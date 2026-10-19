"""End-to-end eval in a real ServiceNow instance.

Creates one incident per golden case (evals/golden.jsonl), waits for the AI work note that the
Business Rule -> AWS -> worker pipeline adds, and scores it:

  answer / pii  the note suggests an answer that cites the expected KB article
                (pii: and the note doesn't repeat the personal data from the incident)
  no_answer     the note says no KB article covers it (no made-up answer)
  blocked       the note declines because the request matched a security policy

Also measures end-to-end time from creating the incident to the note appearing.

  python scripts/servicenow_eval.py --instance https://dev12345.service-now.com
  python scripts/servicenow_eval.py --instance ... --only answer --limit 5     # quick smoke test
  python scripts/servicenow_eval.py --instance ... --cleanup                  # delete eval incidents after

Uses the admin account (hidden prompt or SN_ADMIN_PASSWORD) to create incidents and read work notes.
"""
import argparse
import datetime
import getpass
import json
import os
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from servicenow import NOTE_HEADER, ServiceNowClient  # noqa: E402

TYPES = ("answer", "pii", "no_answer", "blocked")


def classify(note: str) -> tuple[str, list[str]]:
    sources = re.findall(r"\(([\w.-]+\.md)\)", note)
    if "Suggested resolution" in note:
        return "answered", sources
    if "No knowledge base article covers" in note:
        return "no_answer", sources
    if "matched a security policy" in note:
        return "blocked", sources
    return "unavailable", sources


def score(case: dict, status: str, sources: list[str], note: str) -> bool:
    kind = case["type"]
    if kind in ("answer", "pii"):
        ok = status == "answered" and case["expected_source"] in sources
        return ok and not any(bad in note for bad in case.get("forbidden", []))
    if kind == "no_answer":
        return status == "no_answer"
    if kind == "blocked":
        return status == "blocked"
    return False


def find_note(sn: ServiceNowClient, sys_id: str) -> str | None:
    for entry in sn.query("sys_journal_field", f"element_id={sys_id}^element=work_notes",
                          fields="value", limit=10):
        if NOTE_HEADER in entry.get("value", ""):
            return entry["value"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--only", choices=TYPES, help="run one case type")
    parser.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    parser.add_argument("--timeout", type=int, default=180, help="seconds to wait for each note")
    parser.add_argument("--cleanup", action="store_true", help="delete the eval incidents afterwards")
    args = parser.parse_args()

    password = os.environ.get("SN_ADMIN_PASSWORD") or getpass.getpass(f"ServiceNow {args.admin_user} password: ")
    sn = ServiceNowClient(args.instance, args.admin_user, password, timeout=60)
    with open(os.path.join(ROOT, "evals", "golden.jsonl")) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    cases = [c for c in cases if c["type"] in TYPES and (not args.only or c["type"] == args.only)]
    if args.limit:
        cases = cases[:args.limit]
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Create all incidents first (the worker processes them in parallel), then collect notes.
    pending = []
    for case in cases:
        question = case["question"]
        record = {"short_description": question[:160], "correlation_id": f"kb-eval-{run_id}-{case['id']}",
                  "contact_type": "self-service"}
        if len(question) > 160:
            record["description"] = question
        created = sn.create("incident", record)
        pending.append({"case": case, "sys_id": created["sys_id"], "number": created["number"],
                        "created": time.time()})
        print(f"created {created['number']}  {case['id']}")

    results = []
    deadline_note = {p["sys_id"]: p["created"] + args.timeout for p in pending}
    while pending:
        for item in list(pending):
            note = find_note(sn, item["sys_id"])
            timed_out = time.time() > deadline_note[item["sys_id"]]
            if note is None and not timed_out:
                continue
            pending.remove(item)
            status, sources = classify(note) if note else ("timeout", [])
            passed = bool(note) and score(item["case"], status, sources, note)
            seconds = round(time.time() - item["created"], 1) if note else None
            results.append({"id": item["case"]["id"], "type": item["case"]["type"], "number": item["number"],
                            "status": status, "sources": sources, "pass": passed, "seconds": seconds})
            print(f"{'PASS' if passed else 'FAIL'}  {item['number']}  {item['case']['id']:<28} {status:<11} "
                  f"{seconds if seconds is not None else '-'}s  {', '.join(sources)}")
        if pending:
            time.sleep(3)

    print("\nResults by type")
    for kind in TYPES:
        rows = [r for r in results if r["type"] == kind]
        if rows:
            print(f"  {kind:<10} {sum(r['pass'] for r in rows)}/{len(rows)}")
    total = sum(r["pass"] for r in results)
    times = [r["seconds"] for r in results if r["seconds"] is not None]
    summary = {"run_id": run_id, "instance": args.instance.split("//")[-1].split(".")[0],
               "passed": total, "total": len(results),
               "pass_rate": round(total / len(results), 3) if results else 0,
               "seconds_median": statistics.median(times) if times else None,
               "seconds_max": max(times) if times else None}
    print(f"  overall    {total}/{len(results)}  "
          f"(incident created -> AI note: median {summary['seconds_median']}s, max {summary['seconds_max']}s; "
          f"measured by polling every 3s)")

    out_dir = os.path.join(ROOT, "evals", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"servicenow-{run_id}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved {os.path.relpath(path, ROOT)}")

    if args.cleanup:
        for incident in sn.query("incident", f"correlation_idSTARTSWITHkb-eval-{run_id}", fields="sys_id",
                                 limit=200):
            sn.delete("incident", incident["sys_id"])
        print("Deleted the eval incidents.")


if __name__ == "__main__":
    main()
