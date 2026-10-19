"""End-to-end eval in a real ServiceNow instance.

Creates one incident per golden case (evals/golden.jsonl), waits for the AI work note that the
Business Rule -> AWS -> worker pipeline adds, and scores it.

Agent mode (default, matches ServiceNowMode=agent):
  answer / pii  the agent sent the fix from the expected KB article and is waiting for the caller
                (except security incidents like phish-clicked: escalated to Security, priority 2)
  no_answer     escalated to a team with a handoff note (no made-up fix sent)
  blocked       escalated for manual review without the agent seeing the text
  outage        3 similar incidents at once: all escalated to Network with priority raised to 2
  confirm       on 2 fixed incidents, a caller reply "it worked" -> Resolved, "still broken" -> In Progress

Suggest mode (--mode suggest, matches ServiceNowMode=suggest):

  answer / pii  the note suggests an answer that cites the expected KB article
                (pii: and the note doesn't repeat the personal data from the incident)
  no_answer     the note says no KB article covers it (no made-up answer)
  blocked       the note declines because the request matched a security policy

Also measures end-to-end time from creating the incident to the note appearing.

  python scripts/servicenow_eval.py --instance https://dev12345.service-now.com
  python scripts/servicenow_eval.py --instance ... --only answer --limit 5     # quick smoke test
  python scripts/servicenow_eval.py --instance ... --cleanup                  # delete eval incidents after
  python scripts/servicenow_eval.py --instance ... --rescore evals/results/servicenow-<run>.json
      # re-check a finished run: notes that arrived after the wait are scored, nothing new is created

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

from agent import AGENT_HEADER, FIX_MARKER  # noqa: E402
from servicenow import NOTE_HEADER, ServiceNowClient  # noqa: E402

TYPES = ("answer", "pii", "no_answer", "blocked")
ESCALATED = re.compile(r"Action: escalated to (.+?)(?:, priority raised to (\d))?\.")
SECURITY_CASES = {"phish-clicked"}  # the KB has an article, but a possible compromise must go to people
OUTAGE = ["VPN keeps disconnecting every few minutes", "The VPN disconnects constantly this morning",
          "VPN disconnecting again and again, can't stay connected"]


def classify_agent(note: str) -> tuple[str, list[str], str, str]:
    sources = re.findall(r"\(([\w.-]+\.md)\)", note.split("Closest knowledge base answer")[0])
    if FIX_MARKER in note:
        return "awaiting_caller", sources, "", ""
    match = ESCALATED.search(note)
    if match:
        return "escalated", sources, match.group(1), match.group(2) or ""
    return "unknown", sources, "", ""


def score_agent(case: dict, status: str, sources: list[str], team: str, priority: str, note: str) -> bool:
    kind = case["type"]
    if case["id"] in SECURITY_CASES:
        return status == "escalated" and team == "Security" and priority == "2"
    if kind in ("answer", "pii"):
        ok = status == "awaiting_caller" and case["expected_source"] in sources
        return ok and not any(bad in note for bad in case.get("forbidden", []))
    if kind == "no_answer":
        return status == "escalated"
    if kind == "blocked":
        return status == "escalated" and "guardrail" in note
    if kind == "outage":
        return status == "escalated" and team == "Network" and priority == "2"
    return False


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


def find_note(sn: ServiceNowClient, sys_id: str, header: str = NOTE_HEADER) -> str | None:
    for entry in sn.query("sys_journal_field", f"element_id={sys_id}^element=work_notes",
                          fields="value", limit=10):
        if header in entry.get("value", "") and "Caller reply at" not in entry.get("value", ""):
            return entry["value"]
    return None


def check_confirmations(sn: ServiceNowClient, results: list, timeout: int) -> list:
    """Reply as the caller on two incidents waiting for confirmation; the agent must resolve only the
    one where the caller says it worked."""
    waiting = [r for r in results if r["status"] == "awaiting_caller"][:2]
    if len(waiting) < 2:
        print("confirm: needs 2 incidents waiting for the caller; skipped")
        return []
    # Posted as the admin account, which is the caller of the eval incidents.
    plan = [(waiting[0], "Yes, that worked. Thanks!", "Resolved"),
            (waiting[1], "No, I tried that and it still doesn't work.", "In Progress")]
    for row, reply, _ in plan:
        found = sn.query("incident", f"number={row['number']}", fields="sys_id", limit=1)[0]
        row["sys_id"] = found["sys_id"]
        sn.update("incident", found["sys_id"], {"comments": reply})
    out = []
    for row, reply, expected in plan:
        deadline, state = time.time() + timeout, ""
        while time.time() < deadline:
            state = sn._request("GET", f"/api/now/table/incident/{row['sys_id']}",
                                query={"sysparm_fields": "state", "sysparm_display_value": "true"})["result"]["state"]
            if state == expected:
                break
            time.sleep(3)
        passed = state == expected
        out.append({"id": f"confirm-{'yes' if expected == 'Resolved' else 'no'}", "type": "confirm",
                    "number": row["number"], "status": state, "sources": [], "pass": passed, "seconds": None})
        print(f"{'PASS' if passed else 'FAIL'}  {row['number']}  caller says {reply!r:<48} -> {state}")
    return out


def print_summary(results: list, run_id: str, instance: str, times: list, path: str) -> None:
    print("\nResults by type")
    for kind in TYPES + ("outage", "confirm"):
        rows = [r for r in results if r["type"] == kind]
        if rows:
            print(f"  {kind:<10} {sum(r['pass'] for r in rows)}/{len(rows)}")
    total = sum(r["pass"] for r in results)
    noted = [r for r in results if r["status"] != "timeout"]
    summary = {"run_id": run_id, "instance": instance.split("//")[-1].split(".")[0],
               "passed": total, "total": len(results),
               "pass_rate": round(total / len(results), 3) if results else 0,
               "notes_received": len(noted),
               "correct_when_noted": f"{sum(r['pass'] for r in noted)}/{len(noted)}",
               "seconds_median": statistics.median(times) if times else None,
               "seconds_max": max(times) if times else None}
    print(f"  overall    {total}/{len(results)}  (notes received: {len(noted)}, correct among them: "
          f"{summary['correct_when_noted']}; created -> note: median {summary['seconds_median']}s, "
          f"max {summary['seconds_max']}s, polling every 3s)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved {os.path.relpath(path, ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--only", choices=TYPES, help="run one case type")
    parser.add_argument("--mode", choices=["agent", "suggest"], default="agent",
                        help="what the stack runs (ServiceNowMode); agent is the default")
    parser.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    parser.add_argument("--timeout", type=int, default=180, help="seconds to wait for each note")
    parser.add_argument("--cleanup", action="store_true", help="delete the eval incidents afterwards")
    parser.add_argument("--rescore", metavar="RESULTS_JSON", help="re-check the incidents of an earlier run")
    args = parser.parse_args()

    password = os.environ.get("SN_ADMIN_PASSWORD") or getpass.getpass(f"ServiceNow {args.admin_user} password: ")
    sn = ServiceNowClient(args.instance, args.admin_user, password, timeout=60)
    with open(os.path.join(ROOT, "evals", "golden.jsonl")) as f:
        golden = {c["id"]: c for c in (json.loads(line) for line in f if line.strip())}

    if args.rescore:
        with open(args.rescore) as f:
            earlier = json.load(f)
        run_id = earlier["summary"]["run_id"]
        results = []
        for row in earlier["results"]:
            if row["status"] == "timeout":
                found = sn.query("incident", f"number={row['number']}", fields="sys_id", limit=1)
                note = find_note(sn, found[0]["sys_id"]) if found else None
                if note:
                    status, sources = classify(note)
                    row.update(status=status, sources=sources, seconds=None, late=True,
                               **{"pass": score(golden[row["id"]], status, sources, note)})
            results.append(row)
            print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['number']}  {row['id']:<28} {row['status']}")
        times = [r["seconds"] for r in results if r.get("seconds") is not None]
        print_summary(results, run_id, args.instance, times, os.path.splitext(args.rescore)[0] + "-rescored.json")
        return

    agent = args.mode == "agent"
    header = AGENT_HEADER if agent else NOTE_HEADER
    # The admin is the caller of the eval incidents, so its replies count as the caller's (confirm step).
    admin_id = sn.query("sys_user", f"user_name={args.admin_user}", fields="sys_id", limit=1)[0]["sys_id"]
    cases = [c for c in golden.values() if c["type"] in TYPES and (not args.only or c["type"] == args.only)]
    if args.limit:
        cases = cases[:args.limit]
    if agent and not args.only:
        cases += [{"id": f"outage-{i + 1}", "type": "outage", "question": q} for i, q in enumerate(OUTAGE)]
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Create all incidents first (the worker processes them in parallel), then collect notes.
    pending = []
    for case in cases:
        question = case["question"]
        record = {"short_description": question[:160], "correlation_id": f"kb-eval-{run_id}-{case['id']}",
                  "contact_type": "self-service", "caller_id": admin_id}
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
            note = find_note(sn, item["sys_id"], header)
            timed_out = time.time() > deadline_note[item["sys_id"]]
            if note is None and not timed_out:
                continue
            pending.remove(item)
            team = priority = ""
            if not note:
                status, sources = "timeout", []
            elif agent:
                status, sources, team, priority = classify_agent(note)
            else:
                status, sources = classify(note)
            passed = bool(note) and (score_agent(item["case"], status, sources, team, priority, note) if agent
                                     else score(item["case"], status, sources, note))
            seconds = round(time.time() - item["created"], 1) if note else None
            results.append({"id": item["case"]["id"], "type": item["case"]["type"], "number": item["number"],
                            "status": status, "team": team, "priority": priority, "sources": sources,
                            "pass": passed, "seconds": seconds})
            print(f"{'PASS' if passed else 'FAIL'}  {item['number']}  {item['case']['id']:<28} {status:<15} "
                  f"{team + (' P' + priority if priority else ''):<22} {seconds if seconds is not None else '-'}s  "
                  f"{', '.join(sources)}")
        if pending:
            time.sleep(3)

    if agent and not args.only:
        results += check_confirmations(sn, results, args.timeout)
    times = [r["seconds"] for r in results if r["seconds"] is not None]
    path = os.path.join(ROOT, "evals", "results", f"servicenow-{args.mode}-{run_id}.json")
    print_summary(results, run_id, args.instance, times, path)
    if args.cleanup:
        for incident in sn.query("incident", f"correlation_idSTARTSWITHkb-eval-{run_id}", fields="sys_id",
                                 limit=200):
            sn.delete("incident", incident["sys_id"])
        print("Deleted the eval incidents.")


if __name__ == "__main__":
    main()
