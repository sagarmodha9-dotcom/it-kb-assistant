"""Fast check of the input guardrail alone against every golden question (no model calls, ~30 s).

  python scripts/check_guardrail.py --stack it-kb-assistant

"blocked" cases must be blocked; every other question must get through. Use it after changing
denied topics or filters, before running the full evals.
"""
import argparse
import json
import os
import sys

import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from guardrails import BedrockGuardrail  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="it-kb-assistant")
    parser.add_argument("--version", default="", help="defaults to the stack's GuardrailVersion; DRAFT works too")
    args = parser.parse_args()
    out = {o["OutputKey"]: o["OutputValue"] for o in
           boto3.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]}
    guard = BedrockGuardrail(out["GuardrailId"], args.version or out["GuardrailVersion"])
    with open(os.path.join(ROOT, "evals", "golden.jsonl")) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    wrong = 0
    for case in cases:
        result = guard.check_input(case["question"])
        should_block = case["type"] == "blocked"
        ok = result.blocked == should_block
        wrong += not ok
        mark = "ok  " if ok else "FAIL"
        print(f"{mark} {case['id']:24} {result.action:10} {','.join(result.reasons)}")
    print(f"\n{len(cases) - wrong}/{len(cases)} correct (version {guard.version})")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
