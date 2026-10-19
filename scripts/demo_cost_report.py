"""Measured usage and cost of the public demo.

  python scripts/demo_cost_report.py                  # last 24 hours of demo traffic
  python scripts/demo_cost_report.py --hours 168      # last week
  python scripts/demo_cost_report.py --cost-explorer  # also AWS's own billed cost by service (last 7 days)

1. Reads the demo function's logs (one JSON line per question) with CloudWatch Logs Insights and
   reports questions, outcomes, latency, and the Claude tokens actually used.
2. Turns tokens into dollars with the model prices you pass in (defaults are Claude Haiku 4.5 list
   prices per million tokens; check the current price on the Anthropic pricing page).
3. With --cost-explorer, asks AWS Cost Explorer what the account was actually billed, by service
   (Lambda, API Gateway, DynamoDB, Bedrock Guardrails, S3 Vectors...). Each Cost Explorer request costs $0.01.

Voyage AI embedding cost is not in AWS: check the usage page in the Voyage dashboard. Each question
embeds only the question text (a few dozen tokens).
"""
import argparse
import datetime
import time

QUERY = """
fields @message
| filter @message like /"event": "demo_ask"/
| parse @message '"status": "*"' as status
| parse @message '"input_tokens": *,' as input_tokens
| parse @message '"output_tokens": *}' as output_tokens
| parse @message '"latency_ms": *}' as latency_ms
| stats count(*) as questions, sum(input_tokens) as input_tokens, sum(output_tokens) as output_tokens,
        avg(latency_ms) as avg_latency_ms, pct(latency_ms, 95) as p95_latency_ms by status
"""


def stack_function(cfn, stack: str, logical_id: str) -> str:
    for page in cfn.get_paginator("list_stack_resources").paginate(StackName=stack):
        for resource in page["StackResourceSummaries"]:
            if resource["LogicalResourceId"] == logical_id:
                return resource["PhysicalResourceId"]
    raise SystemExit(f"{logical_id} not found in {stack}. Was the stack deployed with PublicDemo=true?")


def logs_insights(logs, group: str, hours: float) -> list[dict]:
    end = int(time.time())
    query_id = logs.start_query(logGroupName=group, startTime=end - int(hours * 3600), endTime=end,
                                queryString=QUERY)["queryId"]
    while True:
        result = logs.get_query_results(queryId=query_id)
        if result["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
        time.sleep(1)
    return [{f["field"]: f["value"] for f in row} for row in result.get("results", [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", default="it-kb-assistant")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--input-price", type=float, default=1.00, help="$ per million input tokens")
    parser.add_argument("--output-price", type=float, default=5.00, help="$ per million output tokens")
    parser.add_argument("--cost-explorer", action="store_true", help="also show AWS billed cost by service")
    args = parser.parse_args()

    import boto3
    session = boto3.Session(region_name=args.region)
    function = stack_function(session.client("cloudformation"), args.stack, "DemoAskFunction")
    rows = logs_insights(session.client("logs"), f"/aws/lambda/{function}", args.hours)

    print(f"Public demo usage, last {args.hours:g} hours")
    total_q = total_in = total_out = 0
    for row in rows:
        q, tin, tout = int(row.get("questions", 0)), int(float(row.get("input_tokens") or 0)), \
            int(float(row.get("output_tokens") or 0))
        total_q, total_in, total_out = total_q + q, total_in + tin, total_out + tout
        print(f"  {row.get('status', '?'):<10} {q:>5} questions  avg {float(row.get('avg_latency_ms') or 0):,.0f} ms"
              f"  p95 {float(row.get('p95_latency_ms') or 0):,.0f} ms")
    model_cost = total_in / 1e6 * args.input_price + total_out / 1e6 * args.output_price
    print(f"  total      {total_q:>5} questions, {total_in:,} input + {total_out:,} output tokens")
    print(f"  Claude cost at ${args.input_price}/${args.output_price} per M tokens: ${model_cost:.4f}"
          + (f"  (${model_cost / total_q:.5f} per question)" if total_q else ""))

    if args.cost_explorer:
        ce = boto3.client("ce", region_name="us-east-1")
        today = datetime.date.today()
        result = ce.get_cost_and_usage(
            TimePeriod={"Start": str(today - datetime.timedelta(days=7)), "End": str(today)},
            Granularity="MONTHLY", Metrics=["UnblendedCost"], GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}])
        print("\nAWS billed cost, last 7 days (whole account, from Cost Explorer):")
        total = 0.0
        for period in result["ResultsByTime"]:
            for group in period["Groups"]:
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                if amount >= 0.0001:
                    total += amount
                    print(f"  {group['Keys'][0]:<45} ${amount:.4f}")
        print(f"  {'total':<45} ${total:.4f}")


if __name__ == "__main__":
    main()
