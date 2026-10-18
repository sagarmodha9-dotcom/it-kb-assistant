"""Upload the docs to S3 and sync the Bedrock Knowledge Base (chunk + embed + store in S3 Vectors).

  python scripts/sync_kb.py --stack it-kb-assistant

Reads the bucket, knowledge base, and data source IDs from the CloudFormation stack outputs.
Each doc gets a <name>.metadata.json file with its title, so search results can show it.
Run it again any time a doc changes; files deleted locally are deleted from S3 too.
"""
import argparse
import json
import os
import sys
import time

import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from documents import load_docs  # noqa: E402


def stack_outputs(name: str) -> dict:
    stack = boto3.client("cloudformation").describe_stacks(StackName=name)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="it-kb-assistant")
    parser.add_argument("--docs", default=os.path.join(ROOT, "docs"))
    args = parser.parse_args()

    out = stack_outputs(args.stack)
    bucket, kb_id, ds_id = out["DocsBucketName"], out["KnowledgeBaseId"], out["DataSourceId"]
    s3 = boto3.client("s3")

    keep = set()
    for doc in load_docs(args.docs):
        s3.put_object(Bucket=bucket, Key=doc["source"], Body=doc["text"].encode("utf-8"),
                      ContentType="text/markdown")
        meta_key = doc["source"] + ".metadata.json"
        s3.put_object(Bucket=bucket, Key=meta_key, ContentType="application/json",
                      Body=json.dumps({"metadataAttributes": {"title": doc["title"]}}).encode("utf-8"))
        keep |= {doc["source"], meta_key}
    removed = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            if obj["Key"] not in keep:
                s3.delete_object(Bucket=bucket, Key=obj["Key"])
                removed += 1
    print(f"Uploaded {len(keep) // 2} docs to s3://{bucket} (removed {removed} old files)")

    agent = boto3.client("bedrock-agent")
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)["ingestionJob"]
    print("Ingestion started", end="", flush=True)
    while job["status"] in ("STARTING", "IN_PROGRESS"):
        time.sleep(5)
        print(".", end="", flush=True)
        job = agent.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id,
                                      ingestionJobId=job["ingestionJobId"])["ingestionJob"]
    print(f"\nStatus: {job['status']}")
    print(json.dumps(job.get("statistics", {}), indent=2))
    if job["status"] != "COMPLETE":
        print("Failure reasons:", job.get("failureReasons"))
        sys.exit(1)


if __name__ == "__main__":
    main()
