"""Embed the docs with Voyage and load them into the project's own Amazon S3 Vectors index.

  export VOYAGE_API_KEY=...   (or: python scripts/sync_vectors.py --key-param /it-kb-assistant/voyage-api-key)
  python scripts/sync_vectors.py --stack it-kb-assistant

Each chunk is stored as one vector with metadata: source, title, section (filterable) and text
(non-filterable, so it can be longer than the 2 KB filterable limit). Re-running replaces changed
chunks and deletes chunks that no longer exist, so the index always matches docs/.
"""
import argparse
import os
import sys
import time

import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from documents import chunk_all, load_docs  # noqa: E402
from embeddings import VoyageEmbedder  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="it-kb-assistant")
    parser.add_argument("--docs", default=os.path.join(ROOT, "docs"))
    parser.add_argument("--key-param", default="", help="SSM parameter holding the Voyage API key")
    args = parser.parse_args()

    stack = boto3.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]
    index_arn = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}["ChunkIndexArn"]
    key = os.environ.get("VOYAGE_API_KEY", "")
    if args.key_param:
        key = boto3.client("ssm").get_parameter(Name=args.key_param, WithDecryption=True)["Parameter"]["Value"]

    chunks = chunk_all(load_docs(args.docs))
    started = time.time()
    embedder = VoyageEmbedder(key)
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    print(f"Embedded {len(chunks)} chunks with {embedder.name} in {time.time() - started:.1f}s "
          f"({embedder.tokens_used} tokens)")

    s3v = boto3.client("s3vectors")
    items = [{"key": c["id"], "data": {"float32": v},
              "metadata": {"source": c["source"], "title": c["title"], "section": c["section"], "text": c["text"]}}
             for c, v in zip(chunks, vectors)]
    for start in range(0, len(items), 500):
        s3v.put_vectors(indexArn=index_arn, vectors=items[start:start + 500])

    keep, stale, token = {c["id"] for c in chunks}, [], None
    while True:
        page = s3v.list_vectors(indexArn=index_arn, **({"nextToken": token} if token else {}))
        stale += [v["key"] for v in page.get("vectors", []) if v["key"] not in keep]
        token = page.get("nextToken")
        if not token:
            break
    for start in range(0, len(stale), 500):
        s3v.delete_vectors(indexArn=index_arn, keys=stale[start:start + 500])
    print(f"Stored {len(items)} vectors in {index_arn.split('/', 1)[1]} (deleted {len(stale)} stale)")


if __name__ == "__main__":
    main()
