"""Build the local vector index: chunk the docs, embed every chunk, save to data/index_<embedder>.json.

  python scripts/build_index.py --embedder hash    # offline, no AWS (used by tests and CI)
  python scripts/build_index.py --embedder titan   # real embeddings from Amazon Titan on Bedrock
  python scripts/build_index.py --embedder voyage  # real embeddings from Voyage AI (VOYAGE_API_KEY)
"""
import argparse
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from documents import chunk_all, load_docs  # noqa: E402
from embeddings import build_embedder  # noqa: E402
from retrieval import LocalVectorRetriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["hash", "titan", "voyage"], default="hash")
    parser.add_argument("--docs", default=os.path.join(ROOT, "docs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    chunks = chunk_all(load_docs(args.docs))
    embedder = build_embedder(args.embedder)
    started = time.time()
    if args.embedder == "titan":
        print(f"Embedding {len(chunks)} chunks with Titan (spaced out to stay under Bedrock limits)...")
    index = LocalVectorRetriever.build(chunks, embedder)
    path = os.path.join(ROOT, "data", f"index_{args.embedder}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    index.save(path)
    print(f"Embedded {len(chunks)} chunks with {embedder.name} in {time.time() - started:.1f}s "
          f"({embedder.tokens_used} tokens) -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
