"""Ask the assistant from the terminal.

  python scripts/ask.py "How do I set up MFA on my new phone?"                  # offline (fake model)
  RETRIEVER=local EMBEDDER=titan INDEX_PATH=data/index_titan.json LLM_PROVIDER=bedrock \\
  GUARDRAIL=local python scripts/ask.py "My laptop wants a 48 digit key"         # real models
  python scripts/ask.py --api https://xxxx.execute-api.us-east-1.amazonaws.com "..."  # deployed API
"""
import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--api", help="Deployed API URL; the key is read from ASSISTANT_API_KEY")
    args = parser.parse_args()
    if args.api:
        request = urllib.request.Request(
            args.api.rstrip("/") + "/ask", data=json.dumps({"question": args.question}).encode(),
            headers={"content-type": "application/json", "x-api-key": os.environ.get("ASSISTANT_API_KEY", "")},
            method="POST")
        with urllib.request.urlopen(request, timeout=35) as response:
            result = json.loads(response.read())
    else:
        from factory import build_assistant
        result = build_assistant().ask(args.question)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
