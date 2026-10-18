"""Talk to the AI model.

BedrockLLM    calls a model through Amazon Bedrock's Converse API.
AnthropicLLM  calls Claude directly through the Anthropic Messages API.
FallbackLLM   tries one provider and switches to a backup if it fails (throttling, quota, outage).
FakeLLM       a rule-based stand-in so tests, CI, and demos run without AWS or an API key.

Every model has the same method, so the rest of the code doesn't care which one it uses:
  complete(system, user, max_tokens) -> text
and records `last_usage` = {"input_tokens": n, "output_tokens": n} after each call.
"""
import json
import logging
import re
import urllib.request

logger = logging.getLogger(__name__)


def _usage(input_tokens=0, output_tokens=0) -> dict:
    return {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}


class BedrockLLM:
    provider = "bedrock"

    def __init__(self, model_id: str, client=None, max_attempts: int = 3, read_timeout: int = 15):
        self.model_id = model_id
        self.last_usage = _usage()
        if client is None:
            import boto3  # only needed when actually calling AWS
            from botocore.config import Config
            # botocore's default read timeout is 60s, longer than API Gateway's 30s limit.
            client = boto3.client("bedrock-runtime", config=Config(
                connect_timeout=3, read_timeout=read_timeout,
                retries={"max_attempts": max_attempts, "mode": "standard"}))
        self.client = client

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = response.get("usage", {})
        self.last_usage = _usage(usage.get("inputTokens"), usage.get("outputTokens"))
        return "".join(b.get("text", "") for b in response["output"]["message"]["content"])


class AnthropicLLM:
    """Calls the Anthropic Messages API with only the standard library (no extra packages)."""

    provider = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model_id: str, api_key: str, opener=urllib.request.urlopen, timeout: int = 15):
        if not api_key:
            raise ValueError("Anthropic API key is missing")
        self.model_id = model_id
        self.api_key = api_key
        self.opener = opener
        self.timeout = timeout
        self.last_usage = _usage()

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        request = urllib.request.Request(
            self.URL,
            data=json.dumps({
                "model": self.model_id, "max_tokens": max_tokens, "temperature": 0,
                "system": system, "messages": [{"role": "user", "content": user}],
            }).encode("utf-8"),
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        with self.opener(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        usage = data.get("usage", {})
        self.last_usage = _usage(usage.get("input_tokens"), usage.get("output_tokens"))
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class FallbackLLM:
    """Try the primary provider first; if it raises, use the backup instead."""

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup = backup
        self.last_provider = None

    @property
    def provider(self):
        return self.last_provider or self.primary.provider

    @property
    def last_usage(self):
        used = self.backup if self.last_provider == self.backup.provider else self.primary
        return getattr(used, "last_usage", _usage())

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        try:
            result = self.primary.complete(system, user, max_tokens)
            self.last_provider = self.primary.provider
            return result
        except Exception as error:  # throttling, no quota, outage, etc.
            logger.warning("Primary model (%s) failed, switching to backup (%s): %s",
                           self.primary.provider, self.backup.provider, type(error).__name__)
            result = self.backup.complete(system, user, max_tokens)
            self.last_provider = self.backup.provider
            return result


WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be but by can do does for from how i if in into is it its me my
no not of on or our so than that the their then there these this to up was we what when where which
who why will with you your get got have has need want""".split())


def content_words(text: str) -> set:
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1}


class FakeLLM:
    """Pretends to answer from the documents with simple word overlap. Good enough to test the
    pipeline (prompt building, JSON parsing, citations, refusals) without calling a real model."""

    provider = "fake"

    def __init__(self):
        self.last_usage = _usage()

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        self.last_usage = _usage(len(system + user) // 4, 60)
        question = user.split("<question>")[-1].split("</question>")[0]
        if "<history>" in user:  # follow-up rewrite: borrow the topic of the last user turn
            last = [line[6:] for line in user.split("\n") if line.startswith("User: ")][-1]
            return f"{question} ({last})"
        q_words = content_words(question)
        docs = re.findall(r'<document index="(\d+)"[^>]*>(.*?)</document>', user, re.S)
        best = None
        for index, text in docs:
            for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
                overlap = len(q_words & content_words(sentence))
                if sentence.strip() and (best is None or overlap > best[0]):
                    best = (overlap, int(index), sentence.strip())
        if not best or best[0] < 2:
            return json.dumps({"answerable": False, "answer": "", "citations": []})
        return json.dumps({"answerable": True, "answer": f"{best[2]} [{best[1]}]", "citations": [best[1]]})


def build_llm(provider: str, bedrock_model_id: str = "", anthropic_model_id: str = "",
              anthropic_api_key: str = ""):
    """Pick the model setup by name: bedrock, anthropic, bedrock_with_anthropic_fallback, or fake."""
    # Worst case per call stays under assistant.MIN_SECONDS_FOR_MODEL (20 s).
    if provider == "bedrock":
        return BedrockLLM(bedrock_model_id, max_attempts=2, read_timeout=8)
    if provider == "anthropic":
        return AnthropicLLM(anthropic_model_id, anthropic_api_key, timeout=15)
    if provider == "bedrock_with_anthropic_fallback":
        # One quick Bedrock attempt (no long retry loop), then fail over to Anthropic.
        return FallbackLLM(BedrockLLM(bedrock_model_id, max_attempts=1, read_timeout=8),
                           AnthropicLLM(anthropic_model_id, anthropic_api_key, timeout=10))
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown LLM provider: {provider}")
