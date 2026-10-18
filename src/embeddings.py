"""Turn text into vectors (embeddings) so we can search by meaning instead of exact words.

TitanEmbedder  Amazon Titan Text Embeddings V2 on Bedrock (the same model the Bedrock
               Knowledge Base uses). Vectors are normalized, so cosine similarity = dot product.
HashEmbedder   An offline stand-in: hashes words into a fixed-size vector. It only matches shared
               words, so it is NOT semantic. It exists so tests and CI run without AWS.
"""
import hashlib
import json
import logging
import math
import re
import time

from llm import STOP

logger = logging.getLogger(__name__)

WORD = re.compile(r"[a-z0-9]+")


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / length for v in vector]


class TitanEmbedder:
    """Rate-limit friendly: requests are spaced out (min_interval), throttling errors are retried
    with growing waits, and repeated texts (the same eval question asked 3 times) are cached."""

    name = "titan-v2"
    WAITS = (2, 4, 8, 15, 30, 30, 30)  # seconds to wait after each throttling error

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", dimensions: int = 1024,
                 client=None, min_interval: float = 0.25, sleep=time.sleep):
        self.model_id = model_id
        self.dimensions = dimensions
        self.tokens_used = 0
        self.min_interval = min_interval
        self.sleep = sleep
        self._last_call = 0.0
        self._cache = {}
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 3, "mode": "adaptive"}))
        self.client = client

    def _invoke(self, text: str) -> dict:
        for wait in self.WAITS + (None,):
            gap = self.min_interval - (time.time() - self._last_call)
            if gap > 0:
                self.sleep(gap)
            self._last_call = time.time()
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps({"inputText": text[:20000], "dimensions": self.dimensions, "normalize": True}),
                    contentType="application/json",
                    accept="application/json",
                )
                return json.loads(response["body"].read())
            except Exception as error:
                code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
                if code not in ("ThrottlingException", "ServiceUnavailableException") or wait is None:
                    raise
                logger.warning("Titan throttled, waiting %ss", wait)
                self.sleep(wait)

    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            data = self._invoke(text)
            self.tokens_used += int(data.get("inputTextTokenCount", 0))
            self._cache[text] = data["embedding"]
        return self._cache[text]


class HashEmbedder:
    name = "hash"

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.tokens_used = 0

    def _features(self, text: str) -> list[str]:
        words = [w.rstrip("s") if len(w) > 3 else w
                 for w in WORD.findall(text.lower()) if w not in STOP]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        return normalize(vector)


def build_embedder(name: str):
    if name == "titan":
        return TitanEmbedder()
    if name == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedder: {name}")
