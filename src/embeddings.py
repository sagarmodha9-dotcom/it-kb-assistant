"""Turn text into vectors (embeddings) so we can search by meaning instead of exact words.

TitanEmbedder  Amazon Titan Text Embeddings V2 on Bedrock (the same model the Bedrock
               Knowledge Base uses). Vectors are normalized, so cosine similarity = dot product.
VoyageEmbedder Voyage AI voyage-4 (Anthropic's recommended embeddings provider), called over HTTPS.
               Documents and queries are embedded with different input types, which improves
               retrieval. Batches up to 128 texts per request.
HashEmbedder   An offline stand-in: hashes words into a fixed-size vector. It only matches shared
               words, so it is NOT semantic. It exists so tests and CI run without AWS.
"""
import hashlib
import json
import logging
import math
import os
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


class VoyageEmbedder:
    name = "voyage-4"
    URL = "https://api.voyageai.com/v1/embeddings"
    WAITS = (2, 5, 10, 20, 30, 30)
    BATCH = 128

    def __init__(self, api_key: str, model: str = "voyage-4", dimensions: int = 1024,
                 opener=None, sleep=time.sleep, timeout: int = 10):
        if not api_key:
            raise ValueError("Voyage API key is missing (set VOYAGE_API_KEY)")
        import urllib.request
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.opener = opener or urllib.request.urlopen
        self.sleep = sleep
        self.timeout = timeout
        self.tokens_used = 0
        self._cache = {}

    def _post(self, texts: list[str], input_type: str) -> list[list[float]]:
        import urllib.error
        import urllib.request
        body = json.dumps({"input": texts, "model": self.model, "input_type": input_type,
                           "output_dimension": self.dimensions}).encode("utf-8")
        for wait in self.WAITS + (None,):
            request = urllib.request.Request(self.URL, data=body, method="POST", headers={
                "content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503) or wait is None:
                    raise
                logger.warning("Voyage returned %s, waiting %ss", error.code, wait)
                self.sleep(wait)
        self.tokens_used += int(data.get("usage", {}).get("total_tokens", 0))
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [normalize(r["embedding"]) for r in rows]

    def embed_many(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if (input_type, t) not in self._cache]
        for start in range(0, len(missing), self.BATCH):
            batch = missing[start:start + self.BATCH]
            for text, vector in zip(batch, self._post(batch, input_type)):
                self._cache[(input_type, text)] = vector
        return [self._cache[(input_type, t)] for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_many(texts, "document")

    def embed(self, text: str) -> list[float]:
        """Single text = a search query."""
        return self.embed_many([text], "query")[0]


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


def build_embedder(name: str, api_key: str = ""):
    if name == "titan":
        return TitanEmbedder()
    if name == "voyage":
        return VoyageEmbedder(api_key or os.environ.get("VOYAGE_API_KEY", ""))
    if name == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedder: {name}")
