"""Find the knowledge base chunks most relevant to a question.

Three retrievers, all with the same method: search(query, k) -> list of results, best first.
A result is {"text", "source", "title", "section", "score"}.

BedrockKBRetriever    the deployed setup: Bedrock Knowledge Base (Titan embeddings stored in
                      Amazon S3 Vectors). One API call does embed + vector search.
LocalVectorRetriever  the same idea run locally: a JSON file of chunk embeddings, searched with
                      cosine similarity. Lets you see exactly how vector search works.
KeywordRetriever      BM25 keyword search, the baseline that vector search is compared against.
"""
import json
import math
import re
from collections import Counter

from llm import STOP

WORD = re.compile(r"[a-z0-9]+")


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class LocalVectorRetriever:
    name = "local-vector"

    def __init__(self, chunks: list[dict], embedder):
        """chunks must already have an "embedding" (see build_index)."""
        self.chunks = chunks
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[dict], embedder):
        texts = [c["text"] for c in chunks]
        if hasattr(embedder, "embed_documents"):  # batch API (Voyage): one request for all chunks
            vectors = embedder.embed_documents(texts)
        else:
            vectors = [embedder.embed(t) for t in texts]
        embedded = [{**c, "embedding": v} for c, v in zip(chunks, vectors)]
        return cls(embedded, embedder)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"embedder": self.embedder.name, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, path: str, embedder):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data["embedder"] != embedder.name:
            raise ValueError(f"Index was built with '{data['embedder']}', not '{embedder.name}'")
        return cls(data["chunks"], embedder)

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.embedder.embed(query)  # Titan and Hash vectors are normalized: dot = cosine
        scored = sorted(((dot(q, c["embedding"]), c) for c in self.chunks), key=lambda x: -x[0])
        return [{**{key: c[key] for key in ("text", "source", "title", "section")}, "score": round(s, 4)}
                for s, c in scored[:k]]


class KeywordRetriever:
    """BM25: scores chunks by shared words, weighting rare words more. No understanding of meaning."""

    name = "keyword-bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tokens = [self._tokens(c["text"]) for c in chunks]
        self.avg_len = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        df = Counter(word for tokens in self.tokens for word in set(tokens))
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [w for w in WORD.findall(text.lower()) if w not in STOP]

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = set(self._tokens(query))
        results = []
        for chunk, tokens in zip(self.chunks, self.tokens):
            tf = Counter(tokens)
            score = sum(
                self.idf[w] * tf[w] * (self.k1 + 1)
                / (tf[w] + self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len))
                for w in q if w in tf)
            if score > 0:
                results.append((score, chunk))
        results.sort(key=lambda x: -x[0])
        return [{**{key: c[key] for key in ("text", "source", "title", "section")}, "score": round(s, 4)}
                for s, c in results[:k]]


class BedrockKBRetriever:
    name = "bedrock-kb"

    def __init__(self, knowledge_base_id: str, client=None):
        self.knowledge_base_id = knowledge_base_id
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("bedrock-agent-runtime",
                                  config=Config(connect_timeout=2, read_timeout=4, retries={"max_attempts": 1}))
        self.client = client

    def search(self, query: str, k: int = 5) -> list[dict]:
        response = self.client.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
        )
        results = []
        for item in response.get("retrievalResults", []):
            uri = item.get("location", {}).get("s3Location", {}).get("uri", "")
            source = uri.rsplit("/", 1)[-1] or "unknown"
            metadata = item.get("metadata", {})
            results.append({
                "text": item.get("content", {}).get("text", ""),
                "source": source,
                "title": str(metadata.get("title") or source),
                "section": "",
                "score": round(float(item.get("score", 0.0)), 4),
            })
        return results


class S3VectorsRetriever:
    """Vector search in Amazon S3 Vectors that this project manages itself: chunks are embedded
    (scripts/sync_vectors.py) and stored with their text as metadata; a question is embedded with
    the same model and the index returns the nearest chunks by cosine distance."""

    name = "s3-vectors"

    def __init__(self, index_arn: str, embedder, client=None):
        self.index_arn = index_arn
        self.embedder = embedder
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("s3vectors",
                                  config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 2}))
        self.client = client

    def search(self, query: str, k: int = 5) -> list[dict]:
        response = self.client.query_vectors(
            indexArn=self.index_arn, topK=k, queryVector={"float32": self.embedder.embed(query)},
            returnMetadata=True, returnDistance=True)
        results = []
        for v in response.get("vectors", []):
            meta = v.get("metadata") or {}
            results.append({"text": str(meta.get("text", "")), "source": str(meta.get("source", "")),
                            "title": str(meta.get("title", "")), "section": str(meta.get("section", "")),
                            "score": round(1.0 - float(v.get("distance", 1.0)), 4)})  # cosine distance -> similarity
        return results


class RerankingRetriever:
    """Two-stage retrieval: pull a wider set of candidates (e.g. 15) from any retriever, then have a
    rerank model (Bedrock Rerank API, e.g. Cohere Rerank 3.5) re-score each candidate against the
    question and keep the top k. Embeddings compare question and chunk separately; a reranker reads
    them together, so it's better at telling "close" from "actually answers the question"."""

    def __init__(self, inner, model_arn: str, candidates: int = 15, client=None):
        self.inner = inner
        self.model_arn = model_arn
        self.candidates = candidates
        self.name = f"{getattr(inner, 'name', 'retriever')}+rerank"
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("bedrock-agent-runtime",
                                  config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 1}))
        self.client = client

    def search(self, query: str, k: int = 5) -> list[dict]:
        pool = self.inner.search(query, self.candidates)
        if len(pool) <= 1:
            return pool[:k]
        response = self.client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[{"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": c["text"]}}}
                     for c in pool],
            rerankingConfiguration={"type": "BEDROCK_RERANKING_MODEL", "bedrockRerankingConfiguration": {
                "numberOfResults": min(k, len(pool)), "modelConfiguration": {"modelArn": self.model_arn}}},
        )
        return [{**pool[r["index"]], "score": round(float(r["relevanceScore"]), 4)}
                for r in response.get("results", [])]
