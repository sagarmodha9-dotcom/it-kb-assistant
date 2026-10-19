"""Build the assistant from settings, so Lambda, the web UI, and the evals all wire it the same way.

Settings (environment variables or a dict):
  RETRIEVER          s3-vectors (needs VECTOR_INDEX_ARN) | bedrock-kb (needs KNOWLEDGE_BASE_ID)
                     | local (needs INDEX_PATH) | keyword
  EMBEDDER           voyage | titan | hash  (s3-vectors and local; must match what built the index)
  VOYAGE_API_KEY
  GUARDRAIL          bedrock (needs GUARDRAIL_ID, GUARDRAIL_VERSION) | local | none
  LLM_PROVIDER       bedrock | anthropic | bedrock_with_anthropic_fallback | fake
  MODEL_ID, ANTHROPIC_MODEL_ID, ANTHROPIC_API_KEY
  DOCS_DIR           folder of .md docs (keyword retriever)
  RERANK_MODEL_ARN   optional: rerank a wider candidate set with this Bedrock rerank model
  TOP_K, MIN_SCORE
"""
import os

from assistant import Assistant
from documents import chunk_all, load_docs
from embeddings import build_embedder
from guardrails import BedrockGuardrail, LocalGuardrail, NoGuardrail
from llm import build_llm
from retrieval import (BedrockKBRetriever, KeywordRetriever, LocalVectorRetriever, RerankingRetriever,
                       S3VectorsRetriever)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "RETRIEVER": "local",
    "EMBEDDER": "hash",
    "INDEX_PATH": os.path.join(HERE, "..", "data", "index_hash.json"),
    "DOCS_DIR": os.path.join(HERE, "..", "docs"),
    "GUARDRAIL": "local",
    "LLM_PROVIDER": "fake",
    "MODEL_ID": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "ANTHROPIC_MODEL_ID": "claude-haiku-4-5",
    "TOP_K": "5",
    "MIN_SCORE": "0",
}


def setting(settings: dict, name: str) -> str:
    return str(settings.get(name) or os.environ.get(name) or DEFAULTS.get(name, ""))


def build_retriever(settings: dict):
    base = _base_retriever(settings)
    rerank = setting(settings, "RERANK_MODEL_ARN")
    return RerankingRetriever(base, rerank) if rerank else base


def _base_retriever(settings: dict):
    kind = setting(settings, "RETRIEVER")
    if kind == "bedrock-kb":
        return BedrockKBRetriever(setting(settings, "KNOWLEDGE_BASE_ID"))
    if kind == "s3-vectors":
        return S3VectorsRetriever(setting(settings, "VECTOR_INDEX_ARN"),
                                  build_embedder(setting(settings, "EMBEDDER"), setting(settings, "VOYAGE_API_KEY")))
    if kind == "local":
        embedder = build_embedder(setting(settings, "EMBEDDER"), setting(settings, "VOYAGE_API_KEY"))
        return LocalVectorRetriever.load(setting(settings, "INDEX_PATH"), embedder)
    if kind == "keyword":
        return KeywordRetriever(chunk_all(load_docs(setting(settings, "DOCS_DIR"))))
    raise ValueError(f"Unknown retriever: {kind}")


def build_guardrail(settings: dict):
    kind = setting(settings, "GUARDRAIL")
    if kind == "bedrock":
        return BedrockGuardrail(setting(settings, "GUARDRAIL_ID"), setting(settings, "GUARDRAIL_VERSION"))
    if kind == "local":
        return LocalGuardrail()
    if kind == "none":
        return NoGuardrail()
    raise ValueError(f"Unknown guardrail: {kind}")


def build_assistant(settings: dict | None = None, retriever=None, llm=None, guardrail=None) -> Assistant:
    settings = settings or {}
    return Assistant(
        retriever=retriever or build_retriever(settings),
        llm=llm or build_llm(setting(settings, "LLM_PROVIDER"), setting(settings, "MODEL_ID"),
                             setting(settings, "ANTHROPIC_MODEL_ID"), setting(settings, "ANTHROPIC_API_KEY")),
        guardrail=guardrail or build_guardrail(settings),
        k=int(setting(settings, "TOP_K")),
        min_score=float(setting(settings, "MIN_SCORE")),
    )
