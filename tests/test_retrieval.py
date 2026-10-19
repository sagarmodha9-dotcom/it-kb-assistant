import io
import json

import pytest

from conftest import DOCS
from documents import chunk_all, load_docs
from embeddings import HashEmbedder, TitanEmbedder, normalize
from retrieval import BedrockKBRetriever, KeywordRetriever, LocalVectorRetriever

CHUNKS = chunk_all(load_docs(DOCS))


def test_keyword_finds_right_doc():
    top = KeywordRetriever(CHUNKS).search("BitLocker recovery key", 3)
    assert top[0]["source"] == "bitlocker-recovery.md"
    assert top[0]["score"] >= top[-1]["score"]


def test_keyword_no_match_returns_nothing():
    assert KeywordRetriever(CHUNKS).search("zzzz qqqq") == []


def test_hash_embeddings_normalized_and_deterministic():
    e = HashEmbedder()
    a, b = e.embed("reset my password"), e.embed("reset my password")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_local_vector_search_save_load(tmp_path):
    index = LocalVectorRetriever.build(CHUNKS, HashEmbedder())
    path = tmp_path / "index.json"
    index.save(str(path))
    loaded = LocalVectorRetriever.load(str(path), HashEmbedder())
    top = loaded.search("guest wifi password for visitors", 3)
    assert top[0]["source"] == "wifi-access.md"
    assert "embedding" not in top[0]


def test_index_embedder_mismatch_is_an_error(tmp_path):
    path = tmp_path / "index.json"
    LocalVectorRetriever.build(CHUNKS[:2], HashEmbedder()).save(str(path))

    class Other(HashEmbedder):
        name = "titan-v2"
    with pytest.raises(ValueError):
        LocalVectorRetriever.load(str(path), Other())


def test_titan_request_and_token_count():
    class Client:
        def invoke_model(self, **kwargs):
            self.body = json.loads(kwargs["body"])
            return {"body": io.BytesIO(json.dumps({"embedding": [0.6, 0.8], "inputTextTokenCount": 7}).encode())}
    client = Client()
    e = TitanEmbedder(client=client, dimensions=512)
    assert e.embed("hello") == [0.6, 0.8]
    assert client.body == {"inputText": "hello", "dimensions": 512, "normalize": True}
    assert e.tokens_used == 7


def test_bedrock_kb_maps_results():
    class Client:
        def retrieve(self, **kwargs):
            self.kwargs = kwargs
            return {"retrievalResults": [
                {"content": {"text": "Use GlobalProtect."}, "score": 0.71234,
                 "location": {"s3Location": {"uri": "s3://bucket/vpn-troubleshooting.md"}},
                 "metadata": {"title": "VPN: Connect and Troubleshoot"}},
                {"content": {"text": "x"}, "score": 0.1, "location": {}, "metadata": {}},
            ]}
    client = Client()
    results = BedrockKBRetriever("KB123", client=client).search("vpn", 4)
    assert client.kwargs["knowledgeBaseId"] == "KB123"
    assert client.kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] == 4
    assert results[0] == {"text": "Use GlobalProtect.", "source": "vpn-troubleshooting.md",
                          "title": "VPN: Connect and Troubleshoot", "section": "", "score": 0.7123}
    assert results[1]["source"] == "unknown"


class ThrottleError(Exception):
    def __init__(self, code="ThrottlingException"):
        self.response = {"Error": {"Code": code}}


def test_titan_retries_throttling_and_caches():
    class Client:
        calls = 0

        def invoke_model(self, **kwargs):
            Client.calls += 1
            if Client.calls <= 2:
                raise ThrottleError()
            return {"body": io.BytesIO(json.dumps({"embedding": [1.0], "inputTextTokenCount": 3}).encode())}
    slept = []
    e = TitanEmbedder(client=Client(), sleep=slept.append, min_interval=0)
    assert e.embed("hi") == [1.0] and e.embed("hi") == [1.0]
    assert Client.calls == 3 and slept == [2, 4] and e.tokens_used == 3


def test_titan_other_errors_not_retried():
    class Client:
        def invoke_model(self, **kwargs):
            raise ThrottleError("AccessDeniedException")
    with pytest.raises(ThrottleError):
        TitanEmbedder(client=Client(), sleep=lambda s: None).embed("x")


def test_reranker_reorders_wider_pool():
    from retrieval import RerankingRetriever

    class Inner:
        name = "local-vector"

        def search(self, q, k=5):
            self.k = k
            return [{"text": f"t{i}", "source": f"{i}.md", "title": "", "section": "", "score": 0.5} for i in range(k)]

    class Client:
        def rerank(self, **kwargs):
            self.kwargs = kwargs
            return {"results": [{"index": 7, "relevanceScore": 0.91}, {"index": 2, "relevanceScore": 0.4}]}
    inner, client = Inner(), Client()
    r = RerankingRetriever(inner, "arn:rerank", candidates=10, client=client)
    out = r.search("q", 2)
    assert inner.k == 10 and r.name == "local-vector+rerank"
    assert [o["source"] for o in out] == ["7.md", "2.md"] and out[0]["score"] == 0.91
    cfg = client.kwargs["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert cfg["numberOfResults"] == 2 and cfg["modelConfiguration"]["modelArn"] == "arn:rerank"
    assert len(client.kwargs["sources"]) == 10


class VoyageOpener:
    def __init__(self, fail_first=0, code=429):
        self.requests, self.fail_first, self.code = [], fail_first, code

    def __call__(self, request, timeout):
        import urllib.error
        body = json.loads(request.data)
        self.requests.append((request, body))
        if self.fail_first:
            self.fail_first -= 1
            raise urllib.error.HTTPError(request.full_url, self.code, "busy", {}, None)
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))]
        return io.BytesIO(json.dumps({"data": data[::-1], "usage": {"total_tokens": 9}}).encode())


def test_voyage_batches_caches_and_uses_input_types():
    from embeddings import VoyageEmbedder
    opener = VoyageOpener()
    e = VoyageEmbedder("k", opener=opener, sleep=lambda s: None)
    docs = e.embed_documents(["a", "b", "a"])
    assert docs == [[0.6, 0.8]] * 3  # normalized
    q = e.embed("a")
    e.embed("a")
    assert q == [0.6, 0.8] and len(opener.requests) == 2  # one batch for docs, one for the query, then cached
    req, body = opener.requests[0]
    assert body == {"input": ["a", "b"], "model": "voyage-4", "input_type": "document", "output_dimension": 1024}
    assert req.get_header("Authorization") == "Bearer k"
    assert opener.requests[1][1]["input_type"] == "query" and e.tokens_used == 18


def test_voyage_retries_429_but_not_401():
    import urllib.error
    from embeddings import VoyageEmbedder
    slept = []
    e = VoyageEmbedder("k", opener=VoyageOpener(fail_first=2), sleep=slept.append)
    e.embed("x")
    assert slept == [2, 5]
    with pytest.raises(urllib.error.HTTPError):
        VoyageEmbedder("k", opener=VoyageOpener(fail_first=1, code=401), sleep=lambda s: None).embed("x")
    with pytest.raises(ValueError):
        VoyageEmbedder("")


def test_voyage_splits_large_batches():
    from embeddings import VoyageEmbedder
    opener = VoyageOpener()
    VoyageEmbedder("k", opener=opener).embed_documents([str(i) for i in range(300)])
    assert [len(b["input"]) for _, b in opener.requests] == [128, 128, 44]


def test_s3_vectors_retriever_maps_results():
    from retrieval import S3VectorsRetriever

    class Client:
        def query_vectors(self, **kwargs):
            self.kwargs = kwargs
            return {"vectors": [{"key": "vpn#0", "distance": 0.25, "metadata": {
                "text": "Use GlobalProtect.", "source": "vpn-troubleshooting.md", "title": "VPN", "section": "Connecting"}}]}
    client = Client()
    out = S3VectorsRetriever("arn:index", HashEmbedder(), client=client).search("vpn", 3)
    assert client.kwargs["indexArn"] == "arn:index" and client.kwargs["topK"] == 3
    assert client.kwargs["returnMetadata"] and len(client.kwargs["queryVector"]["float32"]) == 512
    assert out == [{"text": "Use GlobalProtect.", "source": "vpn-troubleshooting.md", "title": "VPN",
                    "section": "Connecting", "score": 0.75}]
