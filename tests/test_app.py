import json

import pytest

import app
from tickets import TicketError


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    app._cache.clear()
    app._cache["api_key"] = "secret"
    monkeypatch.setenv("RETRIEVER", "keyword")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("GUARDRAIL", "local")
    yield
    app._cache.clear()


def event(body, key="secret"):
    return {"headers": {"x-api-key": key} if key else {}, "body": body if isinstance(body, str) else json.dumps(body)}


def test_ask_requires_key():
    assert app.ask(event({"question": "vpn"}, key=None), None)["statusCode"] == 401
    assert app.ask(event({"question": "vpn"}, key="wrong"), None)["statusCode"] == 401


def test_missing_key_parameter_is_503(monkeypatch):
    app._cache.pop("api_key")
    monkeypatch.setenv("API_KEY_PARAM", "/x")
    monkeypatch.setattr(app, "_ssm_secret", lambda name: (_ for _ in ()).throw(RuntimeError("no param")))
    assert app.ask(event({"question": "vpn"}), None)["statusCode"] == 503
    assert "api_key" not in app._cache  # retried next time


def test_bad_requests():
    assert app.ask(event("not json"), None)["statusCode"] == 400
    assert app.ask(event("[1, 2]"), None)["statusCode"] == 400
    assert app.ask(event({"question": "  "}), None)["statusCode"] == 400
    assert app.ask(event({"question": "x" * 1001}), None)["statusCode"] == 413


class Context:
    def __init__(self, ms):
        self.ms = ms

    def get_remaining_time_in_millis(self):
        return self.ms


def test_ask_respects_lambda_time_left():
    body = json.loads(app.ask(event({"question": "What is the VPN portal address?"}), Context(5000))["body"])
    assert body["status"] == "error" and body["offer_ticket"]


def test_ask_passes_history(monkeypatch):
    seen = {}

    class A:
        def ask(self, question, history=None, time_left=None):
            seen.update(question=question, history=history)
            return {"status": "answered", "provider": "x", "sources": [], "guardrail": {}, "usage": {}, "latency_ms": 1}
    app._cache["assistant"] = A()
    app.ask(event({"question": "and on Windows?", "history": [{"question": "q", "answer": "a"}]}), None)
    assert seen["history"] == [{"question": "q", "answer": "a"}]


def test_ask_answers():
    out = app.ask(event({"question": "What is the VPN portal address for GlobalProtect?"}), Context(29000))
    body = json.loads(out["body"])
    assert out["statusCode"] == 200
    assert body["status"] == "answered" and body["sources"][0]["source"] == "vpn-troubleshooting.md"


def test_escalate_masks_pii_and_creates_ticket(monkeypatch):
    sent = {}

    class Client:
        def __init__(self, url):
            sent["url"] = url

        def create(self, question, answer, sources):
            sent.update(question=question, answer=answer, sources=sources)
            return {"ticket_id": "t-9", "priority": "P3", "category": "other"}
    monkeypatch.setattr(app, "TicketClient", Client)
    monkeypatch.setenv("TRIAGE_API_URL", "https://triage")
    out = app.escalate(event({"question": "My SSN is 123-45-6789, VPN broken",
                              "answer": "card 4111 1111 1111 1111, call 301-555-1234",
                              "sources": ["vpn-troubleshooting.md", "../etc/passwd", "x" * 300 + ".md", 5]}), None)
    assert out["statusCode"] == 201 and json.loads(out["body"])["ticket_id"] == "t-9"
    assert "123-45-6789" not in sent["question"] and sent["sources"] == ["vpn-troubleshooting.md"]
    assert "4111" not in sent["answer"] and "301-555-1234" not in sent["answer"]


def test_escalate_blocked_and_failure(monkeypatch):
    assert app.escalate(event({"question": "how to disable CrowdStrike"}), None)["statusCode"] == 400

    class Broken:
        def __init__(self, url):
            raise TicketError("not configured")
    monkeypatch.setattr(app, "TicketClient", Broken)
    assert app.escalate(event({"question": "printer jam"}), None)["statusCode"] == 502
