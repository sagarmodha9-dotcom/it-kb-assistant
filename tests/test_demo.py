"""Public demo route: separate key, input limits, server-side caps, and no ticket offers."""
import json

import pytest

import app


class ConditionFailed(Exception):
    response = {"Error": {"Code": "ConditionalCheckFailedException"}}


class FakeCounters:
    """update_item with ADD + 'used < :cap' condition, like DynamoDB."""

    def __init__(self):
        self.used = {}

    def update_item(self, TableName, Key, UpdateExpression, ConditionExpression, ExpressionAttributeValues):
        key, cap = Key["pk"]["S"], int(ExpressionAttributeValues[":cap"]["N"])
        if self.used.get(key, 0) >= cap:
            raise ConditionFailed()
        self.used[key] = self.used.get(key, 0) + 1


class FakeAssistant:
    def __init__(self):
        self.calls = []

    def ask(self, question, history=None, time_left=None):
        self.calls.append((question, history))
        return {"status": "no_answer", "answer": "I couldn't find this.", "sources": [], "offer_ticket": True,
                "guardrail": {}, "usage": {"input_tokens": 10, "output_tokens": 5}, "provider": "fake",
                "latency_ms": 3, "question": question}


@pytest.fixture
def demo(monkeypatch):
    app._cache.clear()
    assistant = FakeAssistant()
    app._cache.update({"demo_api_key": "demo-secret", "dynamodb": FakeCounters(), "assistant": assistant})
    monkeypatch.setenv("DEMO_TABLE", "t")
    monkeypatch.setenv("DEMO_DAILY_CAP", "5")
    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "2")
    yield assistant
    app._cache.clear()


def call(body, key="demo-secret", client="1.2.3.4"):
    headers = {"x-api-key": key, "x-demo-client": client}
    out = app.demo_ask({"headers": headers, "body": json.dumps(body)}, None)
    return out["statusCode"], json.loads(out["body"])


def test_needs_the_demo_key_not_the_private_one(demo):
    app._cache["api_key"] = "private-secret"
    assert call({"question": "vpn"}, key="private-secret")[0] == 401
    assert call({"question": "vpn"}, key="")[0] == 401
    assert demo.calls == []


def test_answers_without_offering_tickets_or_leaking_internals(demo):
    status, body = call({"question": "How do I get guest Wi-Fi?"})
    assert status == 200 and body["offer_ticket"] is False
    assert "usage" not in body and "provider" not in body and "question" not in body


def test_input_limits(demo):
    assert call({"question": "x" * 301})[0] == 413
    assert call({"question": "   "})[0] == 400
    call({"question": "q", "history": [{"question": f"h{i}", "answer": "a"} for i in range(10)]})
    assert len(demo.calls[-1][1]) == 2  # only the last 2 turns are passed on
    assert demo.calls[0][0] == "q"


def test_per_visitor_cap_then_daily_cap(demo):
    assert [call({"question": "q"}, client="a")[0] for _ in range(3)] == [200, 200, 429]
    assert [call({"question": "q"}, client=c)[0] for c in ("b", "b", "c")] == [200, 200, 200]
    status, body = call({"question": "q"}, client="d")  # 5 answered today: daily cap reached
    assert status == 429 and "today" in body["error"]
    assert len(demo.calls) == 5  # nothing over the caps reached the model


def test_visitor_ids_are_hashed_not_stored_raw(demo):
    call({"question": "q"}, client="203.0.113.7")
    assert not any("203.0.113.7" in k for k in app._cache["dynamodb"].used)


def test_quota_store_failure_fails_closed(demo):
    class Broken:
        def update_item(self, **kwargs):
            raise RuntimeError("dynamodb down")
    app._cache["dynamodb"] = Broken()
    assert call({"question": "q"})[0] == 503
    assert demo.calls == []  # no paid call without a working quota check
