import base64
import hashlib
import hmac
import io
import json
import time
import urllib.error

import pytest

import app
from servicenow import (NOTE_HEADER, RetryLater, ServiceNowClient, ServiceNowError, format_work_note,
                        incident_question, process_incident)

SYS_ID = "a" * 32
KEY = base64.b64encode(b"k" * 32).decode()


# ---------------------------------------------------------------- client
class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Opener:
    """Fake urlopen: returns queued responses (dict -> 200 JSON, int -> HTTPError) and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        reply = self.responses.pop(0)
        if isinstance(reply, int):
            raise urllib.error.HTTPError(request.full_url, reply, "err", {}, io.BytesIO(b"{}"))
        return Response(json.dumps(reply).encode())


def client(opener):
    return ServiceNowClient("dev1.service-now.com", "svc", "pw", opener=opener, sleep=lambda s: None)


def test_client_requires_https_and_credentials():
    with pytest.raises(ServiceNowError):
        ServiceNowClient("http://dev1.service-now.com", "svc", "pw")
    with pytest.raises(ServiceNowError):
        ServiceNowClient("https://dev1.service-now.com", "svc", "")
    assert client(Opener()).base == "https://dev1.service-now.com"


def test_get_incident_and_auth_header():
    opener = Opener({"result": {"sys_id": SYS_ID, "number": "INC0010001"}})
    assert client(opener).get_incident(SYS_ID)["number"] == "INC0010001"
    request = opener.requests[0]
    assert request.get_header("Authorization") == "Basic " + base64.b64encode(b"svc:pw").decode()
    assert f"/api/now/table/incident/{SYS_ID}?" in request.full_url


def test_get_incident_404_is_none_and_retries_5xx():
    assert client(Opener(404)).get_incident(SYS_ID) is None
    opener = Opener(503, {"result": {"number": "INC1"}})
    assert client(opener).get_incident(SYS_ID)["number"] == "INC1"
    with pytest.raises(ServiceNowError) as error:
        client(Opener(503, 503, 503)).get_incident(SYS_ID)
    assert error.value.status == 503


def test_rejects_bad_sys_id_before_any_request():
    opener = Opener()
    with pytest.raises(ServiceNowError):
        client(opener).add_work_note("../../sys_user", "x")
    assert opener.requests == []


def test_add_work_note_patches_only_work_notes():
    opener = Opener({"result": {"sys_id": SYS_ID}})
    client(opener).add_work_note(SYS_ID, "hello")
    assert opener.requests[0].get_method() == "PATCH"
    assert json.loads(opener.requests[0].data) == {"work_notes": "hello"}


# ---------------------------------------------------------------- note + processing
def test_incident_question():
    assert incident_question({"short_description": "VPN fails", "description": ""}) == "VPN fails"
    assert incident_question({"short_description": "VPN fails", "description": "vpn fails"}) == "VPN fails"
    q = incident_question({"short_description": "VPN fails", "description": "Error 809 " * 200})
    assert q.startswith("VPN fails. Details: Error 809") and len(q) < 1000


def answered():
    return {"status": "answered", "answer": "Open the portal [1].",
            "sources": [{"n": 1, "title": "VPN", "source": "vpn-troubleshooting.md"}],
            "guardrail": {"output": {"scores": {"GROUNDING": 0.91, "RELEVANCE": 0.88}}}}


def test_work_notes_say_what_the_ai_did():
    note = format_work_note(answered())
    assert note.startswith(NOTE_HEADER) and "[1] VPN (vpn-troubleshooting.md)" in note
    assert "grounding 0.91" in note and "Review before using" in note
    assert "No knowledge base article" in format_work_note({"status": "no_answer"})
    assert "security policy" in format_work_note({"status": "blocked"})
    assert "unavailable" in format_work_note({"status": "error"})


class FakeSN:
    def __init__(self, incident):
        self.incident = incident
        self.notes = []

    def get_incident(self, sys_id):
        return self.incident

    def add_work_note(self, sys_id, note):
        self.notes.append(note)


class FakeAssistant:
    def __init__(self, result):
        self.result = result
        self.questions = []

    def ask(self, question, history=None, time_left=None):
        self.questions.append(question)
        return self.result


def test_process_incident_writes_note():
    sn = FakeSN({"number": "INC1", "short_description": "VPN fails", "active": "true"})
    summary = process_incident(sn, FakeAssistant(answered()), SYS_ID)
    assert summary["outcome"] == "noted" and summary["sources"] == ["vpn-troubleshooting.md"]
    assert "Suggested resolution" in sn.notes[0]


def test_ai_outage_retries_then_leaves_a_clear_note():
    sn = FakeSN({"number": "INC1", "short_description": "VPN fails"})
    with pytest.raises(RetryLater):
        process_incident(sn, FakeAssistant({"status": "error"}), SYS_ID, final_attempt=False)
    assert sn.notes == []  # no "unavailable" note on a retryable failure
    process_incident(sn, FakeAssistant({"status": "error"}), SYS_ID, final_attempt=True)
    assert "unavailable" in sn.notes[0]


def test_missing_closed_or_empty_incidents_get_no_note():
    assistant = FakeAssistant(answered())
    assert process_incident(FakeSN(None), assistant, SYS_ID)["outcome"] == "not_found"
    assert process_incident(FakeSN({"active": "false", "short_description": "x"}), assistant, SYS_ID)["outcome"] == "inactive"
    assert process_incident(FakeSN({"short_description": " "}), assistant, SYS_ID)["outcome"] == "empty"
    assert assistant.questions == []


# ---------------------------------------------------------------- webhook
class ConditionFailed(Exception):
    response = {"Error": {"Code": "ConditionalCheckFailedException"}}


class FakeDynamo:
    def __init__(self):
        self.items, self.deleted = set(), []

    def put_item(self, TableName, Item, ConditionExpression):
        if Item["sys_id"]["S"] in self.items:
            raise ConditionFailed()
        self.items.add(Item["sys_id"]["S"])

    def delete_item(self, TableName, Key):
        self.deleted.append(Key["sys_id"]["S"])
        self.items.discard(Key["sys_id"]["S"])


class FakeSQS:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send_message(self, QueueUrl, MessageBody):
        if self.fail:
            raise RuntimeError("sqs down")
        self.sent.append(json.loads(MessageBody))


@pytest.fixture
def webhook(monkeypatch):
    app._cache.clear()
    app._cache.update({"sn_webhook_secret": KEY, "sn_webhook_secret_at": time.time(),
                       "dynamodb": FakeDynamo(), "sqs": FakeSQS()})
    monkeypatch.setenv("SN_IDEMPOTENCY_TABLE", "t")
    monkeypatch.setenv("SN_QUEUE_URL", "q")
    monkeypatch.setenv("SN_WEBHOOK_SECRET_PARAM", "/p")
    yield app._cache
    app._cache.clear()


def signed_event(body, key=KEY, ts=None):
    body = body if isinstance(body, str) else json.dumps(body)
    ts = str(int(ts if ts is not None else time.time()))
    sig = base64.b64encode(hmac.new(base64.b64decode(key), f"{ts}.{body}".encode(), hashlib.sha256).digest()).decode()
    return {"headers": {"x-timestamp": ts, "x-signature": sig}, "body": body}


def test_signed_event_is_queued_once(webhook):
    event = signed_event({"sys_id": SYS_ID, "number": "INC0010001"})
    assert app.servicenow_event(event, None)["statusCode"] == 202
    assert webhook["sqs"].sent == [{"sys_id": SYS_ID, "number": "INC0010001"}]
    again = app.servicenow_event(signed_event({"sys_id": SYS_ID, "number": "INC0010001"}), None)
    assert again["statusCode"] == 200 and json.loads(again["body"])["status"] == "duplicate"
    assert len(webhook["sqs"].sent) == 1


def test_bad_tampered_or_old_signatures_are_rejected(webhook):
    assert app.servicenow_event({"headers": {}, "body": json.dumps({"sys_id": SYS_ID})}, None)["statusCode"] == 401
    other_key = base64.b64encode(b"x" * 32).decode()
    assert app.servicenow_event(signed_event({"sys_id": SYS_ID}, key=other_key), None)["statusCode"] == 401
    tampered = signed_event({"sys_id": SYS_ID})
    tampered["body"] = json.dumps({"sys_id": "b" * 32})
    assert app.servicenow_event(tampered, None)["statusCode"] == 401
    assert app.servicenow_event(signed_event({"sys_id": SYS_ID}, ts=time.time() - 600), None)["statusCode"] == 401
    assert webhook["sqs"].sent == []


def test_rotated_key_is_reloaded(webhook, monkeypatch):
    new_key = base64.b64encode(b"n" * 32).decode()
    webhook["sn_webhook_secret_at"] = time.time() - 120
    monkeypatch.setattr(app, "_ssm_secret", lambda name: new_key)
    assert app.servicenow_event(signed_event({"sys_id": SYS_ID}, key=new_key), None)["statusCode"] == 202


def test_invalid_body_or_sys_id(webhook):
    assert app.servicenow_event(signed_event("not json"), None)["statusCode"] == 400
    assert app.servicenow_event(signed_event({"sys_id": "123"}), None)["statusCode"] == 400
    assert app.servicenow_event(signed_event({"sys_id": SYS_ID, "number": "x; drop"}), None)["statusCode"] == 400


def test_queue_failure_releases_the_claim(webhook):
    webhook["sqs"] = FakeSQS(fail=True)
    assert app.servicenow_event(signed_event({"sys_id": SYS_ID}), None)["statusCode"] == 503
    assert webhook["dynamodb"].deleted == [SYS_ID] and SYS_ID not in webhook["dynamodb"].items


def test_missing_secret_is_503(webhook, monkeypatch):
    webhook.pop("sn_webhook_secret")
    monkeypatch.setattr(app, "_ssm_secret", lambda name: (_ for _ in ()).throw(RuntimeError("no param")))
    assert app.servicenow_event(signed_event({"sys_id": SYS_ID}), None)["statusCode"] == 503


# ---------------------------------------------------------------- worker
def record(body, receives=1, message_id="m1"):
    return {"messageId": message_id, "body": body if isinstance(body, str) else json.dumps(body),
            "attributes": {"ApproximateReceiveCount": str(receives)}}


def test_worker_notes_and_reports_only_failures(monkeypatch):
    app._cache.clear()
    sn = FakeSN({"number": "INC1", "short_description": "VPN fails"})
    app._cache.update({"servicenow": sn, "assistant": FakeAssistant(answered())})
    out = app.servicenow_worker({"Records": [record({"sys_id": SYS_ID}), record("garbage", message_id="m2")]}, None)
    assert out == {"batchItemFailures": []} and len(sn.notes) == 1

    app._cache["assistant"] = FakeAssistant({"status": "error"})
    out = app.servicenow_worker({"Records": [record({"sys_id": SYS_ID}, receives=1)]}, None)
    assert out == {"batchItemFailures": [{"itemIdentifier": "m1"}]}
    out = app.servicenow_worker({"Records": [record({"sys_id": SYS_ID}, receives=3)]}, None)
    assert out == {"batchItemFailures": []} and "unavailable" in sn.notes[-1]
    app._cache.clear()


def test_worker_drops_cached_credentials_on_401():
    app._cache.clear()

    class Unauthorized(FakeSN):
        def get_incident(self, sys_id):
            raise ServiceNowError("HTTP 401", 401)
    app._cache.update({"servicenow": Unauthorized(None), "sn_password": "old",
                       "assistant": FakeAssistant(answered())})
    out = app.servicenow_worker({"Records": [record({"sys_id": SYS_ID})]}, None)
    assert out["batchItemFailures"] and "servicenow" not in app._cache and "sn_password" not in app._cache
    app._cache.clear()
