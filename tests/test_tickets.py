import io
import json
import urllib.error

import pytest

from tickets import TicketClient, TicketError


class Opener:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.requests = body, error, []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.body).encode())


def test_creates_ticket_with_context():
    opener = Opener({"ticket_id": "t-1", "priority": "P2", "category": "network", "summary": "x"})
    out = TicketClient("https://api.example/", opener=opener).create(
        "VPN keeps dropping", "I couldn't find this.", ["vpn-troubleshooting.md"])
    assert out == {"ticket_id": "t-1", "priority": "P2", "category": "network"}
    req = opener.requests[0]
    assert req.full_url == "https://api.example/tickets" and req.get_method() == "POST"
    body = json.loads(req.data)
    assert body["subject"] == "VPN keeps dropping"
    assert "vpn-troubleshooting.md" in body["description"] and "I couldn't find this." in body["description"]


def test_long_question_subject_trimmed():
    opener = Opener({"ticket_id": "t"})
    TicketClient("https://a", opener=opener).create("x" * 500)
    assert len(json.loads(opener.requests[0].data)["subject"]) == 120


def test_errors_become_ticket_error():
    with pytest.raises(TicketError):
        TicketClient("")
    with pytest.raises(TicketError):
        TicketClient("https://a", opener=Opener(error=urllib.error.URLError("down"))).create("q")
    with pytest.raises(TicketError):
        TicketClient("https://a", opener=Opener({"error": "bad"})).create("q")


def test_more_failure_shapes_become_ticket_error():
    with pytest.raises(TicketError):  # URL without a scheme
        TicketClient("triage.example.com")
    with pytest.raises(TicketError):
        TicketClient("https://a", opener=Opener(error=ConnectionResetError())).create("q")
    with pytest.raises(TicketError):
        TicketClient("https://a", opener=Opener(["not", "a", "dict"])).create("q")
