"""Public demo: play the caller and watch the ticket's next steps (same policy code, dry run)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lifecycle import CHOICES, play  # noqa: E402


def states(action, choice):
    return [(e["when"], e["state"], e["outcome"]) for e in play(action, choice, "VPN keeps dropping")]


def test_caller_confirms_the_fix():
    assert states("awaiting_caller", "fixed") == [("Caller replies", "Resolved · Solution provided", "resolved")]


def test_caller_says_still_broken():
    assert states("awaiting_caller", "broken") == [("Caller replies", "In Progress · Service Desk",
                                                    "back_to_technician")]


def test_silence_runs_the_3_strike_rule():
    events = play("awaiting_caller", "silent", "VPN keeps dropping")
    assert [e["outcome"] for e in events] == ["reminder_1", "reminder_2", "resolved_no_response"]
    assert events[-1]["state"].endswith("No resolution provided") and "never recorded as a fix" in events[-1]["summary"]
    assert "Last reminder" in events[1]["comment"]


def test_reply_on_an_ai_resolved_ticket():
    assert states("resolved", "thanks")[0][2] == "still_resolved"
    events = play("resolved", "broken", "I'm locked out")
    assert events[0]["outcome"] == "reopened" and "reopened your ticket" in events[0]["comment"]


def test_only_offered_where_the_ai_can_act():
    assert set(CHOICES) == {"awaiting_caller", "resolved"}     # escalations and approvals stay with people
    with pytest.raises(ValueError):
        play("escalated", "silent", "x")
    with pytest.raises(ValueError):
        play("resolved", "silent", "x")                          # nothing to wait for on a resolved ticket


def test_dry_run_needs_no_network_or_model(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    for action, choices in CHOICES.items():
        for key, _ in choices:
            assert play(action, key, "x")
