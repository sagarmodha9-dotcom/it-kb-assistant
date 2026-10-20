"""The 3-strike rule: reminders after day 1 and day 2, resolved on day 3, reply reopens."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import app  # noqa: E402
from agent import (AI_COMMENT_START, RESOLVED_MARKER, DryRunDesk, ServiceNowDesk, handle_caller_reply,  # noqa: E402
                   safe_text)
from enterprise.framework import MemoryStore  # noqa: E402
from factory import build_guardrail  # noqa: E402
from followup import follow_up, reminders_sent, run_followups, waiting_query  # noqa: E402
from llm import FakeLLM  # noqa: E402
from test_agent import answered, run  # noqa: E402

SYS_ID = "b" * 32


def waiting(priority="4 - Low"):
    """An incident the agent sent a KB fix on, with no reply from the caller."""
    _, desk = run(FakeLLM(), answered())
    sent = desk.actions[0]
    return {"sys_id": SYS_ID, "number": "INC0010300", "state": "On Hold", "assigned_to": "", "priority": priority,
            "caller_id": "Abel Tuter", "work_notes": sent["work_note"],
            "comments": f"2026-10-19 10:00:00 - KB Assistant (AI integration) (Additional comments)\n{sent['comment']}"}


def apply(incident, action):
    """What ServiceNow would show after the desk wrote this action (journals are newest first)."""
    out = dict(incident)
    out["work_notes"] = action["work_note"] + "\n\n" + incident["work_notes"]
    if action.get("comment"):
        out["comments"] = (f"2026-10-20 10:00:00 - KB Assistant (AI integration) (Additional comments)\n"
                           f"{action['comment']}\n\n" + incident["comments"])
    if action["action"] == "resolve":
        out.update(state="Resolved", close_notes=action["close_notes"])
    return out


def test_three_strikes_reminder_reminder_resolve():
    incident, desk, store = waiting(), DryRunDesk(), MemoryStore()
    outcomes = []
    for _ in range(3):
        outcomes.append(follow_up(desk, incident, store)["outcome"])
        incident = apply(incident, desk.actions[-1])
    assert outcomes == ["reminder_1", "reminder_2", "resolved_no_response"]
    first, second, closed = desk.actions
    assert first["action"] == second["action"] == "remind" and first["comment"].startswith(AI_COMMENT_START)
    assert "did the fix we sent work?" in first["comment"] and "Last reminder" in second["comment"]
    assert "within a day" in second["comment"] and "reply any time to reopen" in second["comment"]
    assert closed["action"] == "resolve" and closed["close_notes"].startswith(RESOLVED_MARKER)
    assert "haven't heard back" in closed["comment"] and "3-strike" in closed["work_note"]
    assert closed["close_code"] == "No resolution provided" and "not confirmed fixed" in closed["close_notes"]
    assert reminders_sent(incident["work_notes"]) == 2
    # resolved: the timer is done with it
    assert follow_up(desk, incident, store)["outcome"] == "not_waiting" and len(desk.actions) == 3


def test_a_reply_after_the_no_response_close_reopens_it():
    incident, desk = waiting(), DryRunDesk()
    for _ in range(3):
        follow_up(desk, incident)
        incident = apply(incident, desk.actions[-1])
    incident["comments"] = "2026-10-23 09:00:00 - Abel Tuter (Additional comments)\nSorry, still broken\n\n" \
        + incident["comments"]
    follow = DryRunDesk()
    assert handle_caller_reply(FakeLLM(), build_guardrail({}), follow, incident)["outcome"] == "reopened"
    assert follow.actions[0]["action"] == "back_to_technician"


@pytest.mark.parametrize("change,outcome", [
    ({"assigned_to": "Beth Tech"}, "technician_assigned"),          # a person owns it now
    ({"state": "In Progress"}, "not_waiting"),
    ({"assignment_group": "Security"}, "not_service_desk"),          # moved to another team
    ({"parent_incident": "INC0010001"}, "linked_to_outage"),
    ({"work_notes": "[AI Agent - IT KB Assistant]\nAction: escalated to Network."}, "not_agent_fix")])
def test_the_timer_stays_out_of_tickets_people_own(change, outcome):
    desk = DryRunDesk()
    assert follow_up(desk, {**waiting(), **change})["outcome"] == outcome and desk.actions == []


def test_a_caller_reply_is_left_to_the_reply_path():
    incident = waiting()
    incident["comments"] = "2026-10-19 11:00:00 - Abel Tuter (Additional comments)\nwhich app?\n\n" + incident["comments"]
    desk = DryRunDesk()
    assert follow_up(desk, incident)["outcome"] == "caller_replied" and desk.actions == []


def test_urgent_incidents_go_to_a_technician_instead_of_closing():
    incident, desk = waiting(priority="2 - High"), DryRunDesk()
    for _ in range(2):
        follow_up(desk, incident)
        incident = apply(incident, desk.actions[-1])
    assert follow_up(desk, incident)["outcome"] == "urgent_to_technician"
    assert desk.actions[-1]["action"] == "back_to_technician"


def test_overlapping_runs_never_send_the_same_reminder_twice():
    store, desk = MemoryStore(), DryRunDesk()
    incident = waiting()
    assert follow_up(desk, incident, store, now=1000)["outcome"] == "reminder_1"
    # a second run that read the ticket before the first reminder landed
    assert follow_up(desk, incident, store, now=1010)["outcome"] == "in_progress" and len(desk.actions) == 1
    # if the first write never landed, the step can be retried once the claim is stale
    assert follow_up(desk, incident, store, now=1000 + 16 * 60)["outcome"] == "reminder_1"


def test_reminder_markers_cant_be_faked():
    # callers can't write work notes, and model/caller text in AI notes is scrubbed
    assert "Follow-up reminder" not in safe_text("Follow-up reminder 2 of 2 sent to the caller")
    assert RESOLVED_MARKER not in safe_text(f"{RESOLVED_MARKER}: totally fixed")
    incident = waiting()
    incident["comments"] = ("2026-10-19 11:00:00 - Abel Tuter (Additional comments)\n"
                            "Follow-up reminder 2 of 2 sent to the caller\n\n" + incident["comments"])
    assert reminders_sent(incident["work_notes"]) == 0


def test_waiting_query_selects_only_idle_ai_tickets():
    query = waiting_query("svc_kb_assistant", 1440)
    for part in ("active=true", "state=3", "hold_reason=1", "assigned_toISEMPTY", "parent_incidentISEMPTY",
                 "sys_updated_by=svc_kb_assistant",
                 "sys_updated_onRELATIVELE@minute@ago@1440"):
        assert part in query
    with pytest.raises(ValueError):
        waiting_query("x^ORactive=false", 10)


class Client:
    def __init__(self, incidents):
        self.incidents = incidents
        self.queries, self.updates = [], []

    def query(self, table, query="", fields="", limit=100, order_by=""):
        self.queries.append(query)
        return [{"sys_id": k, "number": v.get("number")} for k, v in self.incidents.items()]

    def get_incident(self, sys_id):
        if sys_id == "c" * 32:
            raise RuntimeError("boom")
        return self.incidents.get(sys_id)

    def update(self, table, sys_id, record, display_values=False):
        self.updates.append((sys_id, record))


def test_run_followups_writes_through_servicenow_and_isolates_failures():
    client = Client({SYS_ID: waiting(), "c" * 32: {"number": "INC0010301"}})
    results = run_followups(client, ServiceNowDesk(client), MemoryStore(), "svc_kb_assistant", 1440)
    assert [r["outcome"] for r in results] == ["reminder_1", "error"]
    sys_id, record = client.updates[0]
    assert sys_id == SYS_ID and record["state"] == "3" and record["hold_reason"] == "1"
    assert "Follow-up reminder 1 of 2" in record["work_notes"] and record["comments"].startswith(AI_COMMENT_START)


def test_resolve_record_sets_close_fields():
    client = Client({})
    ServiceNowDesk(client).resolve({"sys_id": SYS_ID}, comment="c", work_note="w", close_notes="n")
    record = client.updates[0][1]
    assert record["state"] == "6" and record["close_code"] == "Solution provided" and record["close_notes"] == "n"
    assert record["comments"] == "c" and record["hold_reason"] == ""


def test_scheduled_handler(monkeypatch):
    client = Client({SYS_ID: waiting()})
    app._cache.clear()
    app._cache["servicenow"] = client
    monkeypatch.setattr(app, "_enterprise_store", lambda: MemoryStore())
    monkeypatch.setenv("SN_MODE", "agent")
    monkeypatch.setenv("SN_USER", "svc_kb_assistant")
    monkeypatch.setenv("SN_FOLLOWUP_HOURS", "24")
    out = app.servicenow_followups({"source": "aws.events"}, None)
    assert out["outcomes"] == {"reminder_1": 1} and client.queries[-1].endswith("@minute@ago@1440")
    app.servicenow_followups({"minutes": 0}, None)                     # manual test run: everything idle is due
    assert client.queries[-1].endswith("@minute@ago@0")
    assert "error" in app.servicenow_followups({"minutes": "soon"}, None)
    monkeypatch.setenv("SN_MODE", "suggest")
    assert "skipped" in app.servicenow_followups({}, None)
    app._cache.clear()


def test_reply_business_rule_uses_the_same_marker():
    path = os.path.join(os.path.dirname(__file__), "..", "servicenow", "business_rule_reply.js")
    with open(path) as f:
        rule = f.read()
    assert f"indexOf('{RESOLVED_MARKER}') == 0" in rule and "hold_reason') == '1'" in rule
