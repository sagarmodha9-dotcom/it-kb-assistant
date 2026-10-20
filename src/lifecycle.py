"""What happens to a ticket after the agent's first move, for the public demo.

A visitor picks how the caller responds ("it works", "still broken", or no reply at all) and this runs
the SAME policy code the ServiceNow integration runs (agent.handle_caller_reply, followup.follow_up)
against a DryRunDesk: nothing is written anywhere, no model is called (replies are classified by the
offline keyword classifier), and the result is the timeline of what the AI would do to the ticket.
"""
from agent import (AGENT_HEADER, AI_COMMENT_START, CONFIRM_MARKER, REOPEN_INVITE, RESOLVED_MARKER, UPDATE_START,
                   DryRunDesk, handle_caller_reply)
from followup import NO_RESPONSE_CLOSE_CODE, follow_up
from guardrails import LocalGuardrail
from llm import FakeLLM

CALLER = "Alex Rivera"
AI_NAME = "KB Assistant (AI integration)"

# The choices that make sense after each first move. Escalations and approvals stay with people.
CHOICES = {
    "awaiting_caller": [("fixed", "Caller replies “Yes, that fixed it”"),
                        ("broken", "Caller replies “No, it's still broken”"),
                        ("silent", "Caller never replies (fast-forward 3 days)")],
    "resolved": [("thanks", "Caller replies “Thanks, it works”"),
                 ("broken", "Caller replies “No, it's still broken”")],
}
# Journal timestamps in ServiceNow's format (the reply parser reads them); the timeline shows "Day N" labels.
STAMP = {0: "2026-01-05 09:00:00", "reply": "2026-01-05 10:00:00", 1: "2026-01-06 09:00:00",
         2: "2026-01-07 09:00:00", 3: "2026-01-08 09:00:00"}
REPLIES = {"fixed": "Yes, that fixed it, thanks", "thanks": "Thanks, it works now", "broken": "No, it's still broken"}


def _entry(when: str, who: str, text: str) -> str:
    return f"{when} - {who} (Comments)\n{text}\n\n"


def start(action: str, question: str) -> dict:
    """The ticket as the agent left it after its first move (On Hold waiting for the caller, or resolved)."""
    incident = {"sys_id": "0" * 32, "number": "DEMO0001", "short_description": question[:160], "priority": "4 - Low",
                "caller_id": CALLER, "assigned_to": "", "assignment_group": "Service Desk"}
    if action == "resolved":
        return {**incident, "state": "Resolved",
                "close_notes": f"{RESOLVED_MARKER}: verified fix [SIMULATED backends: no real change].",
                "work_notes": f"{AGENT_HEADER}\n{RESOLVED_MARKER} (verified fix).",
                "comments": _entry(STAMP[0], AI_NAME, f"{UPDATE_START}. Resolved. {REOPEN_INVITE}")}
    return {**incident, "state": "On Hold",
            "work_notes": f"{AGENT_HEADER}\nAction: sent the results to the caller; On Hold - Awaiting Caller.\n"
                          f"{CONFIRM_MARKER}.",
            "comments": _entry(STAMP[0], AI_NAME, f"{AI_COMMENT_START}. Please reply: did this fix it?")}


def _apply(incident: dict, action: dict, when: str) -> dict:
    """What ServiceNow would show after the desk wrote this action (journals are newest first)."""
    out = dict(incident)
    out["work_notes"] = action.get("work_note", "") + "\n\n" + incident["work_notes"]
    if action.get("comment"):
        out["comments"] = _entry(when, AI_NAME, action["comment"]) + incident["comments"]
    if action["action"] == "resolve":
        out.update(state="Resolved", close_notes=action.get("close_notes", ""))
    elif action["action"] == "back_to_technician":
        out.update(state="In Progress", assignment_group=action.get("team") or "Service Desk")
    return out


def _event(when: str, action: dict | None, outcome: str) -> dict:
    if action is None:
        return {"when": when, "state": "Resolved (unchanged)", "outcome": outcome, "comment": "",
                "summary": "The caller confirmed it works, so the AI leaves the ticket resolved and writes nothing."}
    kind = action["action"]
    if kind == "remind":
        state, summary = "On Hold - Awaiting Caller", "No reply yet: the AI sends a reminder and keeps waiting."
    elif kind == "resolve" and action.get("close_code") == NO_RESPONSE_CLOSE_CODE:
        state = f"Resolved · {NO_RESPONSE_CLOSE_CODE}"
        summary = "Still no reply after two reminders: resolved as no response, never recorded as a fix."
    elif kind == "resolve":
        state, summary = "Resolved · Solution provided", "The caller confirmed the fix, so the AI resolves it."
    else:
        state = f"In Progress · {action.get('team') or 'Service Desk'}"
        summary = ("Reopened for a technician, with the caller's reply quoted in the work note."
                   if outcome == "reopened" else "Handed to a technician with the caller's reply.")
    return {"when": when, "state": state, "outcome": outcome, "comment": action.get("comment", ""),
            "summary": summary}


def play(action: str, choice: str, question: str) -> list[dict]:
    """Timeline of what the AI does next. Each event: when, state, outcome, comment, summary."""
    if choice not in dict(CHOICES.get(action, [])):
        raise ValueError("not a choice for this ticket")
    incident, desk, events = start(action, question), DryRunDesk(), []
    if choice == "silent":
        for day in (1, 2, 3):
            before = len(desk.actions)
            outcome = follow_up(desk, incident)["outcome"]
            if len(desk.actions) == before:
                break
            when = f"Day {day}"
            events.append(_event(when, desk.actions[-1], outcome))
            incident = _apply(incident, desk.actions[-1], STAMP[day])
        return events
    incident = {**incident, "comments": _entry(STAMP["reply"], CALLER, REPLIES[choice]) + incident["comments"]}
    outcome = handle_caller_reply(FakeLLM(), LocalGuardrail(), desk, incident)["outcome"]
    events.append(_event("Caller replies", desk.actions[-1] if desk.actions else None, outcome))
    return events
