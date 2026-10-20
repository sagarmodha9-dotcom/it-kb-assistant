"""Follow-up on tickets waiting for the caller (the "3-strike rule"), run by a scheduled Lambda.

When the agent sends something only the caller can judge (a knowledge base fix, a software install,
a device sync), the incident waits On Hold - Awaiting Caller. If the caller never answers:

    day 1  reminder 1 of 2   "did the fix work? reply yes or no"
    day 2  reminder 2 of 2   "last reminder: we'll close this if we don't hear back"
    day 3  resolved          "no response; reply any time to reopen" (close code "No resolution
                             provided": a timeout is never recorded as a confirmed fix)

A reply at any point goes through the normal caller-reply path instead (Business Rule -> agent), and a
reply after the AI resolved it reopens the ticket for a technician (agent.handle_caller_reply).

ServiceNow is the source of truth, so the job keeps no schedule of its own:
  - candidates: active, On Hold - Awaiting Caller, unassigned, and last updated by the AI's service
    account at least FOLLOWUP_HOURS ago (any human touch or caller comment changes sys_updated_by and
    takes the ticket out of the job's hands);
  - which strike: the number of "Follow-up reminder N of 2" work notes already on the ticket (work
    notes are internal, so a caller can't fake one);
  - each step is claimed in DynamoDB first, so two overlapping runs can't send the same reminder twice.
Never closed this way: urgent incidents and anything whose text looks like a security problem (handed
to a technician instead of a first reminder), tickets moved
to another team (Security, Network, ...), tickets linked to an outage, and anything a person is
assigned to. The agent normally escalates security incidents and outages instead of sending a fix;
these checks are the backstop in code. Closing is left to ServiceNow: the AI only sets Resolved, and the platform's usual
auto-close turns Resolved into Closed later, so the caller can still reopen in between.
"""
import logging
import re
import time

from agent import (AGENT_HEADER, AI_COMMENT_START, CONFIRM_MARKER, DEFAULT_TEAM, REMINDER_START, REOPEN_INVITE,
                   RESOLVED_MARKER, latest_caller_reply, looks_like_security, priority_number, with_service_desk)

logger = logging.getLogger(__name__)

MAX_REMINDERS = 2
FOLLOWUP_HOURS = 24
MAX_PER_RUN = 50
NO_RESPONSE_CLOSE_CODE = "No resolution provided"   # never recorded as a fix: nobody confirmed one
CLAIM_SECONDS = 15 * 60           # a claimed step whose write never landed can be retried after this
REMINDER_COUNT = re.compile(r"Follow-up reminder (\d+) of \d+ sent to the caller")


def waiting_query(service_user: str, min_minutes: int) -> str:
    """Encoded query for tickets the AI is waiting on. Relative dates: no time-zone conversion."""
    if not re.fullmatch(r"[\w.@-]{1,100}", service_user or ""):
        raise ValueError("invalid service user name")
    return ("active=true^state=3^hold_reason=1^assigned_toISEMPTY^parent_incidentISEMPTY"
            f"^sys_updated_by={service_user}^sys_updated_onRELATIVELE@minute@ago@{max(0, int(min_minutes))}")


def reminders_sent(work_notes: str) -> int:
    return max([int(n) for n in REMINDER_COUNT.findall(work_notes or "")] or [0])


def _wait_text(hours: int) -> str:
    return "a day" if hours == 24 else f"{hours} hours" if hours != 1 else "an hour"


def _claim(store, key: str, now: float) -> bool:
    """Claim one step for one ticket. A claim older than CLAIM_SECONDS (its write never landed) is taken
    over with a compare-and-swap, so two runs can't both win it."""
    if store is None:
        return True
    existing = store.get(key)
    if existing is None:
        return store.put(key, {"at": now}, if_absent=True)
    if now - float(existing.get("at", 0)) < CLAIM_SECONDS:
        return False
    return store.replace_if(key, {"at": now}, existing)


def follow_up(desk, incident: dict, store=None, wait_hours: int = FOLLOWUP_HOURS, now: float | None = None,
              close_code: str = NO_RESPONSE_CLOSE_CODE, close_when_silent: bool = True,
              still_current=None) -> dict:
    """One strike for one incident (fetched fresh from ServiceNow). Re-checks everything the query
    selected on, because the ticket may have changed since. `still_current()` is called right before the
    write: if the ticket changed in between (say the caller just replied), nothing is written."""
    notes = str(incident.get("work_notes") or "")
    state = str(incident.get("state") or "").lower()
    if CONFIRM_MARKER not in notes:
        return {"outcome": "not_agent_fix"}
    if not ("on hold" in state or state == "3"):
        return {"outcome": "not_waiting"}
    if str(incident.get("assigned_to") or "").strip():
        return {"outcome": "technician_assigned"}
    if not with_service_desk(incident, desk):
        return {"outcome": "not_service_desk"}   # moved to Security, Network, ...: that team decides
    if str(incident.get("parent_incident") or "").strip():
        return {"outcome": "linked_to_outage"}   # part of an outage: closes with the parent, not on a timer
    if "Caller reply at " in notes or latest_caller_reply(str(incident.get("comments") or "")):
        return {"outcome": "caller_replied"}     # the reply path handles it, not the timer

    sent = reminders_sent(notes)
    urgent = priority_number(incident.get("priority")) <= 2
    security = looks_like_security(incident)
    step = "handoff" if urgent or security else f"reminder{sent + 1}" if sent < MAX_REMINDERS else "close"
    if not _claim(store, f"followup#{incident['sys_id']}#{step}", now or time.time()):
        return {"outcome": "in_progress"}
    if still_current is not None and not still_current():
        return {"outcome": "changed_meanwhile"}  # next run sees the new state (e.g. the caller's reply)

    if urgent or security:
        # Never on a timer: hand it to people as soon as the caller has gone quiet.
        why = ("urgent (priority 1-2)" if urgent else "possibly a security incident (code check on the text)")
        desk.back_to_technician(incident, "Security" if security else DEFAULT_TEAM, "\n".join([
            AGENT_HEADER, f"No reply from the caller yet. This incident is {why}, so the AI doesn't send reminders "
            "or close it: set back to In Progress for a technician."]))
        return {"outcome": "security_to_technician" if security else "urgent_to_technician"}

    if sent < MAX_REMINDERS:
        n = sent + 1
        if n < MAX_REMINDERS:
            ask = ("Just checking in: did the fix we sent work? Please reply yes or no. If it's still not working, "
                   "tell us what's happening and a technician will take over.")
        elif close_when_silent:
            ask = (f"Last reminder: did the fix work? If we don't hear back within {_wait_text(wait_hours)}, we'll "
                   "close this ticket. You can reply any time to reopen it.")
        else:
            ask = ("Last reminder: did the fix work? If we don't hear back, a technician will review your ticket "
                   "before it's closed.")
        desk.remind(incident, comment=f"{REMINDER_START}. {ask}", work_note="\n".join([
            AGENT_HEADER, f"Follow-up reminder {n} of {MAX_REMINDERS} sent to the caller (no reply yet).",
            CONFIRM_MARKER + "."]))
        return {"outcome": f"reminder_{n}"}

    if not close_when_silent:
        desk.back_to_technician(incident, DEFAULT_TEAM, "\n".join([
            AGENT_HEADER, f"No reply from the caller after {MAX_REMINDERS} reminders. Closing on no response is "
            "turned off (ServiceNowNoResponseClose=false): review before closing."]))
        return {"outcome": "no_response_to_technician"}
    desk.resolve(incident,
                 comment=f"{AI_COMMENT_START}. We haven't heard back, so we've resolved this ticket. {REOPEN_INVITE}",
                 work_note="\n".join([
                     AGENT_HEADER, f"Action: resolved after {MAX_REMINDERS} reminders with no reply from the caller "
                     "(3-strike rule).", RESOLVED_MARKER + " (no response).", "",
                     "If the caller replies that it still isn't working, the incident reopens for the Service Desk."]),
                 close_notes=f"{RESOLVED_MARKER}: no response from the caller after {MAX_REMINDERS} reminders "
                             "(3-strike rule). Closed for no response, not confirmed fixed. The caller was told they "
                             "can reply to reopen.",
                 team=DEFAULT_TEAM, close_code=close_code)
    return {"outcome": "resolved_no_response"}


def run_followups(client, desk, store, service_user: str, min_minutes: int, wait_hours: int = FOLLOWUP_HOURS,
                  time_left=None, close_code: str = NO_RESPONSE_CLOSE_CODE, close_when_silent: bool = True) -> list[dict]:
    """One pass over every ticket that has waited long enough. One failure never stops the others."""
    time_left = time_left or (lambda: float("inf"))
    rows = client.query("incident", waiting_query(service_user, min_minutes), fields="sys_id,number",
                        limit=MAX_PER_RUN, order_by="sys_updated_on")
    results = []
    for row in rows:
        if time_left() < 15:
            results.append({"number": row.get("number"), "outcome": "deferred_to_next_run"})
            continue
        sys_id = str(row.get("sys_id", ""))
        try:
            incident = client.get_incident(sys_id)
            if incident is None:
                results.append({"number": row.get("number"), "outcome": "not_found"})
                continue
            stamp = incident.get("sys_updated_on")
            outcome = follow_up(desk, {**incident, "sys_id": sys_id}, store, wait_hours, close_code=close_code,
                                close_when_silent=close_when_silent,
                                still_current=lambda: (client.get_incident(sys_id) or {}).get("sys_updated_on") == stamp)
        except Exception as error:
            logger.exception("Follow-up failed on %s", row.get("number"))
            outcome = {"outcome": "error", "error": type(error).__name__}
        results.append({"number": row.get("number"), **outcome})
    return results
