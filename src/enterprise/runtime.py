"""Wiring the enterprise toolkit into the ServiceNow worker and the public demo, and finishing
approved actions.

Approval decisions come from ServiceNow: an approver sets the incident's Approval field to Approved
or Rejected, a Business Rule sends a signed "approval" event, and handle_approval_decision checks
that this person may approve (right role, not the requester), then runs exactly the approved action
(no model involved). If the action verified its result in the system (license, group access) the
incident is resolved right away; otherwise the caller is asked to confirm. A rejection goes back to
a technician.
"""
import copy
import re

from agent import (AI_COMMENT_START, AGENT_HEADER, CONFIRM_MARKER, DEFAULT_TEAM, _is_resolved, action_lines,
                   executed_lines,
                   looks_like_security, simulated, simulation_notice, verified_fix, verified_resolution,
                   with_service_desk)
from .framework import AuditLog, Context, MemoryStore, Toolkit, ToolError
from .sim import SimBackends
from .workflows import build_registry

DEMO_CALLER = "alex.rivera"     # the public demo always acts as this fictional employee


def caller_id(incident: dict) -> str:
    """The caller's directory id: their ServiceNow user name, exactly (never guessed from a display name).
    No user name -> "" (unknown caller: every account-scoped tool is refused)."""
    value = str(incident.get("caller_id.user_name") or "").strip().lower()
    return value if re.fullmatch(r"[a-z0-9._@-]{1,100}", value) else ""


def build_toolkit(ticket: str, caller: str, store=None, backends=None, simulated_only=False, sleep=None) -> Toolkit:
    ctx = Context(ticket=ticket, caller=caller, store=store or MemoryStore(), audit=AuditLog(),
                  backends=backends or SimBackends(), simulated_only=simulated_only)
    if sleep:
        ctx.sleep = sleep
    return Toolkit(build_registry(), ctx)


def demo_toolkit() -> Toolkit:
    """Public demo: fictional company, in-memory state per request, simulated tools only, no waits."""
    return build_toolkit("DEMO", DEMO_CALLER, simulated_only=True, sleep=lambda s: None)


def demo_llm(llm):
    """A copy of the model client with a short timeout, so the demo always answers within API Gateway's limit."""
    clone = copy.copy(llm)
    for part in ("primary", "backup"):
        if hasattr(clone, part):
            setattr(clone, part, demo_llm(getattr(clone, part)))
    if hasattr(clone, "timeout"):
        clone.timeout = min(clone.timeout, 8)
    return clone


def handle_approval_decision(desk, incident: dict, approver: str, approved: bool, store, backends=None,
                             resolve_simulated: bool = False) -> dict:
    """Run (or cancel) every pending approval on this incident after an approver's decision."""
    ticket = str(incident.get("number") or "")
    pending = (store.get(f"pending#{ticket}") or {"ids": []})["ids"]
    if not pending:
        return {"outcome": "no_pending_approval"}
    caller = caller_id(incident)
    kit = build_toolkit(ticket, caller, store=store, backends=backends)
    results = []
    for approval_id in pending:
        try:
            record = kit.approvals.decide(approval_id, approver, approved, kit.ctx.audit)
        except ToolError as error:
            results.append({"id": approval_id, "error": str(error)})
            continue
        if approved:
            try:
                result = kit.run_approved(approval_id)
            except Exception as error:   # never strand the incident: report it as a failure
                result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            results.append({"id": approval_id, "tool": record["tool"], "ok": bool(result.get("ok")),
                            "error": result.get("error") or result.get("reason")})
        else:
            results.append({"id": approval_id, "tool": record["tool"], "rejected": True})
    undecided = [r["id"] for r in results if r.get("error") and not r.get("tool")]
    store.put(f"pending#{ticket}", {"ids": undecided})
    if not any(r.get("tool") for r in results):
        return {"outcome": "not_authorized", "results": results}   # e.g. the requester tried to approve

    done = [r for r in results if r.get("ok")]
    failed = [r for r in results if r.get("tool") and not r.get("ok") and not r.get("rejected")]
    rejected = [r for r in results if r.get("rejected")]
    audit = "\n".join(f"- {e['action']}: {e.get('tool', '')}" for e in kit.ctx.audit.entries)
    # Close only if nobody else has taken it over since the request: no technician, not linked to an outage,
    # still with the Service Desk, not already resolved, and nothing that looks like a security problem.
    untouched = not str(incident.get("assigned_to") or "").strip() and not str(incident.get("parent_incident") or "").strip() \
        and with_service_desk(incident, desk) and not _is_resolved(str(incident.get("state") or "").lower()) \
        and not looks_like_security(incident)
    if done and not failed and not undecided and untouched and verified_fix(kit.ctx.executed, incident, resolve_simulated):
        # The approved action read its end state back from the system: resolve now, the caller can reopen.
        steps = [f"- approved by {approver}"] + [f"- {e['action']}: {e.get('tool', '')}" for e in kit.ctx.audit.entries]
        comment, note, close_notes = verified_resolution(kit.ctx.caller_messages, kit.ctx.executed, steps)
        desk.resolve(incident, comment=comment, work_note=note, close_notes=close_notes, team=DEFAULT_TEAM)
        return {"outcome": "executed_resolved", "results": results}
    if done and not failed:
        comment = "\n".join([AI_COMMENT_START + ". Your request was approved and completed:", "",
                             *simulation_notice(kit.ctx.executed), *action_lines(kit.ctx.caller_messages, kit.ctx.executed), "",
                             "Please reply to let us know: is it working now? If yes, we'll close the ticket."])
        note = "\n".join([AGENT_HEADER, f"Approved by {approver}; the approved action(s) ran"
                          + (" [SIMULATED backends: no real change]" if simulated(kit.ctx.executed) else "") + ".",
                          CONFIRM_MARKER + ".", "", "Actions executed:", *executed_lines(kit.ctx.executed), "",
                          "Audit:", audit])
        # keep the ticket with whoever owns it now (Security, another team); only an unowned one goes to the Service Desk
        desk.send_fix(incident, comment=comment, work_note=note,
                      team=DEFAULT_TEAM if with_service_desk(incident, desk) else None)
        return {"outcome": "executed_awaiting_caller", "results": results}
    if rejected:
        desk.back_to_technician(incident, DEFAULT_TEAM, "\n".join([AGENT_HEADER, f"Rejected by {approver}.", "", audit]),
                                comment=f"{AI_COMMENT_START}. Your request was not approved by {approver}. "
                                        "A technician will follow up with options.")
        return {"outcome": "rejected", "results": results}
    desk.back_to_technician(incident, DEFAULT_TEAM, "\n".join(
        [AGENT_HEADER, f"Approved by {approver}, but the action failed: "
         + "; ".join(f"{r['tool']}: {r.get('error')}" for r in failed), "Needs a technician.", "", audit]))
    return {"outcome": "failed_after_approval", "results": results}
