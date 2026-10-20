"""The public demo's ticket desk: the real agent working a demo ticket, with nothing real behind it.

What is real here (the same code the private ServiceNow integration runs):
  - agent.TicketAgent (decision rules, kb1-only fixes, priority caps, escalation notes),
  - agent.ServiceNowDesk (the exact field writes the worker sends to ServiceNow's Table API),
  - the enterprise framework (registry -> policy -> identity -> approvals -> executor -> audit),
  - agent.handle_caller_reply, enterprise.runtime.handle_approval_decision, followup.follow_up,
  - agent.new_incident_check (the worker's duplicate/ownership guard).
What is not:
  - SandboxServiceNow below stands in for the ServiceNow Table API: an in-memory record per visitor
    session. There is no URL, no credential and no network code in this module.
  - Directory, device, license, mailbox and HR actions run on enterprise/sim.py (a fictional company),
    through a toolkit built with simulated_only=True.
  - Tool choices for sample tickets are replayed from a script; for custom tickets they are the live
    model's choices, returned by the rate-limited public /demo/ask route (a dry run on the server) and
    replayed here through the same code so the ticket stays interactive.
  - Caller replies are classified by the offline keyword classifier (production uses Claude).
"""
import datetime
import json

from agent import TEAMS, ServiceNowDesk, TicketAgent, handle_caller_reply, new_incident_check
from enterprise.framework import MemoryStore
from enterprise.runtime import build_toolkit, handle_approval_decision
from enterprise.sim import SimBackends
from enterprise.workflows import build_registry
from followup import follow_up
from guardrails import LocalGuardrail
from llm import FakeLLM
from servicenow import incident_question
from showcase import CAPABILITIES

AI_AUTHOR = "KB Assistant (AI integration)"
SERVICE_USER = "svc_kb_assistant"
STATE = {"1": "New", "2": "In Progress", "3": "On Hold", "6": "Resolved", "7": "Closed", "8": "Canceled"}
HOLD = {"1": "Awaiting Caller", "": ""}
APPROVAL = {"not requested": "Not Yet Requested", "requested": "Requested", "approved": "Approved",
            "rejected": "Rejected"}
PRIORITY = {1: "1 - Critical", 2: "2 - High", 3: "3 - Moderate", 4: "4 - Low", 5: "5 - Planning"}
MATRIX = {("1", "1"): 1, ("1", "2"): 2, ("2", "1"): 2, ("2", "2"): 3, ("1", "3"): 3, ("3", "1"): 3,
          ("2", "3"): 4, ("3", "2"): 4, ("3", "3"): 4}
MAX_CUSTOM = 200
REGISTRY = build_registry()
DESK_TOOLS = {"send_fix_to_caller", "escalate_incident", "find_similar_incidents", "link_to_outage",
              "request_approval", "complete_and_confirm"}


# ---------------------------------------------------------------- the stand-in for ServiceNow
class SandboxServiceNow:
    """In-memory incident with ServiceNow's semantics (state codes, journals, impact x urgency priority,
    reference fields). Implements only what ServiceNowDesk and the agent call: get_incident, query, update."""

    def __init__(self, record: dict, recent: list | None = None, start: datetime.datetime | None = None):
        self.rec = record
        self.recent = recent or []
        self.clock = start or datetime.datetime(2026, 1, 5, 9, 0, 0)
        self.journal = []        # {"at", "author", "kind": comments|work_notes, "text"}
        self.changes = []        # {"at", "author", "fields": {name: (old, new)}}
        self.actor = AI_AUTHOR
        self.mod_count = 0

    # time
    def now(self) -> str:
        return self.clock.strftime("%Y-%m-%d %H:%M:%S")

    def tick(self, **delta):
        self.clock += datetime.timedelta(**delta)

    # reads
    def _journal_text(self, kind: str) -> str:
        label = "Comments" if kind == "comments" else "Work notes"
        return "".join(f"{e['at']} - {e['author']} ({label})\n{e['text']}\n\n"
                       for e in reversed(self.journal) if e["kind"] == kind)

    def display(self) -> dict:
        r = self.rec
        return {"sys_id": r["sys_id"], "number": r["number"], "short_description": r["short_description"],
                "description": r.get("description", ""), "active": "false" if r["state"] in ("7", "8") else "true",
                "state": STATE[r["state"]], "hold_reason": HOLD.get(r.get("hold_reason", ""), ""),
                "priority": PRIORITY[r["priority"]], "impact": r["impact"], "urgency": r["urgency"],
                "assignment_group": r.get("assignment_group", ""), "assigned_to": r.get("assigned_to", ""),
                "caller_id": r["caller_name"], "caller_id.user_name": r["caller_user"],
                "approval": APPROVAL[r.get("approval", "not requested")], "close_code": r.get("close_code", ""),
                "close_notes": r.get("close_notes", ""), "parent_incident": r.get("parent_incident", ""),
                "work_notes": self._journal_text("work_notes"), "comments": self._journal_text("comments"),
                "sys_updated_on": f"{self.now()}#{self.mod_count}", "sys_mod_count": str(self.mod_count)}

    def get_incident(self, sys_id: str) -> dict | None:
        return self.display() if sys_id == self.rec["sys_id"] else None

    def query(self, table: str, query: str = "", fields: str = "", limit: int = 100, order_by: str = "") -> list:
        if table == "sys_user_group":
            name = query.split("=", 1)[-1]
            return [{"sys_id": f"group:{name}"}] if name in TEAMS else []
        if table == "incident" and query.startswith("number="):
            number = query.split("=", 1)[1]
            return [{"sys_id": f"incident:{number}"}] if any(i["number"] == number for i in self.recent) else []
        if table == "incident":
            return [{**i, "caller_id.name": i.get("caller", "Someone Else")} for i in self.recent][:limit]
        return []

    # writes
    def update(self, table: str, sys_id: str, record: dict, display_values: bool = False) -> dict:
        if table != "incident" or sys_id != self.rec["sys_id"]:
            raise ValueError("the sandbox only holds this one demo incident")
        before = dict(self.rec)
        for field, value in record.items():
            value = str(value)
            if field in ("comments", "work_notes"):
                if value:
                    self.journal.append({"at": self.now(), "author": self.actor, "kind": field, "text": value})
            elif field == "assignment_group":
                self.rec["assignment_group"] = value.split(":", 1)[-1]
            elif field == "parent_incident":
                self.rec["parent_incident"] = value.split(":", 1)[-1]
            else:
                self.rec[field] = value
        if "impact" in record or "urgency" in record:
            self.rec["priority"] = MATRIX.get((self.rec["impact"], self.rec["urgency"]), 4)
        if self.rec["state"] != "3":
            self.rec["hold_reason"] = ""
        tracked = ("state", "hold_reason", "priority", "assignment_group", "approval", "close_code", "parent_incident")
        diff = {f: (before.get(f, ""), self.rec.get(f, "")) for f in tracked if before.get(f, "") != self.rec.get(f, "")}
        if diff:
            self.changes.append({"at": self.now(), "author": self.actor, "fields": diff})
        self.mod_count += 1
        return {"sys_id": sys_id}

    def human(self, author: str, **fields):
        """A person (caller, approver) edits the ticket in the ServiceNow UI."""
        previous, self.actor = self.actor, author
        try:
            self.update("incident", self.rec["sys_id"], fields)
        finally:
            self.actor = previous


# ---------------------------------------------------------------- model stand-ins
class ScriptedPlanner:
    """Replays a fixed list of tool calls (sample tickets: authored; custom tickets: the live model's)."""
    provider = "replay"

    def __init__(self, calls):
        # One model turn per group: the live model may ask for several tools in one turn, and the replay keeps
        # those turns so the agent's step limit counts the same way it did on the server. Scripted samples
        # without a turn number get one call per turn.
        self.turns = []
        last = object()
        for n, c in enumerate(calls):
            name, args = (c["name"], c.get("input") or {}) if isinstance(c, dict) else c
            turn = c.get("turn", f"s{n}") if isinstance(c, dict) else f"s{n}"
            if turn != last or not self.turns:
                self.turns.append([])
            self.turns[-1].append((name, args))
            last = turn
        self.seen = 0
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    def chat(self, system, messages, tools, max_tokens=700):
        if not self.turns:
            return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}
        content = []
        for name, args in self.turns.pop(0):
            self.seen += 1
            content.append({"type": "tool_use", "id": f"c{self.seen}", "name": name, "input": args})
        return {"content": content, "stop_reason": "tool_use"}


class UnavailablePlanner(ScriptedPlanner):
    """The model didn't return in time: the agent's own failure path takes over."""

    def __init__(self):
        super().__init__([])

    def chat(self, *a, **k):
        raise TimeoutError("model did not respond in time")


class RecordedSearch:
    def __init__(self, result):
        self.result = result
        self.llm, self.guardrail = FakeLLM(), LocalGuardrail()

    def ask(self, question, history=None, time_left=None):
        return {**self.result, "question": question}


class RecordingAgent(TicketAgent):
    """The real agent, with every tool call and the exact result the model saw recorded for the timeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def _dispatch(self, incident, question, state, name, args, time_left):
        output = super()._dispatch(incident, question, state, name, args, time_left)
        self.records.append({"tool": name, "input": args, "output": output})
        return output


# ---------------------------------------------------------------- sample tickets
def _kb(source, title, answer, passage):
    return {"status": "answered", "answer": answer, "provider": "recorded", "usage": {}, "guardrail": {},
            "sources": [{"n": 1, "title": title, "source": source, "score": None, "snippet": passage}]}


NO_KB = {"status": "no_answer", "answer": "", "sources": [], "provider": "recorded", "usage": {}, "guardrail": {}}

SAMPLES = [
    {"id": "lockout", "title": "Locked out of my account", "category": "Identity", "expect": "Verified fix, resolved",
     "caller": ("alex.rivera", "Alex Rivera"), "text": "I'm locked out of my account",
     "kb": _kb("password-reset.md", "Reset or Change Your Password",
               "Accounts lock after 10 failed sign-in attempts and unlock automatically after 15 minutes. If you need "
               "access sooner, call the Service Desk at extension 4357.",
               "Accounts lock after 10 failed sign-in attempts and unlock automatically after 15 minutes. If you need "
               "access sooner, call the Service Desk at extension 4357."),
     "plan": [("get_account_status", {}), ("verify_caller_identity", {}), ("unlock_account", {}),
              ("complete_and_confirm", {})]},
    {"id": "vpn", "title": "VPN keeps disconnecting", "category": "Network", "expect": "Guidance, caller confirms",
     "caller": ("alex.rivera", "Alex Rivera"), "text": "My VPN keeps disconnecting",
     "kb": _kb("vpn-troubleshooting.md", "VPN: Connect and Troubleshoot",
               "Disconnect, wait 10 seconds and reconnect; if it continues, restart the laptop [1]. On public Wi-Fi, "
               "switch to a phone hotspot, since many public networks block VPN traffic [1].",
               "Connected but can't reach internal sites: disconnect, wait 10 seconds, reconnect. If it continues, "
               "restart the laptop. \"Gateway not responding\": switch networks (for example from hotel Wi-Fi to a "
               "phone hotspot). Many public networks block VPN traffic."),
     "plan": [("get_service_health", {"service": "vpn"}), ("get_devices", {}), ("find_similar_incidents", {}),
              ("send_fix_to_caller", {"search_id": "kb1"})]},
    {"id": "access", "title": "Access to a SharePoint site", "category": "Access", "expect": "Owner approval",
     "caller": ("taylor.kim", "Taylor Kim"), "text": "I need access to the Finance Reports SharePoint site",
     "kb": _kb("software-access-request.md", "Request Software or Access",
               "Access requests are approved by the data owner [1].",
               "Use the same form to request access to shared drives, shared mailboxes, or applications. The data "
               "owner must approve."),
     "plan": [("find_access_group", {"resource": "Finance Reports SharePoint site"}), ("verify_caller_identity", {}),
              ("add_group_member", {"group": "Finance-Reports"}), ("request_approval", {})]},
    {"id": "outage", "title": "VPN down for several people", "category": "Outage", "expect": "Escalated, priority 2",
     "caller": ("alex.rivera", "Alex Rivera"), "text": "VPN keeps disconnecting",
     "kb": _kb("vpn-troubleshooting.md", "VPN: Connect and Troubleshoot",
               "If more than one coworker reports the same VPN problem at the same time, it may be an outage [1].",
               "Open a ticket if more than one coworker reports the same VPN problem at the same time (that may be "
               "an outage)."),
     "recent": [{"number": "INC0010001", "short_description": "VPN disconnecting all morning", "caller": "Morgan Fox"},
                {"number": "INC0010002", "short_description": "VPN keeps disconnecting again", "caller": "Jordan Lee"}],
     "sim": {"services": {"vpn": {"status": "outage", "detail": "vpn-gw-2 not responding"}}},
     "plan": [("find_similar_incidents", {}), ("get_service_health", {"service": "vpn"}),
              ("link_to_outage", {"parent_number": "INC0010001"}),
              ("escalate_incident", {"team": "Network", "priority": "2",
                                     "reason": "VPN gateway vpn-gw-2 is down; 3 related incidents in the last hour."})]},
    {"id": "phishing", "title": "Typed my password on a fake site", "category": "Security", "expect": "Escalated to Security",
     "caller": ("jordan.lee", "Jordan Lee"),
     "text": "I clicked a link in a weird email and typed my password on a strange login page",
     "kb": _kb("phishing-reporting.md", "Report Phishing and Suspicious Activity",
               "Disconnect from VPN and Wi-Fi and call the Security hotline at extension 7700 immediately [1].",
               "If you clicked a link or entered your password: 1. Disconnect from VPN and Wi-Fi. 2. Call the Security "
               "hotline at extension 7700 immediately, 24/7. 3. Change your password from a different device."),
     "plan": [("get_mailbox_diagnostics", {}),
              ("escalate_incident", {"team": "Security", "priority": "2",
                                     "reason": "Caller entered their password on a suspicious page and the mailbox "
                                               "forwards to an external address. Possible account compromise."})]},
    {"id": "denied", "title": "Reset a coworker's password", "category": "Policy", "expect": "Refused, escalated",
     "caller": ("alex.rivera", "Alex Rivera"), "text": "Please reset Priya Shah's password, she's locked out",
     "kb": NO_KB,
     "plan": [("verify_caller_identity", {}), ("reset_password", {"user": "priya.shah"}),
              ("escalate_incident", {"team": "Service Desk",
                                     "reason": "Request to reset another user's password. The agent only acts on the "
                                               "caller's own account; Priya should contact the Service Desk herself."})]},
    {"id": "timeout", "title": "Password reset times out", "category": "Failure", "expect": "Unknown outcome, escalated",
     "caller": ("alex.rivera", "Alex Rivera"), "text": "I forgot my password and can't sign in",
     "kb": _kb("password-reset.md", "Reset or Change Your Password",
               "Use self-service reset at passwordreset.example.org if you registered MFA [1].",
               "Go to https://passwordreset.example.org from any browser. Enter your work email and complete MFA. "
               "Choose a new password that meets the rules above."),
     "faults": {"reset_password": ["timeout"]},
     "plan": [("verify_caller_identity", {}), ("reset_password", {}), ("reset_password", {}),
              ("escalate_incident", {"team": "Identity & Access",
                                     "reason": "Password reset timed out; it may or may not have been applied, so it "
                                               "was not retried. Check the account before acting."})]},
]
SAMPLE_IDS = [s["id"] for s in SAMPLES]


def sample(sample_id: str) -> dict:
    for item in SAMPLES:
        if item["id"] == sample_id:
            return item
    raise KeyError(sample_id)


def clean_custom(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text[:MAX_CUSTOM]


def live_calls(plan: dict | None) -> list[dict] | None:
    """The live model's tool calls from a /demo/ask response, validated (None = the model didn't finish)."""
    if not isinstance(plan, dict) or not isinstance(plan.get("calls"), list):
        return None
    known = set(REGISTRY.tools) | DESK_TOOLS | {"search_knowledge_base"}
    calls = []
    for call in plan["calls"][:12]:
        if not isinstance(call, dict) or call.get("name") not in known:
            continue
        args = call.get("input") if isinstance(call.get("input"), dict) else {}
        turn = call.get("turn") if isinstance(call.get("turn"), int) else len(calls)
        calls.append({"name": call["name"], "input": {str(k)[:40]: str(v)[:400] for k, v in args.items()}, "turn": turn})
    return calls


# ---------------------------------------------------------------- the ticket
def _target(tool: str) -> tuple[str, bool]:
    """Where a tool call lands, and whether it is simulated."""
    if tool == "search_knowledge_base":
        return "Knowledge base (RAG)", False
    if tool in DESK_TOOLS:
        return "Demo ticket (in memory, not ServiceNow)", False
    t = REGISTRY.get(tool)
    if t is None:
        return "Unknown tool", False
    if t.workflow == 0:
        return "Simulated Entra ID (MFA)", True
    return CAPABILITIES[t.workflow][1], t.simulated


def _verdict(tool: str, out: dict) -> tuple[str, str]:
    """(status, plain words) for one tool result."""
    if not isinstance(out, dict):
        return "ok", ""
    if out.get("denied"):
        return "refused", f"Refused by policy: {out.get('reason', '')}"
    if out.get("needs_identity"):
        return "blocked", "Needs identity verification first"
    if out.get("approval_requested"):
        return "approval", f"Needs approval from {out.get('approver')} ({out.get('approver_role', '').replace('_', ' ')})"
    if out.get("invalid"):
        return "refused", f"Rejected: {out.get('reason', 'invalid arguments')}"
    if out.get("status") == "unknown_outcome" and "earlier attempt" in str(out.get("error")):
        return "refused", "Not run: an earlier attempt has an unknown outcome, and a write is never repeated blindly"
    if out.get("status") == "unknown_outcome":
        return "failed", "Timed out: the outcome is unknown, so it is never retried automatically"
    if out.get("status") in ("failed", "unavailable"):
        return "failed", f"Failed: {out.get('error', '')}"
    if out.get("error"):
        return "refused", str(out["error"])
    if out.get("duplicate"):
        return "ok", "Already done for this ticket: skipped, not repeated"
    if REGISTRY.get(tool) and REGISTRY.get(tool).risk != "read":
        return "ok", "Authorized and executed"
    return "ok", ""


class Ticket:
    """One demo incident and everything that happens to it."""

    def __init__(self, number: str, caller: tuple, text: str, kb: dict, calls, source: str, recent=None, sim=None,
                 faults=None, start=None):
        caller_user, caller_name = caller
        self.source = source                        # "sample" | "live"
        self.kb, self.calls = kb, calls
        self.store = MemoryStore()
        self.backends = SimBackends(faults=faults, scenario=sim)
        record = {"sys_id": "d" * 32, "number": number, "short_description": text, "description": "", "state": "1",
                  "hold_reason": "", "impact": "3", "urgency": "3", "priority": 4, "assignment_group": "",
                  "assigned_to": "", "caller_user": caller_user, "caller_name": caller_name, "approval": "not requested",
                  "close_code": "", "close_notes": "", "parent_incident": ""}
        self.sn = SandboxServiceNow(record, recent, start)
        self.desk = ServiceNowDesk(self.sn)
        self.timeline = []      # what the visitor sees, in order
        self.claims = set()     # webhook idempotency keys (the Lambda claims these in DynamoDB)
        self.last_event = None
        self.outcome = None
        self.executed = []
        self.sn.changes.append({"at": self.sn.now(), "author": caller_name,
                                "fields": {"state": ("", "1"), "priority": ("", 4)}})
        self._log("event", "Incident created", f"{caller_name}: “{text}”", actor=caller_name)

    # ----- timeline
    def _log(self, kind, title, detail="", **extra):
        self.timeline.append({"at": self.sn.now(), "kind": kind, "title": title, "detail": detail, **extra})

    @property
    def record(self) -> dict:
        return self.sn.display()

    # ----- the webhook + queue + worker path
    def _deliver(self, kind: str, payload: dict | None = None) -> str:
        """Business Rule -> signed webhook -> idempotency claim -> queue -> worker. Returns the outcome."""
        version = self.sn.display()["sys_mod_count"]
        key = self.sn.rec["sys_id"] if kind == "new" else f"{self.sn.rec['sys_id']}#{kind}#{version}"
        self.last_event = (kind, payload or {}, key)
        if key in self.claims:
            self._log("guard", "Duplicate webhook event ignored", "Same event key already claimed (the Lambda's "
                      "DynamoDB conditional write), so it never reaches the queue twice.", key=key)
            return "duplicate"
        self.claims.add(key)
        self._log("event", {"new": "New-incident event received", "caller_reply": "Caller-reply event received",
                            "approval": "Approval event received"}[kind],
                  "Signed by the Business Rule, verified, claimed once, queued, picked up by the worker.")
        return self._work(kind, payload or {})

    def redeliver(self, layer: str = "webhook") -> str:
        """Simulate a duplicate: the same webhook call again, or the queue delivering the message twice."""
        if not self.last_event:
            return "nothing"
        kind, payload, key = self.last_event
        if layer == "webhook":
            if key in self.claims:
                self._log("guard", "Duplicate webhook event ignored", "Same event key already claimed (the Lambda's "
                          "DynamoDB conditional write), so the agent doesn't run again.", key=key)
                return "duplicate"
            self.claims.add(key)
            return self._work(kind, payload)
        self._log("event", "Queue redelivered the same message", "SQS delivers at least once; the worker checks the "
                  "ticket before acting.")
        return self._work(kind, payload, redelivery=True)

    def _work(self, kind: str, payload: dict, redelivery: bool = False) -> str:
        incident = self.sn.display()
        if kind == "new":
            skip = new_incident_check(incident)
            if skip:
                self._log("guard", "Worker skipped the event", {"already_handled": "The ticket already has the AI "
                          "agent's work note, so it is never worked twice.", "technician_assigned": "A technician owns "
                          "it.", "inactive": "The incident is closed."}[skip], outcome=skip)
                return skip
            return self._run_agent(incident)
        if kind == "caller_reply":
            result = handle_caller_reply(FakeLLM(), LocalGuardrail(), self.desk, incident)
            words = {"resolved": "The caller confirmed the fix: resolved.",
                     "back_to_technician": "Not a clear yes: back to the Service Desk with the reply quoted.",
                     "reopened": "Reply on an AI-resolved ticket that doesn't say it's fixed: reopened for a technician.",
                     "still_resolved": "The caller says it works: stays resolved, nothing written.",
                     "already_handled": "A reply was already handled: people own the conversation now.",
                     "technician_assigned": "A technician owns it: the AI stays out.",
                     "not_caller": "The reply isn't from the caller: ignored.", "no_reply": "No caller reply found."}
            self._log("policy", "Caller reply handled", words.get(result["outcome"], result["outcome"])
                      + (f" (classified as “{result['verdict']}” by the offline classifier)" if result.get("verdict") else ""),
                      outcome=result["outcome"])
            return result["outcome"]
        if kind == "approval":
            approver, decision = payload["approver"], payload["decision"]
            if str(incident.get("approval") or "").lower() != decision:
                self._log("guard", "Approval event ignored", "The ticket no longer shows that decision.")
                return "ignored"
            result = handle_approval_decision(self.desk, incident, approver, decision == "approved", self.store,
                                              backends=self.backends, resolve_simulated=True, simulated_only=True)
            for r in result.get("results", []):
                if r.get("error") and not r.get("tool"):
                    self._log("policy", "Approval refused", f"{approver}: {r['error']}", status="refused")
            for e in result.get("executed", []):
                self.executed.append(e)
                self._log("tool", e["tool"], "", target=_target(e["tool"])[0], simulated=e["simulated"], status="ok",
                          verdict="Approved action executed in code (the model can't run it)", output={"summary": e["summary"]},
                          input={})
            words = {"executed_resolved": "Approved, executed, verified by reading the system back: resolved.",
                     "executed_awaiting_caller": "Approved and executed: waiting for the caller to confirm.",
                     "rejected": "Rejected by the approver: back to the Service Desk, caller told.",
                     "not_authorized": "Refused: this person can't approve this request. Logged; nothing ran.",
                     "failed_after_approval": "Approved, but the action failed: back to a technician.",
                     "no_pending_approval": "No approval is pending on this ticket."}
            self._log("policy", "Approval decision handled", words.get(result["outcome"], result["outcome"]),
                      outcome=result["outcome"])
            return result["outcome"]
        raise ValueError(kind)

    def _run_agent(self, incident: dict) -> str:
        planner = UnavailablePlanner() if self.calls is None else ScriptedPlanner(self.calls)
        kit = build_toolkit(incident["number"], incident["caller_id.user_name"], store=self.store,
                            backends=self.backends, simulated_only=True, sleep=lambda s: None)
        agent = RecordingAgent(planner, RecordedSearch(self.kb), self.desk, toolkit=kit, max_steps=8,
                               resolve_simulated=True)
        self.sn.tick(seconds=4)
        outcome = agent.run(incident, incident_question(incident) or incident["short_description"])
        kb = outcome.get("kb") or self.kb
        self._log("tool", "search_knowledge_base", "", target=_target("search_knowledge_base")[0], simulated=False,
                  status="ok" if kb.get("status") == "answered" else "info",
                  verdict={"answered": "Answered from " + ", ".join(sorted({s["source"] for s in kb.get("sources", [])})),
                           "no_answer": "No article covers it", "blocked": "Blocked by the input guardrail: the agent "
                           "never sees the text"}.get(kb.get("status"), "Search unavailable"),
                  input={"query": incident["short_description"]},
                  output={"status": kb.get("status"), "sources": [s.get("title") for s in kb.get("sources", [])]})
        for rec in agent.records:
            target, simulated = _target(rec["tool"])
            status, verdict = _verdict(rec["tool"], rec["output"])
            if rec["tool"] == "search_knowledge_base":
                verdict = ("Follow-up search by the model. The replay doesn't rerun retrieval, so this shows the first "
                           "search's articles; only the first search can be sent to the caller.")
            self._log("tool", rec["tool"], "", target=target, simulated=simulated, status=status, verdict=verdict,
                      input=rec["input"], output=rec["output"])
        if self.calls is None:
            self._log("guard", "Model unavailable", "The live model didn't return a plan in time. In production SQS "
                      "retries the message, and the final attempt escalates; that escalation is what you see here.")
        self.executed += kit.ctx.executed
        self.outcome = outcome["action"]
        self.sn.tick(seconds=18)
        words = {"resolved": "Every action read its end state back from the system: resolved without waiting.",
                 "awaiting_caller": "Waiting for the caller: resolved if they confirm, reminders if they don't.",
                 "awaiting_approval": "Waiting for an approver: the approved action will run in code.",
                 "escalated": f"Escalated to {outcome.get('team')}" + (f" at priority {outcome.get('priority')}"
                                                                       if (outcome.get("priority") or 5) <= 2 else "")
                 + ", with a handoff note."}
        self._log("policy", "Agent decision", words.get(outcome["action"], outcome["action"]), outcome=outcome["action"])
        return outcome["action"]

    # ----- what a person does in the ServiceNow UI
    def caller_reply(self, text: str, author: str | None = None) -> str:
        """A comment from the caller (or someone else), then the reply Business Rule's conditions."""
        text = clean_custom(text) or "(empty)"
        author = author or self.sn.rec["caller_name"]
        prev = dict(self.sn.rec)
        self.sn.tick(hours=1)
        self.sn.human(author, comments=text)
        self._log("person", f"{author} commented", f"“{text}”", actor=author)
        awaiting = prev["state"] == "3" and prev.get("hold_reason") == "1"
        ai_resolved = prev["state"] == "6" and str(prev.get("close_notes", "")).startswith("Resolved by the AI agent")
        if not (awaiting or ai_resolved):
            self._log("guard", "Business Rule didn't fire", "The ticket isn't waiting on the caller or resolved by the AI, "
                      "so people own the conversation.")
            return "rule_not_fired"
        if author != self.sn.rec["caller_name"]:
            self._log("guard", "Business Rule didn't fire", "Only the caller's own comments are sent to the agent.")
            return "rule_not_fired"
        return self._deliver("caller_reply")

    def approve(self, approver_user: str, approved: bool = True) -> str:
        """An approver sets the Approval field; the approval Business Rule sends who decided and what."""
        if self.sn.rec.get("approval") != "requested":
            return "not_requested"
        person = self.backends.data["users"].get(approver_user, {}).get("name", approver_user)
        self.sn.tick(minutes=30)
        decision = "approved" if approved else "rejected"
        self.sn.human(person, approval=decision)
        self._log("person", f"{person} set Approval to {decision.capitalize()}", "", actor=person)
        result = self._deliver("approval", {"approver": approver_user, "decision": decision})
        if result == "not_authorized":
            self.sn.human("Demo reset", approval="requested")   # so the real approver can still decide
            self._log("guard", "Approval field set back to Requested", "Sandbox step so the owner can still decide. In "
                      "production an ACL should stop non-approvers editing the field; the agent refuses them either way.")
        return result

    def fast_forward(self) -> str:
        """One day passes; the hourly follow-up job runs (3-strike rule)."""
        self.sn.tick(days=1)
        result = follow_up(self.desk, self.sn.display(), self.store, wait_hours=24)
        words = {"reminder_1": "No reply after a day: reminder 1 of 2 sent.",
                 "reminder_2": "Still no reply: last reminder sent.",
                 "resolved_no_response": "No reply after two reminders: resolved as No resolution provided (never "
                                         "recorded as fixed). The caller can reply to reopen.",
                 "urgent_to_technician": "Urgent ticket: handed to a technician instead of reminders.",
                 "security_to_technician": "Possible security issue: handed to Security instead of reminders.",
                 "not_waiting": "Not waiting on the caller: the follow-up job leaves it alone.",
                 "not_agent_fix": "The AI didn't send a fix on this ticket: nothing to follow up.",
                 "caller_replied": "The caller already replied: the reply path handles it.",
                 "not_service_desk": "Another team owns it: no reminders.", "technician_assigned": "A technician owns it."}
        self._log("policy", "Follow-up job (fast-forward 1 day)", words.get(result["outcome"], result["outcome"]),
                  outcome=result["outcome"])
        return result["outcome"]

    # ----- for the page
    def pending_approvals(self) -> list[dict]:
        ids = (self.store.get(f"pending#{self.sn.rec['number']}") or {"ids": []})["ids"]
        found = []
        for approval_id in ids:
            rec = self.store.get(f"approval#{approval_id}")
            if rec and rec.get("status") == "pending":
                users = self.backends.data["users"]
                found.append({"tool": rec["tool"], "approver": rec.get("approver"), "requested_by": rec["requested_by"],
                              "approver_name": users.get(rec.get("approver"), {}).get("name", rec.get("approver")),
                              "requester_name": users.get(rec["requested_by"], {}).get("name", rec["requested_by"])})
        return found

    def actions(self) -> list[str]:
        """What a visitor can do next, given the ticket's state."""
        r = self.sn.rec
        found = []
        if r["state"] == "3" and r.get("hold_reason") == "1":
            found += ["reply", "fast_forward"]
        if r["state"] == "6" and str(r.get("close_notes", "")).startswith("Resolved by the AI agent"):
            found.append("reply")
        if r.get("approval") == "requested" and self.pending_approvals():
            found.append("approve")
        if self.last_event:
            found.append("redeliver")
        return found


def new_sample(sample_id: str, number: str = "DEMO0001001") -> Ticket:
    sc = sample(sample_id)
    ticket = Ticket(number, sc["caller"], sc["text"], sc["kb"], [{"name": n, "input": a} for n, a in sc["plan"]],
                    "sample", sc.get("recent"), sc.get("sim"), sc.get("faults"))
    ticket._deliver("new")
    return ticket


def new_live(text: str, response: dict, number: str = "DEMO0001001") -> Ticket:
    """A visitor's own incident, after the public /demo/ask route ran the real RAG pipeline and the real model
    (dry run, rate-limited, guardrailed). The model's tool calls are replayed here through the same code."""
    kb = {k: response.get(k) for k in ("status", "answer", "sources", "guardrail")}
    kb.update(provider="live", usage={})
    kb["sources"] = kb.get("sources") or []
    calls = live_calls(response.get("agent"))
    if kb.get("status") == "blocked":
        calls = []
    ticket = Ticket(number, ("alex.rivera", "Alex Rivera"), clean_custom(text), kb, calls, "live")
    ticket._deliver("new")
    return ticket


def as_json(value) -> str:
    return json.dumps(value, indent=1, default=str)[:1500]


def stages(ticket: Ticket) -> list[dict]:
    """The seven-step summary of how the agent worked the ticket."""
    tools = [t for t in ticket.timeline if t["kind"] == "tool"]
    kb = next((t for t in tools if t["title"] == "search_knowledge_base"), None)
    chosen = [t["title"] for t in tools if t["title"] != "search_knowledge_base"]
    writes = [t for t in tools if REGISTRY.get(t["title"]) and REGISTRY.get(t["title"]).risk != "read"]
    refused = [t for t in tools if t.get("status") in ("refused", "blocked")]
    approval = [t for t in tools if t.get("status") == "approval"]
    failed = [t for t in tools if t.get("status") == "failed"]
    done = [t for t in writes if t.get("status") == "ok"]
    rec = ticket.sn.rec

    def st(name, status, detail):
        return {"name": name, "status": status, "detail": detail}
    verified = done and all(e.get("verifies_fix") for e in ticket.executed)
    final = {"6": "Resolved", "3": "On Hold", "2": "In Progress", "1": "New"}.get(rec["state"], STATE[rec["state"]])
    group = rec.get("assignment_group") or ""
    if rec["state"] in ("1", "2") and group and group != "Service Desk":
        outcome = f"Escalated to {group}"
    else:
        outcome = final + (f" · {group}" if group else "")
    return [st("Incident received", "done", rec["caller_name"]),
            st("Knowledge retrieval", "done", (kb or {}).get("verdict", "")),
            st("Tool selection", "done" if chosen else "skipped", f"{len(chosen)} tool call(s)"),
            st("Authorization", "blocked" if refused else ("waiting" if approval else ("done" if writes else "skipped")),
               "refused" if refused else ("approval needed" if approval else ("passed" if writes else "nothing to authorize"))),
            st("Action", "failed" if failed else ("done" if done else "skipped"),
               "outcome unknown" if failed else (f"{len(done)} simulated" if done else "none")),
            st("Verification", "done" if verified else ("waiting" if rec["state"] == "3" else "skipped"),
               "read back" if verified else ("caller confirms" if rec["state"] == "3" else "n/a")),
            st("Resolution or escalation", "done", outcome)]


def isolation_report() -> dict:
    """Facts the page shows about what the demo can and cannot reach (checked by tests)."""
    simulated = sorted(n for n, t in REGISTRY.tools.items() if t.simulated)
    real = sorted(n for n, t in REGISTRY.tools.items() if not t.simulated)
    return {"simulated_tools": simulated, "non_simulated_tools": real,
            "servicenow_client": "SandboxServiceNow (in memory, no URL or credentials)"}

