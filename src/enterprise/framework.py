"""Common framework for the enterprise IT agent: tool registry, authorization, approvals, audit
trail, idempotency, and failure recovery.

Every action the model asks for goes through the same path, in code:

    model asks for a tool
      -> Registry      is it a known tool? (the model can only call what is registered)
      -> Policy        identity verified? allowed for this caller/target? approval needed?
      -> Approvals     sensitive actions wait for an authorized approver (not the requester)
      -> Executor      idempotency key (no double execution), retries for transient errors,
                       timeouts: retry only tools marked idempotent, otherwise stop and escalate
      -> Audit         every decision and result, with secrets redacted
      -> model sees    a redacted result (credentials never reach the model, tickets, or logs)

The model never holds credentials. Tool handlers use backend clients configured on the server
(simulated backends here unless a real integration is configured and authorized).
"""
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

READ, LOW, SENSITIVE = "read", "low", "sensitive"          # risk levels
SECRET_KEYS = re.compile(r"password|passwd|secret|token$|^otp|passcode|credential|api_?key", re.I)


# ---------------------------------------------------------------- errors
class ToolError(Exception):
    """A tool failed in a way that retrying won't fix (bad input, not found, policy in the backend)."""


class TransientError(ToolError):
    """Throttled or temporarily unavailable: safe to retry."""


class ToolTimeout(ToolError):
    """The backend didn't answer in time. The action may or may not have happened."""


# ---------------------------------------------------------------- registry
@dataclass
class Tool:
    name: str
    workflow: int
    description: str
    handler: Callable                    # handler(ctx, args) -> dict (public result, no secrets)
    properties: dict = field(default_factory=dict)
    required: tuple = ()
    risk: str = READ
    simulated: bool = True
    requires_identity: bool = False      # caller identity verified (MFA) before running
    approval: object = None              # approver role (manager, group_owner, license_admin, hr), or
                                         # a function (ctx, args) -> role or None when it depends on the request
    idempotent: bool = True              # safe to repeat if a call timed out
    precheck: Callable | None = None     # precheck(ctx, args) -> reason to deny, or ""

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": {"type": "object", "properties": self.properties, "required": list(self.required)}}


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def by_workflow(self) -> dict[int, list[str]]:
        out = {}
        for t in self.tools.values():
            out.setdefault(t.workflow, []).append(t.name)
        return out


# ---------------------------------------------------------------- state stores
class MemoryStore:
    """Key-value store with conditional put. DynamoStore below has the same interface for Lambda."""

    def __init__(self):
        self.items = {}

    def get(self, key: str) -> dict | None:
        return self.items.get(key)

    def put(self, key: str, value: dict, if_absent: bool = False) -> bool:
        if if_absent and key in self.items:
            return False
        self.items[key] = value
        return True


class DynamoStore:
    """DynamoDB-backed store: pk (string) -> JSON document, conditional writes for idempotency."""

    def __init__(self, client, table: str, ttl_days: int = 30):
        self.client, self.table, self.ttl = client, table, ttl_days

    def get(self, key):
        item = self.client.get_item(TableName=self.table, Key={"pk": {"S": key}}).get("Item")
        return json.loads(item["doc"]["S"]) if item else None

    def put(self, key, value, if_absent=False):
        args = {"TableName": self.table,
                "Item": {"pk": {"S": key}, "doc": {"S": json.dumps(value)},
                         "expires_at": {"N": str(int(time.time()) + self.ttl * 86400)}}}
        if if_absent:
            args["ConditionExpression"] = "attribute_not_exists(pk)"
        try:
            self.client.put_item(**args)
            return True
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise


def args_key(tool: str, args: dict, ticket: str) -> str:
    """Stable hash of exactly what is being done, for idempotency and approval binding."""
    canonical = json.dumps({"tool": tool, "args": args, "ticket": ticket}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


# ---------------------------------------------------------------- audit
def redact(value):
    """Remove anything that looks like a credential, recursively."""
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if SECRET_KEYS.search(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class AuditLog:
    """Append-only trail of every authorization decision and execution, secrets redacted.
    Each entry also goes to the logs as one JSON line (CloudWatch Logs Insights-friendly)."""

    def __init__(self, sink: list | None = None):
        self.entries = sink if sink is not None else []

    def record(self, **entry):
        entry = redact({"ts": round(time.time(), 3), **entry})
        self.entries.append(entry)
        logger.info(json.dumps({"event": "agent_audit", **entry}, default=str))
        return entry


# ---------------------------------------------------------------- context and policy
@dataclass
class Context:
    ticket: str                      # ServiceNow incident number
    caller: str                      # caller's user id in the directory (from ServiceNow, not the model)
    store: object                    # MemoryStore / DynamoStore: approvals + idempotency
    audit: AuditLog
    backends: object                 # simulated or real backend clients
    simulated_only: bool = False     # public demo: refuse anything not simulated
    identity_verified: bool = False  # set only by the identity verification tool's server-side result
    sleep: Callable = time.sleep
    caller_messages: list = field(default_factory=list)   # templated, code-written messages for the caller
    executed: list = field(default_factory=list)          # what actually ran (for the ticket and the trace)


@dataclass
class Decision:
    outcome: str                     # allow | needs_identity | needs_approval | deny
    reason: str = ""
    approval_id: str = ""


class Policy:
    """Authorization layer. The model's arguments are data; identity and approvals come only from
    server-side state."""

    def authorize(self, tool: Tool, args: dict, ctx: Context) -> Decision:
        if ctx.simulated_only and not tool.simulated:
            return Decision("deny", "the public demo only runs simulated tools")
        if tool.precheck:
            reason = tool.precheck(ctx, args)
            if reason:
                return Decision("deny", reason)
        if tool.requires_identity and not ctx.identity_verified:
            return Decision("needs_identity", "verify the caller's identity first (verify_caller_identity)")
        role = tool.approval(ctx, args) if callable(tool.approval) else tool.approval
        if role:
            approval_id = args_key(tool.name, args, ctx.ticket)
            record = ctx.store.get(f"approval#{approval_id}")
            if record and record["status"] == "approved" and not record.get("consumed"):
                return Decision("allow", f"approved by {record['decided_by']}", approval_id)
            if record and record["status"] == "rejected":
                return Decision("deny", f"rejected by {record['decided_by']}")
            if record and record.get("consumed"):
                return Decision("deny", "this approval was already used")
            return Decision("needs_approval", f"needs approval from the {role}", approval_id)
        return Decision("allow")


class Approvals:
    """Approval requests bound to the exact tool + arguments + ticket. An approval can be used once,
    only by that action, and only an approver with the right role who isn't the requester can decide."""

    def __init__(self, store, directory=None):
        self.store = store
        self.directory = directory   # resolves approver roles (simulated directory or a real one)

    def request(self, approval_id: str, tool: Tool, args: dict, ctx: Context) -> dict:
        role = tool.approval(ctx, args) if callable(tool.approval) else tool.approval
        record = {"id": approval_id, "tool": tool.name, "args": args, "ticket": ctx.ticket,
                  "requested_by": ctx.caller, "role": role, "status": "pending",
                  "identity_verified": ctx.identity_verified,  # checked before any approval is requested
                  "approver": self.directory.approver_for(role, ctx.caller, args) if self.directory else None}
        created = self.store.put(f"approval#{approval_id}", record, if_absent=True)
        pending = (self.store.get(f"pending#{ctx.ticket}") or {"ids": []})
        if approval_id not in pending["ids"]:
            pending["ids"].append(approval_id)
            self.store.put(f"pending#{ctx.ticket}", pending)
        ctx.audit.record(ticket=ctx.ticket, action="approval_requested", tool=tool.name, args=args,
                         approver=record["approver"], duplicate=not created)
        return self.store.get(f"approval#{approval_id}")

    def decide(self, approval_id: str, approver: str, approve: bool, audit: AuditLog) -> dict:
        record = self.store.get(f"approval#{approval_id}")
        if not record:
            raise ToolError("unknown approval")
        if record["status"] != "pending":
            raise ToolError(f"approval already {record['status']}")
        if approver == record["requested_by"]:
            raise ToolError("requesters can't approve their own requests")
        if self.directory and not self.directory.has_role(approver, record["role"], record):
            raise ToolError(f"{approver} is not an authorized {record['role']} for this request")
        record.update(status="approved" if approve else "rejected", decided_by=approver)
        self.store.put(f"approval#{approval_id}", record)
        audit.record(ticket=record["ticket"], action="approval_" + record["status"], tool=record["tool"],
                     approver=approver)
        return record

    def consume(self, approval_id: str):
        record = self.store.get(f"approval#{approval_id}")
        record["consumed"] = True
        self.store.put(f"approval#{approval_id}", record)


# ---------------------------------------------------------------- execution
class Executor:
    """Runs one authorized tool call: idempotency, bounded retries, timeout handling."""

    def __init__(self, max_attempts: int = 3, waits=(1, 3)):
        self.max_attempts = max_attempts
        self.waits = waits

    def run(self, tool: Tool, args: dict, ctx: Context) -> dict:
        key = f"exec#{args_key(tool.name, args, ctx.ticket)}"
        if tool.risk != READ:
            done = ctx.store.get(key)
            if done:
                ctx.audit.record(ticket=ctx.ticket, action="duplicate_skipped", tool=tool.name, args=args)
                return {**done, "duplicate": True}
        attempt = 0
        while True:
            attempt += 1
            try:
                result = tool.handler(ctx, args)
                break
            except TransientError as error:
                if attempt >= self.max_attempts:
                    return self._failed(tool, args, ctx, "unavailable", str(error), attempt)
                ctx.audit.record(ticket=ctx.ticket, action="retry", tool=tool.name, attempt=attempt, error=str(error))
                ctx.sleep(self.waits[min(attempt - 1, len(self.waits) - 1)])
            except ToolTimeout as error:
                if tool.idempotent and attempt < self.max_attempts:
                    ctx.audit.record(ticket=ctx.ticket, action="retry_after_timeout", tool=tool.name, attempt=attempt)
                    ctx.sleep(self.waits[min(attempt - 1, len(self.waits) - 1)])
                    continue
                # A write that timed out may have happened: never blindly repeat it.
                return self._failed(tool, args, ctx, "unknown_outcome", str(error), attempt)
            except ToolError as error:
                return self._failed(tool, args, ctx, "failed", str(error), attempt)
        result = {"ok": True, **result}
        if tool.risk != READ:
            ctx.store.put(key, result)
            ctx.executed.append({"tool": tool.name, "workflow": tool.workflow, "simulated": tool.simulated,
                                 "args": redact(args)})
        ctx.audit.record(ticket=ctx.ticket, action="executed", tool=tool.name, args=args, attempts=attempt,
                         simulated=tool.simulated, result=result)
        return result

    @staticmethod
    def _failed(tool, args, ctx, status, error, attempts) -> dict:
        ctx.audit.record(ticket=ctx.ticket, action=status, tool=tool.name, args=args, attempts=attempts, error=error)
        return {"ok": False, "status": status, "error": error, "escalate": True,
                "note": "Do not retry this action. Escalate to a technician." if status == "unknown_outcome" else ""}


class Toolkit:
    """What the agent calls: the full authorize -> approve -> execute -> audit path for one tool."""

    def __init__(self, registry: Registry, ctx: Context, policy: Policy | None = None,
                 executor: Executor | None = None, approvals: Approvals | None = None):
        self.registry, self.ctx = registry, ctx
        self.policy = policy or Policy()
        self.executor = executor or Executor()
        self.approvals = approvals or Approvals(ctx.store, getattr(ctx.backends, "directory", None))
        self.pending_approvals: list[dict] = []
        self.trace: list[dict] = []

    def schemas(self) -> list[dict]:
        return self.registry.schemas()

    def call(self, name: str, args: dict) -> dict:
        tool = self.registry.get(name)
        if tool is None:
            return {"error": f"unknown tool {name}"}
        decision = self.policy.authorize(tool, args, self.ctx)
        self.ctx.audit.record(ticket=self.ctx.ticket, action="authorize", tool=name, args=args,
                              decision=decision.outcome, reason=decision.reason)
        step = {"tool": name, "workflow": tool.workflow, "simulated": tool.simulated, "risk": tool.risk}
        if decision.outcome == "deny":
            self.trace.append({**step, "result": "denied", "detail": decision.reason})
            return {"ok": False, "denied": True, "reason": decision.reason}
        if decision.outcome == "needs_identity":
            self.trace.append({**step, "result": "needs identity", "detail": decision.reason})
            return {"ok": False, "needs_identity": True, "reason": decision.reason}
        if decision.outcome == "needs_approval":
            record = self.approvals.request(decision.approval_id, tool, args, self.ctx)
            self.pending_approvals.append(record)
            self.trace.append({**step, "result": "approval requested",
                               "detail": f"{record['role']}: {record.get('approver') or 'approver'}"})
            return {"ok": False, "approval_requested": True, "approver_role": record["role"],
                    "approver": record.get("approver"),
                    "note": "The action will run automatically once approved. Finish with request_approval."}
        if decision.approval_id:
            self.approvals.consume(decision.approval_id)
        result = self.executor.run(tool, args, self.ctx)
        self.trace.append({**step, "result": "ok" if result.get("ok") else result.get("status", "failed"),
                           "detail": result.get("summary") or result.get("error") or ""})
        return redact(result)

    def run_approved(self, approval_id: str) -> dict:
        """Execute exactly the approved action (no model involved), once."""
        record = self.ctx.store.get(f"approval#{approval_id}")
        if not record or record["status"] != "approved":
            raise ToolError("not approved")
        if record["ticket"] != self.ctx.ticket or record["requested_by"] != self.ctx.caller:
            raise ToolError("approval belongs to a different ticket or caller")
        self.ctx.identity_verified = bool(record.get("identity_verified"))
        return self.call(record["tool"], record["args"])
