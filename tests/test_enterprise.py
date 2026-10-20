"""Enterprise IT agent: the common framework and all 12 workflows.

Each workflow is tested for the cases that apply to it: success, permission denial, identity and
approval gates, retries, duplicate requests, timeouts, and escalation. Workflows marked simulated
run against enterprise/sim.py; 5 (KB part), 7, 8 and 12 use the real ServiceNow/KB code paths with
test doubles for the HTTP layer.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from agent import (CONFIRM_MARKER, REOPEN_INVITE, RESOLVED_MARKER, SIMULATED_NOTICE, DryRunDesk,  # noqa: E402
                   TicketAgent, handle_caller_reply)
from enterprise.framework import (AuditLog, DynamoStore, MemoryStore,  # noqa: E402
                                  args_key, redact)
from enterprise.loganalysis import analyze  # noqa: E402
from enterprise.runtime import (build_toolkit, caller_id, demo_llm, demo_toolkit,  # noqa: E402
                                handle_approval_decision)
from enterprise.sim import LOGS, SimBackends  # noqa: E402
from enterprise.workflows import WORKFLOWS, build_registry  # noqa: E402
from factory import build_guardrail  # noqa: E402
from llm import AnthropicLLM, FakeLLM  # noqa: E402
from test_agent import Assistant, Scripted, answered  # noqa: E402

NO_ANSWER = {"status": "no_answer", "answer": "", "sources": [], "provider": "fake", "guardrail": {},
             "usage": {"input_tokens": 50, "output_tokens": 10}, "latency_ms": 5}


def kit(caller="alex.rivera", faults=None, store=None, ticket="INC0010100", scenario=None):
    return build_toolkit(ticket, caller, store=store or MemoryStore(), backends=SimBackends(faults, scenario),
                         sleep=lambda s: None)


def verified(k):
    assert k.call("verify_caller_identity", {})["ok"]
    return k


# ---------------------------------------------------------------- framework
def test_registry_covers_all_12_workflows():
    by = build_registry().by_workflow()
    # 7, 8 and 12 are ServiceNow-side actions in agent.py (escalate, similar/link, caller confirmation)
    assert {1, 2, 3, 4, 5, 6, 9, 10, 11} <= set(by) and len(WORKFLOWS) == 12


def test_unknown_tool_and_secret_redaction():
    k = kit()
    assert "unknown tool" in k.call("format_disk", {})["error"]
    assert redact({"password": "x", "nested": {"api_key": "y", "ok": 1}, "list": [{"secret": "z"}]}) == \
        {"password": "[REDACTED]", "nested": {"api_key": "[REDACTED]", "ok": 1}, "list": [{"secret": "[REDACTED]"}]}


def test_identity_cant_be_claimed_by_the_model():
    k = kit()
    out = k.call("unlock_account", {"user": "alex.rivera", "identity_verified": True})
    assert out["needs_identity"] and k.ctx.backends.data["users"]["alex.rivera"]["locked"]


def test_dynamo_store_conditional_put():
    class Ddb:
        def __init__(self):
            self.items = {}

        def get_item(self, TableName, Key):
            item = self.items.get(Key["pk"]["S"])
            return {"Item": item} if item else {}

        def put_item(self, TableName, Item, ConditionExpression=None):
            if ConditionExpression and Item["pk"]["S"] in self.items:
                class Failed(Exception):
                    response = {"Error": {"Code": "ConditionalCheckFailedException"}}
                raise Failed()
            self.items[Item["pk"]["S"]] = Item
    store = DynamoStore(Ddb(), "t")
    assert store.put("a", {"x": 1}, if_absent=True) and not store.put("a", {"x": 2}, if_absent=True)
    assert store.get("a") == {"x": 1}


# ---------------------------------------------------------------- 1. password reset and unlock
def test_wf1_unlock_after_identity_verification():
    k = verified(kit())
    assert k.call("get_account_status", {"user": "alex.rivera"})["locked"] is True
    out = k.call("unlock_account", {"user": "alex.rivera"})
    assert out["ok"] and k.ctx.backends.data["users"]["alex.rivera"]["locked"] is False
    assert "unlocked" in k.ctx.caller_messages[0]


def test_wf1_reset_uses_unique_temp_password_never_exposed():
    k = verified(kit())
    out = k.call("reset_password", {"user": "alex.rivera"})
    user = k.ctx.backends.data["users"]["alex.rivera"]
    secret = k.ctx.backends.delivered[0]["secret"]
    assert out["ok"] and out["force_change_at_next_sign_in"] and user["force_change"]
    assert len(secret) == 16 and secret not in str(out) and secret not in str(k.ctx.audit.entries)
    assert secret not in " ".join(k.ctx.caller_messages)
    other = verified(kit())
    other.call("reset_password", {"user": "alex.rivera"})
    assert other.ctx.backends.delivered[0]["secret"] != secret      # unique per reset


def test_wf1_denials_failed_mfa_and_other_users():
    k = kit(caller="jordan.lee")                                      # this user's MFA push is denied
    assert not k.call("verify_caller_identity", {})["ok"] and not k.ctx.identity_verified
    assert k.call("reset_password", {"user": "jordan.lee"})["needs_identity"]
    k = verified(kit())
    assert k.call("reset_password", {"user": "priya.shah"})["denied"]  # someone else's account


def test_wf1_reset_timeout_is_not_repeated_and_duplicate_is_skipped():
    k = verified(kit(faults={"reset_password": ["timeout"]}))
    out = k.call("reset_password", {"user": "alex.rivera"})
    assert out["status"] == "unknown_outcome" and out["escalate"]
    assert k.ctx.backends.calls.count("reset_password") == 1           # a timed-out write is never blindly retried
    store = MemoryStore()
    first = verified(kit(store=store))
    first.call("reset_password", {"user": "alex.rivera"})
    again = verified(kit(store=store))                                  # same ticket, same request (e.g. SQS redelivery)
    assert again.call("reset_password", {"user": "alex.rivera"})["duplicate"]
    assert again.ctx.backends.calls.count("reset_password") == 0


def test_wf1_unlock_retries_after_throttling():
    k = verified(kit(faults={"unlock": ["throttle", "throttle"]}))
    assert k.call("unlock_account", {"user": "alex.rivera"})["ok"]
    assert k.ctx.backends.calls.count("unlock") == 3
    k = verified(kit(faults={"unlock": ["throttle"] * 3}))
    assert k.call("unlock_account", {"user": "alex.rivera"})["status"] == "unavailable"


# ---------------------------------------------------------------- 2. Intune device diagnostics
def test_wf2_device_diagnostics_and_sync():
    k = kit(caller="taylor.kim")
    out = k.call("get_devices", {})
    assert "LAP-0587" in out["summary"] and "BitLocker off" in out["summary"]
    assert k.call("sync_device", {"device_id": "LAP-0587"})["ok"]
    assert k.ctx.backends.data["devices"]["LAP-0587"]["last_check_in_hours"] == 0
    assert k.call("sync_device", {"device_id": "LAP-0412"})["denied"]    # not the caller's device


def test_wf2_read_timeout_is_retried():
    k = kit(caller="taylor.kim", faults={"get_devices": ["timeout"]})
    assert k.call("get_devices", {})["ok"] and k.ctx.backends.calls.count("get_devices") == 2


# ---------------------------------------------------------------- 3. licenses
def test_wf3_license_needs_manager_approval_then_runs_once():
    store = MemoryStore()
    k = verified(kit(store=store))
    out = k.call("assign_license", {"user": "alex.rivera", "sku": "Visio_Plan2"})
    assert out["approval_requested"] and out["approver"] == "priya.shah"
    assert "Visio_Plan2" not in k.ctx.backends.data["users"]["alex.rivera"]["licenses"]
    desk = DryRunDesk()
    incident = {"sys_id": "a" * 32, "number": "INC0010100", "caller_id": "Alex Rivera", "caller_id.user_name": "alex.rivera"}
    assert handle_approval_decision(desk, incident, "alex.rivera", True, store)["outcome"] == "not_authorized"
    backends = SimBackends()
    result = handle_approval_decision(desk, incident, "priya.shah", True, store, backends, resolve_simulated=True)
    # a license is read back from the directory after it's assigned: verified, so resolved right away
    # (only because this test opts in: by default simulated evidence never closes a ticket, see below)
    assert result["outcome"] == "executed_resolved" and "Visio_Plan2" in backends.data["users"]["alex.rivera"]["licenses"]
    assert desk.actions[-1]["action"] == "resolve" and desk.actions[-1]["close_notes"].startswith(RESOLVED_MARKER)
    assert SIMULATED_NOTICE in desk.actions[-1]["comment"] and "SIMULATED" in desk.actions[-1]["close_notes"]
    assert handle_approval_decision(desk, incident, "priya.shah", True, store)["outcome"] == "no_pending_approval"


def test_wf3_no_seats_fails_and_escalates():
    store = MemoryStore()
    k = verified(kit(store=store))
    approval_id = k.call("assign_license", {"user": "alex.rivera", "sku": "PowerBI_Pro"}) and \
        args_key("assign_license", {"user": "alex.rivera", "sku": "PowerBI_Pro"}, "INC0010100")
    k.approvals.decide(approval_id, "priya.shah", True, AuditLog())
    full = SimBackends(scenario={"skus": {"PowerBI_Pro": {"name": "Power BI Pro", "total": 20, "used": 20}}})
    k2 = build_toolkit("INC0010100", "alex.rivera", store=store, backends=full)
    out = k2.run_approved(approval_id)
    assert out["status"] == "failed" and "no Power BI Pro seats" in out["error"]


# ---------------------------------------------------------------- 4. groups and access
def test_wf4_access_request_routes_to_group_owner():
    store = MemoryStore()
    k = verified(kit(caller="jordan.lee".replace("jordan.lee", "taylor.kim"), store=store))
    groups = k.call("find_access_group", {"resource": "Finance Reports SharePoint"})
    assert groups["groups"][0]["group"] == "Finance-Reports" and groups["groups"][0]["owner"] == "sam.ortiz"
    out = k.call("add_group_member", {"group": "Finance-Reports"})
    assert out["approval_requested"] and out["approver"] == "sam.ortiz"
    incident = {"sys_id": "a" * 32, "number": "INC0010100", "caller_id": "Taylor Kim", "caller_id.user_name": "taylor.kim"}
    desk = DryRunDesk()
    assert handle_approval_decision(desk, incident, "priya.shah", True, store)["outcome"] == "not_authorized"
    rejected = handle_approval_decision(desk, incident, "sam.ortiz", False, store)
    assert rejected["outcome"] == "rejected" and "not approved" in desk.actions[-1]["comment"]


# ---------------------------------------------------------------- 5. VPN troubleshooting
def test_wf5_vpn_checks_monitoring_and_uses_the_kb_fix():
    llm = Scripted(("get_service_health", {"service": "vpn"}), ("get_devices", {}),
                   ("find_similar_incidents", {}), ("send_fix_to_caller", {"search_id": "kb1"}))
    desk = DryRunDesk()
    agent = TicketAgent(llm, Assistant(answered()), desk, toolkit=kit())
    out = agent.run({"sys_id": "a" * 32, "number": "INC1", "short_description": "VPN keeps disconnecting",
                     "priority": "4 - Low"}, "VPN keeps disconnecting")
    assert out["action"] == "awaiting_caller" and "vpn.example.org" in desk.actions[0]["comment"]
    assert [s["tool"] for s in out["steps"]][:3] == ["search_knowledge_base", "get_service_health", "get_devices"]
    assert "get_service_health: ok: vpn: healthy" in desk.actions[0]["work_note"]
    assert "[SIMULATED]" in desk.actions[0]["work_note"]


def test_wf5_monitoring_outage_escalates():
    k = kit(scenario={"services": {"vpn": {"status": "outage", "detail": "vpn-gw-2 down"}}})
    assert "outage" in k.call("get_service_health", {"service": "vpn"})["summary"]
    llm = Scripted(("get_service_health", {"service": "vpn"}),
                   ("escalate_incident", {"team": "Network", "priority": "2", "reason": "VPN gateway outage"}))
    desk = DryRunDesk()
    out = TicketAgent(llm, Assistant(answered()), desk, toolkit=k).run(
        {"sys_id": "a" * 32, "number": "INC1", "short_description": "VPN down", "priority": "4 - Low"}, "VPN down")
    assert out["action"] == "escalated" and out["team"] == "Network" and out["priority"] == 2


# ---------------------------------------------------------------- 6. software
def test_wf6_software_policy():
    k = verified(kit())
    assert k.call("deploy_software", {"app": "zoom", "device_id": "LAP-0412"})["ok"]           # auto-approved
    assert k.call("deploy_software", {"app": "utorrent", "device_id": "LAP-0412"})["denied"]   # prohibited
    assert k.call("deploy_software", {"app": "adobe acrobat pro", "device_id": "LAP-0412"})["approval_requested"]
    assert k.call("deploy_software", {"app": "zoom", "device_id": "LAP-0587"})["denied"]       # not their device
    low_disk = verified(kit(caller="taylor.kim"))
    assert low_disk.call("deploy_software", {"app": "zoom", "device_id": "LAP-0587"})["status"] == "failed"


# ---------------------------------------------------------------- 7. triage and assignment (real ServiceNow path)
def test_wf7_triage_assigns_team_and_raises_priority_for_security():
    llm = Scripted(("escalate_incident", {"team": "Security", "priority": "2", "reason": "Clicked a phishing link."}))
    desk = DryRunDesk()
    out = TicketAgent(llm, Assistant(answered()), desk, toolkit=kit()).run(
        {"sys_id": "a" * 32, "number": "INC1", "short_description": "I clicked a bad link", "priority": "4 - Low"}, "x")
    assert out["team"] == "Security" and desk.actions[0]["priority"] == 2


# ---------------------------------------------------------------- 8. outage detection and coordination
def test_wf8_outage_links_parent_and_escalates():
    recent = [{"number": "INC0010001", "short_description": "VPN disconnecting all morning"},
              {"number": "INC0010002", "short_description": "VPN keeps disconnecting again"}]
    llm = Scripted(("find_similar_incidents", {}), ("link_to_outage", {"parent_number": "INC9999999"}),
                   ("link_to_outage", {"parent_number": "INC0010001"}),
                   ("escalate_incident", {"team": "Network", "priority": "2", "reason": "outage"}))
    desk = DryRunDesk(recent)
    out = TicketAgent(llm, Assistant(answered()), desk, toolkit=kit()).run(
        {"sys_id": "a" * 32, "number": "INC1", "short_description": "VPN keeps disconnecting", "priority": "4 - Low"}, "x")
    assert [a["action"] for a in desk.actions] == ["link_parent", "escalate"]      # bogus parent refused
    assert desk.actions[0]["parent"] == "INC0010001" and out["priority"] == 2


# ---------------------------------------------------------------- 9. log analysis
def test_wf9_log_analysis_finds_root_causes():
    result = analyze(LOGS["LAP-0412"])
    assert "lockout" in result["correlations"][0] and not result["needs_hardware"]
    assert analyze(LOGS["LAP-0587"])["needs_hardware"] is True
    k = kit(caller="taylor.kim")
    assert "hardware failure" in k.call("analyze_device_logs", {"device_id": "LAP-0587"})["summary"].lower()
    assert k.call("analyze_device_logs", {"device_id": "LAP-0412"})["denied"]


# ---------------------------------------------------------------- 10. Outlook / Microsoft 365
def test_wf10_mailbox_and_message_trace():
    k = kit(caller="taylor.kim")
    out = k.call("get_mailbox_diagnostics", {})
    assert "98% full" in out["summary"] and "48.7 GB" in out["summary"]
    assert "Junk" in k.call("run_message_trace", {"sender": "vendor@contoso-supplies.example"})["status"]
    flagged = kit(caller="jordan.lee").call("get_mailbox_diagnostics", {})
    assert flagged["external_forwarding"] and "forwarding" not in flagged and "Security" in flagged["summary"]


# ---------------------------------------------------------------- 11. onboarding and offboarding
def test_wf11_onboarding_needs_manager_and_hr_and_resumes_after_failure():
    store = MemoryStore()
    assert kit(caller="sam.ortiz").call("run_onboarding", {"employee": "casey.nguyen"})["denied"]   # not the manager
    k = verified(kit(caller="priya.shah", store=store, faults={"lifecycle:licenses": ["error"]}))
    out = k.call("run_onboarding", {"employee": "casey.nguyen"})
    assert out["approval_requested"] and out["approver"] == "dana.hr"
    approval_id = args_key("run_onboarding", {"employee": "casey.nguyen"}, "INC0010100")
    k.approvals.decide(approval_id, "dana.hr", True, AuditLog())
    failed = k.run_approved(approval_id)
    assert failed["status"] == "failed" and "stopped at 'licenses'" in failed["error"]
    assert "completed: create account, groups" in failed["error"]
    # the approval was used once; a new approval re-runs and resumes after the completed steps
    k.ctx.store.put(f"approval#{approval_id}", {**k.ctx.store.get(f"approval#{approval_id}"), "consumed": False})
    resumed = k.run_approved(approval_id)
    assert resumed["ok"] and "create account: already done" in resumed["steps"]
    assert k.ctx.backends.data["users"]["casey.nguyen"]["force_change"] is True


def test_wf11_offboarding():
    k = verified(kit(caller="priya.shah"))
    assert k.call("run_offboarding", {"employee": "morgan.fox"})["approval_requested"]
    assert verified(kit(caller="morgan.fox")).call("run_offboarding", {"employee": "morgan.fox"})["denied"]
    approval_id = args_key("run_offboarding", {"employee": "morgan.fox"}, "INC0010100")
    k.approvals.decide(approval_id, "dana.hr", True, AuditLog())
    out = k.run_approved(approval_id)
    user = k.ctx.backends.data["users"]["morgan.fox"]
    assert out["ok"] and not user["enabled"] and user["licenses"] == []
    assert k.ctx.backends.data["skus"]["PowerBI_Pro"]["used"] == 11


# ---------------------------------------------------------------- 12. follow-up and resolution verification
def test_wf12_verified_unlock_resolves_right_away_and_a_reply_reopens_it():
    llm = Scripted(("get_account_status", {"user": "alex.rivera"}), ("unlock_account", {"user": "alex.rivera"}),
                   ("verify_caller_identity", {}), ("unlock_account", {"user": "alex.rivera"}),
                   ("complete_and_confirm", {}))
    desk = DryRunDesk()
    agent = TicketAgent(llm, Assistant(NO_ANSWER), desk, toolkit=kit(), resolve_simulated=True)
    incident = {"sys_id": "a" * 32, "number": "INC1", "short_description": "I'm locked out", "priority": "4 - Low"}
    out = agent.run(incident, "I'm locked out")
    assert out["action"] == "resolved" and out["executed"] == ["unlock_account"]
    action = desk.actions[0]
    assert action["action"] == "resolve" and action["close_notes"].startswith(RESOLVED_MARKER)
    assert "unlocked" in action["comment"] and REOPEN_INVITE in action["comment"]
    assert action["comment"].index(SIMULATED_NOTICE) < action["comment"].index("unlocked")   # said up front
    assert "SIMULATED" in action["work_note"] and CONFIRM_MARKER not in action["work_note"]
    resolved = {**incident, "state": "Resolved", "caller_id": "Alex Rivera", "work_notes": action["work_note"],
                "close_notes": action["close_notes"]}

    def said(text, who="Alex Rivera"):
        return {**resolved, "comments": f"2026-10-19 10:05:00 - {who} (Additional comments)\n{text}\n\n"
                                        f"2026-10-19 10:00:00 - KB Assistant (Additional comments)\n{action['comment']}"}
    guard, follow = build_guardrail({}), DryRunDesk()
    assert handle_caller_reply(FakeLLM(), guard, follow, said("Yes it works now, thanks"))["outcome"] == "still_resolved"
    assert handle_caller_reply(FakeLLM(), guard, follow, said("Still broken", "Beth Tech"))["outcome"] == "not_caller"
    assert follow.actions == []                                  # thanks / someone else: nothing written
    result = handle_caller_reply(FakeLLM(), guard, follow, said("No, I still can't sign in"))
    assert result["outcome"] == "reopened" and follow.actions[0]["action"] == "back_to_technician"
    assert "reopened" in follow.actions[0]["comment"]


def test_simulated_evidence_never_closes_a_ticket_by_default():
    script = (("verify_caller_identity", {}), ("unlock_account", {}), ("complete_and_confirm", {}))
    incident = {"sys_id": "a" * 32, "number": "INC1", "short_description": "locked out", "priority": "4 - Low"}
    desk = DryRunDesk()
    out = TicketAgent(Scripted(*script), Assistant(NO_ANSWER), desk, toolkit=kit()).run(incident, "x")
    assert out["action"] == "awaiting_caller" and desk.actions[0]["action"] == "send_fix"
    assert SIMULATED_NOTICE in desk.actions[0]["comment"] and CONFIRM_MARKER in desk.actions[0]["work_note"]
    # a live (real tenant) tool with the same verification resolves without the switch
    live = kit()
    live.registry.tools["unlock_account"].simulated = False
    desk = DryRunDesk()
    out = TicketAgent(Scripted(*script), Assistant(NO_ANSWER), desk, toolkit=live).run(incident, "x")
    assert out["action"] == "resolved" and SIMULATED_NOTICE not in desk.actions[0]["comment"]
    assert "SIMULATED" not in desk.actions[0]["close_notes"]


def test_instant_resolve_needs_a_clean_non_security_run():
    def run_it(script, text="locked out"):
        desk = DryRunDesk()
        out = TicketAgent(Scripted(*script), Assistant(NO_ANSWER), desk, toolkit=kit(), resolve_simulated=True).run(
            {"sys_id": "a" * 32, "number": "INC1", "short_description": text, "priority": "4 - Low"}, text)
        return out, desk
    ok = (("verify_caller_identity", {}), ("unlock_account", {}))
    out, desk = run_it(ok + (("complete_and_confirm", {}),), "I typed my password on a weird site and got locked out")
    assert out["action"] == "escalated" and out["team"] == "Security" and out["priority"] == 2
    handoff = desk.actions[0]
    assert handoff["action"] == "back_to_technician" and handoff["team"] == "Security"
    assert "Security team will review" in handoff["comment"] and "reply" not in handoff["comment"].lower()
    # another write still waiting for an approver: the request isn't done, so nothing is closed
    out, _ = run_it(ok + (("assign_license", {"sku": "Visio_Plan2"}), ("complete_and_confirm", {})))
    assert out["action"] == "awaiting_caller"
    # a write that failed alongside the verified one
    out, desk = run_it(ok + (("reset_password", {"user": "priya.shah"}), ("complete_and_confirm", {})))
    assert out["action"] == "awaiting_caller" and "Not closed automatically" in desk.actions[0]["work_note"]


def test_approved_action_on_a_ticket_someone_took_over_is_not_closed():
    store = MemoryStore()
    k = verified(kit(caller="taylor.kim", store=store))
    k.call("add_group_member", {"group": "Finance-Reports"})
    incident = {"sys_id": "a" * 32, "number": "INC0010100", "caller_id": "Taylor Kim", "caller_id.user_name": "taylor.kim",
                "assigned_to": "Beth Tech", "priority": "4 - Low"}
    desk = DryRunDesk()
    out = handle_approval_decision(desk, incident, "sam.ortiz", True, store, resolve_simulated=True)
    assert out["outcome"] == "executed_awaiting_caller" and desk.actions[-1]["action"] == "send_fix"


def test_approval_path_waits_on_undecided_approvals_and_keeps_the_owning_team():
    store = MemoryStore()
    k = verified(kit(caller="taylor.kim", store=store))
    k.call("add_group_member", {"group": "Finance-Reports"})
    store.put("pending#INC0010100", {"ids": store.get("pending#INC0010100")["ids"] + ["missing-approval"]})
    incident = {"sys_id": "a" * 32, "number": "INC0010100", "caller_id": "Taylor Kim", "caller_id.user_name": "taylor.kim",
                "priority": "4 - Low"}
    desk = DryRunDesk()
    out = handle_approval_decision(desk, incident, "sam.ortiz", True, store, resolve_simulated=True)
    assert out["outcome"] == "executed_awaiting_caller"          # the other approval was never decided
    assert desk.actions[-1]["team"] == "Service Desk"
    store2 = MemoryStore()
    k2 = verified(kit(caller="taylor.kim", store=store2))
    k2.call("add_group_member", {"group": "Finance-Reports"})
    desk = DryRunDesk()
    handle_approval_decision(desk, {**incident, "assignment_group": "Security"}, "sam.ortiz", True, store2,
                             resolve_simulated=True)
    assert desk.actions[-1]["action"] == "send_fix" and desk.actions[-1]["team"] is None   # stays with Security


def test_a_reply_on_a_re_resolved_ticket_with_a_technician_is_left_alone():
    incident = {"sys_id": "a" * 32, "state": "Resolved", "close_notes": RESOLVED_MARKER + ": verified fix",
                "assigned_to": "Beth Tech", "caller_id": "Alex Rivera",
                "comments": "2026-10-19 10:05:00 - Alex Rivera (Additional comments)\nbroken again"}
    desk = DryRunDesk()
    assert handle_caller_reply(FakeLLM(), build_guardrail({}), desk, incident)["outcome"] == "technician_assigned"
    assert desk.actions == []


def test_wf12_unverifiable_action_still_waits_for_the_caller():
    llm = Scripted(("get_devices", {}), ("sync_device", {"device_id": "LAP-0412"}), ("complete_and_confirm", {}))
    desk = DryRunDesk()
    incident = {"sys_id": "a" * 32, "number": "INC1", "short_description": "Laptop not getting settings",
                "priority": "4 - Low"}
    out = TicketAgent(llm, Assistant(NO_ANSWER), desk, toolkit=kit()).run(incident, "x")
    assert out["action"] == "awaiting_caller" and desk.actions[0]["action"] == "send_fix"
    note, comment = desk.actions[0]["work_note"], desk.actions[0]["comment"]
    assert CONFIRM_MARKER in note and SIMULATED_NOTICE in comment and "SIMULATED backends" in note
    journal = (f"2026-10-19 10:05:00 - Alex Rivera (Additional comments)\nYes it works now, thanks\n\n"
               f"2026-10-19 10:00:00 - KB Assistant (Additional comments)\n{comment}")
    follow = DryRunDesk()
    result = handle_caller_reply(FakeLLM(), build_guardrail({}), follow,
                                 {**incident, "state": "On Hold", "caller_id": "Alex Rivera", "work_notes": note,
                                  "comments": journal})
    assert result["outcome"] == "resolved" and follow.actions[0]["action"] == "resolve"


def test_urgent_incidents_are_never_resolved_without_the_caller():
    llm = Scripted(("verify_caller_identity", {}), ("unlock_account", {}), ("complete_and_confirm", {}))
    desk = DryRunDesk()
    out = TicketAgent(llm, Assistant(NO_ANSWER), desk, toolkit=kit()).run(
        {"sys_id": "a" * 32, "number": "INC1", "short_description": "locked out", "priority": "2 - High"}, "x")
    assert out["action"] == "awaiting_caller" and desk.actions[0]["action"] == "send_fix"


def test_wf12_finish_tools_are_gated():
    llm = Scripted(("complete_and_confirm", {}), ("request_approval", {}))
    desk = DryRunDesk()
    out = TicketAgent(llm, Assistant(NO_ANSWER), desk, toolkit=kit()).run(
        {"sys_id": "a" * 32, "number": "INC1", "short_description": "help", "priority": "4 - Low"}, "help")
    assert out["action"] == "escalated" and out["forced"]   # nothing ran, nothing to confirm or approve


# ---------------------------------------------------------------- demo isolation
def test_demo_toolkit_is_simulated_only_and_isolated():
    a, b = demo_toolkit(), demo_toolkit()
    verified(a).call("unlock_account", {"user": "alex.rivera"})
    assert b.ctx.backends.data["users"]["alex.rivera"]["locked"] is True       # state is per request
    assert a.ctx.simulated_only
    real = build_registry()
    real.tools["get_devices"].simulated = False
    a.registry = real
    assert a.call("get_devices", {})["denied"]


def test_demo_llm_has_a_short_timeout_and_caller_mapping():
    llm = AnthropicLLM("m", "k", timeout=15)
    assert demo_llm(llm).timeout == 8 and llm.timeout == 15
    assert caller_id({"caller_id": "Alex Rivera", "caller_id.user_name": "Alex.Rivera"}) == "alex.rivera"
    assert caller_id({"caller_id": "Alex Rivera"}) == ""          # never guessed from the display name
    assert caller_id({"caller_id.user_name": "x; drop"}) == ""


@pytest.mark.parametrize("number,name", sorted(WORKFLOWS.items()))
def test_every_workflow_has_a_name(number, name):
    assert name and 1 <= number <= 12


# ---------------------------------------------------------------- Lambda wiring
def test_worker_enterprise_mode_and_approval_event(monkeypatch):
    import json
    import app
    from test_agent import SN
    monkeypatch.setenv("SN_MODE", "agent")
    monkeypatch.setenv("SN_ENTERPRISE", "true")
    monkeypatch.delenv("ENTERPRISE_TABLE", raising=False)
    store = MemoryStore()
    monkeypatch.setattr(app, "_enterprise_store", lambda: store)
    app._cache.clear()
    assistant = Assistant(NO_ANSWER)
    assistant.llm = Scripted(("verify_caller_identity", {}), ("find_access_group", {"resource": "Finance Reports"}),
                             ("add_group_member", {"group": "Finance-Reports"}), ("request_approval", {}))
    sn = SN({"number": "INC0010200", "short_description": "Need access to Finance Reports", "priority": "4 - Low",
             "caller_id": "Taylor Kim", "caller_id.user_name": "taylor.kim"})
    app._cache.update({"servicenow": sn, "assistant": assistant})
    summary = app.work_incident(sn, assistant, "a" * 32)
    assert summary["outcome"] == "awaiting_approval" and sn.updates[-1]["approval"] == "requested"
    # the approver (group owner) approves in ServiceNow -> approval event -> action runs, caller asked to confirm
    sn.inc.update(approval="Approved", work_notes=sn.updates[-1]["work_notes"])
    # the requester's own "approval" is refused; the decision comes signed from the Business Rule
    monkeypatch.setenv("SN_RESOLVE_SIMULATED", "true")   # dev-instance switch: resolve on simulated evidence
    msg = {"sys_id": "a" * 32, "event": "approval", "decision": "approved", "approver": "taylor.kim"}
    assert app.work_incident(sn, assistant, "a" * 32, "approval", event=msg)["outcome"] == "not_authorized"
    stale = {**msg, "approver": "sam.ortiz", "decision": "rejected"}            # no longer what the incident shows
    assert app.work_incident(sn, assistant, "a" * 32, "approval", event=stale)["outcome"] == "ignored"
    out = app.servicenow_worker({"Records": [{"messageId": "m1", "body": json.dumps({**msg, "approver": "sam.ortiz"}),
                                              "attributes": {"ApproximateReceiveCount": "1"}}]}, None)
    # group membership is read back after the change: verified, so the incident is resolved (Taylor can reopen)
    assert out == {"batchItemFailures": []} and sn.updates[-1]["state"] == "6"
    assert "added to Finance-Reports" in sn.updates[-1]["comments"] and SIMULATED_NOTICE in sn.updates[-1]["comments"]
    app._cache.clear()


def test_approval_events_are_validated_and_deduplicated(monkeypatch):
    import time
    import app
    from test_servicenow import KEY, FakeDynamo, FakeSQS, signed_event
    app._cache.clear()
    app._cache.update({"sn_webhook_secret": KEY, "sn_webhook_secret_at": time.time(),
                       "dynamodb": FakeDynamo(), "sqs": FakeSQS()})
    for name, value in (("SN_IDEMPOTENCY_TABLE", "t"), ("SN_QUEUE_URL", "q"), ("SN_WEBHOOK_SECRET_PARAM", "/p")):
        monkeypatch.setenv(name, value)
    event = {"sys_id": "a" * 32, "number": "INC0010200", "event": "approval", "version": "9",
             "decision": "approved", "approver": "sam.ortiz"}
    assert app.servicenow_event(signed_event(event), None)["statusCode"] == 202
    assert app._cache["sqs"].sent[-1]["approver"] == "sam.ortiz"
    assert app.servicenow_event(signed_event({**event, "version": "10", "approver": "bad user"}), None)["statusCode"] == 400
    assert app.servicenow_event(signed_event({**event, "version": "11", "decision": "maybe"}), None)["statusCode"] == 400
    assert app.servicenow_event(signed_event(event), None)["statusCode"] == 200
    assert app.servicenow_event(signed_event({**event, "version": ""}), None)["statusCode"] == 400
    app._cache.clear()


def test_public_demo_runs_simulated_workflow_trace(monkeypatch):
    import json
    import app
    from test_demo import FakeCounters
    app._cache.clear()
    assistant = Assistant(NO_ANSWER)
    assistant.llm = Scripted(("get_account_status", {"user": "alex.rivera"}), ("verify_caller_identity", {}),
                             ("unlock_account", {"user": "alex.rivera"}), ("complete_and_confirm", {}))
    app._cache.update({"demo_api_key": "k", "dynamodb": FakeCounters(), "assistant": assistant})
    monkeypatch.setenv("DEMO_TABLE", "t")
    out = app.demo_ask({"headers": {"x-api-key": "k", "x-demo-client": "1.2.3.4"},
                        "body": json.dumps({"question": "I'm locked out of my account"})}, None)
    plan = json.loads(out["body"])["agent"]
    assert plan["action"] == "resolved" and plan["dry_run"]
    tools = [(s["tool"], s["simulated"]) for s in plan["steps"]]
    assert ("unlock_account", True) in tools and ("search_knowledge_base", False) in tools
    assert "secret" not in out["body"].lower() and "temporary password" not in out["body"]
    app._cache.clear()



# ---------------------------------------------------------------- security review regressions
def test_reset_runs_once_per_ticket_whatever_the_arguments():
    k = verified(kit())
    for args in ({}, {"user": "alex.rivera"}, {"user": " Alex.Rivera "}, {"user": "alex.rivera", "x": 1}, {"user": None}):
        k.call("reset_password", args)
    assert k.ctx.backends.calls.count("reset_password") == 1 and len(k.ctx.backends.delivered) == 1


def test_timed_out_write_cannot_be_retried_by_the_model_or_a_redelivery():
    store = MemoryStore()
    k = verified(kit(store=store, faults={"reset_password": ["timeout"]}))
    assert k.call("reset_password", {})["status"] == "unknown_outcome"
    assert k.call("reset_password", {})["status"] == "unknown_outcome"
    again = verified(kit(store=store))
    assert again.call("reset_password", {})["status"] == "unknown_outcome"
    assert k.ctx.backends.calls.count("reset_password") == 1 and again.ctx.backends.calls.count("reset_password") == 0


def test_bad_arguments_and_handler_bugs_never_crash():
    k = verified(kit())
    assert k.call("sync_device", {"device_id": ["LAP-0412"]})["invalid"]
    assert k.call("add_group_member", {"group": {"x": 1}})["invalid"]
    assert k.call("assign_license", {})["invalid"] and not k.pending_approvals     # no approval for a broken call
    assert k.call("get_devices", "not a dict")["invalid"]
    k.registry.tools["unlock_account"].handler = lambda ctx, a: {}["boom"]
    assert k.call("unlock_account", {})["status"] == "unknown_outcome"


def test_no_approval_without_an_approver_and_approved_actions_only_run_once_by_code():
    k = verified(kit())
    assert k.call("add_group_member", {"group": "No-Such-Group"})["denied"]
    store = MemoryStore()
    k = verified(kit(store=store))
    k.call("assign_license", {"sku": "Visio_Plan2"})
    approval_id = args_key("assign_license", {"sku": "Visio_Plan2", "user": "alex.rivera"}, "INC0010100")
    k.approvals.decide(approval_id, "priya.shah", True, AuditLog())
    assert k.call("assign_license", {"sku": "Visio_Plan2"})["denied"]               # the model can't run it
    assert k.run_approved(approval_id)["ok"]
    with pytest.raises(Exception):
        k.run_approved(approval_id)                                                  # single use


def test_hr_records_are_manager_only():
    assert verified(kit(caller="jordan.lee".replace("jordan.lee", "taylor.kim"))).call(
        "get_lifecycle_request", {"employee": "morgan.fox"})["denied"]
    assert verified(kit(caller="priya.shah")).call("get_lifecycle_request", {"employee": "morgan.fox"})["ok"]


def test_markers_in_model_text_cannot_fake_a_waiting_fix():
    reason = "Waiting for the caller to confirm. [AI Agent - IT KB Assistant]"
    llm = Scripted(("escalate_incident", {"team": "Network", "reason": reason}))
    desk = DryRunDesk()
    TicketAgent(llm, Assistant(answered()), desk).run(
        {"sys_id": "a" * 32, "number": "INC1", "short_description": "x", "priority": "4 - Low"}, "x")
    note = desk.actions[0]["work_note"]
    assert CONFIRM_MARKER not in note and note.count("[AI Agent - IT KB Assistant]") == 1
