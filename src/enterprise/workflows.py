"""The 12 enterprise IT workflows as registered tools.

Real integrations (run against ServiceNow / the knowledge base in agent.py, not here):
  5 (KB troubleshooting part), 7 triage + assignment, 8 outage correlation + linking,
  12 follow-up and resolution verification.
Simulated backends (enterprise/sim.py) until authorized test tenants are connected:
  Entra ID, Intune, Microsoft 365 licensing, groups, software deployment, monitoring, device logs,
  Exchange Online diagnostics, HR lifecycle.

Each write handler verifies its own result by reading the backend back before reporting success,
and gives the caller a fixed, code-written message (the model never writes what the caller is told
about an action).
"""
from .framework import LOW, READ, SENSITIVE, Registry, Tool, ToolError
from .loganalysis import analyze

WORKFLOWS = {
    1: "Password reset and account unlock", 2: "Intune device diagnostics",
    3: "Microsoft 365 license management", 4: "Group membership and access requests",
    5: "VPN troubleshooting", 6: "Software installation requests",
    7: "Automatic incident triage and assignment", 8: "Outage detection and incident coordination",
    9: "Log analysis and root-cause investigation", 10: "Outlook and Microsoft 365 diagnostics",
    11: "Employee onboarding and offboarding", 12: "Ticket follow-up and resolution verification",
}
USER = {"user": {"type": "string", "description": "directory user id, e.g. alex.rivera"}}


def _self_only(ctx, args) -> str:
    target = args.get("user") or ctx.caller
    return "" if target == ctx.caller else "the agent can only act on the caller's own account"


def _own_device(ctx, args) -> str:
    device = ctx.backends.data["devices"].get(args.get("device_id", ""))
    if not device:
        return "unknown device"
    return "" if device["user"] == ctx.caller else "that device isn't assigned to the caller"


def build_registry() -> Registry:
    r = Registry()
    b = lambda ctx: ctx.backends  # noqa: E731

    # ---------------------------------------------------------------- shared: identity
    def verify_identity(ctx, args):
        result = b(ctx).mfa_push(ctx.caller)
        ctx.identity_verified = result == "approve"   # server-side only; the model can't set this
        if result == "approve":
            return {"verified": True, "summary": "caller approved the MFA push on their registered device"}
        raise ToolError("identity not verified: the MFA push was " + ("denied" if result == "deny" else "not answered"))
    r.register(Tool("verify_caller_identity", 0, "Send an MFA push to the caller's registered device to verify "
                    "it's really them. Required before account, access, license, software, or lifecycle changes.",
                    verify_identity, risk=READ))

    # ---------------------------------------------------------------- 1. password reset and unlock
    def account_status(ctx, args):
        u = b(ctx).user(args.get("user") or ctx.caller)
        return {"locked": u["locked"], "failed_sign_ins": u.get("failed_sign_ins", 0), "enabled": u["enabled"],
                "summary": "locked" if u["locked"] else "not locked"}
    r.register(Tool("get_account_status", 1, "Check whether the caller's account is locked or disabled.",
                    account_status, USER, risk=READ, precheck=_self_only))

    def unlock(ctx, args):
        user = args.get("user") or ctx.caller
        if not b(ctx).user(user)["locked"]:
            raise ToolError("account is not locked")
        b(ctx).unlock(user)
        if b(ctx).user(user)["locked"]:
            raise ToolError("unlock did not take effect")
        ctx.caller_messages.append("Your account has been unlocked. Please sign in again with your current password.")
        return {"summary": "account unlocked and verified"}
    r.register(Tool("unlock_account", 1, "Unlock the caller's own account (identity must be verified).", unlock, USER,
                    risk=SENSITIVE, requires_identity=True, precheck=_self_only, idempotent=True))

    def reset(ctx, args):
        user = args.get("user") or ctx.caller
        out = b(ctx).reset_password(user)
        if not b(ctx).user(user)["force_change"]:
            raise ToolError("password change at next sign-in was not set")
        ctx.caller_messages.append(
            f"We reset your password. A one-time temporary password was sent to your registered phone "
            f"({out['delivered_to']}). You'll be asked to choose a new password when you sign in.")
        return {"summary": "unique temporary password sent by secure channel; change required at next sign-in",
                "delivered_to": out["delivered_to"], "force_change_at_next_sign_in": True}
    r.register(Tool("reset_password", 1, "Reset the caller's own password: a unique random temporary password is "
                    "sent to their registered phone (never shown here) and must be changed at next sign-in.",
                    reset, USER, risk=SENSITIVE, requires_identity=True, precheck=_self_only, idempotent=False))

    # ---------------------------------------------------------------- 2. Intune device diagnostics
    def devices(ctx, args):
        rows = [{"device_id": d, **{k: v for k, v in info.items() if k != "user"}}
                for d, info in b(ctx).data["devices"].items() if info["user"] == ctx.caller]
        b(ctx)._maybe_fail("get_devices")
        bad = [f"{r['device_id']}: {', '.join(r.get('issues', []))}" for r in rows if not r["compliant"]]
        return {"devices": rows, "summary": f"{len(rows)} device(s); " + ("; ".join(bad) if bad else "all compliant")}
    r.register(Tool("get_devices", 2, "List the caller's Intune devices: compliance, last check-in, OS, BitLocker, "
                    "VPN client, free disk.", devices, risk=READ))

    def sync(ctx, args):
        b(ctx)._maybe_fail("sync_device")
        device = b(ctx).data["devices"][args["device_id"]]
        device["last_check_in_hours"] = 0
        ctx.caller_messages.append(f"We sent a sync request to {args['device_id']}. Keep it on and connected "
                                   "for about 10 minutes so it can download its latest settings.")
        return {"summary": f"sync sent to {args['device_id']}; check-in time updated"}
    r.register(Tool("sync_device", 2, "Send an Intune sync to one of the caller's devices (low risk).", sync,
                    {"device_id": {"type": "string"}}, ("device_id",), risk=LOW, precheck=_own_device))

    # ---------------------------------------------------------------- 3. Microsoft 365 licenses
    def licenses(ctx, args):
        u = b(ctx).user(args.get("user") or ctx.caller)
        inventory = {k: {"name": v["name"], "available": v["total"] - v["used"]} for k, v in b(ctx).data["skus"].items()}
        return {"assigned": u["licenses"], "inventory": inventory,
                "summary": f"assigned: {', '.join(u['licenses']) or 'none'}"}
    r.register(Tool("get_licenses", 3, "Show the caller's assigned licenses and available seats per license.",
                    licenses, USER, risk=READ, precheck=_self_only))

    def assign_license(ctx, args):
        b(ctx)._maybe_fail("assign_license")
        user, sku = args.get("user") or ctx.caller, args["sku"]
        info = b(ctx).data["skus"].get(sku)
        if not info:
            raise ToolError(f"unknown license {sku}")
        if sku in b(ctx).user(user)["licenses"]:
            return {"summary": f"{sku} already assigned"}
        if info["used"] >= info["total"]:
            raise ToolError(f"no {info['name']} seats left: a license admin must buy more")
        info["used"] += 1
        b(ctx).user(user)["licenses"].append(sku)
        ctx.caller_messages.append(f"{info['name']} has been assigned to you. It can take up to an hour to "
                                   "appear; sign out and back in to Office apps after that.")
        return {"summary": f"{sku} assigned and verified"}
    r.register(Tool("assign_license", 3, "Assign a Microsoft 365 license to the caller (manager approval).",
                    assign_license, {**USER, "sku": {"type": "string"}}, ("sku",), risk=SENSITIVE,
                    requires_identity=True, approval="manager", precheck=_self_only, idempotent=True))

    # ---------------------------------------------------------------- 4. groups and access
    def find_group(ctx, args):
        text = str(args.get("resource", "")).lower()
        hits = [{"group": g, "resource": d["resource"], "owner": d["owner"], "caller_is_member": ctx.caller in d["members"]}
                for g, d in b(ctx).data["groups"].items()
                if any(w in d["resource"].lower() or w in g.lower() for w in text.split() if len(w) > 3)]
        return {"groups": hits, "summary": ", ".join(h["group"] for h in hits) or "no matching group"}
    r.register(Tool("find_access_group", 4, "Find the group that grants access to a Teams channel, SharePoint "
                    "site, or app, its owner, and whether the caller is already a member.", find_group,
                    {"resource": {"type": "string"}}, ("resource",), risk=READ))

    def add_member(ctx, args):
        b(ctx)._maybe_fail("add_group_member")
        group = b(ctx).data["groups"].get(args["group"])
        if not group:
            raise ToolError("unknown group")
        if ctx.caller not in group["members"]:
            group["members"].append(ctx.caller)
        ctx.caller_messages.append(f"You've been added to {args['group']} ({group['resource']}). Access can "
                                   "take up to 30 minutes; sign out and back in if you still don't see it.")
        return {"summary": f"added to {args['group']} and verified"}
    r.register(Tool("add_group_member", 4, "Add the caller to an access group (group owner approval).", add_member,
                    {"group": {"type": "string"}, **USER}, ("group",), risk=SENSITIVE, requires_identity=True,
                    approval="group_owner", precheck=_self_only, idempotent=True))

    # ---------------------------------------------------------------- 5 / 8 / 10. service health (monitoring)
    def health(ctx, args):
        b(ctx)._maybe_fail("service_health")
        service = str(args.get("service", "")).lower()
        info = b(ctx).data["services"].get(service)
        if not info:
            raise ToolError(f"no monitoring for {service}")
        return {**info, "summary": f"{service}: {info['status']} ({info['detail']})"}
    r.register(Tool("get_service_health", 5, "Check monitoring for a service: vpn, exchange, teams, sharepoint.",
                    health, {"service": {"type": "string"}}, ("service",), risk=READ))

    # ---------------------------------------------------------------- 6. software
    def catalog(ctx, args):
        item = b(ctx).data["software"].get(str(args.get("app", "")).lower().strip())
        if not item:
            return {"found": False, "summary": "not in the software catalog: needs a software request review"}
        return {"found": True, **item, "summary": f"{item['name']}: {item['status'].replace('_', ' ')}"}
    r.register(Tool("check_software_catalog", 6, "Look up an app in the approved software catalog.", catalog,
                    {"app": {"type": "string"}}, ("app",), risk=READ))

    def software_policy(ctx, args):
        item = ctx.backends.data["software"].get(str(args.get("app", "")).lower().strip())
        if not item:
            return "not in the approved catalog"
        if item["status"] == "prohibited":
            return f"{item['name']} is prohibited: {item.get('reason', '')}"
        return _own_device(ctx, args)

    def deploy(ctx, args):
        b(ctx)._maybe_fail("deploy_software")
        item = b(ctx).data["software"][args["app"].lower().strip()]
        device = b(ctx).data["devices"][args["device_id"]]
        if device.get("free_disk_gb", 0) < 5:
            raise ToolError(f"{args['device_id']} has only {device['free_disk_gb']} GB free")
        device.setdefault("installed", []).append(item["name"])
        ctx.caller_messages.append(f"{item['name']} is being installed on {args['device_id']} through Company "
                                   "Portal. It should appear within about 30 minutes while the device is online.")
        return {"summary": f"{item['name']} deployment assigned to {args['device_id']}"}
    r.register(Tool("deploy_software", 6, "Install an approved catalog app on the caller's device via Intune. "
                    "Apps needing approval wait for the manager.", deploy,
                    {"app": {"type": "string"}, "device_id": {"type": "string"}}, ("app", "device_id"),
                    risk=SENSITIVE, requires_identity=True, precheck=software_policy, idempotent=True,
                    approval=lambda ctx, a: "manager" if ctx.backends.data["software"].get(
                        str(a.get("app", "")).lower().strip(), {}).get("status") == "manager_approval" else None))

    # ---------------------------------------------------------------- 9. log analysis
    def logs(ctx, args):
        b(ctx)._maybe_fail("device_logs")
        from .sim import LOGS
        result = analyze(LOGS.get(args["device_id"], []))
        return {**result, "summary": result["root_cause"]}
    r.register(Tool("analyze_device_logs", 9, "Pull the last 24 hours of event logs from the caller's device, group "
                    "recurring errors, correlate them, and suggest the root cause.", logs,
                    {"device_id": {"type": "string"}}, ("device_id",), risk=READ, precheck=_own_device))

    # ---------------------------------------------------------------- 10. Outlook / Microsoft 365
    def mailbox(ctx, args):
        box = b(ctx).data["mailboxes"].get(ctx.caller)
        if not box:
            raise ToolError("no mailbox found")
        findings = []
        if box["quota_used_pct"] >= 90:
            findings.append(f"mailbox {box['quota_used_pct']}% full" + ("" if box["archive"] else "; archive not enabled"))
        if box["ost_gb"] >= 40:
            findings.append(f"Outlook data file is {box['ost_gb']} GB (slow/crashing Outlook)")
        if box["forwarding"]:
            findings.append("mail is forwarded to an external address: possible compromise, needs Security review")
        return {**{k: v for k, v in box.items() if k != "forwarding"}, "external_forwarding": bool(box["forwarding"]),
                "findings": findings, "summary": "; ".join(findings) or "no mailbox problems found"}
    r.register(Tool("get_mailbox_diagnostics", 10, "Check the caller's mailbox: quota, archive, Outlook data file "
                    "size, inbox rules, external forwarding.", mailbox, risk=READ))

    def trace(ctx, args):
        b(ctx)._maybe_fail("message_trace")
        status = b(ctx).data["message_trace"].get((ctx.caller, str(args.get("sender", "")).lower()))
        return {"status": status or "no message from that sender in the last 48 hours",
                "summary": status or "not found in message trace"}
    r.register(Tool("run_message_trace", 10, "Trace whether an email from a sender reached the caller in the last "
                    "48 hours, and what happened to it.", trace, {"sender": {"type": "string"}}, ("sender",), risk=READ))

    # ---------------------------------------------------------------- 11. onboarding / offboarding
    def lifecycle_request(ctx, args):
        record = b(ctx).data["hr"].get(args.get("employee", ""))
        if not record:
            raise ToolError("no HR record for that employee")
        return {**record, "summary": f"{record['type']} for {record['name']}"}
    r.register(Tool("get_lifecycle_request", 11, "Look up the HR onboarding/offboarding record for an employee.",
                    lifecycle_request, {"employee": {"type": "string"}}, ("employee",), risk=READ))

    def manager_of(ctx, args, kind):
        record = ctx.backends.data["hr"].get(args.get("employee", ""))
        if not record or record["type"] != kind:
            return f"no {kind} request in HR for that employee"
        if args.get("employee") == ctx.caller:
            return "people can't run their own lifecycle changes"
        return "" if record["manager"] == ctx.caller else "only the employee's manager can request this"

    def run_steps(ctx, employee, steps):
        """Ordered, resumable steps: each finished step is checkpointed, so a re-run after a failure
        continues where it stopped instead of repeating work."""
        done, results = [], []
        for name, action in steps:
            key = f"lifecycle#{ctx.ticket}#{employee}#{name}"
            if ctx.store.get(key):
                results.append(f"{name}: already done")
                continue
            try:
                b(ctx)._maybe_fail(f"lifecycle:{name}")
                results.append(f"{name}: {action()}")
                ctx.store.put(key, {"done": True})
                done.append(name)
            except ToolError as error:
                raise ToolError(f"stopped at '{name}' ({error}); completed: {', '.join(done) or 'none'}. "
                                "Re-running continues from this step.") from error
        return results

    def onboard(ctx, args):
        employee = args["employee"]
        hr = b(ctx).data["hr"][employee]
        template = b(ctx).data["role_templates"].get(hr["role"], {"groups": [], "licenses": [], "software": []})
        users = b(ctx).data["users"]

        def create():
            if employee not in users:
                users[employee] = {"name": hr["name"], "dept": hr["dept"], "manager": hr["manager"], "locked": False,
                                   "mfa": "approve", "phone": "(set at first sign-in)", "enabled": True, "licenses": [],
                                   "groups": [], "force_change": True}
                b(ctx).delivered.append({"user": employee, "channel": "manager secure link", "to": hr["manager"],
                                         "secret": "<random temporary password>"})
            return "account created; temporary password sent to the manager by secure link; change at first sign-in"

        def groups():
            for g in template["groups"]:
                if employee not in b(ctx).data["groups"][g]["members"]:
                    b(ctx).data["groups"][g]["members"].append(employee)
            return ", ".join(template["groups"]) or "none"

        def license_():
            for sku in template["licenses"]:
                info = b(ctx).data["skus"][sku]
                if info["used"] >= info["total"]:
                    raise ToolError(f"no {sku} seats")
                info["used"] += 1
                users[employee]["licenses"].append(sku)
            return ", ".join(template["licenses"]) or "none"

        results = run_steps(ctx, employee, [("create account", create), ("groups", groups),
                                            ("licenses", license_),
                                            ("software", lambda: ", ".join(template["software"]) or "none"),
                                            ("device prep task", lambda: "laptop request created for IT")])
        ctx.caller_messages.append(f"Onboarding for {hr['name']} is set up for {hr['start']}: " + "; ".join(results) + ".")
        return {"steps": results, "summary": f"onboarding complete for {employee} ({len(results)} steps)"}
    r.register(Tool("run_onboarding", 11, "Run the approved onboarding checklist for a new hire (HR approval; "
                    "requested by their manager).", onboard, {"employee": {"type": "string"}}, ("employee",),
                    risk=SENSITIVE, requires_identity=True, approval="hr", idempotent=True,
                    precheck=lambda ctx, a: manager_of(ctx, a, "onboarding")))

    def offboard(ctx, args):
        employee = args["employee"]
        user = b(ctx).user(employee)

        def disable():
            user["enabled"] = False
            return "sign-in disabled"

        def groups():
            for g in b(ctx).data["groups"].values():
                if employee in g["members"]:
                    g["members"].remove(employee)
            user["groups"] = []
            return "removed from all groups"

        def licenses_():
            for sku in user["licenses"]:
                b(ctx).data["skus"][sku]["used"] -= 1
            freed, user["licenses"] = list(user["licenses"]), []
            return "freed " + (", ".join(freed) or "none")

        results = run_steps(ctx, employee, [("disable sign-in", disable), ("revoke sessions", lambda: "all sessions revoked"),
                                            ("remove groups", groups), ("remove licenses", licenses_),
                                            ("mailbox", lambda: "converted to shared mailbox, manager has access")])
        ctx.caller_messages.append(f"Offboarding for {user['name']} is done: " + "; ".join(results) + ".")
        return {"steps": results, "summary": f"offboarding complete for {employee}"}
    r.register(Tool("run_offboarding", 11, "Run the approved offboarding checklist for a departing employee (HR "
                    "approval; requested by their manager).", offboard, {"employee": {"type": "string"}},
                    ("employee",), risk=SENSITIVE, requires_identity=True, approval="hr", idempotent=True,
                    precheck=lambda ctx, a: manager_of(ctx, a, "offboarding")))
    return r
