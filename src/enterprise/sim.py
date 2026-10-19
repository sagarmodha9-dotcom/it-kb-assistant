"""Simulated enterprise backends: a fictional company's directory (Entra ID-like), Intune, license
inventory, groups, software catalog, monitoring, device logs, Microsoft 365, and HR.

These stand in for Microsoft Graph, Intune, Exchange Online, CloudWatch, and an HR system until real,
authorized test tenants are connected. They behave like the real services where it matters to the
agent: lockouts, MFA push results, license seat limits, group owners, device compliance, service
health, throttling (TransientError), and timeouts (ToolTimeout) can all be scripted for tests.

All people, devices, and data here are fictional.
"""
import copy
import secrets
import string
import time

from .framework import ToolError, ToolTimeout, TransientError

SEED = {
    "users": {
        "alex.rivera": {"name": "Alex Rivera", "dept": "Sales", "title": "Account Executive", "manager": "priya.shah",
                        "locked": True, "failed_sign_ins": 6, "mfa": "approve", "phone": "+1 ***-***-1234",
                        "enabled": True, "licenses": ["M365_E3"], "groups": ["Sales-Team", "VPN-Users"],
                        "force_change": False},
        "jordan.lee": {"name": "Jordan Lee", "dept": "Finance", "title": "Analyst", "manager": "sam.ortiz",
                       "locked": False, "failed_sign_ins": 0, "mfa": "deny", "phone": "+1 ***-***-5678",
                       "enabled": True, "licenses": ["M365_E3"], "groups": ["Finance-Team"], "force_change": False},
        "taylor.kim": {"name": "Taylor Kim", "dept": "Marketing", "title": "Designer", "manager": "sam.ortiz",
                       "locked": False, "failed_sign_ins": 0, "mfa": "approve", "phone": "+1 ***-***-9012",
                       "enabled": True, "licenses": ["M365_E3"], "groups": ["Marketing-Team"], "force_change": False},
        "morgan.fox": {"name": "Morgan Fox", "dept": "Sales", "title": "Sales Associate", "manager": "priya.shah",
                       "locked": False, "failed_sign_ins": 0, "mfa": "approve", "phone": "+1 ***-***-3456",
                       "enabled": True, "licenses": ["M365_E3", "PowerBI_Pro"], "groups": ["Sales-Team", "VPN-Users"],
                       "force_change": False},
        "priya.shah": {"name": "Priya Shah", "dept": "Sales", "title": "Sales Director", "manager": "",
                       "locked": False, "mfa": "approve", "enabled": True, "licenses": ["M365_E5"], "groups": [],
                       "roles": ["manager", "group_owner"], "force_change": False},
        "sam.ortiz": {"name": "Sam Ortiz", "dept": "Finance", "title": "Finance Manager", "manager": "",
                      "locked": False, "mfa": "approve", "enabled": True, "licenses": ["M365_E3"], "groups": [],
                      "roles": ["manager", "group_owner"], "force_change": False},
        "lee.chen": {"name": "Lee Chen", "dept": "IT", "title": "M365 Administrator", "manager": "",
                     "locked": False, "mfa": "approve", "enabled": True, "licenses": ["M365_E5"], "groups": [],
                     "roles": ["license_admin"], "force_change": False},
        "dana.hr": {"name": "Dana Brooks", "dept": "HR", "title": "HR Business Partner", "manager": "",
                    "locked": False, "mfa": "approve", "enabled": True, "licenses": ["M365_E3"], "groups": [],
                    "roles": ["hr"], "force_change": False},
    },
    "groups": {
        "Finance-Reports": {"resource": "Finance Reports SharePoint site", "owner": "sam.ortiz", "members": ["jordan.lee"]},
        "Sales-Team": {"resource": "Sales Teams channel", "owner": "priya.shah", "members": ["alex.rivera", "morgan.fox"]},
        "Marketing-Team": {"resource": "Marketing Teams channel", "owner": "sam.ortiz", "members": ["taylor.kim"]},
        "VPN-Users": {"resource": "Corporate VPN", "owner": "priya.shah", "members": ["alex.rivera", "morgan.fox"]},
        "Finance-Team": {"resource": "Finance Teams channel", "owner": "sam.ortiz", "members": ["jordan.lee"]},
    },
    "skus": {"M365_E3": {"name": "Microsoft 365 E3", "total": 50, "used": 50},
             "M365_E5": {"name": "Microsoft 365 E5", "total": 10, "used": 3},
             "Visio_Plan2": {"name": "Visio Plan 2", "total": 10, "used": 4},
             "PowerBI_Pro": {"name": "Power BI Pro", "total": 20, "used": 12}},
    "devices": {
        "LAP-0412": {"user": "alex.rivera", "os": "Windows 11 23H2", "compliant": True, "last_check_in_hours": 2,
                     "bitlocker": True, "vpn_client": "GlobalProtect 6.2.1", "free_disk_gb": 84},
        "LAP-0587": {"user": "taylor.kim", "os": "Windows 11 22H2", "compliant": False, "last_check_in_hours": 216,
                     "bitlocker": False, "vpn_client": "GlobalProtect 5.2.8", "free_disk_gb": 3,
                     "issues": ["BitLocker off", "OS out of date", "not checked in for 9 days"]},
        "MAC-0233": {"user": "jordan.lee", "os": "macOS 14.6", "compliant": True, "last_check_in_hours": 5,
                     "bitlocker": True, "vpn_client": "GlobalProtect 6.2.1", "free_disk_gb": 120},
    },
    "software": {
        "zoom": {"name": "Zoom Workplace", "status": "auto_approved"},
        "python": {"name": "Python 3.12", "status": "auto_approved"},
        "adobe acrobat pro": {"name": "Adobe Acrobat Pro", "status": "manager_approval", "seats": 5},
        "visio": {"name": "Microsoft Visio", "status": "manager_approval", "license": "Visio_Plan2"},
        "utorrent": {"name": "uTorrent", "status": "prohibited", "reason": "peer-to-peer software is not allowed"},
    },
    "services": {"vpn": {"status": "healthy", "detail": "All VPN gateways normal"},
                 "exchange": {"status": "healthy", "detail": "Exchange Online normal"},
                 "teams": {"status": "healthy", "detail": "Microsoft Teams normal"},
                 "sharepoint": {"status": "healthy", "detail": "SharePoint Online normal"}},
    "mailboxes": {
        "alex.rivera": {"quota_used_pct": 41, "forwarding": "", "ost_gb": 6.2, "rules": 3, "archive": True},
        "taylor.kim": {"quota_used_pct": 98, "forwarding": "", "ost_gb": 48.7, "rules": 12, "archive": False},
        "jordan.lee": {"quota_used_pct": 55, "forwarding": "jordan.lee.personal@gmail.com", "ost_gb": 9.1,
                       "rules": 4, "archive": True},
    },
    "message_trace": {
        ("taylor.kim", "vendor@contoso-supplies.example"): "Delivered to Junk Email folder (spam confidence 6)",
        ("alex.rivera", "client@fabrikam.example"): "Rejected: recipient mailbox full? No - sender domain failed DMARC",
    },
    "hr": {
        "casey.nguyen": {"name": "Casey Nguyen", "type": "onboarding", "role": "Sales Associate", "dept": "Sales",
                         "manager": "priya.shah", "start": "2026-10-26"},
        "morgan.fox": {"name": "Morgan Fox", "type": "offboarding", "dept": "Sales", "manager": "priya.shah",
                       "last_day": "2026-10-23"},
    },
    "role_templates": {"Sales Associate": {"groups": ["Sales-Team", "VPN-Users"], "licenses": ["M365_E5"],
                                           "software": ["zoom"]}},
}

LOGS = {
    "LAP-0412": [
        ("2026-10-19 08:01:12", "Security", 4625, "An account failed to log on. Account: alex.rivera. Failure: bad password"),
        ("2026-10-19 08:01:40", "Security", 4625, "An account failed to log on. Account: alex.rivera. Failure: bad password"),
        ("2026-10-19 08:02:05", "Security", 4625, "An account failed to log on. Account: alex.rivera. Failure: bad password"),
        ("2026-10-19 08:02:31", "Security", 4740, "A user account was locked out. Account: alex.rivera"),
        ("2026-10-19 08:05:00", "DCOM", 10016, "The application-specific permission settings do not grant Local Activation"),
        ("2026-10-19 08:10:00", "DCOM", 10016, "The application-specific permission settings do not grant Local Activation"),
        ("2026-10-19 07:58:44", "Microsoft-Windows-Credential-Manager", 0,
         "Stored credential for 'outlook.office365.com' is out of date"),
    ],
    "LAP-0587": [
        ("2026-10-19 09:12:01", "disk", 7, "The device has a bad block on \\Device\\Harddisk0\\DR0"),
        ("2026-10-19 09:12:44", "disk", 7, "The device has a bad block on \\Device\\Harddisk0\\DR0"),
        ("2026-10-19 09:15:02", "Ntfs", 55, "A corruption was discovered in the file system structure on volume C:"),
        ("2026-10-19 09:20:30", "NETLOGON", 5719, "This computer was not able to set up a secure session with a domain controller"),
        ("2026-10-19 09:25:30", "Application Error", 1000, "Faulting application OUTLOOK.EXE, faulting module mso.dll"),
        ("2026-10-19 09:31:10", "Application Error", 1000, "Faulting application OUTLOOK.EXE, faulting module mso.dll"),
    ],
}


class SimBackends:
    """One fictional company. `faults` scripts failures per operation for tests:
    {"reset_password": ["timeout"], "assign_license": ["throttle", "throttle"]}."""

    def __init__(self, faults: dict | None = None, scenario: dict | None = None):
        self.data = copy.deepcopy(SEED)
        for key, value in (scenario or {}).items():
            self.data[key].update(value)
        self.faults = {k: list(v) for k, v in (faults or {}).items()}
        self.delivered = []          # secure-channel deliveries (temporary passwords); never returned
        self.calls = []
        self.directory = SimDirectoryRoles(self)

    def _maybe_fail(self, op: str):
        self.calls.append(op)
        queue = self.faults.get(op)
        if queue:
            fault = queue.pop(0)
            if fault == "throttle":
                raise TransientError(f"{op}: throttled (HTTP 429)")
            if fault == "timeout":
                raise ToolTimeout(f"{op}: no response within the timeout")
            if fault == "error":
                raise ToolError(f"{op}: backend rejected the request")

    def user(self, user_id: str) -> dict:
        user = self.data["users"].get(user_id)
        if not user:
            raise ToolError(f"user {user_id} not found")
        return user

    # identity
    def mfa_push(self, user_id: str) -> str:
        self._maybe_fail("mfa_push")
        return self.user(user_id)["mfa"]          # approve | deny | timeout

    # passwords (Entra ID-like)
    def reset_password(self, user_id: str) -> dict:
        self._maybe_fail("reset_password")
        user = self.user(user_id)
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_"
        while True:   # unique, random, meets complexity: upper, lower, digit, symbol, 16 chars
            temp = "".join(secrets.choice(alphabet) for _ in range(16))
            if (any(c.islower() for c in temp) and any(c.isupper() for c in temp) and any(c.isdigit() for c in temp)
                    and any(c in "!@#$%^&*-_" for c in temp)):
                break
        user.update(force_change=True, locked=False, failed_sign_ins=0, password_set_at=time.time())
        self.delivered.append({"user": user_id, "channel": "sms", "to": user["phone"], "secret": temp})
        return {"delivered_to": user["phone"], "force_change_at_next_sign_in": True}

    def unlock(self, user_id: str) -> None:
        self._maybe_fail("unlock")
        self.user(user_id).update(locked=False, failed_sign_ins=0)


class SimDirectoryRoles:
    """Who may approve what (stand-in for Entra ID roles, managers, and group owners)."""

    def __init__(self, backends: SimBackends):
        self.b = backends

    def approver_for(self, role: str, requester: str, args: dict) -> str:
        users = self.b.data["users"]
        if role == "manager":
            target = args.get("user") or requester
            return users.get(target, {}).get("manager") or users.get(requester, {}).get("manager", "")
        if role == "group_owner":
            return self.b.data["groups"].get(args.get("group", ""), {}).get("owner", "")
        return next((u for u, d in users.items() if role in d.get("roles", [])), "")

    def has_role(self, user: str, role: str, request: dict) -> bool:
        if user not in self.b.data["users"]:
            return False
        return self.approver_for(role, request["requested_by"], request["args"]) == user
