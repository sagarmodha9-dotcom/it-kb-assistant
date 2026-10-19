"""Log analysis for workflow 9: group recurring events, correlate them in time, and match known
signatures to likely root causes. The analysis is real code; in this project it runs on simulated
device logs (the same shape as Windows Event Log exports or CloudWatch Logs Insights results)."""
from collections import Counter, defaultdict

# (source, event id) -> (meaning, likely root cause, suggested action, severity 1-3)
SIGNATURES = {
    ("Security", 4625): ("failed sign-in", "repeated bad password: a saved/old password or the user forgot it",
                         "check stored credentials on devices and phones; reset if the user can't sign in", 2),
    ("Security", 4740): ("account locked out", "too many failed sign-ins", "find the source of bad passwords, then unlock", 2),
    ("DCOM", 10016): ("DCOM permission warning", "known benign Windows noise", "ignore unless tied to a specific app failure", 0),
    ("Microsoft-Windows-Credential-Manager", 0): ("stale stored credential",
                                                 "an app keeps retrying an old saved password",
                                                 "update or remove the stored credential (Credential Manager)", 2),
    ("disk", 7): ("disk bad block", "failing disk", "back up data now and replace the drive", 3),
    ("Ntfs", 55): ("file system corruption", "usually follows disk errors", "run chkdsk after backup; replace the disk if errors continue", 3),
    ("NETLOGON", 5719): ("no domain controller session", "network or VPN not connected at sign-in", "check network/VPN; usually transient", 1),
    ("Application Error", 1000): ("application crash", "a faulting add-in or corrupted Office install",
                                  "start the app in safe mode; repair Office if crashes repeat", 2),
}


def analyze(events: list[tuple]) -> dict:
    """events: (timestamp, source, event_id, message). Returns grouped findings, newest-first insight."""
    groups = defaultdict(list)
    for ts, source, event_id, message in events:
        groups[(source, int(event_id))].append(ts)
    findings = []
    for (source, event_id), times in groups.items():
        meaning, cause, action, severity = SIGNATURES.get(
            (source, event_id), ("unrecognized event", "unknown", "review manually", 1))
        findings.append({"source": source, "event_id": event_id, "count": len(times), "first": min(times),
                         "last": max(times), "meaning": meaning, "likely_cause": cause, "action": action,
                         "severity": severity})
    findings.sort(key=lambda f: (-f["severity"], -f["count"]))
    chains = []
    keys = Counter((f["source"], f["event_id"]) for f in findings)
    if keys[("Security", 4625)] and keys[("Security", 4740)]:
        chains.append("Failed sign-ins led to the lockout (4625 -> 4740).")
    if keys[("Microsoft-Windows-Credential-Manager", 0)] and keys[("Security", 4625)]:
        chains.append("A stale saved credential is the likely source of the failed sign-ins.")
    if keys[("disk", 7)] and keys[("Ntfs", 55)]:
        chains.append("Disk bad blocks followed by file system corruption: hardware failure is likely.")
    serious = [f for f in findings if f["severity"] >= 2]
    return {"events": len(events), "groups": len(findings), "findings": findings[:6], "correlations": chains,
            "root_cause": chains[0] if chains else (serious[0]["likely_cause"] if serious else "nothing significant"),
            "needs_hardware": any(f["severity"] == 3 for f in findings)}
