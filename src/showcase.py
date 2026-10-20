"""Public demo: the 12-capability catalog and the measured evaluation results shown on the page.

The interactive ticket desk lives in sandbox.py.
"""
import json

from enterprise.workflows import WORKFLOWS, build_registry

# ---------------------------------------------------------------- capability catalog
# kind: real | mixed | simulated. Wording matches the README table.
CAPABILITIES = {
    1: ("simulated", "Simulated Entra ID", "Account status, MFA-verified unlock or reset; a unique temporary "
        "password goes to the registered phone, change required at next sign-in.", "Identity, own account only"),
    2: ("simulated", "Simulated Intune", "Compliance, last check-in, BitLocker, OS and disk; sends an Intune sync.",
        "Own devices only"),
    3: ("simulated", "Simulated Microsoft 365", "Assigned licenses and seat inventory; assigns a license.",
        "Identity, manager approval, seat check"),
    4: ("simulated", "Simulated Entra ID", "Finds the group behind a site or channel and its owner; adds the caller.",
        "Identity, group-owner approval"),
    5: ("mixed", "KB real, monitoring simulated", "Service health and VPN client version, a cited KB fix, or "
        "escalation to Network.", "None beyond the agent rules"),
    6: ("simulated", "Simulated Intune + catalog", "Catalog policy (auto-approved, needs approval, prohibited), disk "
        "check, deployment.", "Identity, manager approval when required"),
    7: ("real", "ServiceNow", "Team, priority (only up, at most 2) and a handoff note.", "Fixed team list"),
    8: ("mixed", "ServiceNow real, monitoring simulated", "3+ similar incidents in 60 minutes: links the parent "
        "incident, priority 2, no self-service.", "Links only to a related incident"),
    9: ("mixed", "Analysis real, logs simulated", "Groups recurring events, correlates chains (failed sign-ins to "
        "lockout, disk errors to corruption), names a root cause.", "Own devices only"),
    10: ("simulated", "Simulated Exchange Online", "Quota, archive, Outlook data file size, external forwarding "
         "(flagged for Security), message trace.", "Own mailbox only"),
    11: ("simulated", "Simulated Entra ID / M365 / HR", "Role-template onboarding checklist and offboarding; resumes "
         "after a failed step.", "Manager requests, HR approval, identity"),
    12: ("real", "ServiceNow", "Verified fix resolves; caller confirmation resolves; no reply gets two reminders then "
         "closes as no response; any other reply reopens it.", "Caller only; never urgent, security or outage"),
}


def capabilities() -> list[dict]:
    tools = build_registry().by_workflow()
    return [{"n": n, "name": WORKFLOWS[n], "kind": kind, "integration": where, "does": does, "gates": gates,
             "tools": tools.get(n, [])} for n, (kind, where, does, gates) in sorted(CAPABILITIES.items())]


# ---------------------------------------------------------------- measured results (see README "Evaluation")
EVAL_RUN = "evals/results/s3-vectors-voyage-anthropic-bedrock.json"


def load_eval(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def eval_summary(data: dict) -> dict:
    """Headline numbers straight from the committed eval output, so the page can't drift from it."""
    e2e, retrieval = data["end_to_end"], data["retrieval"]
    sem, bm25 = retrieval["s3-vectors (voyage)"], retrieval["keyword-bm25 (baseline)"]
    return {"passed": e2e["passed"], "total": e2e["total"], "by_type": e2e["by_type"], "runs": data["config"]["runs"],
            "avg_latency_s": round(e2e["avg_latency_ms"] / 1000, 1), "p95_latency_s": round(e2e["p95_latency_ms"] / 1000, 1),
            "cost_per_question": e2e["est_model_cost_per_question_usd"],
            "hit1": {"semantic": sem["all"]["hit@1"], "bm25": bm25["all"]["hit@1"], "n": sem["all"]["n"],
                     "semantic_paraphrased": sem["paraphrased"]["hit@1"], "bm25_paraphrased": bm25["paraphrased"]["hit@1"],
                     "n_paraphrased": sem["paraphrased"]["n"]},
            "failures": sorted({f["id"] for f in e2e.get("failures", [])})}
