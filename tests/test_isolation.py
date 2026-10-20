"""The public demo route can't reach ServiceNow or anything privileged: checked against template.yaml itself,
so a future change that widens the demo function's permissions fails here before it is deployed."""
import json
import os

import pytest

yaml = pytest.importorskip("yaml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CfnLoader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {f"!{suffix}": value}


CfnLoader.add_multi_constructor("!", _tag)


@pytest.fixture(scope="module")
def template():
    with open(os.path.join(ROOT, "template.yaml")) as f:
        return yaml.load(f, Loader=CfnLoader)


def demo(template):
    return template["Resources"]["DemoAskFunction"]


def test_demo_function_has_no_servicenow_settings(template):
    env = demo(template)["Properties"]["Environment"]["Variables"]
    shared = template["Globals"]["Function"]["Environment"]["Variables"]
    for name in list(env) + list(shared):
        assert not name.startswith(("SN_", "SERVICENOW")), name
        assert name not in ("QUEUE_URL", "STATE_TABLE", "IDEMPOTENCY_TABLE", "TRIAGE_API_URL"), name


def test_demo_function_permissions_are_minimal(template):
    policies = demo(template)["Properties"]["Policies"]
    text = json.dumps(policies)
    for forbidden in ("ServiceNow", "Sn", "Queue", "sqs:", "EnterpriseState", "Idempotency", "PutItem",
                      "DeleteItem", "secretsmanager", "iam:", "lambda:Invoke"):
        assert forbidden not in text, forbidden
    statements = [p for p in policies if "Statement" in p]
    actions = sorted(s["Action"] for p in statements for s in p["Statement"])
    assert actions == ["dynamodb:UpdateItem", "ssm:GetParameter"]
    assert [p for p in policies if "Statement" not in p] == [{"!Ref": "AssistantRuntimePolicy"}]


def test_shared_runtime_policy_reads_only(template):
    """The policy the demo shares with /ask: retrieval, model, guardrail and two key reads, nothing that writes."""
    statements = template["Resources"]["AssistantRuntimePolicy"]["Properties"]["PolicyDocument"]["Statement"]
    text = json.dumps(statements)
    for forbidden in ("ServiceNow", "Sn", "sqs:", "dynamodb:", "s3:Put", "iam:", "secretsmanager"):
        assert forbidden not in text, forbidden


def test_demo_function_only_serves_its_own_route(template):
    events = demo(template)["Properties"]["Events"]
    assert [e["Properties"]["Path"] for e in events.values()] == ["/demo/ask"]


def test_demo_runs_simulated_dry_runs(template):
    env = demo(template)["Properties"]["Environment"]["Variables"]
    assert env["DEMO_AGENT"] == "true" and env["DEMO_ENTERPRISE"] == "true"
