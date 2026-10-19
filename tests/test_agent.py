"""Ticket agent: decisions, code-enforced limits, ServiceNow writes, caller confirmation, demo dry run."""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app  # noqa: E402
from agent import (AGENT_HEADER, DryRunDesk, ServiceNowDesk, TicketAgent, agent_plan,  # noqa: E402
                   handle_caller_reply, latest_caller_reply)
from factory import build_assistant, build_guardrail  # noqa: E402
from llm import AnthropicLLM, FakeLLM, _to_converse  # noqa: E402

SYS_ID = "a" * 32


def answered(source="vpn-troubleshooting.md", title="VPN"):
    return {"status": "answered", "answer": "Open **vpn.example.org** and sign in [1].",
            "sources": [{"n": 1, "title": title, "source": source, "score": 0.9, "snippet": "x"}],
            "guardrail": {"output": {"scores": {"GROUNDING": 0.9}}}, "usage": {"input_tokens": 100, "output_tokens": 20},
            "provider": "fake", "question": "q"}


class Assistant:
    """Stands in for the RAG assistant: returns a fixed result for every search."""

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.llm = FakeLLM()
        self.guardrail = build_guardrail({})

    def ask(self, question, history=None, time_left=None):
        self.calls.append(question)
        return {**self.result, "question": question}


class Scripted:
    """An LLM that plays back tool calls, to test what the code allows whatever the model asks for."""
    provider = "scripted"

    def __init__(self, *calls):
        self.calls = list(calls)
        self.seen = 0
        self.last_usage = {"input_tokens": 10, "output_tokens": 5}

    def chat(self, system, messages, tools, max_tokens=700):
        self.seen += 1
        if not self.calls:
            return {"content": [{"type": "text", "text": "I am not sure."}], "stop_reason": "end_turn"}
        name, args = self.calls.pop(0)
        return {"content": [{"type": "tool_use", "id": f"t{self.seen}", "name": name, "input": args}],
                "stop_reason": "tool_use"}


def incident(priority="4 - Low", text="VPN keeps disconnecting"):
    return {"sys_id": SYS_ID, "number": "INC1", "short_description": text, "priority": priority}


def run(llm, result, priority="4 - Low", recent=None, text="VPN keeps disconnecting"):
    desk = DryRunDesk(recent)
    out = TicketAgent(llm, Assistant(result), desk).run(incident(priority, text), text)
    return out, desk


# ---------------------------------------------------------------- decisions
def test_answered_low_priority_sends_fix_and_waits_for_caller():
    out, desk = run(FakeLLM(), answered())
    assert out["action"] == "awaiting_caller" and len(desk.actions) == 1
    action = desk.actions[0]
    assert action["action"] == "send_fix" and "vpn.example.org" in action["comment"]
    assert "**" not in action["comment"] and "did this fix it?" in action["comment"]
    assert action["work_note"].startswith(AGENT_HEADER) and "Not resolved yet" in action["work_note"]
    assert [s["tool"] for s in out["steps"]] == ["search_knowledge_base", "find_similar_incidents", "send_fix_to_caller"]


def test_no_answer_escalates_to_a_team_with_a_handoff_note():
    out, desk = run(FakeLLM(), {"status": "no_answer", "answer": "", "sources": []}, text="My monitor shows no display")
    assert out["action"] == "escalated" and out["team"] == "Hardware"
    note = desk.actions[0]["work_note"]
    assert "Action: escalated to Hardware" in note and "What the AI tried:" in note and "no article covers it" in note


def test_send_fix_is_refused_for_urgent_incidents():
    llm = Scripted(("send_fix_to_caller", {"search_id": "kb1"}),
                   ("escalate_incident", {"team": "Network", "reason": "Urgent."}))
    out, desk = run(llm, answered(), priority="2 - High")
    assert out["action"] == "escalated" and [a["action"] for a in desk.actions] == ["escalate"]
    assert any("refused" in s["detail"] for s in out["steps"])


def test_send_fix_needs_a_verified_answer_not_model_text():
    llm = Scripted(("send_fix_to_caller", {"search_id": "kb1"}), ("send_fix_to_caller", {"search_id": "made-up"}))
    out, desk = run(llm, {"status": "no_answer", "answer": "", "sources": []})
    assert out["action"] == "escalated" and out["team"] == "Service Desk"  # forced: no decision reached
    assert [a["action"] for a in desk.actions] == ["escalate"]


def test_outage_blocks_self_service_and_raises_priority():
    recent = [{"number": "INC7", "short_description": "VPN disconnecting all morning"},
              {"number": "INC8", "short_description": "VPN keeps dropping and disconnecting"}]
    llm = Scripted(("find_similar_incidents", {}), ("send_fix_to_caller", {"search_id": "kb1"}),
                   ("escalate_incident", {"team": "Network", "reason": "Several users."}))
    out, desk = run(llm, answered(), recent=recent)
    assert out["action"] == "escalated" and out["priority"] == 2 and out["related"] == ["INC7", "INC8"]
    assert desk.actions[0]["priority"] == 2 and "INC7, INC8" in desk.actions[0]["work_note"]


def test_priority_only_goes_up_and_never_past_2():
    out, desk = run(Scripted(("escalate_incident", {"team": "Security", "priority": "1", "reason": "x"})),
                    answered(), priority="4 - Low")
    assert out["priority"] == 2 and desk.actions[0]["priority"] == 2
    out, desk = run(Scripted(("escalate_incident", {"team": "Security", "priority": "3", "reason": "x"})),
                    answered(), priority="2 - High")
    assert out["priority"] == 2 and desk.actions[0]["priority"] is None  # unchanged, never lowered


def test_unknown_team_goes_to_service_desk():
    out, _ = run(Scripted(("escalate_incident", {"team": "Facilities", "reason": "x"})), answered())
    assert out["team"] == "Service Desk"


def test_blocked_incident_never_reaches_the_agent_model():
    llm = Scripted(("send_fix_to_caller", {"search_id": "kb1"}))
    out, desk = run(llm, {"status": "blocked", "answer": "", "sources": []})
    assert llm.seen == 0 and out["action"] == "escalated" and out["forced"] is True
    assert "guardrail" in desk.actions[0]["work_note"]


def test_model_that_never_decides_is_escalated():
    out, desk = run(Scripted(), answered())
    assert out["action"] == "escalated" and out["forced"] is True and len(desk.actions) == 1


INDEX = os.path.join(os.path.dirname(__file__), "..", "data", "index_hash.json")


def test_search_limit_and_full_pipeline_with_real_assistant():
    if not os.path.exists(INDEX):  # CI builds it first: python scripts/build_index.py --embedder hash
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "build_index.py"),
                        "--embedder", "hash"], check=True, capture_output=True)
    assistant = build_assistant({})
    desk = DryRunDesk()
    out = TicketAgent(FakeLLM(), assistant, desk).run(incident(text="How do I set up MFA on a new phone?"),
                                                      "How do I set up MFA on a new phone?")
    assert out["action"] == "awaiting_caller" and out["kb"]["status"] == "answered"
    llm = Scripted(*[("search_knowledge_base", {"query": f"q{i}"}) for i in range(5)])
    out, _ = run(llm, answered())
    assert sum(1 for s in out["steps"] if s["tool"] == "search_knowledge_base") == 3


# ---------------------------------------------------------------- caller confirmation
def fixed_incident(reply, state="On Hold", assigned=""):
    out, desk = run(FakeLLM(), answered())
    comment = desk.actions[0]["comment"]
    journal = f"2026-10-19 10:00:00 - KB Assistant (Additional comments)\n{comment}\n\n"
    if reply:
        journal = f"2026-10-19 10:05:00 - Abel Tuter (Additional comments)\n{reply}\n\n" + journal
    return {"sys_id": SYS_ID, "number": "INC1", "state": state, "assigned_to": assigned, "caller_id": "Abel Tuter",
            "work_notes": desk.actions[0]["work_note"], "comments": journal}


@pytest.mark.parametrize("reply,outcome,action", [
    ("Yes that worked, thanks!", "resolved", "resolve"),
    ("No, it still won't connect", "back_to_technician", "back_to_technician"),
    ("Which app do I open?", "back_to_technician", "back_to_technician")])
def test_only_a_confirmed_fix_is_resolved(reply, outcome, action):
    desk = DryRunDesk()
    result = handle_caller_reply(FakeLLM(), build_guardrail({}), desk, fixed_incident(reply))
    assert result["outcome"] == outcome and desk.actions[0]["action"] == action
    assert "Caller reply at 2026-10-19 10:05:00" in desk.actions[0]["work_note"]


def test_reply_handling_skips_when_nothing_to_do():
    guard = build_guardrail({})
    assert handle_caller_reply(FakeLLM(), guard, DryRunDesk(), fixed_incident(""))["outcome"] == "no_reply"
    assert handle_caller_reply(FakeLLM(), guard, DryRunDesk(), fixed_incident("ok", state="Resolved"))["outcome"] == "not_waiting"
    assert handle_caller_reply(FakeLLM(), guard, DryRunDesk(), fixed_incident("ok", assigned="Beth"))["outcome"] == "technician_assigned"
    inc = fixed_incident("Yes it worked")
    inc["work_notes"] += "\nCaller reply at 2026-10-19 10:05:00: yes"
    assert handle_caller_reply(FakeLLM(), guard, DryRunDesk(), inc)["outcome"] == "already_handled"
    assert handle_caller_reply(FakeLLM(), guard, DryRunDesk(), {"work_notes": "", "state": "On Hold"})["outcome"] == "not_agent_fix"


def test_latest_caller_reply_parses_the_journal():
    journal = ("2026-10-19 10:05:00 - Abel Tuter (Additional comments)\nIt works\nnow\n\n"
               "2026-10-19 10:00:00 - KB Assistant (Additional comments)\nHi, this is the IT help desk's AI assistant...")
    assert latest_caller_reply(journal) == {"at": "2026-10-19 10:05:00", "author": "Abel Tuter", "text": "It works\nnow"}


# ---------------------------------------------------------------- ServiceNow writes
class Client:
    def __init__(self, groups=("Service Desk", "Network")):
        self.groups = groups
        self.updates = []

    def query(self, table, query="", fields="", limit=100, order_by=""):
        if table == "sys_user_group":
            name = query.split("=", 1)[1]
            return [{"sys_id": f"g-{name}"}] if name in self.groups else []
        return [{"number": "INC9", "short_description": "x"}]

    def update(self, table, sys_id, record, display_values=False):
        self.updates.append(record)


def test_servicenow_desk_records():
    client = Client()
    desk = ServiceNowDesk(client)
    desk.send_fix(incident(), comment="c", work_note="w", team="Service Desk")
    assert client.updates[-1] == {"state": "3", "hold_reason": "1", "comments": "c", "work_notes": "w",
                                  "assignment_group": "g-Service Desk"}
    desk.escalate(incident(), team="Hardware", priority=2, work_note="w")  # no Hardware group -> Service Desk
    assert client.updates[-1] == {"work_notes": "w", "assignment_group": "g-Service Desk", "impact": "2", "urgency": "1"}
    desk.confirm_resolved(incident(), "w", "notes")
    assert client.updates[-1]["state"] == "6" and client.updates[-1]["close_code"] == "Solution provided"
    desk.back_to_technician(incident(), "Network", "w")
    assert client.updates[-1] == {"state": "2", "hold_reason": "", "work_notes": "w", "assignment_group": "g-Network"}


class SN:
    def __init__(self, inc):
        self.inc = inc
        self.updates = []

    def get_incident(self, sys_id):
        return self.inc

    def query(self, *a, **k):
        return []

    def update(self, table, sys_id, record, display_values=False):
        self.updates.append(record)


def test_worker_agent_mode_end_to_end(monkeypatch):
    monkeypatch.setenv("SN_MODE", "agent")
    app._cache.clear()
    sn = SN({"number": "INC1", "short_description": "VPN fails", "priority": "4 - Low", "active": "true"})
    app._cache.update({"servicenow": sn, "assistant": Assistant(answered())})
    out = app.servicenow_worker({"Records": [{"messageId": "m1", "body": json.dumps({"sys_id": SYS_ID}),
                                              "attributes": {"ApproximateReceiveCount": "1"}}]}, None)
    assert out == {"batchItemFailures": []} and sn.updates[-1]["state"] == "3"
    # already worked: a repeat event never acts twice
    sn.inc["work_notes"] = sn.updates[-1]["work_notes"]
    summary = app.work_incident(sn, app._cache["assistant"], SYS_ID)
    assert summary["outcome"] == "already_handled" and len(sn.updates) == 1
    app._cache.clear()


def test_worker_retries_when_the_model_is_down_then_escalates(monkeypatch):
    monkeypatch.setenv("SN_MODE", "agent")

    class Down(FakeLLM):
        def chat(self, *a, **k):
            raise TimeoutError("model down")
    assistant = Assistant(answered())
    assistant.llm = Down()
    sn = SN({"number": "INC1", "short_description": "VPN fails", "priority": "4 - Low"})
    app._cache.clear()
    app._cache.update({"servicenow": sn, "assistant": assistant})
    rec = {"messageId": "m1", "body": json.dumps({"sys_id": SYS_ID}), "attributes": {"ApproximateReceiveCount": "1"}}
    assert app.servicenow_worker({"Records": [rec]}, None) == {"batchItemFailures": [{"itemIdentifier": "m1"}]}
    assert sn.updates == []
    rec["attributes"]["ApproximateReceiveCount"] = "3"
    assert app.servicenow_worker({"Records": [rec]}, None) == {"batchItemFailures": []}
    assert "escalated to Service Desk" in sn.updates[-1]["work_notes"]
    app._cache.clear()


def test_agent_plan_is_safe_to_show_publicly():
    out, _ = run(FakeLLM(), answered())
    plan = agent_plan(out)
    assert set(plan) == {"action", "team", "priority", "steps", "dry_run"} and plan["dry_run"] is True
    assert "kb" not in plan and "usage" not in plan


# ---------------------------------------------------------------- model layer
def test_converse_conversion_round_trip():
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {"a": 1}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}]
    out = _to_converse(messages)
    assert out[0]["content"] == [{"text": "hi"}]
    assert out[1]["content"][0]["toolUse"] == {"toolUseId": "t1", "name": "x", "input": {"a": 1}}
    assert out[2]["content"][0]["toolResult"]["toolUseId"] == "t1"


def test_anthropic_chat_sends_tools_and_returns_tool_calls():
    sent = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(request, timeout):
        sent.update(json.loads(request.data))
        return Response(json.dumps({"content": [{"type": "tool_use", "id": "t1", "name": "find_similar_incidents",
                                                 "input": {}}], "stop_reason": "tool_use",
                                    "usage": {"input_tokens": 5, "output_tokens": 3}}).encode())
    llm = AnthropicLLM("claude-haiku-4-5", "k", opener=opener)
    reply = llm.chat("sys", [{"role": "user", "content": "x"}], [{"name": "find_similar_incidents"}])
    assert sent["tools"] == [{"name": "find_similar_incidents"}] and reply["stop_reason"] == "tool_use"
    assert reply["content"][0]["name"] == "find_similar_incidents" and llm.last_usage["input_tokens"] == 5


# ---------------------------------------------------------------- webhook + public demo
def test_caller_reply_events_are_queued_once_per_version(monkeypatch):
    from test_servicenow import FakeDynamo, FakeSQS, KEY, signed_event
    import time
    app._cache.clear()
    app._cache.update({"sn_webhook_secret": KEY, "sn_webhook_secret_at": time.time(),
                       "dynamodb": FakeDynamo(), "sqs": FakeSQS()})
    for name, value in (("SN_IDEMPOTENCY_TABLE", "t"), ("SN_QUEUE_URL", "q"), ("SN_WEBHOOK_SECRET_PARAM", "/p")):
        monkeypatch.setenv(name, value)
    reply = {"sys_id": SYS_ID, "number": "INC0010001", "event": "caller_reply", "version": "7"}
    assert app.servicenow_event(signed_event({"sys_id": SYS_ID, "number": "INC0010001"}), None)["statusCode"] == 202
    assert app.servicenow_event(signed_event(reply), None)["statusCode"] == 202
    assert app.servicenow_event(signed_event(reply), None)["statusCode"] == 200  # duplicate
    assert app.servicenow_event(signed_event({**reply, "version": "8"}), None)["statusCode"] == 202
    assert app.servicenow_event(signed_event({**reply, "version": "x"}), None)["statusCode"] == 400
    assert app.servicenow_event(signed_event({**reply, "event": "delete"}), None)["statusCode"] == 400
    assert [m["event"] for m in app._cache["sqs"].sent] == ["new", "caller_reply", "caller_reply"]
    app._cache.clear()


def test_demo_shows_the_agent_plan_without_writing(monkeypatch):
    from test_demo import FakeCounters
    app._cache.clear()
    app._cache.update({"demo_api_key": "k", "dynamodb": FakeCounters(), "assistant": Assistant(answered())})
    monkeypatch.setenv("DEMO_TABLE", "t")
    out = app.demo_ask({"headers": {"x-api-key": "k", "x-demo-client": "1.2.3.4"},
                        "body": json.dumps({"question": "VPN keeps disconnecting"})}, None)
    body = json.loads(out["body"])
    assert out["statusCode"] == 200 and body["agent"]["action"] == "awaiting_caller" and body["agent"]["dry_run"]
    assert body["status"] == "answered" and "sn" not in app._cache and "servicenow" not in app._cache
    app._cache.clear()


def test_review_fixes_caller_identity_quotes_and_only_kb1():
    guard = build_guardrail({})
    inc = fixed_incident("Yes that worked, thanks!")
    inc["comments"] = inc["comments"].replace("Abel Tuter", "Beth Tech")  # a technician, not the caller
    desk = DryRunDesk()
    assert handle_caller_reply(FakeLLM(), guard, desk, inc)["verdict"] == "not_caller"
    assert desk.actions[0]["action"] == "back_to_technician"
    # quoted email text is cut off: "No" survives, the quoted "fixed" text doesn't
    reply = latest_caller_reply("2026-10-19 10:05:00 - Abel Tuter (Additional comments)\nNo luck\n"
                                "On Mon, IT wrote:\n> this should fix it, thanks\n")
    assert reply["text"] == "No luck"
    # only one reply is ever handled
    inc = fixed_incident("Yes it worked")
    inc["work_notes"] = "Caller reply at 2026-10-19 09:00:00 from Abel Tuter: no\n" + inc["work_notes"]
    assert handle_caller_reply(FakeLLM(), guard, DryRunDesk(), inc)["outcome"] == "already_handled"
    # a fix from a model-chosen query (kb2) is refused; only kb1 can reach the caller
    llm = Scripted(("search_knowledge_base", {"query": "something else"}), ("send_fix_to_caller", {"search_id": "kb2"}))
    out, desk = run(llm, answered())
    assert any("only kb1" in s["detail"] for s in out["steps"]) and desk.actions[0]["action"] == "escalate"


def test_reply_classifier_failure_retries_then_hands_back():
    class Broken(FakeLLM):
        def complete(self, *a, **k):
            raise TimeoutError("down")
    from agent import AgentUnavailable
    with pytest.raises(AgentUnavailable):
        handle_caller_reply(Broken(), build_guardrail({}), DryRunDesk(), fixed_incident("yes"), final_attempt=False)
    desk = DryRunDesk()
    assert handle_caller_reply(Broken(), build_guardrail({}), desk, fixed_incident("yes"))["outcome"] == "back_to_technician"


def test_out_of_time_raises_for_retry_instead_of_guessing():
    from agent import AgentUnavailable
    agent = TicketAgent(FakeLLM(), Assistant(answered()), DryRunDesk(), min_step_seconds=15)
    with pytest.raises(AgentUnavailable):
        agent.run(incident(), "VPN", time_left=lambda: 5, raise_on_model_error=True)
