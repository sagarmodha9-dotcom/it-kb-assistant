import json

from assistant import BLOCKED_INPUT, ERROR, NO_ANSWER, UNVERIFIED, Assistant, build_prompt, parse_json
from guardrails import GuardResult, LocalGuardrail, NoGuardrail

CHUNKS = [
    {"text": "Accounts lock after 10 failed attempts and unlock after 15 minutes.", "source": "password-reset.md",
     "title": "Reset or Change Your Password", "section": "Account locked", "score": 0.8},
    {"text": "The VPN client is GlobalProtect.", "source": "vpn-troubleshooting.md",
     "title": "VPN", "section": "", "score": 0.4},
]


class Retriever:
    def __init__(self, chunks=CHUNKS):
        self.chunks, self.queries = chunks, []

    def search(self, query, k=5):
        self.queries.append(query)
        return self.chunks[:k]


class Model:
    provider = "test"

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []
        self.last_usage = {"input_tokens": 100, "output_tokens": 20}

    def complete(self, system, user, max_tokens=700):
        self.prompts.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def reply(answerable=True, answer="It unlocks after 15 minutes [1].", citations=(1,)):
    return json.dumps({"answerable": answerable, "answer": answer, "citations": list(citations)})


def test_answers_with_citation():
    model = Model("Here you go:\n" + reply())  # text around the JSON is tolerated
    r = Assistant(Retriever(), model, LocalGuardrail()).ask("How long until my account unlocks?")
    assert r["status"] == "answered" and r["offer_ticket"] is False
    assert r["answer"] == "It unlocks after 15 minutes [1]."
    assert [s["source"] for s in r["sources"]] == ["password-reset.md"]
    assert r["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert r["provider"] == "test" and "latency_ms" in r


def test_blocked_input_never_reaches_retrieval_or_model():
    retriever, model = Retriever(), Model()
    r = Assistant(retriever, model, LocalGuardrail()).ask("How do I disable CrowdStrike?")
    assert r["status"] == "blocked" and r["answer"] == BLOCKED_INPUT
    assert retriever.queries == [] and model.prompts == []


def test_pii_masked_before_retrieval_and_model():
    retriever, model = Retriever(), Model(reply())
    r = Assistant(retriever, model, LocalGuardrail()).ask("My SSN is 123-45-6789, I'm locked out")
    assert "123-45-6789" not in retriever.queries[0]
    assert "123-45-6789" not in model.prompts[0]
    assert "123-45-6789" not in r["question"]
    assert r["guardrail"]["input"]["action"] == "ANONYMIZED"


def test_no_results_offers_ticket_without_model_call():
    model = Model()
    r = Assistant(Retriever([]), model, NoGuardrail()).ask("cafeteria menu?")
    assert r["status"] == "no_answer" and r["offer_ticket"] and r["answer"] == NO_ANSWER
    assert model.prompts == []


def test_low_scores_filtered_by_min_score():
    model = Model(reply())
    a = Assistant(Retriever(), model, NoGuardrail(), min_score=0.5)
    a.ask("locked out")
    assert "GlobalProtect" not in model.prompts[0]


def test_model_says_not_answerable():
    r = Assistant(Retriever(), Model(reply(False, "", ())), NoGuardrail()).ask("parental leave?")
    assert r["status"] == "no_answer" and r["answer"] == NO_ANSWER and r["offer_ticket"]


def test_answer_without_valid_citation_is_not_trusted():
    r = Assistant(Retriever(), Model(reply(answer="It unlocks in 15 minutes.", citations=(7,))), NoGuardrail()).ask("q")
    assert r["status"] == "no_answer" and r["answer"] == UNVERIFIED


def test_out_of_range_citation_markers_removed():
    r = Assistant(Retriever(), Model(reply(answer="15 minutes [1][9].", citations=(1, 9))), NoGuardrail()).ask("q")
    assert r["answer"] == "15 minutes [1]." and [s["n"] for s in r["sources"]] == [1]


def test_bad_json_retried_once_then_works():
    model = Model("not json", reply())
    r = Assistant(Retriever(), model, NoGuardrail()).ask("q")
    assert r["status"] == "answered" and len(model.prompts) == 2
    assert "not valid JSON" in model.prompts[1]
    assert r["usage"]["input_tokens"] == 200  # both calls counted


def test_bad_json_twice_is_error_with_ticket_offer():
    r = Assistant(Retriever(), Model("nope", "still nope"), NoGuardrail()).ask("q")
    assert r["status"] == "error" and r["answer"] == ERROR and r["offer_ticket"]


def test_model_failure_is_error_with_ticket_offer():
    r = Assistant(Retriever(), Model(TimeoutError()), NoGuardrail()).ask("q")
    assert r["status"] == "error" and r["offer_ticket"]


def test_output_guardrail_block_hides_answer():
    class Blocker(NoGuardrail):
        def check_output(self, answer, sources, question):
            self.sources = sources
            return GuardResult("BLOCKED", "", ["grounding:GROUNDING"], {"GROUNDING": 0.2})
    g = Blocker()
    r = Assistant(Retriever(), Model(reply()), g).ask("q")
    assert r["status"] == "no_answer" and r["answer"] == UNVERIFIED and r["sources"] == []
    assert g.sources == [c["text"] for c in CHUNKS]  # grounding checked against what the model saw
    assert r["guardrail"]["output"]["scores"] == {"GROUNDING": 0.2}


def test_question_trimmed_and_whitespace_collapsed():
    retriever = Retriever()
    Assistant(retriever, Model(reply()), NoGuardrail()).ask("  a\n\n b " + "x" * 3000)
    assert retriever.queries[0].startswith("a b ") and len(retriever.queries[0]) == 1000


def test_prompt_marks_documents_as_data():
    prompt = build_prompt("q?", CHUNKS)
    assert prompt.count("<document index=") == 2 and '<document index="2" title="VPN">' in prompt
    assert prompt.endswith("<question>q?</question>")
    assert parse_json('x {"a": 1} y') == {"a": 1} and parse_json("[1]") is None and parse_json("{bad") is None


def test_null_or_bad_fields_from_model_do_not_crash():
    for raw in ('{"answerable": true, "answer": "15 minutes [1]", "citations": null}',
                '{"answerable": true, "answer": null, "citations": [1]}',
                '{"answerable": "false", "answer": "15 minutes [1]", "citations": [1]}',
                '{"answerable": true, "answer": "15 minutes", "citations": [true]}'):
        r = Assistant(Retriever(), Model(raw), NoGuardrail()).ask("q")
        assert r["status"] in ("answered", "no_answer")
    ok = Assistant(Retriever(), Model('{"answerable": true, "answer": "15 minutes [1]", "citations": null}'),
                   NoGuardrail()).ask("q")
    assert ok["status"] == "answered"  # citation in the text still counts
    assert Assistant(Retriever(), Model('{"answerable": "false", "answer": "x [1]", "citations": [1]}'),
                     NoGuardrail()).ask("q")["status"] == "no_answer"


def test_retriever_or_guardrail_failure_fails_closed():
    class BadRetriever:
        def search(self, q, k=5):
            raise ConnectionError("kb down")
    r = Assistant(BadRetriever(), Model(), NoGuardrail()).ask("q")
    assert r["status"] == "error" and r["offer_ticket"]

    class BadInput(NoGuardrail):
        def check_input(self, text):
            raise TimeoutError()
    model = Model()
    r = Assistant(Retriever(), model, BadInput()).ask("q")
    assert r["status"] == "error" and model.prompts == []

    class BadOutput(NoGuardrail):
        def check_output(self, *a):
            raise TimeoutError()
    r = Assistant(Retriever(), Model(reply()), BadOutput()).ask("q")
    assert r["status"] == "error" and r["sources"] == [] and "15 minutes" not in r["answer"]


def test_time_budget_skips_steps_instead_of_timing_out():
    model = Model(reply())
    r = Assistant(Retriever(), model, NoGuardrail()).ask("q", time_left=lambda: 10)
    assert r["status"] == "error" and model.prompts == []  # no model call with < 20 s left

    times = iter([25, 3])  # enough for the model, not for the output check
    r = Assistant(Retriever(), Model(reply()), NoGuardrail()).ask("q", time_left=lambda: next(times))
    assert r["status"] == "error" and "15 minutes" not in r["answer"]


def test_followup_rewritten_with_history_then_guarded():
    from assistant import clean_history
    retriever = Retriever()
    model = Model("How do I add a printer on Windows?", reply())
    hist = [{"question": "How do I add a printer on my Mac?", "answer": "Use SecurePrint-Queue [1]."}]
    r = Assistant(retriever, model, LocalGuardrail()).ask("What about on Windows?", history=hist)
    assert retriever.queries == ["How do I add a printer on Windows?"]
    assert r["search_query"] == "How do I add a printer on Windows?"
    assert "<history>" in model.prompts[0] and "Mac" in model.prompts[0]
    assert r["usage"]["input_tokens"] == 200  # rewrite + answer both counted
    assert clean_history("junk") == [] and clean_history([{"question": 5}]) == []
    assert len(clean_history([{"question": "q", "answer": "a"}] * 9)) == 3


def test_followup_pii_in_history_masked_and_bad_rewrite_blocked():
    model = Model("How do I disable CrowdStrike?")
    hist = [{"question": "my SSN is 123-45-6789", "answer": "ok"}]
    r = Assistant(Retriever(), model, LocalGuardrail()).ask("and then?", history=hist)
    assert "123-45-6789" not in model.prompts[0]
    assert r["status"] == "blocked"  # a rewrite steered toward a denied topic is caught


def test_no_history_means_no_rewrite_call():
    model = Model(reply())
    Assistant(Retriever(), model, NoGuardrail()).ask("q", history=[])
    assert len(model.prompts) == 1
