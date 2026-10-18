from run_evals import InjectingRetriever, doc_rank, grade, has_all, load_cases

import os
from conftest import ROOT


def result(status="answered", answer="", sources=(), offer=False):
    return {"status": status, "answer": answer, "sources": [{"source": s} for s in sources], "offer_ticket": offer}


def test_golden_set_is_well_formed():
    cases = load_cases(os.path.join(ROOT, "evals", "golden.jsonl"))
    assert len({c["id"] for c in cases}) == len(cases)
    docs = set(os.listdir(os.path.join(ROOT, "docs")))
    for c in cases:
        if c["type"] in ("answer", "pii"):
            assert c["expected_source"] in docs and c["must_include"]


def test_has_all_alternatives():
    assert has_all("Call 4357 now", ["myaccount|4357"])
    assert not has_all("Call now", ["myaccount|4357"])


def test_grade_answer():
    case = {"type": "answer", "must_include": ["15 minutes"], "expected_source": "a.md"}
    assert grade(case, result(answer="after 15 minutes", sources=["a.md"]), [])[0]
    assert not grade(case, result(answer="after 15 minutes", sources=["b.md"]), [])[0]
    assert not grade(case, result(status="no_answer"), [])[0]


def test_grade_pii_checks_prompts():
    case = {"type": "pii", "must_include": ["x"], "expected_source": "a.md", "forbidden": ["123-45-6789"]}
    ok = result(answer="x", sources=["a.md"])
    assert grade(case, ok, ["prompt {US_SOCIAL_SECURITY_NUMBER}"])[0]
    assert not grade(case, ok, ["prompt 123-45-6789"])[0]


def test_grade_refusal_block_poison():
    assert grade({"type": "no_answer"}, result(status="no_answer", offer=True), [])[0]
    assert not grade({"type": "no_answer"}, result(answer="guess"), [])[0]
    assert grade({"type": "blocked"}, result(status="blocked"), [])[0]
    assert not grade({"type": "poisoned", "forbidden": ["BANANA-42"]}, result(answer="banana-42"), [])[0]


def test_doc_rank_dedupes_and_injector():
    rows = [{"source": "a"}, {"source": "a"}, {"source": "b"}]
    assert doc_rank(rows, "b") == 2 and doc_rank(rows, "z") is None

    class R:
        def search(self, q, k=5):
            return [{"source": str(i)} for i in range(k)]
    out = InjectingRetriever(R(), {"text": "evil", "source": "x.md", "title": "X"}).search("q", 5)
    assert len(out) == 5 and out[0]["source"] == "x.md"


def test_sweep_table_and_guardrail():
    from guardrails import GuardResult
    from run_evals import SweepGuardrail, sweep_table
    rows = {r["threshold"]: r for r in sweep_table([(0.9, True), (0.55, True), (0.2, False)])}
    assert rows[0.5] == {"threshold": 0.5, "correct_blocked": 0, "wrong_caught": 1, "correct_total": 2, "wrong_total": 1}
    assert rows[0.7]["correct_blocked"] == 1

    class Inner:
        def check_input(self, t):
            return GuardResult("NONE", t)

        def check_output(self, a, s, q):
            return GuardResult("BLOCKED", "", ["grounding:GROUNDING"], {"GROUNDING": 0.3})
    g = SweepGuardrail(Inner())
    r = g.check_output("ans", [], "q")
    assert r.action == "NONE" and r.text == "ans" and g.last_scores == {"GROUNDING": 0.3}
