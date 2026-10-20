"""The Streamlit page, driven headless with streamlit.testing (offline: no API URL, no keys)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "app.py")
NAV = ["AI ticket desk", "Ask the assistant", "12 capabilities", "ServiceNow, live", "Architecture", "Evaluation"]


@pytest.fixture
def app(monkeypatch):
    for name in ("ASSISTANT_API_URL", "ASSISTANT_API_KEY", "PUBLIC_DEMO", "PORTFOLIO_URL", "LINKEDIN_URL"):
        monkeypatch.delenv(name, raising=False)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def html(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def button(at, key):
    found = [b for b in at.button if b.key == key]
    assert found, f"no button {key}: {[b.key for b in at.button]}"
    return found[0]


def show(at, section):
    at.session_state["nav_target"] = section
    at.run()
    assert not at.exception, at.exception


def test_every_section_renders(app):
    for section in NAV:
        show(app, section)
        assert html(app)


def test_simulated_banner_and_no_credentials_claim(app):
    page = html(app)
    assert "SIMULATED" in page
    assert "no ServiceNow credentials" in page


def test_sample_ticket_resolves_and_shows_the_work(app):
    button(app, "sample-lockout").click().run()
    assert not app.exception
    page = html(app)
    assert "Resolved" in page and "unlock_account" in page and "SIMULATED" in page
    assert "demo ticket, not ServiceNow" in page


def test_caller_and_fast_forward_buttons(app):
    button(app, "sample-vpn").click().run()
    button(app, "act-ff").click().run()
    assert "reminder 1 of 2" in html(app)
    button(app, "act-worked").click().run()
    assert "The caller confirmed the fix" in html(app)


def test_approval_buttons(app):
    button(app, "sample-access").click().run()
    button(app, "act-appr-req").click().run()
    assert "approve this request" in html(app)
    button(app, "act-appr-ok").click().run()
    assert "resolved" in html(app).lower()
    assert app.session_state["ticket"].sn.rec["state"] == "6"


def test_duplicate_buttons(app):
    button(app, "sample-lockout").click().run()
    button(app, "act-dup").click().run()
    assert "Duplicate webhook event ignored" in html(app)
    button(app, "act-dupq").click().run()
    assert "never worked twice" in html(app)


def test_live_incident_needs_the_api(app):
    submit = [b for b in app.button if "live AI agent" in str(b.label)]
    assert submit and submit[0].disabled
    assert "sample incidents work offline" in html(app)


def test_chat_example_answers_with_sources(app):
    show(app, "Ask the assistant")
    button(app, "ex-1").click().run()
    app.run()
    assert not app.exception
    assert app.session_state["turns"], "the example question was asked"
    assert app.session_state["turns"][0]["answer"]


def test_capabilities_list_all_twelve(app):
    show(app, "12 capabilities")
    page = html(app)
    assert all(f"Workflow {n}" in page or f"WORKFLOW {n}" in page.upper() for n in range(1, 13))


def test_html_never_contains_raw_newlines_in_activity(app):
    """Multi-line notes must not break out of the HTML block (st.markdown would turn them into stray lists)."""
    button(app, "sample-lockout").click().run()
    activity = [m.value for m in app.markdown if 'class="kb-act"' in m.value]
    assert activity and all("\n" not in v for v in activity)


def test_header_links(app):
    page = html(app)
    assert 'href="https://www.linkedin.com/in/sagarmodha/"' in page
    assert 'href="https://github.com/sagarmodha9-dotcom/it-kb-assistant"' in page
    assert 'href="https://github.com/sagarmodha9-dotcom"' in page
    assert ">Portfolio<" not in page, "no Portfolio button without a portfolio site"
