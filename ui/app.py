"""Web chat for the IT Knowledge Base Assistant (Streamlit).

  streamlit run ui/app.py

Three modes:
  Deployed:    set ASSISTANT_API_URL and ASSISTANT_API_KEY -> calls the AWS API (POST /ask, /escalate)
  Public demo: also set PUBLIC_DEMO=true and use the demo key -> calls POST /demo/ask only. No ticket
               button, the API URL isn't shown, questions are limited to 300 characters, and the AWS
               side enforces daily and per-visitor caps. Settings can come from Streamlit secrets.
  Local:       otherwise runs the assistant in this process, configured by the same environment
               variables as src/factory.py (offline by default: hash embeddings + fake model)
"""
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from factory import build_assistant, build_guardrail, setting  # noqa: E402
from lifecycle import CHOICES, play  # noqa: E402
from tickets import TicketClient, TicketError  # noqa: E402


def config(name: str, default: str = "") -> str:
    """Environment variable, else Streamlit secret (Streamlit Community Cloud keeps secrets server-side)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets file
        return default


API_URL = config("ASSISTANT_API_URL").rstrip("/")
API_KEY = config("ASSISTANT_API_KEY")
PUBLIC_DEMO = config("PUBLIC_DEMO").lower() in ("1", "true", "yes")
MAX_QUESTION = 300 if PUBLIC_DEMO else 1000
REPO = "https://github.com/sagarmodha9-dotcom/it-kb-assistant"
REPO_DOCS = f"{REPO}/blob/main/docs/"
EXAMPLES = ["How do I set up MFA on my new phone?",
            "I'm locked out of my account",
            "I need access to the Finance Reports SharePoint site",
            "What's on the cafeteria menu today?"]
# (workflow, real or simulated in this project)
AGENT_WORKFLOWS = [("Password reset and account unlock", "Simulated"), ("Intune device diagnostics", "Simulated"),
                   ("Microsoft 365 license management", "Simulated"), ("Group membership and access", "Simulated"),
                   ("VPN troubleshooting", "KB real, monitoring simulated"), ("Software installation", "Simulated"),
                   ("Incident triage and assignment", "Real (ServiceNow)"),
                   ("Outage detection and coordination", "Real (ServiceNow), monitoring simulated"),
                   ("Log analysis and root cause", "Analysis real, logs simulated"),
                   ("Outlook and Microsoft 365 diagnostics", "Simulated"),
                   ("Onboarding and offboarding", "Simulated"),
                   ("Follow-up and resolution verification", "Real (ServiceNow)")]
STEPS = [("Screen the question", "Prompt attacks and off-topic requests are blocked, and personal "
                                 "details are masked before anything else sees them."),
         ("Search the articles", "Voyage AI embeddings in Amazon S3 Vectors find the passages that "
                                 "best match the question."),
         ("Answer with citations", "Claude writes only from those passages and cites each one. "
                                   "Citations that don't point at a real passage are dropped."),
         ("Check before showing", "Amazon Bedrock Guardrails confirms the answer is grounded in the "
                                  "cited text. If it isn't, you get “not found” instead.")]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');
:root {
  --paper: #F3F5F7; --surface: #FFFFFF; --ink: #15202E; --muted: #5A6677; --line: #DCE2E9;
  --cite: #1E5BB8; --cite-bg: #E8EFFA; --ok: #1C7A4C; --ok-bg: #E7F4EC; --warn: #855700; --warn-bg: #FBF3E2;
  --stop: #9A3324; --stop-bg: #FBEBE8;
}
html, body, p, li, h1, h2, h3, h4, label, button, textarea, input, .stMarkdown {
  font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif !important;
}
[data-testid="stDecoration"], [data-testid="stToolbarActions"] { display: none; }
.block-container { max-width: 820px; padding-top: 2.5rem; }

.kb-head h1 { font-size: 2.1rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1.15;
  color: var(--ink); margin: 0 0 .5rem; padding: 0; }
.kb-head p { color: var(--muted); font-size: 1.02rem; line-height: 1.55; max-width: 62ch; margin: 0; }
.kb-note { margin: 1.1rem 0 1.6rem; padding: .7rem .9rem; border: 1px solid var(--line); border-radius: 8px;
  background: var(--surface); color: var(--muted); font-size: .88rem; line-height: 1.5; }
.kb-note strong { color: var(--ink); font-weight: 600; }
.kb-label { font-size: .95rem; font-weight: 600; color: var(--ink); margin: 1.4rem 0 .6rem; }

[class*="st-key-starter-"] button { justify-content: flex-start; text-align: left; min-height: 3.4rem;
  background: var(--surface); border: 1px solid var(--line); border-radius: 8px; color: var(--ink); }
[class*="st-key-starter-"] button:hover { border-color: var(--cite); color: var(--cite); }
[class*="st-key-starter-"] button > div, [class*="st-key-starter-"] button [data-testid="stMarkdownContainer"] {
  width: 100%; justify-content: flex-start; }
[class*="st-key-starter-"] button p { font-size: .93rem; text-align: left; white-space: normal;
  overflow: visible; text-overflow: clip; }

.kb-steps { list-style: none; counter-reset: step; padding: 0; margin: .2rem 0 1rem;
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem 1.6rem; }
.kb-steps li { counter-increment: step; position: relative; padding-left: 2.1rem; margin: 0;
  color: var(--muted); font-size: .88rem; line-height: 1.5; }
.kb-steps li::before { content: counter(step); position: absolute; left: 0; top: .05rem; width: 1.45rem;
  height: 1.45rem; border-radius: 50%; border: 1.5px solid var(--ink); color: var(--ink);
  font-size: .78rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }
.kb-steps b { display: block; color: var(--ink); font-weight: 600; font-size: .93rem; margin-bottom: .1rem; }
@media (max-width: 640px) { .kb-steps { grid-template-columns: 1fr; } .kb-head h1 { font-size: 1.7rem; } }

[data-testid="stChatMessage"] { background: transparent; padding: .9rem 0; }
[class*="st-key-answer-"] { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 1rem 1.15rem 1.05rem; gap: .6rem; }
[class*="st-key-answer-"] .stMarkdown p, [class*="st-key-answer-"] .stMarkdown li { line-height: 1.6; }
sup.kb-cite { display: inline-flex; align-items: center; justify-content: center; min-width: 1.15rem;
  height: 1.15rem; padding: 0 .25rem; margin: 0 .1rem; border-radius: 4px; background: var(--cite-bg);
  color: var(--cite); font-size: .7rem; font-weight: 600; vertical-align: super; line-height: 1; }

.kb-status { margin-bottom: .35rem; display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .9rem; font-size: .82rem;
  color: var(--muted); }
.kb-badge { display: inline-flex; align-items: center; gap: .35rem; padding: .2rem .55rem; border-radius: 999px;
  font-weight: 600; font-size: .8rem; }
.kb-badge.ok { background: var(--ok-bg); color: var(--ok); }
.kb-badge.warn { background: var(--warn-bg); color: var(--warn); }
.kb-badge.stop { background: var(--stop-bg); color: var(--stop); }

.kb-sources { border-top: 1px solid var(--line); padding-top: .75rem; margin-top: .15rem; }
.kb-sources-title { font-size: .8rem; font-weight: 600; color: var(--muted); margin-bottom: .5rem; }
.kb-source { display: grid; grid-template-columns: 1.6rem 1fr; gap: .1rem .6rem; padding: .45rem 0; }
.kb-source + .kb-source { border-top: 1px dashed var(--line); }
.kb-source .n { grid-row: span 2; align-self: start; margin-top: .1rem; width: 1.4rem; height: 1.4rem;
  border-radius: 4px; background: var(--cite-bg); color: var(--cite); font-size: .75rem; font-weight: 600;
  display: flex; align-items: center; justify-content: center; }
.kb-source a { color: var(--ink); font-weight: 600; font-size: .9rem; text-decoration: none; }
.kb-source a:hover { color: var(--cite); text-decoration: underline; }
.kb-source .file { color: var(--muted); font-size: .78rem; margin-left: .4rem; font-weight: 400; }
.kb-source q { color: var(--muted); font-size: .84rem; line-height: 1.5; quotes: "“" "”"; }

.kb-agent { border-top: 1px solid var(--line); padding-top: .75rem; margin-top: .15rem; }
.kb-agent-head { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem .7rem; margin-bottom: .5rem; }
.kb-agent-title { font-size: .8rem; font-weight: 600; color: var(--muted); }
.kb-agent-decision { font-size: .9rem; font-weight: 600; color: var(--ink); }
.kb-agent ol { margin: 0 0 .4rem; padding-left: 1.2rem; }
.kb-agent li { font-size: .84rem; color: var(--muted); line-height: 1.5; margin: .1rem 0; }
.kb-agent li b { color: var(--ink); font-weight: 500; }
.kb-agent-foot { font-size: .78rem; color: var(--muted); }
.kb-next { border-top: 1px solid var(--line); padding-top: .75rem; margin-top: .15rem; }
.kb-next-title { font-size: .8rem; font-weight: 600; color: var(--muted); margin-bottom: .2rem; }
.kb-next-hint { font-size: .8rem; color: var(--muted); margin-bottom: .4rem; }
[class*="st-key-next-"] button { justify-content: flex-start; min-height: 2.4rem; background: var(--surface);
  border: 1px solid var(--line); border-radius: 8px; color: var(--ink); }
[class*="st-key-next-"] button:hover { border-color: var(--cite); color: var(--cite); }
[class*="st-key-next-"] button p { font-size: .85rem; text-align: left; white-space: normal; }
.kb-timeline { list-style: none; margin: .5rem 0 0; padding: 0 0 0 1rem; border-left: 2px solid var(--line); }
.kb-timeline li { position: relative; margin: 0 0 .8rem; padding-left: .3rem; }
.kb-timeline li::before { content: ""; position: absolute; left: -1.42rem; top: .35rem; width: .7rem; height: .7rem;
  border-radius: 50%; background: var(--surface); border: 2px solid var(--cite); }
.kb-timeline .when { font-size: .74rem; font-weight: 600; color: var(--muted); text-transform: uppercase;
  letter-spacing: .04em; }
.kb-timeline .state { font-size: .88rem; font-weight: 600; color: var(--ink); }
.kb-timeline .why { font-size: .84rem; color: var(--muted); line-height: 1.5; }
.kb-timeline blockquote { margin: .35rem 0 0; padding: .45rem .7rem; border-left: 3px solid var(--cite-bg);
  background: var(--paper); font-size: .82rem; color: var(--ink); line-height: 1.5; white-space: pre-line; }
.kb-sim { font-size: .68rem; font-weight: 600; color: var(--warn); background: var(--warn-bg); border-radius: 4px;
  padding: 0 .3rem; margin-left: .2rem; }
[data-testid="stSidebar"] ul.kb-flows { list-style: none; padding: 0; margin: .3rem 0 0; }
[data-testid="stSidebar"] .kb-flows li { font-size: .82rem; padding: .25rem 0; margin: 0; border-bottom: 1px solid var(--line); }
[data-testid="stSidebar"] .kb-flows li span { display: block; color: var(--muted); font-size: .74rem; }

[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h3 { font-size: .95rem; font-weight: 600; margin-top: .6rem; }
.kb-side p { color: var(--muted); font-size: .88rem; line-height: 1.55; }
[data-testid="stSidebar"] ul.kb-stack { list-style: none; padding: 0; margin: .4rem 0 .8rem; }
[data-testid="stSidebar"] .kb-stack li { font-size: .84rem; display: flex; justify-content: space-between; gap: .8rem; padding: .32rem 0;
  border-bottom: 1px solid var(--line); margin: 0; }
.kb-stack li span:first-child { color: var(--muted); }
.kb-stack li span:last-child { color: var(--ink); font-weight: 500; text-align: right; }
.kb-articles { list-style: none; padding: 0; margin: .3rem 0 0; }
[data-testid="stSidebar"] .kb-articles li { margin: 0; padding: .22rem 0; font-size: .86rem; }
.kb-articles a { color: var(--ink); text-decoration: none; }
.kb-articles a:hover { color: var(--cite); text-decoration: underline; }
[data-testid="stSidebar"] p.kb-credit { color: var(--muted); font-size: .8rem; margin-top: 1rem; }
</style>
"""


@st.cache_data
def articles() -> list[tuple[str, str]]:
    found = []
    for name in sorted(os.listdir(os.path.join(ROOT, "docs"))):
        if name.endswith(".md"):
            with open(os.path.join(ROOT, "docs", name)) as f:
                found.append((name, f.readline().lstrip("# ").strip()))
    return found


def visitor_id() -> str:
    """Who's asking, for the per-visitor cap: the client IP from the proxy headers (hashed again on the
    server, never stored raw), falling back to this browser session."""
    try:
        forwarded = st.context.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    except Exception:
        pass
    return st.session_state.setdefault("visitor", os.urandom(8).hex())


def call_api(path: str, payload: dict) -> tuple[int, dict]:
    headers = {"content-type": "application/json", "x-api-key": API_KEY}
    if PUBLIC_DEMO:
        headers["x-demo-client"] = visitor_id()
    request = urllib.request.Request(f"{API_URL}{path}", data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")
    except (urllib.error.URLError, TimeoutError):
        return 503, {"error": "Could not reach the assistant API."}


@st.cache_resource
def local_assistant():
    return build_assistant()


def ask(question: str, history: list) -> dict:
    if API_URL:
        status, body = call_api("/demo/ask" if PUBLIC_DEMO else "/ask", {"question": question, "history": history})
        if status != 200:
            return {"status": "error", "answer": body.get("error", "Something went wrong."), "sources": [],
                    "offer_ticket": not PUBLIC_DEMO, "question": question, "guardrail": {}}
        return body
    return local_assistant().ask(question, history=history)


def escalate(turn: dict) -> tuple[bool, str]:
    payload = {"question": turn["question"], "answer": turn["answer"],
               "sources": [s["source"] for s in turn.get("sources", [])]}
    if API_URL:
        status, body = call_api("/escalate", payload)
        if status == 201:
            return True, f"Ticket {body['ticket_id'][:8]} created (priority {body.get('priority')}). A technician will follow up."
        return False, body.get("error", "Could not create the ticket.")
    guard = build_guardrail({})
    guarded = guard.check_input(turn["question"])
    if guarded.blocked:
        return False, "This request can't be turned into a ticket."
    answer = guard.check_input(turn["answer"])
    try:
        ticket = TicketClient(os.environ.get("TRIAGE_API_URL", "")).create(
            guarded.text, "" if answer.blocked else answer.text, payload["sources"])
    except TicketError as error:
        return False, str(error)
    return True, f"Ticket {ticket['ticket_id'][:8]} created (priority {ticket['priority']}). A technician will follow up."


def answer_markdown(text: str) -> str:
    """The model's answer as safe markdown: HTML escaped, [n] citations turned into small badges."""
    safe = html.escape(text, quote=False)
    return re.sub(r"\[(\d+)\]", r'<sup class="kb-cite">\1</sup>', safe)


def snippet(text: str, limit: int = 220) -> str:
    """First readable lines of a chunk, without markdown headings, trimmed at a word."""
    lines = [ln.strip().lstrip("0123456789.-* ").strip() for ln in text.splitlines()]
    flat = " ".join(ln for ln in lines if ln and not ln.startswith("#"))
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + "…"


def status_html(turn: dict) -> str:
    status = turn.get("status")
    parts = []
    if status == "answered":
        n = len(turn.get("sources", []))
        parts.append(f'<span class="kb-badge ok">✓ Grounded in {n} cited article{"s" if n != 1 else ""}</span>')
        if not PUBLIC_DEMO:
            scores = (turn.get("guardrail", {}).get("output") or {}).get("scores") or {}
            if "GROUNDING" in scores:
                parts.append(f"<span>Grounding score {scores['GROUNDING']:.2f}</span>")
    elif status == "no_answer":
        parts.append('<span class="kb-badge warn">Not covered by the knowledge base</span>')
    elif status == "blocked":
        parts.append('<span class="kb-badge stop">Blocked by the input guardrail</span>')
    else:
        parts.append('<span class="kb-badge stop">No answer</span>')
    if (turn.get("guardrail", {}).get("input") or {}).get("action") == "ANONYMIZED":
        parts.append("<span>Personal details masked</span>")
    if turn.get("latency_ms") is not None:
        parts.append(f"<span>{turn['latency_ms'] / 1000:.1f} s</span>")
    return f'<div class="kb-status">{"".join(parts)}</div>'


def sources_html(sources: list) -> str:
    rows = []
    for s in sources:
        name = html.escape(s["source"])
        rows.append(
            f'<div class="kb-source"><span class="n">{int(s["n"])}</span>'
            f'<div><a href="{REPO_DOCS}{name}" target="_blank">{html.escape(s["title"])}</a>'
            f'<span class="file">{name}</span></div>'
            f'<q>{html.escape(snippet(s.get("snippet", "")))}</q></div>')
    return f'<div class="kb-sources"><div class="kb-sources-title">Sources</div>{"".join(rows)}</div>'


def demo_text(text: str) -> str:
    """Tickets are off in the public demo, so drop any sentence offering one."""
    if not PUBLIC_DEMO or "open a ticket" not in text.lower():
        return text
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if "open a ticket" not in s.lower()]
    return " ".join(kept) or text


AGENT_STEP_NAMES = {"search_knowledge_base": "Searched the knowledge base",
                    "find_similar_incidents": "Checked for similar incidents",
                    "send_fix_to_caller": "Sent the fix to the caller",
                    "escalate_incident": "Escalated",
                    "verify_caller_identity": "Verified identity (MFA push)",
                    "complete_and_confirm": "Sent the results to the caller",
                    "request_approval": "Requested approval",
                    "link_to_outage": "Linked to the outage incident"}


def agent_html(plan: dict) -> str:
    """What the ticket agent would do with this as a new ServiceNow incident (dry run in the demo)."""
    if plan.get("action") == "resolved":
        badge, decision = "ok", ("Would resolve the ticket: the fix was verified by reading the system back "
                                 "(here a simulated directory; on a real ServiceNow ticket that needs a real tenant). "
                                 "The caller can reply to reopen")
    elif plan.get("action") == "awaiting_caller":
        badge, decision = "ok", ("Would send this to the caller and wait: resolved if they confirm, "
                                 "otherwise two daily reminders, then closed for no response")
    elif plan.get("action") == "awaiting_approval":
        badge, decision = "warn", "Would wait for an approver, then run the approved action automatically"
    else:
        badge = "warn"
        decision = f"Would escalate to the {html.escape(str(plan.get('team')))} team"
        if plan.get("priority") and int(plan["priority"]) <= 2:
            decision += f", priority {int(plan['priority'])} - High"
    steps = "".join(f"<li><b>{html.escape(AGENT_STEP_NAMES.get(st_['tool'], st_['tool'].replace('_', ' ').capitalize()))}</b>"
                    + (' <span class="kb-sim">simulated</span>' if st_.get("simulated") else "")
                    + f": {html.escape(str(st_.get('detail', '')))}</li>" for st_ in plan.get("steps", []))
    return (f'<div class="kb-agent"><div class="kb-agent-head"><span class="kb-agent-title">Ticket agent</span>'
            f'<span class="kb-badge {badge}">{"Self-service" if badge == "ok" else ("Approval" if plan.get("action") == "awaiting_approval" else "Escalate")}</span>'
            f'<span class="kb-agent-decision">{decision}</span></div><ol>{steps}</ol>'
            '<div class="kb-agent-foot">Demo caller: Alex Rivera, an employee of a fictional company. Steps marked '
            "simulated run against simulated Entra ID, Intune, and Microsoft 365 backends; nothing real is changed "
            "and nothing is written. In the ServiceNow integration this runs on every new incident. A ticket is "
            "resolved right away only when a real system confirms the fix; otherwise the caller confirms, or it "
            "closes for no response after two reminders. Urgent, security and outage tickets always go to people."
            "</div></div>")


def timeline_html(events: list[dict]) -> str:
    rows = []
    for e in events:
        comment = (f"<blockquote>{html.escape(e['comment'])}</blockquote>" if e.get("comment") else "")
        rows.append(f'<li><div class="when">{html.escape(e["when"])}</div><div class="state">'
                    f'{html.escape(e["state"])}</div><div class="why">{html.escape(e["summary"])}</div>{comment}</li>')
    return f'<ol class="kb-timeline">{"".join(rows)}</ol>'


def next_steps(i: int, turn: dict) -> None:
    """Let the visitor play the caller and watch the ticket's next steps, run by the same policy code as the
    ServiceNow integration (agent.py, followup.py) against a dry-run desk: nothing is written anywhere."""
    plan = turn.get("agent") or {}
    choices = CHOICES.get(plan.get("action"), [])
    if not choices:
        return
    st.markdown('<div class="kb-next"><div class="kb-next-title">What happens next? Play the caller</div>'
                '<div class="kb-next-hint">Runs the same follow-up and reopen code as the live ServiceNow '
                "integration, as a dry run (nothing is sent or written).</div></div>", unsafe_allow_html=True)
    cols = st.columns(len(choices))
    for n, (key, label) in enumerate(choices):
        if cols[n].button(label, key=f"next-{i}-{key}", use_container_width=True):
            turn["lifecycle"] = {"choice": key, "events": play(plan["action"], key, turn.get("question") or "")}
            st.rerun()
    if turn.get("lifecycle"):
        st.markdown(timeline_html(turn["lifecycle"]["events"]), unsafe_allow_html=True)


def show_turn(i: int, turn: dict) -> None:
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        with st.container(key=f"answer-{i}"):
            st.markdown(status_html(turn), unsafe_allow_html=True)
            st.markdown(answer_markdown(demo_text(turn["answer"])), unsafe_allow_html=True)
            if turn.get("sources"):
                st.markdown(sources_html(turn["sources"]), unsafe_allow_html=True)
            if turn.get("agent"):
                st.markdown(agent_html(turn["agent"]), unsafe_allow_html=True)
                next_steps(i, turn)
            if turn.get("ticket"):
                ok, message = turn["ticket"]
                (st.success if ok else st.error)(message)
            if turn.get("offer_ticket") and not PUBLIC_DEMO and not (turn.get("ticket") or (False,))[0]:
                if st.button("Open a ticket", key=f"ticket-{i}"):
                    turn["ticket"] = escalate(turn)
                    st.rerun()


st.set_page_config(page_title="IT Help Desk Assistant", page_icon=":material/support_agent:",
                   layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

intro = ("Ask an IT question in plain words. Answers come only from the company's IT articles, and each "
         "one shows the passages it used so you can check it. If the articles don't cover something, "
         "it says so instead of guessing.")
if PUBLIC_DEMO:
    intro += (" Each answer also shows what the AI ticket agent would do with it as a ServiceNow incident, and "
              "you can play the caller to see what happens next: confirmed, still broken, or no reply at all.")
if not PUBLIC_DEMO:
    intro += " When it can't help, it can open a ticket."
st.markdown(f'<div class="kb-head"><h1>IT Help Desk Assistant</h1><p>{intro}</p></div>', unsafe_allow_html=True)
if PUBLIC_DEMO:
    st.markdown(f'<div class="kb-note"><strong>Public demo.</strong> The knowledge base is {len(articles())} '
                "sample IT articles for a fictional company. Please don't enter personal or real company "
                "information. Ticket creation is turned off, and questions are capped per visitor and per day."
                "</div>", unsafe_allow_html=True)

if PUBLIC_DEMO and not (API_URL and API_KEY):
    st.error("The demo isn't configured yet: ASSISTANT_API_URL and ASSISTANT_API_KEY must be set.")
    st.stop()

turns = st.session_state.setdefault("turns", [])

with st.sidebar:
    if PUBLIC_DEMO:
        st.markdown("### About this demo")
        st.markdown('<div class="kb-side"><p>Retrieval-augmented generation over an IT knowledge base, '
                    "running serverless on AWS. In the private deployment, a ticket agent built on it works "
                    "every new ServiceNow incident: it fixes what it can, sends a verified fix, or escalates with a "
                    "handoff note, and closes a ticket only with evidence (a verified fix, the caller's "
                    "confirmation, or two unanswered reminders).</p></div>", unsafe_allow_html=True)
        stack = [("Model", "Claude Haiku 4.5"), ("Embeddings", "Voyage AI"), ("Vector store", "Amazon S3 Vectors"),
                 ("Safety", "Bedrock Guardrails"), ("Runtime", "AWS Lambda")]
        st.markdown('<ul class="kb-stack">' + "".join(f"<li><span>{k}</span><span>{v}</span></li>" for k, v in stack)
                    + "</ul>", unsafe_allow_html=True)
        st.link_button("Code, architecture and evals", REPO, use_container_width=True)
    elif API_URL:
        st.markdown("### Setup")
        st.write("Mode: **deployed API**")
        st.code(API_URL, language=None)
    else:
        st.markdown("### Setup")
        st.write("Mode: **local**")
        st.write(f"Retriever: `{setting({}, 'RETRIEVER')}`" +
                 (f" ({setting({}, 'EMBEDDER')})" if setting({}, "RETRIEVER") == "local" else ""))
        st.write(f"Model: `{setting({}, 'LLM_PROVIDER')}`")
        st.write(f"Guardrail: `{setting({}, 'GUARDRAIL')}`")
    if PUBLIC_DEMO:
        st.markdown("### What the agent can do")
        st.markdown('<ul class="kb-flows">' + "".join(f"<li>{n}<span>{k}</span></li>" for n, k in AGENT_WORKFLOWS)
                    + "</ul>", unsafe_allow_html=True)
    st.markdown("### What it knows")
    st.markdown('<ul class="kb-articles">' + "".join(
        f'<li><a href="{REPO_DOCS}{name}" target="_blank">{html.escape(title)}</a></li>'
        for name, title in articles()) + "</ul>", unsafe_allow_html=True)
    if turns and st.button("Clear chat", use_container_width=True):
        st.session_state["turns"] = []
        st.rerun()
    if PUBLIC_DEMO:
        st.markdown('<p class="kb-credit">Built by Sagar Modha</p>', unsafe_allow_html=True)

if not turns:
    st.markdown('<div class="kb-label">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for n, example in enumerate(EXAMPLES):
        if cols[n % 2].button(example, key=f"starter-{n}", use_container_width=True):
            st.session_state["pending"] = example
    if PUBLIC_DEMO:
        with st.expander("See it run live in ServiceNow"):
            st.caption("Real incidents on a ServiceNow developer instance, worked by this agent. Account actions "
                       "run against a simulated directory and say so on the ticket.")
            st.image(os.path.join(ROOT, "assets", "servicenow-verified-fix-resolved.png"),
                     caption="Verified fix: unlocked, checked, and resolved in 22 seconds (Solution provided)")
            st.image(os.path.join(ROOT, "assets", "servicenow-no-response-3-strike.png"),
                     caption="No reply: two reminders, then closed as No resolution provided, never as fixed")
    st.markdown('<div class="kb-label">How each answer is checked</div>', unsafe_allow_html=True)
    st.markdown('<ol class="kb-steps">' + "".join(f"<li><b>{t}</b>{d}</li>" for t, d in STEPS) + "</ol>",
                unsafe_allow_html=True)

for i, turn in enumerate(turns):
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(html.escape(turn["asked"], quote=False))
    show_turn(i, turn)

question = (st.chat_input("Describe your IT problem, e.g. my VPN keeps disconnecting", max_chars=MAX_QUESTION)
            or st.session_state.pop("pending", None))
if question:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(html.escape(question, quote=False))
    with st.spinner(f"Searching {len(articles())} IT articles..."):
        history = [{"question": t.get("search_query") or t["question"], "answer": t["answer"]}
                   for t in turns if t.get("status") == "answered"][-3:]
        result = ask(question, history)
    result["asked"] = question
    result.setdefault("question", question)  # the server masks it again before any ticket
    turns.append(result)
    st.rerun()
