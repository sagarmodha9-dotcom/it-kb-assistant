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
import json
import os
import sys
import urllib.error
import urllib.request

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from factory import build_assistant, build_guardrail, setting  # noqa: E402
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
REPO_DOCS = "https://github.com/sagarmodha9-dotcom/it-kb-assistant/blob/main/docs/"
EXAMPLES = ["How do I set up MFA on my new phone?",
            "My laptop is asking for a 48-digit key after an update",
            "A visitor needs Wi-Fi tomorrow. What do they use?",
            "What's on the cafeteria menu today?"]


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


def show_turn(i: int, turn: dict) -> None:
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn.get("sources"):
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for s in turn["sources"]:
                    st.markdown(f"**[{s['n']}] {s['title']}** · [`{s['source']}`]({REPO_DOCS}{s['source']})")
                    st.caption(s["snippet"])
        meta = turn.get("guardrail", {})
        notes = []
        if meta.get("input", {}).get("action") == "ANONYMIZED":
            notes.append("personal info was masked before processing")
        scores = meta.get("output", {}).get("scores") or {}
        if "GROUNDING" in scores:
            notes.append(f"grounding {scores['GROUNDING']:.2f}")
        if turn.get("latency_ms") is not None:
            notes.append(f"{turn['latency_ms']} ms")
        if notes:
            st.caption(" · ".join(notes))
        if turn.get("ticket"):
            ok, message = turn["ticket"]
            (st.success if ok else st.error)(message)
        if turn.get("offer_ticket") and not PUBLIC_DEMO and not (turn.get("ticket") or (False,))[0]:
            if st.button("Open a ticket", key=f"ticket-{i}"):
                turn["ticket"] = escalate(turn)
                st.rerun()


st.set_page_config(page_title="IT Help Assistant", page_icon="💬")
st.title("IT Help Assistant")
if PUBLIC_DEMO:
    st.caption("Answers come only from the IT knowledge base, with sources you can open and check. "
               "If the articles don't cover a question, it says so instead of guessing.")
    st.info("**Public demo.** The knowledge base is 12 sample IT articles for a fictional company. "
            "Please don't enter personal or real company information. Ticket creation is turned off here, "
            "and usage is capped per visitor and per day.", icon="ℹ️")
else:
    st.caption("Answers come only from the IT knowledge base, with sources. If it can't find an answer, "
               "it says so and can open a ticket.")

if PUBLIC_DEMO and not (API_URL and API_KEY):
    st.error("The demo isn't configured yet.")
    st.stop()

with st.sidebar:
    if PUBLIC_DEMO:
        st.subheader("About this demo")
        st.write("RAG over IT articles with Claude, Voyage AI embeddings in Amazon S3 Vectors, and Amazon "
                 "Bedrock Guardrails, running on AWS Lambda.")
        st.markdown("[Code, architecture, and evals on GitHub](https://github.com/sagarmodha9-dotcom/it-kb-assistant)")
        st.subheader("What it knows")
        for name in sorted(os.listdir(os.path.join(ROOT, "docs"))):
            if name.endswith(".md"):
                with open(os.path.join(ROOT, "docs", name)) as f:
                    title = f.readline().lstrip("# ").strip()
                st.markdown(f"- [{title}]({REPO_DOCS}{name})")
    elif API_URL:
        st.subheader("Setup")
        st.write("Mode: **deployed API**")
        st.code(API_URL, language=None)
    else:
        st.subheader("Setup")
        st.write("Mode: **local**")
        st.write(f"Retriever: `{setting({}, 'RETRIEVER')}`" +
                 (f" ({setting({}, 'EMBEDDER')})" if setting({}, "RETRIEVER") == "local" else ""))
        st.write(f"Model: `{setting({}, 'LLM_PROVIDER')}`")
        st.write(f"Guardrail: `{setting({}, 'GUARDRAIL')}`")
    st.subheader("Try")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["pending"] = example
    if st.button("Clear chat"):
        st.session_state["turns"] = []

turns = st.session_state.setdefault("turns", [])
for i, turn in enumerate(turns):
    with st.chat_message("user"):
        st.markdown(turn["asked"])
    show_turn(i, turn)

question = (st.chat_input("Describe your IT problem", max_chars=MAX_QUESTION)
            or st.session_state.pop("pending", None))
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.spinner("Searching the knowledge base..."):
        history = [{"question": t.get("search_query") or t["question"], "answer": t["answer"]}
                   for t in turns if t.get("status") == "answered"][-3:]
        result = ask(question, history)
    result["asked"] = question
    result.setdefault("question", question)  # the server masks it again before any ticket
    turns.append(result)
    st.rerun()
