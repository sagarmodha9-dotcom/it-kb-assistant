"""Web chat for the IT Knowledge Base Assistant (Streamlit).

  streamlit run ui/app.py

Two modes:
  Deployed: set ASSISTANT_API_URL and ASSISTANT_API_KEY -> calls the AWS API (POST /ask, /escalate)
  Local:    otherwise runs the assistant in this process, configured by the same environment
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

API_URL = os.environ.get("ASSISTANT_API_URL", "").rstrip("/")
API_KEY = os.environ.get("ASSISTANT_API_KEY", "")
EXAMPLES = ["How do I set up MFA on my new phone?",
            "My laptop is asking for a 48-digit key after an update",
            "A visitor needs Wi-Fi tomorrow. What do they use?",
            "What's on the cafeteria menu today?"]


def call_api(path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(f"{API_URL}{path}", data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json", "x-api-key": API_KEY},
                                     method="POST")
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
        status, body = call_api("/ask", {"question": question, "history": history})
        if status != 200:
            return {"status": "error", "answer": body.get("error", "Something went wrong."), "sources": [],
                    "offer_ticket": True, "question": question, "guardrail": {}}
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
                    st.markdown(f"**[{s['n']}] {s['title']}** · `{s['source']}`")
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
        if turn.get("offer_ticket") and not (turn.get("ticket") or (False,))[0]:
            if st.button("Open a ticket", key=f"ticket-{i}"):
                turn["ticket"] = escalate(turn)
                st.rerun()


st.set_page_config(page_title="IT Help Assistant", page_icon="💬")
st.title("IT Help Assistant")
st.caption("Answers come only from the IT knowledge base, with sources. If it can't find an answer, "
           "it says so and can open a ticket.")

with st.sidebar:
    st.subheader("Setup")
    if API_URL:
        st.write("Mode: **deployed API**")
        st.code(API_URL, language=None)
    else:
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

question = st.chat_input("Describe your IT problem") or st.session_state.pop("pending", None)
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
