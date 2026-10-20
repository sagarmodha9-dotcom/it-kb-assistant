"""Public demo and web chat for the IT Knowledge Base Assistant and its service-desk agent (Streamlit).

  streamlit run ui/app.py

Three modes:
  Deployed:    set ASSISTANT_API_URL and ASSISTANT_API_KEY -> calls the AWS API (POST /ask, /escalate)
  Public demo: also set PUBLIC_DEMO=true and use the demo key -> calls POST /demo/ask only. No ticket
               button, the API URL isn't shown, questions are limited to 300 characters, and the AWS
               side enforces daily and per-visitor caps. Settings can come from Streamlit secrets.
  Local:       otherwise runs the assistant in this process, configured by the same environment
               variables as src/factory.py (offline by default: hash embeddings + fake model)

The AI ticket desk (src/sandbox.py) runs the real agent code against an in-memory demo ticket and the
simulated company in src/enterprise/sim.py: this app holds no ServiceNow URL or credential and cannot
reach one. Custom incidents use the same rate-limited /demo/ask route as the chat.
Header links: GitHub repo, LinkedIn (override with LINKEDIN_URL), and a Portfolio button only if PORTFOLIO_URL is set.
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
from sandbox import MAX_CUSTOM, SAMPLES, STATE, new_live, new_sample, stages  # noqa: E402
from showcase import EVAL_RUN, capabilities, eval_summary, load_eval  # noqa: E402
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
LIVE_TICKETS_PER_SESSION = 5     # on top of the server's per-visitor and daily caps
GITHUB_PROFILE = "https://github.com/sagarmodha9-dotcom"
REPO = f"{GITHUB_PROFILE}/it-kb-assistant"
REPO_DOCS = f"{REPO}/blob/main/docs/"
PORTFOLIO_URL = config("PORTFOLIO_URL")
LINKEDIN_URL = config("LINKEDIN_URL", "https://www.linkedin.com/in/sagarmodha/")
SAFE_URL = re.compile(r"https://[\w.-]+(/[\w./%?=&#~+-]*)?")
NAV = ["AI ticket desk", "Ask the assistant", "12 capabilities", "ServiceNow, live", "Architecture", "Evaluation"]

EXAMPLES = [(":material/lock:", "Identity", "I'm locked out of my account"),
            (":material/smartphone:", "MFA", "How do I set up MFA on my new phone?"),
            (":material/folder_shared:", "Access", "I need access to the Finance Reports SharePoint site"),
            (":material/vpn_lock:", "Network", "My VPN keeps disconnecting"),
            (":material/phishing:", "Security", "I think I clicked a phishing link"),
            (":material/restaurant:", "Out of scope", "What's on the cafeteria menu today?")]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
  --paper: #F5F7FA; --surface: #FFFFFF; --ink: #0E1B2E; --muted: #5B6B80; --line: #E2E8F0; --navy: #0B2545;
  --blue: #1F4FD1; --blue-bg: #EAF0FD; --ok: #157347; --ok-bg: #E6F4EC; --warn: #8A5A00; --warn-bg: #FCF3DF;
  --stop: #A0301F; --stop-bg: #FBEAE7; --sim: #7A4E00; --sim-bg: #FFF1CC; --real: #0F5C3A; --real-bg: #DFF3E8;
  --note: #F4C542; --radius: 12px;
}
html, body, p, li, h1, h2, h3, h4, label, button, textarea, input, .stMarkdown {
  font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif !important; }
code, pre, .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }
[data-testid="stDecoration"], [data-testid="stToolbarActions"] { display: none; }
.block-container { max-width: 1160px; padding: 1.1rem 1.4rem 4rem; }
@media (max-width: 640px) { .block-container { padding: .7rem .85rem 3rem; } }

.kb-top { display: flex; align-items: center; justify-content: space-between; gap: .8rem; flex-wrap: wrap;
  padding: .5rem 0 .85rem; border-bottom: 1px solid var(--line); }
.kb-brand { display: flex; align-items: center; gap: .6rem; font-weight: 700; color: var(--navy); font-size: 1.02rem; }
.kb-mark { width: 30px; height: 30px; border-radius: 8px; background: var(--navy); color: #fff; display: flex;
  align-items: center; justify-content: center; font-size: .78rem; font-weight: 700; }
.kb-links { display: flex; gap: .45rem; flex-wrap: wrap; }
.kb-links a { text-decoration: none; font-size: .85rem; font-weight: 600; padding: .38rem .8rem; border-radius: 8px;
  border: 1px solid var(--line); color: var(--ink) !important; background: var(--surface); }
.kb-links a.primary { background: var(--navy); border-color: var(--navy); color: #fff !important; }

.kb-hero { padding: 1.9rem 0 1.1rem; }
.kb-eyebrow { font-size: .76rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--blue); }
.kb-hero h1 { font-size: 2.4rem; line-height: 1.12; letter-spacing: -.025em; color: var(--navy); font-weight: 700;
  margin: .45rem 0 .75rem; padding: 0; max-width: 22ch; }
.kb-hero p { font-size: 1.06rem; line-height: 1.6; color: var(--muted); max-width: 66ch; margin: 0; }
.kb-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: 1rem; }
.kb-chip { font-size: .77rem; font-weight: 500; color: var(--navy); background: var(--surface); border: 1px solid var(--line);
  border-radius: 999px; padding: .2rem .65rem; }
@media (max-width: 640px) { .kb-hero { padding-top: 1.1rem; } .kb-hero h1 { font-size: 1.75rem; } .kb-hero p { font-size: .97rem; } }

.kb-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: .7rem; margin: .3rem 0 1.2rem; }
.kb-stat { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: .85rem 1rem; }
.kb-stat .v { font-size: 1.5rem; font-weight: 700; color: var(--navy); letter-spacing: -.02em; }
.kb-stat .l { font-size: .85rem; font-weight: 600; color: var(--ink); margin-top: .1rem; }
.kb-stat .c { font-size: .75rem; color: var(--muted); line-height: 1.45; margin-top: .3rem; }

.st-key-nav [data-testid="stButtonGroup"] > div { flex-wrap: wrap; gap: .3rem; }
.st-key-nav button { font-weight: 600 !important; }

.kb-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 1rem 1.1rem; }
.kb-lede { color: var(--muted); font-size: .93rem; line-height: 1.55; margin: .5rem 0 .8rem; max-width: 80ch; }
.kb-label { font-size: .76rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  margin: 1rem 0 .45rem; }
.kb-badge { display: inline-flex; align-items: center; gap: .3rem; padding: .15rem .55rem; border-radius: 999px;
  font-weight: 600; font-size: .73rem; white-space: nowrap; background: var(--paper); color: var(--muted); }
.kb-badge.ok { background: var(--ok-bg); color: var(--ok); } .kb-badge.warn { background: var(--warn-bg); color: var(--warn); }
.kb-badge.stop { background: var(--stop-bg); color: var(--stop); } .kb-badge.info { background: var(--blue-bg); color: var(--blue); }
.kb-badge.sim { background: var(--sim-bg); color: var(--sim); letter-spacing: .03em; }
.kb-badge.real { background: var(--real-bg); color: var(--real); } .kb-badge.mixed { background: var(--blue-bg); color: var(--blue); }
.kb-banner { display: flex; gap: .7rem; align-items: flex-start; background: var(--sim-bg); border: 1px solid #F2DDA4;
  border-radius: 10px; padding: .65rem .9rem; font-size: .84rem; color: #5C4100; line-height: 1.5; margin: .2rem 0 .9rem; }
.kb-foot { font-size: .77rem; color: var(--muted); line-height: 1.5; margin-bottom: .6rem; }

/* sample incident buttons and chat examples */
[class*="st-key-ex-"] button, [class*="st-key-sample-"] button { justify-content: flex-start; text-align: left;
  min-height: 3.7rem; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  color: var(--ink); padding: .6rem .8rem; }
[class*="st-key-ex-"] button:hover, [class*="st-key-sample-"] button:hover { border-color: var(--blue); color: var(--blue); }
[class*="st-key-ex-"] button p, [class*="st-key-sample-"] button p { font-size: .88rem; text-align: left; white-space: normal; }
[class*="st-key-ex-"] button > div, [class*="st-key-sample-"] button > div { justify-content: flex-start; width: 100%; }

/* ticket form */
.kb-tk { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.kb-tk-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .5rem;
  background: var(--navy); color: #fff; padding: .6rem 1rem; }
.kb-tk-bar .num { font-family: 'IBM Plex Mono', monospace; font-weight: 500; font-size: .95rem; }
.kb-tk-bar .src { font-size: .74rem; opacity: .85; }
.kb-tk-body { padding: .9rem 1rem 1rem; }
.kb-tk-desc { font-size: 1.05rem; font-weight: 600; color: var(--ink); margin-bottom: .7rem; }
.kb-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .55rem .9rem; }
.kb-field .k { font-size: .72rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
.kb-field .v { font-size: .9rem; color: var(--ink); font-weight: 500; margin-top: .1rem; }
.kb-res { margin-top: .8rem; padding: .6rem .75rem; border-radius: 9px; background: var(--paper); border: 1px solid var(--line);
  font-size: .84rem; line-height: 1.5; color: var(--ink); }
.kb-res b { font-weight: 600; }
.kb-pill { display: inline-block; padding: .14rem .55rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }
.kb-pill.s1 { background: var(--blue-bg); color: var(--blue); } .kb-pill.s2 { background: var(--warn-bg); color: var(--warn); }
.kb-pill.s3 { background: #EFE7FB; color: #5B3A9C; } .kb-pill.s6 { background: var(--ok-bg); color: var(--ok); }

/* stage strip */
.kb-stages { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: .4rem; margin: .8rem 0 .2rem; }
.kb-stage { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); padding: .45rem .55rem; }
.kb-stage .n { font-size: .68rem; font-weight: 700; color: var(--muted); }
.kb-stage .t { font-size: .78rem; font-weight: 600; color: var(--ink); line-height: 1.25; }
.kb-stage .d { font-size: .72rem; color: var(--muted); margin-top: .15rem; line-height: 1.3; }
.kb-stage.done { border-top: 3px solid var(--ok); } .kb-stage.waiting { border-top: 3px solid var(--blue); }
.kb-stage.blocked, .kb-stage.failed { border-top: 3px solid var(--stop); } .kb-stage.skipped { border-top: 3px solid var(--line); opacity: .8; }
@media (max-width: 900px) { .kb-stages { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
@media (max-width: 520px) { .kb-stages { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .kb-fields { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.kb-journey { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .6rem; margin: .2rem 0 1rem; }
.kb-step { display: flex; gap: .65rem; align-items: flex-start; background: #fff; border: 1px solid var(--line);
  border-radius: 12px; padding: .75rem .85rem; }
.kb-step .n { flex: none; width: 1.6rem; height: 1.6rem; border-radius: 50%; background: var(--navy, #0E1B2E); color: #fff;
  display: grid; place-items: center; font-size: .8rem; font-weight: 600; }
.kb-step b { display: block; font-size: .9rem; color: var(--ink); }
.kb-step span:not(.n) { font-size: .8rem; color: var(--muted); }
@media (max-width: 640px) { .kb-journey { grid-template-columns: 1fr; } }

/* execution timeline */
.kb-exec { list-style: none; margin: 0; padding: 0; }
.kb-exec > li { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); padding: .55rem .7rem; margin-bottom: .45rem; }
.kb-exec .h { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .5rem; }
.kb-exec .tool { font-family: 'IBM Plex Mono', monospace; font-size: .82rem; font-weight: 500; color: var(--navy); }
.kb-exec .ev { font-size: .85rem; font-weight: 600; color: var(--ink); }
.kb-exec .at { margin-left: auto; font-size: .7rem; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }
.kb-exec .tgt { font-size: .74rem; color: var(--muted); }
.kb-exec .why { font-size: .82rem; color: var(--muted); margin-top: .2rem; line-height: 1.45; }
.kb-exec li.policy { border-left: 3px solid var(--blue); } .kb-exec li.guard { border-left: 3px solid #9AA8BA; }
.kb-exec li.person { border-left: 3px solid #5B3A9C; } .kb-exec li.event { background: var(--paper); }
.kb-exec details { margin-top: .3rem; } .kb-exec summary { font-size: .74rem; color: var(--blue); cursor: pointer; }
.kb-exec pre { font-size: .72rem; background: var(--paper); border: 1px solid var(--line); border-radius: 7px; padding: .45rem .55rem;
  white-space: pre-wrap; word-break: break-word; margin: .3rem 0 0; color: var(--ink); }

/* activity stream (journal) */
.kb-act { list-style: none; margin: 0; padding: 0; }
.kb-act > li { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); padding: .55rem .7rem; margin-bottom: .45rem; }
.kb-act li.wn { border-left: 4px solid var(--note); } .kb-act li.cm { border-left: 4px solid var(--blue); }
.kb-act li.fc { background: var(--paper); }
.kb-act .h { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .4rem; font-size: .74rem; color: var(--muted); }
.kb-act .h b { color: var(--ink); font-weight: 600; }
.kb-act .body { font-size: .8rem; color: var(--ink); white-space: pre-wrap; word-break: break-word; margin-top: .3rem; line-height: 1.5; }
.kb-act .chg { font-size: .8rem; color: var(--ink); margin-top: .2rem; }
.kb-act .chg span { color: var(--muted); }

[class*="st-key-act-"] button { border-radius: 9px; }

/* chat */
[data-testid="stChatMessage"] { background: transparent; padding: .7rem 0; }
[class*="st-key-answer-"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1rem 1.15rem 1.05rem; gap: .55rem; }
[class*="st-key-answer-"] .stMarkdown p, [class*="st-key-answer-"] .stMarkdown li { line-height: 1.62; }
sup.kb-cite { display: inline-flex; align-items: center; justify-content: center; min-width: 1.15rem; height: 1.15rem;
  padding: 0 .25rem; margin: 0 .1rem; border-radius: 4px; background: var(--blue-bg); color: var(--blue); font-size: .7rem;
  font-weight: 700; vertical-align: super; line-height: 1; }
.kb-status { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .8rem; font-size: .8rem; color: var(--muted); }
.kb-src-title { font-size: .76rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--muted);
  border-top: 1px solid var(--line); padding-top: .7rem; }
.kb-passage { font-size: .88rem; line-height: 1.6; color: var(--ink); background: var(--paper); border-left: 3px solid var(--blue);
  border-radius: 6px; padding: .6rem .8rem; white-space: pre-line; }
.kb-passage-meta { font-size: .78rem; color: var(--muted); margin-top: .4rem; } .kb-passage-meta a { color: var(--blue); }
.kb-agent { border-top: 1px solid var(--line); padding-top: .7rem; }
.kb-agent-head { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem .6rem; margin-bottom: .4rem; }
.kb-agent-title { font-size: .76rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }
.kb-agent-decision { font-size: .9rem; font-weight: 600; color: var(--ink); }
.kb-agent ol { margin: 0 0 .35rem; padding-left: 1.2rem; }
.kb-agent li { font-size: .84rem; color: var(--muted); line-height: 1.5; margin: .1rem 0; }
.kb-agent li b { color: var(--ink); font-weight: 500; }

/* capabilities */
.kb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: .75rem; }
.kb-cap { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: .9rem 1rem;
  display: flex; flex-direction: column; gap: .4rem; }
.kb-cap-h { display: flex; align-items: flex-start; justify-content: space-between; gap: .5rem; }
.kb-cap-n { font-size: .72rem; font-weight: 700; color: var(--blue); }
.kb-cap-name { font-weight: 600; color: var(--ink); font-size: .95rem; line-height: 1.3; }
.kb-cap p { margin: 0; font-size: .85rem; color: var(--muted); line-height: 1.5; }
.kb-cap .gate { font-size: .8rem; color: var(--ink); } .kb-cap .gate b { font-weight: 600; }
.kb-tools { display: flex; flex-wrap: wrap; gap: .3rem; }
.kb-tools code { font-size: .72rem; background: var(--paper); border: 1px solid var(--line); border-radius: 5px; padding: .05rem .35rem; color: var(--navy); }

/* ServiceNow evidence */
.kb-ba { display: grid; grid-template-columns: 1fr auto 1fr; gap: .8rem; align-items: stretch; }
.kb-ba .arrow { display: flex; align-items: center; color: var(--muted); font-size: 1.4rem; }
.kb-ba dl { display: grid; grid-template-columns: auto 1fr; gap: .25rem .8rem; margin: .5rem 0 0; font-size: .85rem; }
.kb-ba dt { color: var(--muted); } .kb-ba dd { margin: 0; color: var(--ink); font-weight: 500; }
@media (max-width: 760px) { .kb-ba { grid-template-columns: 1fr; } .kb-ba .arrow { justify-content: center; transform: rotate(90deg); } }
.kb-timeline { list-style: none; margin: .5rem 0 0; padding: 0 0 0 1rem; border-left: 2px solid var(--line); }
.kb-timeline li { position: relative; margin: 0 0 .85rem; padding-left: .35rem; }
.kb-timeline li::before { content: ""; position: absolute; left: -1.43rem; top: .32rem; width: .72rem; height: .72rem;
  border-radius: 50%; background: var(--surface); border: 2px solid var(--blue); }
.kb-timeline .when { font-size: .72rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.kb-timeline .state { font-size: .9rem; font-weight: 600; color: var(--ink); }
.kb-timeline .why { font-size: .84rem; color: var(--muted); line-height: 1.5; }
.kb-out { white-space: pre-wrap; font-size: .78rem; line-height: 1.55; background: #0E1B2E; color: #E6ECF5; border-radius: 10px;
  padding: .8rem .95rem; overflow-x: auto; margin: .3rem 0 0; }

/* architecture */
.kb-lane { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: .9rem 1rem; margin-bottom: .8rem; }
.kb-lane-h { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; font-weight: 600; color: var(--navy); margin-bottom: .6rem; }
.kb-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: .45rem; }
.kb-node { flex: 1 1 140px; min-width: 128px; border: 1px solid var(--line); border-radius: 10px; padding: .55rem .65rem; background: var(--paper); }
.kb-node b { display: block; font-size: .84rem; color: var(--ink); }
.kb-node span { display: block; font-size: .74rem; color: var(--muted); line-height: 1.4; margin-top: .15rem; }
.kb-node.aws { border-top: 3px solid #E08A00; } .kb-node.ai { border-top: 3px solid var(--blue); }
.kb-node.sn { border-top: 3px solid #2E8B57; } .kb-node.ui { border-top: 3px solid var(--navy); } .kb-node.sim { border-top: 3px solid #C9A227; }
.kb-arrow { display: flex; align-items: center; color: var(--muted); font-size: .95rem; }
@media (max-width: 640px) { .kb-arrow { display: none; } .kb-node { flex-basis: 100%; } }

/* tables */
.kb-tablewrap { overflow-x: auto; }
.kb-table { width: 100%; border-collapse: collapse; font-size: .85rem; background: var(--surface); border: 1px solid var(--line); }
.kb-table th { text-align: left; font-size: .72rem; letter-spacing: .05em; text-transform: uppercase; color: var(--muted);
  background: var(--paper); padding: .55rem .75rem; border-bottom: 1px solid var(--line); }
.kb-table td { padding: .55rem .75rem; border-bottom: 1px solid var(--line); color: var(--ink); vertical-align: top; }
.kb-table tr:last-child td { border-bottom: 0; } .kb-table td.num { font-weight: 600; white-space: nowrap; }
.kb-bar { height: 8px; border-radius: 99px; background: var(--paper); border: 1px solid var(--line); overflow: hidden; margin-top: .3rem; }
.kb-bar i { display: block; height: 100%; background: var(--blue); } .kb-bar.base i { background: #9AA8BA; }
.kb-step-h { font-weight: 600; color: var(--ink); font-size: .9rem; }

.kb-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: .8rem; color: var(--muted);
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: .6rem; }
.kb-footer a { color: var(--blue); text-decoration: none; }
.kb-articles { columns: 2; column-gap: 1.5rem; list-style: none; padding: 0; margin: .2rem 0 0; }
.kb-articles li { font-size: .86rem; padding: .2rem 0; break-inside: avoid; }
.kb-articles a { color: var(--ink); text-decoration: none; }
@media (max-width: 640px) { .kb-articles { columns: 1; } }
</style>
"""


# ---------------------------------------------------------------- data and API helpers
@st.cache_data
def articles() -> list[tuple[str, str]]:
    found = []
    for name in sorted(os.listdir(os.path.join(ROOT, "docs"))):
        if name.endswith(".md"):
            with open(os.path.join(ROOT, "docs", name)) as f:
                found.append((name, f.readline().lstrip("# ").strip()))
    return found


@st.cache_data
def evals() -> dict:
    return eval_summary(load_eval(os.path.join(ROOT, EVAL_RUN)))


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
        try:
            return error.code, json.loads(error.read() or b"{}")
        except ValueError:
            return error.code, {}
    except (urllib.error.URLError, TimeoutError, ValueError):
        return 503, {"error": "Could not reach the assistant API."}


@st.cache_resource
def local_assistant():
    return build_assistant()


def ask(question: str, history: list) -> dict:
    if API_URL:
        status, body = call_api("/demo/ask" if PUBLIC_DEMO else "/ask", {"question": question, "history": history})
        if status != 200:
            return {"status": "error", "answer": body.get("error", "Something went wrong."), "sources": [],
                    "offer_ticket": not PUBLIC_DEMO, "question": question, "guardrail": {}, "http_status": status}
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


# ---------------------------------------------------------------- small rendering helpers
def esc(text) -> str:
    # Newlines become entities so st.markdown never sees a blank line inside our HTML (a blank line ends the HTML
    # block and the rest gets parsed as markdown: stray bullets and big gaps). pre-wrap still shows them as breaks.
    return html.escape(str(text if text is not None else "")).replace("\r", "").replace("\n", "&#10;")


def answer_markdown(text: str) -> str:
    """The model's answer as safe markdown: HTML escaped, [n] citations turned into small badges."""
    safe = html.escape(text, quote=False)
    return re.sub(r"\[(\d+)\]", r'<sup class="kb-cite">\1</sup>', safe)


def passage(text: str) -> str:
    """A retrieved chunk for reading: markdown heading marks removed, line breaks kept."""
    lines = [re.sub(r"^#+\s*", "", ln.strip()) for ln in str(text or "").splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def demo_text(text: str) -> str:
    """Tickets are off in the public demo, so drop any sentence offering one."""
    if not PUBLIC_DEMO or "open a ticket" not in text.lower():
        return text
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if "open a ticket" not in s.lower()]
    return " ".join(kept) or text


def link(url: str, text: str, primary: bool = False) -> str:
    if not url or not SAFE_URL.fullmatch(url):
        return ""
    return f'<a class="{"primary" if primary else ""}" href="{esc(url)}" target="_blank">{esc(text)}</a>'


def pretty(value) -> str:
    return esc(json.dumps(value, indent=1, default=str)[:1500])


def go(section: str) -> None:
    st.session_state["nav_target"] = section
    st.rerun()


# ---------------------------------------------------------------- header
def header() -> None:
    links = link(REPO, "GitHub", primary=True) + link(PORTFOLIO_URL, "Portfolio") + link(LINKEDIN_URL, "LinkedIn")
    st.markdown(f'<div class="kb-top"><div class="kb-brand"><div class="kb-mark">IT</div>IT Service Desk Agent</div>'
                f'<div class="kb-links">{links}</div></div>', unsafe_allow_html=True)
    chips = ["RAG with citations", "Agent orchestration (Claude tool use)", "AWS Lambda + API Gateway", "Amazon S3 Vectors",
             "Bedrock Guardrails", "ServiceNow integration", "Measured evaluations"]
    st.markdown(
        '<div class="kb-hero"><div class="kb-eyebrow">Enterprise AI · built by Sagar Modha</div>'
        "<h1>An AI service desk agent that answers, acts, and shows its work.</h1>"
        "<p>It answers IT questions only from the company's knowledge base, citing the exact passages, and works "
        "incidents end to end: it fixes what policy allows, gets an owner's approval where required, and escalates the "
        "rest with a handoff note. Open the ticket desk below to create an incident and watch it investigate, act on a "
        "simulated company and resolve or escalate, with every step on the record.</p>"
        f'<div class="kb-chips">{"".join(f"<span class=kb-chip>{esc(c)}</span>" for c in chips)}</div></div>',
        unsafe_allow_html=True)


def stats() -> None:
    e, h = evals(), evals()["hit1"]
    cards = [(f"{e['passed']}/{e['total']}", "End-to-end eval cases passed",
              f"42 cases × {e['runs']} runs on the deployed stack (S3 Vectors, Claude Haiku 4.5, Guardrails), Oct 18 2026"),
             (f"{h['semantic']:.0%}", "Right article ranked first",
              f"{h['n']} questions, same run. Keyword (BM25) baseline: {h['bm25']:.0%} overall, "
              f"{h['bm25_paraphrased']:.0%} on the {h['n_paraphrased']} paraphrased"),
             ("22 s", "Live ServiceNow incident to verified resolution",
              "Developer instance, Oct 19 2026. The unlock ran on the simulated directory"),
             ("11/11", "Public-demo security checks", "Keys, route isolation, input limits, injection, per-visitor cap; "
              "run against the live API, Oct 19 2026")]
    st.markdown('<div class="kb-stats">' + "".join(
        f'<div class="kb-stat"><div class="v">{esc(v)}</div><div class="l">{esc(lbl)}</div><div class="c">{esc(c)}</div></div>'
        for v, lbl, c in cards) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- AI ticket desk
def field(k: str, v: str) -> str:
    return f'<div class="kb-field"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>'


def ticket_form(t) -> str:
    r = t.record
    code = t.sn.rec["state"]
    state = f'<span class="kb-pill s{esc(code)}">{esc(r["state"])}{(" · " + esc(r["hold_reason"])) if r["hold_reason"] else ""}</span>'
    source = ("Sample incident · tool choices replayed from a script" if t.source == "sample"
              else "Your incident · tool choices made by the live model")
    res = ""
    if r["close_code"]:
        res = (f'<div class="kb-res"><b>Resolution code:</b> {esc(r["close_code"])}<br><b>Resolution notes:</b> '
               f'{esc(r["close_notes"])}</div>')
    fields = "".join([field("State", state), field("Priority", esc(r["priority"])),
                      field("Assignment group", esc(r["assignment_group"] or "(none)")),
                      field("Caller", esc(r["caller_id"]) + " <span class='kb-badge'>fictional</span>"),
                      field("Approval", esc(r["approval"])), field("Parent incident", esc(r["parent_incident"] or "—"))])
    return (f'<div class="kb-tk"><div class="kb-tk-bar"><span class="num">{esc(r["number"])}</span>'
            f'<span class="src">{esc(source)} · demo ticket, not ServiceNow</span></div><div class="kb-tk-body">'
            f'<div class="kb-tk-desc">{esc(r["short_description"])}</div><div class="kb-fields">{fields}</div>{res}</div></div>')


def stages_html(t) -> str:
    cells = "".join(f'<div class="kb-stage {esc(s["status"])}"><div class="n">{i}</div><div class="t">{esc(s["name"])}</div>'
                    f'<div class="d">{esc(s["detail"])}</div></div>' for i, s in enumerate(stages(t), 1))
    return f'<div class="kb-stages">{cells}</div>'


STATUS = {"ok": ("ok", "OK"), "info": ("info", "No match"), "refused": ("stop", "REFUSED"), "blocked": ("stop", "BLOCKED"),
          "approval": ("warn", "NEEDS APPROVAL"), "failed": ("stop", "FAILED")}


def execution_html(t) -> str:
    rows = []
    for e in t.timeline:
        at = f'<span class="at">{esc(e["at"][11:])}</span>'
        if e["kind"] == "tool":
            cls, label = STATUS.get(e.get("status"), ("", e.get("status", "")))
            sim = '<span class="kb-badge sim">SIMULATED</span>' if e.get("simulated") else ""
            io = ""
            if e.get("input") is not None or e.get("output") is not None:
                io = (f"<details><summary>Tool input and output</summary><pre>input: {pretty(e.get('input'))}</pre>"
                      f"<pre>output: {pretty(e.get('output'))}</pre></details>")
            rows.append(f'<li class="tool"><div class="h"><span class="tool">{esc(e["title"])}</span>'
                        f'<span class="kb-badge {cls}">{esc(label)}</span>{sim}{at}</div>'
                        f'<div class="tgt">{esc(e.get("target", ""))}</div>'
                        + (f'<div class="why">{esc(e.get("verdict"))}</div>' if e.get("verdict") else "") + io + "</li>")
        else:
            rows.append(f'<li class="{esc(e["kind"])}"><div class="h"><span class="ev">{esc(e["title"])}</span>{at}</div>'
                        + (f'<div class="why">{esc(e["detail"])}</div>' if e.get("detail") else "") + "</li>")
    return f'<ul class="kb-exec">{"".join(reversed(rows))}</ul>'


LABELS = {"state": ("State", lambda v: STATE.get(str(v), v) if v != "" else "—"), "hold_reason": ("On hold reason",
          lambda v: "Awaiting Caller" if v == "1" else "—"), "priority": ("Priority", lambda v: f"{v}" if v else "—"),
          "assignment_group": ("Assignment group", lambda v: v or "—"), "approval": ("Approval", lambda v: v or "—"),
          "close_code": ("Resolution code", lambda v: v or "—"), "parent_incident": ("Parent incident", lambda v: v or "—")}


def activity_html(t) -> str:
    items = [(e["at"], 1, "wn" if e["kind"] == "work_notes" else "cm", e["author"],
              "Work notes (internal)" if e["kind"] == "work_notes" else "Comments (visible to the caller)",
              f'<div class="body">{esc(e["text"])}</div>') for e in t.sn.journal]
    for c in t.sn.changes:
        lines = "".join(f'<div class="chg">{esc(LABELS[f][0])}: <span>{esc(LABELS[f][1](old))} →</span> '
                        f'{esc(LABELS[f][1](new))}</div>' for f, (old, new) in c["fields"].items() if f in LABELS)
        items.append((c["at"], 0, "fc", c["author"], "Field changes", lines))
    items.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return '<ul class="kb-act">' + "".join(
        f'<li class="{cls}"><div class="h"><span><b>{esc(who)}</b> · {esc(kind)}</span><span>{esc(at)}</span></div>{body}</li>'
        for at, _, cls, who, kind, body in items) + "</ul>"


def create_ticket_panel() -> None:
    st.markdown('<div class="kb-label">Start from a sample incident</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for n, sc in enumerate(SAMPLES):
        if cols[n % 3].button(f"**{sc['category']}** · {sc['title']}  \n{sc['expect']}", key=f"sample-{sc['id']}",
                              use_container_width=True):
            st.session_state["ticket"] = new_sample(sc["id"], next_number())
            st.rerun()
    st.markdown('<div class="kb-label">Or describe your own incident</div>', unsafe_allow_html=True)
    live_ok = bool(API_URL and API_KEY)
    used = st.session_state.get("live_used", 0)
    with st.form("custom", clear_on_submit=False, border=False):
        text = st.text_area("Incident description", max_chars=MAX_CUSTOM, height=90, label_visibility="collapsed",
                            placeholder="e.g. Outlook keeps crashing on my laptop since this morning")
        sent = st.form_submit_button("Submit to the live AI agent", type="primary", icon=":material/send:",
                                     disabled=not live_ok or used >= LIVE_TICKETS_PER_SESSION)
    note = ("The live model reads your text, searches the knowledge base and picks its own tools on the server (a dry "
            f"run; counts toward the demo's hourly cap). Caller: Alex Rivera (fictional). {used}/{LIVE_TICKETS_PER_SESSION} "
            "used this session." if live_ok else "The live model needs the deployed API; sample incidents work offline.")
    st.markdown(f'<div class="kb-foot">{esc(note)}</div>', unsafe_allow_html=True)
    if sent and text.strip():
        with st.spinner("The agent is working your incident..."):
            response = ask(text.strip()[:MAX_CUSTOM], [])
        st.session_state["live_used"] = used + 1
        if response.get("status") == "error":
            st.error(response.get("answer") or "The live agent is unavailable right now. Try a sample incident.")
            return
        st.session_state["ticket"] = new_live(text, response, next_number())
        st.rerun()


def next_number() -> str:
    n = st.session_state.get("ticket_seq", 1000) + 1
    st.session_state["ticket_seq"] = n
    return f"DEMO{n:07d}"


def act(label: str, key: str, fn, *args, **kwargs) -> None:
    if st.button(label, key=f"act-{key}", use_container_width=True, **kwargs):
        fn(*args)
        st.rerun()


def actions_panel(t) -> None:
    available = t.actions()
    caller = t.record["caller_id"]
    st.markdown('<div class="kb-label">Play the other people</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    slot = 0

    def next_col():
        nonlocal slot
        col = cols[slot % 4]
        slot += 1
        return col
    if "reply" in available:
        with next_col():
            act(f"{caller}: “It worked”", "worked", t.caller_reply, "It worked, thanks!", icon=":material/thumb_up:")
        with next_col():
            act(f"{caller}: “Still broken”", "broken", t.caller_reply, "No, it's still broken", icon=":material/thumb_down:")
    if "fast_forward" in available:
        with next_col():
            act("No reply: fast-forward 1 day", "ff", t.fast_forward, icon=":material/fast_forward:")
    if "approve" in available:
        for p in t.pending_approvals()[:1]:
            with next_col():
                act(f"Approve as {p['requester_name']} (requester)", "appr-req", t.approve, p["requested_by"], True,
                    icon=":material/person:")
            with next_col():
                act(f"Approve as {p['approver_name']} (owner)", "appr-ok", t.approve, p["approver"], True,
                    icon=":material/verified:")
            with next_col():
                act(f"Reject as {p['approver_name']}", "appr-no", t.approve, p["approver"], False, icon=":material/block:")
    if "redeliver" in available:
        with next_col():
            act("Send the last event again", "dup", t.redeliver, "webhook", icon=":material/content_copy:")
        with next_col():
            act("Queue delivers it twice", "dupq", t.redeliver, "queue", icon=":material/repeat:")
    with st.expander("Reply as the caller in your own words, or as someone else"):
        with st.form("reply", border=False):
            text = st.text_input("Reply", max_chars=MAX_CUSTOM, placeholder="e.g. it works now but Teams is slow")
            who = st.radio("From", [caller, "A coworker (not the caller)"], horizontal=True)
            if st.form_submit_button("Post comment") and text.strip():
                t.caller_reply(text, None if who == caller else "Morgan Fox")
                st.rerun()
        st.markdown('<div class="kb-foot">Replies are classified by the offline keyword classifier in this demo; the '
                    "ServiceNow integration uses Claude for the same step.</div>", unsafe_allow_html=True)


def section_ticket_desk() -> None:
    st.markdown('<p class="kb-lede">Create an incident and watch the agent work it: knowledge search, tool choice, '
                "identity and permission checks, simulated actions, verification, then resolve, wait for the caller, or "
                "escalate. Then play the caller or the approver and watch the ticket change.</p>", unsafe_allow_html=True)
    st.markdown('<div class="kb-banner"><span class="kb-badge sim">SIMULATED</span><div>Real agent and policy code, '
                "fake world. Account, Microsoft 365, Intune and directory actions run on a simulated company with "
                "fictional people. The ticket lives in your browser session, not in ServiceNow, and this page has no "
                "ServiceNow credentials (see Architecture for how it's isolated)."
                "</div></div>", unsafe_allow_html=True)
    t = st.session_state.get("ticket")
    if t is None:
        st.markdown('<div class="kb-journey">' + "".join(
            f'<div class="kb-step"><span class="n">{n}</span><div><b>{a}</b><span>{b}</span></div></div>'
            for n, (a, b) in enumerate([("Create a ticket", "Pick a sample incident or describe your own."),
                                        ("Watch the agent investigate", "Its searches, tool calls, checks and decisions."),
                                        ("See the outcome", "Verified simulated fix, follow-up with the caller, or escalation.")], 1))
            + "</div>", unsafe_allow_html=True)
        create_ticket_panel()
        return
    top_l, top_r = st.columns([5, 1])
    with top_r:
        if st.button("New incident", icon=":material/add:", use_container_width=True, key="new-ticket"):
            st.session_state.pop("ticket", None)
            st.rerun()
    with top_l:
        st.markdown(ticket_form(t), unsafe_allow_html=True)
    st.markdown(stages_html(t), unsafe_allow_html=True)
    actions_panel(t)
    left, right = st.columns([11, 9])
    with left:
        st.markdown('<div class="kb-label">Agent execution · newest first</div>', unsafe_allow_html=True)
        st.markdown(execution_html(t), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="kb-label">Ticket activity · newest first</div>', unsafe_allow_html=True)
        st.markdown(activity_html(t), unsafe_allow_html=True)


# ---------------------------------------------------------------- chat
def status_html(turn: dict) -> str:
    status, parts = turn.get("status"), []
    if status == "answered":
        n = len(turn.get("sources", []))
        parts.append(f'<span class="kb-badge ok">✓ Grounded in {n} cited passage{"s" if n != 1 else ""}</span>')
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
        parts.append("<span>Personal details masked before search</span>")
    if turn.get("latency_ms") is not None:
        parts.append(f"<span>{turn['latency_ms'] / 1000:.1f} s</span>")
    return f'<div class="kb-status">{"".join(parts)}</div>'


def show_sources(sources: list) -> None:
    """Each cited passage is inspectable: the exact text the answer was written from, and its article."""
    st.markdown(f'<div class="kb-src-title">Sources · {len(sources)} passage{"s" if len(sources) != 1 else ""} '
                "the answer was written from</div>", unsafe_allow_html=True)
    for s in sources:
        name = esc(s.get("source", ""))
        with st.expander(f"[{int(s.get('n', 0))}]  {s.get('title', '')}  ·  {s.get('source', '')}"):
            score = f" · similarity {float(s['score']):.2f}" if isinstance(s.get("score"), (int, float)) else ""
            st.markdown(f'<div class="kb-passage">{esc(passage(s.get("snippet", "")))}</div><div class="kb-passage-meta">'
                        f'From <a href="{REPO_DOCS}{name}" target="_blank">{name}</a> (open the full article){score}</div>',
                        unsafe_allow_html=True)


AGENT_STEP_NAMES = {"search_knowledge_base": "Searched the knowledge base", "find_similar_incidents": "Checked for similar incidents",
                    "send_fix_to_caller": "Sent the fix to the caller", "escalate_incident": "Escalated",
                    "verify_caller_identity": "Verified identity (MFA push)", "complete_and_confirm": "Sent the results to the caller",
                    "request_approval": "Requested approval", "link_to_outage": "Linked to the outage incident"}


def agent_html(plan: dict) -> str:
    decisions = {"resolved": ("ok", "Resolve", "Would resolve it: the fix was verified by reading the simulated system back."),
                 "awaiting_caller": ("ok", "Self-service", "Would send this to the caller and wait for them to confirm."),
                 "awaiting_approval": ("warn", "Approval", "Would wait for an approver, then run the approved action in code.")}
    badge, label, decision = decisions.get(plan.get("action"), ("warn", "Escalate", f"Would escalate to the {esc(plan.get('team'))} team"))
    steps = "".join(f"<li><b>{esc(AGENT_STEP_NAMES.get(s['tool'], s['tool'].replace('_', ' ').capitalize()))}</b>"
                    + (' <span class="kb-badge sim">SIMULATED</span>' if s.get("simulated") else "")
                    + f": {esc(s.get('detail', ''))}</li>" for s in plan.get("steps", []))
    return (f'<div class="kb-agent"><div class="kb-agent-head"><span class="kb-agent-title">Ticket agent · dry run</span>'
            f'<span class="kb-badge {badge}">{label}</span><span class="kb-agent-decision">{decision}</span></div><ol>{steps}</ol>'
            '<div class="kb-foot">Demo caller Alex Rivera (fictional). Nothing real is changed and nothing is written.</div></div>')


def show_turn(i: int, turn: dict) -> None:
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        with st.container(key=f"answer-{i}"):
            st.markdown(status_html(turn), unsafe_allow_html=True)
            st.markdown(answer_markdown(demo_text(turn["answer"])), unsafe_allow_html=True)
            if turn.get("sources"):
                show_sources(turn["sources"])
            if turn.get("agent"):
                st.markdown(agent_html(turn["agent"]), unsafe_allow_html=True)
                if st.button("Open this as a ticket in the AI ticket desk", key=f"open-{i}", icon=":material/open_in_new:"):
                    st.session_state["ticket"] = new_live(turn["asked"], turn, next_number())
                    go("AI ticket desk")
            if turn.get("ticket"):
                ok, message = turn["ticket"]
                (st.success if ok else st.error)(message)
            if turn.get("offer_ticket") and not PUBLIC_DEMO and not (turn.get("ticket") or (False,))[0]:
                if st.button("Open a ticket", key=f"ticket-{i}"):
                    turn["ticket"] = escalate(turn)
                    st.rerun()


def section_chat() -> None:
    st.markdown('<p class="kb-lede">Ask an IT question in plain words. Answers come only from 12 sample IT articles for a '
                "fictional company; open any source to read the exact passage used. Each answer also shows what the ticket "
                "agent would do with it, and you can open it as a ticket.</p>", unsafe_allow_html=True)
    if PUBLIC_DEMO and not (API_URL and API_KEY):
        st.error("The live chat isn't configured: ASSISTANT_API_URL and ASSISTANT_API_KEY must be set.")
        return
    turns = st.session_state.setdefault("turns", [])
    if not turns:
        cols = st.columns(3)
        for n, (icon, tag, example) in enumerate(EXAMPLES):
            if cols[n % 3].button(f"**{tag}**  \n{example}", key=f"ex-{n}", icon=icon, use_container_width=True):
                st.session_state["pending"] = example
                st.rerun()
    for i, turn in enumerate(turns):
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(html.escape(turn["asked"], quote=False))
        show_turn(i, turn)
    question = (st.chat_input("Describe your IT problem, e.g. my VPN keeps disconnecting", max_chars=MAX_QUESTION, key="chat")
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
    left, right = st.columns([3, 1])
    with left.expander(f"What the assistant knows ({len(articles())} articles)"):
        st.markdown('<ul class="kb-articles">' + "".join(
            f'<li><a href="{REPO_DOCS}{esc(name)}" target="_blank">{esc(title)}</a></li>' for name, title in articles())
            + "</ul>", unsafe_allow_html=True)
    if turns and right.button("Clear chat", use_container_width=True, icon=":material/refresh:"):
        st.session_state["turns"] = []
        st.rerun()
    if PUBLIC_DEMO:
        st.markdown('<div class="kb-foot">Please don\'t enter personal or real company information. Questions are capped '
                    "per visitor and per day on the server.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- capabilities
def section_capabilities() -> None:
    st.markdown('<p class="kb-lede">The agent\'s 12 service-desk workflows, all through one framework: allowlisted tools, '
                "server-side identity checks, scoped permissions, single-use approvals, safe retries and a redacted audit trail.</p>",
                unsafe_allow_html=True)
    kinds = {"real": ("real", "Real integration"), "mixed": ("mixed", "Real + simulated"), "simulated": ("sim", "Simulated backend")}
    view = st.segmented_control("Show", ["All", "Real integration", "Real + simulated", "Simulated backend"], default="All",
                                key="capfilter", label_visibility="collapsed") or "All"
    cards = []
    for c in capabilities():
        cls, label = kinds[c["kind"]]
        if view not in ("All", label):
            continue
        tools = "".join(f"<code>{esc(x)}</code>" for x in c["tools"])
        cards.append(f'<div class="kb-cap"><div class="kb-cap-h"><div><div class="kb-cap-n">WORKFLOW {c["n"]}</div>'
                     f'<div class="kb-cap-name">{esc(c["name"])}</div></div><span class="kb-badge {cls}">{esc(label)}</span></div>'
                     f'<p>{esc(c["does"])}</p><div class="gate"><b>Integration:</b> {esc(c["integration"])}</div>'
                     f'<div class="gate"><b>Gates:</b> {esc(c["gates"])}</div>' + (f'<div class="kb-tools">{tools}</div>' if tools else "")
                     + "</div>")
    st.markdown(f'<div class="kb-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="kb-foot" style="margin-top:.8rem">Real: ServiceNow, the knowledge base (RAG + Bedrock Guardrails) '
                "and Claude. Simulated until an authorized test tenant is connected: Microsoft Entra ID, Intune, Microsoft 365, "
                "Exchange Online, monitoring, device logs and HR.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- ServiceNow evidence
SN_TIMELINE = [("20:16:15", "Incident created", "New · Caller Alex Rivera · “I'm locked out of my account”"),
               ("20:16:37", "Agent works it (22 s)", "Knowledge base search, similar-incident check, MFA push, unlock, "
                "read-back. Directory steps marked SIMULATED."),
               ("20:16:37", "Resolved · Solution provided", "Close notes: “verified fix (unlock_account) [SIMULATED backends: "
                "no real change]”. The caller is told they can reply to reopen."),
               ("20:27:40", "Caller replies “Still can't sign in”", "Live testing found the reply was missed: this release "
                "labels comments “(Comments)”, not “(Additional comments)”. Fixed, with a regression test from the real text."),
               ("After the fix", "Reopened for a technician", "A new reply from the caller set the incident back to In Progress "
                "for the Service Desk, with the reply quoted in the work note.")]
SN_WORK_NOTE = """[AI Agent - IT KB Assistant]
Action: completed the actions below and verified each one by reading the SIMULATED backend back after the change; resolved without waiting for the caller.
Resolved by the AI agent (verified fix).

Actions executed:
- unlock_account (workflow 1, SIMULATED)

Steps:
- search_knowledge_base: answered from password-reset.md
- get_account_status: ok: locked [SIMULATED]
- find_similar_incidents: 0 similar in the last 60 min
- verify_caller_identity: ok: caller approved the MFA push on their registered device [SIMULATED]
- unlock_account: ok: account unlocked and verified [SIMULATED]

Why no caller confirmation: each action read the system back after the change and confirmed the end state.
If the caller replies that it still isn't working, the incident reopens for the Service Desk."""


def section_servicenow() -> None:
    st.markdown('<p class="kb-lede">The private integration on a real ServiceNow developer instance, Oct 19 2026 (instance '
                "times). ServiceNow, the signed webhook, the queue and the agent are real; the account action ran on the "
                "simulated directory, and the ticket says so. The public demo can't reach this instance.</p>", unsafe_allow_html=True)
    st.markdown('<div class="kb-ba"><div class="kb-card"><span class="kb-badge">Before · 20:16:15</span><dl>'
                "<dt>Incident</dt><dd>INC0010053</dd><dt>Caller</dt><dd>Alex Rivera</dd><dt>Description</dt>"
                "<dd>I'm locked out of my account</dd><dt>State</dt><dd>New</dd><dt>Assigned to</dt><dd>(nobody)</dd></dl></div>"
                '<div class="arrow">→</div><div class="kb-card"><span class="kb-badge ok">After · 20:16:37 (22 s)</span><dl>'
                "<dt>State</dt><dd>Resolved</dd><dt>Resolution code</dt><dd>Solution provided</dd><dt>Group</dt><dd>Service Desk</dd>"
                "<dt>Work note</dt><dd>Every step, SIMULATED ones labeled</dd><dt>Caller</dt><dd>Told it's a demo; can reply to reopen</dd>"
                "</dl></div></div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="kb-label">Incident timeline · INC0010053</div><ol class="kb-timeline">' + "".join(
            f'<li><div class="when">{esc(w)}</div><div class="state">{esc(s)}</div><div class="why">{esc(d)}</div></li>'
            for w, s, d in SN_TIMELINE) + "</ol>", unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="kb-label">Agent work note (copied from the ticket)</div><pre class="kb-out">{esc(SN_WORK_NOTE)}</pre>',
                    unsafe_allow_html=True)
    st.markdown('<div class="kb-label">Screenshots from the instance</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.image(os.path.join(ROOT, "assets", "servicenow-verified-fix-resolved.png"),
            caption="INC0010053: verified fix, resolved as Solution provided", use_container_width=True)
    b.image(os.path.join(ROOT, "assets", "servicenow-no-response-3-strike.png"),
            caption="INC0010050: no reply, two reminders, closed as No resolution provided", use_container_width=True)
    st.markdown('<div class="kb-foot">For INC0010050 the follow-up job was triggered by hand three times in a row for the test; '
                "on its schedule it runs hourly and sends each reminder after 24 hours without a reply.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- architecture
def node(title: str, detail: str, kind: str) -> str:
    return f'<div class="kb-node {kind}"><b>{esc(title)}</b><span>{esc(detail)}</span></div>'


def flow(*nodes: str) -> str:
    return '<div class="kb-flow">' + '<div class="kb-arrow">→</div>'.join(nodes) + "</div>"


def section_architecture() -> None:
    st.markdown('<p class="kb-lede">Two separate paths. The public demo can only read the knowledge base and run dry runs; the '
                "private ServiceNow path is the only one that can write, through a least-privilege service account.</p>",
                unsafe_allow_html=True)
    demo = flow(node("Browser", "This page", "ui"), node("Streamlit Cloud", "Demo key held server-side; ticket desk in memory", "ui"),
                node("API Gateway", "/demo/ask · own key · 1 req/s", "aws"),
                node("Lambda", "Per-visitor and daily caps (DynamoDB), dry-run agent", "aws"),
                node("Bedrock Guardrails", "Input: PII masking, denied topics, prompt attacks", "ai"),
                node("Retrieval", "Voyage embeddings · Amazon S3 Vectors · top 5", "ai"),
                node("Claude", "Answers only from the passages; picks agent tools", "ai"),
                node("Guardrails", "Output: contextual grounding", "ai"))
    private = flow(node("ServiceNow", "Business Rules on insert, reply, approval", "sn"),
                   node("API Gateway", "HMAC-signed webhook · 5-min replay window", "aws"),
                   node("Lambda + SQS", "Idempotency claim (DynamoDB) · queue · DLQ alarm", "aws"),
                   node("Ticket agent", "Claude tool use · code-enforced rules", "ai"),
                   node("Enterprise framework", "Registry → policy → approvals → executor → audit", "ai"),
                   node("Backends", "ServiceNow Table API (real) · Entra/Intune/M365 (simulated)", "sim"))
    follow = flow(node("EventBridge", "Hourly", "aws"), node("Follow-up Lambda", "Reminders day 1 and 2, no-response close day 3", "aws"),
                  node("ServiceNow", "Resolve or hand to a technician", "sn"))
    st.markdown(f'<div class="kb-lane"><div class="kb-lane-h">Public demo · read-only <span class="kb-badge info">this page</span>'
                f'</div>{demo}</div><div class="kb-lane"><div class="kb-lane-h">ServiceNow integration · private '
                f'<span class="kb-badge">not reachable from the demo</span></div>{private}<div style="height:.6rem"></div>{follow}</div>',
                unsafe_allow_html=True)
    proof = [("This page", "No ServiceNow URL, password or Table API client. The ticket desk writes to an in-memory record "
              "(SandboxServiceNow) and its toolkit refuses any tool not marked simulated."),
             ("Demo Lambda permissions", "Can read the vector index, call the model and guardrail, read its own key "
              "and the model and embedding API keys, and update its usage counter. No access to the ServiceNow password, queue, webhook secret or state table."),
             ("Demo key", "Rejected by /ask, /escalate and /servicenow/events (verified by the live security check)."),
             ("Server agent", "Runs with a dry-run desk and simulated-only tools; returns its plan and tool calls, writes nothing."),
             ("Spend", "Per-visitor (15/hour) and daily (100) caps checked before any paid call; 5 live tickets per page session.")]
    rows = [("Credentials", "The model never holds credentials; tools use server-side clients. Temporary passwords never "
             "appear in tickets, logs or responses."), ("Identity", "Set only by a server-side MFA check; the model can't claim it."),
            ("Scope", "Account, device and mailbox tools act only on the caller's own records."),
            ("Approvals", "Bound to the exact request, single use, never by the requester; approved actions run in code."),
            ("Idempotency", "Every write is claimed before it runs; a write that timed out is never retried blindly.")]
    st.markdown('<div class="kb-label">Why a visitor can\'t change a real system</div><div class="kb-tablewrap"><table class="kb-table">'
                + "".join(f"<tr><td class=num>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in proof) + "</table></div>"
                '<div class="kb-label">Security boundaries in the agent</div><div class="kb-tablewrap"><table class="kb-table">'
                + "".join(f"<tr><td class=num>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in rows) + "</table></div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- evaluation
def bar(value: float, base: bool = False) -> str:
    return f'<div class="kb-bar{" base" if base else ""}"><i style="width:{max(0.0, min(1.0, value)) * 100:.0f}%"></i></div>'


def section_evaluation() -> None:
    e = evals()
    h = e["hit1"]
    st.markdown('<p class="kb-lede">Deterministic checks, no LLM judge. Numbers are read from the committed eval output '
                f'(<a href="{REPO}/blob/main/{EVAL_RUN}" target="_blank">raw JSON</a>) and the README, with the set size and '
                "conditions next to each.</p>", unsafe_allow_html=True)
    labels = {"answer": "Answer: right facts and right article cited", "no_answer": "Not in the docs: says so",
              "blocked": "Denied topics and prompt attacks blocked", "pii": "Personal data masked before the model",
              "poisoned": "Injected instructions in a retrieved chunk ignored", "followup": "Multi-turn follow-ups"}
    rows = "".join(f'<tr><td>{esc(labels.get(k, k))}</td><td class="num">{v["passed"]} / {v["total"]}</td></tr>'
                   for k, v in e["by_type"].items())
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f'<div class="kb-label">End to end · {e["passed"]} / {e["total"]} passed</div><div class="kb-tablewrap">'
                    f'<table class="kb-table"><tr><th>Case type</th><th>Passed</th></tr>{rows}</table></div>'
                    f'<div class="kb-foot" style="margin-top:.5rem">42 cases × {e["runs"]} runs = {e["total"]}, deployed stack in '
                    "us-west-2 (S3 Vectors + Voyage voyage-4, Claude Haiku 4.5, Bedrock Guardrail v5), Oct 18 2026. The 3 failures "
                    "are one question (“People in my meeting say they can't hear me”): a correct answer the relevance check "
                    "withheld. Left visible rather than tuned away.</div>", unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="kb-label">Retrieval · right article ranked first</div><div class="kb-card">'
                    f'<div class="kb-step-h">Semantic (S3 Vectors + Voyage) · {h["semantic"]:.0%}</div>{bar(h["semantic"])}'
                    f'<div class="kb-step-h" style="margin-top:.6rem">BM25 keyword baseline · {h["bm25"]:.0%}</div>{bar(h["bm25"], True)}'
                    f'<div class="kb-foot" style="margin-top:.6rem">{h["n"]} questions. On the {h["n_paraphrased"]} paraphrased '
                    f'ones: {h["semantic_paraphrased"]:.0%} vs {h["bm25_paraphrased"]:.0%}.</div></div>'
                    f'<div class="kb-label">Latency and cost</div><div class="kb-card"><div class="kb-step-h">{e["avg_latency_s"]} s '
                    f'average · {e["p95_latency_s"]} s p95</div><div class="kb-foot">About ${e["cost_per_question"]:.4f} model cost '
                    "per question, measured over the same eval runs.</div></div>", unsafe_allow_html=True)
    other = [("ServiceNow incidents (suggest mode)", "35 / 36", "36 incidents from the golden set on a live developer instance, "
              "Oct 19 2026; every incident got its note"),
             ("Input guardrail check", "42 / 42", "All golden-set questions, after fixing over-blocking the evals found"),
             ("Public-demo security checks", "11 / 11", "Run against the live API, Oct 19 2026"),
             ("Automated tests", "all passing in CI", "Offline unit, integration and UI tests: agent framework, follow-up "
              "policy, the ticket desk and this page")]
    st.markdown('<div class="kb-label">Other measured checks</div><div class="kb-tablewrap"><table class="kb-table">'
                "<tr><th>Check</th><th>Result</th><th>Conditions</th></tr>" + "".join(
                    f'<tr><td>{esc(a)}</td><td class="num">{esc(b)}</td><td>{esc(c)}</td></tr>' for a, b, c in other)
                + "</table></div>", unsafe_allow_html=True)


def footer() -> None:
    links = " · ".join(x for x in [f'<a href="{REPO}" target="_blank">Code, architecture and evals on GitHub</a>',
                                   link(GITHUB_PROFILE, "GitHub profile"),
                                   link(PORTFOLIO_URL, "Portfolio"), link(LINKEDIN_URL, "LinkedIn")] if x)
    st.markdown(f'<div class="kb-footer"><span>Built by Sagar Modha. Fictional company and people; this demo cannot change any '
                f"real system.</span><span>{links}</span></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- page
st.set_page_config(page_title="IT Service Desk Agent · Sagar Modha", page_icon=":material/support_agent:",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
if not PUBLIC_DEMO:
    with st.sidebar:
        st.markdown("### Setup")
        if API_URL:
            st.write("Mode: **deployed API**")
            st.code(API_URL, language=None)
        else:
            st.write("Mode: **local**")
            st.write(f"Retriever: `{setting({}, 'RETRIEVER')}`" +
                     (f" ({setting({}, 'EMBEDDER')})" if setting({}, "RETRIEVER") == "local" else ""))
            st.write(f"Model: `{setting({}, 'LLM_PROVIDER')}`")
            st.write(f"Guardrail: `{setting({}, 'GUARDRAIL')}`")

header()
stats()
if "nav_target" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("nav_target")
section = st.segmented_control("Section", NAV, default=NAV[0], key="nav", label_visibility="collapsed") or NAV[0]
{"AI ticket desk": section_ticket_desk, "Ask the assistant": section_chat, "12 capabilities": section_capabilities,
 "ServiceNow, live": section_servicenow, "Architecture": section_architecture, "Evaluation": section_evaluation}[section]()
footer()
