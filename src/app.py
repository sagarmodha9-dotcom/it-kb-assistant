"""AWS Lambda handlers, called by API Gateway. Both routes need the x-api-key header.

POST /ask       {"question": "...", "history": [...]}    -> cited answer, or an offer to open a ticket
                (history: optional last turns [{"question", "answer"}] for follow-up questions)
POST /escalate  {"question": "...", "answer": "...",     -> opens a ticket in the AI Ticket Triage API
                 "sources": ["vpn-troubleshooting.md"]}
"""
import base64
import hmac
import json
import logging
import os
import re

from assistant import MAX_QUESTION
from factory import build_assistant, build_guardrail
from tickets import TicketClient, TicketError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_cache = {}
SOURCE_NAME = re.compile(r"[\w.-]{1,100}\.md")  # built once per Lambda container and reused on warm starts


def _ssm_secret(name: str) -> str:
    import boto3
    return boto3.client("ssm").get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]


def _api_key() -> str | None:
    if "api_key" not in _cache:
        try:
            _cache["api_key"] = _ssm_secret(os.environ["API_KEY_PARAM"])
        except Exception:
            logger.exception("Could not read the API key parameter")
            return None  # not cached, so it's retried once the parameter exists
    return _cache["api_key"]


def _assistant():
    if "assistant" not in _cache:
        settings = {}
        if "anthropic" in os.environ.get("LLM_PROVIDER", ""):
            try:
                settings["ANTHROPIC_API_KEY"] = _ssm_secret(os.environ["ANTHROPIC_KEY_PARAM"])
            except Exception:
                logger.exception("No Anthropic key; using Bedrock only")
                settings["LLM_PROVIDER"] = "bedrock"
        _cache["assistant"] = build_assistant(settings)
    return _cache["assistant"]


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def _parse(event: dict) -> dict:
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("body must be a JSON object")
    return data


def _check(event: dict):
    """Shared auth + body parsing. Returns (data, None) or (None, error_response)."""
    expected = _api_key()
    if expected is None:
        return None, _response(503, {"error": "API key is not configured on the server."})
    sent = (event.get("headers") or {}).get("x-api-key", "")
    if not hmac.compare_digest(sent.encode(), expected.encode()):
        return None, _response(401, {"error": "Missing or invalid x-api-key header."})
    try:
        data = _parse(event)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None, _response(400, {"error": "Request body must be valid JSON."})
    question = str(data.get("question", "")).strip()
    if not question:
        return None, _response(400, {"error": "'question' is required."})
    if len(question) > MAX_QUESTION:
        return None, _response(413, {"error": f"'question' is limited to {MAX_QUESTION} characters."})
    return data, None


def ask(event, context):
    data, error = _check(event)
    if error:
        return error
    time_left = (lambda: context.get_remaining_time_in_millis() / 1000 - 2) if context else None
    result = _assistant().ask(data["question"], history=data.get("history"), time_left=time_left)
    # One structured log line per question (no question text), easy to query in Logs Insights.
    logger.info(json.dumps({"event": "ask", "status": result["status"], "provider": result["provider"],
                            "sources": [s["source"] for s in result["sources"]],
                            "guardrail": result["guardrail"], "usage": result["usage"],
                            "latency_ms": result["latency_ms"]}))
    return _response(200, result)


def escalate(event, context):
    data, error = _check(event)
    if error:
        return error
    # The client sends the question back, so mask it again here instead of trusting it.
    if "guardrail" not in _cache:
        _cache["guardrail"] = build_guardrail({})
    guarded = _cache["guardrail"].check_input(data["question"])
    if guarded.blocked:
        return _response(400, {"error": "This request can't be turned into a ticket."})
    # Everything the client sends is untrusted: mask the answer too, and only accept article names.
    answer = _cache["guardrail"].check_input(str(data.get("answer", ""))[:1000]) if data.get("answer") else None
    raw_sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    sources = [s for s in raw_sources if isinstance(s, str) and SOURCE_NAME.fullmatch(s)][:10]
    try:
        ticket = TicketClient(os.environ.get("TRIAGE_API_URL", "")).create(
            guarded.text, answer.text if answer and not answer.blocked else "", sources)
    except TicketError as e:
        logger.warning("Escalation failed: %s", e)
        return _response(502, {"error": "Could not create the ticket. Please contact the Service Desk."})
    logger.info(json.dumps({"event": "escalate", "ticket_id": ticket["ticket_id"], "priority": ticket["priority"]}))
    return _response(201, ticket)
