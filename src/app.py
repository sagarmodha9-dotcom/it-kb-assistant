"""AWS Lambda handlers, called by API Gateway. Both routes need the x-api-key header.

POST /ask       {"question": "...", "history": [...]}    -> cited answer, or an offer to open a ticket
                (history: optional last turns [{"question", "answer"}] for follow-up questions)
POST /escalate  {"question": "...", "answer": "...",     -> opens a ticket in the AI Ticket Triage API
                 "sources": ["vpn-troubleshooting.md"]}

Public recruiter demo (deployed only when PublicDemo=true):
POST /demo/ask  {"question": "...", "history": [...]}    separate API key; answers only, no tickets or
                                                          ServiceNow; daily and per-visitor caps enforced here

ServiceNow integration (see servicenow.py and agent.py):
POST /servicenow/events  {"sys_id": "...", "number": "INC0010001"}  called by a ServiceNow Business Rule
                         and signed (x-timestamp + x-signature HMAC-SHA256) -> 202, queued.
                         {"event": "caller_reply", "version": "7", ...} when the caller answers a fix.
SQS worker               SN_MODE=agent (default): the ticket agent sends a verified fix and waits for the
                         caller, or escalates; on caller_reply it resolves only if the caller confirms.
                         SN_MODE=suggest: posts a suggested-answer work note only.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time

from assistant import MAX_QUESTION
from factory import build_assistant, build_guardrail
from agent import (AGENT_HEADER, AgentUnavailable, DryRunDesk, ServiceNowDesk, TicketAgent, agent_plan,
                   handle_caller_reply)
from servicenow import (NOTE_HEADER, NUMBER, SYS_ID, RetryLater, ServiceNowClient, ServiceNowError,
                        incident_question, process_incident)
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
        if os.environ.get("EMBEDDER") == "voyage":
            settings["VOYAGE_API_KEY"] = _ssm_secret(os.environ["VOYAGE_KEY_PARAM"])
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
    try:
        assistant = _assistant()
    except Exception:
        logger.exception("Assistant setup failed")
        return _response(503, {"error": "The assistant is not configured correctly. Please contact the Service Desk."})
    result = assistant.ask(data["question"], history=data.get("history"), time_left=time_left)
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


# ---------------------------------------------------------------- ServiceNow integration
IDEMPOTENCY_DAYS = 7
SIGNATURE_MAX_AGE = 300  # seconds


def _secret(cache_key: str, env_name: str) -> str | None:
    if cache_key not in _cache:
        try:
            _cache[cache_key] = _ssm_secret(os.environ[env_name])
            _cache[cache_key + "_at"] = time.time()
        except Exception:
            logger.exception("Could not read %s", env_name)
            return None
    return _cache[cache_key]


def _aws(service: str):
    if service not in _cache:
        import boto3
        _cache[service] = boto3.client(service)
    return _cache[service]


def _servicenow() -> ServiceNowClient:
    if "servicenow" not in _cache:
        password = _secret("sn_password", "SN_PASSWORD_PARAM")
        _cache["servicenow"] = ServiceNowClient(os.environ.get("SN_INSTANCE_URL", ""),
                                                os.environ.get("SN_USER", ""), password or "", timeout=25)
    return _cache["servicenow"]


def _raw_body(event: dict) -> str:
    body = event.get("body") or ""
    return base64.b64decode(body).decode("utf-8") if event.get("isBase64Encoded") else body


def verify_signature(secret_b64: str, timestamp: str, body: str, signature: str, now: float | None = None) -> bool:
    """HMAC-SHA256 over "<timestamp>.<body>" with the shared key (base64). The timestamp must be
    within 5 minutes, so a captured request can't be replayed later."""
    try:
        sent_at = int(timestamp)
        key = base64.b64decode(secret_b64, validate=True)
    except (TypeError, ValueError):
        return False
    if abs((now or time.time()) - sent_at) > SIGNATURE_MAX_AGE:
        return False
    expected = base64.b64encode(hmac.new(key, f"{timestamp}.{body}".encode("utf-8"), hashlib.sha256).digest())
    return hmac.compare_digest(expected, (signature or "").encode())


def servicenow_event(event, context):
    """Webhook from the ServiceNow Business Rule. Fast: verify signature, de-duplicate, queue, 202."""
    secret = _secret("sn_webhook_secret", "SN_WEBHOOK_SECRET_PARAM")
    if secret is None:
        return _response(503, {"error": "Webhook secret is not configured on the server."})
    headers = event.get("headers") or {}
    try:
        body = _raw_body(event)
    except (ValueError, UnicodeDecodeError):
        return _response(400, {"error": "Request body must be valid JSON."})
    signed = lambda key: verify_signature(key, headers.get("x-timestamp", ""), body,  # noqa: E731
                                          headers.get("x-signature", ""))
    if not signed(secret) and time.time() - _cache.get("sn_webhook_secret_at", 0) > 60:
        # The key may have been rotated (servicenow_setup.py): re-read it, at most once a minute.
        _cache.pop("sn_webhook_secret", None)
        secret = _secret("sn_webhook_secret", "SN_WEBHOOK_SECRET_PARAM") or ""
    if not secret or not signed(secret):
        return _response(401, {"error": "Missing, invalid, or expired request signature."})
    try:
        data = json.loads(body or "{}")
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        return _response(400, {"error": "Request body must be valid JSON."})
    sys_id = str(data.get("sys_id", ""))
    number = str(data.get("number", ""))
    kind = str(data.get("event", "new"))
    version = str(data.get("version", ""))
    if not SYS_ID.fullmatch(sys_id) or (number and not NUMBER.fullmatch(number)):
        return _response(400, {"error": "'sys_id' must be a 32-character ServiceNow sys_id."})
    if kind not in ("new", "caller_reply", "approval") or (kind != "new" and not re.fullmatch(r"\d{1,10}", version)):
        return _response(400, {"error": "Unknown event."})
    extra = {}
    if kind == "approval":
        # who decided and what, as signed by the Business Rule at the moment of the change
        approver, decision = str(data.get("approver", "")), str(data.get("decision", ""))
        if decision not in ("approved", "rejected") or not re.fullmatch(r"[\w.@-]{1,100}", approver):
            return _response(400, {"error": "Invalid approval event."})
        extra = {"approver": approver, "decision": decision, "version": version}

    # Each event is handled once: one agent run per new incident, one per caller reply or approval decision
    # (by record version).
    key = sys_id if kind == "new" else f"{sys_id}#{kind}#{version}"
    table = os.environ["SN_IDEMPOTENCY_TABLE"]
    try:
        _aws("dynamodb").put_item(
            TableName=table,
            Item={"sys_id": {"S": key}, "number": {"S": number or "-"},
                  "expires_at": {"N": str(int(time.time()) + IDEMPOTENCY_DAYS * 86400)}},
            ConditionExpression="attribute_not_exists(sys_id)")
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return _response(200, {"status": "duplicate", "number": number})
        logger.exception("Idempotency check failed")
        return _response(503, {"error": "Temporarily unavailable."})
    try:
        _aws("sqs").send_message(QueueUrl=os.environ["SN_QUEUE_URL"],
                                 MessageBody=json.dumps({"sys_id": sys_id, "number": number, "event": kind, **extra}))
    except Exception:
        logger.exception("Could not queue %s", number or sys_id)
        try:  # undo the claim so ServiceNow's retry (or a re-run) can succeed
            _aws("dynamodb").delete_item(TableName=table, Key={"sys_id": {"S": key}})
        except Exception:
            logger.exception("Could not release idempotency claim for %s", sys_id)
        return _response(503, {"error": "Temporarily unavailable."})
    logger.info(json.dumps({"event": "servicenow_queued", "number": number, "kind": kind}))
    return _response(202, {"status": "queued", "number": number})


def _enterprise_store():
    from enterprise.framework import DynamoStore, MemoryStore
    table = os.environ.get("ENTERPRISE_TABLE")
    return DynamoStore(_aws("dynamodb"), table) if table else MemoryStore()


def _enterprise_on() -> bool:
    return os.environ.get("SN_ENTERPRISE", "false") == "true"


def _desk(client: ServiceNowClient) -> ServiceNowDesk:
    if "desk" not in _cache or _cache["desk"].client is not client:
        _cache["desk"] = ServiceNowDesk(client, close_code=os.environ.get("SN_CLOSE_CODE", "Solution provided"))
    return _cache["desk"]


def work_incident(client, assistant, sys_id: str, kind: str = "new", time_left=None,
                  final_attempt: bool = True, event: dict | None = None) -> dict:
    """Agent mode. New incident: the agent sends a verified fix or escalates. Caller reply: resolve if
    the caller confirms, otherwise back to a technician."""
    incident = client.get_incident(sys_id)
    if incident is None:
        return {"sys_id": sys_id, "outcome": "not_found"}
    incident = {**incident, "sys_id": sys_id}  # the id we verified, not whatever the record echoes
    number = incident.get("number")
    if str(incident.get("active", "true")).lower() == "false":
        return {"sys_id": sys_id, "number": number, "outcome": "inactive"}
    desk = _desk(client)
    if kind == "approval":
        from enterprise.runtime import handle_approval_decision
        event = event or {}
        decision, approver = event.get("decision", ""), event.get("approver", "")
        current = str(incident.get("approval") or "").lower()
        # Act only on the signed decision, and only if the incident still shows it.
        if decision not in ("approved", "rejected") or current != decision or not _enterprise_on():
            return {"sys_id": sys_id, "number": number, "kind": kind, "outcome": "ignored"}
        return {"sys_id": sys_id, "number": number, "kind": kind,
                **handle_approval_decision(desk, incident, approver, decision == "approved", _enterprise_store())}
    if kind == "caller_reply":
        try:
            handled = handle_caller_reply(assistant.llm, assistant.guardrail, desk, incident, final_attempt)
        except AgentUnavailable as error:
            raise RetryLater(f"reply on {number}: {error}") from error
        return {"sys_id": sys_id, "number": number, "kind": kind, **handled}
    notes = str(incident.get("work_notes") or "")
    if AGENT_HEADER in notes or NOTE_HEADER in notes:
        return {"sys_id": sys_id, "number": number, "outcome": "already_handled"}
    if str(incident.get("assigned_to") or "").strip():
        return {"sys_id": sys_id, "number": number, "outcome": "technician_assigned"}
    question = incident_question(incident)
    if not question:
        return {"sys_id": sys_id, "number": number, "outcome": "empty"}
    toolkit = None
    if _enterprise_on():
        from enterprise.runtime import build_toolkit, caller_id
        toolkit = build_toolkit(str(number), caller_id(incident), store=_enterprise_store())
    try:
        agent = TicketAgent(assistant.llm, assistant, desk, toolkit=toolkit)
        outcome = agent.run(incident, question, time_left=time_left, raise_on_model_error=not final_attempt)
    except AgentUnavailable as error:
        raise RetryLater(f"agent unavailable on {number}: {error}") from error
    return {"sys_id": sys_id, "number": number, "outcome": outcome["action"], "team": outcome.get("team"),
            "priority": outcome.get("priority"), "steps": [s["tool"] for s in outcome["steps"]],
            "executed": [e["tool"] for e in (toolkit.ctx.executed if toolkit else [])],
            "provider": outcome.get("provider"), "usage": outcome.get("usage")}


def servicenow_worker(event, context):
    """SQS consumer. Reports per-message failures so only failed incidents are retried."""
    max_receives = int(os.environ.get("SN_MAX_RECEIVES", "3"))
    time_left = (lambda: context.get_remaining_time_in_millis() / 1000 - 3) if context else None
    failures = []
    for record in event.get("Records", []):
        try:
            message = json.loads(record["body"])
            sys_id = str(message["sys_id"])
            kind = str(message.get("event", "new"))
        except (ValueError, KeyError, TypeError):
            logger.error("Dropping malformed message %s", record.get("messageId"))
            continue  # retrying a malformed message would never succeed
        receives = int(record.get("attributes", {}).get("ApproximateReceiveCount", "1"))
        try:
            final = receives >= max_receives
            if os.environ.get("SN_MODE", "agent") == "agent":
                summary = work_incident(_servicenow(), _assistant(), sys_id, kind, time_left=time_left,
                                        final_attempt=final, event=message)
                logger.info(json.dumps({"event": "servicenow_agent", **summary}))
            elif kind == "new":
                summary = process_incident(_servicenow(), _assistant(), sys_id, time_left=time_left,
                                           final_attempt=final)
                logger.info(json.dumps({"event": "servicenow_note", **summary}))
        except (RetryLater, ServiceNowError) as error:
            if getattr(error, "status", None) == 401:  # password rotated: read it again next time
                _cache.pop("servicenow", None)
                _cache.pop("sn_password", None)
            logger.warning("Will retry %s (attempt %s): %s", sys_id, receives, error)
            failures.append({"itemIdentifier": record["messageId"]})
        except Exception:
            logger.exception("Unexpected error on %s (attempt %s)", sys_id, receives)
            failures.append({"itemIdentifier": record["messageId"]})
    return {"batchItemFailures": failures}


# ---------------------------------------------------------------- public recruiter demo
DEMO_MAX_HISTORY = 2
DEMO_AGENT_MIN_SECONDS = 18   # skip the agent plan unless there's clearly time for it
DEMO_AGENT_STEP_SECONDS = 17  # one model call can take up to 15 s
DEMO_ENTERPRISE_STEP_SECONDS = 10  # the demo's model client times out after 8 s


def _demo_limit(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _claim(key: str, cap: int, ttl_seconds: int) -> bool:
    """Atomically count one request against `key`; False once the cap is reached (never goes over)."""
    try:
        _aws("dynamodb").update_item(
            TableName=os.environ["DEMO_TABLE"], Key={"pk": {"S": key}},
            UpdateExpression="ADD used :one SET expires_at = if_not_exists(expires_at, :exp)",
            ConditionExpression="attribute_not_exists(used) OR used < :cap",
            ExpressionAttributeValues={":one": {"N": "1"}, ":cap": {"N": str(cap)},
                                       ":exp": {"N": str(int(time.time()) + ttl_seconds)}})
        return True
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def demo_ask(event, context):
    """Public demo: same assistant, stricter limits, nothing that writes anywhere except the usage counters."""
    expected = _secret("demo_api_key", "DEMO_API_KEY_PARAM")
    if expected is None:
        return _response(503, {"error": "Demo is not configured."})
    headers = event.get("headers") or {}
    if not hmac.compare_digest(headers.get("x-api-key", "").encode(), expected.encode()):
        return _response(401, {"error": "Missing or invalid x-api-key header."})
    try:
        data = _parse(event)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return _response(400, {"error": "Request body must be valid JSON."})
    question = " ".join(str(data.get("question", "")).split())
    max_len = _demo_limit("DEMO_MAX_QUESTION", 300)
    if not question:
        return _response(400, {"error": "'question' is required."})
    if len(question) > max_len:
        return _response(413, {"error": f"Please keep questions under {max_len} characters."})

    # Usage caps, checked before any paid call. The visitor id is a salted hash, never the raw IP.
    client = str(headers.get("x-demo-client", ""))[:200] or "unknown"
    salt = _secret("demo_api_key", "DEMO_API_KEY_PARAM") or ""
    visitor = hashlib.sha256(f"{salt}:{client}".encode()).hexdigest()[:24]
    now = time.gmtime()
    try:
        if not _claim(f"visitor#{visitor}#{time.strftime('%Y%m%d%H', now)}",
                      _demo_limit("DEMO_VISITOR_HOURLY_CAP", 15), 2 * 3600):
            return _response(429, {"error": "You've reached the demo limit for this hour. Please try again later."})
        if not _claim(f"day#{time.strftime('%Y-%m-%d', now)}", _demo_limit("DEMO_DAILY_CAP", 300), 3 * 86400):
            return _response(429, {"error": "The demo has reached today's usage limit. Please try again tomorrow."})
    except Exception:
        logger.exception("Demo quota check failed")
        return _response(503, {"error": "The demo is temporarily unavailable."})

    started = time.time()
    history = data.get("history") if isinstance(data.get("history"), list) else []
    time_left = (lambda: context.get_remaining_time_in_millis() / 1000 - 2) if context else None
    try:
        assistant = _assistant()
    except Exception:
        logger.exception("Assistant setup failed")
        return _response(503, {"error": "The demo is temporarily unavailable."})
    result = assistant.ask(question, history=history[-DEMO_MAX_HISTORY:], time_left=time_left)
    result["offer_ticket"] = False  # no tickets, escalation, or ServiceNow writes in the public demo
    plan = None
    enough_time = time_left is None or time_left() > DEMO_AGENT_MIN_SECONDS
    if result["status"] in ("answered", "no_answer", "blocked") and os.environ.get("DEMO_AGENT", "true") == "true" \
            and enough_time:
        # Show what the ticket agent would do with this as a new incident. DryRunDesk writes nothing.
        try:
            incident = {"sys_id": "0" * 32, "number": "DEMO", "short_description": result.get("question") or question,
                        "priority": "4 - Low"}
            # Each step needs time for one full model call (15 s client timeout), or the plan is skipped:
            # the visitor always gets the answer, never a gateway timeout.
            if os.environ.get("DEMO_ENTERPRISE", "true") == "true":
                # The 12 workflows against a fictional company: simulated tools only, state per request,
                # a short model timeout so the page always answers in time.
                from enterprise.runtime import demo_llm, demo_toolkit
                agent = TicketAgent(demo_llm(assistant.llm), assistant, DryRunDesk(), max_steps=8,
                                    min_step_seconds=DEMO_ENTERPRISE_STEP_SECONDS, toolkit=demo_toolkit())
                incident["caller_id"] = "Alex Rivera"
            else:
                agent = TicketAgent(assistant.llm, assistant, DryRunDesk(), max_steps=3,
                                    min_step_seconds=DEMO_AGENT_STEP_SECONDS)
            outcome = agent.run(incident, incident["short_description"], time_left=time_left, prefetched=result,
                                raise_on_model_error=True)
            plan = agent_plan(outcome)
            for key in result["usage"]:
                result["usage"][key] = outcome["usage"].get(key, 0)  # includes the prefetched search
        except AgentUnavailable as error:
            logger.warning("Demo agent plan skipped: %s", error)
        except Exception:
            logger.exception("Demo agent plan failed")
    logger.info(json.dumps({"event": "demo_ask", "status": result["status"], "provider": result["provider"],
                            "sources": [s["source"] for s in result["sources"]], "usage": result["usage"],
                            "latency_ms": result.get("latency_ms"), "agent": (plan or {}).get("action")}))
    body = {k: result.get(k) for k in
            ("status", "answer", "sources", "offer_ticket", "guardrail", "latency_ms", "search_query")}
    body["agent"] = plan
    body["latency_ms"] = int((time.time() - started) * 1000)
    return _response(200, body)
