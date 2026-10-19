"""Talk to the AI model.

BedrockLLM    calls a model through Amazon Bedrock's Converse API.
AnthropicLLM  calls Claude directly through the Anthropic Messages API.
FallbackLLM   tries one provider and switches to a backup if it fails (throttling, quota, outage).
FakeLLM       a rule-based stand-in so tests, CI, and demos run without AWS or an API key.

Every model has the same methods, so the rest of the code doesn't care which one it uses:
  complete(system, user, max_tokens) -> text
  chat(system, messages, tools, max_tokens) -> {"content": [blocks], "stop_reason": ...}   (tool use)
and records `last_usage` = {"input_tokens": n, "output_tokens": n} after each call.

chat() uses the Anthropic Messages format everywhere (text, tool_use, and tool_result blocks); the
Bedrock class converts to and from the Converse API format, so the agent code is provider-neutral.
"""
import json
import logging
import re
import urllib.request

logger = logging.getLogger(__name__)


def _usage(input_tokens=0, output_tokens=0) -> dict:
    return {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}


class BedrockLLM:
    provider = "bedrock"

    def __init__(self, model_id: str, client=None, max_attempts: int = 3, read_timeout: int = 15):
        self.model_id = model_id
        self.last_usage = _usage()
        if client is None:
            import boto3  # only needed when actually calling AWS
            from botocore.config import Config
            # botocore's default read timeout is 60s, longer than API Gateway's 30s limit.
            client = boto3.client("bedrock-runtime", config=Config(
                connect_timeout=3, read_timeout=read_timeout,
                retries={"max_attempts": max_attempts, "mode": "standard"}))
        self.client = client

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = response.get("usage", {})
        self.last_usage = _usage(usage.get("inputTokens"), usage.get("outputTokens"))
        return "".join(b.get("text", "") for b in response["output"]["message"]["content"])

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 700) -> dict:
        response = self.client.converse(
            modelId=self.model_id, system=[{"text": system}], messages=_to_converse(messages),
            toolConfig={"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                                "inputSchema": {"json": t["input_schema"]}}} for t in tools]},
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = response.get("usage", {})
        self.last_usage = _usage(usage.get("inputTokens"), usage.get("outputTokens"))
        content = []
        for block in response["output"]["message"]["content"]:
            if "text" in block:
                content.append({"type": "text", "text": block["text"]})
            elif "toolUse" in block:
                use = block["toolUse"]
                content.append({"type": "tool_use", "id": use["toolUseId"], "name": use["name"],
                                "input": use.get("input") or {}})
        return {"content": content, "stop_reason": response.get("stopReason", "")}


def _to_converse(messages: list) -> list:
    """Anthropic-format messages -> Bedrock Converse format."""
    out = []
    for message in messages:
        blocks = message["content"] if isinstance(message["content"], list) else [
            {"type": "text", "text": message["content"]}]
        content = []
        for b in blocks:
            if b["type"] == "text":
                content.append({"text": b["text"]})
            elif b["type"] == "tool_use":
                content.append({"toolUse": {"toolUseId": b["id"], "name": b["name"], "input": b["input"]}})
            elif b["type"] == "tool_result":
                content.append({"toolResult": {"toolUseId": b["tool_use_id"],
                                               "content": [{"text": str(b.get("content", ""))}]}})
        out.append({"role": message["role"], "content": content})
    return out


class AnthropicLLM:
    """Calls the Anthropic Messages API with only the standard library (no extra packages)."""

    provider = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model_id: str, api_key: str, opener=urllib.request.urlopen, timeout: int = 15):
        if not api_key:
            raise ValueError("Anthropic API key is missing")
        self.model_id = model_id
        self.api_key = api_key
        self.opener = opener
        self.timeout = timeout
        self.last_usage = _usage()

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.URL, data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        with self.opener(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        usage = data.get("usage", {})
        self.last_usage = _usage(usage.get("input_tokens"), usage.get("output_tokens"))
        return data

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        data = self._post({"model": self.model_id, "max_tokens": max_tokens, "temperature": 0,
                           "system": system, "messages": [{"role": "user", "content": user}]})
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 700) -> dict:
        data = self._post({"model": self.model_id, "max_tokens": max_tokens, "temperature": 0,
                           "system": system, "messages": messages, "tools": tools})
        content = [b for b in data.get("content", []) if b.get("type") in ("text", "tool_use")]
        return {"content": content, "stop_reason": data.get("stop_reason", "")}


class FallbackLLM:
    """Try the primary provider first; if it raises, use the backup instead."""

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup = backup
        self.last_provider = None

    @property
    def provider(self):
        return self.last_provider or self.primary.provider

    @property
    def last_usage(self):
        used = self.backup if self.last_provider == self.backup.provider else self.primary
        return getattr(used, "last_usage", _usage())

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        try:
            result = self.primary.complete(system, user, max_tokens)
            self.last_provider = self.primary.provider
            return result
        except Exception as error:  # throttling, no quota, outage, etc.
            logger.warning("Primary model (%s) failed, switching to backup (%s): %s",
                           self.primary.provider, self.backup.provider, type(error).__name__)
            result = self.backup.complete(system, user, max_tokens)
            self.last_provider = self.backup.provider
            return result

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 700) -> dict:
        try:
            result = self.primary.chat(system, messages, tools, max_tokens)
            self.last_provider = self.primary.provider
            return result
        except Exception as error:
            logger.warning("Primary model (%s) failed, switching to backup (%s): %s",
                           self.primary.provider, self.backup.provider, type(error).__name__)
            result = self.backup.chat(system, messages, tools, max_tokens)
            self.last_provider = self.backup.provider
            return result


WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be but by can do does for from how i if in into is it its me my
no not of on or our so than that the their then there these this to up was we what when where which
who why will with you your get got have has need want""".split())


def content_words(text: str) -> set:
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1}


class FakeLLM:
    """Pretends to answer from the documents with simple word overlap. Good enough to test the
    pipeline (prompt building, JSON parsing, citations, refusals) without calling a real model."""

    provider = "fake"

    def __init__(self):
        self.last_usage = _usage()

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        self.last_usage = _usage(len(system + user) // 4, 60)
        if "<caller_reply>" in user:  # agent.classify_reply
            reply = user.split("<caller_reply>")[-1].split("</caller_reply>")[0].lower()
            if re.search(r"\b(not|didn't|didnt|still|doesn't|won't|no)\b", reply):
                return '{"outcome": "not_fixed"}'
            if re.search(r"\b(worked|fixed|solved|thanks|thank you|all good|yes)\b", reply):
                return '{"outcome": "fixed"}'
            return '{"outcome": "unclear"}'
        question = user.split("<question>")[-1].split("</question>")[0]
        if "<history>" in user:  # follow-up rewrite: borrow the topic of the last user turn
            last = [line[6:] for line in user.split("\n") if line.startswith("User: ")][-1]
            return f"{question} ({last})"
        q_words = content_words(question)
        docs = re.findall(r'<document index="(\d+)"[^>]*>(.*?)</document>', user, re.S)
        best = None
        for index, text in docs:
            for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
                overlap = len(q_words & content_words(sentence))
                if sentence.strip() and (best is None or overlap > best[0]):
                    best = (overlap, int(index), sentence.strip())
        if not best or best[0] < 2:
            return json.dumps({"answerable": False, "answer": "", "citations": []})
        return json.dumps({"answerable": True, "answer": f"{best[2]} [{best[1]}]", "citations": [best[1]]})

    # A scripted stand-in for the ticket agent (agent.py), so tests and offline runs exercise the
    # whole tool loop: search the KB, check for similar incidents, then resolve or escalate.
    FAKE_TEAMS = [("Network", ("vpn", "wi-fi", "wifi", "network", "internet", "outage")),
                  ("Security", ("phishing", "clicked", "malware", "virus", "hacked", "suspicious")),
                  ("Hardware", ("laptop", "printer", "monitor", "screen", "dock", "keyboard", "battery")),
                  ("Identity & Access", ("password", "mfa", "locked", "account", "sign in", "login"))]

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 700) -> dict:
        self.last_usage = _usage(len(json.dumps(messages)) // 4, 40)
        incident = re.search(r"<incident>(.*?)</incident>", messages[0]["content"]
                             if isinstance(messages[0]["content"], str) else messages[0]["content"][0]["text"], re.S)
        text = incident.group(1) if incident else ""
        summary = re.search(r"Short description: (.*)", text)
        question = summary.group(1).strip() if summary else text.strip()
        calls, results = {}, {}
        for m in messages:
            for b in m["content"] if isinstance(m["content"], list) else []:
                if b["type"] == "tool_use":
                    calls[b["id"]] = b
                elif b["type"] == "tool_result":
                    try:
                        results[b["tool_use_id"]] = json.loads(b["content"])
                    except (TypeError, ValueError):
                        results[b["tool_use_id"]] = {}
        names = [c["name"] for c in calls.values()]
        step = len(calls) + 1

        def use(name, args):
            return {"content": [{"type": "tool_use", "id": f"fake_{step}", "name": name, "input": args}],
                    "stop_reason": "tool_use"}

        prompt = messages[0]["content"] if isinstance(messages[0]["content"], str) else messages[0]["content"][0]["text"]
        first = re.search(r"\(search_id kb1\):\n(\{.*\})", prompt, re.S)
        search = json.loads(first.group(1)) if first else {}
        if not search and "search_knowledge_base" not in names:
            return use("search_knowledge_base", {"query": question})
        if "find_similar_incidents" not in names:
            return use("find_similar_incidents", {})
        search = search or next(results.get(i, {}) for i, c in calls.items() if c["name"] == "search_knowledge_base")
        similar = next(results.get(i, {}) for i, c in calls.items() if c["name"] == "find_similar_incidents")
        urgent = "priority: 1" in text.lower() or "priority: 2" in text.lower()
        if search.get("status") == "answered" and not similar.get("possible_outage") and not urgent \
                and "send_fix_to_caller" not in names:
            return use("send_fix_to_caller", {"search_id": search.get("search_id", "")})
        lower = question.lower()
        team = next((t for t, words in self.FAKE_TEAMS if any(w in lower for w in words)), "Service Desk")
        args = {"team": team, "reason": "Not resolvable from the knowledge base."}
        if similar.get("possible_outage"):
            args.update(priority="2", reason="Several similar incidents in the last hour: likely outage.")
        elif team == "Security":
            args.update(priority="2", reason="Possible security incident.")
        return use("escalate_incident", args)


def build_llm(provider: str, bedrock_model_id: str = "", anthropic_model_id: str = "",
              anthropic_api_key: str = ""):
    """Pick the model setup by name: bedrock, anthropic, bedrock_with_anthropic_fallback, or fake."""
    # Worst case per call stays under assistant.MIN_SECONDS_FOR_MODEL (20 s).
    if provider == "bedrock":
        return BedrockLLM(bedrock_model_id, max_attempts=2, read_timeout=8)
    if provider == "anthropic":
        return AnthropicLLM(anthropic_model_id, anthropic_api_key, timeout=15)
    if provider == "bedrock_with_anthropic_fallback":
        # One quick Bedrock attempt (no long retry loop), then fail over to Anthropic.
        return FallbackLLM(BedrockLLM(bedrock_model_id, max_attempts=1, read_timeout=8),
                           AnthropicLLM(anthropic_model_id, anthropic_api_key, timeout=10))
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown LLM provider: {provider}")
