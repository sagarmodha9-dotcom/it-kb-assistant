"""Safety checks before and after the model.

check_input(question)                      runs before retrieval and the model
check_output(answer, sources, question)    runs on the model's answer

Each returns a GuardResult:
  action   "NONE" (passed), "ANONYMIZED" (PII masked, continue with masked text), or "BLOCKED"
  text     the text to use from here on (masked if PII was found)
  reasons  why it intervened, e.g. ["topic:Security bypass", "pii:US_SOCIAL_SECURITY_NUMBER"]
  scores   contextual grounding scores on output, e.g. {"GROUNDING": 0.91, "RELEVANCE": 0.88}

BedrockGuardrail  Amazon Bedrock Guardrails through the ApplyGuardrail API. ApplyGuardrail runs
                  separately from the model call, so the same guardrail protects every provider,
                  including the Anthropic API fallback. On output it runs a contextual grounding
                  check: is the answer supported by the retrieved documents, and relevant?
LocalGuardrail    an offline approximation (regex PII masking, keyword topic and prompt-attack
                  rules, word-overlap grounding) for tests, CI, and local demos.
"""
import re
from dataclasses import dataclass, field

from llm import content_words


@dataclass
class GuardResult:
    action: str
    text: str
    reasons: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == "BLOCKED"


class BedrockGuardrail:
    name = "bedrock"
    MAX_QUERY = 1000          # contextual grounding limits for query and source text
    MAX_SOURCE = 100_000

    def __init__(self, guardrail_id: str, version: str, client=None):
        self.guardrail_id = guardrail_id
        self.version = version
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("bedrock-runtime",
                                  config=Config(connect_timeout=2, read_timeout=4, retries={"max_attempts": 1}))
        self.client = client

    def _apply(self, source: str, content: list) -> dict:
        return self.client.apply_guardrail(guardrailIdentifier=self.guardrail_id,
                                           guardrailVersion=self.version, source=source, content=content)

    @staticmethod
    def _parse(response: dict, original: str) -> GuardResult:
        reasons, scores, blocked = [], {}, False
        for assessment in response.get("assessments", []):
            for topic in assessment.get("topicPolicy", {}).get("topics", []):
                if topic.get("action") == "BLOCKED":
                    blocked = True
                    reasons.append(f"topic:{topic.get('name')}")
            for f in assessment.get("contentPolicy", {}).get("filters", []):
                if f.get("action") == "BLOCKED":
                    blocked = True
                    reasons.append(f"content:{f.get('type')}")
            for group in ("customWords", "managedWordLists"):
                for w in assessment.get("wordPolicy", {}).get(group, []):
                    if w.get("action") == "BLOCKED":
                        blocked = True
                        reasons.append("word")
            sensitive = assessment.get("sensitiveInformationPolicy", {})
            for entity in sensitive.get("piiEntities", []) + sensitive.get("regexes", []):
                if entity.get("action") in ("ANONYMIZED", "BLOCKED"):
                    blocked = blocked or entity["action"] == "BLOCKED"
                    reasons.append(f"pii:{entity.get('type') or entity.get('name')}")
            for f in assessment.get("contextualGroundingPolicy", {}).get("filters", []):
                scores[f.get("type")] = round(float(f.get("score", 0.0)), 3)
                if f.get("action") == "BLOCKED":
                    blocked = True
                    reasons.append(f"grounding:{f.get('type')}")
        if response.get("action") != "GUARDRAIL_INTERVENED":
            return GuardResult("NONE", original, reasons, scores)
        if blocked:
            return GuardResult("BLOCKED", "", reasons, scores)
        # On output the request has 3 blocks (sources, query, answer); the answer is the last one.
        outputs = response.get("outputs") or [{"text": original}]
        return GuardResult("ANONYMIZED", outputs[-1].get("text", original), reasons, scores)

    def check_input(self, text: str) -> GuardResult:
        return self._parse(self._apply("INPUT", [{"text": {"text": text}}]), text)

    def check_output(self, answer: str, sources: list[str], question: str) -> GuardResult:
        source_text = "\n\n".join(sources)[: self.MAX_SOURCE]
        content = [
            {"text": {"text": source_text, "qualifiers": ["grounding_source"]}},
            {"text": {"text": question[: self.MAX_QUERY], "qualifiers": ["query"]}},
            {"text": {"text": answer, "qualifiers": ["guard_content"]}},
        ]
        return self._parse(self._apply("OUTPUT", content), answer)


class LocalGuardrail:
    name = "local"

    PII = [
        ("US_SOCIAL_SECURITY_NUMBER", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        ("CREDIT_DEBIT_CARD_NUMBER", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
        ("PHONE", re.compile(r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
    ]
    # Lines up with the denied topics and prompt-attack filter in the Bedrock guardrail (template.yaml).
    TOPICS = {
        "Security bypass": re.compile(
            r"\b(disable|turn off|uninstall|bypass|get around|kill|stop)\b.{0,40}"
            r"\b(crowdstrike|falcon|antivirus|edr|mfa|security|firewall|bitlocker|monitoring)\b", re.I),
        "Credential sharing": re.compile(
            r"\b(share|give|send|tell)\b.{0,30}\b(my|your|his|her|their|someone'?s?)\b.{0,15}"
            r"\b(password|mfa code|passcode)\b", re.I),
    }
    PROMPT_ATTACK = re.compile(
        r"ignore (all |any |the )?(previous|prior|above|your) (instructions|rules)|"
        r"(reveal|print|show|repeat).{0,20}(system prompt|instructions)|"
        r"\byou are now\b|developer mode|jailbreak", re.I)

    def __init__(self, grounding_threshold: float = 0.5):
        self.grounding_threshold = grounding_threshold

    def check_input(self, text: str) -> GuardResult:
        for name, pattern in self.TOPICS.items():
            if pattern.search(text):
                return GuardResult("BLOCKED", "", [f"topic:{name}"])
        if self.PROMPT_ATTACK.search(text):
            return GuardResult("BLOCKED", "", ["content:PROMPT_ATTACK"])
        return self._mask(text)

    # Answers may quote support emails and phone numbers from the documents, so on output only
    # the high-risk types are masked (matches OutputEnabled settings in template.yaml).
    OUTPUT_PII = ("US_SOCIAL_SECURITY_NUMBER", "CREDIT_DEBIT_CARD_NUMBER")

    def _mask(self, text: str, only: tuple = ()) -> GuardResult:
        reasons = []
        for name, pattern in self.PII:
            if only and name not in only:
                continue
            text, count = pattern.subn("{" + name + "}", text)
            reasons += [f"pii:{name}"] * count
        return GuardResult("ANONYMIZED" if reasons else "NONE", text, reasons)

    def check_output(self, answer: str, sources: list[str], question: str) -> GuardResult:
        answer_words = content_words(re.sub(r"\[\d+\]", "", answer))
        source_words = content_words(" ".join(sources))
        grounding = len(answer_words & source_words) / max(len(answer_words), 1)
        scores = {"GROUNDING": round(grounding, 3)}
        if grounding < self.grounding_threshold:
            return GuardResult("BLOCKED", "", ["grounding:GROUNDING"], scores)
        masked = self._mask(answer, only=self.OUTPUT_PII)
        masked.scores = scores
        return masked


class NoGuardrail:
    name = "none"

    def check_input(self, text: str) -> GuardResult:
        return GuardResult("NONE", text)

    def check_output(self, answer: str, sources: list[str], question: str) -> GuardResult:
        return GuardResult("NONE", answer)
