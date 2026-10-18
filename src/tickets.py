"""Hand a question off to a human by opening a ticket in the AI Ticket Triage API.

POST {TRIAGE_API_URL}/tickets with a subject and description. The triage service sets priority
and category, and P1 tickets trigger an email alert. Only the PII-masked question is sent.
"""
import http.client
import json
import urllib.request


class TicketError(Exception):
    pass


class TicketClient:
    def __init__(self, api_url: str, opener=urllib.request.urlopen, timeout: int = 20):
        if not api_url:
            raise TicketError("Ticket API is not configured (set TRIAGE_API_URL).")
        if not api_url.startswith(("https://", "http://")):
            raise TicketError("TRIAGE_API_URL must start with https://")
        self.api_url = api_url.rstrip("/")
        self.opener = opener
        self.timeout = timeout

    def create(self, question: str, assistant_answer: str = "", sources_checked: list | None = None,
               requester: str = "") -> dict:
        subject = question if len(question) <= 120 else question[:117] + "..."
        description = "\n".join([
            "Opened by the IT Knowledge Base Assistant.",
            f"User question: {question}",
            f"Assistant reply: {assistant_answer or '(none)'}",
            "Articles checked: " + (", ".join(sources_checked) if sources_checked else "(none)"),
        ])[:5000]
        request = urllib.request.Request(
            f"{self.api_url}/tickets",
            data=json.dumps({"subject": subject, "description": description,
                             "requester": requester}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, http.client.HTTPException) as error:  # URLError/timeouts are OSError
            raise TicketError(f"Could not create the ticket: {type(error).__name__}") from error
        if not isinstance(data, dict) or not data.get("ticket_id"):
            raise TicketError("Ticket API did not return a ticket ID.")
        return {"ticket_id": data["ticket_id"], "priority": data.get("priority"),
                "category": data.get("category")}
