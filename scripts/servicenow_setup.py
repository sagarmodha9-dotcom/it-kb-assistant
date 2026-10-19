"""Configure a ServiceNow instance for the KB Assistant integration, repeatably.

Run it after deploying the stack with ServiceNowInstanceUrl set. Safe to run again: every step
creates the record if it's missing and updates it if it exists.

  python scripts/servicenow_setup.py --instance https://dev12345.service-now.com

What it does:
  1. Creates an update set "KB Assistant Integration" and makes it current for the admin user, so
     the configuration below is captured and can be exported as XML (servicenow/update_set.xml).
  2. Creates the service account svc_kb_assistant (web service access only, itil role) with a
     random password, saved to SSM Parameter Store. Nobody needs to know it.
  3. Generates a random HMAC signing key, saved to SSM and to a Private system property
     (private properties are never exported in update sets, so the key never lands in Git).
  4. Sets the webhook URL property from the stack output.
  5. Creates the Business Rules on the incident table: new incident (servicenow/business_rule.js)
     and caller reply (servicenow/business_rule_reply.js, for the agent's confirm-the-fix step).
  6. Creates the assignment groups the agent escalates to, if they're missing.
  7. Checks the service account can log in.

The admin password is read with a hidden prompt (or SN_ADMIN_PASSWORD) and is never stored.
"""
import argparse
import base64
import getpass
import os
import secrets
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent import TEAMS  # noqa: E402
from servicenow import ServiceNowClient, ServiceNowError  # noqa: E402

UPDATE_SET = "KB Assistant Integration"
SERVICE_USER = "svc_kb_assistant"
RULE_NAME = "KB Assistant - AI work note"  # sys_script names are limited to 40 characters
REPLY_RULE_NAME = "KB Assistant - caller reply"


def stack_output(stack: str, region: str, key: str) -> str:
    import boto3
    outputs = boto3.client("cloudformation", region_name=region).describe_stacks(StackName=stack)["Stacks"][0]
    for output in outputs.get("Outputs", []):
        if output["OutputKey"] == key:
            return output["OutputValue"]
    raise SystemExit(f"Stack {stack} has no {key} output. Deploy with ServiceNowInstanceUrl set first.")


def put_secret(name: str, value: str, region: str) -> None:
    import boto3
    boto3.client("ssm", region_name=region).put_parameter(Name=name, Value=value, Type="SecureString",
                                                          Overwrite=True)


def upsert(sn: ServiceNowClient, table: str, query: str, record: dict) -> str:
    found = sn.query(table, query, fields="sys_id", limit=1)
    if found:
        sn.update(table, found[0]["sys_id"], record)
        return found[0]["sys_id"]
    return sn.create(table, record)["sys_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", required=True, help="https://devNNNNN.service-now.com")
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--stack", default="it-kb-assistant")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--password-param", default="/it-kb-assistant/servicenow-password")
    parser.add_argument("--signing-key-param", default="/it-kb-assistant/servicenow-webhook-secret")
    parser.add_argument("--rule-when", choices=["async_always", "after"], default="async_always",
                        help="async_always (default): runs in the background job queue, so saving an incident "
                             "never waits. after: runs right after the save, with a 5s timeout; use it when the "
                             "instance's job queue is backlogged (common on brand-new developer instances)")
    parser.add_argument("--keep-secrets", action="store_true",
                        help="don't rotate the service account password and signing key")
    args = parser.parse_args()

    admin_password = os.environ.get("SN_ADMIN_PASSWORD") or getpass.getpass(f"ServiceNow {args.admin_user} password: ")
    sn = ServiceNowClient(args.instance, args.admin_user, admin_password, timeout=120)
    webhook_url = stack_output(args.stack, args.region, "ServiceNowWebhookUrl")

    print("1. Update set")
    update_set = upsert(sn, "sys_update_set", f"name={UPDATE_SET}",
                        {"name": UPDATE_SET, "state": "in progress",
                         "description": "Business Rule and properties for the IT KB Assistant integration."})
    admin = sn.query("sys_user", f"user_name={args.admin_user}", fields="sys_id", limit=1)[0]["sys_id"]
    upsert(sn, "sys_user_preference", f"user={admin}^name=sys_update_set",
           {"user": admin, "name": "sys_update_set", "value": update_set})

    print(f"2. Service account {SERVICE_USER} (itil role)")
    user = upsert(sn, "sys_user", f"user_name={SERVICE_USER}",
                  {"user_name": SERVICE_USER, "first_name": "KB", "last_name": "Assistant (AI integration)",
                   "web_service_access_only": "true", "active": "true"})
    itil = sn.query("sys_user_role", "name=itil", fields="sys_id", limit=1)[0]["sys_id"]
    if not sn.query("sys_user_has_role", f"user={user}^role={itil}", fields="sys_id", limit=1):
        sn.create("sys_user_has_role", {"user": user, "role": itil})
    if not args.keep_secrets:
        password = secrets.token_urlsafe(24) + "aA1!"  # meets typical complexity rules
        sn.update("sys_user", user, {"user_password": password}, display_values=True)
        put_secret(args.password_param, password, args.region)
        print(f"   password saved to SSM {args.password_param}")

    print("3. Signing key")
    if not args.keep_secrets:
        key = base64.b64encode(secrets.token_bytes(32)).decode()
        put_secret(args.signing_key_param, key, args.region)
        upsert(sn, "sys_properties", "name=kb_assistant.signing_key",
               {"name": "kb_assistant.signing_key", "value": key, "type": "string", "is_private": "true",
                "description": "Base64 HMAC-SHA256 key for signing KB Assistant webhook calls. Private: not exported."})
        print(f"   key saved to SSM {args.signing_key_param} and a Private system property")

    print("4. Webhook URL property")
    upsert(sn, "sys_properties", "name=kb_assistant.webhook_url",
           {"name": "kb_assistant.webhook_url", "value": webhook_url, "type": "string",
            "description": "AWS endpoint that receives new-incident events for the KB Assistant."})

    print("5. Business Rule")
    with open(os.path.join(ROOT, "servicenow", "business_rule.js")) as f:
        script = f.read()
    # STARTSWITH also finds a rule created under an older name, so re-runs update it instead of adding a second rule.
    upsert(sn, "sys_script", "nameSTARTSWITHKB Assistant - AI",
           {"name": RULE_NAME, "collection": "incident", "when": args.rule_when, "action_insert": "true",
            "active": "true", "advanced": "true", "order": "1000", "script": script,
            "description": "Sends new incidents to the IT KB Assistant ticket agent (AWS)."})
    with open(os.path.join(ROOT, "servicenow", "business_rule_reply.js")) as f:
        reply_script = f.read()
    upsert(sn, "sys_script", f"name={REPLY_RULE_NAME}",
           {"name": REPLY_RULE_NAME, "collection": "incident", "when": "after", "action_update": "true",
            "active": "true", "advanced": "true", "order": "1000", "script": reply_script,
            "description": "When the caller replies to the AI agent's fix, lets the agent resolve or hand it back."})
    upsert(sn, "sys_properties", "name=kb_assistant.service_user",
           {"name": "kb_assistant.service_user", "value": SERVICE_USER, "type": "string",
            "description": "The KB Assistant service account; its own comments never trigger the reply rule."})

    print("6. Assignment groups for escalation")
    for team in TEAMS:
        if not sn.query("sys_user_group", f"name={team}", fields="sys_id", limit=1):
            sn.create("sys_user_group", {"name": team, "description": "Created for the KB Assistant ticket agent."})
            print(f"   created {team}")

    print("7. Service account login check")
    if not args.keep_secrets:
        try:
            ServiceNowClient(args.instance, SERVICE_USER, password).query("incident", fields="number", limit=1)
            print("   OK")
        except ServiceNowError as error:
            print(f"   FAILED ({error}). Set the password in ServiceNow by hand and save it to SSM {args.password_param}.")

    print(f"\nDone. Webhook: {webhook_url}\n"
          "Next: create a test incident and watch its work notes. Then in ServiceNow open the update set\n"
          f"'{UPDATE_SET}', mark it Complete, and use 'Export to XML' -> save as servicenow/update_set.xml.")


if __name__ == "__main__":
    main()
