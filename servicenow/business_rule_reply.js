/**
 * Business Rule: "KB Assistant - caller reply"
 * Table: incident | Update: true | When: after (needs `previous`, which async rules don't have)
 *
 * When the AI agent has sent a fix and put the incident On Hold - Awaiting Caller, and the caller
 * then adds a comment, tell AWS so the agent can read the reply: confirmed -> Resolved, anything
 * else -> back to a technician. Signed exactly like the new-incident rule; only ids are sent.
 * Fires only for the caller's own comments; AWS re-checks the author against the caller.
 */
(function executeRule(current, previous) {
    if (!previous || !current.comments.changes()) return;
    if (previous.getValue('state') != '3' || previous.getValue('hold_reason') != '1') return;  // On Hold - Awaiting Caller
    if (gs.getUserName() == gs.getProperty('kb_assistant.service_user', 'svc_kb_assistant')) return;  // our own comment
    if (gs.getUserID() != current.getValue('caller_id')) return;  // only the caller can confirm their fix (AWS checks again)

    var endpoint = gs.getProperty('kb_assistant.webhook_url', '');
    var key = gs.getProperty('kb_assistant.signing_key', '');
    if (!endpoint || !key) return;
    var body = JSON.stringify({ sys_id: current.getUniqueValue(), number: current.getValue('number'),
                                event: 'caller_reply', version: String(current.getValue('sys_mod_count')) });
    var timestamp = String(Math.floor(new GlideDateTime().getNumericValue() / 1000));
    var signature = new GlideCertificateEncryption().generateMac(key, 'HmacSHA256', timestamp + '.' + body);
    try {
        var request = new sn_ws.RESTMessageV2();
        request.setEndpoint(endpoint);
        request.setHttpMethod('post');
        request.setHttpTimeout(5000);
        request.setRequestHeader('Content-Type', 'application/json');
        request.setRequestHeader('X-Timestamp', timestamp);
        request.setRequestHeader('X-Signature', signature);
        request.setRequestBody(body);
        request.executeAsync();  // don't make the caller's save wait on AWS
    } catch (e) {
        gs.error('KB Assistant reply webhook failed for ' + current.getValue('number') + ': ' + e);
    }
})(current, previous);
