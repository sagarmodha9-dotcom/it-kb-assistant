/**
 * Business Rule: "KB Assistant - caller reply"
 * Table: incident | Update: true | When: after (needs `previous`, which async rules don't have)
 *
 * When the caller adds a comment to an incident the AI agent is waiting on (On Hold - Awaiting
 * Caller), tell AWS so the agent can read the reply: confirmed -> Resolved, anything else -> back to a
 * technician. Also when the caller comments on an incident the AI resolved without them (a verified
 * fix, or no reply after two reminders; its close notes start with "Resolved by the AI agent"): unless
 * the reply says it's fixed, the agent reopens it for a technician.
 * Signed exactly like the new-incident rule; only ids are sent. Fires only for the caller's own
 * comments; AWS re-checks the author against the caller and re-reads the incident itself.
 */
(function executeRule(current, previous) {
    if (!previous || !current.comments.changes()) return;
    var awaitingCaller = previous.getValue('state') == '3' && previous.getValue('hold_reason') == '1';
    var aiResolved = previous.getValue('state') == '6'
        && String(previous.getValue('close_notes') || '').indexOf('Resolved by the AI agent') == 0;
    if (!awaitingCaller && !aiResolved) return;
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
