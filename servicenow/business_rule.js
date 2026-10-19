/**
 * Business Rule: "KB Assistant - AI work note"
 * Table: incident | Insert: true | When: async_always (background job queue, so saving never waits)
 *   or after (runs right after the save; the webhook only queues and answers in under a second, and the
 *   5-second timeout below caps the worst case). Use "after" if the instance's job queue is backlogged.
 *
 * Sends only the incident's sys_id and number to the AWS webhook. The worker then reads the incident
 * itself through the Table API, so no incident text travels in this request.
 * The request is signed: HMAC-SHA256 over "<timestamp>.<body>" with the shared key, which AWS
 * verifies (and rejects anything older than 5 minutes).
 *
 * System properties (created by scripts/servicenow_setup.py):
 *   kb_assistant.webhook_url   the stack's ServiceNowWebhookUrl output
 *   kb_assistant.signing_key   base64 HMAC key (marked Private, so it is never exported in update sets)
 */
(function executeRule(current, previous /* null when async */) {
    var endpoint = gs.getProperty('kb_assistant.webhook_url', '');
    var key = gs.getProperty('kb_assistant.signing_key', '');
    if (!endpoint || !key) {
        gs.warn('KB Assistant: webhook URL or signing key is not configured; skipping ' + current.getValue('number'));
        return;
    }
    var body = JSON.stringify({ sys_id: current.getUniqueValue(), number: current.getValue('number') });
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
        var response = request.execute();
        var status = response.getStatusCode();
        if (status != 202 && status != 200) {
            gs.warn('KB Assistant webhook returned HTTP ' + status + ' for ' + current.getValue('number') +
                    ': ' + response.getBody());
        }
    } catch (e) {
        // Never block or break the incident: log and move on. The incident works normally without AI.
        gs.error('KB Assistant webhook failed for ' + current.getValue('number') + ': ' + e);
    }
})(current, previous);
