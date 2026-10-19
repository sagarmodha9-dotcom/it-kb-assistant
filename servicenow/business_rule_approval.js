/**
 * Business Rule: "KB Assistant - approval decision"
 * Table: incident | Update: true | When: after
 *
 * When the AI agent has asked for approval (Approval = Requested) and an approver sets Approval to
 * Approved or Rejected, tell AWS who decided and what (signed). Recommended: an ACL so only approver
 * roles can write the incident's Approval field. The worker checks that this person is an authorized approver for
 * that request (right role, not the requester), then runs exactly the approved action, or tells the
 * caller it was rejected. Signed like the other rules; only ids are sent.
 */
(function executeRule(current, previous) {
    if (!previous || !current.approval.changes() || previous.getValue('approval') != 'requested') return;
    var decision = current.getValue('approval');
    if (decision != 'approved' && decision != 'rejected') return;
    // The person who made this change, captured now and signed: AWS never infers it from later edits.
    var approver = gs.getUserName();

    var endpoint = gs.getProperty('kb_assistant.webhook_url', '');
    var key = gs.getProperty('kb_assistant.signing_key', '');
    if (!endpoint || !key) return;
    var body = JSON.stringify({ sys_id: current.getUniqueValue(), number: current.getValue('number'),
                                event: 'approval', version: String(current.getValue('sys_mod_count')),
                                decision: decision, approver: approver });
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
        request.executeAsync();
    } catch (e) {
        gs.error('KB Assistant approval webhook failed for ' + current.getValue('number') + ': ' + e);
    }
})(current, previous);
