/** Bind this script to the spreadsheet. Deploy as a Web App executing as owner.
 * Set AUDIT_TOKEN in Script Properties and match it on the Flask server.
 * Do not add passwords, hashes, email addresses, names, or IP addresses.
 */
function doPost(e) {
  const expected = PropertiesService.getScriptProperties().getProperty('AUDIT_TOKEN');
  const body = JSON.parse((e && e.postData && e.postData.contents) || '{}');
  if (!expected || body.token !== expected) return ContentService.createTextOutput('{"ok":false}');
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  if (sheet.getLastRow() === 0) sheet.appendRow(['Timestamp UTC', 'Event', 'Outcome', 'Internal user ID']);
  sheet.appendRow([body.recorded_at || new Date().toISOString(), body.event || '', body.outcome || '', body.actor_id || '']);
  return ContentService.createTextOutput('{"ok":true}');
}
