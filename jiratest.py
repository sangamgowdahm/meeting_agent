from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '✅ Replit Flask server is running!'

@app.route('/webhook', methods=['POST', 'GET'])
def handle_webhook():
    if request.method == 'GET':
        return "GET request received — likely a Jira test", 200

    # Check Content-Type
    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported Media Type — Content-Type must be application/json"}), 415

    # Try to parse JSON
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    print("🔔 Webhook Received")

    # Get event type
    event_type = data.get('webhookEvent', 'Unknown')
    print(f"🔹 Event Type: {event_type}")

    # ===============================
    # Handle different Jira events
    # ===============================

    # --- Issue Events ---
    if event_type.startswith("jira:issue_"):
        issue = data.get('issue', {})
        fields = issue.get('fields', {})

        issue_key = issue.get('key')
        summary = fields.get('summary')
        description = fields.get('description')
        status = fields.get('status', {}).get('name')
        reporter = fields.get('reporter', {}).get('displayName')
        assignee = fields.get('assignee', {}).get('displayName') if fields.get('assignee') else 'Unassigned'

        print(f"Issue Key: {issue_key}")
        print(f"Summary: {summary}")
        print(f"Description: {description}")
        print(f"Status: {status}")
        print(f"Reporter: {reporter}")
        print(f"Assignee: {assignee}")

    # --- Sprint Created ---
    elif event_type == "sprint_created":
        sprint = data.get('sprint', {})
        print(f"New Sprint Created: {sprint.get('name')} (ID: {sprint.get('id')})")

    # --- Board Created ---
    elif event_type == "board_created":
        board = data.get('board', {})
        print(f"New Board Created: {board.get('name')} (ID: {board.get('id')})")

    # --- Project Events ---
    elif event_type.startswith("project_"):
        project = data.get('project', {})
        print(f"Project Event: {event_type}")
        print(f"Project Key: {project.get('key')}")
        print(f"Project Name: {project.get('name')}")

    # --- Issue Link Events ---
    elif event_type.startswith("issuelink_"):
        link = data.get('issueLink', {})
        print(f"Issue Link Event: {event_type}")
        print(f"Link Type: {link.get('type', {}).get('name')}")
        print(f"Inward Issue: {link.get('inwardIssue', {}).get('key')}")
        print(f"Outward Issue: {link.get('outwardIssue', {}).get('key')}")

    else:
        print("⚠️ Unhandled webhook event type.")

    return jsonify({"status": "Webhook processed"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
