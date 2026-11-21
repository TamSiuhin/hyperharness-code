"""Compact, source-ordered construction view of one user's history."""
from collections import defaultdict


def short(value, limit):
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[:limit - 3] + "..."


def digest(events):
    sessions = defaultdict(list)
    for event in events:
        row = event.get("event") or {}
        sessions[str(row.get("session_id") or "unknown")].append(row)

    dialogue = []
    activity = []
    for session, rows in sessions.items():
        previous_assistant = ""
        statements = []
        logs = []
        for row in rows:
            content = row.get("content") or ""
            if row.get("kind") == "dialogue":
                if row.get("role") == "assistant":
                    previous_assistant = short(content, 130)
                elif row.get("role") == "user":
                    line = f"{row.get('turn', '')} USER: {short(content, 195)}"
                    if previous_assistant and len(statements) > 0:
                        line += " | PRIOR ASSISTANT PROPOSAL: " + short(previous_assistant, 70)
                    statements.append(line)
                    previous_assistant = ""
            elif row.get("kind") == "log":
                logs.append(f"[{session} {row.get('date', '')}] {short(content, 165)}")
        dialogue.append((session, statements))
        if logs:
            chosen = []
            for line in logs:
                lower = line.lower()
                if any(term in lower for term in ("sent a message", "recorded", "wrote", "added a note", "purchased", "purchase", "ordered", "reminder")):
                    chosen.append(line)
            for line in logs[-4:]:
                if line not in chosen:
                    chosen.append(line)
            activity.append((session, chosen[-4:]))

    result = []
    for session, statements in dialogue:
        result.append("### " + session)
        result.extend(statements)
    if len("\n".join(result)) > 100000:
        result = []
        for session, statements in dialogue:
            result.append("### " + session)
            for line in statements:
                if " | PRIOR ASSISTANT PROPOSAL: " in line:
                    user, prior = line.split(" | PRIOR ASSISTANT PROPOSAL: ", 1)
                    line = short(user, 165) + " | PRIOR ASSISTANT PROPOSAL: " + short(prior, 45)
                else:
                    line = short(line, 210)
                result.append(line)
    dialogue_text = "\n".join(result)
    log_lines = []
    for session, lines in activity:
        log_lines.append("### " + session)
        log_lines.extend(lines)
    return ("USER DIALOGUE (user statements are evidence; prior assistant proposals are context only)\n"
            + dialogue_text + "\n\nSELECTED ACTIVITY (behavior is weaker than a stated preference)\n"
            + "\n".join(log_lines))
