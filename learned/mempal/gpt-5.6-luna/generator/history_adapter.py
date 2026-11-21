"""Compact construction history with dialogue provenance and broad session coverage."""
import collections
import re

FEEDBACK = re.compile(
    r"\b(?:prefer|rather|instead|don't|do not|won't|can't|cannot|too |not |"
    r"good idea|great idea|sounds|works|work for|suitable|practical|unrealistic|"
    r"consider|try|tried|already|actually|need|want|hope|like|dislike|"
    r"difficult|problem|concern|worry|busy|time|cost|expensive|agree|"
    r"right|exactly|yes|no|but|however)\b", re.I)
ACTION = re.compile(
    r"\b(?:sent|posted|wrote|created|updated|edited|replied|scheduled|"
    r"canceled|cancelled|bought|purchased|ordered|attended|completed|"
    r"recorded|started|stopped|changed|joined|shared|liked|disliked)\b", re.I)


def short(value, limit):
    text = " ".join(str(value or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def construction_digest(events, max_chars=90000):
    sessions = collections.OrderedDict()
    for event in events:
        item = event.get("event") or {}
        sessions.setdefault(item.get("session_id", "unknown"), []).append(item)
    blocks = []
    for session_id, rows in sessions.items():
        dialogue = [row for row in rows if row.get("kind") == "dialogue"]
        logs = [row for row in rows if row.get("kind") == "log"]
        user_turns = [(i, row) for i, row in enumerate(dialogue) if row.get("role") == "user"]
        chosen = set()
        if user_turns:
            chosen.add(user_turns[0][0])
            chosen.add(user_turns[-1][0])
        feedback = []
        for i, row in user_turns:
            content = str(row.get("content") or "")
            if i and FEEDBACK.search(content):
                score = len(FEEDBACK.findall(content)) + (2 if any(word in content.lower() for word in
                    ("but", "not", "too ", "instead", "rather", "don't", "won't")) else 0)
                feedback.append((score, i))
        for _, i in sorted(feedback, reverse=True)[:5]:
            chosen.add(i)
        lines = [session_id + " (" + short((rows[-1] if rows else {}).get("date", ""), 24) + ")"]
        for i in sorted(chosen):
            row = dialogue[i]
            if i and dialogue[i - 1].get("role") == "assistant":
                lines.append("  assistant proposed: " + short(dialogue[i - 1].get("content"), 210))
            lines.append("  USER: " + short(row.get("content"), 310))
        selected_logs = [row for row in logs if ACTION.search(str(row.get("content") or ""))]
        selected_logs = selected_logs[-4:] if selected_logs else logs[-2:]
        for row in selected_logs:
            lines.append("  action " + short(row.get("date"), 18) + ": " + short(row.get("content"), 240))
        if len(lines) > 1:
            blocks.append("\n".join(lines))
    header = f"{len(events)} observations across {len(sessions)} sessions. USER lines are the person's words; assistant proposals are context only.\n"
    budget = max_chars - len(header) - len(blocks) * 2
    share = max(400, budget // max(1, len(blocks)))
    return (header + "\n\n".join(block[:share] for block in blocks))[:max_chars]


def digest(events):
    return construction_digest(events)
