"""Unified history digest -> personal section -> host check -> bounded repair generation."""
import ast
import collections
import json
import re
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "#@USER_SECTION@#"
REPAIRS = 3
DIGEST_CHARS = 40000
SECTION_CHARS = 14000
MAX_TOKENS = 12000
EFFORT = 'medium'
FULL_PROGRAM = False


def call(op, **payload):
    sys.stdout.write("@@PHG " + json.dumps({"op": op, **payload}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    if op == "done":
        return None
    line = sys.stdin.readline()
    if not line:
        raise SystemExit("host closed the protocol")
    return json.loads(line)


def clip(value, size):
    text = " ".join(str(value or "").split())
    return text if len(text) <= size else text[:size - 3] + "..."


def day(event):
    stamp = (event.get("event") or {}).get("formatted_timestamp") or ""
    match = re.search(r"(\d\d)/(\d\d)/(\d{4})", stamp)
    return f"{match.group(3)}-{match.group(1)}-{match.group(2)}" if match else str(event.get("ts", ""))


def content_text(details):
    content = details.get("content") or {}
    if not isinstance(content, dict):
        return clip(content, 400)
    parts = [content.get(key) for key in ("title", "caption", "text", "overall_description") if content.get(key)]
    return clip(" | ".join(str(part) for part in parts), 400)


def user_turns(details):
    turns = []
    for turn in details.get("conversation") or []:
        if isinstance(turn, dict) and turn.get("role") == "user" and turn.get("content"):
            turns.append(str(turn["content"]))
    return turns


def dm_text(details):
    texts = []
    for message in details.get("messages") or []:
        if not isinstance(message, dict):
            continue
        text = message.get("text") or message.get("content")
        if not isinstance(text, str):
            forwarded = message.get("forwarded_content") or {}
            text = "[forwarded] " + content_text(forwarded) if forwarded else ""
        sender = message.get("sender") or message.get("sender_id") or message.get("author") or ""
        if text:
            texts.append((str(sender) + ": " if sender else "") + text)
    return texts


def digest(events):
    """Deterministic, compact view of the user's history for the generator model."""
    apps = collections.Counter(event.get("app", "") for event in events)
    kinds = collections.Counter((event.get("event") or {}).get("source_interaction_type", "unspecified") for event in events)
    split = int(len(events) * 0.75)
    positive, negative, late_positive, early_positive = (collections.Counter() for _ in range(4))
    persona, cities, people = collections.Counter(), collections.Counter(), collections.Counter()
    own, chats, forgets, liked, disliked, dms, calendar = [], [], [], [], [], [], []
    for index, event in enumerate(events):
        details = event.get("event") or {}
        app = event.get("app", "")
        interaction = details.get("source_interaction_type", "")
        tags = [str(tag).lower() for tag in details.get("source_hashtags") or []]
        if interaction in ("explicit_positive", "implicit_positive") or details.get("is_self_authored"):
            positive.update(tags)
            (late_positive if index >= split else early_positive).update(tags)
        elif interaction in ("explicit_negative", "implicit_negative"):
            negative.update(tags)
        for item in details.get("preferences") or []:
            if isinstance(item, dict) and item.get("persona_item"):
                persona[(item["persona_item"], item.get("time_horizon") or "")] += 1
        city = (details.get("event_location") or {}).get("city") if isinstance(details.get("event_location"), dict) else None
        if city:
            cities[city] += 1
        relationship = details.get("relationship")
        if relationship and relationship not in ("public", "stranger", "self"):
            people[(details.get("author_id") or "", relationship)] += 1
        action = (details.get("interaction_format") or {}).get("action_label") or ""
        if details.get("is_self_authored"):
            own.append(f"- [{day(event)} {app}] {content_text(details)} {' '.join(tags[:6])}")
        if app in ("chatbot", "ai_studio"):
            turns = user_turns(details)
            note = (details.get("interaction_format") or {}).get("user_message")
            if details.get("ask_to_forget"):
                forgets.append(f"- [{day(event)} {app}] user said: {clip(note, 300)} | topic: {clip(turns[0] if turns else '', 200)}")
            elif turns:
                chats.append(f"- [{day(event)} {app}] {clip(turns[0], 260)}" + (f" / later: {clip(turns[-1], 160)}" if len(turns) > 1 else ""))
        if details.get("is_dm"):
            texts = dm_text(details)
            if texts:
                dms.append(f"- [{day(event)} {app} {relationship or ''}] " + clip(" / ".join(texts[-3:]), 300))
        if app == "calendar":
            entry = details.get("entry") or {}
            calendar.append(f"- [{day(event)}] {details.get('action', '')}: {clip(entry.get('title') or details.get('entry_id') or '', 120)}"
                + (f" ({entry.get('type')})" if entry.get("type") else "")
                + (f" reason: {clip(details.get('removal_reason'), 120)}" if details.get("removal_reason") else "")
                + (f" change: {clip(json.dumps(details.get('diff'), ensure_ascii=False), 200)}" if details.get("diff") else ""))
        if interaction == "explicit_positive" and not details.get("is_self_authored") and app not in ("chatbot", "ai_studio"):
            liked.append(f"- [{day(event)} {app}] {action}: {clip(content_text(details), 200)} {' '.join(tags[:5])}")
        if interaction == "explicit_negative":
            disliked.append(f"- [{day(event)} {app}] {action}: {clip(content_text(details), 140)} {' '.join(tags[:5])}")
    rising = [f"{tag} ({late_positive[tag]})" for tag, _ in late_positive.most_common(60)
        if late_positive[tag] >= 2 and late_positive[tag] * 3 > early_positive[tag]][:20]
    fading = [f"{tag} ({early_positive[tag]})" for tag, _ in early_positive.most_common(60)
        if early_positive[tag] >= 3 and not late_positive[tag]][:15]
    sections = [
        ("Overview", [f"{len(events)} events from {day(events[0]) if events else '?'} to {day(events[-1]) if events else '?'}",
            "apps: " + ", ".join(f"{app} {count}" for app, count in apps.most_common()),
            "interactions: " + ", ".join(f"{kind} {count}" for kind, count in kinds.most_common()),
            "places: " + ", ".join(f"{city} {count}" for city, count in cities.most_common(8))]),
        ("Requests to forget or not personalize (respect these)", forgets[-15:]),
        ("Annotated interests (count, horizon)", [f"- {item} ({count}, {horizon})" for (item, horizon), count in persona.most_common(30)]),
        ("Positive hashtags (engaged or own posts)", [", ".join(f"{tag} {count}" for tag, count in positive.most_common(50))]),
        ("Rising in the latest quarter of the history", [", ".join(rising)]),
        ("Fading (positive earlier, absent in the latest quarter)", [", ".join(fading)]),
        ("Negative hashtags (skipped, hidden, disliked)", [", ".join(f"{tag} {count}" for tag, count in negative.most_common(30))]),
        ("Own posts (most recent last)", own[-30:]),
        ("Own messages to assistants (most recent last)", chats[-45:]),
        ("Direct messages", dms[-15:]),
        ("Calendar", calendar[-25:]),
        ("People (author, relationship, count)", [f"- {author} {relationship} {count}" for (author, relationship), count in people.most_common(12)]),
        ("Explicitly liked or engaged content (most recent last)", liked[-25:]),
        ("Explicitly disliked content", disliked[-12:]),
    ]
    text = ""
    for title, lines in sections:
        lines = [line for line in lines if line and line.strip(", ")]
        if lines:
            text += f"### {title}\n" + "\n".join(lines) + "\n\n"
    return text[:DIGEST_CHARS]


def extract_section(reply):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    return (blocks[-1] if blocks else reply).strip()


def submitted_program(reply, backbone):
    content = extract_section(reply)
    if FULL_PROGRAM:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content
        if any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "PROGRAM_API_VERSION" for target in node.targets) for node in tree.body):
            return content
    return backbone.replace(MARKER, content + "\n")


def main():
    task = json.loads(Path(sys.argv[1]).read_text())
    started = time.time()
    backbone = (HERE / "backbone.py").read_text()
    limits = task["limits"]
    language = task["language"]
    prompt = (HERE / "prompt.md").read_text()
    prompt = prompt.replace("<<DATASET_RUNTIME>>", (HERE / "program_api.md").read_text())
    adapter = HERE / "history_adapter.py"
    history = runpy.run_path(str(adapter))["digest"](task["events"]) if adapter.exists() else digest(task["events"])
    fields = {"<<CONTROL_SECTION>>": (HERE / "control_section.py").read_text().strip(),
        "<<LANGUAGE>>": "builtins " + ", ".join(language["safe_builtins"]) + "; string/list/dict methods "
            + ", ".join(language["safe_methods"]),
        "<<MAX_CHARS>>": str(limits["program_chars"]), "<<MAX_NODES>>": str(limits["program_nodes"]),
        "<<SECTION_CHARS>>": str(SECTION_CHARS), "<<BACKBONE>>": backbone.strip(), "<<DIGEST>>": history,
        "<<STARTING_PROGRAM>>": backbone.replace(MARKER, (HERE / "control_section.py").read_text().strip() + "\n")}
    for key, value in fields.items():
        prompt = prompt.replace(key, value)
    messages = [{"role": "user", "content": prompt}]
    notes = {"digest_chars": len(fields["<<DIGEST>>"]), "prompt_chars": len(prompt), "attempts": []}
    for attempt in range(1 + REPAIRS):
        reply = call("llm", messages=messages, max_tokens=MAX_TOKENS, effort=EFFORT)
        if reply.get("error"):
            notes["attempts"].append({"llm_error": reply["error"]})
            break
        section = extract_section(reply["content"])
        program = submitted_program(reply["content"], backbone)
        result = call("check", program=program)
        notes["attempts"].append({"section_chars": len(section), "ok": result.get("ok"), "error": result.get("error")})
        if result.get("ok"):
            notes["seconds"] = round(time.time() - started, 1)
            call("done", program=program, notes=notes)
            return
        messages += [{"role": "assistant", "content": reply["content"]}, {"role": "user", "content":
            "The host rejected your program:\n" + str(result.get("error"))[:3000]
            + ("\nFix the cause and reply with the complete corrected program in one ```python block." if FULL_PROGRAM
                else "\nFix the cause and reply with the complete corrected section in one ```python block.")}]
    notes["seconds"] = round(time.time() - started, 1)
    call("done", program=None, notes=notes)


if __name__ == "__main__":
    main()
