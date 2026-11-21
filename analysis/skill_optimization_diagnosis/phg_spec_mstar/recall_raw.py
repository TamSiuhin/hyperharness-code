"""M*-style seed `recall_raw`: dump the recent raw history, personalize retrieval only.

recent raw records (no model summary) -> one call writes USER_MODEL + user_route/user_keep -> host check -> repair with
the host error. The personal harness changes only what the runtime retrieves; it adds no guidance text.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKFLOW_VERSION = "mstar-v1"
RECENT_CHARS = 60000
RECORD_CHARS = 1500
SKIP_KEYS = ("id", "offset", "timestamp", "lat", "lon", "precision", "contract")
FOCUS = ("Personalize retrieval only. Put this user's recurring topics, entities and own vocabulary in USER_MODEL. "
    "Write user_route to turn each request into up to three BM25 queries that use this vocabulary, and user_keep to "
    "drop records unlikely to help this user. user_guidance returns an empty string: the answer model sees only the "
    "retrieved evidence.")


def load_base():
    specification = importlib.util.spec_from_file_location("base_generator", HERE / "base_generator.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


base = load_base()


def flat(value, key=""):
    if isinstance(value, dict):
        return " ".join(flat(item, name) for name, item in value.items()
                        if not any(str(name).endswith(skip) for skip in SKIP_KEYS))
    if isinstance(value, list):
        return " ".join(flat(item, key) for item in value)
    text = " ".join(str(value).split())
    return f"{key}={text}" if text not in ("", "None", "False", "[]", "{}") else ""


def record(event, size=RECORD_CHARS):
    body = flat(event["event"]) if isinstance(event.get("event"), dict) else event.get("text")
    return f"[{event.get('id')}] {event.get('app') or ''}: " + base.clip(" ".join(str(body).split()), size)


def build_context(task, state):
    lines, size = [], 0
    for event in reversed(task["events"]):
        line = record(event)
        if size + len(line) + 1 > RECENT_CHARS:
            break
        lines.append(line)
        size += len(line) + 1
    state["notes"]["records"] = len(lines)
    return f"Most recent {len(lines)} of {len(task['events'])} records, oldest first:\n" + "\n".join(reversed(lines))


def ask(state, messages, max_tokens):
    """One model request; request errors are retried while the registered call budget lasts."""
    while state["calls"] < state["limit"]:
        state["calls"] += 1
        reply = base.call("llm", messages=messages, max_tokens=max_tokens, effort=base.EFFORT)
        if not reply.get("error"):
            return reply["content"]
        state["notes"]["llm_errors"].append(str(reply["error"])[:300])
    return None


def write_program(task, state, context):
    backbone = (HERE / "backbone.py").read_text()
    control = (HERE / "control_section.py").read_text().strip()
    language, limits = task["language"], task["limits"]
    fields = {"<<CONTROL_SECTION>>": control, "<<LANGUAGE>>": "builtins " + ", ".join(language["safe_builtins"])
        + "; string/list/dict methods " + ", ".join(language["safe_methods"]),
        "<<MAX_CHARS>>": str(limits["program_chars"]), "<<MAX_NODES>>": str(limits["program_nodes"]),
        "<<SECTION_CHARS>>": str(base.SECTION_CHARS), "<<BACKBONE>>": backbone.strip(), "<<DIGEST>>": context,
        "<<STARTING_PROGRAM>>": backbone.replace(base.MARKER, control + "\n")}
    prompt = (HERE / "prompt.md").read_text()
    for key, value in fields.items():
        prompt = prompt.replace(key, value)
    messages = [{"role": "user", "content": prompt + "\n## Seed design\n" + FOCUS}]
    while True:
        reply = ask(state, messages, base.MAX_TOKENS)
        if reply is None:
            return None
        program = base.submitted_program(reply, backbone)
        result = base.call("check", program=program)
        state["notes"]["attempts"].append({"ok": bool(result.get("ok")), "error": str(result.get("error") or "")[:300]})
        if result.get("ok"):
            return program
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": "The host rejected your "
            "program:\n" + str(result.get("error"))[:3000] + "\nFix the cause and reply with the complete code block."}]


def main():
    task = json.loads(Path(sys.argv[1]).read_text())
    started = time.time()
    state = {"calls": 0, "limit": task["limits"]["llm_calls"], "notes": {"name": "recall_raw",
        "workflow_version": WORKFLOW_VERSION, "llm_errors": [], "attempts": []}}
    context = build_context(task, state)
    state["notes"]["context_chars"] = len(context)
    program = write_program(task, state, context)
    state["notes"]["seconds"] = round(time.time() - started, 1)
    base.call("done", program=program, notes={"workflow_seed": state["notes"]})


if __name__ == "__main__":
    main()
