"""Personalized-harness generator: evidence loop.

Run by the host as `python generator.py task.json` in a sandbox without network. Each step gpt-6-luna chooses one action:
investigate the user's history through the host's history tools (when loop.json enables them), write or edit the user
section, or submit. Written sections are checked by the host; only a section that passed the check is submitted.
The model keeps an evidence ledger (claims with supporting and counter event ids) that is shown back every step.
"""
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "#@USER_SECTION@#"
CONFIG = json.loads((HERE / "loop.json").read_text())
TOOL_ACTIONS = ("search", "list", "read")


def load_digest():
    spec = importlib.util.spec_from_file_location("phg_digest", HERE / "digest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIGEST = load_digest()
call = DIGEST.call


def parse_action(reply):
    match = re.search(r"```json\s*\n?(.*?)(?:```|$)", reply, re.S)
    text = match.group(1) if match else reply.strip()
    try:
        action = json.loads(text)
    except ValueError as error:
        raise ValueError(f"the reply must start with one ```json block holding one action object ({error})") from None
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ValueError("the json block must be an object with an \"action\" field")
    return action


def python_block(reply):
    blocks = re.findall(r"```python\s*\n(.*?)```", reply, re.S)
    return blocks[-1].strip() if blocks else None


def ledger_text(ledger):
    if not ledger:
        return "(empty)"
    lines = []
    for index, entry in enumerate(ledger[-CONFIG["ledger_shown"]:]):
        lines.append(f"{index + 1}. [{entry.get('confidence', '?')}] {entry.get('claim', '')}"
            f" | support {entry.get('support', [])} | counter {entry.get('counter', [])}")
    return "\n".join(lines)


def update_ledger(ledger, action):
    entries = action.get("ledger") or []
    if not isinstance(entries, list):
        return
    for entry in entries[:CONFIG["ledger_per_step"]]:
        if isinstance(entry, dict) and isinstance(entry.get("claim"), str) and entry["claim"].strip():
            ledger.append({"claim": entry["claim"][:400], "support": [str(x) for x in entry.get("support") or []][:8],
                "counter": [str(x) for x in entry.get("counter") or []][:8], "confidence": str(entry.get("confidence") or "")[:10]})


def main():
    task = json.loads(Path(sys.argv[1]).read_text())
    started = time.time()
    limits, language = task["limits"], task["language"]
    backbone = (HERE / "backbone.py").read_text()
    prompt = (HERE / "prompt.md").read_text().split("## Output", 1)[0]
    loop = (HERE / "loop.md").read_text()
    tools = (HERE / "history_tools.md").read_text() if CONFIG["history_tools"] else ""
    fields = {"<<CONTROL_SECTION>>": (HERE / "control_section.py").read_text().strip(),
        "<<LANGUAGE>>": "builtins " + ", ".join(language["safe_builtins"]) + "; string/list/dict methods "
            + ", ".join(language["safe_methods"]),
        "<<MAX_CHARS>>": str(limits["program_chars"]), "<<MAX_NODES>>": str(limits["program_nodes"]),
        "<<SECTION_CHARS>>": str(CONFIG["section_chars"]), "<<BACKBONE>>": backbone.strip(),
        "<<DIGEST>>": DIGEST.digest(task["events"])[:CONFIG["digest_chars"]], "<<HISTORY_TOOLS>>": tools,
        "<<STEPS>>": str(limits["llm_calls"]), "<<WRITE_BY>>": str(CONFIG["write_by_step"])}
    text = prompt + loop
    for key, value in fields.items():
        text = text.replace(key, value)
    notes = {"variant": CONFIG["variant"], "prompt_chars": len(text), "steps": [], "ledger": []}
    opening = "Begin."
    if CONFIG["history_tools"]:
        overview = call("history", action="overview")
        opening = "Begin. History overview from the host:\n" + json.dumps(overview, ensure_ascii=False)[:CONFIG["result_chars"]]
    messages = [{"role": "user", "content": text}, {"role": "user", "content": opening}]
    section, passed, ledger = None, None, []
    for step in range(1, limits["llm_calls"] + 1):
        reply = call("llm", messages=messages, max_tokens=CONFIG["max_tokens"], effort=CONFIG["effort"])
        if reply.get("error"):
            notes["steps"].append({"step": step, "llm_error": reply["error"]})
            break
        content = reply["content"]
        messages.append({"role": "assistant", "content": content})
        record = {"step": step}
        try:
            action = parse_action(content)
            name = action["action"]
            record["action"] = name
            update_ledger(ledger, action)
            if name in TOOL_ACTIONS and CONFIG["history_tools"]:
                payload = {key: action[key] for key in ("query", "filters", "limit", "order", "ids") if key in action}
                answer = call("history", action=name, **payload)
                result = json.dumps(answer, ensure_ascii=False)[:CONFIG["result_chars"]]
                record["error"] = answer.get("error")
            elif name in ("write", "edit"):
                if name == "write":
                    candidate = python_block(content)
                    if candidate is None:
                        raise ValueError("write needs the complete section in one ```python block after the json block")
                else:
                    if section is None:
                        raise ValueError("there is no section to edit yet; use write")
                    candidate = section
                    edits = action.get("edits")
                    if not isinstance(edits, list) or not edits:
                        raise ValueError("edit needs edits: a nonempty list of {old, new}")
                    for edit in edits:
                        old, new = edit.get("old"), edit.get("new")
                        if not isinstance(old, str) or not isinstance(new, str) or candidate.count(old) != 1:
                            raise ValueError("each edit needs an old string occurring exactly once in the current section")
                        candidate = candidate.replace(old, new)
                section = candidate
                check = call("check", program=backbone.replace(MARKER, section + "\n"))
                record.update(section_chars=len(section), ok=check.get("ok"), error=check.get("error"))
                if check.get("ok"):
                    passed = section
                    result = "The host check passed. This section is now the submittable version."
                else:
                    result = "The host rejected the program:\n" + str(check.get("error"))[:3000]
            elif name == "submit":
                if passed is not None:
                    notes["steps"].append(record)
                    break
                result = "Nothing to submit: no section has passed the host check yet."
            else:
                raise ValueError("unknown action " + repr(name))
        except ValueError as error:
            record["error"] = str(error)
            result = "Action rejected: " + str(error)
        notes["steps"].append(record)
        remaining = limits["llm_calls"] - step
        guidance = ""
        if passed is None and remaining <= CONFIG["force_write_remaining"]:
            guidance = "\nYou must write a complete passing section now."
        messages.append({"role": "user", "content": f"Result of step {step}:\n{result}\n\nLedger:\n{ledger_text(ledger)}\n\n"
            f"Remaining model steps: {remaining}." + guidance})
    notes.update(ledger=ledger, seconds=round(time.time() - started, 1))
    call("done", program=backbone.replace(MARKER, passed + "\n") if passed is not None else None, notes=notes)


if __name__ == "__main__":
    main()
