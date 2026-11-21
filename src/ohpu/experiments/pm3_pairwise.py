"""PersonaMem-v3 pick-the-better items on the personalized-harness runtime.

Each program-scored PM3 item (a user request with the benchmark's correct response and its typed inferior response) is
shown as two candidate responses in a hash-randomized A/B order; the executor answers one letter and exact match scores
it. No judge model is involved. Every item sees only the user's sanitized app events strictly before the item time
(the loader's official visibility firewall), held in the same programmable memory store that generated harnesses
query through recall; the answer call is the harness's single model call.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

TASK_KIND = "pm3_pairwise"
EVENT_CHARS = 320
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "pm3_choice", "strict": True, "schema": {
    "type": "object", "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
    "required": ["choice"], "additionalProperties": False}}}
INSTRUCTION = ("Two candidate responses answer this user's request. Pick the one that is better for THIS user, using what "
    "the supplied history shows about them. Some requests call for personalization; others call for restraint (do not "
    "over-personalize, flatter, or surface sensitive or irrelevant personal details). "
    'Return JSON {"choice": "A"} or {"choice": "B"}.')


def _clip(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


def event_text(content: Any, kind: str) -> str:
    """One compact line per sanitized event: time, app, action, feedback type, hashtags, inferred preferences, content."""
    event = content.get("_ohpu_agent_visible_event", content) if isinstance(content, dict) else {}
    if kind == "calendar":
        entry = event.get("entry") or {}
        return _clip(f"[{event.get('formatted_timestamp', '')}] Calendar {event.get('action', '')}: {entry.get('title', '')} "
            f"({entry.get('type', '')})", EVENT_CHARS)
    form = event.get("interaction_format") or {}
    parts = [f"[{event.get('formatted_timestamp', '')}] {form.get('app') or kind}", form.get("action_label") or form.get("action") or "",
        f"feedback={event.get('source_interaction_type', '')}"]
    if event.get("source_hashtags"):
        parts.append(" ".join(event["source_hashtags"][:8]))
    prefs = [item.get("persona_item") for item in event.get("preferences") or [] if isinstance(item, dict) and item.get("persona_item")]
    if prefs:
        parts.append("pref: " + "; ".join(prefs[:3]))
    if form.get("user_message"):
        parts.append("said: " + str(form["user_message"]))
    turns = [turn.get("content", "") for turn in event.get("conversation") or [] if isinstance(turn, dict) and turn.get("role") == "user"]
    if turns:
        parts.append("asked: " + turns[0])
    return _clip(" | ".join(part for part in parts if part), EVENT_CHARS)


def _response(value: Any) -> str:
    return str(value.get("text", "")) if isinstance(value, dict) else str(value or "")


def prepare_user(user) -> dict:
    """Events (time order) and pairwise items with their legal cutoff: events strictly before the item time."""
    events, items = [], []
    for episode in sorted(user.episodes, key=lambda episode: episode.order):
        for history in episode.history_delta:
            events.append({"id": history.item_id, "ts": int(history.date), "app": history.kind, "text": event_text(history.content, history.kind)})
        cutoff = len(events)
        inferior = episode.meta.get("inferior_response")
        if not inferior or episode.gold_actions:
            continue
        good, bad = _response(episode.meta.get("example_response")), _response(inferior)
        if not good.strip() or not bad.strip() or good.strip() == bad.strip():
            continue
        flip = hashlib.sha256(f"pm3-pairwise-order-v1:{episode.episode_id}".encode()).digest()[0] % 2 == 1
        options = {"A": bad, "B": good} if flip else {"A": good, "B": bad}
        items.append({"case_id": f"pm3::{user.user_id}::{episode.episode_id}", "query_id": episode.episode_id, "cutoff": cutoff,
            "query": episode.instruction, "prior_conversation": episode.message_history or [], "options": options,
            "candidates": episode.meta.get("candidates") or [], "target": "B" if flip else "A",
            "inferior_flaw": inferior.get("flaw_kind") if isinstance(inferior, dict) else None,
            "task_type": episode.meta.get("task_type") or "", "expected_behavior": episode.meta.get("expected_behavior") or ""})
    return {"user_id": str(user.user_id), "events": events, "items": items}


def prompt(item: dict) -> str:
    lines = ["User request: " + _clip(item["query"], 2000)]
    if item["prior_conversation"]:
        lines.append("Earlier in this conversation: " + _clip(json.dumps(item["prior_conversation"], ensure_ascii=False), 1500))
    if item["candidates"]:
        lines.append("Candidate items (indexes used by rankings): " + "; ".join(
            f"[{index}] {_clip(candidate.get('title') or candidate.get('caption') or '', 120)} {' '.join((candidate.get('hashtags') or [])[:5])}"
            for index, candidate in enumerate(item["candidates"])))
    lines += ["Response A: " + _clip(item["options"]["A"], 2500), "Response B: " + _clip(item["options"]["B"], 2500), INSTRUCTION]
    return "\n\n".join(lines)


def writer() -> str:
    """Store writer identity: the fixed remember hook every PM3 program shares (the host plays its role)."""
    from ohpu.per_user_adaptation.program import HarnessProgram

    return HarnessProgram(uniform_programs()["no_history"]).fingerprint.hook_sha256["remember"]


def documents(prepared: dict) -> dict[int, str]:
    """Programmable memory documents, one per distinct item cutoff, built host-side in one pass (records never shrink)."""
    from ohpu.per_user_adaptation import memory_backend

    wanted, result, records, identity = sorted({item["cutoff"] for item in prepared["items"]}), {}, [], writer()
    events = prepared["events"]
    position = 0
    for cutoff in wanted:
        while position < cutoff:
            event = events[position]
            day = str(event["ts"])
            records.append({"collection": "observations", "key": f"e{position:05d}", "data": {"app": event["app"], "ordinal": position},
                "text": event["text"], "vector": [], "sources": [hashlib.sha256(f"{prepared['user_id']}:{event['id']}".encode()).hexdigest()],
                "updated_day": day, "expires_on": None})
            position += 1
        payload = {"protocol": memory_backend.PROTOCOL, "user_id": prepared["user_id"], "writer_sha256": identity, "schemas": {},
            "records": copy.copy(records)}
        day = str(events[cutoff - 1]["ts"]) if cutoff else "0"
        result[cutoff] = f"# USER MEMORY (last activity seen: {day})\n\n{memory_backend.HEADING}\n{memory_backend.encode(payload)}\n"
    return result


def run_case(client, prepared: dict, item: dict, source: str, document: str, *, seed: int, full_history_chars: int = 0,
             capture_context: bool = False) -> dict:
    """One pick-the-better answer through the harness runtime; returns a scored row. With ``capture_context``, a wrong
    answer also keeps the executor's system message (the harness prefix and recalled context) for update evidence."""
    from ohpu.per_user_adaptation.core.models import make_minimal_config
    from ohpu.per_user_adaptation.executor import ExecutableHarness, Runner
    from ohpu.per_user_adaptation.program import HarnessProgram
    from ohpu.experiments.lamp import SeededLuna

    program = HarnessProgram(source)
    harness = ExecutableHarness(prepared["user_id"], make_minimal_config(), memory_md=document, program=program,
        memory_backend_enabled=True, memory_as_of=None, memory_full_history_chars=full_history_chars)
    runner = Runner(SeededLuna(client, seed, model=client.model), harness, capture_execution=True)
    text = prompt(item)
    row = {"case_id": item["case_id"], "user_id": prepared["user_id"], "query_id": item["query_id"], "generation_seed": seed,
        "task_type": item["task_type"], "expected_behavior": item["expected_behavior"], "target": item["target"],
        "program_sha256": program.fingerprint.sha256}
    try:
        trace = runner.run(episode_id=item["case_id"], user_id=prepared["user_id"], condition="pm3_pairwise",
            messages=[{"role": "user", "content": text}], query=text, task_kind=TASK_KIND, max_tokens=32, response_format=RESPONSE_FORMAT)
    except ValueError as error:
        # A program that breaks the memory protocol at runtime gets no answer and scores zero with a receipt; other errors abort.
        return {**row, "final_output": "", "reward": 0.0, "num_llm_calls": 0, "total_tokens": 0, "degradations": [],
            "runtime_violation": f"{type(error).__name__}: {error}"[:300]}
    try:
        choice = json.loads(trace.final).get("choice")
    except (TypeError, ValueError, AttributeError):
        choice = next((letter for letter in ("A", "B") if str(trace.final).strip().upper().startswith(letter)), None)
    context = {}
    if capture_context and choice != item["target"]:
        request = next((event for event in trace.execution or [] if event["kind"] == "model_request"), None)
        system = [message["content"] for message in (request or {}).get("messages", []) if message.get("role") == "system"]
        context = {"executor_system": system[0][:14000] if system else ""}
    return {**row, "final_output": choice or str(trace.final)[:40], "reward": float(choice == item["target"]),
        "num_llm_calls": trace.model_calls, "total_tokens": int(trace.usage["prompt"]) + int(trace.usage["completion"]),
        "degradations": list(trace.degradations)[:5], "runtime_violation": None, **context}


class _StubClient:
    model = "pm3-smoke-check"

    def chat(self, messages, **options):
        return {"content": '{"choice": "A"}', "tool_calls": [], "usage": {"prompt": 0, "completion": 0}, "finish_reason": "stop"}


def smoke_check(source: str, history: list[dict], requests: list[dict]) -> int:
    """Run a program end to end on sample requests over a store of the given history lines ({"app", "text"}) with a stub
    model: no model call and no scoring. Raises on a runtime protocol violation, so crashing programs are caught early."""
    validate_program(source)
    events = [{"id": f"h{index}", "ts": index, "app": line.get("app", ""), "text": line["text"]} for index, line in enumerate(history)]
    items = [{"case_id": f"smoke-{index}", "query_id": f"smoke-{index}", "cutoff": len(events), "query": request["request"],
        "prior_conversation": request.get("prior_conversation") or [], "options": {"A": request["response_A"], "B": request["response_B"]},
        "candidates": request.get("candidates") or [], "target": "A", "task_type": request.get("task_type", ""),
        "expected_behavior": request.get("expected_behavior", "")} for index, request in enumerate(requests)]
    prepared = {"user_id": "smoke-check", "events": events, "items": items}
    document = documents(prepared)[len(events)] if items else ""
    for item in items:
        row = run_case(_StubClient(), prepared, item, source, document, seed=0)
        if row["runtime_violation"]:
            raise ValueError(f"runtime violation on a {item['task_type'] or 'sample'} request: {row['runtime_violation']}")
    return len(items)


def smoke_requests(train_items: list[dict]) -> list[dict]:
    """One labelled TRAIN example per task type (first occurrence), the fixed sample for smoke checks."""
    chosen = {}
    for item in train_items:
        chosen.setdefault(item.get("task_type", ""), item)
    return list(chosen.values())


def check_literals(source: str, texts, identifiers, *, literal_chars: int = 40) -> None:
    """A shared reference's string literals may not name a registered user or copy TRAIN item text."""
    import ast

    literals = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    if literals & {str(identifier) for identifier in identifiers}:
        raise ValueError("reference harness contains a registered user identifier")
    corpus = "\x00".join(str(text or "") for text in texts)
    if any(len(value) >= literal_chars and value in corpus for value in literals):
        raise ValueError("reference harness copies TRAIN item text")


def check_reference(source: str, histories: list[list[dict]], train_items: list[dict], identifiers, *, literal_chars: int = 40) -> str:
    """Host checks for a shared reference harness: a valid program whose string literals name no registered user and copy
    no TRAIN item text, and which runs without a protocol violation over every given history."""
    digest = validate_program(source)
    check_literals(source, [item.get(key) for item in train_items for key in ("request", "response_A", "response_B")], identifiers,
        literal_chars=literal_chars)
    requests = smoke_requests(train_items)
    for history in histories:
        smoke_check(source, history, requests)
    return digest


# ---- harness programs -------------------------------------------------------------------------------------------
# remember/guard are fixed: the host builds the event store (one record per event), so remember is never invoked.
FIXED_HOOKS = '''def remember(context):
    return {}


def guard(context):
    return {}
'''

RECALL_SEARCH = '''def recall(context):
    if context.get("stage") != "context" or not context.get("backend"):
        return {}
    settings = USER_MODEL["retrieval"]
    backend = context["backend"]
    if backend.get("step", 0) == 0:
        request = context.get("query", "").split("Response A:", 1)[0]
        return {"memory_query": {"operation": "search", "algorithm": settings["mode"], "collection": "observations",
            "text": request, "limit": settings["top_k"]}}
    texts, refs, size = [], [], 0
    for row in backend.get("result", {}).get("data", []):
        text = row.get("text", "")
        if size + len(text) + 1 <= settings["max_chars"]:
            texts.append(text)
            refs.append(row["ref"])
            size += len(text) + 1
    return {"memory_query": {"operation": "finish", "output": "\\n".join(texts), "source_refs": refs}}
'''

RECALL_RECENT = '''def recall(context):
    if context.get("stage") != "context" or not context.get("backend"):
        return {}
    settings = USER_MODEL["retrieval"]
    backend = context["backend"]
    if backend.get("step", 0) == 0:
        return {"memory_query": {"operation": "sql", "sql": "SELECT text, ref FROM records WHERE collection = ? "
            "ORDER BY CAST(json_extract(data, '$.ordinal') AS INTEGER) DESC LIMIT ?", "parameters": ["observations", settings["top_k"]]}}
    rows = list(reversed((backend.get("result", {}).get("data") or {}).get("rows", [])))
    texts, refs, size = [], [], 0
    for row in rows:
        text = row.get("text", "")
        if size + len(text) + 1 <= settings["max_chars"]:
            texts.append(text)
            refs.append(row["ref"])
            size += len(text) + 1
    return {"memory_query": {"operation": "finish", "output": "\\n".join(texts), "source_refs": refs}}
'''

RECALL_NONE = '''def recall(context):
    if context.get("stage") != "context" or not context.get("backend"):
        return {}
    return {"memory_query": {"operation": "finish", "output": "", "source_refs": []}}
'''

RECALL_SUMMARY = '''def recall(context):
    if context.get("stage") != "context" or not context.get("backend"):
        return {}
    backend = context["backend"]
    if backend.get("step", 0) == 0:
        return {"memory_query": {"operation": "sql", "sql": "SELECT text, ref FROM records WHERE collection = ?", "parameters": ["summary"]}}
    texts, refs = [], []
    for row in (backend.get("result", {}).get("data") or {}).get("rows", []):
        texts.append(row.get("text", ""))
        refs.append(row["ref"])
    return {"memory_query": {"operation": "finish", "output": "\\n".join(texts), "source_refs": refs}}
'''

APPLY_NEUTRAL = '''def apply(context):
    if context.get("stage") != "generation" or context.get("task_kind") != "pm3_pairwise":
        return {}
    base = context.get("base", {})
    if "prefix" not in base.get("expressible", []):
        return {}
    advice = "Judge both responses against this user's supplied history."
    return {"prefix": (base.get("prefix", "") + "\\n" + advice).strip()[:800]}
'''


def program(architecture: str, recall: str, *, mode="bm25", top_k=10, max_chars=6000) -> str:
    model = {"architecture": architecture, "retrieval": {"mode": mode, "top_k": top_k, "max_chars": max_chars}}
    return ("PROGRAM_API_VERSION = 1\nUSER_MODEL = " + repr(model) + "\nCHANGE_MANIFEST = {}\n\n\n" + FIXED_HOOKS + "\n\n"
        + recall + "\n\n" + APPLY_NEUTRAL)


def uniform_programs() -> dict[str, str]:
    """Fixed programs shared by every user."""
    return {"no_history": program("no_history", RECALL_NONE), "recent_20": program("recent", RECALL_RECENT, mode="recent", top_k=20),
        "recent_64": program("recent", RECALL_RECENT, mode="recent", top_k=64, max_chars=12000),
        "bm25_10": program("bm25", RECALL_SEARCH), "bm25_30": program("bm25", RECALL_SEARCH, top_k=30, max_chars=11000),
        "tfidf_10": program("tfidf", RECALL_SEARCH, mode="tfidf"), "summary": program("summary", RECALL_SUMMARY, mode="summary")}


def validate_program(source: str) -> str:
    """Generated programs: registered hooks, fixed remember/guard, safe memory decisions, no complete-history read."""
    import ast
    from ohpu.per_user_adaptation.program import HarnessProgram
    from ohpu.experiments.lamp_seed_harness import validate_decisions

    program_ = HarnessProgram(source)
    tree = ast.parse(source)
    functions = {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    fixed = {node.name: ast.dump(node) for node in ast.parse(FIXED_HOOKS).body}
    if any(functions.get(name) != dump for name, dump in fixed.items()):
        raise ValueError("remember and guard are fixed; copy them unchanged from blank_harness.py")
    if "read_all" in source:
        raise ValueError("read_all is unavailable: build compact context with search, SQL or deterministic summaries")
    validate_decisions(source)
    return program_.fingerprint.sha256


def summarize_events(client, prepared: dict, *, batch=40, limit=1500, seed=7) -> dict[int, str]:
    """Iterative summary baseline (host-side): the executor rewrites a bounded summary after every run of at most `batch` new
    events and at every item time. Returns, per item cutoff, the summary of all earlier events plus the last 10 raw events."""
    events, cutoffs = prepared["events"], sorted({item["cutoff"] for item in prepared["items"]})
    boundaries = sorted(set(range(batch, cutoffs[-1], batch)) | set(cutoffs)) if cutoffs else []
    states, summary, start = {}, "", 0
    for end in boundaries:
        if end > start:
            chunk = "\n".join(event["text"] for event in events[start:end])
            response = client.chat([{"role": "system", "content": "You maintain a concise, factual summary of one user's preferences and "
                "behaviour from their app activity. Record what they engage with or like (positive feedback), what they skip or dislike "
                "(negative feedback), sensitive topics, and any requests about personalization. Never copy raw events."},
                {"role": "user", "content": f"Current summary:\n{summary or '(empty)'}\n\nNew events:\n{chunk}\n\nReturn the complete updated "
                f"summary in at most {limit} characters."}], max_tokens=700, seed=seed)
            summary = _clip(response.get("content") or summary, limit)
        states[end], start = summary, end
    result = {}
    for cutoff in cutoffs:
        tail = [event["text"] for event in events[max(0, cutoff - 10):cutoff]]
        result[cutoff] = (states[cutoff] + ("\nRecent events:\n" + "\n".join(tail) if tail else "")).strip()
    return result


def summary_documents(prepared: dict, summaries: dict[int, str]) -> dict[int, str]:
    """Memory documents holding one summary record per cutoff (the summary baseline reads only that record)."""
    from ohpu.per_user_adaptation import memory_backend

    result, identity = {}, writer()
    for cutoff, text in summaries.items():
        record = {"collection": "summary", "key": "s", "data": {"cutoff": cutoff}, "text": text or "(no history yet)", "vector": [],
            "sources": [hashlib.sha256(f"{prepared['user_id']}:summary:{cutoff}".encode()).hexdigest()],
            "updated_day": str(prepared["events"][cutoff - 1]["ts"]) if cutoff else "0", "expires_on": None}
        payload = {"protocol": memory_backend.PROTOCOL, "user_id": prepared["user_id"], "writer_sha256": identity, "schemas": {}, "records": [record]}
        result[cutoff] = f"# USER MEMORY (last activity seen: {record['updated_day']})\n\n{memory_backend.HEADING}\n{memory_backend.encode(payload)}\n"
    return result
