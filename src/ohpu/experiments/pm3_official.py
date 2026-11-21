"""PersonaMem-v3's own released tasks on the personalized-harness runtime.

Every query runs through the pinned official evaluator (its task builders, scorers and LLM judge) in llm_memory mode via
the OHPU bridge (``pm3_native``): the bridge removes the evaluator's history section from the official prompt, and the
harness program supplies the user's context from a programmable store holding one record per event strictly before the
query time. A program that breaks the memory protocol at runtime leaves the request unanswered (the official scorer
then scores the empty answer) and the violation is recorded.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from typing import Any

from ohpu.benchmarks import pm3_history, pm3_native

EXECUTOR_NAME = "Qwen3.5-9B"
JUDGE_MODEL = "gpt-5.5"  # the official evaluator's default judge
STORE_POINTER = "(This user's history is available to the harness only through its memory store.)"
# With hidden task types a program sees this task kind for every request and must infer the task from the request text.
HIDDEN_TASK = "request"
CONTEXT_CHARS = 14000
HISTORY_CONTRACT = "pm3-native-structured-history-v2"
STORE_LIMITS = {"max_store_chars": 16_000_000 * int(os.environ.get("OHPU_HOST_LIMIT_SCALE", "1")),
    "max_record_chars": 64_000 * int(os.environ.get("OHPU_HOST_LIMIT_SCALE", "1"))}
APPLY_NONE = '''def apply(context):
    return {}
'''
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
        request = context.get("query", "")[:8000]
        return {"memory_query": {"operation": "search", "algorithm": settings["mode"], "collection": "observations",
            "text": request, "limit": settings["top_k"]}}
    texts, refs, size = [], [], 0
    for row in backend.get("result", {}).get("data", []):
        remaining = settings["max_chars"] - size - 1
        if remaining <= 0:
            break
        text = row.get("text", "")[:remaining]
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
        remaining = settings["max_chars"] - size - 1
        if remaining <= 0:
            break
        text = row.get("text", "")[:remaining]
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


def program(architecture, recall, *, mode="bm25", top_k=10, max_chars=6000):
    model = {"architecture": architecture, "retrieval": {"mode": mode, "top_k": top_k, "max_chars": max_chars}}
    return ("PROGRAM_API_VERSION = 1\nUSER_MODEL = " + repr(model) + "\nCHANGE_MANIFEST = {}\n\n\n" + FIXED_HOOKS + "\n\n"
        + recall + "\n\n" + APPLY_NONE)


def uniform_programs() -> dict[str, str]:
    """Native fixed controls: official requests, no A/B response prompt or answer guidance."""
    return {"no_history": program("no_history", RECALL_NONE), "recent_20": program("recent", RECALL_RECENT, mode="recent", top_k=20),
        "recent_64": program("recent", RECALL_RECENT, mode="recent", top_k=64, max_chars=12000),
        "bm25_10": program("bm25", RECALL_SEARCH), "bm25_30": program("bm25", RECALL_SEARCH, top_k=30, max_chars=11000),
        "tfidf_10": program("tfidf", RECALL_SEARCH, mode="tfidf"), "summary": program("summary", RECALL_SUMMARY, mode="summary")}


def validate_program(source: str, *, editable_hooks: bool = False) -> str:
    from ohpu.per_user_adaptation.program import HarnessProgram
    from ohpu.experiments.lamp_seed_harness import validate_decisions

    program = HarnessProgram(source)
    if not editable_hooks:
        functions = {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        fixed = {node.name: ast.dump(node) for node in ast.parse(FIXED_HOOKS).body}
        if any(functions.get(name) != value for name, value in fixed.items()):
            raise ValueError("remember and guard are fixed; copy them unchanged from blank_harness.py")
    if "read_all" in source:
        raise ValueError("read_all is unavailable: build compact context with search or SQL")
    validate_decisions(source)
    return program.fingerprint.sha256


class StoreSnapshot(pm3_native.OnlineSnapshotCache):
    """The official builder's memory slot receives a pointer instead of the store: the bridge strips that section, and
    the harness reads the store (larger than the bridge's snapshot bound) through recall."""

    def attach_memory(self, memory: str | None) -> None:
        super().attach_memory(STORE_POINTER)


def official_runtime(evaluator_root, backend_dir, user_id: str, *, judge: bool = True):
    from ohpu.experiments import pm3_proxy

    runtime = pm3_native.OfficialPM3Runtime(evaluator_root, backend_dir, user_id, model_name=EXECUTOR_NAME, enable_llm_judge=judge)
    # The judge's reply parser is loaded from the verified checkout, which may sit outside this source tree's data root.
    pm3_proxy.configure_official_scoring(evaluator_root, pm3_native.OFFICIAL_EVALUATOR_REVISION)
    runtime.snapshot_cache = StoreSnapshot()
    return runtime


def released_queries(runtime) -> list[dict]:
    """The user's released queries in time order, without task types the official registry retired."""
    from evaluation.task_registry import DROPPED_TASK_TYPES, normalize_task_type

    return [row for row in runtime.load_queries() if normalize_task_type(str(row.get("task_type") or "")) not in DROPPED_TASK_TYPES]


def retired_query_ids(rows: list[dict]) -> list[str]:
    from evaluation.task_registry import DROPPED_TASK_TYPES, normalize_task_type

    return [row["query_id"] for row in rows if normalize_task_type(str(row.get("task_type") or "")) in DROPPED_TASK_TYPES]


def split(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Evaluate the later half by time. A lifecycle pre/post pair cut by the boundary leaves the evaluation (its official
    metric needs both sides, and the earlier side predates the history the personalized harness is written from)."""
    evaluated = rows[len(rows) // 2:]
    inside = {row["query_id"] for row in evaluated}
    sides: dict[str, list[str]] = {}
    for row in rows:
        identity = pm3_native._lifecycle_identity({"query_id": row["query_id"]})
        if identity:
            sides.setdefault(identity[0], []).append(row["query_id"])
    broken = {query for queries in sides.values() if not all(query in inside for query in queries) for query in queries}
    return [row for row in evaluated if row["query_id"] not in broken], sorted(broken & inside)


def complete_split(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """The later half by time without lifecycle sides that lack exactly one partner inside it. Such a side can only score
    as unpaired (no official lifecycle score, so no change to the micro) and would fail the metric-completeness check."""
    evaluated = rows[len(rows) // 2:]
    identities = {row["query_id"]: pm3_native._lifecycle_identity({"query_id": row["query_id"]}) for row in evaluated}
    phases: dict[str, list[str]] = {}
    for identity in identities.values():
        if identity:
            phases.setdefault(identity[0], []).append(identity[1])
    broken = {key for key, values in phases.items() if sorted(values) != ["post", "pre"]}
    dropped = [query for query, identity in identities.items() if identity and identity[0] in broken]
    return [row for row in evaluated if row["query_id"] not in dropped], dropped


def events(user) -> list[dict]:
    """Preserve complete agent-visible events without reading profile or query labels."""
    result = []
    for episode in sorted(user.episodes, key=lambda item: item.order):
        for history in episode.history_delta:
            event = pm3_history.unwrap_event(history.content)
            if event is None:
                if history.kind != "calendar" or not isinstance(history.content, dict):
                    raise ValueError("native history requires a sanitized PM3 event")
                event = history.content
            event = pm3_history.sanitize_event(event)
            result.append({"id": history.item_id, "ts": int(history.date), "app": history.kind,
                "text": json.dumps(event, ensure_ascii=False, sort_keys=True), "event": event,
                "history_contract": HISTORY_CONTRACT})
    return sorted(result, key=lambda event: event["ts"])


def cutoff(events_: list[dict], timestamp: int) -> int:
    return sum(event["ts"] < int(timestamp) for event in events_)


def remember_keeps_default(source: str) -> bool:
    """Whether the remember hook always returns {} (every return is an empty dict literal, no raise): replaying it would
    keep every host-inserted observation, so stores can be built without calling it per event."""
    hook = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "remember")
    returns = [node for node in ast.walk(hook) if isinstance(node, ast.Return)]
    return bool(returns) and all(isinstance(node.value, ast.Dict) and not node.value.keys for node in returns) \
        and not any(isinstance(node, ast.Raise) for node in ast.walk(hook))


def documents(user_id: str, events_: list[dict], cutoffs, *, source: str | None = None, checkpoint: dict | None = None,
              checkpoint_sink: dict | None = None) -> dict[int, str]:
    from ohpu.per_user_adaptation import memory_backend
    from ohpu.per_user_adaptation.program import HarnessProgram

    source = source or uniform_programs()["no_history"]
    validate_program(source, editable_hooks=True)
    program = HarnessProgram(source)
    replay_remember = not remember_keeps_default(source)
    wanted = sorted(set(cutoffs))
    if any(type(value) is not int or not 0 <= value <= len(events_) for value in wanted):
        raise ValueError("invalid causal history cutoff")
    store = memory_backend.MemoryStore(user_id, program.fingerprint.hook_sha256["remember"], day="0", **STORE_LIMITS)
    result, position = {}, 0
    if checkpoint is not None:
        position = checkpoint["cutoff"]
        if type(position) is not int or not 0 <= position <= min(wanted, default=len(events_)):
            raise ValueError("checkpoint exceeds the requested causal prefix")
        expected = hashlib.sha256(json.dumps(events_[:position], sort_keys=True).encode()).hexdigest()
        if checkpoint["history_sha256"] != expected:
            raise ValueError("checkpoint history changed")
        payload = memory_backend.MemoryStore.payload(checkpoint["document"], **STORE_LIMITS)
        if payload["user_id"] != user_id or payload["writer_sha256"] != store.writer_sha256:
            raise ValueError("checkpoint owner or remember implementation changed")
        store.schemas = payload["schemas"]
        store.records = {(record["collection"], record["key"]): record for record in payload["records"]}
        store.day = str(events_[position - 1]["ts"]) if position else "0"
        for index, event in enumerate(events_[:position]):
            store.sources[f"event:{index}"] = [hashlib.sha256(f"{user_id}:{event['id']}".encode()).hexdigest()]
        for record in store.records.values():
            store.sources[store.reference(record)] = list(record["sources"])
        if "sources" in checkpoint:
            store.sources = {key: list(value) for key, value in checkpoint["sources"].items()}
        elif replay_remember:
            raise ValueError("A custom remember checkpoint requires complete source references")

    def invoke(context):
        return program.call("remember", context).patch

    def no_model(*args, **kwargs):
        raise ValueError("remember cannot call models")

    for limit in wanted:
        while position < limit:
            event = events_[position]
            store.day = str(event["ts"])
            key, ref = f"e{position:05d}", f"event:{position}"
            receipts = [hashlib.sha256(f"{user_id}:{event['id']}".encode()).hexdigest()]
            store.sources[ref] = receipts
            data = {"app": event["app"], "ordinal": position, "ts": event["ts"], "event_id": event["id"]}
            native = event.get("event")
            if native is not None:
                data["history_contract"] = HISTORY_CONTRACT
                data.update({name: native[name] for name in ("source_object_id", "author_id", "relationship", "is_self_authored",
                    "is_dm", "thread_id", "recipient_id", "participants", "event_location") if name in native})
            record = {"collection": "observations", "key": key, "data": data, "text": event["text"], "vector": [],
                "sources": receipts, "updated_day": str(event["ts"]), "expires_on": None}
            if len(memory_backend.encode(record)) > STORE_LIMITS["max_record_chars"]:
                raise ValueError("native event exceeds registered record capacity; refusing to truncate")
            store.records["observations", key] = record
            store.sources[store.reference(record)] = receipts
            context = {"stage": "memory_write", "source_events": [{"index": position, "key": key, "ref": ref,
                "ts": event["ts"], "app": event["app"], "text": event["text"], **({"event": native} if native is not None else {})}]}
            initial = invoke({**context, "backend": {**store.describe(), "phase": "write", "step": 0,
                "state": {}, "result": {"data": None, "truncated": False}, "model_calls_remaining": 0}}) if replay_remember else {}
            if initial:
                memory_backend.execute(store, phase="write", initial=initial, invoke=invoke, context=context,
                    model_call=no_model, record=lambda *args, **kwargs: None, model_budget=0)
            position += 1
        result[limit] = store.snapshot()
        if checkpoint_sink is not None and limit in (wanted[0], wanted[-1]):
            checkpoint_sink[limit] = {"sources": {key: list(value) for key, value in store.sources.items()}}
    return result


def summary_documents(client, user_id: str, events_: list[dict], cutoffs) -> tuple[dict[int, str], dict[int, str]]:
    wanted = sorted(set(cutoffs))
    boundaries = sorted(set(range(40, wanted[-1], 40)) | set(wanted)) if wanted else []
    states, summary, start = {}, "", 0
    for end in boundaries:
        if end > start:
            chunk = "\n".join(event["text"] for event in events_[start:end])
            response = client.chat([{"role": "system", "content": "Maintain a factual summary of this user's causal app events. "
                "Preserve preferences, corrections, authorship, voice, time and disclosure boundaries without inventing facts."},
                {"role": "user", "content": f"Current summary:\n{summary}\n\nNew events:\n{chunk}\n\nReturn at most 1500 characters."}],
                max_tokens=700, seed=7)
            summary = str(response.get("content") or summary)[:1500]
        states[end], start = summary, end
    summaries = {limit: (states[limit] + "\nRecent events:\n" + "\n".join(event["text"]
        for event in events_[max(0, limit - 10):limit]))[:12000] for limit in wanted}
    return summaries, documents_from_summaries(user_id, events_, summaries)


def documents_from_summaries(user_id, events_, summaries):
    from ohpu.per_user_adaptation import memory_backend
    from ohpu.per_user_adaptation.program import HarnessProgram

    identity = HarnessProgram(uniform_programs()["no_history"]).fingerprint.hook_sha256["remember"]
    result = {}
    for limit, text in summaries.items():
        store = memory_backend.MemoryStore(user_id, identity, day=str(events_[limit - 1]["ts"]) if limit else "0", **STORE_LIMITS)
        store.sources["summary"] = [hashlib.sha256(f"{user_id}:summary:{limit}".encode()).hexdigest()]
        store.upsert({"collection": "summary", "key": "summary", "data": {"cutoff": limit},
            "text": text or "(no history yet)", "source_refs": ["summary"]})
        result[limit] = store.snapshot()
    return result


_HISTORY_HEADING = re.compile(r"(?im)^[ \t]*##[ \t]+(?:cross-app[ \t]+user|user|full user)[ \t]+history[ \t]*\(time-masked[^\n]*\n?")


def official_request(prompt: Any) -> str:
    """The official prompt without its history: the delimited memory slot and its heading line. The bridge's own section
    stripping also deletes everything after the slot up to the next heading, which removes the output format of the
    proactive tasks (it follows the slot without a heading)."""
    if isinstance(prompt, list):
        source = "\n\n".join(str(message.get("content") or "") for message in prompt
            if isinstance(message, dict) and message.get("role") == "user")
    else:
        source = str(prompt or "")
    request = _HISTORY_HEADING.sub("", pm3_native._HISTORY_SENTINEL_RE.sub("", source)).strip()
    if STORE_POINTER in request or len(request) > pm3_native.MAX_HARNESS_REQUEST_CHARS:
        raise pm3_native.HarnessInputBoundaryError("PM3 request still carries the memory slot or exceeds the harness bound")
    return request


def _adapter_factory(violations: list[str], *, hide_task_type: bool = False):
    class Adapter(pm3_native.HarnessQueryLLM):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if hide_task_type:
                self.task_type = HIDDEN_TASK

        def query_llm(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
            self.current_request = official_request(prompt)
            try:
                return super().query_llm(prompt, *args, **kwargs)
            except pm3_native.HarnessInputBoundaryError:
                raise
            except ValueError as error:
                violations.append(f"{type(error).__name__}: {error}"[:300])
                return ""

    return lambda llm, bundle, query_timestamp, **options: Adapter(llm, bundle, **options)


def _harness(user_id: str, source: str, document: str, *, editable_hooks: bool = False, tool_model_calls: int = 0,
             hook_model_calls: dict | None = None, decision_client=None, decision_calls: int = 0):
    from ohpu.per_user_adaptation import memory_backend
    from ohpu.per_user_adaptation.core.models import make_minimal_config
    from ohpu.per_user_adaptation.executor import ExecutableHarness
    from ohpu.per_user_adaptation.program import HarnessProgram

    if decision_calls and (not editable_hooks or decision_client is None or hook_model_calls is not None):
        raise ValueError("decision-model calls require editable hooks and a host decision client")
    if hook_model_calls is not None:
        if (not editable_hooks or tool_model_calls or set(hook_model_calls) != {"recall", "apply", "guard"}
                or any(type(value) is not int or not 0 <= value <= 1 for value in hook_model_calls.values())):
            raise ValueError("invalid per-hook model-call budget")
        return ExecutableHarness(user_id, make_minimal_config(), memory_md=document, program=HarnessProgram(source),
            memory_backend_enabled=True, memory_as_of=None, guard_repair_enabled=bool(hook_model_calls["guard"]),
            model_call_limit=1 + sum(hook_model_calls.values()), hook_model_calls=dict(hook_model_calls),
            memory_store_limits=dict(STORE_LIMITS))
    if type(tool_model_calls) is not int or not 0 <= tool_model_calls <= (memory_backend.MAX_MODEL_CALLS if editable_hooks else 0):
        raise ValueError("invalid recall model-tool budget")
    return ExecutableHarness(user_id, make_minimal_config(), memory_md=document, program=HarnessProgram(source),
        memory_backend_enabled=True, memory_as_of=None, guard_repair_enabled=not editable_hooks,
        model_call_limit=1 + tool_model_calls if editable_hooks else None, memory_store_limits=dict(STORE_LIMITS),
        decision_client=decision_client, decision_calls=decision_calls)


class _ContextRecorder:
    """Executor wrapper that keeps each call's system message (the harness prefix and recalled context)."""

    def __init__(self, llm):
        self.llm, self.systems = llm, []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)

    def chat(self, messages, **options):
        self.systems.append(next((str(message.get("content") or "") for message in messages if message.get("role") == "system"), ""))
        return self.llm.chat(messages, **options)


def run_query(runtime, row: dict, *, llm, judge, source: str, document: str, run_dir, condition: str, hide_task_type: bool = False,
              capture_context: bool = False, editable_hooks: bool = False, tool_model_calls: int = 0,
              generation_tape=None, replay_generation: bool = False, decision_client=None, decision_calls: int = 0) -> dict:
    """One official query: official prompt without history -> harness (store + program) -> executor -> official score."""
    violations: list[str] = []
    executor = _ContextRecorder(llm) if capture_context else llm
    replay_options = {"generation_tape": generation_tape, "replay_generation": replay_generation} if generation_tape is not None else {}
    if replay_generation and generation_tape is None:
        raise ValueError("Replay requires a persisted generation tape")
    record, audit = runtime.evaluate(row, llm=executor,
        bundle=_harness(runtime.user_id, source, document, editable_hooks=editable_hooks, tool_model_calls=tool_model_calls,
            decision_client=decision_client, decision_calls=decision_calls),
        judge_client=judge,
        condition=condition, harness_checkpoint=0, run_dir=run_dir,
        query_adapter_factory=_adapter_factory(violations, hide_task_type=hide_task_type), **replay_options)
    usage = [trace.get("usage") or {} for trace in audit["harness_traces"]]
    return {**record, "runtime_violations": violations, "executor_calls": 0 if replay_generation else len(usage),
        **({"replayed_executor_calls": len(usage)} if replay_generation else {}),
        "executor_prompt_tokens": sum(int(item.get("prompt", 0) or 0) for item in usage),
        "executor_completion_tokens": sum(int(item.get("completion", 0) or 0) for item in usage),
        "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
        **({"executor_context": [text[:CONTEXT_CHARS] for text in executor.systems[:3]]} if capture_context else {}),
        # Official tasks may make several answer calls; recall tool calls carry the host memory-model system prompt.
        **(_tool_context(executor.systems) if capture_context and tool_model_calls and executor.systems else {})}


def _tool_context(systems: list[str]) -> dict:
    from ohpu.per_user_adaptation.memory_backend import MODEL_SYSTEM_PROMPT

    answers = [text for text in systems if text != MODEL_SYSTEM_PROMPT]
    return {"answer_context": [answers[-1][:CONTEXT_CHARS]] if answers else [],
        "model_tool_calls": len(systems) - len(answers)}


def capture_request(runtime, row: dict, run_dir, *, preflight: dict | None = None) -> str | None:
    """The official request text the executor would receive for ``row`` (history removed), without a model call or
    judge: the capture adapter returns an empty answer to the official scorer."""
    captured: list[str] = []

    class Capture(pm3_native.HarnessQueryLLM):
        def query_llm(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
            captured.append(official_request(prompt))
            return ""

    record, _ = runtime.evaluate(row, llm=None, bundle=_harness(runtime.user_id, uniform_programs()["no_history"], ""), judge_client=None,
        condition="capture", harness_checkpoint=0, run_dir=run_dir,
        query_adapter_factory=lambda llm, bundle, query_timestamp, **options: Capture(llm, bundle, **options))
    if preflight is not None:
        preflight.update(status=record.get("status"), error=record.get("error", ""))
    return captured[0] if captured else None


class _StubClient:
    model = "pm3-native-smoke-check"

    def chat(self, messages, **options):
        return {"content": '{"response": "smoke answer"}', "tool_calls": [],
            "usage": {"prompt": 0, "completion": 0}, "finish_reason": "stop"}


def smoke_check(source: str, history: list[dict], requests: list[dict], *, hide_task_type: bool = False,
                editable_hooks: bool = False, tool_model_calls: int = 0, document: str | None = None,
                user_id: str = "smoke-check", decision_client=None, decision_calls: int = 0) -> int:
    """Run a program end to end on sample official requests ({"task_type", "request"}) over a store of the given history
    lines ({"app", "text"}) with a stub model: no model call, no scoring. Raises on a runtime protocol violation. A caller
    that already replayed the full history for user "smoke-check" may pass that store as ``document``."""
    from ohpu.experiments.lamp import SeededLuna
    from ohpu.per_user_adaptation.executor import Runner

    validate_program(source, editable_hooks=editable_hooks)
    events_ = [{**line, "id": line.get("id", f"h{index}"), "ts": line.get("ts", index), "app": line.get("app", "")}
        for index, line in enumerate(history)]
    try:
        document = document or documents(user_id, events_, [len(events_)], source=source if editable_hooks else None)[len(events_)]
    except ValueError as error:
        raise ValueError(f"runtime violation in remember: {error}") from error
    stub = _StubClient()
    for index, request in enumerate(requests):
        harness = _harness(user_id, source, document, editable_hooks=editable_hooks, tool_model_calls=tool_model_calls,
            decision_client=decision_client, decision_calls=decision_calls)
        try:
            Runner(SeededLuna(stub, 0, model=stub.model), harness).run(episode_id=f"smoke-{index}", user_id=user_id,
                condition="smoke", messages=[{"role": "user", "content": request["request"]}], tools=None,
                task_kind="pm3_native::" + (HIDDEN_TASK if hide_task_type else request["task_type"]), max_tokens=64,
                query=request["request"])
        except ValueError as error:
            raise ValueError(f"runtime violation on a {request['task_type']} request: {type(error).__name__}: {error}") from error
    return len(requests)


def check_reference(source: str, histories: list[list[dict]], requests: list[dict], texts, identifiers, *, literal_chars: int = 40) -> str:
    """Host checks for a shared reference harness on the native tasks: a valid program whose string literals name no
    registered user and copy no TRAIN label text, and which runs every sample request over every given history without a
    protocol violation while seeing only the hidden task kind."""
    digest = validate_program(source)
    literals = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    if literals & {str(identifier) for identifier in identifiers}:
        raise ValueError("reference harness contains a registered user identifier")
    corpus = "\x00".join(str(text or "") for text in texts)
    if any(len(value) >= literal_chars and value in corpus for value in literals):
        raise ValueError("reference harness copies TRAIN item text")
    for history in histories:
        smoke_check(source, history, requests, hide_task_type=True)
    return digest
