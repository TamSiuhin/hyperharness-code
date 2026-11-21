"""Source-preserving Mem-PAL execution and official signed selection scoring."""
from __future__ import annotations

import hashlib
import json
import statistics

from ohpu.benchmarks import mempal as data
from ohpu.benchmarks.horizon_runtime import QwenClient, pack_messages
from ohpu.benchmarks.muse_runtime import CONTRACT_LIMITS, MuseState
from ohpu.benchmarks.perma_runtime import failure_boundary, validate_declared_limits
from ohpu.experiments import pm3_official as official


RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "mempal_selection", "strict": True,
    "schema": {"type": "object", "properties": {"analysis": {"type": "string", "maxLength": 240,
        "pattern": "^[A-Za-z0-9 .,;:!?()'/-]*$"},
        "selected_solutions": {"type": "array", "minItems": 2, "maxItems": 2,
            "items": {"type": "string", "enum": list(data.LABELS)}}},
        "required": ["analysis", "selected_solutions"], "additionalProperties": False}}}


class MemPalState(MuseState):
    def __init__(self, user_id, source):
        super().__init__(user_id, source, span_retrieval=False, evidence_chars=20000, allow_memory_writes=True)

    def advance(self, events):
        previous_size = len(self.events)
        super().advance(events)
        if len(events) != previous_size and hasattr(self, "read_store"):
            del self.read_store


def answer_user(user, *, client, tokenizer, source=None, baseline=None, input_tokens=32768,
        output_tokens=512, seed=data.SEED, sink=None, error_sink=None):
    if (source is None) == (baseline is None) or baseline not in (None, "no_history", "bm25_10"):
        raise ValueError("Supply one personal program or registered baseline")
    if baseline == "bm25_10":
        source = official.program("bm25", official.RECALL_SEARCH, top_k=10, max_chars=20000)
    state = MemPalState(user["user_id"], source) if source is not None else None
    rows = []
    response = None
    def record_error(failure):
        if error_sink is None:
            return
        details = {**failure, "task_type": "solution_selection", "limits": {**CONTRACT_LIMITS,
            "input_tokens": input_tokens, "output_tokens": output_tokens}}
        if failure["stage"] == "executor" and response is not None:
            details["response"] = response
            details["request_seed"] = request_seed
            details["messages_sha256"] = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
            details["code"] = "executor_incomplete" if response.get("finish_reason") != "stop" else "executor_token_budget"
        error_sink(details)
    for case in user["cases"]:
        response = None
        query = data.request(case)
        retrieval_query = data.request(case, include_logs=False)
        cutoff = case["history_cutoff"]
        if not user["construction_cutoff"] <= cutoff <= len(user["events"]):
            raise ValueError("Invalid causal history cutoff")
        if state:
            with failure_boundary(user["user_id"], case, "remember", record_error):
                state.advance(user["events"][:cutoff])
            with failure_boundary(user["user_id"], case, "recall", record_error):
                evidence, guidance = state.context(None, retrieval_query, apply_guidance=baseline is None)
        else:
            evidence, guidance = {"output": "", "source_refs": []}, ""
        with failure_boundary(user["user_id"], case, "packing", record_error):
            messages, packing = pack_messages(None, evidence["output"], guidance, tokenizer,
                input_tokens=input_tokens, instruction=query,
                system_instruction="Use observed user preferences and current context to select solutions. ")
        request_seed = int.from_bytes(hashlib.sha256(f"{seed}:{user['user_id']}:{case['case_id']}".encode()).digest()[:4], "big") % (2 ** 31)
        with failure_boundary(user["user_id"], case, "executor", record_error):
            response = client.chat(messages, max_tokens=output_tokens, seed=request_seed, response_format=RESPONSE_FORMAT)
            usage = response.get("usage") or {}
            if response.get("finish_reason") != "stop":
                raise ValueError("Incomplete executor response; panel invalid")
            if usage.get("prompt_tokens", 0) > input_tokens or usage.get("completion_tokens", 0) > output_tokens:
                raise ValueError("Executor exceeded registered token budget")
        content = response.get("content") or ""
        if state:
            with failure_boundary(user["user_id"], case, "guard", record_error):
                if state.program.call("guard", {"stage": "validation", "task_kind": None,
                        "query": retrieval_query, "output": content}).patch:
                    raise ValueError("Mem-PAL guard cannot rewrite the answer or score")
        selected, error, value = None, None, -100.0
        try:
            selected = data.parse_output(content)
            value = data.selection_score(selected, case)
        except (ValueError, TypeError) as caught:
            error = str(caught)
        row = {"user": user["user_id"], "query_id": case["case_id"], "task_type": "solution_selection",
            "request": query, "agent_response": content, "prediction": selected,
            "ground_truth": {"positive": case["positive"], "negative": case["negative"]},
            "accuracy": value, "selection_score": value, "format_error": error, "usage": usage,
            "history_count": cutoff if state else 0, "history_cutoff": cutoff, "executor_calls": 1,
            "evidence": evidence, "guidance": guidance, "packing": packing}
        rows.append(row)
        if sink:
            sink(row)
    return rows


def summarize_user(user, rows):
    if not rows or [row["query_id"] for row in rows] != [case["case_id"] for case in user["cases"]]:
        raise ValueError("Incomplete or reordered Mem-PAL evaluation")
    summary = {"evaluation_complete": True, "runtime_violations": [], "queries": len(rows),
        "accuracy_pct_micro": statistics.fmean(row["selection_score"] for row in rows),
        "objective": data.OBJECTIVE, "invalid_outputs": sum(row["format_error"] is not None for row in rows)}
    mask = user.get("evaluation_mask")
    if mask is not None:
        original = mask["original_query_ids"]
        excluded = mask["excluded_query_ids"]
        if (mask.get("protocol") != "mempal-fixed-content-filter-mask-v1"
                or not original or len(original) != len(set(original))
                or len(excluded) != len(set(excluded)) or not set(excluded) <= set(original)
                or [identifier for identifier in original if identifier not in set(excluded)]
                != [case["case_id"] for case in user["cases"]]):
            raise ValueError("Invalid frozen Mem-PAL evaluation mask")
        summary.update(
            original_queries=len(original),
            scored_queries=len(rows),
            filtered_queries=len(excluded),
            coverage=len(rows) / len(original),
            original_evaluation_complete=not excluded,
        )
    return summary


def validate_harness(source, events):
    validate_declared_limits(source, limits=CONTRACT_LIMITS)
    state = MemPalState("validation", source)
    state.advance(events)
    request = 'User query:\nRecommend a practical plan based on my preferences.\nOptions:\nS1: Plan one\nS2: Plan two'
    state.context(None, request, apply_guidance=True)
    if state.program.call("guard", {"stage": "validation", "task_kind": None, "query": request,
            "output": json.dumps({"analysis": "Example", "selected_solutions": ["S1", "S2"]})}).patch:
        raise ValueError("Mem-PAL guard must return an empty patch")
