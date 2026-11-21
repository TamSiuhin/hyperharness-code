"""Bridge OHPU harnesses into PersonaMem-v3's official evaluator.

This module intentionally does not reimplement PM3 prompts, task dispatch, or
metrics.  It loads those components from a pinned checkout of the official PM3
repository and supplies only a ``QueryLLM``-compatible adapter whose generation
call executes the current OHPU :class:`~ohpu.per_user_adaptation.executor.Runner`.

The information boundary is one-way::

    official task builder -> strip raw-history section
    -> current request + bounded online memory -> current harness
    -> answer -> official scorer

Official labels, ground-truth preferences, rubrics, and judge output never flow
back through this bridge as Phase-A reward.  The v1 driver invokes it through
Phase A's observer callbacks, whose return values are ignored by the search.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

from ohpu.benchmarks.native_checkout import module_within, verify_pinned_checkout
from ohpu.experiments import pm3_proxy
from ohpu.llm import LLM, PROVIDER_DEFAULT
from ohpu.per_user_adaptation.executor import ExecutableHarness, Runner


# This is the official checkout audited for the current PM3 data release.  The
# revision is recorded in every run and checked before any label-bearing file is
# opened, so an upstream evaluator change cannot silently alter the experiment.
OFFICIAL_EVALUATOR_REVISION = "fcbc03ec3514cca7db32e81c3a844d3875659404"
EVAL_MODE = "llm_memory"
HISTORY_CUTOFF = "source_timestamp < T_test"
CALENDAR_HISTORY_CUTOFF = "calendar modification ts < T_test"
# The pinned upstream revision folds calendar state with ``ts <= T_test`` even
# though the paper and its other backend streams require events strictly before
# the query time.  Native OHPU runs apply this one-second integer-timestamp
# boundary correction and record it in every audit row.
CALENDAR_BOUNDARY_FIX = "upstream get_calendar_state(T_test - 1)"
# The pinned evaluator's long-context agentic prompt builders have two
# profile.json side channels despite the backend firewall describing that file
# as evaluator-private: focused voice blocks call ``get_full_profile`` and the
# proactive runner injects Big-Five/hidden-persona summaries.  The paper says
# the system receives the time-masked event history and must infer the user from
# it.  OHPU therefore keeps the official task/scorer code but profile-blinds
# those two prompt-construction paths. Judges continue to use the private
# profile after generation.
PROFILE_PROMPT_FIREWALL = (
    "profile.json denied during agent prompt construction; evaluator-private "
    "profile remains available only to official scorers and judges"
)
PROTOCOL = "pm3-paper-strict-official-native-prequential-v1"
EVALUATION_MODE_PREQUENTIAL = "prequential"
EVALUATION_MODE_CAUSAL_POST_ADAPTATION = "causal-post-adaptation"
EVALUATION_MODE_RETROSPECTIVE_FINAL_SELECTED = (
    "retrospective-final-selected"
)
CAUSAL_POST_ADAPTATION_PROTOCOL = (
    "pm3-official-native-causal-post-adaptation-final-selected-v1"
)
RETROSPECTIVE_FINAL_SELECTED_PROTOCOL = (
    "pm3-official-native-retrospective-final-selected-noncausal-v1"
)
LIFECYCLE_TASK_TYPE = "short_vs_long_term_lifecycle"
_FORBIDDEN_AGENT_PROMPT_MARKERS = frozenset({
    "groundtruth_preference",
    "example_response",
    "inferior_response",
    "held_out_idx",
    "hard_negative_idxs",
    "origin_by_idx",
    "expected_behavior",
    "trigger_evidence",
    "jitai_card",
    "gold_matches",
    "fabricated_value",
    "expected_warning_frame",
    "rubric_tags",
    "hidden_persona_labels",
    "update_history",
    "confidence_score_init",
    "confidence_cross_referenced",
    "stereotype_mark",
})
# Current 100-user public cohort peaks at 703,137 serialized history tokens at
# the latest query (audited 2026-08-27). Leave headroom while still failing if a
# future release would require silent reservoir truncation.
# The PM3 evaluator can construct very large long-context prompts.  Those
# prompts are intentionally not an input surface for an online harness: only
# the current task request (with the evaluator's history section removed) is
# forwarded to Runner.  Keep this bound explicit so a future prompt builder
# cannot silently turn the bridge back into a full-history harness.
MAX_HARNESS_REQUEST_CHARS = 32_000
MAX_ONLINE_MEMORY_CHARS = 64_000
DEFAULT_CONTEXT_BUDGET = 800_000

_ENV_LOCK = threading.RLock()
_PM3_ENV_KEYS = (
    "PM3_T_TEST",
    "PM3_USER_ID",
    "PM3_BACKEND_DIR",
    "PM3_RUN_DIR",
    "PM3_QUERY_ID",
    "PM3_TASK_TYPE",
    "PM3_OVERLAY_PATH",
    "PM3_HISTORY_APPS",
    "EVAL_CHRONO_HISTORY",
)


def _finite_metric(value: object) -> float | None:
    """Return an official numeric metric without bool/string coercion."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _lifecycle_identity(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """Recover the E5 canonical/phase attached during dispatch.

    The query-id fallback keeps already-produced CSV-shaped records auditable.
    PM3 constructs those ids as ``e5_<canonical[:40]>_<phase>`` and
    ``e5_pair_<canonical[:40]>``.
    """

    metadata = row.get("_pm3_lifecycle")
    if isinstance(metadata, Mapping):
        key = str(
            metadata.get("pair_key") or metadata.get("canonical_id") or ""
        ).strip()
        phase = str(metadata.get("phase") or "").strip().lower()
        if key and phase in {"pre", "post", "paired"}:
            return key, phase

    query_id = str(row.get("query_id") or "")
    parts = query_id.split(":", 2)
    instance_id = parts[2] if len(parts) == 3 else query_id
    if instance_id.startswith("e5_pair_"):
        key = instance_id[len("e5_pair_"):]
        return (key, "paired") if key else None
    match = re.fullmatch(r"e5_(.+)_(pre|post)", instance_id)
    if match:
        return match.group(1), match.group(2)
    return None


def _synthesize_lifecycle_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recreate the official E5 second pass after single-row dispatch.

    ``run_e5_horizon_lifecycle`` normally receives the whole E5 batch, pairs
    pre/post rows by canonical id, and appends one signed ``lifecycle_score``
    row per pair. ``dispatch_single`` suppresses that second pass because it
    sees only one instance. Apply the same arithmetic within one evaluation
    split, while rejecting missing or ambiguous sides instead of silently
    dropping E5 from the headline metric.
    """

    lifecycle_rows = [
        dict(row)
        for row in rows
        if str(row.get("_normalized_task_type") or row.get("task_type") or "")
        == LIFECYCLE_TASK_TYPE
    ]
    sides: dict[str, dict[str, list[dict[str, Any]]]] = {}
    existing_scores: dict[str, list[dict[str, Any]]] = {}
    opaque_existing_scores: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []

    for row in lifecycle_rows:
        metrics = row.get("_metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        identity = _lifecycle_identity(row)
        lifecycle_score = _finite_metric(metrics.get("lifecycle_score"))
        if lifecycle_score is not None:
            if identity is None:
                opaque_existing_scores.append(row)
            else:
                existing_scores.setdefault(identity[0], []).append(row)
            continue
        if identity is None or identity[1] not in {"pre", "post"}:
            unpaired.append({
                "query_id": str(row.get("query_id") or ""),
                "phase": None,
                "reason": "missing_lifecycle_identity",
            })
            continue
        key, phase = identity
        sides.setdefault(key, {"pre": [], "post": []})[phase].append(row)

    synthetic: list[dict[str, Any]] = []
    complete_pair_keys: set[str] = {
        key for key, value in existing_scores.items() if len(value) == 1
    }
    for key, phase_rows in sorted(sides.items()):
        pre_rows = phase_rows["pre"]
        post_rows = phase_rows["post"]
        if len(pre_rows) != 1 or len(post_rows) != 1:
            for phase, candidates in (("pre", pre_rows), ("post", post_rows)):
                reason = (
                    "missing_other_side" if len(candidates) == 1
                    else "duplicate_side"
                )
                for row in candidates:
                    unpaired.append({
                        "query_id": str(row.get("query_id") or ""),
                        "phase": phase,
                        "reason": reason,
                    })
            continue

        pre, post = pre_rows[0], post_rows[0]
        pre_metrics = pre.get("_metrics") or {}
        post_metrics = post.get("_metrics") or {}
        pre_match = _finite_metric(pre_metrics.get("match_rate_at_3"))
        post_match = _finite_metric(post_metrics.get("match_rate_at_3"))
        if (
            pre.get("status") != "ok"
            or post.get("status") != "ok"
            or pre_match is None
            or post_match is None
        ):
            for phase, row in (
                ("pre", pre), ("post", post)
            ):
                unpaired.append({
                    "query_id": str(row.get("query_id") or ""),
                    "phase": phase,
                    "reason": (
                        "non_ok_side"
                        if row.get("status") != "ok"
                        else "missing_match_rate_at_3"
                    ),
                })
            continue

        complete_pair_keys.add(key)
        if key in existing_scores:
            # A full-batch official result may already contain the pair row.
            # Keep that row as the single source of lifecycle_score.
            continue
        pair_metrics = {
            "pre.match_rate_at_3": pre_match,
            "post.match_rate_at_3": post_match,
            "lifecycle_score": pre_match - post_match,
            "post.hard_violation_at_1": (
                _finite_metric(post_metrics.get("matching_in_top1")) or 0.0
            ),
        }
        pair_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        synthetic.append({
            "query_id": f"e5_pair_{pair_hash}",
            "seq": None,
            "user_id": post.get("user_id") or pre.get("user_id") or "",
            "task_type": LIFECYCLE_TASK_TYPE,
            "ts": post.get("ts"),
            "metrics_json": json.dumps(pair_metrics, ensure_ascii=False),
            "status": "ok",
            "duration_ms": 0,
            "error": "",
            "agent_response": "",
            "_normalized_task_type": LIFECYCLE_TASK_TYPE,
            "_metrics": pair_metrics,
            "_pm3_lifecycle": {
                "pair_key": key,
                "phase": "paired",
                "synthesized_after_single_dispatch": True,
            },
        })

    duplicate_existing = {
        key: len(value) for key, value in existing_scores.items()
        if len(value) != 1
    }
    for count in duplicate_existing.values():
        unpaired.append({
            "query_id": None,
            "phase": "paired",
            "reason": "duplicate_lifecycle_score",
            "count": count,
        })
    if opaque_existing_scores and sides:
        # Without a canonical key, synthesis could double-count an existing
        # score. Retain the official row and fail the completeness audit.
        unpaired.append({
            "query_id": None,
            "phase": "paired",
            "reason": "unkeyed_existing_lifecycle_score_with_side_rows",
            "count": len(opaque_existing_scores),
        })
        synthetic = []
        complete_pair_keys = {
            key for key, value in existing_scores.items() if len(value) == 1
        }

    telemetry = {
        "lifecycle_side_rows": sum(
            len(phase_rows[phase])
            for phase_rows in sides.values()
            for phase in ("pre", "post")
        ) + sum(
            1 for row in lifecycle_rows
            if _finite_metric((row.get("_metrics") or {}).get("lifecycle_score"))
            is None and _lifecycle_identity(row) is None
        ),
        "complete_pairs": len(complete_pair_keys) + len(opaque_existing_scores),
        "synthesized_pair_rows": len(synthetic),
        "existing_lifecycle_score_rows": (
            sum(len(value) for value in existing_scores.values())
            + len(opaque_existing_scores)
        ),
        "unpaired_side_count": len(unpaired),
        "unpaired_sides": unpaired,
        "metric_complete": not unpaired and not duplicate_existing,
    }
    return synthetic, telemetry


def _int_timestamp(value: object, *, label: str) -> int:
    try:
        timestamp = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has a non-integer PM3 timestamp") from exc
    if timestamp <= 0:
        raise ValueError(f"{label} has no positive PM3 timestamp")
    return timestamp


def evaluation_protocol(evaluation_mode: str) -> str:
    """Return the reporting protocol for one explicit PM3 evaluation mode."""

    if evaluation_mode == EVALUATION_MODE_PREQUENTIAL:
        return PROTOCOL
    if evaluation_mode == EVALUATION_MODE_CAUSAL_POST_ADAPTATION:
        return CAUSAL_POST_ADAPTATION_PROTOCOL
    if evaluation_mode == EVALUATION_MODE_RETROSPECTIVE_FINAL_SELECTED:
        return RETROSPECTIVE_FINAL_SELECTED_PROTOCOL
    raise ValueError(f"unsupported PM3 native evaluation mode: {evaluation_mode}")


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PM3 evaluator script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _paper_strict_backend_type(base: type) -> type:
    """Return an official BackendQuery subclass with the paper's strict T cut."""

    class PaperStrictBackendQuery(base):
        def get_calendar_state(
            self,
            user_id: str,
            as_of_timestamp: int,
        ) -> dict[str, dict]:
            # PM3 timestamps are integer seconds. Upstream folds with ``<=``;
            # evaluating at T-1 is exactly the paper's ``modification_ts < T``.
            return super().get_calendar_state(
                user_id,
                int(as_of_timestamp) - 1,
            )

    PaperStrictBackendQuery.__name__ = "PaperStrictBackendQuery"
    return PaperStrictBackendQuery


class _ProfileBlindBackend:
    """Prompt-builder view that delegates events but denies profile state."""

    __slots__ = ("_backend",)

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def _load_profile(self, user_id: str) -> dict[str, Any]:
        del user_id
        return {}

    def get_full_profile(self, user_id: str) -> dict[str, Any]:
        del user_id
        return {}

    def get_profile_summary(self, user_id: str) -> dict[str, Any]:
        del user_id
        return {}

    def get_friend(self, user_id: str, friend_id: str) -> None:
        del user_id, friend_id
        return None


def _install_paper_prompt_firewall(
    ground_truth_builders: ModuleType,
    proactive_actions: ModuleType,
) -> None:
    """Remove evaluator-private profile side channels from agent prompts."""

    current_builder = ground_truth_builders.build_for_task
    if not getattr(current_builder, "_ohpu_profile_blind", False):
        original_builder = current_builder

        def profile_blind_builder(
            task_id: str,
            backend: Any,
            user_id: str,
            t_test: int,
            instance: dict[str, Any],
        ) -> str:
            return original_builder(
                task_id,
                _ProfileBlindBackend(backend),
                user_id,
                t_test,
                instance,
            )

        profile_blind_builder._ohpu_profile_blind = True  # type: ignore[attr-defined]
        ground_truth_builders.build_for_task = profile_blind_builder

    current_summary = proactive_actions._build_user_state_summary
    if not getattr(current_summary, "_ohpu_profile_blind", False):
        def profile_blind_summary(backend: Any, user_id: str) -> str:
            del backend, user_id
            return "(profile withheld; infer user state from causal history)"

        profile_blind_summary._ohpu_profile_blind = True  # type: ignore[attr-defined]
        proactive_actions._build_user_state_summary = profile_blind_summary


class AgentPromptLeakError(RuntimeError):
    """Raised when evaluator-private field names reach the generation prompt."""


def _agent_prompt_audit(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, str):
        rendered = prompt
    else:
        rendered = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    folded = rendered.casefold()
    exposed = sorted(
        marker
        for marker in _FORBIDDEN_AGENT_PROMPT_MARKERS
        if marker.casefold() in folded
    )
    if exposed:
        raise AgentPromptLeakError(
            "PM3 agent prompt exposed evaluator-private fields: "
            + ", ".join(exposed)
        )
    return {
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_characters": len(rendered),
        "forbidden_markers_exposed": [],
    }


class HarnessInputBoundaryError(ValueError):
    """Raised when an evaluator prompt cannot be reduced to an online input."""


# Official PM3 task builders interpolate the time-masked context under one of
# these headings.  The bridge removes that section before a prompt becomes a
# harness message.  The actual online memory is attached to the bundle and is
# therefore rendered by Runner's normal memory block, not copied from this
# evaluator prompt.
_HISTORY_SECTION_RE = re.compile(
    r"(?is)(?:^|\n)[ \t]*##[ \t]+(?:cross-app[ \t]+user|user|full user)[ \t]+history"
    r"[ \t]*\(time-masked\)[^\n]*\n.*?(?=\n[ \t]*##[ \t]+|\Z)"
)
_HISTORY_SENTINEL_RE = re.compile(
    r"(?is)__OHPU_HISTORY_BEGIN__.*?__OHPU_HISTORY_END__"
)


def _online_task_request(prompt: Any) -> tuple[str, dict[str, int | bool]]:
    """Return a bounded task request with evaluator history removed.

    PM3's ``llm_longctx`` path historically passed its complete generated
    prompt into ``query``.  That made the stage hooks observe the benchmark's entire
    history, even when the harness had an online memory.  This function keeps
    the task framing/candidates while deleting both the conventional history
    section and the serializer sentinels.  It never truncates silently: an
    unexpectedly large non-history task prompt fails closed.
    """

    if isinstance(prompt, list):
        # QueryLLM callers sometimes provide a conversation.  System messages
        # can contain evaluator-private scaffolding, so only user turns are an
        # admissible current-request surface.
        pieces = [
            str(message.get("content") or "")
            for message in prompt
            if isinstance(message, Mapping) and message.get("role") == "user"
        ]
        source = "\n\n".join(pieces)
    else:
        source = str(prompt or "")
    without_sentinels, delimited_sections = _HISTORY_SENTINEL_RE.subn("", source)
    sanitized, removed = _HISTORY_SECTION_RE.subn("", without_sentinels)
    request = sanitized.strip()
    if len(request) > MAX_HARNESS_REQUEST_CHARS:
        raise HarnessInputBoundaryError(
            "PM3 current task request exceeds online harness bound "
            f"{MAX_HARNESS_REQUEST_CHARS} characters after history removal "
            f"(got {len(request)})"
        )
    return request, {
        "source_prompt_characters": len(source),
        "history_sections_removed": removed,
        "delimited_memory_sections_removed": delimited_sections,
        "history_characters_removed": max(0, len(source) - len(request)),
        "current_request_characters": len(request),
        "raw_history_forwarded": False,
    }


class OnlineSnapshotCache:
    """A PM3 ``SnapshotCache``-compatible view over one online memory state.

    The official ``SnapshotCache(mode='llm_longctx')`` asks the backend to
    serialize every event before every generation call.  This adapter has no
    backend serialization path: ``get_or_build`` can only return the bounded
    memory snapshot supplied by the current harness bundle.  Consequently,
    moving to another query or T_test cannot accidentally reintroduce raw
    history into the harness prompt.
    """

    mode = "llm_memory"

    def __init__(self, memory: str = "") -> None:
        self._memory = ""
        self._store: dict[tuple[str, int, str | None], tuple[str, dict[str, Any]]] = {}
        self._pending_query: str | None = None
        self.attach_memory(memory)

    def attach_memory(self, memory: str | None) -> None:
        text = str(memory or "").strip()
        if len(text) > MAX_ONLINE_MEMORY_CHARS:
            raise HarnessInputBoundaryError(
                "online PM3 memory exceeds bound "
                f"{MAX_ONLINE_MEMORY_CHARS} characters (got {len(text)})"
            )
        self._memory = text
        self._store.clear()

    def attach_memory_checkpoints(self, checkpoints: Mapping[Any, str] | None) -> None:
        """Compatibility hook; online runs use the current state only."""

        if not checkpoints:
            return
        latest = max(checkpoints, key=lambda value: int(value))
        self.attach_memory(checkpoints[latest])

    def get_or_build(
        self,
        _bq: Any,
        user_id: str,
        t_test: int,
        model: str | None,
        _budget: int | None,
        *,
        query: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del query
        key = (str(user_id), int(t_test), model)
        if key not in self._store:
            # Rough token accounting is sufficient for the audit and avoids
            # importing any evaluator tokenizer into this boundary module.
            stats = {
                "total_tokens": (len(self._memory) + 3) // 4,
                "memory_mode": True,
                "online_memory": True,
                "raw_history_chars": 0,
                "raw_history_serialized": False,
                "memory_characters": len(self._memory),
                "truncated": False,
                "per_app": {},
            }
            # The official builder embeds this string inside a Markdown
            # history section. Memory itself contains Markdown headings, so
            # heading-based stripping alone stops early and leaks the rest
            # into the current request, bypassing recall/no-memory decisions.
            # Delimit the complete supplied snapshot before interpolation;
            # only the harness memory block may expose it to generation.
            marked = "__OHPU_HISTORY_BEGIN__\n" + self._memory + "\n__OHPU_HISTORY_END__"
            self._store[key] = (marked, stats)
        return self._store[key]


class GenerationReplayError(RuntimeError):
    """A fixed-answer rejudge diverged from its original generation contract."""


class GenerationTape:
    """Evaluator-private, in-memory answers for one query's fixed-answer rejudge.

    Identity includes user, query, policy, memory and scope. Every public
    request must match in order; replay never falls back to paid generation.
    """
    def __init__(self, *, measurement_store=None, measurement_slot=None):
        self.identity = None
        self.calls = []
        self.position = 0
        self.sealed = False
        self.failed = False
        self.measurement_store = measurement_store
        self.measurement_slot = measurement_slot

    def capture(self, request_digest, operation):
        identity = {"query_identity": self.identity, "request_digest": request_digest}
        resumed = False
        if self.measurement_store is not None:
            if self.measurement_slot is None:
                raise GenerationReplayError("persistent generation needs an explicit slot")
            call, resumed = self.measurement_store.measure(["native_generation", self.measurement_slot, len(self.calls)], identity, operation)
        else:
            call = operation()
        self.calls.append(copy.deepcopy(call))
        if resumed:
            call = copy.deepcopy(call)
            trace = call["trace"]
            trace["replayed_model_calls"] = trace["model_calls"]
            trace["model_calls"] = 0
            trace["usage"] = {"prompt": 0, "completion": 0, "reasoning": 0}
            trace["generation_replayed"] = True
            trace["measurement_resumed"] = True
        return call

    def begin(self, identity, *, replay):
        self.position = 0
        self.failed = False
        if replay:
            if not self.sealed or self.identity != identity:
                raise GenerationReplayError("generation tape identity mismatch or incomplete capture")
        elif self.identity is not None:
            raise GenerationReplayError("generation tape already captured")
        else:
            self.identity = identity

    def take(self, request_digest):
        if self.position >= len(self.calls) or self.calls[self.position]["request_digest"] != request_digest:
            self.failed = True
            raise GenerationReplayError("fixed-answer request sequence changed")
        call = self.calls[self.position]
        self.position += 1
        return copy.deepcopy(call)

    def finish(self, *, replay):
        if self.failed or (replay and self.position != len(self.calls)):
            raise GenerationReplayError("fixed-answer generation tape was not consumed exactly")
        if not replay:
            self.sealed = True

    def export(self):
        if not self.sealed or self.failed or not isinstance(self.identity, str):
            raise GenerationReplayError("only a completed generation tape can be persisted")
        payload = {"protocol": "pm3-generation-tape-v1", "identity": self.identity, "calls": copy.deepcopy(self.calls)}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        return {"payload": payload, "sha256": hashlib.sha256(encoded).hexdigest()}

    @classmethod
    def restore(cls, document):
        if not isinstance(document, dict) or set(document) != {"payload", "sha256"}:
            raise GenerationReplayError("invalid persisted generation tape")
        payload = document["payload"]
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != document["sha256"]:
            raise GenerationReplayError("persisted generation tape checksum changed")
        if (not isinstance(payload, dict) or set(payload) != {"protocol", "identity", "calls"}
                or payload["protocol"] != "pm3-generation-tape-v1" or not isinstance(payload["identity"], str)
                or not isinstance(payload["calls"], list)):
            raise GenerationReplayError("invalid generation tape payload")
        for call in payload["calls"]:
            if (not isinstance(call, dict) or set(call) != {"request_digest", "answer", "trace"}
                    or not isinstance(call["request_digest"], str) or not isinstance(call["answer"], str)
                    or not isinstance(call["trace"], dict)):
                raise GenerationReplayError("invalid captured answer entry")
        tape = cls()
        tape.identity, tape.calls, tape.sealed = payload["identity"], copy.deepcopy(payload["calls"]), True
        return tape


class HarnessQueryLLM:
    """Official ``QueryLLM`` surface backed by one online OHPU harness."""

    def __init__(
        self,
        llm: LLM,
        bundle: ExecutableHarness,
        *,
        query_id: str,
        task_type: str,
        condition: str,
        current_request: str | None = None,
        max_tokens: int = 4096,
        generation_tape: GenerationTape | None = None,
        replay_generation: bool = False,
    ) -> None:
        self.llm = llm
        self.bundle = bundle
        self.query_id = query_id
        self.task_type = task_type
        self.condition = condition
        self.current_request = current_request
        self.max_tokens = max_tokens
        self.traces: list[dict[str, Any]] = []
        self.prompt_audits: list[dict[str, Any]] = []
        self.generation_tape = generation_tape
        self.replay_generation = replay_generation
        if replay_generation and generation_tape is None:
            raise GenerationReplayError("fixed-answer replay requires a tape")
        if generation_tape is not None:
            from ohpu.per_user_adaptation.retrieval_policy import IMPLEMENTATION_SHA256
            identity = hashlib.sha256(json.dumps({
                "user": bundle.user_id, "query": query_id, "task": task_type,
                "program": bundle.program.fingerprint.sha256 if bundle.program else None,
                "config": bundle.config.as_dict(), "memory": bundle.memory_md,
                "scope": bundle.recall_task_scope, "max_tokens": max_tokens,
                "recall_profile": bundle.recall_profile,
                "pipeline_config": bundle.pipeline_config,
                **({"request_contract": bundle.request_contract} if bundle.request_contract is not None else {}),
                "host_retrieval_sha256": IMPLEMENTATION_SHA256 if bundle.recall_profile == "shared-retrieval" else None,
                "adaptation_source_count": bundle.adaptation_source_count,
                "adaptation_task_scope": bundle.adaptation_task_scope,
                "model": getattr(llm, "model", None),
                "reasoning_effort": getattr(llm, "reasoning_effort", None),
            }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            generation_tape.begin(identity, replay=replay_generation)

    def query_llm(
        self,
        prompt: Any,
        use_history: bool = False,
        image: Any = None,
        image_path: Any = None,
        verbose: bool = False,
        thread_id: Any = None,
        temperature: float | None = None,
        **_kwargs: Any,
    ) -> str:
        del image, image_path, verbose, thread_id, temperature
        del use_history
        if self.current_request is None:
            public_query, input_audit = _online_task_request(prompt)
        else:
            public_query = str(self.current_request).strip()
            if len(public_query) > MAX_HARNESS_REQUEST_CHARS:
                raise HarnessInputBoundaryError(
                    "explicit PM3 current request exceeds online harness bound"
                )
            input_audit = {
                "source_prompt_characters": len(str(prompt or "")),
                "history_sections_removed": 0,
                "history_characters_removed": 0,
                "current_request_characters": len(public_query),
                "raw_history_forwarded": False,
            }
        # The adapter deliberately constructs a one-turn conversation from the
        # reduced request.  The compact online memory is rendered by Runner
        # from bundle.memory_md; no evaluator history message is retained.
        messages = [{"role": "user", "content": public_query}]
        prompt_audit = _agent_prompt_audit(messages)
        prompt_audit.update(input_audit)
        retrieval_query, response_format = public_query, None
        pipeline_request = (self.bundle.pipeline_config or {}).get("request_profile")
        if pipeline_request or self.bundle.request_contract is not None:
            from . import pm3_request
            from ohpu.per_user_adaptation import adaptive_pipeline
            if pipeline_request:
                adaptive_pipeline.validate_execution_policy(self.bundle.pipeline_config)
            if self.bundle.request_contract is not None:
                if self.bundle.request_contract != {
                    "profile": pm3_request.PROFILE, "sha256": pm3_request.IMPLEMENTATION_SHA256,
                }:
                    raise ValueError("native public request contract changed")
            request = pm3_request.prepare(public_query)
            retrieval_query, response_format = request["query"], request["response_format"]
            prompt_audit.update(request["audit"])
        self.prompt_audits.append(prompt_audit)
        request_digest = hashlib.sha256(public_query.encode()).hexdigest()
        if self.replay_generation:
            saved = self.generation_tape.take(request_digest)
            trace = saved["trace"]
            trace["replayed_model_calls"] = trace["model_calls"]
            trace["model_calls"] = 0
            trace["usage"] = {"prompt": 0, "completion": 0, "reasoning": 0}
            trace["generation_replayed"] = True
            self.traces.append(trace)
            return saved["answer"]
        call_index = len(self.traces)
        def generate():
            trace = Runner(self.llm, self.bundle).run(
                episode_id=f"{self.query_id}::native-call-{call_index}",
                user_id=self.bundle.user_id, condition=self.condition,
                messages=messages, tools=None, task_kind=f"pm3_native::{self.task_type}",
                max_tokens=self.max_tokens, query=retrieval_query, response_format=response_format,
            )
            return {"request_digest": request_digest, "trace": trace.as_dict(), "answer": trace.final}
        call = self.generation_tape.capture(request_digest, generate) if self.generation_tape is not None else generate()
        self.traces.append(copy.deepcopy(call["trace"]))
        return call["answer"]

    def query_many(
        self,
        prompts: Iterable[Any],
        temperature: float | None = None,
        allow_batch: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        del allow_batch
        return [
            self.query_llm(prompt, temperature=temperature, **kwargs)
            for prompt in prompts
        ]

    def reset_history(self, thread_id: Any = None) -> None:
        del thread_id

    def get_usage_totals(self) -> dict[str, int]:
        return {
            "calls": sum(not trace.get("generation_replayed", False) for trace in self.traces),
            "input_tokens": sum(
                int((trace.get("usage") or {}).get("prompt", 0) or 0)
                for trace in self.traces
            ),
            "output_tokens": sum(
                int((trace.get("usage") or {}).get("completion", 0) or 0)
                for trace in self.traces
            ),
            "cached_input_tokens": 0,
            "errors": 0,
        }


class JudgeUnavailable(RuntimeError):
    """The judge produced no parseable output after every attempt.

    Both official consumers turn an unparseable reply into a number without
    saying so: ``personalization_rubric.score`` scores 0.0 and
    ``llm_metrics.personalization_leak_check`` defaults ``restraint_score`` to
    a perfect 10. Opposite directions, same problem -- a judge outage becomes
    a measurement. Raising is not enough on its own, because both callers
    swallow exceptions, so ``PrivateJudgeQueryLLM.failures`` also counts them
    for a caller that has to decide whether a pass is scorable.
    """


def _official_json_parser():
    """The evaluator's own response parser, so retries agree with scoring.

    A local parser would retry replies the official ``json_repair`` fallback
    accepts, and accept replies it rejects.
    """

    return pm3_proxy.official_module("data_preparation.utils").extract_json_from_response


class PrivateJudgeQueryLLM:
    """Evaluator-private PM3 judge with no personalized harness attached.

    The request shape matches the official evaluator's ``QueryLLM`` OpenAI
    branch, which sends only model and messages: no completion budget and no
    reasoning effort. A caller-chosen budget is not a neutral knob here --
    reasoning tokens are drawn from it, so a 4096 cap returned empty content,
    and ``personalization_rubric.score`` scores an unparseable judge reply 0.0
    with no error, indistinguishable from a genuinely bad answer.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        model: str,
        reasoning_effort: str | None = "default",
        max_tokens: int | None = None,
        max_attempts: int = 3,
        measurement_store=None,
        measurement_slot=None,
        measurement_identity=None,
        score_store=None,
        seed: int = 7,
        response_cache=None,
        cache_scope=None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.reasoning_effort = (
            PROVIDER_DEFAULT
            if reasoning_effort in (None, "", "default")
            else reasoning_effort
        )
        self.max_tokens = max_tokens
        self.max_attempts = max(1, int(max_attempts))
        self.failures = 0
        self.unusable_replies = 0
        self.measurement_store = measurement_store
        self.measurement_slot = measurement_slot
        self.measurement_identity = measurement_identity
        self.score_store = score_store
        self.seed = seed
        self.response_cache = response_cache
        self.cache_scope = cache_scope
        self.pending_score_replies = []
        self.scoring_replies = []
        self.reply_cache_hits = 0
        self._reply_locks_guard = threading.Lock()
        self._reply_locks = {}
        self._pending_reply_values = {}
        self._measurement_ordinal = 0

    def scoring_identity(self):
        client = getattr(self.llm, "_client", None)
        return {"model": self.model, "endpoint": str(getattr(client, "base_url", "")),
                "api_version": (getattr(client, "default_query", {}) or {}).get("api-version"),
                "reasoning_effort": self.reasoning_effort, "max_tokens": self.max_tokens,
                "seed": self.seed, "max_attempts": self.max_attempts,
                "format_repair_policy": "json_object_retry-v1"}

    def fork(self, *, measurement_slot=None, measurement_identity=None, repeat_index=None, cache_scope=None) -> "PrivateJudgeQueryLLM":
        """Share the metered client, but isolate each scoring job's counters."""
        return PrivateJudgeQueryLLM(
            self.llm, model=self.model, reasoning_effort=self.reasoning_effort,
            max_tokens=self.max_tokens, max_attempts=self.max_attempts,
            measurement_store=self.measurement_store,
            measurement_slot=measurement_slot,
            measurement_identity={"experiment": self.measurement_identity, "measurement": measurement_identity},
            score_store=self.score_store,
            seed=self.seed if repeat_index is None else self.seed + 100 * repeat_index,
            response_cache=self.response_cache,
            cache_scope=self.cache_scope if cache_scope is None else cache_scope,
        )

    def commit_validated_replies(self):
        if self.response_cache is not None:
            for slot, identity, reply in self.pending_score_replies:
                self.response_cache.run(slot, identity, lambda: reply)
            self.pending_score_replies.clear()

    def query_llm(
        self,
        prompt: Any,
        use_history: bool = False,
        temperature: float | None = None,
        **_kwargs: Any,
    ) -> str:
        messages = (
            list(prompt)
            if use_history and isinstance(prompt, list)
            else [{"role": "user", "content": str(prompt or "")}]
        )
        parse = _official_json_parser()
        slot = [self.measurement_slot, self._measurement_ordinal]
        self._measurement_ordinal += 1
        identity = {"context": self.measurement_identity, "messages": messages,
                    "model": self.model, "max_tokens": self.max_tokens,
                    "reasoning_effort": self.reasoning_effort,
                    "temperature": 0.0 if temperature is None else temperature,
                    **({"seed": self.seed} if self.seed != 7 else {})}

        def measure():
            cache_identity = None
            if self.response_cache is not None:
                from ohpu.benchmarks.private_measurements import digest

                if not self.cache_scope:
                    raise ValueError("native judge cache requires an isolated scoring scope")
                cache_identity = {"scope": self.cache_scope, "judge": self.scoring_identity(),
                                  "messages": messages, "temperature": identity["temperature"]}
                cache_slot = ["validated-judge-reply", digest(cache_identity)]
                cached = self.response_cache.completed_value(cache_slot, cache_identity)
                if cached is not None:
                    self.reply_cache_hits += 1
                    self.scoring_replies.append(parse(cached))
                    return cached
            for attempt in range(self.max_attempts):
                attempt_messages = messages
                options = {"seed": self.seed} if self.seed != 7 else {}
                if attempt:
                    attempt_messages = [*messages, {
                        "role": "user",
                        "content": (
                            "FORMAT REPAIR: Evaluate exactly the same fixed response "
                            "using the original rubric and all its required fields. "
                            "Return one nonempty JSON object only, without markdown "
                            "or surrounding commentary. Do not rewrite or replace "
                            "the response being evaluated."
                        ),
                    }]
                    options = {"seed": self.seed + attempt, "response_format": {"type": "json_object"}}
                try:
                    response = self.llm.chat(
                        attempt_messages, model=self.model, max_tokens=self.max_tokens,
                        reasoning_effort=self.reasoning_effort,
                        temperature=identity["temperature"],
                        **options,
                    )
                except Exception:
                    self.failures += 1
                    raise
                content = str(response.get("content") or "")
                if parse(content):
                    self.scoring_replies.append(parse(content))
                    if cache_identity is not None:
                        self.pending_score_replies.append((cache_slot, cache_identity, content))
                    return content
                self.unusable_replies += 1
                if self.measurement_store is not None:
                    self.measurement_store.record_unusable_reply(slot, identity, response)
            self.failures += 1
            raise JudgeUnavailable(
                f"judge returned no parseable JSON in {self.max_attempts} attempts"
            )

        def measured_reply():
            if self.response_cache is None:
                return measure()
            request_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            with self._reply_locks_guard:
                lock = self._reply_locks.setdefault(request_key, threading.Lock())
            with lock:
                if request_key in self._pending_reply_values:
                    reply = self._pending_reply_values[request_key]
                    self.scoring_replies.append(parse(reply))
                    return reply
                reply = measure()
                self._pending_reply_values[request_key] = reply
                return reply

        if self.measurement_store is None:
            return measured_reply()
        if self.measurement_slot is None:
            raise ValueError("persistent judge needs an explicit measurement slot")
        return self.measurement_store.run(["judge", slot], identity, measured_reply)

    def query_many(
        self,
        prompts: Iterable[Any],
        temperature: float | None = None,
        allow_batch: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        del allow_batch
        return [
            self.query_llm(prompt, temperature=temperature, **kwargs)
            for prompt in prompts
        ]

    def __call__(self, prompt: Any, **kwargs: Any) -> str:
        return self.query_llm(prompt, **kwargs)


class _ConditionClusterCaches:
    """Isolate PM3's response-bearing repetition caches between eval arms."""

    _ATTR_BY_TASK = {
        "over_personalization_repetition_recsys": "_C1C_CLUSTER_CACHE",
        "over_personalization_repetition_chatbot": "_C1D_CLUSTER_CACHE",
    }

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.stores: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        self.origins: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    def activate(
        self,
        namespace: str,
        task_type: str,
        instance: Mapping[str, Any],
        *,
        query_id: str,
        t_test: int,
        harness_checkpoint: int,
    ) -> dict[str, Any]:
        attr = self._ATTR_BY_TASK.get(task_type)
        cluster_id = str(instance.get("cluster_id") or "")
        if attr is None or not cluster_id:
            return {"enabled": False}
        cache = self.stores.setdefault(namespace, {}).setdefault(attr, {})
        origins = self.origins.setdefault(namespace, {}).setdefault(attr, {})
        # The official task module reads this global directly. The whole native
        # dispatch runs under _ENV_LOCK, so swapping the arm-owned dictionary is
        # atomic with respect to every other PM3 query in this process.
        setattr(self.module, attr, cache)
        return {
            "enabled": True,
            "namespace": namespace,
            "cache_attr": attr,
            "cluster_id": cluster_id,
            "cache": cache,
            "origins": origins,
            "hit_before_dispatch": cluster_id in cache,
            "current_origin": {
                "query_id": query_id,
                "t_test": int(t_test),
                "harness_checkpoint": f"W{harness_checkpoint}",
            },
        }

    @staticmethod
    def finish(ticket: Mapping[str, Any]) -> dict[str, Any]:
        if not ticket.get("enabled"):
            return {"enabled": False}
        cluster_id = str(ticket["cluster_id"])
        cache = ticket["cache"]
        origins = ticket["origins"]
        if cluster_id in cache and cluster_id not in origins:
            origins[cluster_id] = dict(ticket["current_origin"])
        origin = origins.get(cluster_id)
        return {
            "enabled": True,
            "namespace": ticket["namespace"],
            "cluster_id": cluster_id,
            "hit": bool(ticket["hit_before_dispatch"]),
            "origin": dict(origin) if isinstance(origin, Mapping) else None,
        }


@contextlib.contextmanager
def _official_query_environment(
    *,
    row: Mapping[str, Any],
    user_id: str,
    backend_dir: Path,
    run_dir: Path,
):
    """Set PM3's process-global row context under a cross-thread lock."""

    values = {
        "PM3_T_TEST": str(row["ts"]),
        "PM3_USER_ID": user_id,
        "PM3_BACKEND_DIR": str(backend_dir),
        "PM3_RUN_DIR": str(run_dir),
        "PM3_QUERY_ID": str(row.get("query_id") or ""),
        "PM3_TASK_TYPE": str(row.get("task_type") or ""),
        "PM3_OVERLAY_PATH": str(run_dir / "writes.jsonl"),
        # The paper/default arm is full cross-platform history.
        "PM3_HISTORY_APPS": "",
        "EVAL_CHRONO_HISTORY": "1",
    }
    with _ENV_LOCK:
        previous = {key: os.environ.get(key) for key in _PM3_ENV_KEYS}
        os.environ.update(values)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _install_e5_response_type_guard(task_module):
    parser = task_module.extract_json_from_response
    if getattr(parser, "_ohpu_e5_type_guard", False):
        return

    def object_response(raw_response):
        parsed = parser(raw_response)
        return parsed if isinstance(parsed, dict) else {}

    object_response._ohpu_e5_type_guard = True
    task_module.extract_json_from_response = object_response


class OfficialPM3Runtime:
    """One user's official task dispatcher and strict time-masked backend."""

    def __init__(
        self,
        evaluator_root: str | Path,
        backend_dir: str | Path,
        user_id: str,
        *,
        model_name: str,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
        enable_llm_judge: bool = True,
        expected_revision: str = OFFICIAL_EVALUATOR_REVISION,
    ) -> None:
        self.evaluator_root = Path(evaluator_root).expanduser().resolve()
        self.backend_dir = Path(backend_dir).expanduser().resolve()
        self.user_id = str(user_id)
        self.model_name = model_name
        self.context_budget = int(context_budget)
        self.enable_llm_judge = bool(enable_llm_judge)
        required = (
            self.evaluator_root / "evaluation" / "run_eval.py",
            self.evaluator_root / "evaluation" / "run_eval_dispatch.py",
            self.evaluator_root / "scripts" / "aggregate_eval.py",
            self.backend_dir / self.user_id / "test.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "PM3 official evaluator/data files are missing: "
                + ", ".join(missing)
            )

        self.revision = verify_pinned_checkout(
            self.evaluator_root,
            expected_revision=expected_revision,
            label="PM3",
            include_untracked=True,
            fetch_hint=(
                "use the pinned git checkout produced by "
                "scripts/fetch_pm3_evaluator.sh"
            ),
        )

        existing = sys.modules.get("evaluation")
        if existing is not None and not module_within(
            existing, self.evaluator_root
        ):
            raise RuntimeError(
                "a different top-level 'evaluation' package is already loaded; "
                "start PM3 native evaluation in a fresh process"
            )
        root_text = str(self.evaluator_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        self.run_eval = importlib.import_module("evaluation.run_eval")
        self.dispatch = importlib.import_module("evaluation.run_eval_dispatch")
        self.backend = importlib.import_module("evaluation.backend_query")
        self.inference = importlib.import_module("evaluation.inference_utils")
        _install_e5_response_type_guard(
            importlib.import_module("evaluation.tasks.e5_horizon_lifecycle")
        )
        _install_e5_response_type_guard(
            importlib.import_module("evaluation.tasks.new_suggestions")
        )
        aggregate_name = (
            "_ohpu_pm3_aggregate_"
            + hashlib.sha256(root_text.encode()).hexdigest()[:12]
        )
        self.aggregate = sys.modules.get(aggregate_name) or _load_script(
            self.evaluator_root / "scripts" / "aggregate_eval.py",
            aggregate_name,
        )
        strict_backend = _paper_strict_backend_type(self.backend.BackendQuery)
        self.bq = strict_backend(self.backend_dir)
        # The official cache is intentionally not used here: its long-context
        # mode serializes all backend events.  The online adapter can only
        # expose the current bundle memory snapshot.
        self.snapshot_cache = OnlineSnapshotCache()
        self._cluster_caches = _ConditionClusterCaches(
            importlib.import_module("evaluation.tasks.over_personalization")
        )
        _install_paper_prompt_firewall(
            importlib.import_module("evaluation.ground_truth_builders"),
            importlib.import_module("evaluation.tasks.proactive_actions"),
        )

    def load_queries(self, *, limit: int = 0) -> list[dict[str, Any]]:
        path = self.backend_dir / self.user_id / "test.json"
        rows = [dict(row) for row in self.run_eval._load_queries(path)]
        if limit > 0:
            rows = rows[:limit]
        checked: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("user_id") or self.user_id) != self.user_id:
                raise PermissionError(
                    f"PM3 query {row.get('query_id')} belongs to another user"
                )
            timestamp = _int_timestamp(
                row.get("ts"), label=str(row.get("query_id") or "query")
            )
            try:
                instance = json.loads(str(row.get("instance_json") or "{}"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"PM3 query {row.get('query_id')} has invalid instance_json"
                ) from exc
            effective = instance.get("t_test")
            if effective is None:
                effective = instance.get("source_timestamp")
            if effective is not None and int(effective) != timestamp:
                raise ValueError(
                    f"PM3 query {row.get('query_id')} row ts={timestamp} but "
                    f"task T_test={effective}"
                )
            row["ts"] = str(timestamp)
            checked.append(row)
        task_registry = importlib.import_module("evaluation.task_registry")
        normalized_tasks = {
            task_registry.normalize_task_type(
                str(row.get("task_type") or "")
            )
            for row in checked
        }
        unsupported = sorted(
            normalized_tasks
            - set(task_registry.TASK_TYPE_META)
            - set(task_registry.DROPPED_TASK_TYPES)
        )
        if unsupported:
            raise RuntimeError(
                "released PM3 queries are not covered by the pinned official "
                f"task registry: {unsupported}"
            )
        return sorted(
            checked,
            key=lambda row: (
                int(row["ts"]), int(row.get("seq") or 0), row["query_id"]
            ),
        )

    def _history_audit(self, row: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = int(row["ts"])
        visible_timestamps: list[int] = []
        per_app: dict[str, int] = {}
        for app in self.backend.APPS:
            events = self.bq.get_events(
                user_id=self.user_id,
                app=app,
                since_timestamp=timestamp,
            )
            per_app[app] = len(events)
            for event in events:
                event_ts = int(event.get("source_timestamp") or 0)
                if event_ts >= timestamp:
                    raise RuntimeError(
                        f"official PM3 backend leaked event {event_ts} into "
                        f"T_test={timestamp}"
                    )
                if "split" in event:
                    raise RuntimeError("official PM3 history exposed event split")
                for preference in event.get("preferences") or []:
                    forbidden = {
                        "split", "over_personalization_irrelevant",
                        "update_history", "confidence_score_init",
                        "confidence_cross_referenced", "stereotype_mark",
                        "hidden_persona_labels",
                    }.intersection(preference)
                    if forbidden:
                        raise RuntimeError(
                            "official PM3 history exposed scoring fields: "
                            f"{sorted(forbidden)}"
                        )
                visible_timestamps.append(event_ts)
        calendar_modifications = self.bq.get_calendar_modifications(
            self.user_id,
            timestamp,
        )
        calendar_timestamps = [
            int(modification.get("ts") or 0)
            for modification in calendar_modifications
            if isinstance(modification, Mapping)
        ]
        if any(value >= timestamp for value in calendar_timestamps):
            raise RuntimeError(
                "official PM3 calendar leaked a same/future modification into "
                f"T_test={timestamp}"
            )
        raw_calendar = self.bq._load_calendar_modifications(self.user_id)
        equal_calendar_excluded = sum(
            1
            for modification in raw_calendar
            if isinstance(modification, Mapping)
            and int(modification.get("ts") or 0) == timestamp
        )
        history_text, stats = self.snapshot_cache.get_or_build(
            self.bq,
            self.user_id,
            timestamp,
            self.model_name,
            self.context_budget,
        )
        if bool(stats.get("truncated", False)):
            raise RuntimeError(
                f"PM3 online memory exceeded context budget "
                f"{self.context_budget}; increase --pm3-context-budget rather "
                "than evaluating a truncated online snapshot"
            )
        if int(stats.get("raw_history_chars", 0) or 0) != 0:
            raise RuntimeError(
                "PM3 online snapshot cache serialized raw history; refusing "
                "to send it to the harness"
            )
        forbidden_calendar_keys = (
            "linked_preferences",
            "is_preference_driven",
            "relation_to_social",
        )
        exposed_calendar_keys = [
            key for key in forbidden_calendar_keys
            if f'"{key}"' in history_text
        ]
        if exposed_calendar_keys:
            raise RuntimeError(
                "official PM3 long-context history exposed calendar persona "
                f"annotations: {exposed_calendar_keys}"
            )
        return {
            "cutoff": HISTORY_CUTOFF,
            "calendar_cutoff": CALENDAR_HISTORY_CUTOFF,
            "calendar_boundary_fix": CALENDAR_BOUNDARY_FIX,
            "t_test": timestamp,
            "strict": all(
                value < timestamp
                for value in (*visible_timestamps, *calendar_timestamps)
            ),
            "event_count": len(visible_timestamps) + len(calendar_timestamps),
            "app_event_count": len(visible_timestamps),
            "max_event_timestamp": max(visible_timestamps, default=None),
            "events_by_app": per_app,
            "calendar_modification_count": len(calendar_timestamps),
            "calendar_max_timestamp": max(calendar_timestamps, default=None),
            "same_timestamp_calendar_modifications_excluded": (
                equal_calendar_excluded
            ),
            "history_tokens": int(stats.get("total_tokens", 0) or 0),
            "history_truncated": bool(stats.get("truncated", False)),
            "online_memory": True,
            "online_memory_characters": int(
                stats.get("memory_characters", len(history_text)) or 0
            ),
            "raw_history_characters": int(
                stats.get("raw_history_chars", 0) or 0
            ),
            "history_sha256": hashlib.sha256(
                history_text.encode("utf-8")
            ).hexdigest(),
            "profile_visible": False,
            "scoring_fields_visible": False,
        }

    def evaluate(
        self,
        row: Mapping[str, Any],
        *,
        llm: LLM,
        bundle: ExecutableHarness,
        judge_client: PrivateJudgeQueryLLM | None,
        condition: str,
        harness_checkpoint: int,
        run_dir: str | Path,
        evaluator_namespace: str | None = None,
        evaluation_protocol: str = PROTOCOL,
        causal_deployment_protocol: bool = True,
        generation_tape: GenerationTape | None = None,
        replay_generation: bool = False,
        query_adapter_factory: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if bundle.user_id != self.user_id:
            raise PermissionError("PM3 native harness belongs to another user")
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "writes.jsonl").touch(exist_ok=True)
        query_id = str(row.get("query_id") or "")
        task_type = str(row.get("task_type") or "")
        instance = json.loads(str(row.get("instance_json") or "{}"))
        # Memory is a turn-start snapshot.  The online updater outside this
        # bridge may replace bundle.memory_md between PM3 rows/windows; this
        # call makes the current state visible to the official task builder
        # without replaying any prior history.
        self.snapshot_cache.attach_memory(bundle.memory_md)
        adapter_options = dict(
            query_id=query_id, task_type=task_type, condition=condition,
            generation_tape=generation_tape, replay_generation=replay_generation,
        )
        adapter = (query_adapter_factory(
            llm, bundle, query_timestamp=int(row["ts"]), **adapter_options,
        ) if query_adapter_factory is not None else HarnessQueryLLM(
            llm,
            bundle,
            **adapter_options,
        ))
        with _official_query_environment(
            row=row,
            user_id=self.user_id,
            backend_dir=self.backend_dir,
            run_dir=output_dir,
        ):
            history_audit = self._history_audit(row)
            cache_ticket = self._cluster_caches.activate(
                evaluator_namespace or condition,
                task_type,
                instance,
                query_id=query_id,
                t_test=int(row["ts"]),
                harness_checkpoint=harness_checkpoint,
            )
            try:
                self.snapshot_cache._pending_query = instance.get("user_query")
                context = self.dispatch.DispatchContext(
                    user_id=self.user_id,
                    bq=self.bq,
                    llm_client=adapter,
                    judge_client=(
                        judge_client if self.enable_llm_judge else None
                    ),
                    mode=EVAL_MODE,
                    snapshot_cache=self.snapshot_cache,
                    model_name=self.model_name,
                    claude_model=self.model_name,
                    context_budget=self.context_budget,
                    enable_llm_judge=(
                        self.enable_llm_judge and judge_client is not None
                    ),
                    dry_run=False,
                )
                started = time.time()
                result = self.dispatch.dispatch_single(
                    task_type, instance, context
                )
                record = self.run_eval._pack_rec(
                    query_id,
                    row.get("seq"),
                    self.user_id,
                    task_type,
                    row.get("ts"),
                    result,
                    int((time.time() - started) * 1000),
                )
                if task_type == LIFECYCLE_TASK_TYPE:
                    result_row = result if isinstance(result, Mapping) else {}
                    phase = str(
                        result_row.get("phase") or instance.get("phase") or ""
                    ).lower()
                    canonical_id = str(
                        result_row.get("canonical_id")
                        or instance.get("canonical_id")
                        or ""
                    )
                    if canonical_id and phase in {"pre", "post", "paired"}:
                        record["_pm3_lifecycle"] = {
                            "pair_key": canonical_id,
                            "canonical_id": canonical_id,
                            "phase": phase,
                        }
            except (AgentPromptLeakError, GenerationReplayError):
                # A benchmark/API error can be represented as a failed official
                # row. A label-firewall violation invalidates the whole run and
                # must stop before a misleading score can be reported.
                raise
            except Exception as exc:  # preserve an official-shaped failed row
                tb = traceback.format_exc()
                last_frame = next(
                    (
                        line.strip()
                        for line in reversed(tb.splitlines())
                        if line.startswith("  File ")
                    ),
                    "",
                )
                record = {
                    "query_id": query_id,
                    "seq": row.get("seq"),
                    "user_id": self.user_id,
                    "task_type": task_type,
                    "ts": row.get("ts"),
                    "metrics_json": "",
                    "status": "error",
                    "duration_ms": 0,
                    "error": (
                        f"{type(exc).__name__}: {exc} | {last_frame}"
                    )[:1000],
                    "agent_response": "",
                }
            cache_audit = self._cluster_caches.finish(cache_ticket)
            if generation_tape is not None:
                generation_tape.finish(replay=replay_generation)
        audit = {
            "protocol": evaluation_protocol,
            "official_mode": EVAL_MODE,
            "official_evaluator_revision": self.revision,
            "official_evaluator_worktree_clean": True,
            "calendar_boundary_fix": CALENDAR_BOUNDARY_FIX,
            "profile_prompt_firewall": PROFILE_PROMPT_FIREWALL,
            "condition": condition,
            "harness_checkpoint": f"W{harness_checkpoint}",
            "effective_harness_checkpoint": (
                ((cache_audit.get("origin") or {}).get("harness_checkpoint"))
                or f"W{harness_checkpoint}"
            ),
            "causal_deployment_protocol": causal_deployment_protocol,
            "retrospective_non_causal": not causal_deployment_protocol,
            "native_feedback_used_for_adaptation": False,
            "history": history_audit,
            "official_response_cache": cache_audit,
            # Kept for report compatibility: this now means RAW evaluator
            # history, which is never forwarded. The compact online memory is
            # reported separately so the two channels cannot be conflated.
            "history_consumed_by_generation": False,
            "online_memory_supplied_to_generation": bool(adapter.traces),
            "harness_input_boundary": {
                "max_current_request_characters": MAX_HARNESS_REQUEST_CHARS,
                "max_online_memory_characters": MAX_ONLINE_MEMORY_CHARS,
                "online_memory_characters": max(
                    [item["memory_characters"] for item in getattr(adapter, "preference_audits", [])]
                    or [len(bundle.memory_md)]
                ),
                "raw_history_characters": 0,
                "raw_history_forwarded": any(
                    bool(item.get("raw_history_forwarded"))
                    for item in adapter.prompt_audits
                ),
            },
            "agent_prompt_firewall": {
                "status": "passed",
                "calls": len(adapter.prompt_audits),
                "prompts": adapter.prompt_audits,
            },
            "harness_traces": adapter.traces,
            "preference_adaptation": getattr(adapter, "preference_audits", []),
            "generation_replay": {"enabled": replay_generation,
                                  "captured": generation_tape is not None and not replay_generation,
                                  "verified_exact_sequence": replay_generation},
        }
        return record, audit

    def summarize(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Use PM3's own task registry and aggregation implementation."""

        repeat_count = max((len(row.get("_judge_repeats") or []) for row in records), default=0)
        if repeat_count:
            summaries = []
            for repeat_index in range(repeat_count):
                repeated = []
                for row in records:
                    metrics = row.get("_judge_repeats")
                    if metrics is not None and len(metrics) != repeat_count:
                        raise ValueError("native judge repeat panel is incomplete")
                    repeated.append({**{key: value for key, value in row.items() if key != "_judge_repeats"},
                        **({"metrics_json": json.dumps(metrics[repeat_index])} if metrics is not None else {})})
                summaries.append(self.summarize(repeated))
            first = summaries[0]
            if not all(summary.get("evaluation_complete") for summary in summaries):
                return {**first, "evaluation_complete": False, "invalid_reason": "incomplete repeated native score"}
            def mean_tables(tables):
                merged = []
                for position, row in enumerate(tables[0]):
                    corresponding = [table[position] for table in tables]
                    if any(other.get("task_type") != row.get("task_type") or other.get("n") != row.get("n") for other in corresponding):
                        raise ValueError("native repeat cohorts differ")
                    numeric = [other.get("accuracy_pct") for other in corresponding]
                    merged.append({**row, **({"accuracy_pct": sum(numeric) / len(numeric)}
                        if all(isinstance(value, (int, float)) for value in numeric) else {})})
                return merged
            return {**first, "score_0_1": sum(summary["score_0_1"] for summary in summaries) / repeat_count,
                    "accuracy_pct_micro": sum(summary["accuracy_pct_micro"] for summary in summaries) / repeat_count,
                    "accuracy_pct_micro_judge": (sum(summary["accuracy_pct_micro_judge"] for summary in summaries) / repeat_count
                        if all(summary.get("accuracy_pct_micro_judge") is not None for summary in summaries) else None),
                    "token_accuracy_table": mean_tables([summary["token_accuracy_table"] for summary in summaries]),
                    "fixed_answer_judge_repeats": repeat_count,
                    "judge_repeat_scores": [summary["score_0_1"] for summary in summaries],
                    "by_task_repeats": [summary["by_task"] for summary in summaries],
                    "by_task_scope": "first repeat; see by_task_repeats and averaged token_accuracy_table"}
        task_registry = importlib.import_module("evaluation.task_registry")
        prepared: list[dict[str, Any]] = []
        official_rows: list[dict[str, Any]] = []
        released_tasks: set[str] = set()
        dropped_rows: list[dict[str, Any]] = []
        for source in records:
            row = dict(source)
            task = task_registry.normalize_task_type(
                str(row.get("task_type") or "")
            )
            released_tasks.add(task)
            if task in task_registry.DROPPED_TASK_TYPES:
                row["_normalized_task_type"] = task
                dropped_rows.append(row)
                continue
            metrics_raw = row.get("metrics_json") or ""
            try:
                metrics = json.loads(metrics_raw) if metrics_raw else {}
            except (TypeError, json.JSONDecodeError):
                metrics = {}
            row["_metrics"] = metrics if isinstance(metrics, dict) else {}
            row["_normalized_task_type"] = task
            prepared.append(row)
            official_rows.append(dict(source))
        lifecycle_pairs, lifecycle_pairing = _synthesize_lifecycle_pairs(prepared)
        aggregation_rows = [*prepared, *lifecycle_pairs]
        aggregation_official_rows = [
            *official_rows,
            *(
                {
                    key: value for key, value in row.items()
                    if not key.startswith("_")
                }
                for row in lifecycle_pairs
            ),
        ]
        e6 = self.aggregate._e6_paired_f1(aggregation_rows)
        table = self.aggregate._build_token_accuracy_table(
            aggregation_rows, e6
        )
        micro = next(
            (
                row.get("accuracy_pct")
                for row in table
                if row.get("task_type") == "ALL (micro, row-weighted)"
            ),
            None,
        )
        judge_micro = next(
            (
                row.get("accuracy_pct")
                for row in table
                if row.get("task_type")
                == "ALL-JUDGE (micro, judge tasks only)"
            ),
            None,
        )
        numeric_micro = (
            float(micro) if isinstance(micro, (int, float)) else None
        )
        scored_tasks = sorted({
            task_registry.normalize_task_type(
                str(row.get("task_type") or "")
            )
            for row in prepared
        })
        dropped_by_task: dict[str, int] = {}
        for row in dropped_rows:
            task = str(row["_normalized_task_type"])
            dropped_by_task[task] = dropped_by_task.get(task, 0) + 1
        n_queries = len(prepared)
        n_ok = sum(row.get("status") == "ok" for row in prepared)
        n_errors = sum(row.get("status") != "ok" for row in prepared)
        metric_complete = bool(lifecycle_pairing["metric_complete"])
        evaluation_complete = (
            n_queries > 0 and n_errors == 0 and n_ok == n_queries
            and metric_complete
        )
        invalid_reason = None
        if n_errors:
            invalid_reason = (
                f"official native evaluation contains {n_errors} error row(s)"
            )
        elif n_ok < n_queries:
            invalid_reason = (
                "official native evaluation has fewer successful rows than "
                f"the scored query count ({n_ok}/{n_queries})"
            )
        elif n_queries == 0:
            invalid_reason = "official native evaluation produced no scored queries"
        elif not metric_complete:
            invalid_reason = (
                "official native lifecycle metric is incomplete: "
                f"{lifecycle_pairing['unpaired_side_count']} unpaired or "
                "ambiguous side(s)"
            )
        return {
            "protocol": PROTOCOL,
            "official_evaluator_revision": self.revision,
            "official_evaluator_worktree_clean": True,
            "official_mode": EVAL_MODE,
            "strict_history_cutoff": HISTORY_CUTOFF,
            "strict_calendar_cutoff": CALENDAR_HISTORY_CUTOFF,
            "calendar_boundary_fix": CALENDAR_BOUNDARY_FIX,
            "profile_prompt_firewall": PROFILE_PROMPT_FIREWALL,
            # ``n_queries`` remains the official headline denominator for
            # compatibility. The explicit released/scored fields prevent the
            # public release's retired rows from disappearing silently.
            "n_queries": len(prepared),
            "n_queries_released": len(records),
            "n_queries_scored": len(prepared),
            "n_queries_dropped_by_official_registry": len(dropped_rows),
            "n_ok": n_ok,
            "n_errors": n_errors,
            "metric_complete": metric_complete,
            "evaluation_complete": evaluation_complete,
            "invalid_reason": invalid_reason,
            "lifecycle_pairing": lifecycle_pairing,
            "released_task_type_count": len(released_tasks),
            "scored_task_type_count": len(scored_tasks),
            "released_task_types": sorted(released_tasks),
            "scored_task_types": scored_tasks,
            "dropped_task_types": sorted(dropped_by_task),
            "dropped_queries_by_task": dropped_by_task,
            "task_types": scored_tasks,
            "accuracy_pct_micro": numeric_micro,
            "accuracy_pct_micro_judge": (
                float(judge_micro)
                if isinstance(judge_micro, (int, float)) else None
            ),
            "score_0_1": (
                numeric_micro / 100.0 if numeric_micro is not None else None
            ),
            "e6_paired": e6,
            "by_task": self.run_eval._summarize_by_task(
                aggregation_official_rows
            ),
            "token_accuracy_table": table,
        }
