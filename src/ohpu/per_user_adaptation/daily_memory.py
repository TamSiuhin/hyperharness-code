"""Persistent, causal maintenance of one user's compact persona memory.

The engine owns ordering, persistence, validation, and the event envelope.  The
learned program shapes the rest through the ``remember`` hook: which observed
events reach the rewriter, which channels matter, how large the document may
be, and one bounded clause of summarization guidance.  That is what makes
persistent user modelling part of the harness code rather than a fixed engine
prompt sitting beside it.  The document's SHAPE is nobody's contract: the seed
layout is a starting suggestion and the rewriter may grow, drop, rename or
reorder sections from there.

The boundary is deliberate.  A program may not reorder the cursor, skip the
causal firewall, write files, or keep an event out of the committed cursor: a
dropped event still advances state, it simply does not reach the rewriter.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ohpu.benchmarks import pm3_history
from ohpu.experiments.observed_history import EventCursor, ObservedEvent
from . import memory_backend, structured_memory
from .program import (
    HarnessProgram,
    MAX_MEMORY_GUIDANCE_CHARS,
    ProgramExecutionError,
    audit_token,
)


MEMORY_PROTOCOL = "daily-full-rewrite-v1"
MEMORY_STATE_SCHEMA_VERSION = 1
MEMORY_COMPLETION_TOKENS = 4096

INITIAL_MEMORY_MD = """# USER MEMORY (last activity seen: none yet)

## Who they are
(none yet)

## Interests & preferences
(none yet)

## People & places
(none yet)

## Currently active
(none yet)
"""

MAINTENANCE_SYSTEM_PROMPT = """You maintain a running MEMORY of ONE user... maintain the memory with ADD, EDIT,
REMOVE, and MERGE actions; and the memory should contain only the resulting
persona profile rather than a raw log of activity.
Start with `# USER MEMORY (last activity seen: <date>)`, keep the whole memory
within 4096 tokens, and output the FULL updated memory each call.
For current calendar commitments, keep the entry identity, title, and exact start
and end strings from timestamps_utc, including the UTC offset and full date.
Do not convert these timestamps to local time or infer a time from an event title.
Unix seconds and their host-rendered UTC strings are authoritative when a source
formatted date disagrees. Apply later changes by entry_id: a field's to value
replaces its from value, including an updated end time. Remove cancelled entries.
Store the resulting current calendar state, not the sequence of modifications."""
DEFAULT_MAINTENANCE_POLICY_SHA256 = hashlib.sha256(
    MAINTENANCE_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()

REPAIR_SUFFIX = """

REPAIR REQUIRED: Return one complete valid MEMORY document only. It must start
with `# USER MEMORY (last activity seen: <date>)`, stay within 4096 tokens, and
contain the resulting persona only -- no action list, change log, or
chronological event log."""

# The document's SHAPE is not a contract. INITIAL_MEMORY_MD seeds a reasonable
# starting layout and the rewriter is free to grow, drop, rename or reorder
# sections from there as the user's evidence warrants. The engine bounds size
# and content class -- no raw event log, no envelope leak, no truncation -- and
# nothing else. An earlier revision enforced a fixed vocabulary, exact
# membership and ordering; on runs/verify-pm3-4u20-5iter-v3-20260902 that cost
# six of twenty update windows across all four users, every one of them for a
# document that was perfectly usable.
MAX_MEMORY_DOCUMENT_CHARS = 12_000
MIN_MEMORY_DOCUMENT_CHARS = 400


def initial_memory_md() -> str:
    """The empty document a rematerialization run folds events into."""

    return INITIAL_MEMORY_MD


_HEADER_RE = re.compile(r"^# USER MEMORY \(last activity seen: [^\n)]*\)$")
_ACTION_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s*)?(?:ADD|EDIT|REMOVE|MERGE)(?:\s|:|->|=>|\||$)"
)
_LOG_HEADING_RE = re.compile(
    r"(?im)^#{1,6}\s*(?:raw\s+)?(?:activity|event|chronological|change|action)\s+log\s*$"
)
_ENVELOPE_LEAK_RE = re.compile(
    r"(?im)^(?:CURRENT COMPLETE MEMORY|USER EVENTS FOR\b|Return the full updated MEMORY)"
)
_DATED_LOG_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s*)?(?:"
    r"\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+-]+Z?)?"
    r"|\d{9,13}"
    r")(?:\s*[:|–—]\s*|\s+-\s+|\s+\S)"
)
_JSON_EVENT_LINE_RE = re.compile(
    r'(?im)^\s*(?:[-*+]\s*)?\{[^\n]*(?:"channel"|"value"|"context")[^\n]*\}\s*$'
)
_ACTIVITY_BULLET_RE = re.compile(
    r"(?im)^\s*[-*+]\s*(?:(?:the\s+)?(?:user|they|he|she)\s+)?"
    r"(?:recently\s+)?(?:clicked|watched|viewed|liked|disliked|searched|"
    r"bought|purchased|visited|opened|booked|asked|said|posted|reacted|"
    r"played|listened|read|wrote|used|selected|chose)\b"
)


def _cursor_to_json(cursor: EventCursor | None) -> dict[str, Any] | None:
    if cursor is None:
        return None
    return {
        "episode_order": cursor.episode_order,
        "episode_rank": cursor.episode_rank,
        "item_rank": cursor.item_rank,
        "event_rank": cursor.event_rank,
        "timestamp": cursor.timestamp,
    }


def _cursor_from_json(value: object) -> EventCursor | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("last_committed_cursor must be an object or null")
    expected = {
        "episode_order", "episode_rank", "item_rank", "event_rank", "timestamp"
    }
    if set(value) != expected:
        raise ValueError("last_committed_cursor has an invalid schema")
    try:
        timestamp = str(value.get("timestamp") or "")
        if len(timestamp) > 128 or "\n" in timestamp or "\r" in timestamp:
            raise ValueError("last_committed_cursor timestamp is invalid")
        return EventCursor(
            episode_order=int(value["episode_order"]),
            episode_rank=int(value["episode_rank"]),
            item_rank=int(value["item_rank"]),
            event_rank=int(value["event_rank"]),
            timestamp=timestamp,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid last_committed_cursor") from exc


@dataclass(frozen=True)
class MemoryState:
    """The complete persisted state; it intentionally contains no event body."""

    user_id: str
    last_committed_cursor: EventCursor | None = None
    last_rewrite_day: str | None = None
    schema_version: int = MEMORY_STATE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "last_committed_cursor": _cursor_to_json(self.last_committed_cursor),
            "last_rewrite_day": self.last_rewrite_day,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryState":
        expected = {
            "schema_version",
            "user_id",
            "last_committed_cursor",
            "last_rewrite_day",
        }
        if set(value) != expected:
            raise ValueError("memory state has an invalid schema")
        version = value.get("schema_version")
        if version != MEMORY_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported memory state schema {version!r}; "
                f"expected {MEMORY_STATE_SCHEMA_VERSION}"
            )
        user_id = value.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("memory state user_id must be a non-empty string")
        rewrite_day = value.get("last_rewrite_day")
        if rewrite_day is not None and not isinstance(rewrite_day, str):
            raise ValueError("last_rewrite_day must be a string or null")
        if rewrite_day is not None and not re.fullmatch(
            r"(?:\d{4}-\d{2}-\d{2}|logical-order--?\d+)", rewrite_day
        ):
            raise ValueError("last_rewrite_day has an invalid format")
        return cls(
            user_id=user_id,
            last_committed_cursor=_cursor_from_json(
                value.get("last_committed_cursor")
            ),
            last_rewrite_day=rewrite_day,
            schema_version=version,
        )


@dataclass(frozen=True)
class MemorySnapshot:
    memory_md: str
    state: MemoryState
    maintenance_policy_sha256: str = DEFAULT_MAINTENANCE_POLICY_SHA256


@dataclass(frozen=True)
class MemoryUpdateResult:
    success: bool
    logical_days: tuple[str, ...]
    committed_cursor: EventCursor | None
    error: str | None = None
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    errors: int = 0
    repaired: bool = False
    plan: "MemoryPlan | None" = None
    structured_capacity: dict[str, Any] | None = None
    backend_report: dict[str, Any] | None = None
    backend_execution: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "rewrite_day": self.logical_days[-1] if self.logical_days else None,
            "source_days": list(self.logical_days),
            "committed_cursor": _cursor_to_json(self.committed_cursor),
            "error": self.error,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "errors": self.errors,
            "repaired": self.repaired,
            "program_plan": self.plan.as_dict() if self.plan is not None else None,
            **({"structured_capacity": self.structured_capacity} if self.structured_capacity is not None else {}),
            **({"memory_backend": self.backend_report} if self.backend_report is not None else {}),
        }


@dataclass
class MemoryUpdateStats:
    updates: list[MemoryUpdateResult] = field(default_factory=list)

    def add(self, result: MemoryUpdateResult) -> None:
        self.updates.append(result)

    def totals(self) -> dict[str, int]:
        return {
            key: sum(getattr(update, key) for update in self.updates)
            for key in (
                "calls",
                "cache_hits",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "errors",
            )
        }


@dataclass
class _DayBucket:
    day: str
    events: list[ObservedEvent] = field(default_factory=list)
    closed: bool = False


@dataclass
class _CallAccounting:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    errors: int = 0

    def add_response(
        self,
        response: Mapping[str, Any],
    ) -> None:
        if bool(response.get("_cache_hit", False)):
            self.cache_hits += 1
            return

        usage = response.get("usage")
        if isinstance(usage, Mapping):
            self.prompt_tokens += _integer(usage.get("prompt") or usage.get("prompt_tokens"))
            self.completion_tokens += _integer(
                usage.get("completion") or usage.get("completion_tokens")
            )
            self.reasoning_tokens += _integer(
                usage.get("reasoning") or usage.get("reasoning_tokens")
            )


def _integer(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class MemoryServiceRefusal(RuntimeError):
    """A provider refusal that must not enter the memory repair loop."""


class MemoryIntegrityError(RuntimeError):
    pass


class MemoryDocumentError(ValueError):
    """The rewriter's output violated the engine's document contract.

    Every message raised as this type is a fixed structural string with no
    prompt, event, or profile content, so it is safe to report. That matters
    operationally: without it a run that dies on "the document exceeded the
    token budget" is indistinguishable from one that died on a provider error.
    """


def _safe_error(exc: Exception, stage: str) -> str:
    """Describe updater failures without copying prompts, events, or profiles."""

    if isinstance(exc, MemoryDocumentError):
        return f"MemoryDocumentError: {exc}"
    # Client libraries may raise ValueError with request or response fragments,
    # so every other exception message is not report-safe.
    return f"{type(exc).__name__}: memory updater {stage} failed"


def estimate_token_count(text: str) -> int:
    """A deterministic conservative count used when no tokenizer is available."""

    return len(re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7f]|[^\w\s]", text))


def memory_sha256(memory_md: str) -> str:
    return hashlib.sha256(memory_md.encode("utf-8")).hexdigest()


def last_activity_header(memory_md: str) -> str | None:
    first = memory_md.splitlines()[0] if memory_md else ""
    return first if _HEADER_RE.fullmatch(first) else None


def normalize_logical_day(
    activity_timestamp: object | None,
    *,
    episode_timestamp: object | None = None,
    episode_order: int,
) -> str:
    """Normalize Unix/ISO activity time to a UTC day with causal fallbacks."""

    for candidate in (activity_timestamp, episode_timestamp):
        parsed = _parse_utc_datetime(candidate)
        if parsed is not None:
            return parsed.date().isoformat()
    return f"logical-order-{episode_order}"


def _parse_utc_datetime(value: object | None) -> dt.datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        try:
            stamp = float(text)
            magnitude = abs(stamp)
            if magnitude >= 1e14:
                stamp /= 1_000_000
            elif magnitude >= 1e11:
                stamp /= 1_000
            return dt.datetime.fromtimestamp(stamp, tz=dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        try:
            parsed_date = dt.date.fromisoformat(candidate)
        except ValueError:
            return None
        return dt.datetime.combine(parsed_date, dt.time(), tzinfo=dt.timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def event_logical_day(event: ObservedEvent) -> str:
    return normalize_logical_day(
        event.cursor.timestamp,
        episode_order=event.cursor.episode_order,
    )


def render_event_batch(events: Sequence[ObservedEvent]) -> str:
    """Render only the already-sanitized evidence fields, never provenance."""

    rows: list[str] = []
    for event in events:
        payload = {
            "channel": event.channel,
            "value": event.value,
        }
        if event.context:
            payload["context"] = event.context
        if isinstance(event.history_payload, Mapping):
            # PM3 textual memory is built from the same official compact event
            # view as its long-context baseline. The payload was leak-stripped
            # by the loader and becomes visible only after this event's cursor.
            payload["history_event"] = pm3_history.bounded_compact_event(
                event.history_payload
            )
        rows.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return "\n".join(rows)


@dataclass(frozen=True)
class MemoryPlan:
    """The remember decision for one memory rewrite batch.

    ``kept`` is the subset of the batch that reaches the rewriter.  The batch's
    final cursor is committed either way: a program decides what is worth
    remembering, never what the engine has observed.
    """

    max_chars: int | None = None
    guidance: str = ""
    kept_indices: tuple[int, ...] = ()
    dropped_count: int = 0
    applied_keys: tuple[str, ...] = ()
    effect_digest: str = ""
    error: str | None = None
    state_schema: dict[str, Any] | None = field(default=None, repr=False)
    state_records: list[dict[str, Any]] | None = field(default=None, repr=False)
    backend: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def effective(self) -> bool:
        return bool(self.applied_keys)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook": "remember",
            "max_chars": self.max_chars,
            "guidance_chars": len(self.guidance),
            "events_kept": len(self.kept_indices),
            "events_dropped": self.dropped_count,
            "applied_keys": list(self.applied_keys),
            "effective": self.effective,
            "effect_digest": self.effect_digest,
            "error": self.error,
            "structured_state": self.state_schema is not None or self.state_records is not None,
        }


def default_memory_plan(event_count: int) -> MemoryPlan:
    return MemoryPlan(kept_indices=tuple(range(event_count)))


def memory_event_view(event: ObservedEvent) -> dict[str, Any]:
    """The bounded per-event view a remember plan may branch on."""

    return {
        "channel": str(event.channel or ""),
        "value": str(event.value or "")[:400],
        "context": str(event.context or "")[:200],
        "day": event_logical_day(event),
    }


def compute_memory_plan(
    program: HarnessProgram | None,
    *,
    current_memory: str,
    logical_days: Sequence[str],
    events: Sequence[ObservedEvent],
    backend_enabled: bool = False,
) -> MemoryPlan:
    """Ask ``remember`` how to compile this batch; fail closed to the default."""

    fallback = default_memory_plan(len(events))
    if program is None or not events:
        return fallback
    context = {
        "logical_days": list(logical_days),
        "events": [
            {"index": index, **memory_event_view(event)}
            for index, event in enumerate(events)
        ],
        "channels": sorted({str(event.channel or "") for event in events}),
        # What the document happens to look like right now. It is context for
        # a guidance clause, not a schema: the rewriter may change it.
        "current_memory_sections": list(_document_sections(current_memory)),
        **structured_memory.prepare_context(current_memory, events, logical_days[-1]),
        "base": {
            "max_chars": None,
            "guidance": "",
            "keep_events": list(range(len(events))),
        },
    }
    if backend_enabled:
        store, public = memory_backend.prepare_write(current_memory, events, logical_days[-1], events[0].user_id,
            program.fingerprint.hook_sha256["remember"])
        context.update(public)
        context["backend"] = {**store.describe(), "phase": "write", "step": 0,
            "state": {}, "result": {"data": None, "truncated": False}, "model_calls_remaining": memory_backend.MAX_MODEL_CALLS}
    try:
        outcome = program.call("remember", context)
    except ProgramExecutionError as exc:
        return MemoryPlan(
            kept_indices=fallback.kept_indices,
            error=str(exc)[:200],
        )
    patch = outcome.patch or {}
    if "memory_backend" in patch and not backend_enabled:
        return MemoryPlan(kept_indices=fallback.kept_indices, error="programmable_memory_not_enabled", backend=patch["memory_backend"])
    schema = patch.get("state_schema")
    if schema is not None:
        try:
            structured_memory.validate_schema(schema)
        except Exception:
            return MemoryPlan(kept_indices=fallback.kept_indices,
                              error="invalid_structured_state_schema")

    dropped_channels = {
        str(name) for name in (patch.get("drop_channels") or [])
        if isinstance(name, str)
    }
    requested_keep = patch.get("keep_events")
    keep_filter: set[int] | None = None
    if isinstance(requested_keep, list):
        keep_filter = {
            int(value)
            for value in requested_keep
            if isinstance(value, int) and not isinstance(value, bool)
        }
    kept = [
        index
        for index, event in enumerate(events)
        if str(event.channel or "") not in dropped_channels
        and (keep_filter is None or index in keep_filter)
    ]
    # An empty selection is a real decision, not an error: a day of pure
    # noise should cost neither a rewrite nor a place in the document.  The
    # engine still advances the cursor over the whole batch, so dropping
    # everything cannot stall or replay the stream.  Because memory is
    # day-bucketed, forcing the batch back in here would have let an all-noise
    # day defeat a channel filter that works on every mixed day.
    selection_applied: list[str] = []
    if dropped_channels and len(kept) != len(events):
        selection_applied.append("drop_channels")
    if keep_filter is not None and len(kept) != len(events):
        selection_applied.append("keep_events")

    priority = [
        str(name) for name in (patch.get("priority_channels") or [])
        if isinstance(name, str)
    ]
    if priority:
        rank = {name: position for position, name in enumerate(priority)}
        ordered = sorted(
            kept,
            key=lambda index: (
                rank.get(str(events[index].channel or ""), len(rank)),
                index,
            ),
        )
        if ordered != kept:
            selection_applied.append("priority_channels")
            kept = ordered

    max_chars: int | None = None
    raw_max = patch.get("max_chars")
    if isinstance(raw_max, int) and not isinstance(raw_max, bool):
        max_chars = max(
            MIN_MEMORY_DOCUMENT_CHARS,
            min(int(raw_max), MAX_MEMORY_DOCUMENT_CHARS),
        )
        selection_applied.append("max_chars")

    guidance = patch.get("guidance")
    guidance_text = (
        guidance.strip()[:MAX_MEMORY_GUIDANCE_CHARS]
        if isinstance(guidance, str) else ""
    )
    if guidance_text:
        selection_applied.append("guidance")
    for key in ("state_schema", "state_records", "memory_backend"):
        if key in patch:
            selection_applied.append(key)

    applied = tuple(sorted(set(selection_applied)))
    return MemoryPlan(
        max_chars=max_chars,
        guidance=guidance_text,
        state_schema=schema,
        state_records=patch.get("state_records"),
        backend=patch.get("memory_backend"),
        kept_indices=tuple(kept),
        dropped_count=len(events) - len(kept),
        applied_keys=applied,
        effect_digest=audit_token({
            "max_chars": max_chars,
            "kept": list(kept),
            "guidance": guidance_text,
            "state_schema": schema,
            "state_records": patch.get("state_records"),
            **({"memory_backend": patch["memory_backend"]} if "memory_backend" in patch else {}),
        }) if applied else "",
    )


def _document_sections(memory_md: str) -> tuple[str, ...]:
    return tuple(
        line[3:].strip()
        for line in str(memory_md or "").splitlines()
        if line.startswith("## ")
    )


def maintenance_prompt_for_plan(base_prompt: str, plan: MemoryPlan) -> str:
    """Render the engine prompt for one remember decision.

    Everything here is a request, not a contract: the document is accepted on
    the engine's bounds alone, so a rewriter that overshoots a budget or lays
    the sections out differently still produces a usable memory.
    """

    parts = [base_prompt.strip()]
    if plan.max_chars is not None:
        parts.append(
            f"Keep the whole document within {plan.max_chars} characters."
        )
    if plan.guidance:
        parts.append("User-specific compilation guidance: " + plan.guidance)
    return "\n".join(parts)


def build_maintenance_user_message(
    current_memory: str,
    logical_days: Sequence[str],
    events: Sequence[ObservedEvent],
) -> str:
    day_label = ", ".join(logical_days)
    return (
        "CURRENT COMPLETE MEMORY\n"
        f"{current_memory.rstrip()}\n\n"
        f"USER EVENTS FOR {day_label}\n"
        f"{render_event_batch(events)}\n\n"
        "Return the full updated MEMORY document only."
    )


def validate_memory_document(
    text: str,
    *,
    finish_reason: object | None = None,
    completion_tokens: int | None = None,
    raw_events: Sequence[ObservedEvent] = (),
) -> str:
    """Return normalized full memory or raise ``MemoryDocumentError``.

    What is checked is what the engine owns: the document is not truncated, it
    carries the activity header the cursor bookkeeping reads, and it is a
    persona rather than a copy of the event stream. Its section layout is the
    rewriter's business -- headings may be added, dropped, renamed or reordered
    between rewrites, and reading one back never depends on which layout the
    version that compiled it happened to choose.
    """

    if str(finish_reason or "").casefold() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }:
        raise MemoryDocumentError("memory output was length-truncated")
    document = str(text or "").strip()
    if not document:
        raise MemoryDocumentError("memory output is empty")
    first_line = document.splitlines()[0]
    if not _HEADER_RE.fullmatch(first_line):
        raise MemoryDocumentError("memory output has an invalid header")
    if memory_backend.is_backend(document):
        try:
            memory_backend.MemoryStore.payload(document)
        except Exception:
            raise MemoryDocumentError("invalid programmable memory document") from None
        return document + "\n"
    if structured_memory.is_structured(document):
        try:
            structured_memory.read_records(document)
        except (TypeError, ValueError):
            raise MemoryDocumentError("invalid structured memory document") from None
    if _ACTION_LINE_RE.search(document) or _LOG_HEADING_RE.search(document):
        raise MemoryDocumentError("memory output contains an action or chronological log")
    if _ENVELOPE_LEAK_RE.search(document):
        raise MemoryDocumentError("memory output copied the maintenance envelope")
    if _JSON_EVENT_LINE_RE.search(document):
        raise MemoryDocumentError("memory output contains a raw event row")
    if len(_ACTIVITY_BULLET_RE.findall(document)) >= 2:
        raise MemoryDocumentError("memory output contains a per-activity bullet log")
    if len(_DATED_LOG_LINE_RE.findall(document)) >= 2:
        raise MemoryDocumentError("memory output contains a raw dated chronology")
    _validate_no_verbatim_event_log(document, raw_events)
    measured_tokens = max(completion_tokens or 0, estimate_token_count(document))
    # Treat an exactly exhausted completion budget as truncated too.  Some
    # providers report ``stop`` even when the response consumed the complete
    # allowance, so accepting 4096 here would make truncation depend on a
    # provider-specific finish reason.
    if measured_tokens >= MEMORY_COMPLETION_TOKENS:
        raise MemoryDocumentError("memory output exhausts or exceeds 4096 tokens")
    return document + "\n"


def _validate_no_verbatim_event_log(
    document: str,
    events: Sequence[ObservedEvent],
) -> None:
    """Reject contextual evidence that was copied as an activity dump.

    Persona bullets can legitimately preserve one short preference verbatim,
    so the fail-closed threshold is either one long raw payload or two distinct
    event payloads reproduced verbatim in profile lines.
    """
    if not events:
        return

    def normalize(value: object) -> str:
        return " ".join(str(value or "").casefold().split())

    profile_lines = {
        normalize(re.sub(r"^\s*[-*+]\s*", "", line))
        for line in document.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    normalized_document = normalize(document)
    line_matches: set[str] = set()
    for event in events:
        for raw in (event.value, event.context):
            payload = normalize(raw)
            if not payload:
                continue
            if len(payload) >= 8 and any(
                payload in profile_line for profile_line in profile_lines
            ):
                line_matches.add(payload)
            if len(payload) >= 160 and payload in normalized_document:
                raise MemoryDocumentError("memory output copied a long raw event payload")
    if len(line_matches) >= 2:
        raise MemoryDocumentError("memory output copied multiple raw events verbatim")


def correct_last_activity_header(memory_md: str, last_activity: str) -> str:
    lines = memory_md.rstrip().splitlines()
    if not lines or (
        lines[0] != "# USER MEMORY" and not _HEADER_RE.fullmatch(lines[0])
    ):
        raise MemoryDocumentError("cannot correct an invalid memory header")
    lines[0] = f"# USER MEMORY (last activity seen: {last_activity})"
    return "\n".join(lines) + "\n"


def _state_json(state: MemoryState) -> str:
    return json.dumps(state.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _read_state(path: Path, user_id: str | None = None) -> MemoryState:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("memory state must be a JSON object")
    state = MemoryState.from_dict(raw)
    if user_id is not None and state.user_id != user_id:
        raise ValueError(
            f"memory state belongs to {state.user_id!r}, not {user_id!r}"
        )
    return state


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_fsynced_temp(path: Path, text: str, *, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    temp_path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _write_fsynced_temp(path, text, prefix=f".{path.name}.tmp.")
    try:
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _pending_state_prefix(state_path: Path) -> str:
    return f".{state_path.name}.pending."


def _atomic_replace_memory_then_state(
    memory_path: Path,
    state_path: Path,
    memory_md: str,
    state: MemoryState,
) -> None:
    """Commit memory then state, retaining recoverable metadata between replaces."""

    if memory_path.parent != state_path.parent:
        raise ValueError("memory and state must live in the same directory")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    digest = memory_sha256(memory_md)
    transaction = uuid.uuid4().hex
    memory_temp = _write_fsynced_temp(
        memory_path, memory_md, prefix=f".{memory_path.name}.tmp."
    )
    pending_state = _write_fsynced_temp(
        state_path,
        _state_json(state),
        prefix=f"{_pending_state_prefix(state_path)}{digest}.{transaction}.",
    )
    _fsync_directory(memory_path.parent)
    try:
        os.replace(memory_temp, memory_path)
        _fsync_directory(memory_path.parent)
        # If the process dies here, the pending filename binds the new state to
        # the exact canonical memory bytes without storing an event body.
        os.replace(pending_state, state_path)
        _fsync_directory(memory_path.parent)
    finally:
        memory_temp.unlink(missing_ok=True)
        # Deliberately keep pending_state on a failed second replace so startup
        # can finish the transaction after matching the canonical memory hash.


def _recover_interrupted_commit(
    memory_path: Path,
    state_path: Path,
    *,
    user_id: str,
) -> bool:
    if not memory_path.exists():
        return False
    digest = memory_sha256(memory_path.read_text(encoding="utf-8"))
    prefix = _pending_state_prefix(state_path)
    candidates: list[tuple[EventCursor | None, float, Path, MemoryState]] = []
    for path in state_path.parent.glob(f"{prefix}*"):
        suffix = path.name[len(prefix):]
        if not suffix.startswith(digest + "."):
            continue
        try:
            state = _read_state(path, user_id)
            modified = path.stat().st_mtime
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((state.last_committed_cursor, modified, path, state))
    if not candidates:
        return False
    # Cursor order is authoritative. mtime breaks the initial-state tie.
    _cursor, _modified, chosen, _state = max(
        candidates,
        key=lambda row: (
            row[0] is not None,
            row[0] or EventCursor(-1, -1, -1, -1),
            row[1],
        ),
    )
    os.replace(chosen, state_path)
    _fsync_directory(state_path.parent)
    return True


@contextlib.contextmanager
def _memory_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".memory.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize_memory_files(
    memory_path: str | Path,
    state_path: str | Path,
    user_id: str,
) -> None:
    memory = Path(memory_path)
    state = Path(state_path)
    if memory.parent != state.parent:
        raise ValueError("memory and state must live in the same directory")
    with _memory_lock(memory.parent):
        _recover_interrupted_commit(memory, state, user_id=user_id)
        if not memory.exists() and not state.exists():
            _atomic_replace_memory_then_state(
                memory, state, INITIAL_MEMORY_MD, MemoryState(user_id=user_id)
            )
        elif memory.exists() and not state.exists():
            existing = validate_memory_document(
                memory.read_text(encoding="utf-8")
            )
            if existing != INITIAL_MEMORY_MD:
                raise ValueError(
                    "non-initial memory.md is missing its causal state"
                )
            _atomic_write_text(state, _state_json(MemoryState(user_id=user_id)))
        elif state.exists() and not memory.exists():
            raise ValueError("memory state exists but memory.md is missing")
        else:
            validate_memory_document(memory.read_text(encoding="utf-8"))
            _read_state(state, user_id)


class MemoryWorkspace:
    """A private copy-on-write fork of canonical memory and state.

    ``workspace_dir`` is the private ``.memoharness`` directory.  Leaving it
    unset creates an automatically cleaned temporary directory.  Exiting the
    context never commits; callers must invoke :meth:`commit` explicitly.
    """

    def __init__(
        self,
        canonical_memory_path: str | Path,
        canonical_state_path: str | Path,
        user_id: str,
        workspace_dir: str | Path | None = None,
    ) -> None:
        self.canonical_memory_path = Path(canonical_memory_path)
        self.canonical_state_path = Path(canonical_state_path)
        self.user_id = user_id
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if workspace_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="ohpu-memory-")
            self.directory = Path(self._temporary.name)
        else:
            self.directory = Path(workspace_dir)
            self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.directory / "memory.md"
        self.state_path = self.directory / "memory.state.json"
        self._closed = False
        self._committed = False

        initialize_memory_files(
            self.canonical_memory_path, self.canonical_state_path, user_id
        )
        with _memory_lock(self.canonical_memory_path.parent):
            _recover_interrupted_commit(
                self.canonical_memory_path,
                self.canonical_state_path,
                user_id=user_id,
            )
            canonical_memory = self.canonical_memory_path.read_text(encoding="utf-8")
            canonical_state = _read_state(self.canonical_state_path, user_id)
        with _memory_lock(self.directory):
            _atomic_replace_memory_then_state(
                self.memory_path,
                self.state_path,
                canonical_memory,
                canonical_state,
            )

    @classmethod
    def fork(
        cls,
        canonical_memory_path: str | Path,
        canonical_state_path: str | Path,
        user_id: str,
        workspace_dir: str | Path | None = None,
    ) -> "MemoryWorkspace":
        return cls(
            canonical_memory_path,
            canonical_state_path,
            user_id,
            workspace_dir,
        )

    @property
    def committed(self) -> bool:
        return self._committed

    def commit(self) -> None:
        if self._closed:
            raise RuntimeError("memory workspace is closed")
        with _memory_lock(self.directory):
            _recover_interrupted_commit(
                self.memory_path, self.state_path, user_id=self.user_id
            )
            private_memory = self.memory_path.read_text(encoding="utf-8")
            validate_memory_document(private_memory)
            private_state = _read_state(self.state_path, self.user_id)
        with _memory_lock(self.canonical_memory_path.parent):
            _recover_interrupted_commit(
                self.canonical_memory_path,
                self.canonical_state_path,
                user_id=self.user_id,
            )
            canonical_state = _read_state(
                self.canonical_state_path, self.user_id
            )
            if (
                canonical_state.last_committed_cursor is not None
                and (
                    private_state.last_committed_cursor is None
                    or private_state.last_committed_cursor
                    < canonical_state.last_committed_cursor
                )
            ):
                raise RuntimeError("refusing to commit a stale memory workspace")
            _atomic_replace_memory_then_state(
                self.canonical_memory_path,
                self.canonical_state_path,
                private_memory,
                private_state,
            )
        self._committed = True

    def discard(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._temporary is not None:
            self._temporary.cleanup()

    def __enter__(self) -> "MemoryWorkspace":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class DailyMemoryManager:
    """Observe causal user events and periodically rewrite a full persona file."""

    def __init__(
        self,
        llm: Any,
        user_id: str,
        memory_path: str | Path,
        state_path: str | Path | None = None,
        *,
        rewrite_days: int = 1,
        model: str | None = None,
        seed: int = 7,
        maintenance_prompt: str = MAINTENANCE_SYSTEM_PROMPT,
        maintenance_prompt_path: str | Path | None = None,
        repair_suffix: str = REPAIR_SUFFIX,
        program: HarnessProgram | None = None,
        on_observe: Any = None,
        backend_enabled: bool = False,
    ) -> None:
        if (
            isinstance(rewrite_days, bool)
            or not isinstance(rewrite_days, int)
            or rewrite_days < 1
        ):
            raise ValueError("rewrite_days must be a positive integer")
        if maintenance_prompt_path is not None:
            maintenance_prompt = Path(maintenance_prompt_path).read_text(
                encoding="utf-8"
            )
        if not isinstance(maintenance_prompt, str) or not maintenance_prompt.strip():
            raise ValueError("maintenance_prompt must be a non-empty string")
        if not isinstance(repair_suffix, str) or not repair_suffix.strip():
            raise ValueError("repair_suffix must be a non-empty string")
        self.llm = llm
        self.user_id = user_id
        self.memory_path = Path(memory_path)
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else self.memory_path.with_name("memory.state.json")
        )
        self.rewrite_days = int(rewrite_days)
        self.model = model
        self.seed = seed
        self.maintenance_prompt = maintenance_prompt.strip()
        self.repair_suffix = repair_suffix
        self.maintenance_policy_sha256 = hashlib.sha256(
            (self.maintenance_prompt if repair_suffix == REPAIR_SUFFIX else json.dumps(
                {"maintenance_prompt": self.maintenance_prompt, "repair_suffix": repair_suffix}, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        # The ``remember`` hook compiles the persistent user model.
        # Leaving it unset keeps the fixed engine compiler, which is what the
        # ablation arm and every legacy bundle need.
        self.program = program
        self.backend_enabled = backend_enabled
        # Lets the host accumulate exactly the causal prefix this manager
        # accepted, which is what a harness-versioned replay has to fold. It
        # is notification only: the callback cannot change what is observed.
        self.on_observe = on_observe
        self.stats = MemoryUpdateStats()
        self._buckets: list[_DayBucket] = []
        self._pending_cursors: set[EventCursor] = set()
        self._as_of_day: str | None = None

        initialize_memory_files(self.memory_path, self.state_path, user_id)
        self._reload_from_disk()

    @property
    def memory_md(self) -> str:
        return self.render()

    @property
    def state(self) -> MemoryState:
        return self._state

    def render(self) -> str:
        return memory_backend.prune_expired(structured_memory.prune_expired(self._memory_md, self._as_of_day), self._as_of_day)

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            memory_md=self.render(),
            state=self._state,
            maintenance_policy_sha256=self.maintenance_policy_sha256,
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        if snapshot.state.user_id != self.user_id:
            raise ValueError("cannot restore another user's memory snapshot")
        memory = validate_memory_document(snapshot.memory_md)
        self._check_backend_identity(memory)
        with _memory_lock(self.memory_path.parent):
            _atomic_replace_memory_then_state(
                self.memory_path, self.state_path, memory, snapshot.state
            )
        self._buckets.clear()
        self._pending_cursors.clear()
        self._as_of_day = snapshot.state.last_rewrite_day
        self._reload_from_disk()

    def observe(
        self, events: Iterable[ObservedEvent]
    ) -> list[MemoryUpdateResult]:
        """Observe ordered strict-user events and flush each newly closed period."""

        results = self._flush_ready(force=False)
        ordered = list(events)
        previous: EventCursor | None = None
        for event in ordered:
            if previous is not None and event.cursor < previous:
                raise ValueError("observed events must be in EventCursor order")
            previous = event.cursor
            if event.user_id != self.user_id:
                raise ValueError(
                    f"observed event belongs to {event.user_id!r}, not {self.user_id!r}"
                )
            if not event.is_strict_user_evidence:
                continue
            if (
                self._state.last_committed_cursor is not None
                and event.cursor <= self._state.last_committed_cursor
            ):
                continue
            if event.cursor in self._pending_cursors:
                continue
            high_water = self._pending_high_water()
            if high_water is not None and event.cursor <= high_water:
                raise ValueError("new observed event precedes pending memory evidence")
            day = event_logical_day(event)
            if self._buckets and day != self._buckets[-1].day:
                self._buckets[-1].closed = True
                results.extend(self._flush_ready(force=False))
            if not self._buckets or self._buckets[-1].closed:
                self._buckets.append(_DayBucket(day=day))
            self._buckets[-1].events.append(event)
            self._pending_cursors.add(event.cursor)
            if self.on_observe is not None:
                self.on_observe(event)
        return results

    def flush(self, *, force: bool = True) -> MemoryUpdateResult | None:
        """Flush pending evidence; finite replays should call with ``force=True``."""

        results = self._flush_ready(force=force)
        return results[-1] if results else None

    def advance_to(
        self,
        timestamp: object | None = None,
        *,
        episode_order: int,
    ) -> list[MemoryUpdateResult]:
        """Close activity buckets when the current episode enters another day.

        This is intentionally separate from :meth:`observe`: a task on day D
        may have no newly visible event on D, but must still see memory flushed
        through day D-1.  An empty interval is never materialized as evidence.
        """

        results = self._flush_ready(force=False)
        self._as_of_day = normalize_logical_day(timestamp, episode_order=episode_order)
        if not self._buckets:
            return results
        boundary_day = normalize_logical_day(
            timestamp,
            episode_order=episode_order,
        )
        if boundary_day != self._buckets[-1].day:
            self._buckets[-1].closed = True
            results.extend(self._flush_ready(force=False))
        return results

    def flush_all(self, *, force: bool = True) -> list[MemoryUpdateResult]:
        """Variant preserving all results when multiple periods flush at once."""

        return self._flush_ready(force=force)

    def close_observation_window(self) -> list[MemoryUpdateResult]:
        """Close the current bucket without forcing a short rewrite period.

        Harness windows and memory periods are independent clocks. At a known
        causal window boundary no more events belong to the current bucket, so
        it may be closed; a rewrite still occurs only after ``rewrite_days``
        closed buckets have accumulated. A finite stream may separately call
        :meth:`flush_all` with ``force=True`` for its final partial period.
        """

        self._sync_recovered_commit()
        if self._buckets:
            self._buckets[-1].closed = True
        return self._flush_ready(force=False)

    def recover(self) -> bool:
        """Finish a memory-before-state transaction and refresh this manager."""

        before = self._state.last_committed_cursor
        self._sync_recovered_commit()
        return self._state.last_committed_cursor != before

    def report(self) -> dict[str, Any]:
        # A state replace can fail after durable memory bytes were installed.
        # Reporting is a safe recovery boundary and must describe disk truth.
        self._sync_recovered_commit()
        return {
            "protocol": MEMORY_PROTOCOL,
            "user_id": self.user_id,
            "rewrite_days": self.rewrite_days,
            "maintenance_policy_sha256": self.maintenance_policy_sha256,
            "updates": [update.as_dict() for update in self.stats.updates],
            "totals": self.stats.totals(),
            "memory": {
                "sha256": memory_sha256(self._memory_md),
                "token_count": estimate_token_count(self._memory_md),
                "last_activity": last_activity_header(self._memory_md),
                "sections": list(_document_sections(self._memory_md)),
            },
            # The remember stage is audited exactly like the runtime
            # stages: a returned plan is not evidence, an applied key is.
            "program_memory": {
                "enabled": self.program is not None,
                "program_sha256": (
                    self.program.fingerprint.sha256
                    if self.program is not None else None
                ),
                "invocations": sum(
                    1 for update in self.stats.updates if update.plan is not None
                ),
                "effective": sum(
                    1
                    for update in self.stats.updates
                    if update.plan is not None and update.plan.effective
                ),
                "applied_keys": sorted({
                    key
                    for update in self.stats.updates
                    if update.plan is not None
                    for key in update.plan.applied_keys
                }),
                "events_dropped": sum(
                    update.plan.dropped_count
                    for update in self.stats.updates
                    if update.plan is not None
                ),
                "errors": sorted({
                    update.plan.error
                    for update in self.stats.updates
                    if update.plan is not None and update.plan.error
                }),
            },
        }

    def reset(self) -> None:
        """Clear memory back to the seed document.

        This is the first half of a harness-versioned rematerialization: the
        new program must fold the observed prefix itself rather than inherit a
        document its predecessor compiled.
        """

        self.restore(MemorySnapshot(
            memory_md=initial_memory_md(),
            state=MemoryState(user_id=self.user_id),
        ))

    def rematerialize(
        self,
        events: Iterable[ObservedEvent],
    ) -> list[MemoryUpdateResult]:
        """Rebuild memory from empty by streaming an observed prefix.

        The fold is still day-bucketed, so each rewrite sees only the bounded
        current document plus one batch of new events; replaying a long prefix
        never produces a long context.
        """

        self.reset()
        results: list[MemoryUpdateResult] = []
        for event in events:
            results.extend(self.observe((event,)))
            if any(not result.success for result in results):
                return results
        results.extend(self.flush_all(force=True))
        return results

    def _reload_from_disk(self) -> None:
        with _memory_lock(self.memory_path.parent):
            _recover_interrupted_commit(
                self.memory_path, self.state_path, user_id=self.user_id
            )
            memory = self.memory_path.read_text(encoding="utf-8")
            self._memory_md = validate_memory_document(memory)
            self._check_backend_identity(self._memory_md)
            self._state = _read_state(self.state_path, self.user_id)

    def _check_backend_identity(self, document):
        if memory_backend.is_backend(document):
            payload = memory_backend.MemoryStore.payload(document)
            if not self.backend_enabled or self.program is None:
                raise ValueError("programmable memory requires an enabled writer")
            if payload["user_id"] != self.user_id or payload["writer_sha256"] != self.program.fingerprint.hook_sha256["remember"]:
                raise ValueError("memory owner or writer changed; rebuild from the causal prefix")

    def _sync_recovered_commit(self) -> None:
        previous_cursor = self._state.last_committed_cursor
        self._reload_from_disk()
        current_cursor = self._state.last_committed_cursor
        if current_cursor is None or current_cursor == previous_cursor:
            return
        remaining: list[_DayBucket] = []
        self._pending_cursors.clear()
        for bucket in self._buckets:
            bucket.events = [event for event in bucket.events if event.cursor > current_cursor]
            if bucket.events:
                remaining.append(bucket)
                self._pending_cursors.update(event.cursor for event in bucket.events)
        self._buckets = remaining

    def _pending_high_water(self) -> EventCursor | None:
        if not self._buckets:
            return None
        return self._buckets[-1].events[-1].cursor

    def _flush_ready(self, *, force: bool) -> list[MemoryUpdateResult]:
        self._sync_recovered_commit()
        if force and self._buckets:
            self._buckets[-1].closed = True
        results: list[MemoryUpdateResult] = []
        while self._buckets:
            closed = 0
            for bucket in self._buckets:
                if not bucket.closed:
                    break
                closed += 1
            if closed >= self.rewrite_days:
                count = self.rewrite_days
            elif force and closed:
                count = closed
            else:
                break
            batch = self._buckets[:count]
            result = self._rewrite(batch)
            self.stats.add(result)
            results.append(result)
            if not result.success:
                break
            removed = self._buckets[:count]
            self._buckets = self._buckets[count:]
            for bucket in removed:
                for event in bucket.events:
                    self._pending_cursors.discard(event.cursor)
        return results

    def _rewrite(self, buckets: Sequence[_DayBucket]) -> MemoryUpdateResult:
        days = tuple(bucket.day for bucket in buckets)
        events = [event for bucket in buckets for event in bucket.events]
        if not events:
            return MemoryUpdateResult(
                success=True,
                logical_days=days,
                committed_cursor=self._state.last_committed_cursor,
            )
        plan = compute_memory_plan(
            self.program,
            current_memory=self._memory_md,
            logical_days=days,
            events=events,
            backend_enabled=self.backend_enabled,
        )
        # The program chooses what is worth remembering; the engine still
        # commits the cursor over the WHOLE batch, so a dropped event can
        # never be re-observed and can never stall the stream.
        selected_events = [events[index] for index in plan.kept_indices]
        commit_cursor = events[-1].cursor
        use_backend = plan.backend is not None or memory_backend.is_backend(self._memory_md)
        if use_backend and not self.backend_enabled:
            return MemoryUpdateResult(success=False, logical_days=days, committed_cursor=self._state.last_committed_cursor,
                error="programmable_memory_not_enabled", errors=1, plan=plan)
        use_structured = plan.state_schema is not None or plan.state_records is not None or structured_memory.is_structured(self._memory_md)
        structured_capacity = {} if use_structured else None
        if not selected_events and not use_structured and not use_backend:
            # Nothing survived the plan. Advance state without a model call.
            state = MemoryState(
                user_id=self.user_id,
                last_committed_cursor=commit_cursor,
                last_rewrite_day=days[-1],
            )
            try:
                with _memory_lock(self.memory_path.parent):
                    _recover_interrupted_commit(
                        self.memory_path, self.state_path, user_id=self.user_id
                    )
                    _atomic_replace_memory_then_state(
                        self.memory_path, self.state_path, self._memory_md, state
                    )
            except Exception as exc:
                return MemoryUpdateResult(
                    success=False,
                    logical_days=days,
                    committed_cursor=self._state.last_committed_cursor,
                    error=_safe_error(exc, "persistence"),
                    errors=1,
                    plan=plan,
                )
            self._state = state
            return MemoryUpdateResult(
                success=True,
                logical_days=days,
                committed_cursor=state.last_committed_cursor,
                plan=plan,
            )
        user_message = build_maintenance_user_message(
            self._memory_md, days, selected_events
        )
        system_prompt = maintenance_prompt_for_plan(self.maintenance_prompt, plan)
        accounting = _CallAccounting()
        last_error = "invalid memory output"
        repaired = False
        memory: str | None = None
        backend_report, backend_execution = None, []
        if use_backend:
            def record_backend(kind, **details):
                backend_execution.append({"kind": kind, **details})

            def backend_model(messages, **options):
                accounting.calls += 1
                kwargs = {"model": self.model, "seed": self.seed, "max_tokens": MEMORY_COMPLETION_TOKENS,
                    "reasoning_effort": "none", **options}
                record_backend("memory_model_request", messages=messages, kwargs=kwargs)
                response = self.llm.chat(messages, **kwargs)
                accounting.add_response(response)
                record_backend("memory_model_response", response=response)
                return response

            try:
                store, context = memory_backend.prepare_write(self._memory_md, events, days[-1], self.user_id,
                    self.program.fingerprint.hook_sha256["remember"], kept_indices=plan.kept_indices)
                context.update(logical_days=list(days), events=[{"index": index, **memory_event_view(events[index])}
                    for index in plan.kept_indices])

                def invoke_backend(values):
                    record_backend("hook_input", hook="remember", context=values)
                    outcome = self.program.call("remember", values)
                    record_backend("hook_output", hook="remember", patch=outcome.patch, report=outcome.report)
                    return outcome.patch

                outcome = memory_backend.execute(store, phase="write",
                    initial={"memory_backend": plan.backend or {"operation": "finish"}}, invoke=invoke_backend,
                    context=context, model_call=backend_model, record=record_backend)
                memory = validate_memory_document(outcome["document"])
                backend_report = outcome["report"]
                plan = replace(plan, effect_digest=audit_token(backend_report),
                    applied_keys=tuple(sorted(set(plan.applied_keys) | {"memory_backend"})))
            except Exception as error:
                record_backend("memory_backend_error", error_type=type(error).__name__)
                return MemoryUpdateResult(success=False, logical_days=days, committed_cursor=self._state.last_committed_cursor,
                    error=_safe_error(error, "programmable backend"), calls=accounting.calls,
                    cache_hits=accounting.cache_hits, prompt_tokens=accounting.prompt_tokens,
                    completion_tokens=accounting.completion_tokens, reasoning_tokens=accounting.reasoning_tokens,
                    errors=accounting.errors + 1, plan=plan, backend_execution=backend_execution)
        elif use_structured:
            state_context = structured_memory.prepare_context(self._memory_md, events, days[-1])
            state_context["state_events"] = [row for row in state_context["state_events"] if row["index"] in plan.kept_indices]
            user_message = json.dumps(state_context, ensure_ascii=False)
            system_prompt = (
                "Maintain bounded state for one user using only the supplied causal evidence. "
                "Return a JSON object with one records array. Each record has key, data, source_refs "
                "and optional expires_on (ISO date). Each key must be unique, contain 1 to 64 ASCII "
                "characters, start with a letter, and use only letters, digits, underscores, dots or hyphens. "
                "Use short stable identifiers rather than full preference sentences as keys. "
                "source_refs must reference event:N or state:KEY "
                "from the input. Return the full replacement state; omission deletes a record. "
                f"Keep at most {structured_memory.MAX_RECORDS} records, each below {structured_memory.MAX_RECORD_CHARS} "
                f"serialized characters including its source receipts, and the full document below {structured_memory.MAX_DOCUMENT_CHARS} characters. "
                f"A record may inherit at most {structured_memory.MAX_SOURCES} distinct source receipts; avoid merging unrelated records. "
                "Use the exact state:KEY reference to retain an existing record. Consolidate redundant records "
                "and omit stale or weak information before exceeding capacity. "
                "Order records by retention priority, highest first. The host validates all records, then "
                "omits records whose complete source receipts, record size or total document size exceed capacity; "
                "it does not truncate source receipts. Put explicit current exclusions and durable supported facts first. "
                "Do not turn unknown behavior into positive or negative preferences. "
                "Preserve context, conflicts and uncertainty where the evidence requires them. "
                "Do not reproduce raw event logs. Data for every record must match this schema: "
                + json.dumps(plan.state_schema or {}, ensure_ascii=False)
                + "\nUser-specific update guidance: " + plan.guidance
            )
            direct_records = plan.state_records
            if direct_records is None and not state_context["state_events"]:
                direct_records = [{"key": row["key"], "data": row["data"],
                    "source_refs": [row["ref"]], "expires_on": row["expires_on"]}
                    for row in state_context["state_records"]]
            if direct_records is not None:
                try:
                    memory = structured_memory.compile_records(direct_records,
                        document=self._memory_md, events=events, day=days[-1],
                        schema=plan.state_schema, kept_indices=plan.kept_indices,
                        overflow_policy="priority", capacity_report=structured_capacity)
                    memory = validate_memory_document(memory, raw_events=events)
                except Exception:
                    return MemoryUpdateResult(success=False, logical_days=days,
                        committed_cursor=self._state.last_committed_cursor,
                        error="MemoryDocumentError: invalid structured memory proposal", errors=1, plan=plan)
        # A program-chosen budget is a request, not a contract: the engine's
        # own bounds are the safety property. The audit reports whether the
        # budget was honoured; overshooting it never fails the rewrite.
        budget_applied = plan.max_chars is None
        for attempt in range(0 if memory is not None else 2):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_message + (
                        (
                            ("\nReturn a complete valid JSON state object with records only." if use_structured else self.repair_suffix)
                            + f"\nPrevious attempt failed: {last_error}."
                            + (
                                "\nRebuild the complete resulting memory within 2500 output tokens. "
                                "Compress the entire document, merging redundant facts while preserving "
                                "supported preferences, explicit negative constraints and active goals. "
                                "Do not append a continuation or copy a chronological activity log."
                                if "length-truncated" in last_error or "exhausts or exceeds" in last_error else ""
                            )
                            + (
                                "\nRewrite activity bullets into concise lasting facts or preferences. "
                                "Do not start bullets with watched, clicked, asked, said, posted, "
                                "or other past-tense activity verbs; merge redundant activities."
                                if "per-activity bullet log" in last_error else ""
                            )
                            + (
                                f"\nAim for at most {plan.max_chars} characters."
                                if plan.max_chars is not None else ""
                            )
                        )
                        if attempt else ""
                    ),
                },
            ]
            kwargs: dict[str, Any] = {
                "max_tokens": MEMORY_COMPLETION_TOKENS,
                "reasoning_effort": "none",
                "seed": self.seed + attempt,
            }
            if self.model is not None:
                kwargs["model"] = self.model
            if use_structured:
                kwargs["response_format"] = structured_memory.generation_response_format(plan.state_schema, state_context)
            accounting.calls += 1
            try:
                response = self.llm.chat(messages, **kwargs)
                if not isinstance(response, Mapping):
                    raise ValueError("memory updater returned a non-object response")
                accounting.add_response(response)
                if response.get("_memory_render_error"):
                    raise MemoryDocumentError(str(response["_memory_render_error"]))
                usage = response.get("usage")
                completion_tokens: int | None = None
                if isinstance(usage, Mapping):
                    completion_tokens = _integer(
                        usage.get("completion") or usage.get("completion_tokens")
                    ) or None
                # The activity date is engine-owned. Accept the model's common
                # bare heading and normalize it before validation.
                if use_structured:
                    try:
                        payload = json.loads(str(response.get("content") or ""))
                        if not isinstance(payload, dict) or set(payload) != {"records"}:
                            raise ValueError("invalid state envelope")
                        draft = structured_memory.compile_records(payload["records"],
                            document=self._memory_md, events=events, day=days[-1],
                            schema=plan.state_schema, kept_indices=plan.kept_indices,
                            overflow_policy="priority", capacity_report=structured_capacity)
                    except ValueError as exc:
                        if str(exc) == "invalid or duplicate structured memory key":
                            raise MemoryDocumentError("invalid or duplicate structured memory key; use unique short ASCII identifiers starting with a letter") from None
                        raise MemoryDocumentError("invalid structured memory proposal") from None
                    except Exception:
                        raise MemoryDocumentError("invalid structured memory proposal") from None
                else:
                    draft = correct_last_activity_header(
                        str(response.get("content") or ""), days[-1]
                    )
                if memory_backend.is_backend(draft):
                    raise MemoryDocumentError("model cannot author a programmable memory envelope")
                memory = validate_memory_document(
                    draft,
                    finish_reason=response.get("finish_reason"),
                    completion_tokens=completion_tokens,
                    raw_events=events,
                )
                if plan.max_chars is not None:
                    budget_applied = len(memory) <= plan.max_chars
                break
            except Exception as exc:  # one fixed repair attempt covers call/validation
                accounting.errors += 1
                last_error = _safe_error(exc, "call or validation")
                if isinstance(exc, (MemoryServiceRefusal, MemoryIntegrityError)):
                    break
                repaired = True
        if memory is not None:
            if not budget_applied and "max_chars" in plan.applied_keys:
                # Report what the run actually got. Leaving the key in
                # applied_keys would credit the program with a bound the
                # document did not respect.
                plan = replace(
                    plan,
                    applied_keys=tuple(
                        key for key in plan.applied_keys if key != "max_chars"
                    ),
                )
            state = MemoryState(
                user_id=self.user_id,
                last_committed_cursor=commit_cursor,
                last_rewrite_day=days[-1],
            )
            try:
                with _memory_lock(self.memory_path.parent):
                    _recover_interrupted_commit(
                        self.memory_path, self.state_path, user_id=self.user_id
                    )
                    disk_state = _read_state(self.state_path, self.user_id)
                    if (
                        disk_state.last_committed_cursor is not None
                        and disk_state.last_committed_cursor
                        > commit_cursor
                    ):
                        raise RuntimeError("stale concurrent rewrite")
                    if disk_state.last_committed_cursor == commit_cursor:
                        # Another process completed this exact causal prefix.
                        # Keep its full document rather than racing a second
                        # nondeterministic rewrite over the same event.
                        memory = validate_memory_document(
                            self.memory_path.read_text(encoding="utf-8")
                        )
                        state = disk_state
                    else:
                        _atomic_replace_memory_then_state(
                            self.memory_path, self.state_path, memory, state
                        )
            except Exception as exc:
                accounting.errors += 1
                return MemoryUpdateResult(
                    success=False,
                    logical_days=days,
                    committed_cursor=self._state.last_committed_cursor,
                    error=_safe_error(exc, "persistence"),
                    calls=accounting.calls,
                    cache_hits=accounting.cache_hits,
                    prompt_tokens=accounting.prompt_tokens,
                    completion_tokens=accounting.completion_tokens,
                    reasoning_tokens=accounting.reasoning_tokens,
                    errors=accounting.errors,
                    repaired=repaired,
                    plan=plan,
                    backend_report=backend_report,
                    backend_execution=backend_execution,
                )
            self._memory_md = memory
            self._state = state
            return MemoryUpdateResult(
                success=True,
                logical_days=days,
                committed_cursor=state.last_committed_cursor,
                calls=accounting.calls,
                cache_hits=accounting.cache_hits,
                prompt_tokens=accounting.prompt_tokens,
                completion_tokens=accounting.completion_tokens,
                reasoning_tokens=accounting.reasoning_tokens,
                errors=accounting.errors,
                repaired=repaired,
                plan=plan,
                structured_capacity=structured_capacity,
                backend_report=backend_report,
                backend_execution=backend_execution,
            )
        return MemoryUpdateResult(
            success=False,
            logical_days=days,
            committed_cursor=self._state.last_committed_cursor,
            error=last_error,
            calls=accounting.calls,
            cache_hits=accounting.cache_hits,
            prompt_tokens=accounting.prompt_tokens,
            completion_tokens=accounting.completion_tokens,
            reasoning_tokens=accounting.reasoning_tokens,
            errors=accounting.errors,
            repaired=repaired,
            plan=plan,
        )


__all__ = [
    "DailyMemoryManager",
    "DEFAULT_MAINTENANCE_POLICY_SHA256",
    "INITIAL_MEMORY_MD",
    "MAINTENANCE_SYSTEM_PROMPT",
    "MEMORY_COMPLETION_TOKENS",
    "MEMORY_PROTOCOL",
    "MEMORY_STATE_SCHEMA_VERSION",
    "MemorySnapshot",
    "MemoryState",
    "MemoryUpdateResult",
    "MemoryUpdateStats",
    "MemoryWorkspace",
    "REPAIR_SUFFIX",
    "build_maintenance_user_message",
    "correct_last_activity_header",
    "estimate_token_count",
    "event_logical_day",
    "initialize_memory_files",
    "last_activity_header",
    "memory_sha256",
    "normalize_logical_day",
    "render_event_batch",
    "validate_memory_document",
]
