"""Causal, evaluator-independent projection of observed user history.

This module is infrastructure, not a training objective.  It reads only
``Episode.history_delta`` plus structural episode identity/order and exposes a
typed, chronological event view used by PM3 proxy construction and daily
memory.  Native instructions, checkpoints, gold actions, targets, environments
and ``Episode.meta`` are deliberately outside its information boundary.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ohpu.ir import Episode, HistoryItem

Producer = Literal["user", "assistant", "tool", "environment", "unknown"]
Channel = Literal["text", "review", "reaction", "action"]
TargetKind = Literal["text", "categorical"]

_TEXT_CHANNELS = frozenset({"text", "review"})
_UNTRUSTED_ORIGIN = re.compile(
    r"oracle|annotat|pseudo|simulat|ground.?truth|answer.?key|gold|judge|"
    r"reward|trace|profile|preferences?|reveals?|checkpoints?|meta(?:data)?",
    re.I,
)
_PROJECTED_CURSOR = "_ohpu_source_event_cursor"


@dataclass(frozen=True, order=True)
class EventCursor:
    """A total causal order independent of timestamp string formatting."""

    episode_order: int
    episode_rank: int
    item_rank: int
    event_rank: int
    timestamp: str = field(default="", compare=False)

    @classmethod
    def after_all(cls) -> "EventCursor":
        return cls(2**63 - 1, 2**63 - 1, 2**63 - 1, 2**63 - 1)


@dataclass(frozen=True)
class EventProvenance:
    """Lineage retained from the observed-history record, never rendered."""

    source_episode_id: str
    source_item_id: str
    source_kind: str
    declared_origin: str
    declared_producer: str

    @property
    def is_untrusted(self) -> bool:
        return bool(
            _UNTRUSTED_ORIGIN.search(self.declared_origin)
            or _UNTRUSTED_ORIGIN.search(self.declared_producer)
        )


@dataclass(frozen=True)
class ObservedEvent:
    event_id: str
    user_id: str
    source_group: str
    cursor: EventCursor
    producer: Producer
    channel: Channel
    value: str
    context: str
    provenance: EventProvenance
    # Context-only exposure (for example PM3 ``feed_visible``) may be supplied
    # to later tasks, but is not evidence that the user liked/disliked it and
    # must never become preference evidence or a persona-memory update.
    is_context_only: bool = False
    # Dataset-native, already-sanitized history fields that become visible only
    # after this event's cursor. They never enter the event's public query.
    history_payload: Any = field(default=None, compare=False, repr=False)

    @property
    def target_kind(self) -> TargetKind:
        return "text" if self.channel in _TEXT_CHANNELS else "categorical"

    @property
    def is_strict_user_evidence(self) -> bool:
        return (
            self.producer == "user"
            and not self.is_context_only
            and not self.provenance.is_untrusted
        )


def _clean(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _declared(record: Any, parent: Any, keys: Sequence[str]) -> str:
    for candidate in (record, parent):
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            value = candidate.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, dict):
                for nested in ("origin", "producer", "source", "type", "kind"):
                    if value.get(nested) not in (None, ""):
                        return _clean(value[nested], 160)
                return _clean(json.dumps(value, ensure_ascii=False, sort_keys=True), 160)
            return _clean(value, 160)
    return ""


def _origin(record: Any, parent: Any, producer: Producer) -> tuple[str, str]:
    declared_producer = _declared(
        record, parent, ("producer", "actor", "author", "source_role", "role")
    )
    declared_origin = _declared(
        record, parent,
        ("provenance", "source_origin", "target_origin", "origin", "generated_by"),
    )
    if not declared_origin:
        declared_origin = {
            "user": "observed_user",
            "assistant": "observed_assistant_unverified",
            "tool": "observed_tool",
            "environment": "observed_environment",
            "unknown": "observed_unknown",
        }[producer]
    return declared_origin, declared_producer


def _provenance(
    episode: Episode,
    item: HistoryItem,
    record: Any,
    producer: Producer,
) -> EventProvenance:
    origin, declared_producer = _origin(record, item.content, producer)
    return EventProvenance(
        source_episode_id=episode.episode_id,
        source_item_id=item.item_id,
        source_kind=item.kind,
        declared_origin=origin,
        declared_producer=declared_producer,
    )
