"""Canonical intermediate representation for per-user harness learning.

Every dataset is normalized into this shape so the adaptation/eval pipeline never
touches a dataset-specific field. The load-bearing design decision is that the
unit of evaluation is a **Checkpoint**, i.e. an (episode, preference) pair --
not an episode. See proposal section 7.2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

Dimension = Literal["context", "tools", "generation", "orchestration", "memory", "output"]
Split = Literal["adaptation", "validation", "test"]
Decidable = Literal["program", "judge"]


@dataclass
class Preference:
    """One oracle preference belonging to a user.

    `explicit` is the single most important flag in this schema: the headline
    result of the project is restricted to preferences that were never stated
    outright, only revealed behaviorally (proposal section 6.7b, defense 3).
    """
    pref_id: str
    user_id: str
    text: str
    category: str = ""            # dataset-native grouping, e.g. 饮食偏好
    dimension: Optional[Dimension] = None   # which harness slot it should adapt
    explicit: Optional[bool] = None         # stated outright in history vs only revealed
    first_seen_order: int = 0     # episode order index at which it enters the oracle
    last_status: str = "create"   # create | update | delete | unchanged
    status_history: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)  # HistoryItem ids


@dataclass
class HistoryItem:
    """One piece of user history revealed before some episode."""
    item_id: str
    user_id: str
    order: int                    # which episode boundary revealed it
    date: str = ""
    kind: str = "chat"            # chat | behavior | dialogue | noise | session
    content: Any = None
    is_noise: bool = False
    reveals: list[str] = field(default_factory=list)   # pref_ids, when known


@dataclass
class Checkpoint:
    """THE evaluation unit: does this trajectory satisfy this preference here?

    `pref_id` is what makes section 7.2 computable. A checkpoint with no
    pref_id is a task-correctness criterion, not a personalization one, and is
    scored separately as a guardrail.
    """
    cp_id: str
    episode_id: str
    user_id: str
    text: str
    pref_id: Optional[str] = None
    decidable: Decidable = "judge"
    predicate: Optional[dict[str, Any]] = None   # machine-checkable form, when derivable
    applicable: bool = True       # False => no opportunity to violate; excluded from denominator
    source: str = ""              # which raw field it came from


@dataclass
class Episode:
    """One task the agent must execute for this user."""
    episode_id: str
    user_id: str
    order: int                    # position in the user's timeline; the T knob
    split: Optional[Split] = None
    timestamp: str = ""
    domain: str = ""
    instruction: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    message_history: list[dict[str, Any]] = field(default_factory=list)
    gold_actions: list[dict[str, Any]] = field(default_factory=list)
    gold_targets: list[str] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    history_delta: list[HistoryItem] = field(default_factory=list)  # revealed BEFORE this ep
    skills_tested: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    user_id: str
    dataset: str
    profile: dict[str, Any] = field(default_factory=dict)
    preferences: list[Preference] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)   # time-ordered

    # -- convenience ------------------------------------------------------
    def history_upto(self, order: int) -> list[HistoryItem]:
        """D_u^{<=T}: every history item revealed strictly before episode `order`."""
        out: list[HistoryItem] = []
        for ep in self.episodes:
            if ep.order <= order:
                out.extend(ep.history_delta)
        return out

    def prefs_known_at(self, order: int) -> list[Preference]:
        """Oracle preferences that exist as of episode `order` (drift-aware)."""
        return [p for p in self.preferences
                if p.first_seen_order <= order and p.last_status != "delete"]


@dataclass
class Corpus:
    dataset: str
    users: list[User] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for u in self.users:
                f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")

    @staticmethod
    def stats(users: list[User]) -> dict[str, Any]:
        import statistics as st
        eps = [len(u.episodes) for u in users]
        cps = [len(e.checkpoints) for u in users for e in u.episodes]
        linked = sum(1 for u in users for e in u.episodes
                     for c in e.checkpoints if c.pref_id)
        total_cp = sum(cps) or 1
        return {
            "users": len(users),
            "episodes": sum(eps),
            "episodes_per_user_min": min(eps) if eps else 0,
            "episodes_per_user_max": max(eps) if eps else 0,
            "episodes_per_user_mean": round(st.mean(eps), 2) if eps else 0,
            "preferences": sum(len(u.preferences) for u in users),
            "checkpoints": sum(cps),
            "checkpoints_linked_to_pref": linked,
            "link_rate": round(linked / total_cp, 3),
            "history_items": sum(len(e.history_delta) for u in users for e in u.episodes),
        }
