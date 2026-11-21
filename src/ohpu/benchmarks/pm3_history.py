"""PersonaMem-v3's official agent-visible history projection.

The upstream evaluator exposes time-masked per-app events while withholding
``profile.json`` and scoring-only fields.  Its long-context and textual-memory
modes then compact those safe events before model ingestion.  OHPU uses the
same two-stage contract: the loader stores a physically sanitized event and
runtime retrieval/memory render the upstream-compatible compact view.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

HISTORY_CONTRACT = "pm3-official-agent-history-v1"
CONTRACT_KEY = "_ohpu_history_contract"
VISIBLE_EVENT_KEY = "_ohpu_agent_visible_event"

# Mirrored from the official ``evaluation/backend_query.py`` firewall.
_LEAK_EVENT_FIELDS = frozenset({"split"})
_LEAK_PREFERENCE_FIELDS = frozenset({
    "split",
    "over_personalization_irrelevant",
    "update_history",
    "confidence_score_init",
    "confidence_cross_referenced",
    "stereotype_mark",
    "hidden_persona_labels",
})


def sanitize_preference(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if str(key) not in _LEAK_PREFERENCE_FIELDS
    }


def sanitize_event(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the official leak-stripped event snapshot representation."""
    event = {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if str(key) not in _LEAK_EVENT_FIELDS
    }
    preferences = event.get("preferences")
    if isinstance(preferences, list):
        event["preferences"] = [
            safe
            for raw in preferences
            if (safe := sanitize_preference(raw)) is not None
        ]
    return event


def _content_fields(event: Mapping[str, Any]) -> dict[str, Any]:
    content = event.get("content")
    return dict(content) if isinstance(content, Mapping) else {}


def public_stimulus(event: Mapping[str, Any]) -> str:
    """Render the current non-label query used to predict an event reaction.

    The observed action, interaction polarity, inferred preference annotations,
    and every scoring field are deliberately absent.  Those become historical
    evidence only after the event has occurred.
    """
    interaction = event.get("interaction_format")
    interaction = dict(interaction) if isinstance(interaction, Mapping) else {}
    content = _content_fields(event)
    conversation = event.get("conversation")
    public_conversation: list[dict[str, Any]] = []
    if isinstance(conversation, list):
        public_conversation = [
            {
                "role": str(message.get("role") or ""),
                "content": _clip(message.get("content"), 300),
            }
            for message in conversation[-6:]
            if isinstance(message, Mapping)
            and message.get("role") == "user"
            and message.get("content")
        ]
    payload: dict[str, Any] = {
        "app": interaction.get("app"),
        "user_message": interaction.get("user_message"),
        "hashtags": event.get("source_hashtags") or [],
        "content_type": event.get("content_type"),
        "title": content.get("title"),
        "caption": content.get("caption"),
        "text": content.get("text") or content.get("body"),
        "description": content.get("overall_description"),
        # Current-event generation annotations remain hidden. Only role and
        # text are part of the public stimulus; ``embeds_pref_idx`` becomes
        # visible later when the completed event enters ordinary history.
        "conversation": public_conversation,
    }
    clean = {key: item for key, item in payload.items() if item not in (None, "", [])}
    return json.dumps(clean, ensure_ascii=False, sort_keys=True) if clean else ""


def wrap_event(
    value: Mapping[str, Any],
    *,
    sanitize: bool = True,
) -> dict[str, Any]:
    """Store one safe PM3 history event plus its causal visibility envelope."""
    safe = sanitize_event(value) if sanitize else copy.deepcopy(dict(value))
    interaction_type = str(safe.get("source_interaction_type") or "").strip()
    if not interaction_type:
        raise ValueError("PM3 history event has no source_interaction_type")
    interaction = safe.get("interaction_format")
    app = str(interaction.get("app") or "") if isinstance(interaction, Mapping) else ""
    object_id = str(safe.get("source_object_id") or "")
    return {
        CONTRACT_KEY: HISTORY_CONTRACT,
        VISIBLE_EVENT_KEY: safe,
        "observed_event": {
            "visibility": "observed",
            "producer": "user",
            "channel": "reaction",
            "value": interaction_type,
            "context": public_stimulus(safe),
            "provenance": "observed_user",
            "session_id": f"{app}:{object_id}" if app or object_id else "",
        },
    }


def unwrap_event(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get(CONTRACT_KEY) != HISTORY_CONTRACT:
        return None
    event = value.get(VISIBLE_EVENT_KEY)
    return dict(event) if isinstance(event, Mapping) else None


def compact_event(value: Mapping[str, Any], *, app: str = "") -> dict[str, Any]:
    """Mirror the official long-context/textual-memory event projection."""
    event = sanitize_event(value)
    interaction = event.get("interaction_format")
    interaction = dict(interaction) if isinstance(interaction, Mapping) else {}
    content = _content_fields(event)
    compact: dict[str, Any] = {
        "app": app or interaction.get("app"),
        "oid": event.get("source_object_id"),
        "t": event.get("source_timestamp"),
        "when": event.get("formatted_timestamp"),
        "action": interaction.get("action_label"),
        "user_message": interaction.get("user_message"),
        "type": event.get("source_interaction_type"),
        "hashtags": event.get("source_hashtags") or [],
        "content_type": event.get("content_type"),
        "author_id": event.get("author_id"),
        "is_self_authored": event.get("is_self_authored"),
        "is_dm": event.get("is_dm"),
        "title": content.get("title"),
        "caption": content.get("caption"),
        "desc": content.get("overall_description"),
        "conversation": event.get("conversation") or [],
    }
    if event.get("is_trending"):
        compact.update({
            "is_trending": True,
            "trending_topic": event.get("trending_topic"),
            "trending_relevance": event.get("trending_relevance"),
        })
    if event.get("is_ad"):
        compact["is_ad"] = True
    location = event.get("event_location")
    if isinstance(location, Mapping) and location.get("city"):
        compact["location"] = {
            str(key): copy.deepcopy(item)
            for key, item in location.items()
            if item is not None
        }
    preferences = event.get("preferences")
    if isinstance(preferences, list):
        compact["inferred_prefs"] = [
            {
                "persona_item": preference.get("persona_item"),
                "category": preference.get("category"),
            }
            for preference in preferences
            if isinstance(preference, Mapping)
            and (preference.get("persona_item") or preference.get("category"))
        ]
    return {
        key: item
        for key, item in compact.items()
        if item not in (None, "", [])
    }


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def bounded_compact_event(
    value: Mapping[str, Any],
    *,
    app: str = "",
) -> dict[str, Any]:
    """Compact PM3 history for bounded retrieval and memory-update prompts.

    This follows the official textual-memory renderer's six-turn and per-field
    clipping policy while retaining structured field names for OHPU.
    """
    compact = compact_event(value, app=app)
    for key, limit in (
        ("user_message", 200),
        ("title", 120),
        ("caption", 160),
        ("desc", 220),
    ):
        if compact.get(key):
            compact[key] = _clip(compact[key], limit)
    conversation = compact.get("conversation")
    if isinstance(conversation, list):
        compact["conversation"] = []
        for message in conversation[:6]:
            if not isinstance(message, Mapping) or not message.get("content"):
                continue
            public_message: dict[str, Any] = {
                "role": str(message.get("role") or ""),
                "content": _clip(message.get("content"), 120),
            }
            # The official long-context path copies the conversation object,
            # including this one-based alignment annotation. Preserve it in
            # the bounded representation instead of silently dropping it.
            indices = message.get("embeds_pref_idx")
            if isinstance(indices, list):
                public_message["embeds_pref_idx"] = copy.deepcopy(indices)
            compact["conversation"].append(public_message)
    preferences = compact.get("inferred_prefs")
    if isinstance(preferences, list):
        compact["inferred_prefs"] = [
            {
                "persona_item": _clip(preference.get("persona_item"), 80),
                "category": _clip(preference.get("category"), 80),
            }
            for preference in preferences[:4]
            if isinstance(preference, Mapping)
        ]
    return compact


def render_compact_event(value: Mapping[str, Any], *, app: str = "") -> str:
    return json.dumps(
        bounded_compact_event(value, app=app),
        ensure_ascii=False,
        sort_keys=True,
    )
