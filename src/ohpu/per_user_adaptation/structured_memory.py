"""Bounded user-defined state with host-owned causal source references."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any, Mapping, Sequence

from ohpu.experiments.observed_history import ObservedEvent
from .preference_learning import BOUNDARY, public_observation

HEADING = "## Harness state (structured-v1)"
PREFIX = "- [state] "
MAX_RECORDS = 32
MAX_DOCUMENT_CHARS = 12_000
MAX_RECORD_CHARS = 2_000
MAX_SOURCES = 16
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def is_structured(document: str) -> bool:
    return HEADING in document.splitlines()


def read_records(document: str) -> list[dict[str, Any]]:
    if not is_structured(document):
        return []
    records = []
    seen = set()
    for line in document.splitlines()[1:]:
        if not line.strip() or line == HEADING:
            continue
        if not line.startswith(PREFIX):
            raise ValueError("structured memory contains an unrecognized row")
        record = json.loads(line[len(PREFIX):])
        if not isinstance(record, dict) or set(record) != {"key", "data", "sources", "updated_day", "expires_on"}:
            raise ValueError("invalid structured memory record")
        key = record["key"]
        if not isinstance(key, str) or not _KEY.fullmatch(key) or key in seen:
            raise ValueError("invalid or duplicate structured memory key")
        seen.add(key)
        sources = record["sources"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES or any(
            not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{64}", source) for source in sources
        ):
            raise ValueError("structured memory record has no valid source receipts")
        if not isinstance(record["updated_day"], str):
            raise ValueError("invalid structured memory update day")
        expiry = record["expires_on"]
        if expiry is not None and (not isinstance(expiry, str) or not _DAY.fullmatch(expiry)):
            raise ValueError("invalid structured memory expiry")
        if expiry is not None:
            date.fromisoformat(expiry)
        if len(_json(record)) > MAX_RECORD_CHARS:
            raise ValueError("structured memory record exceeds size limit")
        records.append(record)
    if len(records) > MAX_RECORDS or len(document) > MAX_DOCUMENT_CHARS:
        raise ValueError("structured memory exceeds capacity")
    return records


def _render(records: Sequence[Mapping[str, Any]], header: str) -> str:
    return header + "\n\n" + HEADING + "\n" + "".join(PREFIX + _json(record) + "\n" for record in records)


def prune_expired(document: str, day: str | None) -> str:
    if not is_structured(document) or not day or not _DAY.fullmatch(day):
        return document
    records = read_records(document)
    kept = [record for record in records if not record["expires_on"] or record["expires_on"] > day]
    return document if len(kept) == len(records) else _render(kept, document.splitlines()[0])


def prepare_context(document: str, events: Sequence[ObservedEvent], day: str) -> dict[str, Any]:
    if events and (len({event.user_id for event in events}) != 1 or any(
        not event.is_strict_user_evidence for event in events
    )):
        raise ValueError("structured memory requires one user's verified events")
    if any(left.cursor >= right.cursor for left, right in zip(events, events[1:])):
        raise ValueError("structured memory events must be strictly causal")
    previous = read_records(prune_expired(document, day))
    projected = []
    forget_at = -1
    for index, event in enumerate(events):
        public = public_observation(event)
        if public["action"] == "asked_to_forget" or any(
            message["role"] == "user" and BOUNDARY.search(message["content"])
            for message in public["messages"]
        ):
            forget_at = index
        projected.append({"ref": f"event:{index}", "index": index,
            "action": public["action"], "topics": public["topics"],
            "item": public["item"][:1200],
            "messages": [{"role": message["role"], "content": message["content"][:800]}
                         for message in public["messages"][-6:]]})
    if forget_at >= 0:
        previous = []
        projected = projected[forget_at + 1:]
    return {"state_records": [{"ref": "state:" + record["key"], **record} for record in previous],
            "state_events": projected, "forget_boundary_applied": forget_at >= 0}


def validate_schema(schema: Mapping[str, Any]):
    from jsonschema import Draft202012Validator

    if not isinstance(schema, dict) or len(_json(schema)) > 4_000:
        raise ValueError("state schema must be a bounded object")

    def check(value, depth=0):
        if depth > 10:
            raise ValueError("state schema is too deeply nested")
        if isinstance(value, dict):
            if any(key in value for key in ("$ref", "$dynamicRef", "pattern", "patternProperties")):
                raise ValueError("state schema cannot load references or run regular expressions")
            for child in value.values():
                check(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                check(child, depth + 1)

    check(schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def generation_response_format(schema: Mapping[str, Any] | None, context: Mapping[str, Any]) -> dict[str, Any]:
    def closed(value):
        if not isinstance(value, dict):
            return False
        allowed = {"type", "properties", "required", "additionalProperties", "items", "enum",
                   "description", "title", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"}
        if set(value) - allowed:
            return False
        kind = value.get("type")
        if kind == "object":
            properties = value.get("properties", {})
            return (value.get("additionalProperties") is False
                and set(value.get("required", [])) == set(properties)
                and all(closed(child) for child in properties.values()))
        if kind == "array":
            return closed(value.get("items"))
        return kind in ("string", "number", "integer", "boolean", "null")

    if not closed(schema):
        return {"type": "json_object"}
    references = [record["ref"] for name in ("state_records", "state_events") for record in context[name]]
    if not references:
        return {"type": "json_object"}
    record_schema = {"type": "object", "properties": {
        "key": {"type": "string", "pattern": "^" + _KEY.pattern + "$"},
        "data": schema,
        "source_refs": {"type": "array", "items": {"type": "string", "enum": references},
            "minItems": 1, "maxItems": MAX_SOURCES},
        "expires_on": {"anyOf": [{"type": "string", "pattern": "^" + _DAY.pattern + "$"}, {"type": "null"}]},
    }, "required": ["key", "data", "source_refs", "expires_on"], "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": "causal_user_memory", "strict": True,
        "schema": {"type": "object", "properties": {"records": {
            "type": "array", "items": record_schema, "maxItems": MAX_RECORDS}},
            "required": ["records"], "additionalProperties": False}}}


def compile_records(
    proposal: Sequence[Mapping[str, Any]], *, document: str,
    events: Sequence[ObservedEvent], day: str, schema: Mapping[str, Any] | None = None,
    kept_indices: Sequence[int] | None = None,
    overflow_policy: str = "reject", capacity_report: dict[str, Any] | None = None,
) -> str:
    if overflow_policy not in {"reject", "priority"}:
        raise ValueError("unknown structured memory capacity policy")
    if not isinstance(proposal, list) or len(proposal) > MAX_RECORDS:
        raise ValueError("structured memory proposal must be a bounded record list")
    context = prepare_context(document, events, day)
    sources = {record["ref"]: record["sources"] for record in context["state_records"]}
    kept = set(range(len(events))) if kept_indices is None else set(kept_indices)
    for record in context["state_events"]:
        index = record["index"]
        if index in kept:
            event = events[index]
            sources[record["ref"]] = [hashlib.sha256(
                _json([event.user_id, event.event_id, event.source_group, event.cursor.timestamp]).encode()
            ).hexdigest()]
    validator = None
    if schema is not None:
        validator = validate_schema(schema)
    records = []
    keys = set()
    for candidate in proposal:
        if not isinstance(candidate, Mapping) or not {"key", "data", "source_refs"} <= set(candidate) or set(candidate) - {"key", "data", "source_refs", "expires_on"}:
            raise ValueError("structured memory proposal has invalid fields")
        refs = candidate["source_refs"]
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in sources for ref in refs):
            raise ValueError("structured memory references unavailable evidence")
        if validator is not None:
            validator.validate(candidate["data"])
        key = candidate["key"]
        if not isinstance(key, str) or not _KEY.fullmatch(key) or key in keys:
            raise ValueError("invalid or duplicate structured memory key")
        keys.add(key)
        expiry = candidate.get("expires_on")
        if expiry is not None:
            if not isinstance(expiry, str) or not _DAY.fullmatch(expiry):
                raise ValueError("invalid structured memory expiry")
            date.fromisoformat(expiry)
        receipts = sorted({receipt for ref in refs for receipt in sources[ref]})
        record = {"key": candidate["key"], "data": candidate["data"], "sources": receipts,
                  "updated_day": day, "expires_on": candidate.get("expires_on")}
        records.append(record)
    header = f"# USER MEMORY (last activity seen: {day})"
    if overflow_policy == "priority":
        retained = []
        counts = {"policy": "priority-v1", "proposed_records": len(records), "retained_records": 0,
            "expired_records": 0, "omitted_source_capacity": 0,
            "omitted_record_capacity": 0, "omitted_document_capacity": 0}
        for record in records:
            if record["expires_on"] and record["expires_on"] <= day:
                counts["expired_records"] += 1
            elif len(record["sources"]) > MAX_SOURCES:
                counts["omitted_source_capacity"] += 1
            elif len(_json(record)) > MAX_RECORD_CHARS:
                counts["omitted_record_capacity"] += 1
            elif len(_render([*retained, record], header)) > MAX_DOCUMENT_CHARS:
                counts["omitted_document_capacity"] += 1
            else:
                retained.append(record)
        counts["retained_records"] = len(retained)
        records = retained
        if capacity_report is not None:
            capacity_report.clear()
            capacity_report.update(counts)
    rendered = _render(records, header)
    read_records(rendered)
    return prune_expired(rendered, day)


def user_state(document: str) -> dict[str, Any]:
    return {record["key"]: record["data"] for record in read_records(document)}