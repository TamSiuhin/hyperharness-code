"""Read the OPPU LaMP archive without exposing query labels as user history."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile


TASKS = {
    "citation": ("LaMP_1", ("id", "title", "abstract", "date", "citation")),
    "movie_tagging": ("LaMP_8", ("id", "description", "tag", "date")),
    "news_categorize": ("LaMP_2", ("id", "text", "title", "category", "date")),
    "product_rating": ("LaMP_3", ("id", "text", "score", "date")),
    "news_headline": ("LaMP_4", ("id", "text", "title", "date")),
    "scholarly_title": ("LaMP_5", ("id", "abstract", "title", "date")),
    "tweet_paraphrase": ("LaMP_7", ("id", "text", "date")),
}


@dataclass(frozen=True)
class LaMPQuery:
    query_id: str
    input: str
    target: str

    def public(self) -> dict[str, str]:
        return {"id": self.query_id, "input": self.input}


@dataclass(frozen=True)
class OPPUUser:
    task: str
    user_id: str
    history_json: str
    queries: tuple[LaMPQuery, ...]

    def history(self) -> list[dict[str, Any]]:
        return json.loads(self.history_json)


@dataclass(frozen=True)
class OPPUTask:
    name: str
    native_task: str
    users: tuple[OPPUUser, ...]
    source_sha256: dict[str, str]


def _identifier(value: Any, kind: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"invalid {kind} identifier")
    return str(value)


def _text(value: Any, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid {kind} text")
    return value


def load_oppu(archive_path: str | Path, task: str, *, include_others: bool = False) -> OPPUTask:
    if task not in TASKS:
        raise ValueError(f"unsupported OPPU task: {task}")
    native_task, history_fields = TASKS[task]
    prefix = "user_more_100_history" if task == "tweet_paraphrase" else "user_top_100_history"
    names = [f"data/{task}/{prefix}.json", f"data/{task}/{prefix}_label.json"]
    if include_others:
        names.append(f"data/{task}/user_others.json")
    with zipfile.ZipFile(archive_path) as archive:
        if any(archive.namelist().count(name) != 1 for name in names):
            raise ValueError("expected exactly one copy of each requested OPPU member")
        raw = {name: archive.read(name) for name in names}
    records, labels = (json.loads(raw[name]) for name in names[:2])
    if not isinstance(records, list) or not records:
        raise ValueError("OPPU task has no users")
    if not isinstance(labels, dict) or labels.get("task") != native_task:
        raise ValueError("OPPU label task does not match the requested task")
    if not isinstance(labels.get("golds"), list):
        raise ValueError("OPPU labels must contain a golds list")
    targets = {}
    for label in labels["golds"]:
        if not isinstance(label, dict):
            raise ValueError("invalid OPPU label record")
        identifier = _identifier(label.get("id"), "query")
        if identifier in targets:
            raise ValueError("duplicate OPPU label identifier")
        targets[identifier] = _text(label.get("output"), "target")
    if include_others:
        others = json.loads(raw[names[2]])
        if not isinstance(others, list):
            raise ValueError("OPPU other-user pool must be a list")
        for record in others:
            if not isinstance(record, dict) or not isinstance(record.get("query"), list):
                raise ValueError("invalid OPPU other-user record")
            for question in record["query"]:
                if not isinstance(question, dict):
                    raise ValueError("invalid OPPU query record")
                identifier = _identifier(question.get("id"), "query")
                if identifier in targets:
                    raise ValueError("duplicate OPPU label identifier")
                targets[identifier] = _text(question.get("gold"), "target")
        records = records + others
    users, user_ids, query_ids = [], set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid OPPU user record")
        user_id = _identifier(record.get("user_id"), "user")
        if user_id in user_ids:
            raise ValueError("duplicate OPPU user identifier")
        user_ids.add(user_id)
        history = record.get("profile")
        questions = record.get("query")
        if not isinstance(history, list) or not history or not isinstance(questions, list) or not questions:
            raise ValueError(f"user {user_id} needs history and questions")
        projected = []
        for item in history:
            if not isinstance(item, dict) or not all(field in item for field in history_fields):
                raise ValueError(f"incomplete {task} history record for {user_id}")
            projected.append({field: item[field] for field in history_fields})
        queries = []
        for question in questions:
            if not isinstance(question, dict):
                raise ValueError("invalid OPPU query record")
            query_id = _identifier(question.get("id"), "query")
            if query_id in query_ids:
                raise ValueError("duplicate OPPU query identifier")
            query_ids.add(query_id)
            target = targets.get(query_id)
            if target is None or question.get("gold") != target:
                raise ValueError(f"embedded and separate labels disagree for {query_id}")
            queries.append(LaMPQuery(query_id, _text(question.get("input"), "query"), target))
        users.append(OPPUUser(task, user_id, json.dumps(projected, ensure_ascii=False), tuple(queries)))
    if query_ids != set(targets):
        raise ValueError("query and label coverage differ")
    return OPPUTask(task, native_task, tuple(users), {
        name: hashlib.sha256(content).hexdigest() for name, content in raw.items()
    })