"""Leakage-checked, history-supervised LaMP pilot data and evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from ohpu.benchmarks.loaders.lamp import OPPUUser
from ohpu.per_user_adaptation.executor import Runner


EXECUTOR_MODEL = "gpt-5.6-luna"
MOVIE_TAGS = (
    "sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia",
    "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking",
    "social commentary", "violence", "true story",
)
PROMPTS = {
    "movie_tagging": "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [" + ", ".join(MOVIE_TAGS) + "] description: ",
    "product_rating": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: ",
    "news_headline": "Generate a headline for the following article: ",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def content_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def query_content(task: str, query: str) -> str:
    if task not in PROMPTS or not query.startswith(PROMPTS[task]):
        raise ValueError("query does not match the registered LaMP task template")
    content = query[len(PROMPTS[task]):]
    if not content.strip() and task != "movie_tagging":
        raise ValueError("LaMP query has no source text")
    return content


def prepare_movie_user(user: OPPUUser, *, selection_fraction: float = 0.2) -> dict[str, Any]:
    if user.task != "movie_tagging" or not 0 < selection_fraction < 1:
        raise ValueError("movie preparation needs movie_tagging and a fraction between zero and one")
    history = user.history()
    queries = {content_key(query_content(user.task, query.input)) for query in user.queries}
    if any(query.target not in MOVIE_TAGS for query in user.queries):
        raise ValueError("native movie target is outside the registered label set")
    excluded, groups = [], {}
    for index, record in enumerate(history):
        date.fromisoformat(record["date"])
        if not isinstance(record["description"], str) or record["tag"] not in MOVIE_TAGS:
            raise ValueError("invalid movie history source or tag")
        key = content_key(record["description"])
        if key and key in queries:
            excluded.append({"index": index, "record_id": str(record["id"]),
                "record_sha256": digest(record), "reason": "native_query_source_in_profile"})
        else:
            groups.setdefault(key, []).append(index)
    ordered = sorted(groups, key=lambda key: (max(history[index]["date"] for index in groups[key]), digest(key)))
    if len(ordered) < 2:
        raise ValueError("movie user needs at least two distinct history groups")
    holdout = max(1, math.ceil(len(ordered) * selection_fraction))
    if holdout >= len(ordered):
        raise ValueError("movie selection leaves no training history")
    selection_keys = set(ordered[-holdout:])
    train_indices = sorted(index for key, indices in groups.items() if key not in selection_keys for index in indices)
    selection_indices = sorted(index for key, indices in groups.items() if key in selection_keys for index in indices)
    eligible = sorted(train_indices + selection_indices, key=lambda index: (history[index]["date"], index))
    cases = {"train": [], "selection": [], "test": []}
    for partition, indices in (("train", train_indices), ("selection", selection_indices)):
        for index in indices:
            record = history[index]
            visible = [position for position in train_indices
                if history[position]["date"] < record["date"]
                and content_key(history[position]["description"]) != content_key(record["description"])]
            visible.sort(key=lambda position: (history[position]["date"], position))
            cases[partition].append({"case_id": f"oppu::movie_tagging::{user.user_id}::history::{index}",
                "query": PROMPTS[user.task] + record["description"], "target": record["tag"],
                "history_indices": visible, "history_target_index": index, "cutoff": record["date"],
                "source": "history_supervision"})
    for query in user.queries:
        cases["test"].append({"case_id": f"oppu::movie_tagging::{user.user_id}::native::{query.query_id}",
            "query_id": query.query_id, "query": query.input, "target": query.target,
            "history_indices": eligible, "cutoff": None, "source": "provided_native_query"})
    return {"protocol": "oppu-most-active-movie-history-v1", "task": user.task, "user_id": user.user_id,
        "native_task_alias": "LaMP_2M", "archive_task_name": "LaMP_8", "profile": history,
        "profile_sha256": digest(history), "cases": cases, "original_history_count": len(history),
        "original_query_count": len(user.queries), "excluded_native_query_history": excluded,
        "empty_description_history_indices": [index for index, record in enumerate(history) if not record["description"].strip()],
        "empty_description_policy": "retain missing-input records and native tags; never synthesize a description",
        "history_selection_fraction": selection_fraction, "history_selection_indices": selection_indices,
        "history_selection_policy": "latest content groups by last observed date, SHA-256 tie-breaker",
        "cold_start_training_cases_retained": sum(not case["history_indices"] for case in cases["train"]),
        "selection_sources_excluded_from_train_and_selection_memory": True,
        "native_memory_scope": "all provided eligible profile records, including historical selection records, only after cohort lock",
        "native_query_time_scope": "OPPU-provided profile; query timestamps unavailable, no independent temporal-validity claim",
        "test_sampling": "none; every provided native query is retained"}


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix="." + path.name + ".", suffix=".tmp") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        try:
            path.hardlink_to(stream.name)
        except FileExistsError:
            with Path(stream.name).open("rb") as candidate, path.open("rb") as existing:
                while chunk := candidate.read(64 * 1024):
                    if existing.read(len(chunk)) != chunk:
                        raise ValueError(f"immutable experiment artifact changed: {path.name}")
                if existing.read(1):
                    raise ValueError(f"immutable experiment artifact changed: {path.name}")


class SeededLuna:
    def __init__(self, client: Any, seed: int, *, preserve_request_seed: bool = False,
                 model: str = EXECUTOR_MODEL):
        if client.model != model:
            raise ValueError(f"LaMP execution requires {model}")
        self.client, self.seed = client, seed
        self.preserve_request_seed = preserve_request_seed
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def chat(self, messages: list[dict], **options: Any) -> dict:
        if options.get("model", self.model) != self.model:
            raise ValueError("execution or memory request attempted to change the registered executor model")
        effective_seed = options.get("seed", self.seed) if self.preserve_request_seed else self.seed
        return self.client.chat(messages, **{**options, "seed": effective_seed, "model": self.model})
