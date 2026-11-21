"""Fixed sampled-candidate retrieval and exact-ID scoring for HyperHarness."""
from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from typing import Any

import numpy as np


OUTPUT_KEYS = {"task2": "queries", "task3": "item_descriptions_timewise"}
METRICS = ("ndcg10", "hr1", "hr5", "hr10", "mrr")
BLAIR_MODEL = "hyp1231/blair-roberta-base"
BLAIR_REVISION = "88adfe2b621cb202dee9aabb19b59de8f622844c"


def normalized_title(title: str) -> str:
    if not isinstance(title, str):
        raise ValueError("Product title must be a string")
    value = " ".join(unicodedata.normalize("NFKC", title).split()).casefold()
    if value in {"", "none", "null", "nan", "n/a"}:
        raise ValueError("Product title is missing or invalid")
    return value


def stable_seed(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:16], "big")


def parse_output(content: str, task: str) -> list[str]:
    key = OUTPUT_KEYS[task]
    def unique_keys(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate output object key")
            result[name] = value
        return result
    value = json.loads(content, object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or set(value) != {key}:
        raise ValueError(f"Output must contain only {key}")
    texts = value[key]
    if not isinstance(texts, list) or len(texts) != 10:
        raise ValueError("Output must contain exactly ten strings")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every output must be a nonempty string")
    return texts


def candidate_ids(catalog: list[dict[str, Any]], *, user_id: str, target_index: int,
                  positive_id: str, excluded_ids: set[str], positive_titles: set[str],
                  seed: int = 20261022, negatives: int = 100) -> list[str]:
    by_id = {item["item_id"]: item for item in catalog}
    if len(by_id) != len(catalog) or positive_id not in by_id:
        raise ValueError("Candidate catalog must have unique IDs and contain the target")
    if negatives < 1:
        raise ValueError("Negative count must be positive")
    blocked_titles = {normalized_title(title) for title in positive_titles}
    blocked_titles.add(normalized_title(by_id[positive_id]["title"]))
    available = sorted(item_id for item_id, item in by_id.items()
        if item_id != positive_id and item_id not in excluded_ids
        and normalized_title(item["title"]) not in blocked_titles)
    if len(available) < negatives:
        raise ValueError("Not enough eligible negative items")
    generator = random.Random(stable_seed(seed, f"{user_id}:{target_index}"))
    selected = [positive_id, *generator.sample(available, negatives)]
    generator.shuffle(selected)
    return selected


def normalized_vectors(values) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a nonempty finite matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Embedding vectors must have nonzero finite norms")
    return vectors / norms


def rank_candidates(text_vectors, item_vectors, item_ids: list[str]) -> tuple[list[str], list[float]]:
    texts = normalized_vectors(text_vectors)
    items = normalized_vectors(item_vectors)
    if texts.shape[0] != 10 or texts.shape[1] != items.shape[1]:
        raise ValueError("Expected ten compatible query embeddings")
    if len(item_ids) != len(items) or len(set(item_ids)) != len(item_ids):
        raise ValueError("Candidate IDs and embeddings must align without duplicates")
    scores = np.max(texts @ items.T, axis=0)
    order = np.argsort(-scores, kind="stable")
    return [item_ids[index] for index in order], [float(scores[index]) for index in order]


def rank_metrics(ranked_ids: list[str], positive_id: str) -> dict[str, float]:
    if len(ranked_ids) != len(set(ranked_ids)) or positive_id not in ranked_ids:
        raise ValueError("Ranking must contain the positive exactly once and no duplicate IDs")
    rank = ranked_ids.index(positive_id) + 1
    return {"ndcg10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "hr1": float(rank <= 1), "hr5": float(rank <= 5), "hr10": float(rank <= 10), "mrr": 1.0 / rank}


def invalid_metrics() -> dict[str, float]:
    return dict.fromkeys(METRICS, 0.0)


def score_user(records, candidates, *, item_ids, item_vectors, encode):
    indexed = {(row["task"], row["request_index"]): row for row in records}
    expected = {("task2", index) for index in range(10)} | {("task3", 0)}
    if len(records) != 11 or set(indexed) != expected:
        raise ValueError("Incomplete or duplicate Task2/3 prediction requests")
    if len(candidates) != 10 or {case["target_index"] for case in candidates} != set(range(10)):
        raise ValueError("Incomplete candidate panel")
    user_ids = {row["user_id"] for row in records} | {case["user_id"] for case in candidates}
    if len(user_ids) != 1:
        raise ValueError("Cross-user scoring panel")
    locations = {item_id: index for index, item_id in enumerate(item_ids)}
    if len(locations) != len(item_ids) or len(item_vectors) != len(item_ids):
        raise ValueError("Item embedding catalog is inconsistent")
    vectors = {}
    output = []
    for task in ("task2", "task3"):
        for candidate in sorted(candidates, key=lambda row: row["target_index"]):
            index = candidate["target_index"]
            prediction = indexed[task, index if task == "task2" else 0]
            ids = candidate["candidate_ids"]
            if len(ids) != 101 or len(set(ids)) != 101 or candidate["positive_id"] not in ids:
                raise ValueError("Every case requires 101 unique IDs and one positive")
            ranking, scores = [], []
            if prediction["format_error"] is not None:
                metrics = invalid_metrics()
            else:
                texts = parse_output(prediction["content"], task)
                identity = tuple(texts)
                if identity not in vectors:
                    vectors[identity] = encode(texts)
                ranking, scores = rank_candidates(vectors[identity],
                    np.asarray(item_vectors[[locations[item_id] for item_id in ids]]), ids)
                metrics = rank_metrics(ranking, candidate["positive_id"])
            output.append({"user": prediction["user_id"], "task_type": task,
                "query_id": f"horizon::{task}::{prediction['user_id']}::{index}",
                "request_id": prediction["request_id"], "target_index": index,
                "positive_id": candidate["positive_id"], "ranking": ranking, "ranking_scores": scores,
                "metrics": metrics, "accuracy": 100 * metrics["ndcg10"],
                "agent_response": prediction["content"], "format_error": prediction["format_error"],
                "executor_context": [prediction.get("packing", {}).get("injected_evidence", "")],
                "guidance": prediction.get("guidance", "")})
    task_means = {task: {metric: sum(row["metrics"][metric] for row in output if row["task_type"] == task) / 10
        for metric in METRICS} for task in ("task2", "task3")}
    return output, {"evaluation_complete": True, "runtime_violations": [], "cases": 20,
        "requests": 11, "task_metrics": task_means,
        "accuracy_pct_micro": 50 * (task_means["task2"]["ndcg10"] + task_means["task3"]["ndcg10"]),
        "invalid_format_requests": sum(row["format_error"] is not None for row in records)}


class BlairEncoder:
    def __init__(self, *, device="cpu", cache_dir=None, batch_size=64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(BLAIR_MODEL, revision=BLAIR_REVISION, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(BLAIR_MODEL, revision=BLAIR_REVISION, cache_dir=cache_dir)
        self.model = self.model.to(device=device, dtype=torch.float32).eval()
        torch.set_num_threads(2)
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        self.contract = {"model": BLAIR_MODEL, "revision": BLAIR_REVISION,
            "pooling": "last_hidden_state[:,0]", "normalization": "l2", "max_length": 512,
            "dtype": "float32", "device": device, "torch": torch.__version__}

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot encode empty text")
        batches = []
        for offset in range(0, len(texts), self.batch_size):
            inputs = self.tokenizer(texts[offset:offset + self.batch_size], padding=True,
                truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                output = self.model(**inputs).last_hidden_state[:, 0]
            batches.append(output.float().cpu().numpy())
        return normalized_vectors(np.concatenate(batches))
