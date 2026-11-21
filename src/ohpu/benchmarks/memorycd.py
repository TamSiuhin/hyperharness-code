"""Causal, user-disjoint MemoryCD four-task data contracts."""
from __future__ import annotations

import hashlib
import gzip
import json
import math
from pathlib import Path
import random
import statistics


DATASET_REPOSITORY = "WZDavid/MemoryCD"
DATASET_REVISION = "14b934ce3f76f96b6c3c2efa173525f0da62952e"
DOMAINS = ("Beauty_and_Personal_Care", "Books", "Electronics", "Home_and_Kitchen")
TARGET_DOMAIN = "Home_and_Kitchen"
SEED = 20261022
PROTOCOL = "memorycd-causal-cross-domain-four-task-v2"
TASKS = ("item_ranking", "rating_prediction", "review_summarization", "review_generation")
TEXT_OUTPUTS = {"review_summarization": "title", "review_generation": "review"}
OBJECTIVE = "equal-user mean of four equal-weight task scores: NDCG@5, 1-MAE/4, summary ROUGE-L, review ROUGE-L; scaled 0-100"


def stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def roles(user_ids, *, train_users=100, seed=SEED):
    ordered = sorted(user_ids)
    if len(ordered) != len(set(ordered)) or not 0 < train_users < len(ordered):
        raise ValueError("Require unique users and nonempty disjoint TRAIN/TEST cohorts")
    selected = set(random.Random(seed).sample(ordered, train_users))
    return {"seed": seed, "unit": "user", "train": sorted(selected),
        "test": [user for user in ordered if user not in selected], "official_split": False}


def product_view(item_id, metadata):
    meta = metadata.get(item_id)
    if not meta or not isinstance(meta.get("title"), str) or not meta["title"].strip():
        raise ValueError(f"Missing product title: {item_id}")
    def text(value, limit):
        return (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))[:limit]

    return {"item_id": item_id, "title": meta["title"][:500],
        "category": text(meta.get("main_category", ""), 200),
        "description": text(meta.get("description", []), 1000),
        "features": text(meta.get("features", []), 1000)}


def prepare_user(record, metadata, *, num_test=3, seed=SEED, candidate_metadata=None):
    user = record["user_id"]
    interactions = record["interactions"]
    targets = sorted(interactions.get(TARGET_DOMAIN, []), key=lambda row: row["timestamp"])
    if num_test < 1 or len(targets) < num_test:
        raise ValueError(f"Insufficient target interactions: {user}")
    targets = targets[-num_test:]
    cutoff = targets[0]["timestamp"]
    history, excluded, shortened, missing_history_titles = [], 0, 0, 0
    for domain in DOMAINS:
        if domain == TARGET_DOMAIN:
            continue
        for row in interactions.get(domain, []):
            if row["timestamp"] >= cutoff:
                excluded += 1
                continue
            try:
                item = product_view(row["parent_asin"], metadata)
            except ValueError:
                item = {"item_id": row["parent_asin"], "title": "", "category": domain}
                missing_history_titles += 1
            review = row.get("text") or ""
            title = row.get("title") or ""
            shortened += len(review) > 2000 or len(title) > 300
            history.append({"timestamp": row["timestamp"], "domain": domain,
                "product": {name: item[name] for name in ("item_id", "title", "category")},
                "rating": row["rating"], "review_title": title[:300], "review": review[:2000]})
    history.sort(key=lambda row: (row["timestamp"], row["domain"], row["product"]["item_id"]))
    if not history:
        raise ValueError(f"No strictly causal source-domain history: {user}")
    cases = []
    for index, row in enumerate(targets):
        target = row["parent_asin"]
        negatives = row.get("negative_parent_asins", [])
        if not negatives or len(negatives) != len(set(negatives)) or target in negatives:
            raise ValueError(f"Invalid supplied candidate set: {user}/{index}")
        missing = [item for item in negatives if candidate_metadata is not None and item not in candidate_metadata]
        negatives = [item for item in negatives if item not in missing]
        if not negatives:
            raise ValueError("No eligible negative candidates")
        candidates = [target, *negatives]
        random.Random(stable_seed(seed, user, index)).shuffle(candidates)
        rating = row.get("rating")
        if type(rating) not in (int, float) or not math.isfinite(rating) or not 1 <= rating <= 5:
            raise ValueError(f"Invalid rating: {user}/{index}")
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in ("text", "title")):
            raise ValueError(f"Four-task evaluation requires a review body and title: {user}/{index}")
        def visible(item_id):
            if candidate_metadata is None:
                return product_view(item_id, metadata)
            item = candidate_metadata[item_id]
            return {"item_id": item_id, "title": item["title"][:500], "category": item["category"][:200]}

        cases.append({"case_id": f"{user}:{index}", "timestamp": row["timestamp"],
            "candidates": [visible(item_id) for item_id in candidates],
            "product": visible(target), "gold_item_id": target, "gold_rating": rating,
            "gold_review": row.get("text") or "", "gold_review_title": row.get("title") or "",
            "excluded_untitled_negatives": missing})
    return {"user_id": user, "cutoff": cutoff, "history": history, "cases": cases,
        "excluded_noncausal_source_interactions": excluded, "shortened_history_texts": shortened,
        "missing_history_titles": missing_history_titles}


def history_events(user):
    return [{"id": f"h{index}", "ts": index + 1, "app": row["domain"],
        "text": json.dumps(row, ensure_ascii=False, sort_keys=True), "event": row}
        for index, row in enumerate(user["history"])]


TEXT_OUTPUT_LIMITS = {"review_summarization": 160, "review_generation": 3000}


def request(case, task):
    if task == "item_ranking":
        count = min(5, len(case["candidates"]))
        return (f"Select and rank the TOP {count} of these {len(case['candidates'])} candidates for this user. "
            f"Return only JSON {{\"ranking\":[item_ids]}} with exactly {count} DIFFERENT candidate IDs, "
            "most relevant first. Do not repeat or invent IDs.\n") + json.dumps(
            {"candidates": case["candidates"]}, ensure_ascii=False, sort_keys=True)
    if task == "rating_prediction":
        return "Predict this user's rating. Return only JSON {\"rating\":integer from 1 to 5}.\n" + json.dumps(
            {"product": case["product"]}, ensure_ascii=False, sort_keys=True)
    if task in TEXT_OUTPUTS:
        visible = {"product": case["product"], "rating": case["gold_rating"]}
        if task == "review_summarization":
            visible["review"] = case["gold_review"]
            instruction = ('Summarize the supplied review as a concise title (3-10 words), matching this user\'s '
                'writing style and sentiment. Use at most 160 characters. Return only JSON {"title":"review title"}.\n')
        else:
            instruction = ('Write a detailed review of the supplied product, matching this user\'s writing style, '
                'and the supplied rating sentiment. Keep the review within 3000 characters; be concise when '
                'history is insufficient. Do not repeat phrases or pad the review. '
                'Return only JSON {"review":"review body"}.\n')
        return instruction + json.dumps(visible, ensure_ascii=False, sort_keys=True)
    raise ValueError("Unsupported MemoryCD task")


def output_schema(case, task):
    if task == "item_ranking":
        identifiers = [item["item_id"] for item in case["candidates"]]
        properties = {"ranking": {"type": "array", "items": {"type": "string", "enum": identifiers},
            "minItems": min(5, len(identifiers)), "maxItems": min(5, len(identifiers))}}
    elif task == "rating_prediction":
        properties = {"rating": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}
    elif task in TEXT_OUTPUTS:
        limit = TEXT_OUTPUT_LIMITS[task]
        properties = {TEXT_OUTPUTS[task]: {"type": "string", "minLength": 1, "maxLength": limit}}
    else:
        raise ValueError("Unsupported MemoryCD task")
    return {"type": "json_schema", "json_schema": {"name": "memorycd_" + task, "strict": True,
        "schema": {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}}}


def parse_output(text, task, case):
    def unique_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate output key")
            result[name] = value
        return result

    value = json.loads(text, object_pairs_hook=unique_pairs)
    if task == "item_ranking":
        if not isinstance(value, dict) or set(value) != {"ranking"}:
            raise ValueError("Expected ranking JSON")
        ranking = value["ranking"]
        expected = [item["item_id"] for item in case["candidates"]]
        if (not isinstance(ranking, list) or any(not isinstance(item, str) for item in ranking)
                or len(ranking) != min(5, len(expected)) or len(set(ranking)) != len(ranking)
                or not set(ranking) <= set(expected)):
            raise ValueError("Ranking must contain the required number of distinct allowed candidate IDs")
        return ranking
    if task in TEXT_OUTPUTS:
        field = TEXT_OUTPUTS[task]
        if (not isinstance(value, dict) or set(value) != {field}
            or not isinstance(value[field], str) or not value[field].strip()
            or len(value[field]) > TEXT_OUTPUT_LIMITS[task]):
            raise ValueError(f"Expected nonempty {field} JSON")
        return value[field]
    if task != "rating_prediction" or not isinstance(value, dict) or set(value) != {"rating"}:
        raise ValueError("Expected rating JSON")
    if type(value["rating"]) is not int or not 1 <= value["rating"] <= 5:
        raise ValueError("Rating must be an integer in [1, 5]")
    return value["rating"]


def ranking_metrics(ranking, target):
    position = ranking.index(target) + 1 if target in ranking else math.inf
    return {**{f"ndcg@{limit}": 1 / math.log2(position + 1) if position <= limit else 0.0
        for limit in (1, 3, 5)}, **{f"recall@{limit}": float(position <= limit) for limit in (1, 3, 5)}}


def text_metrics(prediction, reference):
    from rouge_score.rouge_scorer import RougeScorer
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Missing text-task reference")
    scores = RougeScorer(["rouge1", "rougeL"], use_stemmer=True).score(reference, prediction)
    tokens, expected = prediction.split(), [reference.split()]
    smoothing = SmoothingFunction().method1
    return {"rouge1": scores["rouge1"].fmeasure, "rougeL": scores["rougeL"].fmeasure,
        "bleu1": sentence_bleu(expected, tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        "bleu4": sentence_bleu(expected, tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)}


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(source):
    source = Path(source)
    receipt = json.loads((source / "source.json").read_text())
    if receipt["repository"] != DATASET_REPOSITORY or receipt["revision"] != DATASET_REVISION:
        raise ValueError("MemoryCD source revision differs from the contract")
    for name, entry in receipt["files"].items():
        if file_sha(source / name) != entry["sha256"]:
            raise ValueError(f"MemoryCD source changed: {name}")
    metadata = {}
    for domain in DOMAINS:
        with gzip.open(source / "meta" / f"{domain}.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                metadata.setdefault(row["parent_asin"], {name: row.get(name) for name in
                    ("title", "main_category", "description", "features")})
    with gzip.open(source / "users/cross_domain_users_sampled.jsonl.gz", "rt", encoding="utf-8") as stream:
        users = [json.loads(line) for line in stream]
    return users, metadata, receipt


def prepare_dataset(source, out, *, train_users=100, seed=SEED, num_test=3, metadata_catalog=None):
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError("Use a fresh prepared-data directory")
    records, metadata, receipt = load_source(source)
    split = roles([record["user_id"] for record in records], train_users=train_users, seed=seed)
    candidate_metadata = None
    catalog_receipt = None
    if metadata_catalog is not None:
        import duckdb
        required = set()
        for record in records:
            for row in sorted(record["interactions"][TARGET_DOMAIN], key=lambda row: row["timestamp"])[-num_test:]:
                required.update([row["parent_asin"], *row["negative_parent_asins"]])
        with duckdb.connect() as connection:
            rows = connection.execute("SELECT item_id,title,category FROM read_parquet(?) "
                "JOIN (SELECT unnest(?) AS item_id) USING (item_id)",
                [str(metadata_catalog), sorted(required)]).fetchall()
        candidate_metadata = {item_id: {"title": title, "category": category or ""}
            for item_id, title, category in rows if isinstance(title, str) and title.strip()}
        for item_id in required - set(candidate_metadata):
            item = metadata.get(item_id, {})
            if isinstance(item.get("title"), str) and item["title"].strip():
                candidate_metadata[item_id] = {"title": item["title"], "category": item.get("main_category") or ""}
        catalog_receipt = {"path": str(Path(metadata_catalog).resolve()), "sha256": file_sha(metadata_catalog),
            "source": "Amazon Reviews 2023 public metadata, revision 2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e",
            "required_items": len(required), "available_items": len(candidate_metadata)}
    users = [prepare_user(record, metadata, num_test=num_test, seed=seed, candidate_metadata=candidate_metadata)
        for record in sorted(records, key=lambda row: row["user_id"])]
    counts = [len(user["history"]) for user in users]
    candidates = [len(case["candidates"]) for user in users for case in user["cases"]]
    summary = {"protocol": PROTOCOL, "source": receipt, "roles": split, "num_test": num_test, "tasks": list(TASKS),
        "candidate_metadata": catalog_receipt,
        "candidate_policy": "provided positive and negatives; uniform title/category only; untitled negatives excluded and audited, never resampled",
        "user_count": len(users), "target_domain": TARGET_DOMAIN,
        "source_domains": [domain for domain in DOMAINS if domain != TARGET_DOMAIN],
        "history_policy": "fixed source-domain history strictly before the earliest held-out target; no target-domain history",
        "text_limits": {"product_title": 500, "product_category": 200, "product_description": 1000,
            "product_features": 1000, "history_review": 2000, "history_review_title": 300},
        "history": {"min": min(counts), "median": statistics.median(counts), "mean": statistics.fmean(counts),
            "max": max(counts), "total": sum(counts)},
        "candidate_counts": {"min": min(candidates), "max": max(candidates)},
        "excluded_noncausal_source_interactions": sum(user["excluded_noncausal_source_interactions"] for user in users),
        "shortened_history_texts": sum(user["shortened_history_texts"] for user in users),
        "missing_history_titles": sum(user["missing_history_titles"] for user in users),
        "excluded_untitled_negatives": sum(len(case["excluded_untitled_negatives"]) for user in users for case in user["cases"]),
        "queries": {name: {"users": len(split[name]), "per_task": len(split[name]) * num_test,
            "model_requests": len(split[name]) * num_test * len(TASKS)} for name in ("train", "test")}}
    out.mkdir(parents=True)
    for name, value in (("roles.json", split), ("summary.json", summary)):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    with (out / "users.jsonl").open("x") as stream:
        for user in users:
            stream.write(json.dumps(user, ensure_ascii=False) + "\n")
    if candidate_metadata is not None:
        with (out / "candidate_metadata.jsonl").open("x") as stream:
            for item_id in sorted(candidate_metadata):
                stream.write(json.dumps({"item_id": item_id, **candidate_metadata[item_id]}, ensure_ascii=False) + "\n")
    files = {path.name: file_sha(path) for path in out.iterdir() if path.is_file()}
    (out / "registration.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-test", type=int, default=3)
    parser.add_argument("--metadata-catalog", type=Path)
    args = parser.parse_args()
    summary = prepare_dataset(args.source, args.out, train_users=args.train_users, seed=args.seed,
        num_test=args.num_test, metadata_catalog=args.metadata_catalog)
    print(json.dumps({name: summary[name] for name in ("user_count", "history", "queries", "candidate_counts")}, indent=2))


if __name__ == "__main__":
    main()
