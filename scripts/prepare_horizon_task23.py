"""Prepare title-validated, frozen HORIZON Task2/3 candidate data."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from ohpu.benchmarks import horizon
from ohpu.benchmarks.horizon_retrieval import candidate_ids, normalized_title, stable_seed
from prepare_horizon_cohort import digest_file, download_file, fetch_json, source_manifest, write_json


PROTOCOL = "horizon-task23-sampled101-hyperharness-v1"


def metadata_catalog(cohort, cache, catalog):
    import duckdb

    categories = set(json.loads((cohort / "plan.json").read_text())["sources"]["categories"]
                     if cohort is not None else source_manifest()["categories"])
    base = f"https://huggingface.co/api/datasets/{horizon.AMAZON_REPOSITORY}/tree/{horizon.AMAZON_REVISION}"
    entries = fetch_json(base + "/raw/meta_categories?recursive=false&expand=false")
    entries = sorted((entry for entry in entries if Path(entry["path"]).stem.removeprefix("meta_") in categories),
        key=lambda entry: entry["path"])
    if len(entries) != len(categories):
        raise ValueError("Incomplete pinned metadata category listing")
    sources = {"revision": horizon.AMAZON_REVISION, "files": [{"path": entry["path"],
        "bytes": entry["size"], "sha256": entry["lfs"]["oid"]} for entry in entries]}
    manifest = catalog.with_suffix(".manifest.json")
    if catalog.exists():
        receipt = json.loads(manifest.read_text())
        if receipt["sources"] != sources or digest_file(catalog) != receipt["sha256"]:
            raise ValueError("Cached title catalog differs from its receipt")
        return receipt
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda entry: download_file(entry, cache), sources["files"]))
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute("SET memory_limit='64GB'")
    connection.execute("SET preserve_insertion_order=false")
    def title_key(title):
        try:
            return normalized_title(title)
        except ValueError:
            return None
    connection.create_function("title_key", title_key, return_type="VARCHAR", null_handling="special")
    parts = []
    for path in paths:
        part = path.with_suffix(".titles.parquet")
        if not part.exists():
            connection.read_json(str(path), format="newline_delimited",
                columns={"parent_asin": "VARCHAR", "title": "VARCHAR", "main_category": "VARCHAR"},
                maximum_object_size=64 * 1024 * 1024).create_view("metadata_source", replace=True)
            connection.sql("SELECT parent_asin AS item_id, title, main_category AS category, title_key(title) AS title_key "
                "FROM metadata_source WHERE parent_asin IS NOT NULL").write_parquet(str(part), compression="zstd")
        parts.append(str(part))
        print(json.dumps({"metadata_projected": path.name}), flush=True)
    connection.read_parquet(parts, filename=True).create_view("title_parts", replace=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    partial = catalog.with_suffix(".partial.parquet")
    connection.sql("""SELECT item_id, title, category, title_key FROM title_parts WHERE title_key IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY item_id ORDER BY filename DESC, title DESC)=1
        ORDER BY item_id""").write_parquet(str(partial), compression="zstd")
    partial.rename(catalog)
    count = connection.execute("SELECT count(*) FROM read_parquet(?)", [str(catalog)]).fetchone()[0]
    receipt = {"sources": sources, "sha256": digest_file(catalog), "items": count,
        "title_policy": "NFKC and whitespace-normalized casefold key; exclude empty/none/null/nan/n-a titles",
        "duplicate_id_policy": "last lexicographic source filename, then title descending"}
    write_json(manifest, receipt)
    return receipt


def prepare(cohort, catalog, output, *, seed=20261022):
    import duckdb
    import pyarrow as arrow

    if output.exists():
        raise FileExistsError("Use a fresh prepared-release directory")
    source_stats = json.loads((cohort / "stats.json").read_text())
    for name, digest in source_stats["artifacts_sha256"].items():
        if digest_file(cohort / name) != digest:
            raise ValueError(f"Source cohort changed: {name}")
    users = [json.loads(line) for line in (cohort / "users.jsonl").read_text().splitlines()]
    if len(users) != 500:
        raise ValueError("Expected the registered 500-user cohort")
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute("SET memory_limit='64GB'")
    connection.read_parquet(str(catalog)).create_view("catalog", replace=True)
    needed = set()
    for user in users:
        history = [event for event in user["events"] if event["timestamp"] < horizon.CUTOFF_MS]
        future = sorted((event for event in user["events"] if event["timestamp"] >= horizon.CUTOFF_MS),
            key=lambda event: (event["timestamp"], event["item_id"]))[:10]
        user["history"] = history
        user["targets"] = future
        needed.update(event["item_id"] for event in history + future)
    connection.register("needed", arrow.table({"item_id": sorted(needed)}))
    columns = ("item_id", "title", "category", "title_key")
    selected_rows = connection.execute("SELECT catalog.* FROM catalog SEMI JOIN needed USING(item_id)").fetchall()
    titles = {row[0]: dict(zip(columns, row)) for row in selected_rows}
    failures, exclusions = [], []
    for user in users:
        missing_targets = [event["item_id"] for event in user["targets"] if event["item_id"] not in titles]
        missing_history = [event["item_id"] for event in user["history"] if event["item_id"] not in titles]
        history_count = len(user["history"]) - len(missing_history)
        if missing_targets or history_count <= 500 or len(user["targets"]) != 10:
            failures.append({"user_id": user["user_id"], "missing_targets": missing_targets, "valid_history_count": history_count})
        if missing_history:
            exclusions.append({"user_id": user["user_id"], "missing_history_ids": missing_history})
    output.mkdir(parents=True)
    write_json(output / "metadata_audit.json", {"required_items": len(needed), "found_items": len(titles),
        "failures": failures, "history_exclusions": exclusions})
    if failures:
        raise ValueError(f"Metadata gate failed for {len(failures)} users; audit retained, no role/candidate registration")
    roles = horizon.role_split([user["user_id"] for user in users], seed=seed)
    catalog_ids = connection.execute("SELECT item_id FROM catalog ORDER BY item_id").fetchnumpy()["item_id"]
    proposals = {}
    proposed_ids = set()
    for user in users:
        for index, target in enumerate(user["targets"]):
            key = f"{user['user_id']}:{index}"
            generator = random.Random(stable_seed(seed, "proposal:" + key))
            indices = generator.sample(range(len(catalog_ids)), min(256, len(catalog_ids)))
            values = sorted({target["item_id"], *(str(catalog_ids[position]) for position in indices)})
            proposals[key] = values
            proposed_ids.update(values)
    connection.register("proposed", arrow.table({"item_id": sorted(proposed_ids)}))
    proposal_rows = connection.execute("SELECT catalog.* FROM catalog SEMI JOIN proposed USING(item_id)").fetchall()
    proposal_titles = {row[0]: dict(zip(columns, row)) for row in proposal_rows}
    candidates = []
    required_vectors = set()
    for user in users:
        observed = {event["item_id"] for event in user["events"]}
        target_titles = {titles[event["item_id"]]["title"] for event in user["targets"]}
        for index, target in enumerate(user["targets"]):
            key = f"{user['user_id']}:{index}"
            selected = candidate_ids([proposal_titles[item_id] for item_id in proposals[key]],
                user_id=user["user_id"], target_index=index, positive_id=target["item_id"],
                excluded_ids=observed, positive_titles=target_titles, seed=seed)
            candidates.append({"user_id": user["user_id"], "target_index": index,
                "positive_id": target["item_id"], "candidate_ids": selected})
            required_vectors.update(selected)
    prepared_users = []
    for user in users:
        events = [event for event in user["history"] if event["item_id"] in titles] + user["targets"]
        events = [{**event, "title": titles[event["item_id"]]["title"]} for event in events]
        events.sort(key=lambda event: (event["timestamp"], event["item_id"]))
        prepared_users.append({"user_id": user["user_id"], "events": events,
            "history_count": sum(event["timestamp"] < horizon.CUTOFF_MS for event in events),
            "target_ids": [event["item_id"] for event in user["targets"]]})
    write_json(output / "roles.json", roles)
    for name, rows in (("users.jsonl", prepared_users), ("candidates.jsonl", candidates),
            ("items.jsonl", [proposal_titles[item_id] for item_id in sorted(required_vectors)])):
        with (output / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    plan = {"protocol": PROTOCOL, "cohort_root": str(cohort.resolve()), "source_cohort_sha256": digest_file(cohort / "cohort.json"),
        "catalog_sha256": digest_file(catalog), "seed": seed, "users": len(users), "train_users": len(roles["train"]),
        "test_users": len(roles["test"]), "candidate_lists": len(candidates), "unique_candidate_items": len(required_vectors),
        "negative_sampling": "Uniform 256-ID catalog proposal then uniform 100 eligible negatives; insufficient proposal stops registration",
        "history_exclusions": sum(len(row["missing_history_ids"]) for row in exclusions),
        "metric": "exact-ID NDCG@10", "fusion": "max cosine over ten strings", "input_tokens": 32768, "output_tokens": 768,
        "task2_requests": 5000, "task3_requests": 500, "task2_cases": 5000, "task3_cases": 5000}
    write_json(output / "plan.json", plan)
    write_json(output / "registration.json", {"files": {path.name: digest_file(path) for path in output.iterdir() if path.is_file()}})
    return plan


def main():
    parser = argparse.ArgumentParser(description="Build the pinned Amazon title catalog used for MemoryCD candidates.")
    parser.add_argument("command", choices=("metadata",))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--public-categories", action="store_true", help="Use the pinned upstream category list")
    args = parser.parse_args()
    if not args.public_categories:
        parser.error("Use --public-categories")
    print(json.dumps(metadata_catalog(None, args.cache, args.catalog), indent=2), flush=True)


if __name__ == "__main__":
    main()
