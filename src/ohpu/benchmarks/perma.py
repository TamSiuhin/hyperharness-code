"""Causal, user-disjoint PERMA multiple-choice data contracts."""
from __future__ import annotations

import hashlib
from datetime import date
import json
from pathlib import Path
import random
import re
import shutil
import subprocess


DATASET_REPOSITORY = "ustclsc/PERMA"
DATASET_REVISION = "440e64e4fb8baec6f7ad10c1de135505f93e7cb1"
PROTOCOL = "perma-causal-mcq-user-half-v1"
SEED = 20261022
USERS = ("user108", "user109", "user112", "user123", "user1377", "user334", "user354", "user419", "user507", "user914")
OBJECTIVE = "equal-user mean MCQ accuracy across Clean SD/MD checkpoints; scaled 0-100"


def roles(user_ids, *, seed=SEED):
    ordered = sorted(user_ids)
    if len(ordered) < 2 or len(ordered) % 2 or len(ordered) != len(set(ordered)):
        raise ValueError("A 1:1 user split requires an even number of distinct users")
    selected = set(random.Random(seed).sample(ordered, len(ordered) // 2))
    return {"seed": seed, "unit": "user", "train": sorted(selected),
        "test": [user for user in ordered if user not in selected], "official_split": False}


def validate_mcq(case):
    if not isinstance(case.get("question"), str) or not case["question"].strip():
        raise ValueError("MCQ requires a nonempty question")
    options = case.get("options")
    if not isinstance(options, str) or re.findall(r"(?m)^([A-H]):\s", options) != list("ABCDEFGH"):
        raise ValueError("MCQ requires eight ordered A-H options")
    if case.get("gold_label") not in list("ABCDEFGH"):
        raise ValueError("MCQ requires a valid gold label")
    if type(case.get("task_type")) is not int or case["task_type"] not in (1, 2, 3):
        raise ValueError("MCQ requires a temporal checkpoint type")


def request(case):
    validate_mcq(case)
    return ("Choose the most appropriate response to the user query using the available memory.\n"
        + "User query:\n" + case["question"] + "\nOptions:\n" + case["options"]
        + "\nReturn only the option letter A-H, without explanation.")


def parse_output(content):
    if not isinstance(content, str) or content.strip() not in list("ABCDEFGH"):
        raise ValueError("Expected exactly one option letter A-H")
    return content.strip()


def history_events(context, question_date):
    cutoff = date.fromisoformat(question_date)
    events, excluded = [], 0
    for messages, timestamp in context:
        if date.fromisoformat(timestamp[:10]) > cutoff:
            excluded += 1
            continue
        pair = []
        for message in messages:
            if message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str):
                raise ValueError("Invalid history message")
            if message["role"] == "user" and pair:
                events.append((timestamp, pair))
                pair = []
            pair.append({"role": message["role"], "content": message["content"]})
        if pair:
            events.append((timestamp, pair))
    result = []
    for index, (timestamp, messages) in enumerate(events):
        safe = {"date": timestamp, "messages": messages}
        result.append({"id": f"h{index}", "ts": index + 1, "app": "conversation",
            "text": json.dumps(safe, ensure_ascii=False), "event": safe})
    return result, excluded


def prepare_user(user, sources, metadata):
    contexts, cases, seen, excluded = {}, [], set(), 0
    for domain, rows in sources.items():
        for row in rows:
            key = f"{row['task_id']}_{row['type']}"
            if key in seen:
                raise ValueError("Duplicate MCQ checkpoint")
            seen.add(key)
            meta = metadata[key]
            validate_mcq(meta)
            if meta["task_id"] != row["task_id"] or meta["task_type"] != row["type"]:
                raise ValueError("MCQ metadata does not match its checkpoint")
            events, removed = history_events(row["context"], meta["question_date"]) if row["type"] != 1 else ([], 0)
            excluded += removed
            digest = hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()
            contexts[digest] = events
            cases.append({"case_id": key, "domain": domain, "context_id": digest,
                **{name: meta[name] for name in ("question", "options", "gold_label", "task_type")}})
    if not cases or not any(case["task_type"] != 1 for case in cases):
        raise ValueError("Require nonempty MCQ evaluation with history checkpoints")
    histories = [contexts[case["context_id"]] for case in cases if case["task_type"] != 1]
    common = list(min(histories, key=len))
    for history in histories:
        length = 0
        for previous, current in zip(common, history):
            if previous != current:
                break
            length += 1
        common = common[:length]
    return {"user_id": user, "contexts": contexts, "construction_events": common,
        "cases": cases, "excluded_future_sessions": excluded}


def read_records(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require_local_storage(path, *, minimum_free_bytes=1024 ** 3):
    path = Path(path).resolve()
    existing = path
    while not existing.exists():
        existing = existing.parent
    mount = json.loads(subprocess.check_output(
        ["findmnt", "--json", "--target", str(existing), "--output", "SOURCE,FSTYPE,TARGET"], text=True))
    filesystem = mount["filesystems"][0]
    if (filesystem["fstype"] not in ("ext4", "xfs", "overlay", "tmpfs", "btrfs", "zfs")
            or "rslex" in filesystem["source"].lower()):
        raise ValueError(f"Node-local storage required, refusing {path}: {filesystem}")
    if shutil.disk_usage(existing).free < minimum_free_bytes:
        raise ValueError(f"Insufficient local free space at {path}")
    return path


def prepare_dataset(source, out, *, download=False, seed=SEED):
    source = require_local_storage(source)
    out = require_local_storage(out)
    if out.exists():
        raise ValueError("Prepared output already exists; use a fresh directory")
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(DATASET_REPOSITORY, repo_type="dataset", revision=DATASET_REVISION,
            local_dir=source, cache_dir=source / ".cache/hub", max_workers=4, allow_patterns=["tasks/*/input_data_c.json",
                "tasks/*/input_data_multi_c.json", "evaluation/*/meta/overall/*.json"])
    users, files = [], {}
    def load(path):
        payload = path.read_bytes()
        files[str(path.relative_to(source))] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    for user in USERS:
        sources = {domain: load(source / "tasks" / user / filename)["overall"] for domain, filename in
            (("single", "input_data_c.json"), ("multi", "input_data_multi_c.json"))}
        keys = [f"{row['task_id']}_{row['type']}" for rows in sources.values() for row in rows]
        metadata = {key: load(source / "evaluation" / user / "meta/overall" / f"{key}.json") for key in keys}
        users.append(prepare_user(user, sources, metadata))
    split = roles([user["user_id"] for user in users], seed=seed)
    summary = {"dataset": "perma", "protocol": PROTOCOL, "setting": "clean", "domains": ["single", "multi"],
        "objective": OBJECTIVE, "source": {"repository": DATASET_REPOSITORY, "revision": DATASET_REVISION},
        "split": {role: {"users": len(split[role]), "queries": sum(len(user["cases"]) for user in users
            if user["user_id"] in split[role])} for role in ("train", "test")},
        "construction_history_counts": {user["user_id"]: len(user["construction_events"]) for user in users},
        "excluded_future_sessions": sum(user["excluded_future_sessions"] for user in users)}
    out.mkdir(parents=True)
    for name, value in (("roles.json", split), ("summary.json", summary), ("source_files.json", files)):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    with (out / "users.jsonl").open("x") as stream:
        for user in users:
            stream.write(json.dumps(user, ensure_ascii=False) + "\n")
    registration = {"files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in out.iterdir()}}
    (out / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    return summary
