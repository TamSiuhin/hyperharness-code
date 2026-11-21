"""MUSE-Bench 1m histories with a reproducible user-disjoint research split."""
from __future__ import annotations

from datetime import date
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from ohpu.benchmarks.perma import require_local_storage


DATASET_REPOSITORY = "anonymous111111111/MUSE-Bench"
DATASET_REVISION = "dc671dc5da12152bb1c7b054f54569df6daa2362"
DATASET_FILE = "1m/data.parquet"
PROTOCOL = "muse-1m-user50-train50-test-v1"
SEED = 20261023
QUERY_TYPES = ("understand-personalized", "understand-experience", "apply-user-single",
    "apply-user-multi", "apply-user-temporal", "apply-assistant", "evaluate-interference",
    "evaluate-counterfactual")
OBJECTIVE = "equal-user mean five-way MCQ accuracy over all eight query types; scaled0-100"


def roles(user_ids, *, seed=SEED, train_count=50, test_count=50):
    ordered = sorted(user_ids)
    if (len(ordered) != len(set(ordered)) or train_count < 1 or test_count < 1
            or len(ordered) < train_count + test_count):
        raise ValueError("Require distinct users and sufficient data for the registered split")
    random.Random(seed).shuffle(ordered)
    return {"seed": seed, "unit": "user", "train": sorted(ordered[:train_count]),
        "test": sorted(ordered[train_count:train_count + test_count]),
        "excluded": sorted(ordered[train_count + test_count:]), "official_split": False,
        "source_split": "test", "history_configuration": "1m"}


def history_events(sessions):
    events, seen = [], set()
    for session in sessions:
        identifier, timestamp = session["session_id"], session["session_date"]
        date.fromisoformat(timestamp)
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Require distinct nonempty session IDs")
        seen.add(identifier)
        messages = []
        def append_chunk():
            payload = {"session_id": identifier, "date": timestamp, "messages": list(messages)}
            events.append({"id": f"h{len(events)}", "ts": len(events) + 1, "app": "conversation",
                "event": payload, "text": json.dumps(payload, ensure_ascii=False)})
        for turn, message in enumerate(session["dialogue"]):
            if message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str):
                raise ValueError("Invalid public dialogue message")
            content = message["content"]
            for offset in range(0, max(1, len(content)), 3000):
                fragment = {"role": message["role"], "content": content[offset:offset + 3000],
                    "turn_index": turn, "character_offset": offset}
                trial = {"session_id": identifier, "date": timestamp, "messages": messages + [fragment]}
                if messages and len(json.dumps(trial, ensure_ascii=False)) > 12000:
                    append_chunk()
                    messages = []
                messages.append(fragment)
        if messages or not session["dialogue"]:
            append_chunk()
    return events


def prepare_user(row, *, expected_sessions=750):
    if type(row["person_index"]) is not int or row["person_index"] < 0:
        raise ValueError("Invalid person index")
    if len(row["sessions"]) != expected_sessions:
        raise ValueError("Expected the full 1m history configuration")
    user = f"muse_{row['person_index']:04d}"
    events = history_events(row["sessions"])
    cases, seen = [], set()
    for query in row["queries"]:
        identifier = query["query_id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Require distinct query IDs per user")
        seen.add(identifier)
        case = {"case_id": identifier, "question": query["query"], "query_date": query["query_date"],
            "query_type": query["query_type"], "options": [
                {"label": option["label"], "content": option["content"]} for option in query["options"]],
            "gold_label": query["correct_label"]}
        validate_mcq(case)
        cases.append(case)
    if not cases:
        raise ValueError("User has no MCQs")
    return {"user_id": user, "person_index": row["person_index"], "events": events, "cases": cases,
        "sessions": len(row["sessions"]), "history_characters": sum(len(message["content"])
            for session in row["sessions"] for message in session["dialogue"])}


def validate_mcq(case):
    if not isinstance(case["question"], str) or not case["question"].strip():
        raise ValueError("Question must be nonempty")
    date.fromisoformat(case["query_date"])
    if case["query_type"] not in QUERY_TYPES:
        raise ValueError("Unknown MUSE query type")
    if [option["label"] for option in case["options"]] != list("ABCDE"):
        raise ValueError("Require the original five ordered A-E options")
    if any(not isinstance(option["content"], str) or not option["content"].strip() for option in case["options"]):
        raise ValueError("Empty option")
    if case["gold_label"] not in "ABCDE" or len(case["gold_label"]) != 1:
        raise ValueError("Invalid answer label")


def request(case, *, expose_query_type=False):
    validate_mcq(case)
    options = "\n".join(f"{option['label']}: {option['content']}" for option in case["options"])
    task = f" Query type: {case['query_type']}." if expose_query_type else ""
    return (f"Today is {case['query_date']}.{task}\n"
        "Use the user's supplied history only when relevant, and distinguish contradictions from unrelated preferences.\n"
        f"User query:\n{case['question']}\nOptions:\n{options}\n"
        "Return only the option letter A-E, without explanation.")


def parse_output(content):
    if not isinstance(content, str) or content.strip() not in list("ABCDE"):
        raise ValueError("Expected exactly one option letter A-E")
    return content.strip()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def construction_digest(events, max_chars=160000):
    import json
    import re

    signals = re.compile(r"\b(?:i prefer|my usual|i usually|i always|i never|i switched|i changed|"
        r"from now on|no longer|allergic|remember that|my favorite|my favourite|i need|i want|"
        r"i like|i dislike|i work|i live|my goal|my schedule)\b", re.I)
    sessions = {}
    for event in events:
        payload = event["event"]
        session = sessions.setdefault(payload["session_id"], {"date": payload["date"], "turns": {}})
        for message in payload["messages"]:
            turn = session["turns"].setdefault(message["turn_index"], {"role": message["role"], "parts": []})
            turn["parts"].append((message["character_offset"], message["content"]))
    selected = []
    for identifier, session in sorted(sessions.items(), key=lambda item: (item[1]["date"], item[0])):
        turns = [(index, turn["role"], "".join(text for offset, text in sorted(turn["parts"])))
            for index, turn in sorted(session["turns"].items())]
        candidates = [(len(signals.findall(text)), index, text) for index, role, text in turns if role == "user"]
        score, index, text = max(candidates, default=(0, -1, ""))
        match = signals.search(text)
        start = max(0, match.start() - 60) if match else 0
        response = next((content for position, role, content in turns if position > index and role == "assistant"), "")
        selected.append({"session_id": identifier, "date": session["date"],
            "user": text[start:start + (640 if score else 120)], "assistant": response[:160 if score else 60]})
    result = {"total_history_events": len(events), "total_sessions": len(sessions),
        "covered_sessions": len(selected), "selection": "All sessions scanned; date-ordered, source-only excerpts; full history remains available",
        "sessions": selected}
    rendered = json.dumps(result, ensure_ascii=False)
    while len(rendered) > max_chars:
        before = len(rendered)
        for session in selected:
            for field in ("user", "assistant"):
                session[field] = session[field][:int(len(session[field]) * 0.8)]
        rendered = json.dumps(result, ensure_ascii=False)
        if len(rendered) == before:
            raise ValueError("History preview cannot retain all session metadata within the registered budget")
    return rendered


def prepare_dataset(source, out, *, download=False, seed=SEED):
    import duckdb
    source, out = require_local_storage(source), require_local_storage(out)
    if out.exists():
        raise ValueError("Use a fresh prepared output directory")
    if download:
        from huggingface_hub import hf_hub_download
        hf_hub_download(DATASET_REPOSITORY, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION,
            local_dir=source, cache_dir=source / ".cache/hub")
    parquet = source / DATASET_FILE
    source_digest = file_sha(parquet)
    out.mkdir(parents=True)
    (out / "users").mkdir()
    users, counts, records = [], {}, {}
    with duckdb.connect() as connection:
        query = connection.execute("SELECT * FROM read_parquet(?) ORDER BY person_index", [str(parquet)])
        names = [column[0] for column in query.description]
        while values := query.fetchone():
            original = dict(zip(names, values))
            user = prepare_user(original)
            identifier = user["user_id"]
            if identifier in counts:
                raise ValueError("Duplicate persona index")
            users.append(identifier)
            filename = out / "users" / f"{identifier}.json"
            with filename.open("x") as stream:
                json.dump(user, stream, ensure_ascii=False)
            records[str(filename.relative_to(out))] = file_sha(filename)
            dates = [session["session_date"] for session in original["sessions"]]
            counts[identifier] = {"queries": len(user["cases"]), "sessions": user["sessions"],
                "empty_sessions": sum(not session["dialogue"] for session in original["sessions"]),
                "chunks": len(user["events"]), "history_characters": user["history_characters"],
                "event_text_characters": sum(len(event["text"]) for event in user["events"]),
                "query_types": dict(Counter(case["query_type"] for case in user["cases"])),
                "chronological_sessions": dates == sorted(dates),
                "queries_before_latest_session": sum(case["query_date"] < max(dates) for case in user["cases"])}
    split = roles(users, seed=seed)
    summary = {"dataset": "muse", "protocol": PROTOCOL, "configuration": "1m", "source_split": "test",
        "source": {"repository": DATASET_REPOSITORY, "revision": DATASET_REVISION,
            "file": DATASET_FILE, "sha256": source_digest, "license": "CC-BY-4.0"},
        "objective": OBJECTIVE, "query_types": list(QUERY_TYPES), "users": counts,
        "split": {role: {"users": len(split[role]), "queries": sum(counts[user]["queries"] for user in split[role]),
            "sessions": sum(counts[user]["sessions"] for user in split[role]),
            "history_characters": sum(counts[user]["history_characters"] for user in split[role])}
            for role in ("train", "test")},
        "history_policy": "All released 1m sessions retained losslessly, per official visible-input protocol; no prefix truncation",
        "visibility": "sessions and query/date/type/options public; person profile, correct_label and answer_session_ids private",
        "research_disclosure": "User-requested 50/50 research split of an evaluation-only benchmark; not an official training split"}
    for name, value in (("roles.json", split), ("summary.json", summary)):
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
        records[name] = file_sha(out / name)
    with (out / "registration.json").open("x") as stream:
        json.dump({"files": records}, stream, indent=2)
    return summary
