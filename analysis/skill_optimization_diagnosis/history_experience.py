"""Build fresh no-skill S0 experience from strictly earlier history predicting later history."""
import ast
import asyncio
from collections import Counter
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE = load("history_experience_phase", HERE / "phase_b.py")
SEMANTIC = load("history_semantic_retrieval", HERE / "semantic_experience.py")
PILOT = PHASE.PILOT
read, write, sha, now = PILOT.read, PILOT.write, PILOT.sha, PHASE.now


def windows(events, user, adapter):
    ordered = sorted(events, key=lambda event: (date.fromisoformat(json.loads(event["text"])["date"]), event["ts"]))
    dates = [json.loads(event["text"])["date"] for event in ordered]
    boundaries = [index for index in range(1, len(ordered)) if dates[index - 1] < dates[index]]
    if not boundaries:
        raise ValueError("History needs at least two distinct dates for causal prediction")
    starts = sorted({boundaries[0], *[min(boundaries, key=lambda index: (abs(index - len(ordered) * fraction), index))
        for fraction in (1 / 3, 2 / 3)]})
    cases = adapter.history_cases(ordered, user)
    stages = []
    for ordinal, (start, end) in enumerate(zip(starts, starts[1:] + [len(ordered)]), start=1):
        rows = [case for case in cases if start <= case["history_cutoff"] < end]
        if rows:
            stages.append({"arm": f"s0_h{ordinal}", "construction_cutoff": start, "end_exclusive": end,
                "rows": rows, "construction_last_date": dates[start - 1], "first_prediction_date": rows[0]["date"]})
    if not stages or sum(len(stage["rows"]) for stage in stages) != len(cases):
        raise ValueError("Causal windows must cover every eligible later-history case exactly once")
    return ordered, stages


def assert_causal(events, stage, row):
    cutoff = row["history_cutoff"]
    if not stage["construction_cutoff"] <= cutoff < stage["end_exclusive"]:
        raise ValueError("Query outside its registered S0 history window")
    if any(json.loads(event["text"])["date"] >= row["date"] for event in events[:cutoff]):
        raise ValueError("Same-day or future history would leak into the prediction")
    if json.loads(events[stage["construction_cutoff"] - 1]["text"])["date"] >= row["date"]:
        raise ValueError("S0 construction history would leak into the prediction")


def stage_for(plan, user, arm):
    return next(stage for stage in plan["history_windows"][user] if stage["arm"] == arm)


def evaluation_users(plan):
    return plan["test_users"] if plan.get("evaluation_split") == "test" else plan["validation_users"]


CONSTRUCTION_PROMPT = PHASE.BASE_PROMPT.replace("Do not inspect other directories", "Do not inspect outside this workspace") + (
    " No skill is installed or invoked. Construct S0 independently from the supplied history; no existing personalized "
    "program, scored feedback or future history is available. The current history is the only user-specific evidence. "
    "Subsequent observations can enter the runtime memory, so distinguish historical evidence from permanent rules."
)
EXPERIENCE_PROMPT = (
    " Also read experience.json and its linked case files. These are fresh S0 programs constructed without skills from "
    "other TRAIN users' earlier history, with actual predictions on strictly later history. They are observational "
    "success/failure cases, not optimized code revisions or universally validated rules. Inspect successes, mistakes, "
    "the visible history cutoff and executor context to understand applicability and failure modes. Never transfer "
    "donor labels, preferences, cue lists or answers as this user's facts. Use this user's own history to decide which "
    "mechanisms fit and to derive all parameters. State which evidence was considered, adopted or rejected in "
    "CHANGE_MANIFEST. All S0 history predictions were recorded once; no skill, distilled guide or feedback tuning was used."
)


def register(source, out, repetitions=2, retrieval="qwen3_embedding", *, reuse_training_from=None, repair_memory_key=False):
    if out.exists() or repetitions < 2:
        raise ValueError("Fresh directory and at least two validation repetitions required")
    if retrieval not in ("qwen3_embedding", "history_statistics"):
        raise ValueError("Unknown experience retrieval method")
    PILOT.verify(source)
    previous = read(source / "plan.json")
    if previous["dataset"] != "OPPU/movie_tagging" or previous["executor"]["model"] != "Qwen3.5-9B":
        raise ValueError("Expected the registered movie cohort and Qwen executor")
    donors, validation = previous["donor_users"], previous["transfer_users"]
    if set(donors) & set(validation) or set(donors + validation) & set(previous.get("test_users", [])):
        raise ValueError("TRAIN, validation and reserved TEST users must be disjoint")
    adapter, _ = PILOT.host(source)
    histories, stages, source_hashes = {}, {}, {"plan.json": sha(source / "plan.json")}
    for user in donors + validation:
        path = source / "private/users" / f"{user}.json"
        private = read(path)
        if len(private["events"]) <= 50:
            raise ValueError("Cohort users must have more than 50 usable history records")
        source_hashes[str(path.relative_to(source))] = sha(path)
        if user in donors:
            ordered, stages[user] = windows(private["events"], user, adapter)
            histories[user] = {"user_id": user, "events": ordered, "rows": []}
        else:
            histories[user] = private
    out.mkdir(parents=True)
    shutil.copytree(source / "source_snapshot", out / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    for name in PHASE.DRIVERS:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, out / name)
    for name in ("phase_b.py", "local_to_skill.py", "history_experience.py", "semantic_experience.py"):
        shutil.copyfile(HERE / name, out / name)
    construction_files = {}
    for user, private in histories.items():
        write(out / "private/users" / f"{user}.json", private)
        for stage in stages.get(user, []):
            path = f"private/construction/{stage['arm']}/{user}.json"
            write(out / path, {"events": private["events"][:stage["construction_cutoff"]]})
            construction_files.setdefault(stage["arm"], {})[user] = path
    train_arms = sorted({stage["arm"] for values in stages.values() for stage in values})
    validation_arms = {f"{role}_r{iteration}": role for iteration in range(1, repetitions + 1) for role in ("no_bank", "context")}
    history_queries = sum(len(stage["rows"]) for values in stages.values() for stage in values)
    validation_queries = len(validation_arms) * sum(len(histories[user]["rows"]) for user in validation)
    generation_jobs = sum(len(values) for values in stages.values()) + len(validation) * len(validation_arms)
    calls = 1 + sum(previous["hook_model_calls"].values())
    plan = {key: previous[key] for key in ("dataset", "seed", "allowed_tags", "answer_max_tokens", "hook_model_calls",
        "executor", "optimizer", "optimizer_model", "generation_timeout_seconds")}
    plan.update(protocol="fresh-no-skill-s0-causal-history-experience-v1", created_utc=now(),
        data_source=str(source), source_files=source_hashes, donor_users=donors, validation_users=validation,
        users=donors + validation, test_users_reserved=previous.get("test_users", []),
        arms=train_arms + list(validation_arms), arm_roles={**{arm: "no_bank" for arm in train_arms}, **validation_arms},
        validation_arms=list(validation_arms), repetitions=repetitions, minimum_gain_pp=2.0,
        construction_history_files=construction_files, history_windows=stages,
        construction_prompt=CONSTRUCTION_PROMPT, experience_prompt=EXPERIENCE_PROMPT,
        skill_policy="No skill generation, optimization, installation or invocation at any stage.",
        source_policy="Reuse only raw own-history/native-validation inputs and runtime interfaces. Do not import old "
            "harnesses, predictions, DEV/AUDIT results, experience banks, skills or generation workspaces.",
        history_policy="Sort by original date and event order. Use day boundaries near one third and two thirds plus "
            "the first available past-day prefix. Construct a fresh S0 per distinct window using only the prefix. "
            "Score all eligible unique later-history descriptions once; each query sees only strictly earlier dates. "
            "Past gold observations may enter memory after their day finishes; S0 code stays fixed within its window. "
            "Same-day and repeated-description targets are not scored as independent future cases.",
        retrieval={"cases": 5, "per_donor": 1, "minimum_predictions": 1,
            "selection": "Closest prefix-history statistics, standardized over donors; not selected by accuracy."},
        sol_slots=2, score_workers=2,
        comparison="Fresh history-experience vs fresh no-experience construction on identical users, questions and "
            "generation-session/inference limits, two repetitions. Require mean user gain >=2pp, positive one-sided "
            "95% user-bootstrap lower bound, positive gain in each repetition and no validation runtime violations. "
            "No best-draw selection or significance-based early stopping; context token counts differ.",
        caveat="Previously inspected validation cohort, not an independent test. Historical predictions describe "
            "S0 behavior and cannot by themselves prove that a mechanism causes transfer gains.",
        failure_policy="Retain every attempted generation and prediction; no automatic retry, fallback or candidate replacement.",
        upper_bounds={"sol_sessions": generation_jobs, "generation_jobs": generation_jobs,
            "history_predictions": history_queries, "validation_predictions": validation_queries,
            "query_attempts": history_queries + validation_queries, "executor_calls_per_query": calls,
            "executor_calls": (history_queries + validation_queries) * calls})
    if retrieval == "qwen3_embedding":
        plan["retrieval"] = SEMANTIC.policy()
        plan["retrieval"]["embedding_input_texts"] = history_queries + sum(len(histories[user]["events"]) for user in validation)
        plan["protocol"] = "fresh-no-skill-s0-causal-history-semantic-v2"
    if reuse_training_from is not None:
        reuse = reuse_training(reuse_training_from, out, plan, repair_memory_key=repair_memory_key)
        plan["training_reuse"] = {"reference": str(reuse_training_from), "receipt": "training_reuse.json"}
        plan["source_policy"] += " Reuse exact compatible fresh no-skill S0 history artifacts from the named predecessor; "
        plan["source_policy"] += "retain any explicitly recorded operational key repair and original failure, never relabel it as Sol-generated."
        write(out / "remaining_budget.json", {"original_upper_bounds": plan["upper_bounds"],
            "new_sol_sessions": generation_jobs - reuse["harnesses"],
            "new_query_attempts": history_queries + validation_queries - reuse["predictions"],
            "max_new_executor_calls": (history_queries + validation_queries - reuse["predictions"]) * calls})
    write(out / "plan.json", plan)
    write(out / "registration.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print(json.dumps({"out": str(out), "training_users": len(donors), "s0_windows": sum(map(len, stages.values())),
        "budget": plan["upper_bounds"]}), flush=True)
    return plan


def normalize_memory_label_key(source):
    hook = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "remember")
    text = ast.get_source_segment(source, hook)
    if text.count('"key": tag,') != 1:
        raise ValueError("The operational repair only supports the observed label-as-key defect")
    replacement = text.replace('"key": tag,', '"key": tag.replace(" ", "_"),')
    return source[:source.index(text)] + replacement + source[source.index(text) + len(text):]


def reuse_training(reference, out, plan, *, repair_memory_key=False):
    PILOT.verify(reference)
    previous = read(reference / "plan.json")
    if previous["protocol"] != "fresh-no-skill-s0-causal-history-experience-v1":
        raise ValueError("Only fresh no-skill history training can be reused")
    if read(reference / "operations/status.json")["phase"] not in ("completed", "attention_required"):
        raise ValueError("Training reference must not be running")
    for key in ("donor_users", "validation_users", "history_windows", "seed", "executor", "optimizer",
            "hook_model_calls", "construction_prompt", "generation_timeout_seconds"):
        if previous[key] != plan[key]:
            raise ValueError("Incompatible training reuse: " + key)
    for path in (out / "source_snapshot").rglob("*"):
        if path.is_file() and sha(path) != sha(reference / path.relative_to(out)):
            raise ValueError("Runtime changed across training reuse")
    for name in PHASE.DRIVERS:
        if sha(out / name) != sha(reference / name):
            raise ValueError("Frozen driver changed across training reuse: " + name)
    receipt = {"reference": str(reference), "files": {}, "harnesses": 0, "predictions": 0, "repairs": []}
    adapter, _ = PILOT.host(out)
    from ohpu.experiments import pm3_official as official
    def copy(path, relative=None):
        relative = relative or path.relative_to(reference)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        receipt["files"][str(relative)] = {"source": str(path), "sha256": sha(path)}
    for user in plan["donor_users"]:
        if sha(reference / "private/users" / f"{user}.json") != sha(out / "private/users" / f"{user}.json"):
            raise ValueError("Raw history changed across training reuse")
        events = read(out / "private/users" / f"{user}.json")["events"]
        for stage in plan["history_windows"][user]:
            arm = stage["arm"]
            original = reference / "harness" / arm / user
            if not (original / "harness.json").exists():
                if original.exists():
                    raise ValueError("Unfinished source generation needs explicit recovery")
                continue
            harness = read(original / "harness.json")
            if not harness["generation"]["valid"]:
                raise ValueError("Invalid generation cannot be replaced silently")
            score_dir = reference / "scores" / arm / user
            repair = repair_memory_key and user == "8000905" and arm == "s0_h1"
            if repair:
                if (score_dir / "records.jsonl").exists() or (score_dir / "attempts.jsonl").exists():
                    raise ValueError("Key repair requires no scored or attempted predictions for this panel")
                source = normalize_memory_label_key(harness["source"])
                cutoffs = sorted({row["history_cutoff"] for row in stage["rows"]})
                prefix = stage["construction_cutoff"]
                before = official.documents(user, events[:prefix], [prefix], source=harness["source"])[prefix]
                documents = official.documents(user, events[:max(cutoffs)], cutoffs, source=source)
                from ohpu.per_user_adaptation.memory_backend import MemoryStore
                assert MemoryStore.payload(before)["records"] == MemoryStore.payload(documents[prefix])["records"]
                adapter.validate_harness(source, events[:prefix], [], hook_model_calls=plan["hook_model_calls"])
                record = {"user": user, "arm": arm, "before_sha256": hashlib.sha256(harness["source"].encode()).hexdigest(),
                    "after_sha256": hashlib.sha256(source.encode()).hexdigest(), "history_cutoffs_validated": cutoffs,
                    "model_calls": 0, "change": "remember key: replace spaces in tag with underscores; all other code unchanged",
                    "provenance": "Operational repair, not a new Sol-generated candidate; original source and failure retained."}
                receipt["repairs"].append(record)
                for path in original.glob("*.json"):
                    copy(path, Path("operations/original_key_failure") / path.name)
                log = reference / "operations" / f"score-{arm}-{user}.log"
                copy(log, Path("operations/original_key_failure/score.log"))
                write(out / "harness" / arm / user / "harness.json", {"source": source,
                    "generation": {**harness["generation"], "operational_repair": record}})
                write(out / "harness" / arm / user / "attempt.json", {"reused_original_session": str(original), "operational_repair": record})
            else:
                for path in original.glob("*.json"):
                    copy(path)
                if (score_dir / "summary.json").exists():
                    rows = list(PILOT.records(reference, arm, user).values())
                    if [row["query_id"] for row in rows] != [row["query_id"] for row in stage["rows"]]:
                        raise ValueError("Source history score ledger is incomplete")
                    for path in score_dir.glob("*"):
                        if path.is_file():
                            copy(path)
                    receipt["predictions"] += len(rows)
                elif score_dir.exists() and list(score_dir.iterdir()):
                    raise ValueError("Partial history scoring requires explicit recovery")
            receipt["harnesses"] += 1
    if repair_memory_key and len(receipt["repairs"]) != 1:
        raise ValueError("Requested operational key repair did not match exactly one known failure")
    write(out / "training_reuse.json", receipt)
    return receipt


def score_history(out, arm, user, client=None):
    plan = read(out / "plan.json")
    if user not in plan["donor_users"]:
        raise ValueError("History experience can only use TRAIN users")
    stage = stage_for(plan, user, arm)
    directory = out / "scores" / arm / user
    if (directory / "summary.json").exists():
        return
    if (directory / "errors.jsonl").exists():
        raise RuntimeError("Prior failed history prediction requires review; no retry")
    harness = read(out / "harness" / arm / user / "harness.json")
    if not harness["generation"]["valid"]:
        raise ValueError("Invalid S0 cannot produce scored experience")
    events = read(out / "private/users" / f"{user}.json")["events"]
    adapter, native = PILOT.host(out)
    comparison = PILOT.load(out / "executor_swap.py", "history_prediction_executor")
    from ohpu.experiments import pm3_official as official
    if client is None:
        client = native.executor(out)
        client.max_retries = 1
    client = comparison.RefusalClient(client)
    query = comparison.refusal_aware_query(adapter.run_query)
    directory.mkdir(parents=True, exist_ok=True)
    ledger, attempts = directory / "records.jsonl", directory / "attempts.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    expected = [row["query_id"] for row in stage["rows"]]
    if [row["query_id"] for row in records] != expected[:len(records)]:
        raise ValueError("History prediction ledger is not the registered prefix")
    attempted = {json.loads(line)["query_id"] for line in attempts.read_text().splitlines()} if attempts.exists() else set()
    pending = stage["rows"][len(records):]
    for row in pending:
        assert_causal(events, stage, row)
    cutoffs = [row["history_cutoff"] for row in pending]
    documents = official.documents(user, events[:max(cutoffs)], cutoffs, source=harness["source"]) if cutoffs else {}
    for row in pending:
        if row["query_id"] in attempted:
            raise RuntimeError("Unresolved history attempt requires review; no retry")
        with attempts.open("a") as stream:
            stream.write(json.dumps({"query_id": row["query_id"], "started_utc": now()}) + "\n")
        try:
            record = query(user, row, harness["source"], documents[row["history_cutoff"]], client, plan)
        except Exception as error:
            with (directory / "errors.jsonl").open("a") as stream:
                stream.write(json.dumps({"query_id": row["query_id"], "error": f"{type(error).__name__}: {error}"}) + "\n")
            raise
        prior = Counter(json.loads(event["text"])["tag"] for event in events[:row["history_cutoff"]])
        majority = min(prior, key=lambda tag: (-prior[tag], tag))
        record.update(request=row["request"], observed_tag=row["target"], date=row["date"],
            history_cutoff=row["history_cutoff"], construction_cutoff=stage["construction_cutoff"],
            construction_history_sha256=sha(out / plan["construction_history_files"][arm][user]),
            majority_tag_before=majority, majority_accuracy=100.0 * (majority == row["target"]),
            observed_tag_count_before=prior[row["target"]])
        with ledger.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
    write(directory / "summary.json", {"execution_complete": True, "n_queries": len(records),
        "accuracy_pct_micro": statistics.fmean(row["accuracy"] for row in records),
        "majority_accuracy_pct": statistics.fmean(row["majority_accuracy"] for row in records),
        "runtime_violations": sum(len(row["runtime_violations"]) for row in records)})


def verify_bank(out):
    registration = read(out / "experience/registration.json")
    frozen = {**registration["source_prediction_files"], **registration["files"]}
    if any(sha(out / name) != digest for name, digest in frozen.items()):
        raise ValueError("Frozen history experience changed")


def inherit_evaluation(out):
    plan = read(out / "plan.json")
    receipt_path = out / "operations/inherited_evaluation.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if any(sha(out / name) != item["sha256"] for name, item in receipt["files"].items()):
            raise ValueError("Inherited TEST scores changed")
        return receipt
    reference = Path(plan["inherit_evaluation"]["reference"])
    PILOT.verify(reference)
    verify_bank(reference)
    if read(reference / "operations/status.json")["phase"] != "completed":
        raise ValueError("Previous cohort must finish before TEST result inheritance; do not duplicate its pending work")
    original = read(reference / "plan.json")
    for key in ("retrieval", "construction_prompt", "experience_prompt", "hook_model_calls", "preserve_hooks", "seed",
            "executor", "allowed_tags", "answer_max_tokens", "hook_framework", "history_summary", "repetitions"):
        if plan[key] != original[key]:
            raise ValueError("Changed method cannot inherit TEST scores: " + key)
    candidates = []
    for user in plan["inherit_evaluation"]["users"]:
        if sha(out / "private/users" / f"{user}.json") != sha(reference / "private/users" / f"{user}.json"):
            raise ValueError("Inherited TEST history/questions differ")
        expected = [row["query_id"] for row in read(out / "private/users" / f"{user}.json")["rows"]]
        for arm in plan["validation_arms"]:
            harness_dir = reference / "harness" / arm / user
            score_dir = reference / "scores" / arm / user
            if not read(harness_dir / "harness.json")["generation"]["valid"] or not read(score_dir / "summary.json")["execution_complete"]:
                raise ValueError("Cannot inherit invalid or incomplete TEST panels")
            records = list(PILOT.records(reference, arm, user).values())
            if [row["query_id"] for row in records] != expected:
                raise ValueError("Inherited TEST ledger differs")
            if any(row["runtime_violations"] or row["executor_calls"] != 1 for row in records):
                raise ValueError("Inherited TEST runtime/call budget differs")
            PHASE.preserve_hooks((out / "inputs/scaffold.py").read_text(), read(harness_dir / "harness.json")["source"], plan["preserve_hooks"])
            if plan["arm_roles"][arm] == "context" and sha(out / "retrieval" / arm / f"{user}.json") != sha(reference / "retrieval" / arm / f"{user}.json"):
                raise ValueError("Inherited TEST retrieval differs")
            candidates.extend(path for path in harness_dir.glob("*.json"))
            candidates.extend(path for path in score_dir.glob("*") if path.is_file())
    receipt = {"utc": now(), "reference": str(reference), "users": plan["inherit_evaluation"]["users"], "files": {}}
    for path in candidates:
        relative = path.relative_to(reference)
        target = out / relative
        if target.exists():
            raise ValueError("Incomplete inheritance requires review; no overwriting existing artifacts")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        receipt["files"][str(relative)] = {"source": str(path), "sha256": sha(path)}
    write(receipt_path, receipt)
    return receipt


def retrieve(bank, target, count):
    best = {}
    for entry in bank["entries"]:
        distance = PHASE.distance(bank, target, entry["history_signature"])
        rank = (distance, -entry["performance"]["queries"], entry["entry_id"])
        if entry["donor"] not in best or rank < best[entry["donor"]][0]:
            best[entry["donor"]] = (rank, entry)
    selected = sorted(best.values(), key=lambda value: value[0])[:count]
    if len(selected) != count:
        raise ValueError("Insufficient distinct TRAIN donors for frozen retrieval")
    return {"experience": {"origin": bank["origin"], "target_history_signature": target,
        "interpretation": "Cases show actual no-skill S0 outcomes, not revision gains. A success or mistake alone does "
            "not establish a general rule; compare actual prior evidence and applicability to the target history.",
        "cases": [entry for _, entry in selected]},
        "audit": {"selected_entries": [entry["entry_id"] for _, entry in selected],
            "distances": [rank[0] for rank, _ in selected], "selection_uses_accuracy": False}}


def build_bank(out, embedder=None):
    if (out / "experience/registration.json").exists():
        verify_bank(out)
        return read(out / "experience/bank.json")
    if (out / "experience").exists():
        raise RuntimeError("Unfinished bank construction requires review")
    plan = read(out / "plan.json")
    donors = sorted(plan["donor_users"], key=lambda user: hashlib.sha256(user.encode()).hexdigest())
    aliases = {user: f"donor-{index:02d}" for index, user in enumerate(donors, start=1)}
    identifiers = sorted(set(plan["users"] + plan["test_users_reserved"]), key=len, reverse=True)
    def sanitize(value):
        return json.loads(PHASE._redact(json.dumps(value, ensure_ascii=False), identifiers))
    entries, assets, hashes, profiles = [], {}, {}, []
    for user in donors:
        events = read(out / "private/users" / f"{user}.json")["events"]
        profiles.append({"alias": aliases[user], "signature": PHASE.signature(events)})
        for stage in plan["history_windows"][user]:
            arm = stage["arm"]
            harness_path = out / "harness" / arm / user / "harness.json"
            ledger_path = out / "scores" / arm / user / "records.jsonl"
            harness = read(harness_path)
            summary = read(ledger_path.parent / "summary.json")
            records = list(PILOT.records(out, arm, user).values())
            if (not harness["generation"]["valid"] or not summary["execution_complete"] or
                    [row["query_id"] for row in records] != [row["query_id"] for row in stage["rows"]]):
                raise ValueError("Cannot freeze incomplete or invalid S0 experience")
            for path in (harness_path, ledger_path, ledger_path.parent / "summary.json"):
                hashes[str(path.relative_to(out))] = sha(path)
            entry_id = f"{aliases[user]}-{arm}"
            prefix = f"experience_cases/{entry_id}"
            paths = {"s0_program": prefix + "/s0.py", "construction_history": prefix + "/construction_history.jsonl",
                "observation_timeline": prefix + "/observation_timeline.jsonl", "history_predictions": prefix + "/predictions.json"}
            assets[paths["s0_program"]] = PHASE._redact(harness["source"], identifiers)
            for key, end in (("construction_history", stage["construction_cutoff"]), ("observation_timeline", stage["end_exclusive"])):
                assets[paths[key]] = "".join(json.dumps(sanitize(event), ensure_ascii=False) + "\n" for event in events[:end])
            assets[paths["history_predictions"]] = json.dumps(sanitize(records), ensure_ascii=False, indent=2) + "\n"
            successes = [row for row in records if row["accuracy"] == 100]
            failures = [row for row in records if row["accuracy"] != 100]
            confusions = Counter((row["observed_tag"], row["predicted_tag"]) for row in failures)
            entries.append({"entry_id": entry_id, "donor": aliases[user], "kind": "fresh_no_skill_s0_history_prediction",
                "history_signature": PHASE.signature(events[:stage["construction_cutoff"]]), "files": paths,
                "construction": {"empty_scaffold": True, "skill": None, "prior_experience": None,
                    "history_events": stage["construction_cutoff"], "last_date": stage["construction_last_date"],
                    "program_fixed_within_window": True},
                "performance": {"queries": len(records), "correct": len(successes), "incorrect": len(failures),
                    "accuracy_pct": summary["accuracy_pct_micro"], "majority_accuracy_pct": summary["majority_accuracy_pct"],
                    "runtime_violations": summary["runtime_violations"],
                    "executor_calls": sum(row["executor_calls"] for row in records)},
                "confusions": [{"observed": gold, "predicted": predicted, "count": count}
                    for (gold, predicted), count in confusions.most_common()],
                "orientation_examples": [sanitize({key: row.get(key) for key in ("query_id", "request", "observed_tag",
                    "predicted_tag", "accuracy", "date", "history_cutoff", "observed_tag_count_before", "runtime_violations")})
                    for group in (successes, failures) for row in group[:2]],
                "author_design_not_validated": sanitize(PHASE._assignments(harness["source"]).get("CHANGE_MANIFEST", {})),
                "evidence_scope": "Every eligible prediction is retained once. observation_timeline includes hindsight labels "
                    "for reconstruction only; each prediction saw events[:history_cutoff], with dates strictly before its date. "
                    "S0 generation saw only construction_history. Raw accuracy is not a causal mechanism gain."})
    values = {feature: [profile["signature"][feature] for profile in profiles] for feature in PHASE.FEATURES}
    bank = {"version": "fresh-no-skill-s0-history-experience-v1", "origin": "TRAIN user history only; fresh empty-scaffold "
        "no-skill S0 programs evaluated prospectively on later history. No native DEV/AUDIT/VALID/TEST feedback or old candidates.",
        "feature_names": list(PHASE.FEATURES), "feature_mean": {key: statistics.fmean(items) for key, items in values.items()},
        "feature_std": {key: statistics.pstdev(items) or 1.0 for key, items in values.items()},
        "donors": profiles, "entries": entries, "total_history_predictions": sum(entry["performance"]["queries"] for entry in entries)}
    write(out / "experience/attempt.json", {"started_utc": now(), "source_prediction_files": hashes})
    write(out / "experience/bank.json", bank)
    for name, content in assets.items():
        path = out / "inputs/context_assets" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(content)
    semantic = plan["retrieval"].get("method") == "memoharness_qwen3_embedding_v1"
    semantic_files = []
    if semantic:
        import numpy as np
        embedder = embedder or SEMANTIC.QwenEmbeddings(plan["retrieval"])
        cases, vectors = SEMANTIC.build_index(bank, assets, embedder)
        matrix_path = out / "experience/embedding_vectors.npy"
        with matrix_path.open("xb") as stream:
            np.save(stream, vectors, allow_pickle=False)
        index_path = out / "experience/embedding_index.json"
        write(index_path, {"policy": plan["retrieval"], "bank_sha256": sha(out / "experience/bank.json"),
            "matrix_sha256": sha(matrix_path), "shape": list(vectors.shape), "cases": cases,
            "document_text_sha256": hashlib.sha256(json.dumps([case["request"] for case in cases]).encode()).hexdigest()})
        semantic_files = [matrix_path, index_path]
    for user in plan["validation_users"]:
        events = read(out / "private/users" / f"{user}.json")["events"]
        if semantic:
            retrieved = SEMANTIC.retrieve(bank, cases, vectors, events, embedder, plan["retrieval"])
            retrieved["audit"]["index_sha256"] = sha(index_path)
        else:
            retrieved = retrieve(bank, PHASE.signature(events), plan["retrieval"]["cases"])
        for arm in plan["validation_arms"]:
            if plan["arm_roles"][arm] == "context":
                write(out / "retrieval" / arm / f"{user}.json", retrieved)
    if semantic:
        telemetry = out / "experience/embedding_usage.json"
        write(telemetry, {"configuration": plan["retrieval"], "usage": getattr(embedder, "calls", {}),
            "completed_utc": now(), "indexed_predictions": len(cases), "target_users": len(plan["validation_users"])})
        semantic_files.append(telemetry)
    frozen = [out / "experience/bank.json", *semantic_files, *[out / "inputs/context_assets" / name for name in assets],
        *[path for path in (out / "retrieval").rglob("*.json")]]
    write(out / "experience/registration.json", {"created_utc": now(), "source_prediction_files": hashes,
        "files": {str(path.relative_to(out)): sha(path) for path in frozen}})
    return bank


def generate(out, arm, user, invoke=None):
    plan = read(out / "plan.json")
    if user in plan["donor_users"]:
        stage_for(plan, user, arm)
        assert plan["arm_roles"][arm] == "no_bank" and user in plan["construction_history_files"][arm]
    elif user in evaluation_users(plan) and arm in plan["validation_arms"]:
        verify_bank(out)
    else:
        raise ValueError("Unregistered no-skill construction")
    PHASE.generate(out, arm, user, invoke=invoke)
    bundle = out / "harness" / arm / user / "workspace"
    if (bundle / ".agents").exists() or list(bundle.rglob("SKILL.md")):
        raise ValueError("Skills are forbidden in history-experience generation")


def score(out, arm, user, client=None):
    plan = read(out / "plan.json")
    if user in plan["donor_users"]:
        score_history(out, arm, user, client=client)
    elif user in evaluation_users(plan) and arm in plan["validation_arms"]:
        verify_bank(out)
        PHASE.score(out, arm, user, client=client)
    else:
        raise ValueError("Unregistered scoring panel")


def summarize(out):
    plan = read(out / "plan.json")
    verify_bank(out)
    users = evaluation_users(plan)
    panels, violations, calls, call_mismatches = {}, {}, {}, 0
    for arm in plan["validation_arms"]:
        panels[arm] = {}
        violations[arm] = calls[arm] = 0
        for user in users:
            rows = list(PILOT.records(out, arm, user).values())
            expected = [row["query_id"] for row in read(out / "private/users" / f"{user}.json")["rows"]]
            if [row["query_id"] for row in rows] != expected or not read(out / "scores" / arm / user / "summary.json")["execution_complete"]:
                raise ValueError("Incomplete validation comparison")
            if not read(out / "harness" / arm / user / "harness.json")["generation"]["valid"]:
                raise ValueError("Invalid validation generation")
            panels[arm][user] = statistics.fmean(row["accuracy"] for row in rows)
            violations[arm] += sum(len(row["runtime_violations"]) for row in rows)
            calls[arm] += sum(row["executor_calls"] for row in rows)
            if plan.get("expected_calls_per_prediction"):
                call_mismatches += sum(row["executor_calls"] != plan["expected_calls_per_prediction"] for row in rows)
    means = {role: {user: statistics.fmean(panels[f"{role}_r{iteration}"][user] for iteration in range(1, plan["repetitions"] + 1))
        for user in users} for role in ("no_bank", "context")}
    gains = {user: means["context"][user] - means["no_bank"][user] for user in users}
    repeated = {str(iteration): statistics.fmean(panels[f"context_r{iteration}"][user] - panels[f"no_bank_r{iteration}"][user]
        for user in users) for iteration in range(1, plan["repetitions"] + 1)}
    paired = {**PILOT.paired(list(gains.values())), "per_user": gains, "repeat_mean_gain": repeated}
    supported = (paired["mean_gain"] >= plan["minimum_gain_pp"] and paired["lower_bound_95"] > 0 and
        all(value > 0 for value in repeated.values()) and not any(violations.values()) and not call_mismatches)
    bank = read(out / "experience/bank.json")
    result = {"conclusion": "history_experience_supported_on_development" if supported else "history_experience_not_confirmed",
        "user_mean_accuracy": {role: statistics.fmean(values.values()) for role, values in means.items()},
        "comparison": paired, "per_repetition_accuracy": panels, "runtime_violations": violations,
        "executor_calls": calls, "model_call_mismatches": call_mismatches, "experience_cases": len(bank["entries"]), "history_predictions": bank["total_history_predictions"],
        "caveat": plan["caveat"], "completed_utc": now()}
    if plan.get("evaluation_split") == "test":
        result.update(conclusion="history_experience_supported_on_combined_test" if supported else "history_experience_not_confirmed",
            evaluation_split="test", test_user_count=len(users), prior_roles=plan["prior_roles"],
            independent_unseen_test=False, test_user_ids=users)
    write(out / "summary.json", result)
    return result


async def run(out, execute=None):
    PILOT.verify(out)
    plan = read(out / "plan.json")
    operation = out / "operations"
    operation.mkdir(exist_ok=True)
    slots = {"generate": asyncio.Semaphore(plan["sol_slots"]), "score": asyncio.Semaphore(plan["score_workers"])}
    failures = []
    def status(phase):
        (operation / "status.json").write_text(json.dumps({"phase": phase, "utc": now()}))
        print(phase, flush=True)
    async def subprocess(action, arm, user):
        with (operation / f"{action}-{arm}-{user}.log").open("a") as stream:
            process = await asyncio.create_subprocess_exec(sys.executable, "-I", "-B", str(out / "history_experience.py"),
                "job", "--out", str(out), "--action", action, "--arm", arm, "--user", user,
                stdout=stream, stderr=asyncio.subprocess.STDOUT)
            if await process.wait():
                raise RuntimeError(f"Job failed: {action}/{arm}/{user}")
    async def job(action, arm, user):
        async with slots[action]:
            if failures:
                return False
            try:
                await (execute or subprocess)(action, arm, user)
                return True
            except Exception as error:
                failures.append(str(error))
                return False
    async def training(user):
        for stage in plan["history_windows"][user]:
            if not (await job("generate", stage["arm"], user) and await job("score", stage["arm"], user)):
                return
    async def validation(index, user):
        for iteration in range(1, plan["repetitions"] + 1):
            order = ("no_bank", "context") if (index + iteration) % 2 else ("context", "no_bank")
            for role in order:
                arm = f"{role}_r{iteration}"
                if not (await job("generate", arm, user) and await job("score", arm, user)):
                    return
    try:
        if (out / "summary.json").exists():
            status("completed")
            return
        if plan.get("inherit_evaluation"):
            status("inherit_completed_evaluation")
            inherit_evaluation(out)
        if plan.get("execution_mode") != "frozen_bank_validation":
            status("fresh_s0_history_predictions")
            await asyncio.gather(*(training(user) for user in plan["donor_users"]))
            if failures:
                raise RuntimeError(str(failures))
            status("freeze_history_experience")
            build_bank(out)
        else:
            verify_bank(out)
        status("test_no_skill" if plan.get("evaluation_split") == "test" else "validation_no_skill")
        await asyncio.gather(*(validation(index, user) for index, user in enumerate(evaluation_users(plan))))
        if failures:
            raise RuntimeError(str(failures))
        summarize(out)
        status("completed")
    except Exception:
        status("attention_required")
        raise
