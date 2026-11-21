"""English Mem-PAL solution selection: 50 TRAIN and 50 TEST users."""
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location("mempal_shared_adapter", HERE / "muse_phg.py")
MUSE = importlib.util.module_from_spec(specification)
specification.loader.exec_module(MUSE)
PHG = MUSE.PHG
read, write, sha, now = PHG.read, PHG.write, PHG.sha, PHG.now
BASELINES = ()
PROTOCOL = "mempal-en-selection-hyperharness-train50-test50-v1"
REPLICATED_PROTOCOL = "mempal-en-selection-hyperharness-independent-replicates-v2"
SINGLE_PASS_PROTOCOL = "mempal-en-selection-hyperharness-single-pass-significance-v3"
API = """# Mem-PAL English solution-selection runtime
Generate one complete restricted Python harness from this user's history logs and dialogues only.
Construction receives the released history partition, never query topics, requirements, options, labels,
synthetic profiles, background traits or hidden needs. All construction sessions are available through tools.
Runtime follows official sample-order availability: history plus earlier query dialogues and current logs.
The current/future query dialogues and candidate feedback are never visible before answering.
Some original log timestamps are later than their sample date; retain official declared availability and
do not reinterpret those timestamps as permission to access a later sample. English is an upstream LLM translation.
The public request contains the detailed requirement and eight S1-S8 options, as in official solution selection.
The final executor additionally sees current activity logs. Retrieval receives
the requirement and options, with current logs searchable in memory. Runtime task_kind is None.
Choose exactly two distinct solutions. Positive selections score +1, negative -1, neutral 0, then multiply by50.
The primary objective is equal-user mean signed score [-100,100]; query-micro official score is also reported.
Do not confuse this with accuracy. Malformed outputs receive -100; runtime failures invalidate a user panel.
The neutral initial program is uniform BM25 top10, max20000 evidence characters, without guidance.
All personal code, config and helpers are editable. remember may add source-bound derived records but must
retain all original observations unchanged. recall uses bounded BM25/SQL and finish(output, source_refs).
apply can return only a prefix of at most1500characters; guard must return {} and cannot rewrite outputs.
There is no host-side automatic completion and no intermediate model call. Only the final host-owned Qwen call
returns JSON with analysis (brief English explanation) and selected_solutions (two distinct S1-S8 strings).
Budgets:32768inputtokens,512outputtokens,64000read-state chars,8000write-state chars,20000evidence chars,
64source_refs,32memory operations,20000interpreter steps per hook. Model and scoring budgets are host-owned.
Raw observations are JSON with kind, session_id, date, content, and role/turn for dialogue fragments.
Use actual retrieved record refs, not session IDs or guessed observation ordinals. No future/other-user data.
Ground factual rules in observed sources. Preserve qualifiers, contexts, negations and corrections. Empty
guidance or no intervention is valid when unsupported. Option text is public, but option-position heuristics,
copied TRAIN answers and hard-coded selected_solutions are forbidden. Only TRAIN feedback guides optimization.
"""


def host(out):
    native, official = MUSE.BASE.host(out)
    native.PERSONALIZED_FRAMEWORK = API + "\n" + PHG.FULL_PROGRAM_CONTRACT
    return native, official


def initial_spec(directory):
    from ohpu.benchmarks.mempal import construction_digest
    MUSE.initial_spec(directory, personalization="combined")
    contract = API + "\n" + PHG.FULL_PROGRAM_CONTRACT
    prompt = PHG.full_program_prompt(contract,
        "Use this user's source history to learn useful request-specific retrieval and evidence-grounded guidance. "
        "Treat all eight candidate texts symmetrically, distinguish accepted and rejected suggestions from dialogue, "
        "and preserve the neutral BM25 control. Observations are English logs and dialogues; no labels enter construction.")
    (directory / "prompt.md").write_text(prompt)
    (directory / "program_api.md").write_text(contract)
    (directory / "history_adapter.py").write_text("import json\n\n" + inspect.getsource(construction_digest)
        + "\n\ndef digest(events):\n    return construction_digest(events)\n")


def program_check(out, user, events, client=None, *, cached=False):
    from ohpu.benchmarks.mempal_runtime import validate_harness
    def check(source):
        try:
            validate_harness(source, events)
        except Exception as error:
            return f"{type(error).__name__}: {error}"[:3000]
        return None
    return check


def framework(plan, native):
    replication = (" Each user is independently generated and evaluated twice. Scores average those two draws, "
        "not two copies of one program. Adoption requires a complete cohort, improvement in both replicates, "
        "and a positive lower 95% user-paired bootstrap bound versus the parent. Inspect replicate results in "
        "diagnostics. All candidate specification changes must come from this optimizer session; the host does "
        "not install a fixed retrieval strategy or transform your output program. The whole generator remains editable."
        if plan["phg"].get("replicates", 1) > 1 else "")
    single_pass = (" Each candidate is generated and evaluated exactly once per TRAIN user. Do not request repeats, "
        "replace failed slots, alter r0, change the metric or confidence level, or select a convenient subset. "
        "Adoption requires all users, a positive mean versus the current parent, and a positive lower bound of the "
        "two-sided 95% user-paired bootstrap interval versus the fixed original r0. Inspect user_risk.json and "
        "r0_cases.json in every round's TRAIN diagnostics: improve broad user benefit and reduce large regressions "
        "instead of relying on a few extreme gains. Distinguish retrieval losses, misleading personal guidance and "
        "runtime defects from the saved evidence, guidance, source programs and cases; choose and implement your own "
        "generator changes. All candidate code must come from this optimizer, not a host-authored runtime transform. "
        "The unchanged 50-user TEST is not available for optimization or reselection."
        if plan["phg"].get("selection_policy") == "significant_vs_r0" else "")
    return PHG.optimizer_framework(plan, task_api=API + "\n" + PHG.FULL_PROGRAM_CONTRACT,
        task_context="Mem-PAL English eight-option, two-selection task. Optimize equal-user signed score, "
            "not five-way accuracy. Full program editing is allowed, source history and host limits are fixed. "
            "Read runtime failures and paired TRAIN regressions. No TEST feedback or per-user answer copying." + replication + single_pass)


def replica_context(out, index, *, version=None, user=None):
    plan = read(out / "plan.json")
    if not 0 <= index < plan["phg"].get("replicates", 1):
        raise ValueError("Unregistered generation replicate")
    root = out / "replicates" / str(index)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (root / "plan.json").exists():
            replica = copy.deepcopy(plan)
            replica["seed"] += index
            replica["phg"]["replicates"] = 1
            replica["replicate_index"] = index
            write(root / "plan.json", replica)
        if version is not None and not (root / "generator" / f"v{version}.json").exists():
            spec, receipt = PHG.accepted_spec(out, version)
            shutil.copytree(spec, root / "generator" / f"v{version}")
            write(root / "generator" / f"v{version}.json", receipt)
        if user is not None and not (root / "private/native" / (user + ".json")).exists():
            target = root / "private/native" / (user + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / "private/native" / (user + ".json"), target)
    return root


def tool_script(out, function, *arguments):
    return ("import importlib.util\nimport sys\nfrom pathlib import Path\n"
        f"spec=importlib.util.spec_from_file_location('mempal_tools',{str(out / 'mempal_phg.py')!r})\n"
        "driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)\n"
        f"driver.{function}(Path({str(out)!r}),Path('.').resolve()," + ", ".join(arguments) + ")\n")


def generate(out, arm, user):
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["phg"]["arms"]:
        raise ValueError("Unregistered generation")
    if plan["split"] == "TEST" and not (out / "selection.json").exists():
        raise ValueError("TEST requires frozen TRAIN selection")
    if plan["phg"].get("replicates", 1) > 1:
        return generate_replicates(out, arm, user)
    return PHG.generate_harness(out, arm, user, version=plan["phg"]["arms"][arm],
        events=PHG.construction_events(out, plan, user), check_factory=program_check, cached=False)


def generate_replicates(out, arm, user):
    plan = read(out / "plan.json")
    directory = out / "harness" / arm / user
    if (directory / "harness.json").exists():
        if not read(directory / "harness.json")["generation"]["valid"]:
            raise ValueError("Prior replicated generation failed; no replacement")
        return
    directory.mkdir(parents=True, exist_ok=False)
    version = plan["phg"]["arms"][arm]
    events = PHG.construction_events(out, plan, user)
    write(directory / "started.json", {"user": user, "arm": arm, "version": version,
        "replicates": plan["phg"]["replicates"], "utc": now()})
    receipts = []
    error = None
    try:
        for index in range(plan["phg"]["replicates"]):
            root = replica_context(out, index, version=version, user=user)
            try:
                PHG.generate_harness(root, arm, user, version=version, events=events,
                    check_factory=program_check, cached=False)
            finally:
                receipt = root / "harness" / arm / user / "harness.json"
                if receipt.exists():
                    receipts.append(read(receipt))
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    usages = [row["generation"].get("usage", {}) for row in receipts]
    usage = {key: sum(row.get(key, 0) or 0 for row in usages) for key in (usages[0] if usages else {})}
    write(directory / "harness.json", {"source": receipts[0]["source"] if receipts else "",
        "replica_sources": [row["source"] for row in receipts],
        "generation": {"valid": error is None and len(receipts) == plan["phg"]["replicates"],
            "error": error, "version": version, "generator_version": version,
            "replicates": plan["phg"]["replicates"], "completed_replicates": len(receipts),
            "usage": usage, "seconds": sum(row["generation"].get("seconds", 0) for row in receipts)}})
    if error:
        raise RuntimeError(error)


def score(out, arm, user, *, client=None, tokenizer=None):
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["phg"]["arms"]:
        raise ValueError("Scoring outside registered cohort")
    if plan["phg"].get("replicates", 1) == 1:
        return _score_once(out, arm, user, client=client, tokenizer=tokenizer)
    from ohpu.benchmarks.mempal_runtime import summarize_user
    directory = out / "scores" / arm / user
    if (directory / "summary.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=False)
    panels = []
    try:
        for index in range(plan["phg"]["replicates"]):
            root = replica_context(out, index, user=user)
            _score_once(root, arm, user, client=client, tokenizer=tokenizer)
            panels.append(PHG.records(root / "scores" / arm / user / "records.jsonl"))
        expected = [case["case_id"] for case in read(out / "private/native" / (user + ".json"))["cases"]]
        if any([row["query_id"] for row in panel] != expected for panel in panels):
            raise ValueError("Replicated evaluation query mismatch")
        rows = []
        for values in zip(*panels):
            value = statistics.fmean(row["selection_score"] for row in values)
            rows.append({**values[0], "accuracy": value, "selection_score": value,
                "executor_calls": sum(row["executor_calls"] for row in values),
                "replica_results": list(values), "replica_predictions": [row["prediction"] for row in values]})
        with (directory / "records.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        native = read(out / "private/native" / (user + ".json"))
        summary = summarize_user(native, rows)
        summary["invalid_outputs"] = sum(row["format_error"] is not None for panel in panels for row in panel)
        write(directory / "summary.json", {**summary, "user": user, "arm": arm, "utc": now(),
            "replicates": len(panels), "replicate_means": [statistics.fmean(row["selection_score"] for row in panel) for panel in panels]})
    except Exception as error:
        if not (directory / "failure.json").exists():
            write(directory / "failure.json", {"user": user, "arm": arm, "stage": "replicated_score",
                "error": f"{type(error).__name__}: {error}", "completed_replicates": len(panels), "utc": now()})
        raise


def _score_once(out, arm, user, *, client=None, tokenizer=None):
    from ohpu.benchmarks.mempal_runtime import QwenClient, answer_user, summarize_user
    from ohpu.benchmarks.perma_runtime import failure_details
    plan = read(out / "plan.json")
    directory = out / "scores" / arm / user
    if (directory / "summary.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=False)
    try:
        harness = read(out / "harness" / arm / user / "harness.json")
        if not harness["generation"]["valid"]:
            raise ValueError("Invalid personal harness")
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer_model"], revision=plan["tokenizer_revision"], cache_dir=plan["model_cache"])
        client = client or QwenClient(base_url=plan["executor_url"], model=plan["executor_model"])
        native = read(out / "private/native" / (user + ".json"))
        with (directory / "records.jsonl").open("x") as stream:
            def sink(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            def error_sink(error):
                write(directory / "failure.json", {**error, "arm": arm, "utc": now()})
            rows = answer_user(native, source=harness["source"], client=client, tokenizer=tokenizer,
                seed=plan["seed"], input_tokens=plan["input_tokens"], output_tokens=plan["output_tokens"], sink=sink, error_sink=error_sink)
    except Exception as error:
        if not (directory / "failure.json").exists():
            write(directory / "failure.json", {**failure_details(error, user=user, stage="score"), "arm": arm, "utc": now()})
        raise
    write(directory / "summary.json", {**summarize_user(native, rows), "user": user, "arm": arm, "utc": now()})


def diagnose(out, version):
    plan = read(out / "plan.json")
    if plan["split"] != "TRAIN":
        raise ValueError("Diagnostics are restricted to TRAIN")
    arm = PHG.round_arm(plan, version)
    initial_arm = PHG.round_arm(plan, 0)
    target = out / "diagnostics" / f"round{version}"
    if (target / "done.json").exists():
        return
    (target / "programs").mkdir(parents=True)
    parent_arm = PHG.diagnostic_parent(out, plan, version)
    cases, comparisons, failures = [], [], []
    r0_cases = []
    single_pass = plan["phg"].get("selection_policy") == "significant_vs_r0"
    for user in plan["test_users"]:
        def panel(name):
            path = out / "scores" / name / user / "records.jsonl" if name else None
            return {row["query_id"]: row for row in PHG.records(path)} if path and path.exists() else {}
        failure = out / "scores" / arm / user / "failure.json"
        if failure.exists():
            failures.append({key: value for key, value in read(failure).items() if key != "response"})
        rows = list(panel(arm).values())
        initial = panel(initial_arm) if version else {}
        feedback = PHG.diagnostic_feedback(rows, initial, panel(parent_arm), parent_arm=parent_arm)
        cases.extend(feedback["cases"])
        comparisons.extend(feedback["parent_cases"])
        if single_pass:
            initial_feedback = PHG.diagnostic_feedback(rows, {}, initial, parent_arm=initial_arm)
            r0_cases.extend(initial_feedback["parent_cases"])
        harness = out / "harness" / arm / user / "harness.json"
        if harness.exists() and read(harness).get("source"):
            receipt = read(harness)
            (target / "programs" / (user + ".py")).write_text(receipt["source"])
            for index, source in enumerate(receipt.get("replica_sources", [])[1:], start=1):
                (target / "programs" / (user + f"_replicate{index}.py")).write_text(source)
    write(target / "cases.json", cases)
    write(target / "parent_cases.json", comparisons)
    write(target / "runtime_failures.json", failures)
    if single_pass:
        current = PHG.arm_scores(out, plan, arm)
        initial_scores = PHG.arm_scores(out, plan, initial_arm)
        deltas = sorted(({"user": user, "r0": initial_scores[user], "candidate": current[user],
                          "delta": current[user] - initial_scores[user]} for user in current.keys() & initial_scores.keys()),
                        key=lambda row: (row["delta"], row["user"]))
        denominator = len(plan["test_users"])
        positive = [row["delta"] for row in deltas if row["delta"] > 0]
        negative = [row["delta"] for row in deltas if row["delta"] < 0]
        write(target / "user_risk.json", {"split": "TRAIN", "reference": "fixed original r0",
            "registered_users": denominator, "paired_users": len(deltas), "complete": len(deltas) == denominator,
            "missing_candidate_users": sorted(set(plan["test_users"]) - set(current)),
            "wins": len(positive), "losses": len(negative), "ties": len(deltas) - len(positive) - len(negative),
            "positive_contribution": sum(positive) / denominator,
            "negative_contribution": sum(negative) / denominator,
            "top_five_positive_contribution": sum(sorted(positive, reverse=True)[:5]) / denominator,
            "users_below_minus_10": [row["user"] for row in deltas if row["delta"] < -10],
            "users": deltas, "note": "Diagnostic only; no user exclusion, metric reweighting or TEST feedback"})
        write(target / "r0_cases.json", r0_cases)
    PHG.diagnostics_summary(out)
    write(target / "done.json", {"utc": now(), "cases": len(cases)})


def summarize(out):
    from ohpu.benchmarks.mempal import OBJECTIVE
    plan = read(out / "plan.json")
    values = {arm: PHG.arm_scores(out, plan, arm) for arm in plan["phg"]["arms"]}
    selected = "test_optimized" if plan["split"] == "TEST" else PHG.round_arm(plan, read(out / "selection.json")["version"])
    result = {"dataset": "mempal", "language": "en", "task": "solution_selection", "split": plan["split"],
        "objective": OBJECTIVE, "selection": read(out / "selection.json"), "utc": now(),
        "arms": {}, "comparisons": {}}
    for arm, scores in values.items():
        rows = [row for user in scores for row in PHG.records(out / "scores" / arm / user / "records.jsonl")]
        complete = set(scores) == set(plan["test_users"])
        result["arms"][arm] = {"users": len(scores), "complete": complete,
            "mean": statistics.fmean(scores.values()) if complete else None,
            "available_user_mean": statistics.fmean(scores.values()) if scores else None,
            "official_query_micro_score": statistics.fmean(row["selection_score"] for row in rows) if complete else None,
            "invalid_outputs": sum(row["format_error"] is not None for row in rows)}
        if arm != selected and scores and values[selected]:
            result["comparisons"][arm] = PHG.paired(values[selected], scores, selected + "_minus_" + arm)
    write(out / "summary.json", result)
    return result


def register(out, prepared, host_source, *, iterations=10, reference=None, model_cache="/opt/model-apis/huggingface", replicates=1,
             selection_policy=None):
    from ohpu.benchmarks import mempal as data
    from ohpu.benchmarks.mempal_runtime import CONTRACT_LIMITS
    out, prepared = data.require_local_storage(out), data.require_local_storage(prepared)
    if out.exists() or not 1 <= iterations <= 10:
        raise ValueError("Require a fresh registration and 1-10 optimizer iterations")
    if replicates not in (1, 2):
        raise ValueError("Register one or two independent generation replicates")
    if selection_policy not in (None, "mean_gain", "significant_vs_r0"):
        raise ValueError("Unknown selection policy")
    for name, expected in read(prepared / "registration.json")["files"].items():
        if sha(prepared / name) != expected:
            raise ValueError("Prepared data changed")
    summary, roles = read(prepared / "summary.json"), read(prepared / "roles.json")
    if summary["protocol"] != data.PROTOCOL or roles != data.roles(list(summary["users"]), seed=roles["seed"]):
        raise ValueError("Prepared split or protocol changed")
    previous = selected = None
    if reference:
        PHG.BASE.HISTORY.PILOT.verify(reference)
        previous, selected = read(reference / "plan.json"), read(reference / "selection.json")
        replicates = previous["phg"].get("replicates", 1)
        inherited_policy = previous["phg"].get("selection_policy", "mean_gain")
        if selection_policy is not None and selection_policy != inherited_policy:
            raise ValueError("TEST must inherit the TRAIN selection policy")
        selection_policy = inherited_policy
        if (previous["protocol"] not in (PROTOCOL, REPLICATED_PROTOCOL, SINGLE_PASS_PROTOCOL) or previous["split"] != "TRAIN"
                or previous["prepared_registration_sha256"] != sha(prepared / "registration.json")
                or set(PHG.arm_scores(reference, previous, PHG.round_arm(previous, selected["version"]))) != set(roles["train"])):
            raise ValueError("TEST requires a matching complete frozen TRAIN selection")
    selection_policy = selection_policy or "mean_gain"
    if selection_policy == "significant_vs_r0" and replicates != 1:
        raise ValueError("Significant-vs-r0 policy requires exactly one generation per user")
    users = roles["test"] if reference else roles["train"]
    out.mkdir(parents=True)
    shutil.copytree(reference / "source_snapshot/src" if reference else HERE.parent.parent / "src", out / "source_snapshot/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in PHG.DRV.HOST_FILES:
        shutil.copyfile((reference or host_source) / name, out / name)
    for name in (*PHG.DRV.CODE_FILES, "pm3_decision_harness.py", "pm3_generator_harness.py", "horizon_phg.py",
            "memorycd_phg.py", "perma_phg.py", "muse_phg.py", "mempal_phg.py"):
        shutil.copyfile((reference or HERE) / name, out / name)
    if reference:
        for version in sorted({0, selected["version"]}):
            directory, receipt = PHG.accepted_spec(reference, version)
            shutil.copytree(directory, out / "generator" / f"v{version}")
            write(out / "generator" / f"v{version}.json", receipt)
        write(out / "selection.json", selected)
        write(out / "rounds/parent.json", {"version": selected["version"], "history": []})
    else:
        initial_spec(out / "generator/v0")
        PHG.check_spec(out / "generator/v0", open_contract=True, require_model_code=True, allow_decision_tools=False)
        write(out / "generator/v0.json", {"version": 0, "parent": None, "outcome": "accepted", "error": None,
            "spec_sha256": PHG.spec_sha(out / "generator/v0"), "source": "fresh English Mem-PAL full-program generator"})
    (out / "inputs").mkdir()
    shutil.copyfile(out / "generator/v0/control_program.py", out / "inputs/shared_control.py")
    splits = {}
    for user in users:
        native = read(prepared / "users" / (user + ".json"))
        write(out / "private/native" / (user + ".json"), native)
        write(out / "private/users" / (user + ".json"), {"events": native["events"], "rows": []})
        splits[user] = {"construction_cutoff": native["construction_cutoff"], "evaluated": [case["case_id"] for case in native["cases"]]}
    methods = {"test_initial": 0, "test_optimized": selected["version"]} if reference else {f"phg_r{version}": version for version in range(iterations + 1)}
    requests = sum(len(value["evaluated"]) for value in splits.values())
    plan = {"protocol": PROTOCOL, "framework": "HyperHarness", "dataset": "mempal", "language": "en",
        "task": "solution_selection", "created_utc": now(), "split": "TEST" if reference else "TRAIN",
        "seed": roles["seed"], "host_limit_scale": 1, "test_users": users, "heldout_users": [] if reference else roles["test"],
        "splits": splits, "personalization": "combined", "program_edit_scope": PHG.PROGRAM_EDIT_SCOPE,
        "host_contract": dict(CONTRACT_LIMITS), "evidence_chars": 20000, "task_type_visibility": "hidden",
        "retrieval": {"backend": "uniform-bm25-top10-v1", "initial_control_sha256": sha(out / "inputs/shared_control.py")},
        "prepared": str(prepared), "prepared_registration_sha256": sha(prepared / "registration.json"), "source": summary["source"],
        "history_policy": summary["history_policy"], "research_disclosure": summary["research_disclosure"],
        "arms": list(methods), "objective": data.OBJECTIVE,
        "executor_model": "Qwen3.5-9B", "executor_url": "http://127.0.0.1:8001/v1",
        "tokenizer_model": "Qwen/Qwen3.5-9B", "tokenizer_revision": MUSE.BASE.BASE.TOKENIZER_REVISION,
        "model_cache": model_cache, "input_tokens": 32768, "output_tokens": 512, "score_workers": 2,
        "optimizer_model": "gpt-6-sol", "optimizer": {"credential_file": PHG.GENERATOR["credential_file"], "api_version": "2025-04-01-preview"},
        "external_baselines": {"arms": [], "files": {}},
        "phg": {"rounds": 0 if reference else iterations + 1, "arms": methods, "luna_workers": 2,
            "replicates": replicates, "selection_policy": selection_policy,
            "require_model_generated_code": True, "open_generation_contract": True, "decision_model_enabled": False,
            "decision_policy": "disabled", "generator": {**PHG.GENERATOR, "request_timeout_seconds": 600, "transport_retries": 1,
                "retries": "one recorded transient transport retry; failed candidate slots never replayed"},
            "limits": {**PHG.LIMITS, "wall_seconds": 7800}, "dev_users": users[:2],
            "optimizer": {**PHG.OPTIMIZER, "model": "gpt-6-sol", "sessions": 0 if reference else iterations,
                "reasoning_effort": "medium", "luna_relay_calls": None, "decision_relay_calls": 0}},
        "decision_model": {"calls_per_request": 0, "arms": [], "host_limits": {}},
        "study_design": {"baseline_groups": {}, "method_arms": list(methods)},
        "selection_rule": "Complete TRAIN panel; strictly positive paired equal-user signed-score gain; ties retain parent; no TEST selection",
        "failure_policy": "No executor retries; incomplete/runtime failures invalidate panels; malformed choices score -100; failures retained",
        "concurrency": {"generation": 2, "executor": 2, "optimizer": 1},
        "monitoring": {"interval_seconds": 30, "stalled_seconds": 6000},
        "upper_bounds": {"personal_programs": len(users) * len(methods) * replicates, "sol_sessions": 0 if reference else iterations,
            "executor_requests": requests * len(methods) * replicates, "requests_per_arm": requests},
        "scheduled_test": {"users": len(roles["test"]), "personal_programs": 2 * len(roles["test"]) * replicates,
            "executor_requests": 2 * summary["split"]["test"]["queries"] * replicates,
            "sol_sessions": 0, "selection_feedback": False},
        "queue": {"maximum_train_chains": 2}}
    if reference:
        for key in ("host_contract", "program_edit_scope", "executor_model", "executor_url", "tokenizer_model", "tokenizer_revision",
                "model_cache", "input_tokens", "output_tokens", "concurrency", "score_workers", "optimizer"):
            plan[key] = copy.deepcopy(previous[key])
        for key in ("generator", "limits"):
            plan["phg"][key] = copy.deepcopy(previous["phg"][key])
    if replicates > 1:
        plan["protocol"] = REPLICATED_PROTOCOL
        plan["selection_rule"] = "Complete replicated TRAIN panel; positive paired mean at both independent generations and lower 95% user bootstrap bound above zero; no TEST selection"
        plan["optimization_origin"] = {"candidate_author": "outer gpt-6-sol optimizer only", "assistant_runtime_transform": False,
            "initial_generator": "unchanged original Mem-PAL r0", "replicate_generation": "independent model calls and programs",
            "replicate_executor_seed": "registered seed plus replicate index"}
    if selection_policy == "significant_vs_r0":
        plan["protocol"] = SINGLE_PASS_PROTOCOL
        plan["selection_rule"] = "Complete single-pass TRAIN panel; positive mean versus parent and lower two-sided 95% user bootstrap bound above zero versus fixed r0; no TEST selection"
        plan["optimization_origin"] = {"candidate_author": "outer gpt-6-sol optimizer only", "assistant_runtime_transform": False,
            "initial_generator": "unchanged original Mem-PAL r0", "replicate_generation": "one program and one evaluation per user per candidate"}
    write(out / "plan.json", plan)
    write(out / "registration.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    return plan


async def run(out, execute=None):
    plan = read(out / "plan.json")
    await PHG.run_pipeline(out, execute=execute, entrypoint="mempal_phg.py", score_stages=("score",),
        score_workers={"score": plan["score_workers"]}, baseline_arms=BASELINES, control_arms=(),
        diagnose_round=diagnose, summary=summarize, heldout_users=plan["test_users"] if plan["split"] == "TEST" else (),
        heldout_arms=("test_initial", "test_optimized") if plan["split"] == "TEST" else ())


ordinary_round_summary = PHG.round_summary


def single_pass_round_summary(out, version):
    plan = read(out / "plan.json")
    if plan["split"] != "TRAIN" or plan["phg"].get("replicates", 1) != 1:
        raise ValueError("Single-pass selection requires TRAIN and exactly one generation per user")
    arm = PHG.round_arm(plan, version)
    path = out / "rounds/parent.json"
    parent = read(path) if path.exists() else {"version": None, "history": []}
    scores = PHG.arm_scores(out, plan, arm)
    complete = set(scores) == set(plan["test_users"])
    comparison, versus_r0 = {}, {}
    if complete and parent["version"] is not None:
        previous = PHG.round_arm(plan, parent["version"])
        initial = PHG.round_arm(plan, 0)
        reference = PHG.arm_scores(out, plan, previous)
        baseline = PHG.arm_scores(out, plan, initial)
        if set(reference) != set(plan["test_users"]) or set(baseline) != set(plan["test_users"]):
            raise ValueError("Selection requires complete parent and unchanged r0 panels")
        comparison = PHG.paired(scores, reference, arm + "_minus_" + previous)
        versus_r0 = PHG.paired(scores, baseline, arm + "_minus_" + initial)
    significant = bool(versus_r0 and versus_r0["ci95_pp"][0] > 0)
    adopted = complete and (parent["version"] is None or comparison["mean_pp"] > 0 and significant)
    decision = {"version": version, "parent": parent["version"], "evaluated_users": len(scores),
        "complete": complete, "adopted": adopted, "mean_versus_parent": comparison.get("mean_pp"),
        "comparison": comparison, "versus_r0": versus_r0, "significant_vs_r0": significant,
        "selection_scope": "TRAIN only; adaptive selection is not confirmatory TEST evidence"}
    result = {"version": version, "arm": arm, "users": scores,
        "user_mean": statistics.fmean(scores.values()) if complete else None,
        "parent_decision": decision, "generation": PHG.generation_stats(out, plan, arm),
        "receipt": read(out / "generator" / f"v{version}.json"),
        "comparisons": {"parent": comparison, "r0": versus_r0}}
    write(out / "rounds" / f"round{version}.json", result)
    if adopted:
        parent["version"] = version
    parent["history"].append(decision)
    PHG.replace(path, parent)
    if parent["version"] is None:
        raise RuntimeError("Initial single-pass panel is incomplete")
    return result


def round_summary(out, version):
    plan = read(out / "plan.json")
    if plan["phg"].get("selection_policy") == "significant_vs_r0":
        return single_pass_round_summary(out, version)
    if plan["phg"].get("replicates", 1) == 1:
        return ordinary_round_summary(out, version)
    if plan["split"] != "TRAIN":
        raise ValueError("Selection is restricted to TRAIN")
    arm = PHG.round_arm(plan, version)
    path = out / "rounds/parent.json"
    parent = read(path) if path.exists() else {"version": None, "history": []}
    scores = PHG.arm_scores(out, plan, arm)
    complete = set(scores) == set(plan["test_users"])
    comparison, replicate_deltas = {}, []
    if complete and parent["version"] is not None:
        previous = PHG.round_arm(plan, parent["version"])
        reference = PHG.arm_scores(out, plan, previous)
        if set(reference) != set(plan["test_users"]):
            raise ValueError("Selected parent lost its complete panel")
        comparison = PHG.paired(scores, reference, arm + "_minus_" + previous)
        for index in range(plan["phg"]["replicates"]):
            replicate_deltas.append(statistics.fmean(read(out / "scores" / arm / user / "summary.json")["replicate_means"][index]
                - read(out / "scores" / previous / user / "summary.json")["replicate_means"][index] for user in plan["test_users"]))
    adopted = complete and (parent["version"] is None or min(replicate_deltas) > 0 and comparison["ci95_pp"][0] > 0)
    decision = {"version": version, "parent": parent["version"], "evaluated_users": len(scores),
        "complete": complete, "adopted": adopted, "mean_versus_parent": comparison.get("mean_pp"),
        "replicate_deltas": replicate_deltas, "comparison": comparison}
    result = {"version": version, "arm": arm, "users": scores,
        "user_mean": statistics.fmean(scores.values()) if complete else None,
        "parent_decision": decision, "generation": PHG.generation_stats(out, plan, arm),
        "receipt": read(out / "generator" / f"v{version}.json"), "comparisons": {"parent": comparison}}
    write(out / "rounds" / f"round{version}.json", result)
    if adopted:
        parent["version"] = version
    parent["history"].append(decision)
    PHG.replace(path, parent)
    if parent["version"] is None:
        raise RuntimeError("Initial replicated panel is incomplete")
    return result


PHG.BASE.host = host
PHG.program_check = program_check
PHG.framework = framework
PHG.tool_script = tool_script
PHG.REFERENCES = BASELINES
PHG.round_summary = round_summary
check_generator_workspace = PHG.check_generator_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "register-test", "preflight", "run", "study", "job"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--host-source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--test-out", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--replicates", type=int, choices=(1, 2), default=1)
    parser.add_argument("--selection-policy", choices=("mean_gain", "significant_vs_r0"))
    parser.add_argument("--action", choices=("generate", "score", "optimize"))
    parser.add_argument("--arm")
    parser.add_argument("--user")
    args = parser.parse_args()
    os.environ["OHPU_HOST_LIMIT_SCALE"] = "1"
    out = args.out.resolve()
    if args.command in ("register", "register-test"):
        if args.command == "register-test" and not args.reference:
            raise ValueError("TEST requires frozen TRAIN reference")
        plan = register(out, args.prepared.resolve(), args.host_source.resolve(), iterations=args.iterations,
            reference=args.reference.resolve() if args.reference else None, replicates=args.replicates,
            selection_policy=args.selection_policy)
        print(json.dumps({"out": str(out), "split": plan["split"], "bounds": plan["upper_bounds"]}))
        return
    host(out)
    PHG.BASE.HISTORY.PILOT.verify(out)
    if args.command == "preflight":
        MUSE.BASE.preflight(out)
        return
    if args.command in ("run", "study"):
        if args.command == "study" and (not args.test_out or args.test_out.exists() or read(out / "plan.json")["split"] != "TRAIN"):
            raise ValueError("Study requires TRAIN and fresh TEST destination")
        MUSE.BASE.preflight(out)
        PHG.replace(out / "operations/controller.json", {"pid": os.getpid(), "utc": now(), "command": args.command})
        asyncio.run(run(out))
        if args.command == "study":
            plan = read(out / "plan.json")
            target = args.test_out.resolve()
            register(target, Path(plan["prepared"]), out, reference=out)
            subprocess.run([sys.executable, "-I", "-B", str(target / "mempal_phg.py"), "run", "--out", str(target)], check=True)
            write(out / "study_summary.json", {"train": read(out / "summary.json"), "test": read(target / "summary.json"), "utc": now()})
        return
    plan = read(out / "plan.json")
    if args.action == "generate":
        PHG.load_generator_credentials(plan)
        generate(out, args.arm, args.user)
    elif args.action == "score":
        score(out, args.arm, args.user)
    elif args.action == "optimize":
        if plan["split"] != "TRAIN" or args.arm not in plan["phg"]["arms"]:
            raise ValueError("Optimizer restricted to TRAIN")
        PHG.MECHANISM.load_optimizer_credentials(plan)
        PHG.optimize(out, plan["phg"]["arms"][args.arm])
    else:
        raise ValueError("job requires an action")


if __name__ == "__main__":
    main()
