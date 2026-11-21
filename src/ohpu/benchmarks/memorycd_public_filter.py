"""Frozen MemoryCD common-sample scoring hooks, without operations or recovery code."""
from datetime import datetime, timezone
import functools
import json
import statistics
from pathlib import Path

from ohpu.benchmarks.memorycd_runtime import SPARSE_FILTER_PROTOCOL as PROTOCOL


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def records(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def registered_native(out, user):
    native = read(out / "private/native" / (user + ".json"))
    policy = read(out / "inputs/filter_exclusions.json")
    if policy["protocol"] != PROTOCOL:
        raise ValueError("Unregistered MemoryCD filter policy")
    native["filter_policy"] = PROTOCOL
    native["known_filtered_queries"] = policy["users"].get(user, [])
    return native


def score(namespace, out, arm, user, *, client=None, tokenizer=None):
    from ohpu.benchmarks.memorycd_runtime import answer_user, summarize_user
    from ohpu.benchmarks.luna_executor import LunaClient, LunaTokenizer
    host = namespace["PHG"]
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["arms"]:
        raise ValueError("Unregistered MemoryCD scoring")
    directory = out / "scores" / arm / user
    if (directory / "summary.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=False)
    source = baseline = None
    if arm in ("no_history", "bm25_10"):
        baseline = arm
    elif arm == "shared_control":
        source = (out / "inputs/shared_control.py").read_text()
    elif arm in plan.get("shared_harness", {}).get("arms", {}):
        source = host.load("shared_harness_source", out / "shared_harness.py").program_source(out, arm)
    else:
        harness = read(out / "harness" / arm / user / "harness.json")
        if not harness["generation"]["valid"]:
            raise ValueError("Cannot execute invalid harness")
        source = harness["source"]
    native = registered_native(out, user)
    retained = records(out / "reference/filter_prefixes" / arm / (user + ".jsonl"))
    write(directory / "started.json", {"utc": now(), "user": user, "arm": arm,
        "retained_records": len(retained), "protocol": PROTOCOL}, exclusive=True)
    tokenizer = tokenizer or LunaTokenizer()
    client = client or LunaClient(base_url=plan["executor_url"], model=plan["executor_model"])
    with (directory / "records.jsonl").open("x") as stream:
        def sink(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        for row in retained:
            sink(row)
        def error_sink(error):
            write(directory / "failure.json", {**error, "user": user, "arm": arm, "utc": now()})
        try:
            rows = answer_user(native, source=source, baseline=baseline, client=client, tokenizer=tokenizer,
                seed=plan["seed"], input_tokens=plan["input_tokens"], output_tokens=plan["output_tokens"],
                sink=sink, error_sink=error_sink, retained=retained)
            summary = summarize_user(native, rows)
        except Exception as error:
            if not (directory / "failure.json").exists():
                error_sink({"exception_type": type(error).__name__, "code": "score_runtime_error"})
            raise
    write(directory / "summary.json", {**summary, "user": user, "arm": arm, "utc": now()}, exclusive=True)


def comparison(out, plan, left_arm, right_arm, host):
    from ohpu.benchmarks.memorycd_runtime import common_sample_scores
    left_users = host.arm_scores(out, plan, left_arm)
    right_users = host.arm_scores(out, plan, right_arm)
    left, right, coverage, unavailable = {}, {}, {}, {}
    for user in plan["test_users"]:
        if user not in left_users or user not in right_users:
            unavailable[user] = "incomplete_user_panel"
            continue
        try:
            paired = common_sample_scores(records(out / "scores" / left_arm / user / "records.jsonl"),
                records(out / "scores" / right_arm / user / "records.jsonl"), allow_missing_tasks=True)
        except ValueError as error:
            unavailable[user] = str(error)
            continue
        left[user], right[user], coverage[user] = paired["left"], paired["right"], paired
    result = host.paired(left, right, left_arm + "_minus_" + right_arm)
    result.update(protocol=PROTOCOL, all_registered_users_comparable=not unavailable,
        common_queries=sum(row["queries"] for row in coverage.values()), unavailable_users=unavailable)
    return result, coverage, left, right


def round_summary(namespace, out, version):
    host = namespace["PHG"]
    plan = read(out / "plan.json")
    if plan["split"] != "TRAIN":
        raise ValueError("No TEST-based selection")
    arm = host.round_arm(plan, version)
    path = out / "rounds/parent.json"
    parent = read(path) if path.exists() else {"version": None, "history": []}
    scores = host.arm_scores(out, plan, arm)
    complete = set(scores) == set(plan["test_users"])
    comparisons, coverage = {}, {}
    for baseline in namespace["BASELINES"]:
        result, paired, _, _ = comparison(out, plan, arm, baseline, host)
        comparisons[arm + "_minus_" + baseline] = result
        coverage[baseline] = paired
    decision = {"version": version, "parent": parent["version"], "evaluated_users": len(scores),
        "complete": complete, "adopted": complete and parent["version"] is None, "protocol": PROTOCOL}
    if parent["version"] is not None:
        parent_arm = host.round_arm(plan, parent["version"])
        result, paired, left, right = comparison(out, plan, arm, parent_arm, host)
        comparisons[arm + "_minus_" + parent_arm] = result
        coverage[parent_arm] = paired
        decision.update(adopted=complete and host.mean_gain_accept(left, right, plan["test_users"]),
            mean_versus_parent=result.get("mean_pp"), common_queries=result["common_queries"],
            all_registered_users_comparable=result["all_registered_users_comparable"])
    summary = {"version": version, "arm": arm, "completed_utc": now(), "users": scores,
        "user_mean": statistics.fmean(scores.values()) if scores else None, "comparisons": comparisons,
        "comparison_coverage": coverage, "generation": host.generation_stats(out, plan, arm),
        "receipt": read(out / "generator" / ("v" + str(version) + ".json")), "parent_decision": decision}
    write(out / "rounds" / ("round" + str(version) + ".json"), summary, exclusive=True)
    if decision["adopted"]:
        parent["version"] = version
    parent["history"].append(decision)
    write(path, parent)
    return summary


def summarize(namespace, out):
    host = namespace["PHG"]
    plan = read(out / "plan.json")
    selected = "test_optimized" if plan["split"] == "TEST" else host.round_arm(plan, read(out / "selection.json")["version"])
    summary = {"framework": "HyperHarness", "dataset": "memorycd", "split": plan["split"],
        "utc": now(), "selection": read(out / "selection.json"), "registered_users": len(plan["test_users"]),
        "shared_baseline": host.shared_baseline_contract(), "evaluation_scope": PROTOCOL,
        "objective": "Equal-user mean of available-task means; filtered samples omitted",
        "arms": {}, "comparisons": {}, "task_comparisons": {}, "ablation_comparisons": {},
        "task_ablation_comparisons": {}, "comparison_coverage": {},
        "missingness_note": "Empty tasks are missing, not zero. Every user remains required. Only pairwise common samples/tasks support contrasts; not the strict four-task score."}
    for arm in plan["arms"]:
        scores = host.arm_scores(out, plan, arm)
        panels = [read(out / "scores" / arm / user / "summary.json") for user in scores]
        complete = set(scores) == set(plan["test_users"])
        task_values = {task: [row["task_scores"][task] for row in panels if row["task_scores"][task] is not None]
            for task in namespace["data_tasks"]}
        panel = {"users": len(scores), "complete": complete,
            "mean": statistics.fmean(scores.values()) if complete else None,
            "available_user_mean": statistics.fmean(scores.values()) if scores else None,
            "task_users": {task: len(values) for task, values in task_values.items()},
            "task_means": {task: statistics.fmean(values) if len(values) == len(plan["test_users"]) else None
                for task, values in task_values.items()},
            "available_task_means": {task: statistics.fmean(values) if values else None for task, values in task_values.items()},
            "original_queries": sum(row["original_queries"] for row in panels),
            "scored_queries": sum(row["scored_queries"] for row in panels),
            "filtered_queries": sum(row["filtered_queries"] for row in panels),
            "original_full_users": sum(row["original_evaluation_complete"] for row in panels),
            "invalid_outputs": sum(row["invalid_outputs"] for row in panels),
            "task_metrics": {}, "available_task_metrics": {}, "task_metric_users": {}}
        for task in namespace["data_tasks"]:
            metrics = [row["ranking_metrics"] if task == "item_ranking" else row["rating_metrics"]
                if task == "rating_prediction" else row["text_metrics"][task] for row in panels]
            keys = (tuple(f"{metric}@{limit}" for metric in ("ndcg", "recall") for limit in (1, 3, 5))
                if task == "item_ranking" else ("mae", "rmse") if task == "rating_prediction" else
                ("rouge1", "rougeL", "bleu1", "bleu4"))
            available = {key: [row[key] for row in metrics if row.get(key) is not None] for key in keys}
            panel["task_metric_users"][task] = {key: len(values) for key, values in available.items()}
            panel["available_task_metrics"][task] = {key: statistics.fmean(values) if values else None
                for key, values in available.items()}
            panel["task_metrics"][task] = {key: statistics.fmean(values) if len(values) == len(plan["test_users"]) else None
                for key, values in available.items()}
        panel["rating_mae"] = panel["task_metrics"]["rating_prediction"]["mae"]
        summary["arms"][arm] = panel
        if arm == selected:
            continue
        result, coverage, _, _ = comparison(out, plan, selected, arm, host)
        group = "ablation_comparisons" if arm == "shared_control" else "comparisons"
        task_group = "task_ablation_comparisons" if arm == "shared_control" else "task_comparisons"
        summary[group][arm] = result
        summary["comparison_coverage"][arm] = coverage
        summary[task_group][arm] = {task: host.paired(
            {user: row["left_tasks"][task] for user, row in coverage.items() if task in row["left_tasks"]},
            {user: row["right_tasks"][task] for user, row in coverage.items() if task in row["right_tasks"]}, selected + arm + task)
            for task in namespace["data_tasks"]}
    write(out / "summary.json", summary)
    import csv
    with (out / "results.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "arm", "users", "available_task_score", *namespace["data_tasks"], "scored_queries", "filtered_queries"])
        for arm, panel in summary["arms"].items():
            writer.writerow([plan["split"], arm, panel["users"], panel["mean"],
                *[panel["available_task_means"][task] for task in namespace["data_tasks"]], panel["scored_queries"], panel["filtered_queries"]])
    return summary


def install_hooks(namespace):
    from ohpu.benchmarks.memorycd import TASKS
    namespace["data_tasks"] = TASKS
    namespace["score"] = functools.partial(score, namespace)
    namespace["round_summary"] = functools.partial(round_summary, namespace)
    namespace["summarize"] = functools.partial(summarize, namespace)
    original_framework = namespace["framework"]
    def framework(plan, native):
        return original_framework(plan, native) + (
            "\n\nAuthoritative filter-successor rule: confirmed content-filter samples are skipped without retry or zero imputation. "
            "Every registered user must retain an evaluable sample. Empty tasks are missing, never zero. Candidate adoption requires positive equal-user, "
            "equal-available-task gain on the candidate/parent common successful query IDs for every registered TRAIN user. "
            "Available-sample marginal means are not the adoption criterion. Non-filter errors invalidate panels. "
            "Do not edit host masks, filter classifications, scoring or cohort. TEST never selects versions.\n")
    namespace["framework"] = framework
    namespace["PHG"].round_summary = namespace["round_summary"]
    namespace["PHG"].framework = framework
