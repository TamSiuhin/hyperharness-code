"""Train and evaluate HyperHarness on causal user-disjoint MemoryCD."""
from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location("memorycd_shared_adapter", HERE / "horizon_phg.py")
BASE = importlib.util.module_from_spec(specification)
specification.loader.exec_module(BASE)
PHG = BASE.PHG
read, write, sha, now = PHG.read, PHG.write, PHG.sha, PHG.now
BASELINES = ()
PROTOCOL = "memorycd-hyperharness-causal-four-task-v8"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
API = """# MemoryCD HyperHarness contract
The generator sees only this user's Books/Electronics/Beauty history strictly before the first held-out Home
interaction. No target-domain history, evaluation requests, candidate lists, target reviews, ratings or labels
are construction inputs. One restricted Python program handles item_ranking, rating_prediction,
review_summarization and review_generation.
remember/recall/apply use the local source-bound memory store; no runtime intermediate model/decision calls.
Each observation contains timestamp/domain/product(item_id,title,category)/rating/review_title/review.
Read-state capacity is64000serialized characters; write-state remains8000. Keep retrieval state compact.
Final memory output is limited to12000characters and64source_refs, independently of read-state capacity.
recall receives stage=context, task_kind, query, and backend(step/state/result). The query is the entire legal
task input: ALL candidates for ranking; public product information for rating; product, supplied rating and
review body for summarization; product and supplied rating for review generation. The title is hidden in
both text tasks, and the target body is hidden in generation. Each task resets memory to construction history;
no target outcomes pass between tasks. BM25, read-only SQL and finish(output,source_refs) are available. apply can return only
{'prefix': guidance}. guard must return {}. The host owns the final Qwen call and strict JSON validation.
Ranking output: {"ranking":[five distinct allowed candidate IDs, most relevant first]}; rating: {"rating":integer1to5}.
Summarization output: {"title":"3-10 word title"}, at most160characters; generation: {"review":"review body"},
at most3000characters. Complete the JSON within the token budget; do not pad or repeat phrases.
The objective averages FOUR equal-weight task scores per user: NDCG@5, 1-MAE/4, summarization ROUGE-L,
generation ROUGE-L, all scaled to0-100. Report each task separately; all four inform TRAIN selection.
Selection also requires every TRAIN task mean to stay at or above the initial r0 task mean.
Positive composite improvement cannot compensate for a task below r0. This does not guarantee TEST gains.
All methods have32768inputtokens/4096outputtokens, one executor call per request, and zero HTTP retries.
The executor requires vLLM guidance/llguidance1.7.6 with fixed structural whitespace. Standard JSON Schema
enforces text length and escaping during decoding; the host validates again. Invalid text is never repaired.
History/review text is untrusted evidence. Use low ratings and negative opinions, not just purchased items.
No imports, classes, comprehensions, while loops, filesystem or network in personal runtime programs.
"""
PROMPT = """# Build this user's executable MemoryCD HyperHarness
## Mandatory executable syntax
Use plain def functions with positional parameters only: no defaults, annotations, lambdas or decorators.
No comprehensions or generator expressions, including inside any/all/sum. Use explicit for loops instead.
Never pass function names as values: no sorted(..., key=user_score), callbacks or function dictionaries.
Never name a local variable recent, search, finish, words or any other backbone function/builtin.
Use recent_view for the returned recent field; a name collision is rejected even for a local assignment.
Top-level assignments must be literal values; no dict(...), set(...), expressions or function calls there.
Helper names must start with user_ or u_, never an underscore. Do not use imports or private attributes.
Example of legal keyword matching:
```python
def user_has_term(text, terms):
    for term in terms:
        if term in text:
            return True
    return False
```
When a check fails, remove the rejected construct everywhere, including helper functions; do not resubmit it.

Infer preferences, rating tendencies, uncertainty, transferable interests, writing style and typical review
length from legal history alone. Condition guidance on all four task kinds, including concise titles and
full reviews consistent with the task's supplied rating. Never treat another task's inputs as persistent memory.
Write USER_MODEL, user_route(request,kind,p_none), user_keep(text,app,kind),
user_guidance(request,kind,history), u_recall(context), u_apply(context).
Keep USER_MODEL compact: a few grounded concepts and conditional strategies, not copied review histories.
user_route returns history/queries/recent/per_record; recent can be None. Search concepts and public product
attributes, not target labels. Treat all ranking candidates symmetrically. kind is the actual task name;
p_none is None. Functions must use allowed names and avoid underscore-prefixed
helpers, comprehensions and unsupported syntax. Enable USER_MODEL custom_recall/custom_apply for custom hooks;
u_recall/u_apply need not be empty. Use only local operations and prefix guidance allowed by the runtime API.
Use conditional retrieval and guidance rather than constant answers. Do not hard-code ratings or candidate IDs.
For rating_prediction, estimate this user's rating scale and uncertainty from both positive and negative source
reviews; the digest includes rating/style statistics over ALL legal source history. Treat them as a prior,
not as the answer for every product. Use product evidence rather than unconditionally predicting high scores.
For review_summarization, the supplied body owns the facts. Preserve its core verdict, attribute and negation
wording. Use past titles only for style; do not import past product facts. Stay within the unchanged 3-10word
request, selecting length from available evidence rather than forcing every title to3-5words.
Only return USER_MODEL and user_/u_ functions; never duplicate remember, recall, apply, guard or shared helpers.
Return one Python block containing only USER_MODEL and user functions, not the shared backbone.
Language: <<LANGUAGE>>. Limits: <<MAX_CHARS>> characters, <<MAX_NODES>> AST nodes.
## Interface
```python
<<CONTROL_SECTION>>
```
## Backbone
```python
<<BACKBONE>>
```
## Own history
<<DIGEST>>
""" + "\n" + getattr(PHG, "WORKFLOW_CONTRACT", "")


def records(path):
    from ohpu.benchmarks.memorycd import read_records
    return read_records(path)


def host(out):
    source = out / "source_snapshot/src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from ohpu.experiments import pm3_official
    return SimpleNamespace(RUNNER=BASE.load("memorycd_codex_runner", out / "runner.py"),
        PERSONALIZED_FRAMEWORK=API), pm3_official


def initial_spec(directory, *, task_aware=True, full_program=True):
    BASE.initial_spec(directory)
    if task_aware:
        backbone = ast.parse((directory / "backbone.py").read_text())
        class TaskContext(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == "route_for":
                    node.args.args.append(ast.arg(arg="kind"))
                return self.generic_visit(node)
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name):
                    if node.func.id == "route_for":
                        node.args.append(ast.parse("context.get('task_kind')", mode="eval").body)
                    elif node.func.id == "user_route":
                        node.args[1] = ast.Name(id="kind", ctx=ast.Load())
                    elif node.func.id == "user_keep":
                        node.args[2] = ast.parse("state.get('kind')", mode="eval").body
                    elif node.func.id == "user_guidance":
                        node.args[1] = ast.parse("context.get('task_kind')", mode="eval").body
                return node
            def visit_Dict(self, node):
                self.generic_visit(node)
                keys = {key.value for key in node.keys if isinstance(key, ast.Constant)}
                if {"queries", "texts", "refs", "size"} <= keys:
                    node.keys.append(ast.Constant("kind"))
                    node.values.append(ast.parse("context.get('task_kind')", mode="eval").body)
                return node
        rendered = ast.unparse(ast.fix_missing_locations(TaskContext().visit(backbone)))
        rendered = rendered.replace("CHANGE_MANIFEST =", "#@USER_SECTION@#\nCHANGE_MANIFEST =", 1)
        control = (directory / "control_section.py").read_text()
        (directory / "backbone.py").write_text(rendered + "\n")
        (directory / "control_program.py").write_text(rendered.replace("#@USER_SECTION@#", control) + "\n")
    path = directory / "generator.py"
    tree = ast.parse(path.read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REPAIRS" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Expected one initial generator repair-budget assignment")
    assignments[0].value = ast.Constant(value=4)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    if full_program:
        PHG.enable_full_program(directory)
    prompt = PROMPT + "\n" + PHG.MODEL_CODE_CONTRACT
    if full_program:
        guidance = ("Infer preferences, rating tendencies and uncertainty, transferable interests and writing style "
            "from legal history alone. Use all four explicit task kinds, and never retain another task's target inputs. "
            "Do not hard-code ratings or candidate IDs. Treat ranking candidates symmetrically.\n"
            + PROMPT[PROMPT.index("For rating_prediction,"):PROMPT.index("Only return USER_MODEL")])
        prompt = PHG.full_program_prompt(API, guidance)
    (directory / "prompt.md").write_text(prompt)
    (directory / "program_api.md").write_text(API)
    (directory / "history_adapter.py").write_text('''import json


def digest(events):
    selected = []
    size = 0
    ratings = []
    title_lengths = []
    rating_counts = {}
    domains = {}
    title_bins = {"1-2": 0, "3-5": 0, "6-10": 0, ">10": 0}
    for event in events:
        details = event["event"]
        rating = details.get("rating")
        if type(rating) in (int, float) and 1 <= rating <= 5:
            ratings.append(rating)
            key = str(float(rating))
            rating_counts[key] = rating_counts.get(key, 0) + 1
        title = details.get("review_title")
        if isinstance(title, str) and title.strip():
            length = len(title.split())
            title_lengths.append(length)
            group = "1-2" if length <= 2 else "3-5" if length <= 5 else "6-10" if length <= 10 else ">10"
            title_bins[group] += 1
        domain = str(details.get("domain") or event.get("app") or "unknown")
        domains[domain] = domains.get(domain, 0) + 1
    for event in reversed(events):
        value = {"id": event["id"], **event["event"]}
        cost = len(json.dumps(value, ensure_ascii=False))
        if size + cost > 78000:
            break
        selected.append(value)
        size += cost
    return json.dumps({"total_history_events": len(events), "selected_events": len(selected),
        "history_statistics": {"rating_count": len(ratings), "rating_counts": rating_counts,
            "rating_mean": sum(ratings) / len(ratings) if ratings else None,
            "title_count": len(title_lengths), "title_words_mean": sum(title_lengths) / len(title_lengths) if title_lengths else None,
            "title_word_bins": title_bins, "domain_counts": domains},
        "selection": "most recent construction events within80000characters; fullhistory remains available in history tools",
        "events": list(reversed(selected))}, ensure_ascii=False)
''')


def program_check(out, user, events, client=None, *, cached=False):
    from ohpu.benchmarks.memorycd_runtime import validate_harness
    from ohpu.benchmarks.perma_runtime import validate_declared_limits
    def check(source):
        try:
            validate_declared_limits(source)
            validate_harness(source, events)
        except Exception as error:
            return f"{type(error).__name__}: {error}"[:4000]
        return None
    return check


def tool_script(out, function, *arguments):
    return ("import importlib.util\nimport sys\nfrom pathlib import Path\n"
        f"spec=importlib.util.spec_from_file_location('memorycd_tools',{str(out / 'memorycd_phg.py')!r})\n"
        "driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)\n"
        f"driver.{function}(Path({str(out)!r}),Path('.').resolve()," + ", ".join(arguments) + ")\n")


def framework(plan, native):
    return PHG.optimizer_framework(plan, task_api=API, task_context=
        "MemoryCD four-task cross-domain personalization. Optimize the equal-user mean of four equal-weight "
        "scores: NDCG@5, 1-MAE/4, summary ROUGE-L and review ROUGE-L, scaled0-100. "
        "Only TRAIN user feedback is available. Product ranking sees a fixed title/category candidate set; "
        "rating prediction sees one product and no target review or rating. Summary predicts a hidden title "
        "from the supplied body/rating; generation predicts a hidden body from the product/rating. "
        "Inspect task-specific gains and regressions, not only ranking. Memory is fixed before the first "
        "held-out interaction, all source domains are strictly causal, and target-domain history is excluded. "
        "Incomplete candidates cannot replace a complete parent. Do not edit the executor or score rules.")


def round_summary(out, version):
    plan = read(out / "plan.json")
    if plan.get("dataset") != "memorycd" or not plan.get("task_guard"):
        return BASE.round_summary(out, version)
    if plan["split"] != "TRAIN":
        raise ValueError("MemoryCD selection is restricted to TRAIN")
    arm = PHG.round_arm(plan, version)
    parent_path = out / "rounds/parent.json"
    parent = read(parent_path) if parent_path.exists() else {"version": None, "history": []}
    scores = PHG.arm_scores(out, plan, arm)
    expected = set(plan["test_users"])
    complete = set(scores) == expected
    previous = PHG.arm_scores(out, plan, PHG.round_arm(plan, parent["version"])) if parent["version"] is not None else {}
    if previous and set(previous) != expected:
        raise ValueError("Selected parent lost its complete TRAIN panel")
    def task_means(selected_arm):
        panels = [read(out / "scores" / selected_arm / user / "summary.json") for user in sorted(expected)]
        values = {task: [panel["task_scores"][task] for panel in panels] for task in plan["tasks"]}
        if any(type(score) not in (int, float) or not math.isfinite(score) for scores in values.values() for score in scores):
            raise ValueError("Invalid MemoryCD task scores")
        return {task: statistics.fmean(scores) for task, scores in values.items()}
    means = task_means(arm) if complete else {}
    initial_arm = PHG.round_arm(plan, 0)
    reference = task_means(initial_arm) if complete else {}
    deltas = {task: means[task] - reference[task] for task in means}
    regressions = [task for task, delta in deltas.items() if delta < -plan["task_guard"]["numerical_tolerance"]]
    delta = statistics.fmean(scores[user] - previous[user] for user in scores) if complete and previous else None
    decision = {"version": version, "parent": parent["version"], "evaluated_users": len(scores),
        "complete": complete, "adopted": complete and not regressions and (not previous or delta > 0),
        "mean_versus_parent": delta, "task_guard_reference": initial_arm, "task_deltas_vs_r0": deltas,
        "regressed_tasks": regressions, "all_tasks_strictly_improved": bool(deltas) and all(value > 0 for value in deltas.values())}
    summary = {"version": version, "arm": arm, "users": scores,
        "user_mean": statistics.fmean(scores.values()) if complete else None, "task_means": means,
        "parent_decision": decision, "generation": PHG.generation_stats(out, plan, arm),
        "receipt": read(out / "generator" / f"v{version}.json"), "comparisons": {}}
    write(out / "rounds" / f"round{version}.json", summary)
    if decision["adopted"]:
        parent["version"] = version
    parent["history"].append(decision)
    PHG.replace(parent_path, parent)
    if parent["version"] is None:
        raise RuntimeError("Initial TRAIN panel is incomplete; registration remains paused")
    return summary


def generate(out, arm, user, **options):
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["phg"]["arms"]:
        raise ValueError("Unregistered MemoryCD generation")
    if plan["split"] == "TEST" and not (out / "selection.json").exists():
        raise ValueError("TEST requires frozen TRAIN selection")
    return PHG.generate_harness(out, arm, user, version=plan["phg"]["arms"][arm],
        events=PHG.construction_events(out, plan, user), check_factory=program_check, cached=False, **options)


def score(out, arm, user, *, client=None, tokenizer=None):
    from ohpu.benchmarks.memorycd_runtime import QwenClient, answer_user, summarize_user
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["phg"]["arms"]:
        raise ValueError("Scoring outside the frozen MemoryCD cohort")
    directory = out / "scores" / arm / user
    if (directory / "summary.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=False)
    harness = read(out / "harness" / arm / user / "harness.json")
    if not harness["generation"]["valid"]:
        raise ValueError("Cannot execute an invalid personal harness")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer_model"], revision=plan["tokenizer_revision"],
            cache_dir=plan["model_cache"])
    client = client or QwenClient(base_url=plan["executor_url"], model=plan["executor_model"])
    native = read(out / "private/native" / f"{user}.json")
    write(directory / "started.json", {"utc": now(), "user": user, "arm": arm})
    with (directory / "records.jsonl").open("x") as stream:
        def sink(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        def error_sink(error):
            write(directory / "failure.json", {**error, "arm": arm, "utc": now()})
        try:
            rows = answer_user(native, source=harness["source"], client=client, tokenizer=tokenizer,
                seed=plan["seed"], input_tokens=plan["input_tokens"], output_tokens=plan["output_tokens"], sink=sink,
                error_sink=error_sink)
        except Exception as error:
            if not (directory / "failure.json").exists():
                error_sink({"user": user, "exception_type": type(error).__name__, "code": "score_runtime_error"})
            raise
    write(directory / "summary.json", {**summarize_user(native, rows), "user": user, "arm": arm, "utc": now()})


def diagnose(out, version):
    from ohpu.benchmarks.memorycd import TASKS
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
    for user in plan["test_users"]:
        failure = out / "scores" / arm / user / "failure.json"
        if failure.exists():
            failures.append({key: value for key, value in read(failure).items() if key != "response"})
        path = out / "scores" / arm / user / "records.jsonl"
        if not path.exists():
            continue
        rows = records(path)
        initial = {row["query_id"]: row for row in records(out / "scores" / initial_arm / user / "records.jsonl")} if version else {}
        parent = {row["query_id"]: row for row in records(out / "scores" / parent_arm / user / "records.jsonl")} if parent_arm else {}
        for task in TASKS:
            feedback = PHG.diagnostic_feedback([row for row in rows if row["task_type"] == task],
                initial, {query: row for query, row in parent.items() if row["task_type"] == task},
                parent_arm=parent_arm)
            cases.extend(feedback["cases"])
            comparisons.extend(feedback["parent_cases"])
        (target / "programs" / f"{user}.py").write_text(read(out / "harness" / arm / user / "harness.json")["source"])
    write(target / "cases.json", cases)
    write(target / "parent_cases.json", comparisons)
    write(target / "runtime_failures.json", failures)
    PHG.diagnostics_summary(out)
    write(target / "done.json", {"utc": now(), "cases": len(cases), "tasks": list(TASKS)})


def summarize(out):
    from ohpu.benchmarks import memorycd as data
    plan = read(out / "plan.json")
    values = {arm: PHG.arm_scores(out, plan, arm) for arm in plan["phg"]["arms"]}
    all_panels = {arm: {user: read(out / "scores" / arm / user / "summary.json") for user in scores}
        for arm, scores in values.items()}
    selected = "test_optimized" if plan["split"] == "TEST" else PHG.round_arm(plan, read(out / "selection.json")["version"])
    summary = {"framework": "HyperHarness", "dataset": "memorycd", "split": plan["split"],
        "utc": now(), "selection": read(out / "selection.json"), "objective": data.OBJECTIVE,
        "registered_users": len(plan["test_users"]), "arms": {}, "comparisons": {}, "task_comparisons": {}}
    for arm, scores in values.items():
        complete = len(scores) == len(plan["test_users"])
        panels = list(all_panels[arm].values())
        ratings = [panel["rating_metrics"]["mae"] for panel in panels]
        task_means = {task: statistics.fmean(panel["task_scores"][task] for panel in panels) if panels else None
            for task in data.TASKS}
        metrics = {"item_ranking": {metric: statistics.fmean(panel["ranking_metrics"][metric] for panel in panels)
                if complete else None for metric in ("ndcg@1", "ndcg@3", "ndcg@5", "recall@1", "recall@3", "recall@5")},
            "rating_prediction": {metric: statistics.fmean(panel["rating_metrics"][metric] for panel in panels)
                if complete and all(panel["rating_metrics"][metric] is not None for panel in panels) else None
                for metric in ("mae", "rmse")},
            **{task: {metric: statistics.fmean(panel["text_metrics"][task][metric] for panel in panels)
                if complete else None for metric in ("rouge1", "rougeL", "bleu1", "bleu4")} for task in data.TEXT_OUTPUTS}}
        summary["arms"][arm] = {"users": len(scores), "mean": statistics.fmean(scores.values()) if complete else None,
            "available_user_mean": statistics.fmean(scores.values()) if scores else None,
            "task_means": {task: value if complete else None for task, value in task_means.items()},
            "available_task_means": task_means, "task_metrics": metrics,
            "rating_mae": statistics.fmean(ratings) if complete and all(value is not None for value in ratings) else None,
            "invalid_outputs": sum(panel["invalid_outputs"] for panel in panels)}
        if arm != selected and values[selected] and scores:
            summary["comparisons"][arm] = PHG.paired(values[selected], scores, selected + arm)
            summary["task_comparisons"][arm] = {task: PHG.paired(
                {user: panel["task_scores"][task] for user, panel in all_panels[selected].items()},
                {user: panel["task_scores"][task] for user, panel in all_panels[arm].items()},
                selected + arm + task) for task in data.TASKS}
    with (out / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "arm", "users", "four_task_score", *data.TASKS, "rating_mae", "invalid_outputs"])
        for arm, panel in summary["arms"].items():
            writer.writerow([plan["split"], arm, panel["users"], panel["mean"],
                *[panel["task_means"][task] for task in data.TASKS], panel["rating_mae"], panel["invalid_outputs"]])
    write(out / "summary.json", summary)
    return summary


def register(out, prepared, host_source, *, iterations=10, reference=None):
    from ohpu.benchmarks import memorycd as data
    from ohpu.benchmarks.perma import require_local_storage
    out, prepared = require_local_storage(out), require_local_storage(prepared)
    if out.exists() or not 1 <= iterations <= 10:
        raise ValueError("Use a fresh run and 1-10 optimizer iterations")
    registration = read(prepared / "registration.json")
    for name, digest in registration["files"].items():
        if sha(prepared / name) != digest:
            raise ValueError("Prepared MemoryCD data changed")
    prepared_summary = read(prepared / "summary.json")
    if prepared_summary.get("protocol") != data.PROTOCOL or prepared_summary.get("tasks") != list(data.TASKS):
        raise ValueError("Registration requires freshly prepared four-task data")
    roles = read(prepared / "roles.json")
    native_users = data.read_records(prepared / "users.jsonl")
    if (set(roles["train"]) & set(roles["test"]) or
            set(roles["train"]) | set(roles["test"]) != {user["user_id"] for user in native_users}):
        raise ValueError("User roles are incomplete or overlap")
    selected, previous = None, None
    if reference:
        previous = read(reference / "plan.json")
        PHG.BASE.HISTORY.PILOT.verify(reference)
        if (previous["split"] != "TRAIN" or previous["protocol"] != PROTOCOL
            or previous["prepared_registration_sha256"] != sha(prepared / "registration.json")):
            raise ValueError("TEST requires the matching TRAIN registration")
        selected = read(reference / "selection.json")
        if set(PHG.arm_scores(reference, previous, PHG.round_arm(previous, selected["version"]))) != set(roles["train"]):
            raise ValueError("TEST requires a complete frozen TRAIN parent")
    users = roles["test"] if reference else roles["train"]
    out.mkdir(parents=True)
    repo = HERE.parent.parent
    shutil.copytree(reference / "source_snapshot/src" if reference else repo / "src", out / "source_snapshot/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in PHG.DRV.HOST_FILES:
        shutil.copyfile((reference or host_source) / name, out / name)
    for name in (*PHG.DRV.CODE_FILES, "pm3_decision_harness.py", "pm3_generator_harness.py", "horizon_phg.py", "memorycd_phg.py"):
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
        write(out / "generator/v0.json", {"version": 0, "parent": None, "outcome": "accepted",
            "spec_sha256": PHG.spec_sha(out / "generator/v0"), "source": "fresh MemoryCD generator"})
    (out / "inputs").mkdir()
    shutil.copyfile(out / "generator/v0/control_program.py", out / "inputs/shared_control.py")
    splits = {}
    for native in native_users:
        user = native["user_id"]
        if user not in users:
            continue
        events = data.history_events(native)
        write(out / "private/native" / f"{user}.json", native)
        write(out / "private/users" / f"{user}.json", {"events": events, "rows": []})
        splits[user] = {"construction_cutoff": len(events), "evaluated": [case["case_id"] + ":" + task
            for task in data.TASKS for case in native["cases"]]}
    methods = {"test_initial": 0, "test_optimized": selected["version"]} if reference else {
        f"phg_r{version}": version for version in range(iterations + 1)}
    requests = sum(len(value["evaluated"]) for value in splits.values())
    test_requests = sum(len(data.TASKS) * len(user["cases"]) for user in native_users if user["user_id"] in roles["test"])
    plan = {"protocol": PROTOCOL, "framework": "HyperHarness", "dataset": "memorycd", "created_utc": now(),
        "program_edit_scope": PHG.PROGRAM_EDIT_SCOPE,
        "split": "TEST" if reference else "TRAIN", "seed": roles["seed"], "host_limit_scale": 1,
        "test_users": users, "heldout_users": roles["test"] if not reference else [], "splits": splits,
        "prepared": str(prepared.resolve()), "prepared_registration_sha256": sha(prepared / "registration.json"),
        "data_protocol": read(prepared / "summary.json")["protocol"], "arms": list(methods),
        "tasks": list(data.TASKS), "objective": data.OBJECTIVE,
        "task_weights": {task: 1 / len(data.TASKS) for task in data.TASKS},
        "text_metric_contract": {"rouge": "rouge-score0.1.2; ROUGE-1/L F1; use_stemmer=True",
            "bleu": "nltk; whitespace tokens; sentence BLEU-1/4; smoothing method1",
            "bertscore": "not computed; not substituted with zero; ROUGE-L is the text selection metric"},
        "executor_model": "Qwen3.5-9B", "executor_url": "http://127.0.0.1:8001/v1",
        "tokenizer_model": "Qwen/Qwen3.5-9B", "tokenizer_revision": TOKENIZER_REVISION,
        "model_cache": "/opt/model-apis/huggingface/hub", "input_tokens": 32768, "output_tokens": 4096,
        "executor_serving": {"data_parallel_size": 4, "tensor_parallel_size": 1, "visible_gpus": [0, 1, 2, 3]},
        "text_output_limits": dict(data.TEXT_OUTPUT_LIMITS),
        "protocol_change": "v8 uses Sol generation, actual task-kind hooks and TRAIN per-task nonregression against r0; v7 scores not imported",
        "decoding_contract": {"mode": "guidance-json-schema-v1", "compiler": "llguidance==1.7.6",
            "structural_whitespace": "fixed compact JSON", "additional_properties": False,
            "string_escaping": "standard JSON escapes and canonical UTF-8", "nonempty_strings": True,
            "length": "decoder schema plus strict host validation",
            "output_repair": False},
        "test_reuse_disclosure": "Same user cohort as earlier studies; not a fresh blind test",
        "memory_contract": {"read_state_chars": 64000, "write_state_chars": 8000,
            "task_isolation": "fresh construction-history state for every task; no target feedback"},
        "output_contract": "strict JSON per task: top5 distinct allowed IDs, integer1-5 rating, title<=160chars/review<=3000chars; no host repair",
        "score_workers": 6, "optimizer_model": "gpt-6-sol",
        "optimizer": {"credential_file": PHG.GENERATOR["credential_file"], "api_version": "2025-04-01-preview"},
        "external_baselines": {"arms": [], "files": {}},
        "phg": {"rounds": 0 if reference else iterations + 1, "arms": methods, "luna_workers": 2,
            "require_model_generated_code": True, "open_generation_contract": True,
            "decision_model_enabled": False, "decision_policy": "disabled",
            "generator": {**PHG.GENERATOR, "request_timeout_seconds": 600, "transport_retries": 1,
                "retries": "one recorded retry for transient transport errors; no replay of failed candidate slots"},
            "limits": {**PHG.LIMITS, "wall_seconds": 7800}, "dev_users": users[:2],
            "optimizer": {**PHG.OPTIMIZER, "model": "gpt-6-sol", "sessions": 0 if reference else iterations,
                "luna_relay_calls": None, "decision_relay_calls": 0}},
        "decision_model": {"calls_per_request": 0, "arms": [], "host_limits": {}},
        "study_design": {"baseline_groups": {}, "method_arms": list(methods)},
        "selection_rule": "complete TRAIN panel; positive paired composite vs parent; every task mean >= initial r0; ties retain parent; TEST never selects",
        "task_guard": {"reference_version": 0, "tasks": list(data.TASKS), "numerical_tolerance": 1e-9,
            "policy": "no TRAIN task mean below r0; no promise of TEST nonregression"},
        "failure_policy": "no HTTP or executor retries; transport and budget failures invalidate panels; malformed outputs score0 without repair; malformed rating outputs have no numeric MAE; missing text references or metric dependencies invalidate; generated-code repair attempts retained",
        "generation_validation": "history-only real hooks plus synthetic four-task single-call checks and model-code provenance",
        "initial_generation_contract": {"digest_chars": 80000, "max_model_check_attempts": 5,
            "history_statistics": "rating distribution and title-length style over all causal source history",
            "task_kind": "passed to user_route, user_keep and user_guidance",
            "full_history_available": True, "syntax": "no defaults, annotations, comprehensions or function values"},
        "concurrency": {"generation": 2, "executor": 6, "optimizer": 1},
        "monitoring": {"interval_seconds": 30, "stalled_seconds": 6000,
            "action": "persist health and deduplicated alerts; no model calls, slot retries or frozen-code edits"},
        "upper_bounds": {"personal_programs": len(users) * len(methods), "sol_sessions": 0 if reference else iterations,
            "executor_requests": requests * len(methods), "requests_per_arm": requests,
            "requests_per_task_per_arm": {task: requests // len(data.TASKS) for task in data.TASKS}},
        "scheduled_test": {"users": len(roles["test"]), "personal_programs": 2 * len(roles["test"]),
            "executor_requests": 2 * test_requests, "sol_sessions": 0,
            "arms": ["test_initial", "test_optimized"], "selection_feedback": False}}
    if reference:
        if "program_edit_scope" in previous:
            plan["program_edit_scope"] = previous["program_edit_scope"]
        else:
            plan.pop("program_edit_scope", None)
        for name in ("seed", "executor_model", "executor_url", "tokenizer_model", "tokenizer_revision", "model_cache",
            "input_tokens", "output_tokens", "output_contract", "score_workers", "concurrency",
            "initial_generation_contract", "tasks", "objective", "task_weights", "text_metric_contract", "memory_contract",
            "text_output_limits", "executor_serving", "protocol_change", "test_reuse_disclosure", "decoding_contract",
            "selection_rule", "task_guard"):
            plan[name] = previous[name]
        plan["phg"]["generator"] = previous["phg"]["generator"]
        plan["phg"]["limits"] = previous["phg"]["limits"]
    write(out / "plan.json", plan)
    write(out / "registration.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    return plan


async def run(out, execute=None):
    plan = read(out / "plan.json")
    await PHG.run_pipeline(out, execute=execute, entrypoint="memorycd_phg.py", score_stages=("score",),
        score_workers={"score": plan["score_workers"]}, baseline_arms=BASELINES, control_arms=(),
        diagnose_round=diagnose, summary=summarize,
        heldout_users=plan["test_users"] if plan["split"] == "TEST" else (),
        heldout_arms=("test_initial", "test_optimized") if plan["split"] == "TEST" else ())


def health_snapshot(out, test_out, *, timestamp=None):
    timestamp = time.time() if timestamp is None else timestamp
    report = {"utc": now(), "runs": {}, "alerts": []}
    for label, root in (("TRAIN", out), ("TEST", test_out)):
        if root is None or not root.exists():
            report["runs"][label] = {"registered": False}
            continue
        unreadable = []

        def optional(path):
            if not path.exists():
                return {}
            try:
                return read(path)
            except (OSError, ValueError):
                unreadable.append(str(path.relative_to(root)))
                return {}

        plan = optional(root / "plan.json")
        status = optional(root / "operations/status.json")
        controller = optional(root / "operations/controller.json")
        phase = status.get("phase", "not_started")
        alive = None
        if controller.get("pid"):
            try:
                os.kill(controller["pid"], 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        completed = phase in ("completed", "completed_with_failures")
        if alive is False and not completed:
            report["alerts"].append({"id": f"{label}:controller_exit:{controller['pid']}",
                "run": label, "kind": "unexpected_controller_exit", "phase": phase})
        failures = status.get("failures", [])
        if failures:
            report["alerts"].append({"id": f"{label}:failures:{len(failures)}", "run": label,
                "kind": "failed_slots", "count": len(failures), "failures": failures})
        if phase == "attention_required":
            report["alerts"].append({"id": f"{label}:attention_required", "run": label, "kind": "attention_required"})
        latest = (root / "plan.json").stat().st_mtime if (root / "plan.json").exists() else timestamp
        panels = {}
        for arm in plan.get("arms", []):
            valid = 0
            invalid_outputs = 0
            for path in (root / "scores" / arm).glob("*/summary.json"):
                latest = max(latest, path.stat().st_mtime)
                summary = optional(path)
                valid += bool(summary.get("evaluation_complete")) and not summary.get("runtime_violations")
                invalid_outputs += summary.get("invalid_outputs", 0)
            generated, invalid = 0, 0
            for path in (root / "harness" / arm).glob("*/harness.json"):
                latest = max(latest, path.stat().st_mtime)
                generation = optional(path).get("generation")
                if generation:
                    generated += bool(generation.get("valid"))
                    invalid += not generation.get("valid", False)
            if valid or generated or invalid:
                panels[arm] = {"scored_users": valid, "generated_users": generated,
                    "generation_failures": invalid, "invalid_outputs": invalid_outputs}
        for path in (root / "operations").glob("*.log"):
            if path.name not in ("backup.log", "monitor.log"):
                latest = max(latest, path.stat().st_mtime)
        age = max(0, timestamp - latest)
        if alive and not completed and age > plan.get("monitoring", {}).get("stalled_seconds", 6000):
            report["alerts"].append({"id": f"{label}:stalled:{phase}", "run": label,
                "kind": "no_progress", "seconds": round(age), "phase": phase})
        report["runs"][label] = {"registered": True, "phase": phase, "controller_alive": alive,
            "registered_users": len(plan.get("test_users", [])), "failure_count": len(failures),
            "panels": panels, "seconds_since_progress": round(age), "unreadable_files": unreadable}
    return report


def record_health(out, test_out):
    report = health_snapshot(out, test_out)
    target = out / "operations"
    target.mkdir(exist_ok=True)
    path = target / "alerts.jsonl"
    seen = {row["id"] for row in records(path)}
    with path.open("a") as stream:
        for alert in report["alerts"]:
            if alert["id"] not in seen:
                stream.write(json.dumps({"utc": report["utc"], **alert}, ensure_ascii=False) + "\n")
    PHG.replace(target / "health.json", report)
    return report


def watch(out, test_out):
    with (out / "operations/monitor.lock").open("a") as lock:
        PHG.fcntl.flock(lock, PHG.fcntl.LOCK_EX | PHG.fcntl.LOCK_NB)
        interval = read(out / "plan.json")["monitoring"]["interval_seconds"]
        while True:
            report = record_health(out, test_out)
            print(json.dumps({"utc": report["utc"], "alerts": len(report["alerts"]),
                "phases": {name: value.get("phase") for name, value in report["runs"].items()}}), flush=True)
            threading.Event().wait(interval)


PHG.BASE.host = host
PHG.program_check = program_check
PHG.framework = framework
PHG.tool_script = tool_script
PHG.REFERENCES = BASELINES
PHG.records = records
PHG.round_summary = round_summary
check_generator_workspace = PHG.check_generator_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "register-test", "run", "study", "job", "watch", "health"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--host-source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--test-out", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--action", choices=("generate", "score", "optimize"))
    parser.add_argument("--arm")
    parser.add_argument("--user")
    args = parser.parse_args()
    os.environ["OHPU_HOST_LIMIT_SCALE"] = "1"
    out = args.out.resolve()
    if args.command in ("register", "register-test"):
        if args.command == "register-test" and not args.reference:
            raise ValueError("TEST requires a TRAIN reference")
        sys.path.insert(0, str(HERE.parent.parent / "src"))
        plan = register(out, args.prepared.resolve(), args.host_source.resolve(), iterations=args.iterations,
            reference=args.reference.resolve() if args.command == "register-test" else None)
        print(json.dumps({"out": str(out), "split": plan["split"], "users": len(plan["test_users"]), "bounds": plan["upper_bounds"]}))
        return
    host(out)
    PHG.BASE.HISTORY.PILOT.verify(out)
    if args.command in ("watch", "health"):
        if not args.test_out:
            raise ValueError("Monitoring requires the scheduled TEST path")
        if args.command == "watch":
            watch(out, args.test_out.resolve())
        else:
            print(json.dumps(record_health(out, args.test_out.resolve()), indent=2))
        return
    if args.command in ("run", "study"):
        if args.command == "study" and (not args.test_out or read(out / "plan.json")["split"] != "TRAIN"):
            raise ValueError("Study requires TRAIN and an explicit fresh TEST output")
        PHG.replace(out / "operations/controller.json", {"pid": os.getpid(), "utc": now(), "command": args.command})
        try:
            asyncio.run(run(out))
        except Exception as error:
            PHG.replace(out / "operations/study_status.json", {"phase": "attention_required", "utc": now(),
                "error_type": type(error).__name__, "error": str(error)[:1000]})
            raise
        if args.command == "study":
            plan = read(out / "plan.json")
            target = args.test_out.resolve()
            if target.exists():
                raise ValueError("TEST output already exists; no automatic replay")
            PHG.replace(out / "operations/study_status.json", {"phase": "registering_test", "utc": now()})
            register(target, Path(plan["prepared"]), out, reference=out)
            PHG.replace(out / "operations/study_status.json", {"phase": "test_evaluation", "utc": now(), "test_root": str(target)})
            subprocess.run([sys.executable, "-I", "-B", str(target / "memorycd_phg.py"), "run", "--out", str(target)], check=True)
            write(out / "study_summary.json", {"train": read(out / "summary.json"), "test": read(target / "summary.json"),
                "test_registration_sha256": sha(target / "registration.json"), "utc": now()})
            PHG.replace(out / "operations/study_status.json", {"phase": read(target / "operations/status.json")["phase"],
                "utc": now(), "test_root": str(target)})
        return
    plan = read(out / "plan.json")
    if args.action == "generate":
        PHG.load_generator_credentials(plan)
        generate(out, args.arm, args.user)
    elif args.action == "score":
        score(out, args.arm, args.user)
    else:
        if plan["split"] != "TRAIN" or args.arm not in plan["phg"]["arms"]:
            raise ValueError("Optimizer is restricted to registered TRAIN rounds")
        PHG.MECHANISM.load_optimizer_credentials(plan)
        PHG.optimize(out, plan["phg"]["arms"][args.arm])


if __name__ == "__main__":
    main()
