"""Train HyperHarness on PERMA MCQs, then evaluate frozen generators on held-out users."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location("perma_shared_adapter", HERE / "memorycd_phg.py")
BASE = importlib.util.module_from_spec(specification)
specification.loader.exec_module(BASE)
PHG = BASE.PHG
read, write, sha, now = PHG.read, PHG.write, PHG.sha, PHG.now
BASELINES = ("no_history", "bm25_10", "shared_control")
PROTOCOL = "perma-clean-mcq-hyperharness-user-half-v2"
PERSONALIZATION_MODES = ("retrieval_only", "guidance_only", "combined")
API = """# PERMA MCQ HyperHarness contract
One restricted Python program per user is generated from the common prefix of that user's nonzero-memory
checkpoints. No questions, options, labels, profiles, preference annotations or later history enter generation.
At execution each checkpoint has its own source-bound memory containing only that checkpoint's public dialogues.
Type1 bypasses ALL personal code and guidance, as in the upstream zero-memory baseline. No answer feedback persists.
Observations contain date/messages with role/content only. remember/recall/apply use local memory with BM25,
read-only SQL and finish(output,source_refs). No runtime model/decision calls; one host-owned final Qwen call.
recall receives stage=context, task_kind=perma_mcq, query and backend(step/state/result). query contains the
question and ALL eight options, never a gold label. apply may return only {'prefix': guidance}; guard must be {}.
Read-state capacity is64000characters; write-state remains8000. FINAL memory output is at most12000characters,
at most64source_refs, at most32memory operations and20000interpreter steps PER hook. These limits are distinct.
Guidance is at most1500characters. user_guidance(request,kind,evidence) receives a dict with output/source_refs,
not a boolean. Return an empty string unless specific current retrieved evidence supports the guidance.
All arms have32768inputtokens/32outputtokens. Do not raise backbone max_chars above12000 or top_k above64.
Choose the best response to the user query, using actual dialogue facts rather than plausible stereotypes.
Output exactly one letter A-H. Options and provided labels are preserved; do not assume a fixed correct letter.
TRAIN selection maximizes equal-user mean accuracy across Clean single/multi-domain Type1/2/3 checkpoints.
Report checkpoint and domain breakdowns separately. TEST must never guide optimization or reselection.
No imports, classes, comprehensions, while loops, filesystem or network in generated runtime programs.
"""
PROMPT = """# Build this user's executable PERMA HyperHarness
Infer grounded preferences, useful retrieval concepts, corrections and uncertainty from ONLY the available
conversation prefix. The same program later ingests more conversations at legal checkpoints. Do not freeze
early preferences as permanent facts; use runtime evidence for updated preferences and cross-domain requests.
Write USER_MODEL, user_route(request,kind,p_none), user_keep(text,app,kind),
user_guidance(request,kind,evidence), u_recall(context), u_apply(context).
user_route returns history/queries/recent/per_record; recent may be None. kind/p_none default to None.
Use all option texts symmetrically for evidence search, not option positions. Do not hard-code an answer.
Preserve the full request as the first retrieval query; the backbone balances long options symmetrically.
Start from the strong shared BM25 retrieval policy. Personalization is optional when evidence is weak.
Keep user_keep permissive unless retrieved text directly supports exclusion. Do not hard-code a fixed answer.
evidence is a dict: evidence.get('output','') is retrieved text and evidence.get('source_refs',[]) are citations.
Default user_guidance to ''. Only emit brief conditional guidance supported by the actual evidence, not merely
construction-prefix beliefs or a nonempty memory. Empty guidance is valid and preferred to generic reminders.
u_recall/u_apply return {} unless custom hooks are enabled. Preserve remember and empty guard scaffolding.
Use plain def functions with positional parameters only: no defaults, annotations, decorators or lambdas.
No comprehensions, generator expressions, function values/callbacks, imports, or private attributes.
Helpers start with user_ or u_. Top-level assignments are literals only. Use explicit for loops.
When a check rejects syntax, remove it everywhere before retrying. Return one Python block with USER_MODEL
and user functions, not the shared backbone. Language: <<LANGUAGE>>.
Limits: <<MAX_CHARS>> characters, <<MAX_NODES>> AST nodes.
## Interface
```python
<<CONTROL_SECTION>>
```
## Backbone
```python
<<BACKBONE>>
```
## Own construction history
<<DIGEST>>
"""


def host(out):
    source = out / "source_snapshot/src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from ohpu.experiments import pm3_official
    return SimpleNamespace(RUNNER=BASE.BASE.load("perma_codex_runner", out / "runner.py"),
        PERSONALIZED_FRAMEWORK=API), pm3_official


def initial_spec(directory, *, personalization="combined"):
    BASE.initial_spec(directory, task_aware=False, full_program=False)
    backbone = ast.parse((directory / "backbone.py").read_text())
    replacements = ast.parse('''
def balanced_query(request):
    if "\\nOptions:\\n" not in request:
        return request[:8000]
    header, options = request.split("\\nOptions:\\n", 1)
    sections = {}
    current = ""
    for line in options.split("\\n"):
        if len(line) > 2 and line[0] in "ABCDEFGH" and line[1] == ":":
            current = line[0]
            sections[current] = line[2:]
        elif current and not line.startswith("Return only"):
            sections[current] = sections[current] + " " + line
    result = header[:600]
    for label in "ABCDEFGH":
        result = result + "\\n" + label + ": " + sections.get(label, "")[:900]
    return result

def apply(context):
    if context.get("stage") != "generation" or "prefix" not in (context.get("base") or {}).get("expressible", []):
        return {}
    if USER_MODEL.get("custom_apply"):
        return u_apply(context)
    evidence = context.get("backend_evidence") or {"output": "", "source_refs": []}
    if not evidence.get("source_refs"):
        return {}
    text = str(user_guidance(str(context.get("query") or ""), context.get("task_kind"), evidence) or "").strip()
    return {"prefix": text[:BACKBONE["retrieval"]["guidance_chars"]]} if text else {}
''')
    class Queries(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if (isinstance(node.value, ast.Name) and node.value.id in ("request", "text")
                    and isinstance(node.slice, ast.Slice) and isinstance(node.slice.upper, ast.Constant)
                    and node.slice.upper.value == 8000):
                return ast.copy_location(ast.Call(func=ast.Name(id="balanced_query", ctx=ast.Load()),
                    args=[node.value], keywords=[]), node)
            return node
    backbone = Queries().visit(backbone)
    backbone.body = [node for node in backbone.body if not isinstance(node, ast.FunctionDef) or node.name != "apply"]
    backbone.body.extend(replacements.body)
    for node in backbone.body:
        if isinstance(node, ast.FunctionDef) and node.name == "collect":
            loop = next(statement for statement in node.body if isinstance(statement, ast.For))
            loop.body.insert(0, ast.If(test=ast.Compare(left=ast.Call(func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Subscript(value=ast.Name(id="state", ctx=ast.Load()), slice=ast.Constant("refs"), ctx=ast.Load())],
                keywords=[]), ops=[ast.GtE()], comparators=[ast.Constant(64)]), body=[ast.Break()], orelse=[]))
    rendered = ast.unparse(ast.fix_missing_locations(backbone))
    marker = "CHANGE_MANIFEST ="
    rendered = rendered.replace(marker, "#@USER_SECTION@#\n" + marker, 1)
    section = ast.parse((directory / "control_section.py").read_text())
    for node in section.body:
        if isinstance(node, ast.FunctionDef) and node.name == "user_guidance":
            node.args.args[-1].arg = "evidence"
    control = ast.unparse(section)
    (directory / "backbone.py").write_text(rendered + "\n")
    (directory / "control_section.py").write_text(control + "\n")
    (directory / "control_program.py").write_text(rendered.replace("#@USER_SECTION@#", control) + "\n")
    mode_contract = {"retrieval_only": "Only retrieval is evaluated; the host skips apply. Keep user_guidance empty.",
        "guidance_only": "Retrieval is frozen to shared_control; only evidence-grounded user_guidance is evaluated.",
        "combined": "Both retrieval and evidence-grounded guidance are evaluated; empty guidance remains valid."}[personalization]
    (directory / "prompt.md").write_text(PROMPT + "\nRegistered ablation: " + mode_contract + "\n" + PHG.MODEL_CODE_CONTRACT
        + "\n" + getattr(PHG, "WORKFLOW_CONTRACT", ""))
    (directory / "program_api.md").write_text(API)


def program_check(out, user, events, client=None, *, cached=False):
    from ohpu.benchmarks.perma_runtime import failure_details, validate_harness
    def check(source):
        try:
            validate_harness(source, events)
        except Exception as error:
            return json.dumps({"error": str(error)[:2000], "contract": failure_details(error)}, sort_keys=True)
        return None
    return check


def tool_script(out, function, *arguments):
    return ("import importlib.util\nimport sys\nfrom pathlib import Path\n"
        f"spec=importlib.util.spec_from_file_location('perma_tools',{str(out / 'perma_phg.py')!r})\n"
        "driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)\n"
        f"driver.{function}(Path({str(out)!r}),Path('.').resolve()," + ", ".join(arguments) + ")\n")


def framework(plan, native):
    return PHG.optimizer_framework(plan, task_api=API, task_context=
        "PERMA Clean eight-choice accuracy, equal user weighting. TRAIN users only, no TEST feedback. "
        "Construction uses the common prefix of nonzero-memory checkpoints, never full later histories. "
        "Type1 has no memory and bypasses all personal code. Separate source-bound memories at Type2/3; "
        "do not copy TRAIN facts, user identifiers, option-position heuristics or answers into the generator. "
        "Incomplete candidates cannot replace complete parents. All score rules and host code are frozen. "
        "Read runtime_failures.json before changing retrieval: fix contract violations first. "
        "Type1 is reported but excluded from actionable feedback because personal code never runs there. "
        f"Registered personalization mode: {plan.get('personalization', 'retrieval_only')}. "
        "Honor this mode; empty guidance is valid. Do not optimize by raising host limits.")


def generate(out, arm, user, **options):
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["phg"]["arms"]:
        raise ValueError("Unregistered PERMA generation")
    if plan["split"] in ("TEST", "VALIDATION") and not (out / "selection.json").exists():
        raise ValueError("TEST requires frozen TRAIN selection")
    return PHG.generate_harness(out, arm, user, version=plan["phg"]["arms"][arm],
        events=PHG.construction_events(out, plan, user), check_factory=program_check, cached=False, **options)


def score(out, arm, user, *, client=None, tokenizer=None):
    from ohpu.benchmarks.perma_runtime import QwenClient, answer_user, failure_details, summarize_user
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["arms"]:
        raise ValueError("Scoring outside the frozen PERMA cohort")
    directory = out / "scores" / arm / user
    if (directory / "summary.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=False)
    source = baseline = None
    if arm in ("no_history", "bm25_10"):
        baseline = arm
    elif arm == "shared_control":
        source = (out / "inputs/shared_control.py").read_text()
    else:
        harness = read(out / "harness" / arm / user / "harness.json")
        if not harness["generation"]["valid"]:
            raise ValueError("Cannot execute an invalid personal harness")
        source = harness["source"]
    try:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer_model"], revision=plan["tokenizer_revision"],
                cache_dir=plan["model_cache"])
        client = client or QwenClient(base_url=plan["executor_url"], model=plan["executor_model"])
        native = read(out / "private/native" / f"{user}.json")
        with (directory / "records.jsonl").open("x") as stream:
            def sink(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            def error_sink(error):
                write(directory / "failure.json", {**error, "arm": arm, "utc": now()})
            rows = answer_user(native, source=source, baseline=baseline, client=client, tokenizer=tokenizer,
                seed=plan["seed"], input_tokens=plan["input_tokens"], output_tokens=plan["output_tokens"], sink=sink,
                reference_source=(out / "inputs/shared_control.py").read_text(),
                error_sink=error_sink, personalization=plan.get("personalization", "retrieval_only") if arm not in BASELINES else "combined")
    except Exception as error:
        if not (directory / "failure.json").exists():
            write(directory / "failure.json", {**failure_details(error, user=user, stage="score"), "arm": arm, "utc": now()})
        raise
    write(directory / "summary.json", {**summarize_user(native, rows), "user": user, "arm": arm, "utc": now()})


def diagnose(out, version):
    from ohpu.benchmarks.perma import read_records
    plan = read(out / "plan.json")
    arm = PHG.round_arm(plan, version)
    target = out / "diagnostics" / f"round{version}"
    if (target / "done.json").exists():
        return
    (target / "programs").mkdir(parents=True)
    parent_arm = PHG.diagnostic_parent(out, plan, version)
    cases, comparisons, failures = [], [], []
    for user in plan["test_users"]:
        rows = [row for row in read_records(out / "scores" / arm / user / "records.jsonl") if str(row["task_type"]) != "1"]
        baseline = {row["query_id"]: row for row in read_records(out / "scores/shared_control" / user / "records.jsonl")}
        parent = {row["query_id"]: row for row in read_records(out / "scores" / parent_arm / user / "records.jsonl")} if parent_arm else {}
        feedback = PHG.diagnostic_feedback(rows, baseline, parent, parent_arm=parent_arm)
        cases.extend(feedback["cases"])
        comparisons.extend(feedback["parent_cases"])
        failure = out / "scores" / arm / user / "failure.json"
        if failure.exists():
            failures.append(read(failure))
        program = out / "harness" / arm / user / "harness.json"
        if program.exists():
            (target / "programs" / f"{user}.py").write_text(read(program)["source"])
    write(target / "cases.json", cases)
    write(target / "parent_cases.json", comparisons)
    write(target / "runtime_failures.json", failures)
    PHG.diagnostics_summary(out)
    write(target / "done.json", {"utc": now(), "cases": len(cases), "runtime_failures": len(failures)})


def summarize(out):
    from ohpu.benchmarks import perma as data
    plan = read(out / "plan.json")
    values = {arm: PHG.arm_scores(out, plan, arm) for arm in plan["arms"]}
    selected = "test_optimized" if plan["split"] != "TRAIN" else PHG.round_arm(plan, read(out / "selection.json")["version"])
    result = {"framework": "HyperHarness", "dataset": "perma", "split": plan["split"], "utc": now(),
        "selection": read(out / "selection.json"), "objective": data.OBJECTIVE,
        "registered_users": len(plan["test_users"]), "arms": {}, "comparisons": {}}
    for arm, scores in values.items():
        complete = len(scores) == len(plan["test_users"])
        rows = [row for user in scores for row in data.read_records(out / "scores" / arm / user / "records.jsonl")]
        breakdown = {field: {str(key): {"queries": len(subset),
            "micro_accuracy": statistics.fmean(row["accuracy"] for row in subset) if complete else None}
            for key in sorted({row[field] for row in rows})
            if (subset := [row for row in rows if row[field] == key])} for field in ("task_type", "domain")}
        result["arms"][arm] = {"users": len(scores), "mean": statistics.fmean(scores.values()) if complete else None,
            "available_user_mean": statistics.fmean(scores.values()) if scores else None,
            "micro_accuracy": statistics.fmean(row["accuracy"] for row in rows) if complete and rows else None,
            "panels": breakdown}
        if arm != selected and values[selected] and scores:
            result["comparisons"][arm] = PHG.paired(values[selected], scores, selected + arm)
    write(out / "summary.json", result)
    return result


def register(out, prepared, host_source, *, iterations=10, reference=None,
        executor_url="http://127.0.0.1:8001/v1", model_cache="/opt/model-apis/huggingface",
    credential_file=PHG.GENERATOR["credential_file"], personalization=None,
    development_users=None, validation_users=None):
    from ohpu.benchmarks import perma as data
    from ohpu.benchmarks.perma_runtime import CONTRACT_LIMITS
    out = data.require_local_storage(out)
    prepared = data.require_local_storage(prepared)
    model_cache = str(data.require_local_storage(model_cache))
    if personalization is not None and personalization not in PERSONALIZATION_MODES:
        raise ValueError("Unregistered personalization mode")
    if out.exists() or not 1 <= iterations <= 10:
        raise ValueError("Use a fresh run and 1-10 optimizer iterations")
    for name, digest in read(prepared / "registration.json")["files"].items():
        if sha(prepared / name) != digest:
            raise ValueError("Prepared PERMA data changed")
    prepared_summary = read(prepared / "summary.json")
    if prepared_summary["protocol"] != data.PROTOCOL:
        raise ValueError("Wrong prepared data protocol")
    roles = read(prepared / "roles.json")
    native_users = data.read_records(prepared / "users.jsonl")
    if roles != data.roles([user["user_id"] for user in native_users], seed=roles["seed"]):
        raise ValueError("Invalid 1:1 user roles")
    if (development_users is None) != (validation_users is None):
        raise ValueError("A development fold requires both TRAIN and validation users")
    if development_users is not None:
        if (not development_users or not validation_users or set(development_users) & set(validation_users)
                or set(development_users) | set(validation_users) != set(roles["train"])
                or len(development_users) != len(set(development_users)) or len(validation_users) != len(set(validation_users))):
            raise ValueError("Development folds must partition only original TRAIN users")
        if reference:
            raise ValueError("Evaluation roles must be inherited from the frozen TRAIN reference")
    selected, previous = None, None
    if reference:
        PHG.BASE.HISTORY.PILOT.verify(reference)
        previous = read(reference / "plan.json")
        if (previous["split"] != "TRAIN" or previous["protocol"] != PROTOCOL or
                previous["prepared_registration_sha256"] != sha(prepared / "registration.json")):
            raise ValueError("TEST requires the matching TRAIN registration")
        selected = read(reference / "selection.json")
        if set(PHG.arm_scores(reference, previous, PHG.round_arm(previous, selected["version"]))) != set(previous["test_users"]):
            raise ValueError("TEST requires a complete frozen TRAIN parent")
        if personalization is not None and personalization != previous["personalization"]:
            raise ValueError("TEST personalization must match the frozen TRAIN contract")
        personalization = previous["personalization"]
    else:
        personalization = personalization or "combined"
    evaluation_users = (previous.get("validation_users") or roles["test"]) if reference else (validation_users or roles["test"])
    users = evaluation_users if reference else (development_users or roles["train"])
    evaluation_split = "VALIDATION" if (previous and previous.get("validation_users")) or validation_users else "TEST"
    evaluation_requests = sum(len(user["cases"]) for user in native_users if user["user_id"] in evaluation_users)
    out.mkdir(parents=True)
    repo = HERE.parent.parent
    shutil.copytree(reference / "source_snapshot/src" if reference else repo / "src", out / "source_snapshot/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in PHG.DRV.HOST_FILES:
        shutil.copyfile((reference or host_source) / name, out / name)
    for name in (*PHG.DRV.CODE_FILES, "pm3_decision_harness.py", "pm3_generator_harness.py",
            "horizon_phg.py", "memorycd_phg.py", "perma_phg.py"):
        shutil.copyfile((reference or HERE) / name, out / name)
    if reference:
        for version in sorted({0, selected["version"]}):
            directory, receipt = PHG.accepted_spec(reference, version)
            shutil.copytree(directory, out / "generator" / f"v{version}")
            write(out / "generator" / f"v{version}.json", receipt)
        write(out / "selection.json", selected)
        write(out / "rounds/parent.json", {"version": selected["version"], "history": []})
    else:
        initial_spec(out / "generator/v0", personalization=personalization)
        PHG.check_spec(out / "generator/v0", open_contract=True, require_model_code=True, allow_decision_tools=False)
        write(out / "generator/v0.json", {"version": 0, "parent": None, "outcome": "accepted",
            "spec_sha256": PHG.spec_sha(out / "generator/v0"), "source": "fresh PERMA generator"})
    (out / "inputs").mkdir()
    shutil.copyfile(out / "generator/v0/control_program.py", out / "inputs/shared_control.py")
    splits = {}
    for native in native_users:
        user = native["user_id"]
        if user not in users:
            continue
        events = native["construction_events"]
        write(out / "private/native" / f"{user}.json", native)
        write(out / "private/users" / f"{user}.json", {"events": events, "rows": []})
        splits[user] = {"construction_cutoff": len(events), "evaluated": [case["case_id"] for case in native["cases"]]}
    methods = {"test_initial": 0, "test_optimized": selected["version"]} if reference else {
        f"phg_r{version}": version for version in range(iterations + 1)}
    requests = sum(len(value["evaluated"]) for value in splits.values())
    plan = {"protocol": PROTOCOL, "framework": "HyperHarness", "dataset": "perma", "created_utc": now(),
        "split": evaluation_split if reference else "TRAIN", "seed": roles["seed"], "host_limit_scale": 1,
        "test_users": users, "heldout_users": evaluation_users if not reference else [], "splits": splits,
        "validation_users": list(validation_users or []) if not reference else [],
        "personalization": personalization,
        "host_contract": dict(CONTRACT_LIMITS),
        "validation_policy": "declared bounds plus synthetic short/long history and long A-H options; no model calls or heldout data",
        "test_reuse_disclosure": "Original TEST was observed in v1; subsequent use is a reused-cohort comparison, not a fresh blind test",
        "prepared": str(prepared.resolve()), "prepared_registration_sha256": sha(prepared / "registration.json"),
        "data_protocol": data.PROTOCOL, "arms": [*BASELINES, *methods], "objective": data.OBJECTIVE,
        "executor_model": "Qwen3.5-9B", "executor_url": executor_url,
        "executor_serving": {"data_parallel_size": 4, "tensor_parallel_size": 1,
            "visible_gpus": [0, 1, 2, 3], "host": "127.0.0.1"},
        "storage": {"runtime": "verified node-local filesystem", "backup_interval_seconds": 300,
            "publication": "local tar.gz; exclusive new ASCII filename copy; SHA256 verification; no mount rename",
            "backup_failure": "stop supervised process group; preserve local and partial remote copies"},
        "tokenizer_model": "Qwen/Qwen3.5-9B", "tokenizer_revision": BASE.TOKENIZER_REVISION,
        "model_cache": str(model_cache), "input_tokens": 32768, "output_tokens": 32,
        "score_workers": 2, "optimizer_model": "gpt-6-sol",
        "optimizer": {"credential_file": str(credential_file), "api_version": "2025-04-01-preview"},
        "external_baselines": {"arms": [], "files": {}},
        "phg": {"rounds": 0 if reference else iterations + 1, "arms": methods, "luna_workers": 2,
            "require_model_generated_code": True, "open_generation_contract": True,
            "decision_model_enabled": False, "decision_policy": "disabled",
            "generator": {**PHG.GENERATOR, "credential_file": str(credential_file),
                "request_timeout_seconds": 180, "transport_retries": 1,
                "retries": "one recorded transient retry; no replay of failed candidate slots"},
            "limits": {**PHG.LIMITS, "wall_seconds": 2400}, "dev_users": users[:2],
            "optimizer": {**PHG.OPTIMIZER, "model": "gpt-6-sol", "sessions": 0 if reference else iterations,
                "luna_relay_calls": None, "decision_relay_calls": 0}},
        "decision_model": {"calls_per_request": 0, "arms": [], "host_limits": {}},
        "study_design": {"baseline_groups": {"non_personalized": ["no_history"], "retrieval": ["bm25_10"],
            "shared_policy": ["shared_control"]}},
        "selection_rule": "complete TRAIN panel; positive paired equal-user mean; ties retain parent; TEST never selects",
        "failure_policy": "zero executor retries; transport failures invalidate panels; malformed letters score0; retain all generation attempts",
        "concurrency": {"generation": 2, "executor": 2, "optimizer": 1,
            "optimizer_scope": "one shared generator; dependent iterations sequential, not per-user Sol sessions"},
        "monitoring": {"interval_seconds": 30, "stalled_seconds": 6000},
        "upper_bounds": {"personal_programs": len(users) * len(methods), "sol_sessions": 0 if reference else iterations,
            "executor_requests": requests * (len(BASELINES) + len(methods)), "requests_per_arm": requests},
        "scheduled_test": {"split": evaluation_split, "users": len(evaluation_users), "personal_programs": 2 * len(evaluation_users),
            "executor_requests": 5 * evaluation_requests, "sol_sessions": 0,
            "arms": [*BASELINES, "test_initial", "test_optimized"], "selection_feedback": False},
        "visibility": {"generation": "common prefix of nonzero-memory checkpoints; history only",
            "type1": "no memory, no personal hooks", "type2_type3": "independent checkpoint context; no later sessions",
            "same_day": "retain upstream session ordering when intra-day question time is unavailable",
            "private_fields": "profiles, preferences, affinity_links, task_goal and task_description excluded"}}
    if reference:
        for name in ("seed", "executor_model", "executor_url", "tokenizer_model", "tokenizer_revision", "model_cache",
                "input_tokens", "output_tokens", "score_workers", "concurrency", "optimizer"):
            plan[name] = previous[name]
        plan["phg"]["generator"] = previous["phg"]["generator"]
        plan["phg"]["limits"] = previous["phg"]["limits"]
    write(out / "plan.json", plan)
    write(out / "registration.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    return plan


async def run(out, execute=None):
    plan = read(out / "plan.json")
    await PHG.run_pipeline(out, execute=execute, entrypoint="perma_phg.py", score_stages=("score",),
        score_workers={"score": plan["score_workers"]}, baseline_arms=BASELINES, control_arms=(),
        diagnose_round=diagnose, summary=summarize,
        heldout_users=plan["test_users"] if plan["split"] != "TRAIN" else (),
        heldout_arms=("test_initial", "test_optimized") if plan["split"] != "TRAIN" else ())


def preflight(out):
    import urllib.request
    plan = read(out / "plan.json")
    PHG.load_generator_credentials(plan)
    PHG.MECHANISM.load_optimizer_credentials(plan)
    with urllib.request.urlopen(plan["executor_url"] + "/models", timeout=10) as response:
        models = json.load(response)
    if plan["executor_model"] not in {model["id"] for model in models["data"]}:
        raise ValueError("Registered executor model is not served")
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(plan["tokenizer_model"], revision=plan["tokenizer_revision"], cache_dir=plan["model_cache"])
    PHG.replace(out / "operations/preflight.json", {"utc": now(), "status": "ready"})


PHG.BASE.host = host
PHG.program_check = program_check
PHG.framework = framework
PHG.tool_script = tool_script
PHG.REFERENCES = BASELINES
check_generator_workspace = PHG.check_generator_workspace
