"""Register PersonaMem-v3 native requests for HyperHarness.

Every evaluated query is a released PM3 request run through the pinned official evaluator (its task builders, scorers
and judge, llm_memory mode). Harness programs see only the request text and causal history before each query: the task
kind is the constant pm3_native::request, never the fine-grained task type. Registration writes the private per-user
requests, the TRAIN/TEST roles and the evaluated-query split consumed by scripts/paper_pm3.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil

# The evaluator's sentence embedder must load identically in every process; on a GPU filled by vLLM it silently fell back.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent


def _load(name, *paths):
    path = next(path for path in paths if path.exists())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


CORE, CORE_PATH = _load("pm3_skillopt_core", HERE / "pm3_skillopt.py", HERE / "run_pm3_skillopt.py")
OFFICIAL_DRIVER, OFFICIAL_DRIVER_PATH = _load("pm3_official_driver", HERE / "pm3_official_driver.py", HERE / "run_pm3_official.py")
RUNNER = CORE.RUNNER
read, write, sha, now = RUNNER.read, RUNNER.write, RUNNER.sha, RUNNER.now

SEED = 31017


def _hidden(text, replacements):
    for old, new in replacements:
        if old not in text:
            raise ValueError("official contract text changed: " + old)
        text = text.replace(old, new)
    return text


PERSONALIZED_HOOKS = """
## Complete program interface

The interfaces below specify capabilities and hard limits, not preferred strategies. No hook, algorithm, memory
representation, retrieval depth or output policy is preferred in advance. Allocate behavior across all four hooks
from user evidence and measured task outcomes. An empty operation and a nonempty implementation are both allowed;
neither is a goal. Do not preserve a hook solely because a scaffold or an earlier skill left it unchanged.

Define all four hooks in harness.py:
- `remember(context)`: editable memory-write hook, called at stage "memory_write" once per causal history event.
    The host first inserts that event into observations, then supplies source_events with one item containing index,
    key, ref, ts, app, text and event (the complete sanitized structured source, with full content and conversations).
    Return {} at step 0 to keep this default, or return {"memory_backend": {...}} for
    schema, upsert, delete, clear, search, SQL and finish operations. Once started, return operations until finish.
    An upsert record supplies collection, key, data, text and source_refs from the event or existing retrieved records;
    a delete supplies collection and key. backend.step, backend.result.data and backend.state carry loop progress.
    You may filter, summarize or organize evidence, but cannot access future events, other users or call models.
- `recall(context)`: editable evidence-retrieval hook. At stage "context", use the request and backend step/result
    to return a memory_query for search, read-only SQL or finish. Finish returns output (<= 12000 characters) and
    source_refs for the records used. With no relevant evidence, finish with empty output and an empty source_refs list.
    Memory SQL (remember or recall) is one read-only SELECT over the table records(collection, key, data, text, sources,
    updated_day, expires_on, ref): at most 8,000 characters of SQL, at most 128 bound parameters (exactly one per
    placeholder), about a million SQLite steps of work, at most 64 returned rows, and only the functions abs, avg,
    coalesce, count, ifnull, instr, json_array_length, json_extract, json_type, length, like, lower, max, min, nullif,
    round, substr, sum and upper. Breaking a limit is a runtime violation. Official requests can be far longer and stores
    far larger than the samples and user_history.jsonl, so bound SQL size, parameters and work independently of both.
- `apply(context)`: editable generation-guidance hook. At stage "generation", return {"prefix": str} (<= 800
    characters) only when prefix is in context["base"]["expressible"], otherwise return {}. Preserve the requested format.
- `guard(context)`: editable output-checking hook. Inspect query, output and phase. At phase "validate", return
    {"violations": [str, ...]} to report constraints; at phase "postprocess", return {"output": str} to revise the
    answer deterministically, or {} to keep the host default. At postprocess, context["recalled"] holds the output
    recall finished with for this request, so a revision can use the evidence recall found. A provider-pinned format may skip the validation phase.
    PM3 disables automatic model repair: violations are recorded, and postprocessing is the available correction path.
    Do not invent personal facts or change the required format; the host's hard validators still take precedence.

Also define `PROGRAM_API_VERSION = 1`, `USER_MODEL` (a JSON-serialisable dict containing the evidence-grounded user
model and any settings the hooks read), and `CHANGE_MANIFEST` (a dict describing the implemented changes). These are
module-level fields, not additional callable hooks. All four hooks are editable within the host's limits.

## User-model integration

Use USER_MODEL as the single source of user-specific preferences and policy parameters. Hooks or their helpers must
read the relevant fields to make user-specific decisions; do not duplicate those values in hard-coded hook rules.
Reading only retrieval settings does not activate preference fields. Keep descriptive evidence separate from decision
fields, and identify each decision field's consuming hooks in CHANGE_MANIFEST. For a relevant input, changing a
decision field should change its consumer's decision without editing the hook code; irrelevant inputs may be unchanged.
Not every hook must use every field, and shared mechanics need not be user-specific. USER_MODEL is not automatically
injected into the executor prompt or persisted when mutated: durable memory updates go through remember's memory_backend.
"""
FRAMEWORK = _hidden(OFFICIAL_DRIVER.FRAMEWORK, [
    ("context[\"task_kind\"] == \"pm3_native::<task_type>\";",
        "context[\"task_kind\"] == \"pm3_native::request\" for every request (the task type is not revealed; infer it from the "
        "request text);"),
    ("apply(context): at context[\"stage\"] == \"generation\", context[\"task_kind\"] names the official task type; return",
        "apply(context): at context[\"stage\"] == \"generation\" (the task kind is again \"pm3_native::request\"), return")])
PERSONALIZED_FRAMEWORK = _hidden(FRAMEWORK, [
    ('harness.py defines PROGRAM_API_VERSION = 1, USER_MODEL (a JSON-serialisable dict you design; keep its "retrieval" entry\n'
        'with "mode", "top_k", "max_chars" if your recall reads it), CHANGE_MANIFEST, and four hooks: remember(context) and',
        'harness.py defines PROGRAM_API_VERSION = 1, USER_MODEL (a JSON-serialisable dict whose structure you design),\n'
        'CHANGE_MANIFEST, and four hooks: remember(context) and'),
    ("guard(context) are fixed (copy them unchanged); recall(context) and apply(context) are yours.",
        "guard(context), recall(context) and apply(context) are all editable."),
    ("(see the seeds for the allowed style)", "(see the interface scaffold for the allowed style)")]) + PERSONALIZED_HOOKS
PERSONALIZED_SCAFFOLD = '''PROGRAM_API_VERSION = 1
USER_MODEL = {}
CHANGE_MANIFEST = {}


def remember(context):
    return {}


def recall(context):
    if context.get("stage") != "context" or not context.get("backend"):
        return {}
    return {"memory_query": {"operation": "finish", "output": "", "source_refs": []}}


def apply(context):
    return {}


def guard(context):
    return {}
'''
CHECKER = """import json
import sys
sys.path.insert(0, {src!r})
from ohpu.experiments import pm3_official
source = open("harness.py").read()
history = [json.loads(line) for line in open("user_history.jsonl")]
requests = json.load(open("sample_requests.json"))
print(pm3_official.validate_program(source, editable_hooks={editable_hooks!r}), "ran",
    pm3_official.smoke_check(source, history, requests, hide_task_type=True, editable_hooks={editable_hooks!r},
        tool_model_calls={tool_model_calls!r}),
    "sample requests without errors")
"""


def host(out):
    return OFFICIAL_DRIVER.host(out)


def _register_user(job):
    """Registration worker: released queries, the evaluated split, and every evaluated query's official request."""
    out, evaluator_root, backend, uid, events, query_scope = job
    _, official = host(out)
    runtime = official.official_runtime(evaluator_root, backend, uid, judge=False)
    rows = runtime.load_queries() if query_scope == "all" else official.released_queries(runtime)
    evaluated, lifecycle = (rows, []) if query_scope == "all" else official.complete_split(rows)
    retired_queries = official.retired_query_ids(rows) if query_scope == "all" else []
    raw_rows = json.loads((Path(backend) / uid / "test.json").read_text())
    if query_scope == "all" and (len(rows) != len(raw_rows) or
            {row["query_id"] for row in rows} != {row["query_id"] for row in raw_rows}):
        raise ValueError("official loader did not preserve the complete raw query set: " + uid)
    preflight_errors = {}
    requests, unbuildable, clusters = {}, [], {}
    for row in evaluated:
        if row["query_id"] in retired_queries:
            continue
        preflight = {}
        try:
            request = official.capture_request(runtime, row, Path(out) / "private/capture" / uid, preflight=preflight)
        except Exception:
            if query_scope == "all":
                raise
            request = None
        if preflight.get("status") == "error" and query_scope == "all":
            error = str(preflight.get("error") or "").split(" | ", 1)[0]
            if error != "KeyError: 'source_timestamp'":
                raise RuntimeError("unexpected native preflight failure: " + error)
            preflight_errors[row["query_id"]] = error
        try:
            cluster = json.loads(row.get("instance_json") or "{}").get("cluster_id") or None
        except (TypeError, ValueError, AttributeError):
            cluster = None
        # A repetition cluster is dispatched once on its first row; sibling rows are answered from that dispatch's cache.
        if request and cluster:
            clusters.setdefault(cluster, request)
        elif cluster in clusters:
            request = clusters[cluster]
        if request:
            requests[row["query_id"]] = request
        else:
            unbuildable.append(row["query_id"])
    kept = evaluated if query_scope == "all" else [row for row in evaluated if row["query_id"] in requests]
    split = {"construction_cutoff": official.cutoff(events, int(kept[0]["ts"])), "evaluated": [row["query_id"] for row in kept],
        "dropped_unpaired_lifecycle": lifecycle, "excluded_unbuildable": [] if query_scope == "all" else unbuildable,
        "request_capture_missing": unbuildable, "expected_data_errors": preflight_errors,
        "raw_query_count": len(raw_rows), "construction_timestamp": int(kept[0]["ts"]),
        "official_retired_queries": retired_queries}
    return uid, rows, split, requests


def cohort_roles(ranked, train, test, previous=None):
    cohort = ranked[:train + test]
    if min(train, test) < 1 or len(cohort) != train + test:
        raise ValueError("not enough users for the requested cohort")
    roles = dict(previous or {})
    if set(roles) - set(cohort) or set(roles.values()) - {"train", "test"}:
        raise ValueError("previous cohort does not match the requested users")
    remaining = {"train": train - list(roles.values()).count("train"),
        "test": test - list(roles.values()).count("test")}
    if min(remaining.values()) < 0:
        raise ValueError("requested sizes would reassign an existing user")
    for index, uid in enumerate(cohort):
        if uid in roles:
            continue
        role = "train" if index % 4 in (0, 3) else "test"
        if not remaining[role]:
            role = "test" if role == "train" else "train"
        roles[uid] = role
        remaining[role] -= 1
    return {uid: roles[uid] for uid in cohort}


def register(out, *, data_root, evaluator_root, manifest_from, train=25, test=75, processes=16,
             query_scope="all", roles_from=None, optimizer_model="gpt-6-sol"):
    """Register native PM3 users, official requests and executor settings for HyperHarness."""
    out = Path(out)
    if query_scope not in ("all", "later_half") or min(train, test, processes) < 1:
        raise ValueError("invalid registration settings")
    if not isinstance(optimizer_model, str) or not optimizer_model.strip():
        raise ValueError("optimizer model must be a deployment name")
    if out.exists():
        raise RuntimeError("fresh directory required; do not replace an attempt")
    workspace = HERE.parent
    shutil.copytree(workspace / "src", out / "source_snapshot/src", ignore=shutil.ignore_patterns("__pycache__"))
    for source, name in ((Path(__file__), "pm3_native_skillopt.py"), (CORE_PATH, "pm3_skillopt.py"), (CORE.PM3_PATH, "pm3_harness.py"),
            (OFFICIAL_DRIVER_PATH, "pm3_official_driver.py"), (CORE.PM3.RUNNER_PATH, "runner.py"), (RUNNER.BASE_PATH, "base_runner.py")):
        shutil.copyfile(source, out / name)
    loader, official = host(out)
    backend = Path(data_root) / "backend"
    corpus = loader.load(str(data_root), strip_leaks=True)
    events = {str(user.user_id): official.events(user) for user in corpus.users}
    ranked = sorted(events, key=lambda uid: (-len(events[uid]), hashlib.sha256(f"pm3-cohort-v1:{uid}".encode()).hexdigest()))
    previous_roles = read(Path(roles_from) / "plan.json")["roles"] if roles_from is not None else None
    roles = cohort_roles(ranked, train, test, previous_roles)
    cohort = list(roles)
    jobs = [(str(out), str(evaluator_root), str(backend), uid, events[uid], query_scope) for uid in cohort]
    with multiprocessing.get_context("fork").Pool(processes) as pool:
        results = pool.map(_register_user, jobs)
    splits = {}
    for uid, rows, split, requests in results:
        splits[uid] = split
        write(out / "private/users" / f"{uid}.json", {"user_id": uid, "events": events[uid], "rows": rows})
        write(out / "private/requests" / f"{uid}.json", requests)
    executor = read(Path(manifest_from) / "manifest.json")
    write(out / "manifest.json", {key: executor.get(key) for key in ("executor_model", "executor_base_url", "executor_thinking",
        "executor_sampling_profile", "executor_repetition_penalty", "model_request_timeout_seconds", "model_connection_keepalive")})
    train_users = [uid for uid in cohort if roles[uid] == "train"]
    test_users = [uid for uid in cohort if roles[uid] == "test"]
    evaluated = {role: sum(len(splits[uid]["evaluated"]) for uid in users) for role, users in (("train", train_users), ("test", test_users))}
    plan = {"created_utc": now(), "data_root": str(data_root), "evaluator_root": str(evaluator_root),
        "evaluator_revision": official.pm3_native.OFFICIAL_EVALUATOR_REVISION, "judge_model": official.JUDGE_MODEL,
        "optimizer_model": optimizer_model, "seed": SEED, "roles": roles, "train_users": train_users, "test_users": test_users,
        "splits": splits, "evaluated_queries": evaluated, "query_scope": query_scope,
        "roles_from": str(roles_from) if roles_from else None,
        "roles_source_sha256": sha(Path(roles_from) / "plan.json") if roles_from else None}
    write(out / "plan.json", plan)
    write(out / "registration.json", {"created_utc": now(), "files": {str(path.relative_to(out)): sha(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "registration.json"}})
    return plan


def executor(out):
    RUNNER.load_credentials()
    client = RUNNER.base.executor_client(out, read(Path(out) / "manifest.json"))
    client.cache = None
    return client


def judge(plan):
    from ohpu.benchmarks import pm3_native
    from ohpu.llm import LLM

    return pm3_native.PrivateJudgeQueryLLM(LLM(model=plan["judge_model"], cache=None), model=plan["judge_model"])


def embedder_device():
    """The official similarity metric silently falls back to Jaccard when its sentence embedder fails to load; refuse that."""
    from evaluation import metrics

    if not metrics.has_embeddings():
        raise RuntimeError("official evaluator embedder failed to load; refusing its lexical fallback")
    return str(metrics._EMBED_MODEL.device)


def main():
    parser = argparse.ArgumentParser(description="Register native PersonaMem-v3 inputs for HyperHarness.")
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("--out", type=Path, required=True)
    registration.add_argument("--data-root", type=Path, required=True)
    registration.add_argument("--evaluator-root", type=Path, required=True)
    registration.add_argument("--manifest-from", type=Path, required=True, help="Directory whose manifest.json holds the executor settings.")
    registration.add_argument("--optimizer-model", default="gpt-6-sol")
    registration.add_argument("--train-users", type=int, default=25)
    registration.add_argument("--test-users", type=int, default=75)
    registration.add_argument("--query-scope", choices=("later_half", "all"), default="all")
    registration.add_argument("--roles-from", type=Path)
    options = parser.parse_args()
    plan = register(options.out, data_root=options.data_root, evaluator_root=options.evaluator_root,
        manifest_from=options.manifest_from, train=options.train_users, test=options.test_users,
        query_scope=options.query_scope, roles_from=options.roles_from, optimizer_model=options.optimizer_model)
    print(json.dumps({key: plan[key] for key in ("train_users", "test_users", "evaluated_queries")}), flush=True)


if __name__ == "__main__":
    main()
