"""One fixed MUSE 1m 50-user TRAIN / 50-user TEST HyperHarness study."""
from __future__ import annotations

import ast
import fcntl
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import statistics

HERE = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location("muse_shared_adapter", HERE / "perma_phg.py")
BASE = importlib.util.module_from_spec(specification)
specification.loader.exec_module(BASE)
PHG = BASE.PHG
read, write, sha, now = PHG.read, PHG.write, PHG.sha, PHG.now
BASELINES = ("no_history", "bm25_10", "shared_control")
LEGACY_PROTOCOL = "muse-1m-hyperharness-train50-test50-v3"
COMBINED_LEGACY_PROTOCOL = "muse-1m-hyperharness-train50-test50-v4"
UNIFORM_LEGACY_PROTOCOL = "muse-1m-hyperharness-train50-test50-v5"
HIDDEN_LEGACY_PROTOCOL = "muse-1m-hyperharness-train50-test50-v6"
EVIDENCE_LEGACY_PROTOCOL = "muse-1m-hyperharness-train50-test50-v7"
GROUNDED_LEGACY_PROTOCOL = "muse-1m-hyperharness-train50-test50-v8"
PROTOCOL = "muse-1m-hyperharness-train50-test50-v9"
UNIFORM_BACKEND = "uniform-bm25-top10-v1"
PERSONALIZATION_POLICY = "source-grounded-personal-control-v1"
PROGRAM_EDIT_SCOPE = PHG.PROGRAM_EDIT_SCOPE
PERSONALIZATION_MODES = ("retrieval_only", "combined")
API = """# MUSE 1m HyperHarness contract
One history-only restricted Python program per user. Every released session from the 1m configuration is
available at construction and runtime; no history prefix truncation. A bounded digest is only a preview:
history tools can search the complete history. All 750 sessions, roles, contents, IDs and dates are retained.
The initial preview scans every session and is capped at160000characters, never a tail-only preview.
Session ordering is the released ordering, not necessarily chronological. Use explicit dates for temporal facts.
Persona profiles, questions/options, correct_label and answer_session_ids are NOT construction inputs.
Runtime query includes public query date/type, question and ALL five original options A-E. Option E is abstention.
All eight query types are evaluated: understand-personalized, understand-experience, apply-user-single,
apply-user-multi, apply-user-temporal, apply-assistant, evaluate-interference, evaluate-counterfactual.
Use relevant evidence; ignore irrelevant memories and reconcile contradictory or outdated preferences.
remember must keep the default complete observation store. No write hook customization. recall may use BM25,
read-only SQL and finish(output, source_refs). The registered intervention is retrieval only: apply is not run.
Only apply-user-multi uses cached, source-bound900character spans and match-centered snippets. The host searches the
public question and all five options independently, without option-label weighting, plus optional query terms.
Each component is split into128term searches (up to512uniformly sampled terms per component) and merged by RRF60.
For other tasks, preserve legacy full-record BM25. The initial recall bypasses memory for evaluate-interference;
six other types use the original BM25 top10/6000character policy. These routes were chosen on TRAIN only.
Collect all planned multi-task searches before selecting final evidence. route and keep receive the actual task kind.
Do not replace span snippets with record prefixes. The bm25_10 baseline retains its original backend.
guard must return {}. There is one final host-owned Qwen call, no intermediate model/decision calls or feedback writes.
Limits: read state64000chars, output12000chars,64source_refs,32memory operations,20000steps per hook;
write state8000chars, final Qwen32768inputtokens/32outputtokens. Full 1m history is stored, NOT put into one prompt.
The single fixed50TRAIN/50TEST split is a custom research split of an evaluation-only benchmark, not official.
Optimize complete-panel equal-user mean five-way accuracy on TRAIN only. TEST never selects or repairs anything.
No imports, classes, comprehensions, callbacks, while loops, filesystem or network in runtime programs.
"""
PROMPT = """# Build this user's executable MUSE memory retrieval program
Use only this user's full available conversation history. Infer useful retrieval concepts and uncertainty,
not answers or stereotypes. History includes distractors; preserve all observations and retrieve selectively.
Handle preferences, past experiences, temporal changes, assistant behavior, irrelevant memories and contradictions.
Use the full current request as the first retrieval query and treat A-E options symmetrically. Never select
by option position. Start from shared BM25 retrieval. user_keep should stay permissive without strong evidence.
Use dates in retrieved sessions rather than assuming released session order is chronological.
Write USER_MODEL, user_route(request,kind,p_none), user_keep(text,app,kind), user_guidance(request,kind,evidence),
u_recall(context), u_apply(context). Route returns history/queries/recent/per_record; recent may be None.
Keep user_guidance empty, u_apply empty, and preserve the default remember and empty guard scaffolding.
Return ONLY USER_MODEL and user_/u_ functions. Never redefine remember, recall, apply, guard or shared helpers:
those are already present in the backbone and duplicate definitions fail validation.
Use plain def functions with positional parameters only. No defaults, annotations, decorators, lambdas,
comprehensions, function values, imports or private attributes. Helpers start user_ or u_; assignments are literals.
Do not shadow any function or builtin name. After a syntax failure remove that unsupported syntax everywhere.
Return one Python block with USER_MODEL and user functions, not the backbone. Language: <<LANGUAGE>>.
Limits: <<MAX_CHARS>>characters, <<MAX_NODES>>AST nodes.
## Interface
```python
<<CONTROL_SECTION>>
```
## Backbone
```python
<<BACKBONE>>
```
## History preview (complete history remains available through tools)
<<DIGEST>>
"""


PERSONALIZATION_CONTRACT = """## Source-grounded personal control
Preserve original evidence by default. Omit per_record or set it to None unless a specific evidence-allocation
need justifies clipping. There is no fixed 1400-character ceiling: an explicit positive per_record is bounded
by BACKBONE['retrieval']['max_chars'], and all records together must still fit the registered total budget.
Do not shorten every user/assistant turn to a fixed prefix, summarize source text, or allocate arbitrary small
record caps merely to make the program look personalized. Keep dates, qualifications, later corrections and
the actual assistant recommendation when relevant. More records are not inherently better than intact evidence.

Learn executable rules from this user's construction history, not a generic personality description. Useful
rules resolve this user's aliases or recurring entities, locate an explicitly referenced earlier plan, or
distinguish a documented correction from an older context-specific preference. Associate every factual rule
with an observed session_id, date and supporting source excerpt; use actual retrieved ref values at runtime.
An observation event ID is not a storage ref. Do not invent facts or copy the illustrative categories as universal
rules. If the preview is insufficient, the generator may use its registered history tools before making a claim.
Distinguish user statements from assistant suggestions and statements about other people. Do not implement
latest-always-wins or generic caution as a substitute for evidence. Rule activation must use request_core(request)
and legal source evidence, never prompt boilerplate, option positions, task categories or evaluation feedback.
The personal module must contain genuinely model-written executable behavior, not only a profile dictionary.
No intervention for an unsupported request is valid; never manufacture differences or force a preference.
Any answer guidance must convey relevant source-supported user information rather than repeat stock advice.
Guidance remains subject to the registered personalization mode and length limit.

## Optional session completion inside the personal module
The host and shared backbone do NOT complete sessions after recall. Only generated personal code may decide
whether completion is useful, which matched session to expand and when to stop. Enable USER_MODEL.custom_recall
and implement u_recall(context) only when that program needs a custom retrieval workflow. With the flag absent
or false, the default recall remains unchanged. Do not enable completion for every user or request by default.
Missing context for an explicitly referenced earlier list or a documented later correction can justify a
targeted expansion; a vague topic match alone does not. Retain the original top10 query as the initial search.

Use read-only, parameterized SQL on records, not filesystem access or string slicing of JSON. text is the
original JSON containing session_id, date and messages; data contains the numeric ordinal. For example, to
page through the session of an already retrieved reference in original order, query:
SELECT ref, text, json_extract(data, '$.ordinal') AS ordinal FROM records
WHERE collection = 'observations'
AND json_extract(text, '$.session_id') =
    (SELECT json_extract(text, '$.session_id') FROM records WHERE ref = ?)
AND json_extract(data, '$.ordinal') > ?
ORDER BY json_extract(data, '$.ordinal') LIMIT 1
Bind parameters [matched_ref, last_ordinal], starting last_ordinal at -1. Read rows from
context['backend']['result']['data']['rows']; carry the cursor, accumulated text and deduplicated references
in memory_query.state. The next hook invocation advances the state machine; no while loop is required.
Select the matched_ref from this user's authorized search results; do not hard-code storage ordinals from
construction event IDs. Return only text and refs actually obtained from the store. Respect SQL response
truncation, read-state limits, the total evidence budget,64refs and32memory operations, including search and
finish. Stop before those limits, and do not claim complete recovery if expansion ended early. Session
completion adds local reads, not additional runtime model calls.

Both personal and neutral arms have the same query capabilities, source history and limits. The empty module
must remain plain BM25 top10. Moving a uniform hand-written completion routine into every personal program is
not itself evidence of user-specific value; evaluate the contribution of the history-generated module.
"""


def runtime_api(plan):
    mode = plan.get("personalization", "retrieval_only")
    if mode not in PERSONALIZATION_MODES:
        raise ValueError("Unsupported MUSE personalization mode")
    api = API
    if plan.get("task_type_visibility") == "hidden":
        start, end = api.index("Runtime query includes"), api.index("Use relevant evidence")
        api = api[:start] + (
            "Runtime query includes the date, question and ALL five original options A-E. Option E is abstention.\n"
            "Dataset task labels are evaluation metadata only. Runtime task_kind is None for all hooks;\n"
            "infer relevance and the needed behavior from the question and history, never a supplied task category.\n") + api[end:]
    if plan.get("retrieval", {}).get("backend") == UNIFORM_BACKEND:
        evidence_chars = plan.get("evidence_chars", plan.get("retrieval", {}).get("initial_output_chars", 6000))
        start, end = api.index("Only apply-user-multi"), api.index("guard must return")
        api = api[:start] + (
            "All task kinds use the same full-record BM25 backend, without host-side task routing or span expansion.\n"
            f"The initial shared program uses BM25 top10 and packs up to{evidence_chars}characters for every task kind.\n"
            "The empty personal module matches bm25_10; it never skips history based on a task label.\n"
            "Generated programs may learn retrieval and evidence selection within the registered limits.\n") + api[end:]
    if "evidence_chars" in plan:
        api = api.replace("output12000chars", f"output{plan['evidence_chars']}chars")
    if plan.get("personalization_policy") == PERSONALIZATION_POLICY:
        api += "\n" + PERSONALIZATION_CONTRACT
    if plan.get("program_edit_scope") == PROGRAM_EDIT_SCOPE:
        api = api.replace("remember must keep the default complete observation store. No write hook customization.",
            "remember may add source-bound derived records, but must retain every original observation unchanged.")
        api = api.replace("Retain the original top10 query as the initial search.",
            "The neutral program remains BM25 top10; the personal program may choose its own bounded query workflow.")
        api += "\n" + PHG.FULL_PROGRAM_CONTRACT
    if mode == "retrieval_only":
        return api
    return api.replace("The registered intervention is retrieval only: apply is not run.",
        "Retrieval, answer guidance and custom local workflows are evaluated. apply returns {'prefix': guidance}; "
        "only public request/history/evidence may inform guidance.").replace(
        "For other tasks, preserve legacy full-record BM25.",
        "Other task kinds use the legacy full-record BM25 backend, but generated code may change their local retrieval workflow.") + "\n" + PHG.WORKFLOW_CONTRACT


def host(out):
    native, official = BASE.host(out)
    native.PERSONALIZED_FRAMEWORK = runtime_api(read(out / "plan.json"))
    return native, official


def initial_spec(directory, *, personalization="combined", retrieval_backend=UNIFORM_BACKEND):
    from ohpu.benchmarks.muse import construction_digest
    from ohpu.benchmarks.muse_runtime import EVIDENCE_CHARS
    policy = PERSONALIZATION_POLICY if retrieval_backend == UNIFORM_BACKEND else None
    api = runtime_api({"personalization": personalization, "retrieval": {"backend": retrieval_backend},
        "task_type_visibility": "hidden", "evidence_chars": EVIDENCE_CHARS,
        "personalization_policy": policy, "program_edit_scope": PROGRAM_EDIT_SCOPE})
    BASE.initial_spec(directory, personalization="retrieval_only")
    PHG.enable_full_program(directory)
    backbone = ast.parse((directory / "backbone.py").read_text())
    for node in ast.walk(backbone):
        if isinstance(node, ast.Constant) and node.value == "ABCDEFGH":
            node.value = "ABCDE"
    replacements = ast.parse(RETRIEVAL_BACKBONE)
    if retrieval_backend == UNIFORM_BACKEND:
        replacements.body = [node for node in replacements.body
            if not isinstance(node, ast.FunctionDef) or node.name not in ("recall", "route_for")]
        replacements.body.extend(ast.parse(UNIFORM_RECALL).body)
    names = {node.name for node in replacements.body if isinstance(node, ast.FunctionDef)}
    backbone.body = [node for node in backbone.body if not isinstance(node, ast.FunctionDef) or node.name not in names]
    for node in backbone.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "BACKBONE" for target in node.targets):
            value = ast.literal_eval(node.value)
            value["retrieval"].update(top_k=12, max_chars=11000, max_queries=3)
            if retrieval_backend == UNIFORM_BACKEND:
                value["retrieval"].update(top_k=10, max_chars=EVIDENCE_CHARS)
            node.value = ast.parse(repr(value), mode="eval").body
    backbone.body.extend(replacements.body)
    rendered = ast.unparse(ast.fix_missing_locations(backbone))
    rendered = rendered.replace("CHANGE_MANIFEST =", "#@USER_SECTION@#\nCHANGE_MANIFEST =", 1)
    control = (directory / "control_section.py").read_text()
    (directory / "backbone.py").write_text(rendered + "\n")
    (directory / "control_program.py").write_text(rendered.replace("#@USER_SECTION@#", control) + "\n")
    guidance = PROMPT[PROMPT.index("Use only this user's"):PROMPT.index("Write USER_MODEL")]
    guidance = guidance.replace("Use the full current request as the first retrieval query and treat A-E options symmetrically.",
        "Choose retrieval using the visible question and treat A-E option contents symmetrically.")
    prompt = PHG.full_program_prompt(api, guidance)
    (directory / "prompt.md").write_text(prompt)
    (directory / "program_api.md").write_text(api)
    (directory / "history_adapter.py").write_text(inspect.getsource(construction_digest) +
        "\n\ndef digest(events):\n    return construction_digest(events)\n")


UNIFORM_RECALL = '''
def request_core(request):
    text = str(request or '')
    if 'User query:\\n' in text:
        text = text.split('User query:\\n', 1)[1]
    return text.split('\\nOptions:\\n', 1)[0].strip()[:2000]

def route_for(request, kind):
    route = user_route(request, kind, None) or {}
    settings = BACKBONE['retrieval']
    queries = [request]
    for query in route.get('queries') or []:
        text = str(query or '').strip()
        if text and text != request and '\\nOptions:\\n' not in text and text not in queries and len(queries) < settings['max_queries']:
            queries.append(text[:settings['query_chars']])
    per_record = settings['max_chars']
    if route.get('per_record'):
        per_record = min(max(int(route['per_record']), 1), settings['max_chars'])
    return {'history': bool(route.get('history', True)), 'queries': queries, 'per_record': per_record}

def recall(context):
    if context.get('stage') != 'context' or not context.get('backend'):
        return {}
    if USER_MODEL.get('custom_recall'):
        return u_recall(context)
    backend = context['backend']
    if backend.get('step', 0) == 0:
        route = route_for(str(context.get('query') or ''), context.get('task_kind'))
        if not route['history']:
            return finish('', [])
        state = {'queries': route['queries'][1:], 'kind': context.get('task_kind'),
            'texts': [], 'refs': [], 'size': 0, 'per_record': route['per_record']}
        return search(route['queries'][0][:8000], state)
    state = backend.get('state') or {}
    for row in (backend.get('result') or {}).get('data') or []:
        remaining = BACKBONE['retrieval']['max_chars'] - state['size'] - 1
        if remaining <= 0:
            break
        ref = row.get('ref')
        text = str(row.get('text') or '')
        app = row.get('app') or (row.get('data') or {}).get('app') or ''
        if not ref or ref in state['refs'] or not user_keep(text, str(app), state['kind']):
            continue
        text = text[:min(remaining, state['per_record'])]
        state['texts'].append(text)
        state['refs'].append(ref)
        state['size'] = state['size'] + len(text) + 1
    if state['queries'] and state['size'] < BACKBONE['retrieval']['max_chars'] - 1:
        query = state['queries'][0]
        state['queries'] = state['queries'][1:]
        return search(query, state)
    return finish('\\n'.join(state['texts']), state['refs'])
'''


RETRIEVAL_BACKBONE = '''
def route_for(request, kind):
    route = user_route(request, kind, None) or {}
    settings = BACKBONE['retrieval']
    queries = [request]
    for query in route.get('queries') or []:
        text = str(query or '').strip()
        if text and text != request and '\\nOptions:\\n' not in text and text not in queries and len(queries) < settings['max_queries']:
            queries.append(text[:settings['query_chars']])
    per_record = route.get('per_record') or 1400
    per_record = min(max(int(per_record), 400), 1400)
    return {'history': bool(route.get('history', True)), 'queries': queries, 'per_record': per_record}

def collect(state, rows):
    rank = 0
    for row in rows or []:
        rank = rank + 1
        ref = row.get('ref')
        if not ref:
            continue
        text = str(row.get('text') or '')
        app = row.get('app') or (row.get('data') or {}).get('app') or ''
        if not user_keep(text, str(app), state['kind']):
            continue
        if ref not in state['candidates']:
            if len(state['candidates']) >= 32:
                continue
            state['candidates'][ref] = {'score': 0, 'text': text[:state['per_record']]}
        state['candidates'][ref]['score'] = state['candidates'][ref]['score'] + 1 / (60 + rank)

def selected(state):
    ranked = []
    for ref in state['candidates']:
        value = state['candidates'][ref]
        ranked.append([value['score'], ref, value['text']])
    texts = []
    refs = []
    size = 0
    for candidate in sorted(ranked, reverse=True):
        remaining = BACKBONE['retrieval']['max_chars'] - size - 1
        if remaining < 200:
            break
        text = candidate[2][:remaining]
        texts.append(text)
        refs.append(candidate[1])
        size = size + len(text) + 1
    return finish('\\n'.join(texts), refs)

def legacy_recall(context):
    backend = context['backend']
    if backend.get('step', 0) == 0:
        return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations',
            'text': context.get('query', '')[:8000], 'limit': 10}}
    texts = []
    refs = []
    size = 0
    for row in backend.get('result', {}).get('data', []):
        remaining = 6000 - size - 1
        if remaining <= 0:
            break
        text = row.get('text', '')[:remaining]
        texts.append(text)
        refs.append(row['ref'])
        size = size + len(text) + 1
    return finish('\\n'.join(texts), refs)

def recall(context):
    if context.get('stage') != 'context' or not context.get('backend'):
        return {}
    if USER_MODEL.get('custom_recall'):
        return u_recall(context)
    if context.get('task_kind') == 'evaluate-interference':
        return finish('', [])
    if context.get('task_kind') != 'apply-user-multi':
        return legacy_recall(context)
    backend = context['backend']
    if backend.get('step', 0) == 0:
        route = route_for(str(context.get('query') or ''), context.get('task_kind'))
        if not route['history']:
            return finish('', [])
        state = {'queries': route['queries'][1:], 'per_record': route['per_record'],
            'kind': context.get('task_kind'), 'candidates': {}}
        return search(route['queries'][0], state)
    state = backend.get('state') or {}
    data = (backend.get('result') or {}).get('data')
    collect(state, data)
    if state['queries']:
        query = state['queries'][0]
        state['queries'] = state['queries'][1:]
        return search(query, state)
    return selected(state)
'''


def program_check(out, user, events, client=None, *, cached=False):
    from ohpu.benchmarks.muse_runtime import validate_harness
    plan = read(out / "plan.json")
    uniform = plan.get("retrieval", {}).get("backend") == UNIFORM_BACKEND
    def check(source):
        try:
            validate_harness(source, events, expose_query_type=plan.get("task_type_visibility") != "hidden",
                evidence_chars=plan.get("host_contract", {}).get("output_chars", 12000),
                **({"allow_memory_writes": True} if plan.get("program_edit_scope") == PROGRAM_EDIT_SCOPE else {}),
                **({"span_retrieval": False} if uniform else {}))
        except Exception as error:
            return str(error)[:2000]
        return None
    return check


def tool_script(out, function, *arguments):
    return ("import importlib.util\nimport sys\nfrom pathlib import Path\n"
        f"spec=importlib.util.spec_from_file_location('muse_tools',{str(out / 'muse_phg.py')!r})\n"
        "driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)\n"
        f"driver.{function}(Path({str(out)!r}),Path('.').resolve()," + ", ".join(arguments) + ")\n")


def framework(plan, native):
    remember_policy = ("remember may add source-bound derived records, but must retain every original observation unchanged."
        if plan.get("program_edit_scope") == PROGRAM_EDIT_SCOPE else "Do not customize remember.")
    return PHG.optimizer_framework(plan, task_api=runtime_api(plan), task_context=
        "MUSE 1m five-choice accuracy with equal user weighting, eight query types. TRAIN only. "
        "History-only construction, complete histories. Registered personalization: "
        + plan.get("personalization", "retrieval_only") + ". No TEST feedback, "
        "no persona profiles or answer provenance. Do not copy TRAIN user facts, IDs or answers into the generator. "
        "Read runtime_failures.json and fix contract errors first. Incomplete panels cannot replace complete parents. "
        "Do not raise host limits, add runtime models, or optimize a different split. " + remember_policy)


def generate(out, arm, user, **options):
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["phg"]["arms"]:
        raise ValueError("Unregistered MUSE generation")
    if plan["split"] == "TEST" and not (out / "selection.json").exists():
        raise ValueError("TEST requires frozen TRAIN selection")
    return PHG.generate_harness(out, arm, user, version=plan["phg"]["arms"][arm],
        events=PHG.construction_events(out, plan, user), check_factory=program_check, cached=False, **options)


def score(out, arm, user, *, client=None, tokenizer=None):
    plan = read(out / "plan.json")
    if (plan.get("task_type_visibility") == "hidden" and
            plan.get("retrieval", {}).get("backend") == UNIFORM_BACKEND and arm in ("bm25_10", "shared_control")):
        if user not in plan["test_users"] or arm not in plan["arms"]:
            raise ValueError("Scoring outside frozen MUSE cohort")
        operations = out / "operations"
        operations.mkdir(exist_ok=True)
        with (operations / ("bm25-" + user + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _score(out, "bm25_10", user, client=client, tokenizer=tokenizer)
            if arm == "shared_control":
                if sha(out / "inputs/shared_control.py") != plan["retrieval"]["initial_control_sha256"]:
                    raise ValueError("Initial shared program changed from the registered BM25 seed")
                target = out / "scores/shared_control" / user
                if not target.exists():
                    shutil.copytree(out / "scores/bm25_10" / user, target)
                    write(operations / "baseline_aliases" / (user + ".json"),
                        {"arm": "shared_control", "source_arm": "bm25_10", "new_model_calls": 0,
                            "source_summary_sha256": sha(target / "summary.json")})
        return
    return _score(out, arm, user, client=client, tokenizer=tokenizer)


def _score(out, arm, user, *, client=None, tokenizer=None):
    from ohpu.benchmarks.muse_runtime import QwenClient, answer_user, summarize_user
    from ohpu.benchmarks.perma_runtime import failure_details
    plan = read(out / "plan.json")
    if user not in plan["test_users"] or arm not in plan["arms"]:
        raise ValueError("Scoring outside frozen MUSE cohort")
    runtime_api(plan)
    directory = out / "scores" / arm / user
    if (directory / "summary.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=False)
    source = baseline = None
    try:
        if arm in ("no_history", "bm25_10"):
            baseline = arm
        elif arm == "shared_control":
            source = (out / "inputs/shared_control.py").read_text()
        elif arm in plan.get("shared_harness", {}).get("arms", {}):
            source = PHG.load("shared_harness_source", HERE / "shared_harness.py").program_source(out, arm)
        else:
            harness = read(out / "harness" / arm / user / "harness.json")
            if not harness["generation"]["valid"]:
                raise ValueError("Cannot run an invalid personal harness")
            source = harness["source"]
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
                write(directory / "failure.json", {**error, "arm": arm, "utc": now(),
                    "limits": {**error.get("limits", {}), **plan.get("host_contract", {})}})
            rows = answer_user(native, source=source, baseline=baseline, client=client, tokenizer=tokenizer,
                seed=plan["seed"], input_tokens=plan["input_tokens"], output_tokens=plan["output_tokens"],
                sink=sink, error_sink=error_sink,
                expose_query_type=plan.get("task_type_visibility") != "hidden",
                evidence_chars=plan.get("host_contract", {}).get("output_chars", 12000),
                baseline_evidence_chars=plan.get("evidence_chars", 6000),
                **({"allow_memory_writes": True} if plan.get("program_edit_scope") == PROGRAM_EDIT_SCOPE else {}),
                **({"span_retrieval": False} if plan.get("retrieval", {}).get("backend") == UNIFORM_BACKEND else {}),
                **({"apply_guidance": True} if plan.get("personalization") == "combined" else {}))
    except Exception as error:
        if not (directory / "failure.json").exists():
            write(directory / "failure.json", {**failure_details(error, user=user, stage="score"), "arm": arm, "utc": now()})
        raise
    provenance = {"shared_program_sha256": PHG.hashlib.sha256(source.encode()).hexdigest()} if arm in plan.get("shared_harness", {}).get("arms", {}) else {}
    write(directory / "summary.json", {**summarize_user(native, rows), **provenance, "user": user, "arm": arm,
        "personalization": plan.get("personalization", "retrieval_only"), "utc": now()})


def summarize(out):
    from ohpu.benchmarks.muse import OBJECTIVE
    from ohpu.benchmarks.perma import read_records
    plan = read(out / "plan.json")
    values = {arm: PHG.arm_scores(out, plan, arm) for arm in plan["arms"]}
    selected = "test_optimized" if plan["split"] == "TEST" else PHG.round_arm(plan, read(out / "selection.json")["version"])
    result = {"framework": "HyperHarness", "dataset": "muse", "configuration": "1m", "split": plan["split"],
        "personalization": plan.get("personalization", "retrieval_only"),
            "shared_baseline": PHG.shared_baseline_contract(),
        "utc": now(), "selection": read(out / "selection.json"), "objective": OBJECTIVE,
        "registered_users": len(plan["test_users"]), "arms": {}, "comparisons": {}, "ablation_comparisons": {}}
    for arm, scores in values.items():
        complete = set(scores) == set(plan["test_users"])
        rows = [row for user in scores for row in read_records(out / "scores" / arm / user / "records.jsonl")]
        result["arms"][arm] = {"users": len(scores), "mean": statistics.fmean(scores.values()) if complete else None,
            "available_user_mean": statistics.fmean(scores.values()) if scores else None,
            "micro_accuracy": statistics.fmean(row["accuracy"] for row in rows) if complete and rows else None,
            "query_types": {kind: {"queries": len(subset), "micro_accuracy": statistics.fmean(row["accuracy"] for row in subset)
                if complete else None} for kind in sorted({row["task_type"] for row in rows})
                if (subset := [row for row in rows if row["task_type"] == kind])}}
        if arm != selected and values[selected] and scores:
            comparison_group = "ablation_comparisons" if arm == "shared_control" else "comparisons"
            result[comparison_group][arm] = PHG.paired(values[selected], scores, selected + arm)
    write(out / "summary.json", result)
    return result


def register(out, prepared, host_source, *, iterations=10, reference=None,
        executor_url="http://127.0.0.1:8001/v1", model_cache="/opt/model-apis/huggingface",
    credential_file=PHG.GENERATOR["credential_file"], personalization=None):
    from ohpu.benchmarks import muse as data
    from ohpu.benchmarks.muse_runtime import CONTRACT_LIMITS
    out, prepared = data.require_local_storage(out), data.require_local_storage(prepared)
    model_cache = str(data.require_local_storage(model_cache))
    if out.exists() or not 1 <= iterations <= 10:
        raise ValueError("Use a fresh run and 1-10 optimization iterations")
    for name, digest in read(prepared / "registration.json")["files"].items():
        if sha(prepared / name) != digest:
            raise ValueError("Prepared MUSE data changed")
    summary, roles = read(prepared / "summary.json"), read(prepared / "roles.json")
    if summary["protocol"] != data.PROTOCOL or roles != data.roles(list(summary["users"]), seed=roles["seed"]):
        raise ValueError("Invalid MUSE source or fixed50/50 roles")
    previous = selected = None
    if reference:
        PHG.BASE.HISTORY.PILOT.verify(reference)
        previous, selected = read(reference / "plan.json"), read(reference / "selection.json")
        if (previous["split"] != "TRAIN" or previous["protocol"] not in (
                PROTOCOL, LEGACY_PROTOCOL, COMBINED_LEGACY_PROTOCOL, UNIFORM_LEGACY_PROTOCOL,
                HIDDEN_LEGACY_PROTOCOL, EVIDENCE_LEGACY_PROTOCOL, GROUNDED_LEGACY_PROTOCOL)
                or previous["prepared_registration_sha256"] != sha(prepared / "registration.json")
                or set(PHG.arm_scores(reference, previous, PHG.round_arm(previous, selected["version"]))) != set(roles["train"])):
            raise ValueError("TEST requires matching complete frozen TRAIN selection")
        inherited = previous.get("personalization", "retrieval_only")
        if personalization is not None and personalization != inherited:
            raise ValueError("TEST personalization must match the frozen TRAIN contract")
        personalization = inherited
    else:
        personalization = personalization or "combined"
    runtime_api({"personalization": personalization})
    users = roles["test"] if reference else roles["train"]
    out.mkdir(parents=True)
    shutil.copytree(reference / "source_snapshot/src" if reference else HERE.parent.parent / "src", out / "source_snapshot/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in PHG.DRV.HOST_FILES:
        shutil.copyfile((reference or host_source) / name, out / name)
    for name in (*PHG.DRV.CODE_FILES, "pm3_decision_harness.py", "pm3_generator_harness.py", "horizon_phg.py",
            "memorycd_phg.py", "perma_phg.py", "muse_phg.py"):
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
            "spec_sha256": PHG.spec_sha(out / "generator/v0"), "source": "fresh MUSE generator; no learned PERMA code"})
    (out / "inputs").mkdir()
    shutil.copyfile(out / "generator/v0/control_program.py", out / "inputs/shared_control.py")
    splits = {}
    for user in users:
        native = read(prepared / "users" / f"{user}.json")
        write(out / "private/native" / f"{user}.json", native)
        write(out / "private/users" / f"{user}.json", {"events": native["events"], "rows": []})
        splits[user] = {"construction_cutoff": len(native["events"]), "evaluated": [case["case_id"] for case in native["cases"]]}
    methods = {"test_initial": 0, "test_optimized": selected["version"]} if reference else {
        f"phg_r{version}": version for version in range(iterations + 1)}
    requests = sum(len(value["evaluated"]) for value in splits.values())
    plan = {"protocol": PROTOCOL,
        "framework": "HyperHarness", "dataset": "muse", "configuration": "1m",
        "created_utc": now(), "split": "TEST" if reference else "TRAIN", "seed": roles["seed"], "host_limit_scale": 1,
        "test_users": users, "heldout_users": [] if reference else roles["test"], "splits": splits,
        "personalization": personalization, "task_type_visibility": "hidden", "host_contract": dict(CONTRACT_LIMITS),
        "evidence_chars": CONTRACT_LIMITS["output_chars"],
        "personalization_policy": PERSONALIZATION_POLICY,
        "program_edit_scope": PROGRAM_EDIT_SCOPE,
        "construction_preview": {"policy": "scan all sessions; date-ordered source-only excerpts", "max_chars": 160000,
            "labels_or_answer_session_ids": False},
        "retrieval": {"backend": UNIFORM_BACKEND, "span_tasks": [], "initial_no_history_tasks": [],
            "initial_control_sha256": sha(out / "inputs/shared_control.py"),
            "initial_top_k": 10, "initial_output_chars": CONTRACT_LIMITS["output_chars"], "initial_query_chars": 8000,
            "routing_selection": "none; identical BM25 top10 initial policy for all task kinds",
            "baseline_bm25": "unchanged legacy full-record BM25"},
        "prepared": str(prepared), "prepared_registration_sha256": sha(prepared / "registration.json"),
        "data_protocol": data.PROTOCOL, "source": summary["source"], "arms": [*BASELINES, *methods], "objective": data.OBJECTIVE,
        "research_disclosure": summary["research_disclosure"], "history_policy": summary["history_policy"],
        "executor_model": "Qwen3.5-9B", "executor_url": executor_url,
        "executor_serving": {"data_parallel_size": 4, "tensor_parallel_size": 1, "visible_gpus": [0, 1, 2, 3], "host": "127.0.0.1"},
        "tokenizer_model": "Qwen/Qwen3.5-9B", "tokenizer_revision": BASE.BASE.TOKENIZER_REVISION,
        "model_cache": model_cache, "input_tokens": 32768, "output_tokens": 32, "score_workers": 2,
        "optimizer_model": "gpt-6-sol", "optimizer": {"credential_file": str(credential_file), "api_version": "2025-04-01-preview"},
        "external_baselines": {"arms": [], "files": {}},
        "phg": {"rounds": 0 if reference else iterations + 1, "arms": methods, "luna_workers": 2,
            "require_model_generated_code": True, "open_generation_contract": True, "decision_model_enabled": False,
            "decision_policy": "disabled", "generator": {**PHG.GENERATOR, "credential_file": str(credential_file),
                "request_timeout_seconds": 600, "transport_retries": 1,
                "retries": "one recorded transient retry; failed candidate slots never replayed"},
            "limits": {**PHG.LIMITS, "wall_seconds": 7800}, "dev_users": users[:2],
            "optimizer": {**PHG.OPTIMIZER, "model": "gpt-6-sol", "sessions": 0 if reference else iterations,
                "luna_relay_calls": None, "decision_relay_calls": 0}},
        "decision_model": {"calls_per_request": 0, "arms": [], "host_limits": {}},
        "study_design": {"baseline_groups": {"non_personalized": ["no_history"], "retrieval": ["bm25_10"]},
            "ablation_groups": {"fixed_shared_policy": ["shared_control"]}, "shared_baseline": PHG.shared_baseline_contract()},
        "selection_rule": "complete TRAIN panel; positive paired equal-user mean; ties retain parent; no TEST selection",
        "prestudy_gate": {"candidate": "shared_control", "reference": "bm25_10", "users": 50,
            "rule": "verify empty-personalization seed; no required improvement over the identical BM25 policy",
            "scope": "model-free initialization check; no TRAIN-selected initial task routing", "workers": 2},
        "failure_policy": "zero executor retries; transport failures invalidate panels; malformed letters score0; retain failed attempts",
        "concurrency": {"generation": 2, "executor": 2, "optimizer": 1,
            "optimizer_scope": "one shared generator; dependent iterations sequential, not per-user Sol sessions"},
        "monitoring": {"interval_seconds": 30, "stalled_seconds": 6000},
        "storage": {"runtime": "verified node-local", "backup_interval_seconds": 300,
            "publication": "local tar.gz; exclusive ASCII filename copy and SHA256 reread; no mount renames",
            "backup_failure": "stop supervised process group; preserve all artifacts"},
        "upper_bounds": {"personal_programs": len(users) * len(methods), "sol_sessions": 0 if reference else iterations,
            "executor_requests": requests * (2 + len(methods)), "requests_per_arm": requests},
        "scheduled_test": {"split": "TEST", "users": 50, "personal_programs": 100,
            "executor_requests": 4 * summary["split"]["test"]["queries"], "sol_sessions": 0,
            "arms": [*BASELINES, "test_initial", "test_optimized"], "selection_feedback": False},
        "visibility": {"generation": "complete sessions only, no questions/options/gold/profile/provenance",
            "runtime": "full released sessions plus query/date/options; no task category or answer feedback",
            "history_window": "1m dataset fully stored and searchable; final executor input budget32768tokens"}}
    if reference:
        if "program_edit_scope" in previous:
            plan["program_edit_scope"] = previous["program_edit_scope"]
        else:
            plan.pop("program_edit_scope", None)
        if "personalization_policy" in previous:
            plan["personalization_policy"] = previous["personalization_policy"]
        else:
            plan.pop("personalization_policy", None)
        plan["host_contract"] = previous["host_contract"]
        if "evidence_chars" in previous:
            plan["evidence_chars"] = previous["evidence_chars"]
        else:
            plan.pop("evidence_chars", None)
        plan["task_type_visibility"] = previous.get("task_type_visibility", "exposed")
        plan["phg"]["skip_baselines"] = previous["phg"].get("skip_baselines", False)
        plan["visibility"] = previous["visibility"]
        plan["upper_bounds"]["executor_requests"] = previous["scheduled_test"]["executor_requests"]
        for name in ("seed", "executor_model", "executor_url", "tokenizer_model", "tokenizer_revision", "model_cache",
            "input_tokens", "output_tokens", "score_workers", "concurrency", "optimizer", "protocol", "retrieval", "prestudy_gate"):
            plan[name] = previous[name]
        plan["phg"]["generator"], plan["phg"]["limits"] = previous["phg"]["generator"], previous["phg"]["limits"]
    write(out / "plan.json", plan)
    write(out / "registration.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    return plan


async def run(out, execute=None):
    plan = read(out / "plan.json")
    await PHG.run_pipeline(out, execute=execute, entrypoint="muse_phg.py", score_stages=("score",),
        score_workers={"score": plan["score_workers"]},
        baseline_arms=() if plan["phg"].get("skip_baselines") else BASELINES, control_arms=(),
        diagnose_round=BASE.diagnose, summary=summarize,
        heldout_users=plan["test_users"] if plan["split"] == "TEST" else (),
        heldout_arms=("test_initial", "test_optimized") if plan["split"] == "TEST" else ())


PHG.BASE.host = host
PHG.program_check = program_check
PHG.framework = framework
PHG.tool_script = tool_script
PHG.REFERENCES = BASELINES
check_generator_workspace = PHG.check_generator_workspace
