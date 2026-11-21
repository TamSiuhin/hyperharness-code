"""Shared registration, harness validation and scoring helpers for the paper drivers."""
import ast
import hashlib
import importlib.util
import json
import random
import statistics
import sys


HOOK_MODEL_TOOLS = """
## Registered hook model calls

This study lets recall, apply and guard each make at most ONE extra executor-model call per request, beyond the single
answer call; remember cannot call models. Per request there are therefore at most 4 executor calls. Exceeding a hook's
budget is a runtime violation and the request is left unanswered. Every extra call is optional, not a preferred strategy.
- recall: at any recall step you may return {"memory_query": {"operation": "model", "prompt": str (<= 8000 characters),
    "input": JSON value (<= 32000 serialized characters), "schema": optional JSON schema}}. The executor answers once (at most
    2048 tokens) treating input as memory evidence; the next recall step receives backend.result.data with text, data,
    valid_json and truncated. backend.model_calls_remaining shows the budget. Model output is not evidence: a finished
    recall must still cite source_refs of the retrieved records it relies on.
- apply: at stage "generation" you may also return "workflow" (context["base"]["parameter_domains"]["workflow"]):
    "single_call" (default, answer only); "plan_execute" with optional "plan_prompt" (one planning call, then the answer);
    "refine" with optional "refine_prompt" (the answer, then one rewrite call whose output replaces it); or "program".
    With "program", apply is then called at stage "step" (context: query, state, observations, step_index, memory_records,
    base.model_calls_remaining) and returns {"operation": "analyze"|"generate"|"select_memory"|"finish", ...}: analyze
    takes "prompt" (<= 4000 characters) and makes one internal JSON analysis call over the request and memory; generate makes
    the answer call with an optional "prompt"; finish may give "final_output"; "state" (<= 8000 characters JSON) persists.
    Model results arrive in observations. apply has 2 calls including the answer, so "plan_execute_refine" or more than
    one analyze/generate beyond the answer exceeds the budget.
- guard: at phase "validate", nonempty {"violations": [...]} triggers one executor repair call that rewrites the answer;
    an optional "repair_prompt" (<= 800 characters) replaces the generic repair instruction. Postprocess then runs on the
    repaired output. Return no violations to skip the call.
"""


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host(out):
    sys.path.insert(0, str(out / "source_snapshot/src"))
    adapter = load(out / "lamp_pm3.py", "local_skill_adapter")
    return adapter, adapter.load_native(out)


def local_delta(before, after, edit_scope="one_function"):
    if edit_scope not in ("one_function", "program"):
        raise ValueError("Unknown edit scope: " + edit_scope)
    def parts(source):
        functions, fixed = {}, []
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = ast.dump(node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("USER_MODEL", "CHANGE_MANIFEST"):
                ast.literal_eval(node.value)
            else:
                fixed.append(ast.dump(node))
        return functions, fixed
    original, old_fixed = parts(before)
    revised, new_fixed = parts(after)
    changed = sorted(name for name in original.keys() | revised.keys() if original.get(name) != revised.get(name))
    if edit_scope == "program":
        def contract(source):
            hooks, version = {}, None
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name in ("remember", "recall", "apply", "guard"):
                    if node.name in hooks:
                        raise ValueError("Duplicate hook definition")
                    hooks[node.name] = ast.dump(node.args)
                elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PROGRAM_API_VERSION" for target in node.targets):
                    if version is not None:
                        raise ValueError("Duplicate PROGRAM_API_VERSION")
                    version = ast.literal_eval(node.value)
            if set(hooks) != {"remember", "recall", "apply", "guard"} or version is None:
                raise ValueError("Program edits must preserve the four-hook API")
            return hooks, version

        if contract(before) != contract(after):
            raise ValueError("Program edits must preserve hook signatures and PROGRAM_API_VERSION")
        return changed
    if old_fixed != new_fixed or original.keys() != revised.keys() or len(changed) > 1:
        raise ValueError("Local edit may change USER_MODEL/CHANGE_MANIFEST and at most one existing function")
    return changed


def verify(out):
    if any(sha(out / name) != digest for name, digest in read(out / "registration.json")["files"].items()):
        raise ValueError("Pilot registration changed")


def records(out, arm, user):
    return {row["query_id"]: row for row in map(json.loads, (out / "scores" / arm / user / "records.jsonl").read_text().splitlines())}


def paired(values):
    generator = random.Random(37)
    distribution = sorted(statistics.fmean(generator.choices(values, k=len(values))) for _ in range(4000))
    return {"mean_gain": statistics.fmean(values), "lower_bound_95": distribution[200],
        "better": sum(value > 0 for value in values), "worse": sum(value < 0 for value in values)}
