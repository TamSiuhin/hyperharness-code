"""Manage frozen run artifacts and PersonaMemv3 TRAIN and TEST execution."""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
from datetime import datetime, timezone


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def function_source(text, name):
    return next(ast.get_source_segment(text, node) for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == name)


def verify(root, *, include=None):
    for name, expected in read(root / "registration.json")["files"].items():
        if include is not None and not include(Path(name)):
            continue
        if sha(root / name) != expected:
            raise ValueError("Registered file changed: " + str(root / name))


def copy_tree(source, target, hashes):
    def copy_file(original, destination):
        shutil.copyfile(original, destination)
        digest = sha(Path(original))
        if sha(Path(destination)) != digest:
            raise ValueError("Experiment copy mismatch")
        hashes[str(original)] = digest
        return destination
    shutil.copytree(source, target, copy_function=copy_file,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.sqlite*", "official", "capture"))


def freeze(root):
    excluded = {"registration.json", "rounds/parent.json", "diagnostics/summary.json", "results.csv"}
    write(root / "registration.json", {"files": {str(path.relative_to(root)): sha(path)
        for path in root.rglob("*") if path.is_file() and str(path.relative_to(root)) not in excluded
        and "operations" not in path.relative_to(root).parts}})
    verify(root)


def complete(root, arm, users):
    for user in users:
        path = root / "scores" / arm / user / "summary.json"
        if not path.exists():
            return False
        row = read(path)
        if not row.get("evaluation_complete") or row.get("runtime_violations"):
            return False
    return True


def register_pm3_test(train, out, *, experiment_key="reference_pipeline"):
    if out.exists():
        raise ValueError("Use a fresh PM3 TEST directory")
    plan = read(train / "plan.json")
    selected = read(train / "selection.json")
    binding = plan[experiment_key]["native_test_reference"]
    native_root = Path(binding["root"])
    native_inputs = {"plan.json", "registration.json", "manifest.json"}
    native_inputs.update(f"private/{kind}/{user}.json"
        for user in plan["heldout_users"] for kind in ("users", "requests"))
    for name, digest in binding["files"].items():
        if name in native_inputs and sha(native_root / name) != digest:
            raise ValueError("Native PM3 TEST reference changed")
    native = read(native_root / "plan.json")
    users = native["test_users"]
    if users != plan["heldout_users"] or set(users) & set(plan["test_users"]):
        raise ValueError("PM3 held-out split changed")
    if not complete(train, f"phg_r{selected['version']}", plan["test_users"]):
        raise ValueError("Selected TRAIN program must have a complete panel before TEST")
    hashes = {}
    out.mkdir(parents=True)
    for name in ("source_snapshot", "inputs"):
        copy_tree(train / name, out / name, hashes)
    for path in train.iterdir():
        if path.is_file() and (path.suffix == ".py" or path.name == "manifest.json"):
            shutil.copyfile(path, out / path.name)
    for version in sorted({0, selected["version"]}):
        copy_tree(train / "generator" / f"v{version}", out / "generator" / f"v{version}", hashes)
        shutil.copyfile(train / "generator" / f"v{version}.json", out / "generator" / f"v{version}.json")
    splits = {}
    for user in users:
        splits[user] = native["splits"][user]
        expected = set(splits[user]["evaluated"])
        private = read(native_root / "private/users" / f"{user}.json")
        rows = [row for row in private["rows"] if row["query_id"] in expected]
        if len(rows) != len(expected):
            raise ValueError("PM3 TEST queries missing")
        requests = read(native_root / "private/requests" / f"{user}.json")
        write(out / "private/users" / f"{user}.json", {**private, "rows": rows})
        write(out / "private/requests" / f"{user}.json", {key: requests[key] for key in sorted(expected)})
    methods = {"phg_test_r0": 0}
    if selected["version"]:
        methods[f"phg_test_r{selected['version']}"] = selected["version"]
    test_plan = copy.deepcopy(plan)
    test_plan.update(split="TEST", created_utc=now(), test_users=users, heldout_users=[], splits=splits,
        arms=list(methods), active_arms=list(methods), comparisons={},
        arm_roles=dict.fromkeys(methods, "phg"), external_baselines={"arms": [], "files": {}},
        concurrency={**plan["concurrency"], "optimizer": 0})
    test_plan["phg"].update(rounds=0, arms=methods, dev_users=[], start_round=0)
    test_plan["phg"]["optimizer"]["sessions"] = 0
    test_plan["upper_bounds"] = {"sol_sessions": 0, "personal_programs": len(users) * len(methods),
        "query_evaluations": sum(len(value["evaluated"]) for value in splits.values()) * len(methods)}
    write(out / "plan.json", test_plan)
    write(out / "selection.json", selected)
    write(out / "rounds/parent.json", {"version": selected["version"], "history": []})
    freeze(out)


def run_pm3(out, *, experiment_key="reference_pipeline"):
    verify(out)
    sys.path.insert(0, str(out / "source_snapshot/src"))
    specification = importlib.util.spec_from_file_location("pm3_driver", out / "pm3_generator_harness.py")
    phg = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(phg)
    phg.REFERENCES = ()
    plan = read(out / "plan.json")
    phg.DRV.host_limit_environment(out)
    test = plan["split"] == "TEST"
    personal_arms = tuple(plan["phg"]["arms"])

    def summary(root):
        selected = read(root / "selection.json")
        prefix = "phg_test_r" if test else "phg_r"
        selected_arm = prefix + str(selected["version"])
        initial_arm = prefix + "0"
        values = {arm: phg.arm_scores(root, plan, arm) for arm in personal_arms}
        result = {"utc": now(), "method": "personalized_hyperharness_generator", "dataset": "pm3", "split": plan["split"],
            "registered_users": len(plan["test_users"]), "selection": selected, "arms": {}, "comparisons": {},
            experiment_key: plan[experiment_key], "objective": "selected minus this experiment's personal r0"}
        for arm, scores in values.items():
            result["arms"][arm] = {"users": len(scores), "complete": len(scores) == len(plan["test_users"]),
                "mean": statistics.fmean(scores.values()) if len(scores) == len(plan["test_users"]) else None,
                "available_user_mean": statistics.fmean(scores.values()) if scores else None}
            if arm != selected_arm and scores and values[selected_arm]:
                result["comparisons"][arm] = phg.paired(values[selected_arm], scores, selected_arm + "_minus_" + arm)
        result["initial_arm"] = initial_arm
        phg.replace(root / "summary.json", result)
        return result

    phg.replace(out / "operations/controller.json", {"pid": os.getpid(), "utc": now(), "method": "personalized_hyperharness_generator"})
    asyncio.run(phg.run_pipeline(out, baseline_arms=(), control_arms=(),
        heldout_users=plan["test_users"] if test else (),
        heldout_arms=personal_arms if test else (), summary=summary))
