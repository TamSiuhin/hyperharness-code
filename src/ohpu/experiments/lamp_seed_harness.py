"""Blank-start, seed-assisted executable harness construction contracts."""
from __future__ import annotations

import ast



# Empty starting skill for learning a construction method from scratch; the host prompt still carries the full contract.


# Keys the host memory backend reads for each lookup operation (memory_backend.execute).
DECISION_KEYS = {"sql": {"operation", "sql", "parameters", "state"},
    "search": {"operation", "collection", "text", "vector", "limit", "algorithm", "state"}}


def validate_decisions(source):
    """Reject literal sql/search decisions with unregistered keys; they would fail only when that branch runs."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict) or None in node.keys:
            continue
        keys = {key.value for key in node.keys if isinstance(key, ast.Constant)}
        operation = next((value.value for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "operation" and isinstance(value, ast.Constant)), None)
        if operation in DECISION_KEYS:
            unknown = sorted(str(key) for key in keys - DECISION_KEYS[operation])
            if unknown or (operation == "sql" and "sql" not in keys):
                raise ValueError(f"memory_query {operation!r} uses keys {sorted(map(str, keys))}; registered keys are "
                    f"{sorted(DECISION_KEYS[operation])} (sql needs 'sql' and optional 'parameters')")
    return True


# Counterfactual probes change one registered USER_MODEL setting of an existing program; no model call builds them.


# A verified reference is a complete shared harness inside the skill package (memory_evolution.GENERATION_SKILL_REFERENCE).
