"""Host-owned editing boundaries for controlled personal-harness experiments."""
from __future__ import annotations

import ast


MARKER = "#@USER_SECTION@#"
SCOPES = ("section", "full")


def definitions(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            raise ValueError("Permission experiment requires plain top-level functions and single-name assignments")
        if name in result:
            raise ValueError(f"Duplicate top-level definition: {name}")
        result[name] = node
    return result


def validate_program(source, backbone, scope):
    if scope not in SCOPES:
        raise ValueError("Unknown personal-harness editing scope")
    if scope == "full":
        return
    if backbone.count(MARKER) != 1:
        raise ValueError("Section scope requires exactly one registered personal-section marker")
    shared = definitions(backbone)
    generated = definitions(source)
    for name, node in shared.items():
        if name not in generated or ast.dump(generated[name]) != ast.dump(node):
            raise ValueError(f"Section scope forbids replacing shared definition: {name}")
    shared_data = {name for name, node in shared.items() if not isinstance(node, ast.FunctionDef)}
    for name, node in generated.items():
        if name in shared:
            continue
        if name != "USER_MODEL" and not name.startswith(("user_", "u_")):
            raise ValueError(f"Section scope requires USER_MODEL or user_/u_ names: {name}")
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in shared_data:
                raise ValueError(f"Personal section cannot access shared global data directly: {child.id}")
            if isinstance(child, ast.Attribute) and child.attr.startswith("__"):
                raise ValueError("Personal section cannot use runtime introspection")


def scope_contract(scope, contract="v1"):
    if scope not in SCOPES:
        raise ValueError("Unknown personal-harness editing scope")
    if scope == "section" and contract == "v2":
        return ("Generate only the personal section: USER_MODEL and user_/u_ functions or helper data. "
                "The host assembles it into the shared v2 backbone, which turns user_route and USER_MODEL['config'] "
                "into a bounded per-request plan (query mode, extra queries, top_k, evidence and per-record budgets, "
                "fusion, recent view, guidance budget) and applies user_keep, user_weight and user_guidance. "
                "Do not redefine shared functions, constants, or entrypoints, or access shared global data. "
                "This restriction applies to per-user programs, not to the outer optimizer's shared spec edits.")
    if scope == "section":
        return ("Generate only the personal section: USER_MODEL and user_/u_ functions or helper data. "
                "The host assembles it into the current shared backbone. Do not redefine shared functions, "
                "constants, or entrypoints, or access shared global data. Existing user_route, user_keep, "
                "user_guidance, u_recall and u_apply extension points remain available when provided by the backbone. "
                "This restriction applies to per-user programs, not to the outer optimizer's shared spec edits.")
    return ("You may generate the complete executable personal program, replacing the example backbone's "
            "functions, constants and control flow, or voluntarily use its personal extension points. "
            "Complete-program capability is optional, not mandatory. Preserve the host runtime entrypoints.")