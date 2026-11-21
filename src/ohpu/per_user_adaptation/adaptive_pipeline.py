"""Versioned recall/apply action policy with public features and numeric learning.

All evidence comes from the current user's causal memory. The host supplies
candidate actions; a pure program chooses one using a small utility model.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import threading
from collections import OrderedDict
from pathlib import Path

from . import retrieval_policy
from .memory_actions import LocalSemanticEncoder, action_memory

PROFILE = "adaptive-pipeline-v1"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ACTIONS = ("shared_default", "none_default", "hybrid_default",
           "shared_grounded", "none_grounded", "hybrid_grounded")
FEATURES = ("intercept", "query_length", "memory_size", "lexical_best",
            "lexical_margin", "selection_overlap", "shared_fraction",
            "hybrid_fraction", "personal_reference", "explicit_optout")
GROUNDED_GUIDANCE = (
    "Answer the current request directly. Use remembered personal information only "
    "when it materially helps this request and is supported by the supplied evidence. "
    "Do not infer additional personal attributes. Respect explicit instructions to "
    "avoid personalization. Incorporate relevant constraints naturally; avoid "
    "unnecessary announcements about remembering the user. If evidence is absent "
    "or irrelevant, give a helpful answer without adding personal claims."
)
_ENCODERS = {}
_INPUTS = OrderedDict()
_LOCK = threading.RLock()


def encoder_for(config):
    key = (config["encoder_path"], json.dumps(config["encoder_identity"], sort_keys=True))
    with _LOCK:
        if key not in _ENCODERS:
            encoder = LocalSemanticEncoder(config["encoder_path"])
            if encoder.identity != config["encoder_identity"]:
                raise ValueError("pipeline encoder identity changed")
            _ENCODERS[key] = encoder
        return _ENCODERS[key]


def inputs(query, memory, encoder, *, baseline_mode="shared", feature_profile=None):
    from .executor import complete_memory_records
    from ..benchmarks.private_measurements import digest
    if baseline_mode not in {"shared", "base"}:
        raise ValueError("invalid pipeline baseline mode")
    feature_names(feature_profile)
    key = digest([query, memory, encoder.identity, baseline_mode, feature_profile])
    # Encoding/cache mutation is serialized; model generation remains parallel.
    with _LOCK:
        if key in _INPUTS:
            _INPUTS.move_to_end(key)
            return _INPUTS[key]
        records = complete_memory_records(memory)
        choices = {}
        for name, action in (("shared", "corrected_shared"), ("none", "no_memory"), ("hybrid", "hybrid")):
            selected, audit = action_memory(action, query, memory, encoder=encoder)
            if name == "hybrid":
                # A distinct, predeclared precision-oriented intervention.
                ids = audit["selected_record_ids"][:6]
                from .executor import ordered_memory_records
                selected = ordered_memory_records(memory, ids)[0] if ids else ""
                audit = {**audit, "selected_record_ids": ids, "selected_records": len(ids)}
            ids = audit["selected_record_ids"]
            recall = {"memory_mode": "select", "memory_fact_ids": ids} if ids else {"memory_mode": "no_user_memory"}
            choices[name] = {"memory": selected, "recall": recall, "audit": audit}
        ranked = retrieval_policy.rank_records(query, records, .75)
        scores = [s for s, _ in ranked] + [0., 0.]
        left, right = (set(choices[n]["audit"]["selected_record_ids"]) for n in ("shared", "hybrid"))
        features = [1., min(len(query) / 4000., 1.), min(len(memory) / 12000., 1.),
                    scores[0], scores[0] - scores[1], len(left & right) / max(1, len(left | right)),
                    len(left) / max(1, len(records)), len(right) / max(1, len(records)),
                    float(bool(re.search(r"\b(my|mine|me|prefer|favorite)\b|我的|我喜欢", query, re.I))),
                    float(bool(re.search(r"do not personali[sz]e|don.t use.*(history|memory)|without.*personal|不要.*(个性化|历史|记忆)", query, re.I)))]
        candidates = []
        for action in ACTIONS:
            name, style = action.split("_")
            c = choices[name]
            candidates.append({**c, "apply": {"prefix": GROUNDED_GUIDANCE} if style == "grounded" else {}})
        if baseline_mode == "base":
            # Preserve the original out-of-scope control. Keep the other five
            # interventions and the TRAIN feature representation unchanged.
            candidates[0] = {"memory": memory, "recall": {"memory_mode": "base"}, "apply": {},
                "audit": {"selected_record_ids": [r["id"] for r in records],
                          "selected_records": len(records), "reason": "original_out_of_scope_baseline"}}
        value = {"features": features, "candidates": candidates}
        if feature_profile:
            from . import evidence_features
            value = evidence_features.augment(query, records, encoder, value)
        _INPUTS[key] = value
        while len(_INPUTS) > 256:
            _INPUTS.popitem(last=False)
        return value


def feature_names(profile=None):
    if profile is None:
        return FEATURES
    from . import evidence_features
    if profile != evidence_features.PROFILE:
        raise ValueError("unknown pipeline feature profile")
    return evidence_features.feature_names(FEATURES)


def validate_model(model):
    if not isinstance(model, dict) or set(model) not in ({"coefficients", "minimum_gain"}, {"coefficients", "minimum_gain", "feature_profile"}):
        raise ValueError("invalid pipeline model fields")
    names = feature_names(model.get("feature_profile"))
    matrix = model["coefficients"]
    if not isinstance(matrix, list) or len(matrix) != len(ACTIONS):
        raise ValueError("invalid pipeline action count")
    values = [model["minimum_gain"]]
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(names):
            raise ValueError("invalid pipeline feature count")
        values.extend(row)
    if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 10 for v in values):
        raise ValueError("pipeline model must contain bounded finite numbers")
    if not 0 <= model["minimum_gain"] <= 1:
        raise ValueError("invalid minimum predicted gain")
    return model


def choose(model, features):
    if len(features) != len(feature_names(model.get("feature_profile"))) or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in features):
        raise ValueError("invalid pipeline decision features")
    best, value = 0, model["minimum_gain"]
    for i, weights in enumerate(model["coefficients"]):
        score = sum(x * w for x, w in zip(features, weights))
        if score > value:
            best, value = i, score
    return best


_SOURCE = '''PROGRAM_API_VERSION = 1
USER_MODEL = MODEL_LITERAL
CHANGE_MANIFEST = {"recall": "Choose evidence by predicted request-level gain.", "apply": "Choose grounded guidance by the same predicted action."}

def select_action(context):
    if not context.get("adaptation_enabled", True):
        return 0
    features = context["pipeline_features"]
    best = 0
    value = USER_MODEL["minimum_gain"]
    for i in range(len(USER_MODEL["coefficients"])):
        score = 0.0
        for j in range(len(features)):
            score += features[j] * USER_MODEL["coefficients"][i][j]
        if score > value:
            best = i
            value = score
    return best

def remember(context):
    return {}

def recall(context):
    if context.get("stage") != "context":
        return {}
    return context["pipeline_candidates"][select_action(context)]["recall"]

def apply(context):
    if context.get("stage") != "generation":
        return {}
    return context["pipeline_candidates"][select_action(context)]["apply"]

def guard(context):
    return {}
'''


def source(model):
    return _SOURCE.replace("MODEL_LITERAL", json.dumps(validate_model(model), allow_nan=False))


def validate_source(text):
    tree = ast.parse(text)
    model = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "USER_MODEL":
            model = validate_model(ast.literal_eval(node.value))
    if model is None or ast.dump(tree) != ast.dump(ast.parse(source(model))):
        raise ValueError("pipeline program may change only its numeric utility model")
    return model


def execution_policy(scope, adaptation_scope, encoder_path, encoder_identity, *, public_stimulus=False, feature_profile=None):
    config = {"profile": PROFILE, "recall_task_scope": list(scope),
        "stage_task_scope": {"recall:context": list(scope), "apply:generation": list(adaptation_scope)},
        "adaptation_task_scope": list(adaptation_scope), "encoder_path": str(encoder_path),
        "encoder_identity": encoder_identity, "implementation_sha256": IMPLEMENTATION_SHA256}
    if public_stimulus:
        from ..benchmarks import pm3_request
        config.update(request_profile=pm3_request.PROFILE, request_sha256=pm3_request.IMPLEMENTATION_SHA256)
    if feature_profile:
        from . import evidence_features
        feature_names(feature_profile)
        config.update(feature_profile=feature_profile, feature_sha256=evidence_features.IMPLEMENTATION_SHA256)
    return validate_execution_policy(config)


def scope_contains(task, scope):
    return task in scope or ("pm3_native::*" in scope and isinstance(task, str) and task.startswith("pm3_native::"))


def validate_execution_policy(config):
    keys = {"profile", "recall_task_scope", "stage_task_scope", "adaptation_task_scope", "encoder_path", "encoder_identity", "implementation_sha256"}
    optional = {"legacy_recall_task_scope", "request_profile", "request_sha256", "feature_profile", "feature_sha256"}
    if not keys <= set(config) or set(config) - keys - optional or config["profile"] != PROFILE:
        raise ValueError("invalid pipeline execution policy")
    if "request_profile" in config or "request_sha256" in config:
        from ..benchmarks import pm3_request
        if config.get("request_profile") != pm3_request.PROFILE or config.get("request_sha256") != pm3_request.IMPLEMENTATION_SHA256:
            raise ValueError("pipeline public request contract changed")
    if config["implementation_sha256"] != IMPLEMENTATION_SHA256:
        raise ValueError("pipeline implementation changed")
    if "feature_profile" in config or "feature_sha256" in config:
        from . import evidence_features
        if config.get("feature_profile") != evidence_features.PROFILE or config.get("feature_sha256") != evidence_features.IMPLEMENTATION_SHA256 or not config.get("request_profile"):
            raise ValueError("pipeline evidence feature contract changed")
    scope, adapt = config["recall_task_scope"], config["adaptation_task_scope"]
    for values in (scope, adapt):
        if not isinstance(values, list) or not values or any(not isinstance(s, str) or not s.startswith("pm3_native::") for s in values) or len(values) != len(set(values)):
            raise ValueError("invalid pipeline task scope")
    if "legacy_recall_task_scope" in config:
        legacy = config["legacy_recall_task_scope"]
        if not isinstance(legacy, list) or any(not isinstance(s, str) or not s.startswith("pm3_native::") or "*" in s for s in legacy):
            raise ValueError("invalid legacy pipeline scope")
    if not all(scope_contains(t, scope) for t in adapt) or config["stage_task_scope"] != {"recall:context": scope, "apply:generation": adapt}:
        raise ValueError("invalid pipeline stage scope")
    if not isinstance(config["encoder_path"], str) or not config["encoder_path"].startswith("/") or not isinstance(config["encoder_identity"], dict):
        raise ValueError("invalid pipeline encoder configuration")
    return config
