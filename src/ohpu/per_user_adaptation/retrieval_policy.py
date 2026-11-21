"""Shared, local retrieval with a bounded, fact-free per-user parameter delta.

Word overlap and character trigrams are lexical features, NOT embeddings.
Missing semantic or temporal metadata is not replaced by invented features.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import math
from pathlib import Path
import re

PROFILE = "shared-retrieval"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFAULTS = {"lexical_weight": .75, "min_relevance": .04, "top_k": 12,
            "budget_chars": 12000, "abstain_below": 0.0}
BOUNDS = {"lexical_weight": (-.2, .2), "min_relevance": (-.03, .03),
          "top_k": (-6, 6), "budget_chars": (-4000, 4000), "abstain_below": (0, .08)}
ZERO_DELTA = {key: 0 for key in DEFAULTS}
SHRINKAGE_SOURCES = 20
SOURCE = '''PROGRAM_API_VERSION = 1
CHANGE_MANIFEST = {"recall": "Adjust bounded retrieval parameters using training evidence."}
USER_MODEL = {"lexical_weight": 0, "min_relevance": 0, "top_k": 0, "budget_chars": 0, "abstain_below": 0}

def remember(context):
    return {}

def recall(context):
    if context.get("stage") == "context":
        return {"retrieval_delta": USER_MODEL}
    return {}

def apply(context):
    return {}

def guard(context):
    return {}
'''


def validate_delta(delta):
    if not isinstance(delta, dict) or set(delta) != set(BOUNDS):
        raise ValueError("retrieval delta must have exactly the five supported numeric parameters")
    for key, value in delta.items():
        lo, hi = BOUNDS[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lo <= value <= hi):
            raise ValueError(f"retrieval delta outside bounds: {key}")
    return delta


def validate_source(source):
    """Only a numeric USER_MODEL and a short change note may vary."""
    tree, template = ast.parse(source), ast.parse(SOURCE)
    delta = None
    for parsed in (tree, template):
        for node in parsed.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == "USER_MODEL":
                    values = validate_delta(ast.literal_eval(node.value))
                    if parsed is tree:
                        delta = values
                    node.value = ast.Constant(value=None)
                elif name == "CHANGE_MANIFEST":
                    note = ast.literal_eval(node.value)
                    if (not isinstance(note, dict) or set(note) != {"recall"}
                            or not isinstance(note["recall"], str)
                            or len(note["recall"]) > 240):
                        raise ValueError("invalid shared retrieval change note")
                    node.value = ast.Constant(value=None)
    if delta is None or ast.dump(tree) != ast.dump(template):
        raise ValueError("shared retrieval may change only USER_MODEL numbers and CHANGE_MANIFEST text")
    return delta


def parameters(delta, source_count):
    validate_delta(delta)
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 0:
        raise ValueError("invalid independent training source count")
    shrink = source_count / (source_count + SHRINKAGE_SOURCES)
    values = {key: DEFAULTS[key] + shrink * delta[key] for key in DEFAULTS}
    values["top_k"] = round(values["top_k"])
    values["budget_chars"] = round(values["budget_chars"])
    return values, shrink


_STOP = frozenset("a an the and or to of for in on at with is are be as it this that i you your user task output response json memory using only from by every all include".split())


def _tokens(text):
    return {w for w in re.findall(r"\w+", text.casefold()) if len(w) > 1 and w not in _STOP}


def _trigrams(tokens):
    return {word[i:i + 3] for word in tokens for i in range(len(word) - 2)}


def rank_records(query, records, lexical_weight):
    words = [_tokens(r["text"]) for r in records]
    df = Counter(w for tokens in words for w in tokens)
    q = _tokens(query)
    # Instruction-only terms absent from memory must not dilute relevance.
    q &= set(df)
    grams = _trigrams(_tokens(query))
    scores = []
    for record, tokens in zip(records, words):
        idf = lambda w: math.log(1 + len(records) / (1 + df[w]))
        denominator = sum(idf(w) for w in tokens)
        lexical = sum(idf(w) for w in tokens & q) / denominator if denominator else 0.0
        record_grams = _trigrams(tokens)
        chars = len(record_grams & grams) / len(record_grams) if record_grams else 0.0
        scores.append((lexical_weight * lexical + (1 - lexical_weight) * chars, record))
    return sorted(scores, key=lambda pair: (-pair[0], pair[1]["id"]))


def shared_recall(query, records, delta, source_count, *, empty_policy="abstain"):
    """Select complete entries; failed selection never silently expands memory.

    ``legacy-base`` exists only for explicitly labelled diagnostic controls.
    Runtime callers use the corrected, monotone ``abstain`` behavior.
    """
    if empty_policy not in {"abstain", "legacy-base"}:
        raise ValueError("unknown empty retrieval policy")
    params, shrink = parameters(delta, source_count)
    ranked = rank_records(query, records, params["lexical_weight"])
    audit = {"policy": PROFILE, "implementation_sha256": IMPLEMENTATION_SHA256,
             "parameters": params, "shrinkage": shrink,
             "available_record_ids": [r["id"] for r in records],
             "semantic_features": False, "temporal_features": False,
             "empty_selection_policy": empty_policy,
             "selection_revision": "explicit-empty-v5"}
    empty = {"memory_mode": "base" if empty_policy == "legacy-base" else "no_user_memory"}
    if not ranked:
        return empty, {**audit, "reason": "no_addressable_records"}
    if ranked[0][0] < params["abstain_below"]:
        return {"memory_mode": "no_user_memory"}, {**audit, "reason": "below_abstention_threshold"}
    selected, chars, relevant = [], 0, 0
    for score, record in ranked:
        if score < params["min_relevance"] or len(selected) >= params["top_k"]:
            break
        relevant += 1
        cost = len(record["text"]) + len(record["section"]) + 8
        if chars + cost <= params["budget_chars"]:
            selected.append(record["id"])
            chars += cost
    if not selected:
        return empty, {**audit, "reason": "budget_cannot_fit_record" if relevant else "no_relevant_records"}
    return {"memory_mode": "select", "memory_fact_ids": selected}, {**audit, "reason": "ranked_existing_evidence"}
