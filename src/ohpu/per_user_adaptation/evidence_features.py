"""Public request/evidence signals; no task labels, verifier, or future state.

Polarity and temporal markers are lexical cues, not verified preferences or
inferred timestamps. The model may learn their usefulness from TRAIN rewards.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

PROFILE = "public-evidence-v1"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INTENTS = ("ranking", "optout", "utility", "conversation")
SIGNALS = ("shared_semantic_best", "hybrid_semantic_best", "semantic_difference",
           "shared_negative_fraction", "hybrid_negative_fraction",
           "shared_temporal_fraction", "hybrid_temporal_fraction",
           "shared_empty", "hybrid_empty", "mixed_polarity_cue")
NEGATIVE = re.compile(r"\b(?:dislikes?|avoids?|hates?|allerg\w*|never|cannot|can't|no longer|not interested)\b|不喜欢|避免|过敏|不再", re.I)
POSITIVE = re.compile(r"\b(?:likes?|enjoys?|loves?|prefers?)\b|喜欢|偏好", re.I)
TEMPORAL = re.compile(r"\b(?:currently|previously|formerly|recently|no longer|used to|until|since|expired)\b|\b20\d{2}[-/]\d{1,2}\b|目前|过去|不再|过期", re.I)
OPTOUT = re.compile(r"\b(?:do not|don't|don’t|without|avoid)\b.{0,60}\b(?:personali[sz]\w*|memory|history|preferences?)\b|不要.{0,30}(?:个性化|记忆|历史|偏好)", re.I | re.S)


def current_request(query):
    # The shared public-stimulus parser preserves these headings. Do not let
    # earlier user/assistant turns determine an explicit opt-out in this turn.
    match = re.search(r"(?m)^(?:Current user query|User query|User question[^\n]*)\n", query)
    if match:
        rest = query[match.end():]
        return re.split(r"\n\n(?:Prior |Original |Candidate|User context)", rest, maxsplit=1)[0]
    return query


def intent(query):
    request = current_request(query)
    if OPTOUT.search(request):
        return "optout"
    if re.search(r"(?m)^Candidate[^\n]*\n", query):
        return "ranking"
    if re.search(r"\b(?:translate|translation|proofread|rewrite|summarize|summarise|grammar|debug|calculate)\b|翻译|改写|总结|语法|计算", request, re.I):
        return "utility"
    return "conversation"


def feature_names(base):
    return tuple(base) + tuple("intent_" + name for name in INTENTS) + SIGNALS


def augment(query, records, encoder, prepared):
    selected_intent = intent(query)
    scores = {}
    if records:
        vectors = encoder.encode([current_request(query), *(r["text"] for r in records)])
        if len(vectors) != len(records) + 1 or not vectors[0] or any(
            len(v) != len(vectors[0]) or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError("invalid evidence feature embeddings")
        for record, vector in zip(records, vectors[1:]):
            scale = math.sqrt(sum(x*x for x in vectors[0]) * sum(x*x for x in vector))
            scores[record["id"]] = max(0., min(1., sum(a*b for a,b in zip(vectors[0], vector)) / scale)) if scale else 0.
    by_id = {r["id"]: r for r in records}
    def signals(candidate):
        rs = [by_id[i] for i in candidate["audit"]["selected_record_ids"]]
        return (max((scores[r["id"]] for r in rs), default=0.),
                sum(bool(NEGATIVE.search(r["text"])) for r in rs) / max(1, len(rs)),
                sum(bool(TEMPORAL.search(r["text"])) for r in rs) / max(1, len(rs)), float(not rs))
    left, right = signals(prepared["candidates"][0]), signals(prepared["candidates"][2])
    mixed = float(any(NEGATIVE.search(r["text"]) for r in records) and any(POSITIVE.search(r["text"]) for r in records))
    extra = [float(selected_intent == t) for t in INTENTS] + [left[0], right[0], right[0]-left[0],
        left[1], right[1], left[2], right[2], left[3], right[3], mixed]
    return {**prepared, "features": prepared["features"] + extra,
            "feature_audit": {"profile": PROFILE, "intent": selected_intent,
                              "temporal_values": "lexical_cues_only", "private_labels_used": False}}
