"""Fixed memory actions for paired history diagnostics, with no private labels.

The oracle action is a separate evaluator-only operation. It is never a
deployable action or a source of router features. Semantic encoding is local,
explicitly pinned by file hashes, and fails rather than downloading a model.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from ohpu.benchmarks.private_measurements import digest
from . import retrieval_policy as policy
from .executor import complete_memory_records, ordered_memory_records

ACTIONS = ("no_memory", "legacy_shared", "corrected_shared", "hybrid", "support_oracle")
DEPLOYABLE_ACTIONS = ("no_memory", "corrected_shared", "hybrid")


class LocalSemanticEncoder:
    """A frozen local SentenceTransformer; no Hub/network fallback."""

    def __init__(self, model_path):
        root = Path(model_path).expanduser().resolve(strict=True)
        if not root.is_dir() or not (root / "modules.json").is_file():
            raise ValueError("semantic encoder requires a local SentenceTransformer snapshot")
        # SentenceTransformer loads these formats. Hash exactly the local
        # assets and reject custom executable code; do not hash Hub git data.
        assets = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(root.rglob("*")) if p.is_file()
                  and p.suffix in {".json", ".txt", ".safetensors", ".bin", ".model"}}
        if not any(p.endswith((".safetensors", ".bin")) for p in assets):
            raise ValueError("semantic model weights are missing")
        if any(root.rglob("*.py")):
            raise ValueError("custom model code is not supported")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(root), device="cpu", local_files_only=True,
                                         trust_remote_code=False)
        self.identity = {"backend": "sentence-transformers-cpu", "assets_sha256": digest(assets)}
        self.cache = {}

    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors = self.model.encode(missing, normalize_embeddings=True,
                                        show_progress_bar=False, convert_to_numpy=True)
            self.cache.update((t, [float(x) for x in v]) for t, v in zip(missing, vectors))
        return [self.cache[t] for t in texts]


def hybrid_recall(query, records, encoder):
    if encoder is None:
        raise ValueError("hybrid action requires an explicit local semantic encoder")
    if not records:
        return {"memory_mode": "no_user_memory"}, {"reason": "no_addressable_records"}
    vectors = encoder.encode([query, *(r["text"] for r in records)])
    if len(vectors) != len(records) + 1 or not vectors[0]:
        raise ValueError("semantic embedding count or dimension mismatch")
    if any(len(v) != len(vectors[0]) or not all(math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("invalid semantic embeddings")
    def cosine(a, b):
        scale = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
        return sum(x*y for x,y in zip(a,b)) / scale if scale else 0.0
    lexical = {r["id"]: score for score, r in policy.rank_records(query, records, .75)}
    # Predeclared diagnostic settings, not thresholds fitted to held-out scores.
    ranked = sorted(((.5 * lexical[r["id"]] + .5 * max(0., cosine(vectors[0], vector)), r)
                     for r, vector in zip(records, vectors[1:])), key=lambda x: (-x[0], x[1]["id"]))
    selected, chars, relevant = [], 0, 0
    for score, record in ranked:
        if score < .15 or len(selected) >= policy.DEFAULTS["top_k"]:
            break
        relevant += 1
        cost = len(record["text"]) + len(record["section"]) + 8
        if chars + cost <= policy.DEFAULTS["budget_chars"]:
            selected.append(record["id"])
            chars += cost
    audit = {"semantic_features": True, "temporal_features": False,
             "encoder": encoder.identity, "semantic_weight": .5, "minimum_hybrid_score": .15,
             "top_k": policy.DEFAULTS["top_k"], "budget_chars": policy.DEFAULTS["budget_chars"],
             "reason": "ranked_existing_evidence" if selected else
                       "budget_cannot_fit_record" if relevant else "no_relevant_records"}
    return ({"memory_mode": "select", "memory_fact_ids": selected} if selected else
            {"memory_mode": "no_user_memory"}), audit


def action_memory(action, query, memory, *, encoder=None, oracle_ids=None):
    """Return selected existing memory and an audit; unresolved oracle is None."""
    if action not in ACTIONS:
        raise ValueError("unknown memory action")
    records = complete_memory_records(memory)
    if action == "support_oracle":
        if not oracle_ids:
            return None, {"action": action, "available": False,
                          "reason": "support_unresolved", "deployable": False}
        if len(set(oracle_ids)) != len(oracle_ids) or not set(oracle_ids) <= {r["id"] for r in records}:
            raise ValueError("oracle must select unique existing records")
        patch, audit = {"memory_mode": "select", "memory_fact_ids": list(oracle_ids)}, {
            "reason": "exact_supported_existing_records", "diagnostic_only": True}
    elif action == "no_memory":
        patch, audit = {"memory_mode": "no_user_memory"}, {"reason": "explicit_no_memory"}
    elif action == "hybrid":
        patch, audit = hybrid_recall(query, records, encoder)
    else:
        patch, audit = policy.shared_recall(query, records, policy.ZERO_DELTA, 0,
            empty_policy="legacy-base" if action == "legacy_shared" else "abstain")
    mode = patch["memory_mode"]
    selected = (memory if mode == "base" else "" if mode == "no_user_memory" else
                ordered_memory_records(memory, patch["memory_fact_ids"])[0])
    return selected, {**audit, "action": action, "available": True,
        "deployable": action in DEPLOYABLE_ACTIONS, "memory_mode": mode,
        "available_records": len(records), "selected_records": len(records) if mode == "base" else
         0 if mode == "no_user_memory" else len(patch["memory_fact_ids"]),
        "original_memory_sha256": digest(memory), "selected_memory_sha256": digest(selected),
        "selected_record_ids": patch.get("memory_fact_ids", [r["id"] for r in records] if mode == "base" else [])}
