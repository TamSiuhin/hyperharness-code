"""History-only PM3 training, with private official-rubric scoring.

The active objective covers response/restraint pairs, social recommendation,
native chatbot restraint, and real forget before/after pairs. All use native
task/output contracts and the bounded-memory runtime. Mining never reads native
question answers or profile labels. Extraction/classification builders remain
explicit diagnostics for archived contracts, not training.

The decision cursor bounds context; the label-availability cursor bounds when
an observed outcome may enter optimization. Current native-shaped cases also
exclude same-timestamp history, matching downstream native evaluation.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ohpu.benchmarks.native_checkout import module_within, verify_pinned_checkout
from ohpu.paths import DATA_ROOT

PUBLIC_META_KEY = "pm3_proxy"


# Three families survive the 2026-08-31 shortcut audit. The three that were
# dropped were solved by rules that never read a single user-specific label:
#   slate_ranking            ranking by stimulus LENGTH scored nDCG .973 and
#                            top-1 .904 against a .250 chance, because a random
#                            permutation of a 4-item pool already scores .876.
#   restraint_null           the gold is the constant "NONE" for 20/20 users,
#                            so a constant string is a perfect answer.
#   persona_fact_verification  "not in visible history -> UNKNOWN, else has a
#                            content field -> SUPPORTED, else CONTRADICTED"
#                            scored .985 over 20 users (788/800).
# All three inherit one corpus fact: 27,782 of 28,085 implicit_negative
# reactions carry no content field at all, while 2,502/2,503 explicit_positive
# ones do -- so "is there a content object" is a valence oracle.
# Families the official evaluator scores with a judge rather than a formula.
# `evaluation/llm_metrics.py` is upstream's "single source of truth for
# preference-leak detection", and its docstring demotes the similarity-based
# metrics.py versions to a fallback that "should only fire with
# --no-enable_llm_judge". Measured here: an answer that says "I know you have
# been working on your bass guitar lessons" scores cosine 0.41 against that
# very preference, under the 0.5 threshold, so the deterministic path reports
# zero leakage. There is no judge-free way to score this axis.
# A turn shorter than this is a greeting or an acknowledgement: there is no
# answer to write, so there is nothing a leak could hide in.

# reaction_prediction is scored two-way. The four-way label set stays as the
# SOURCE vocabulary (_reaction_label reads it off the corpus), but the task the
# harness is asked to solve is positive-vs-negative: the explicit/implicit
# distinction is a property of how the platform logged the interaction, not of
# the person, and keeping it four-way let the majority class alone score .684.
# The seven keys whose presence in a public stimulus betrays valence. Any
# family that pairs a positive with a negative referent MUST match on this
# signature, or "does the JSON have text in it" answers the question.


_OFFICIAL_EVALUATOR_ROOT = DATA_ROOT / "raw" / "PersonaMem-v3-evaluator"


def configure_official_scoring(root: str | Path, expected_revision: str) -> None:
    """Bind training to the same verified checkout selected for native eval."""
    global _OFFICIAL_EVALUATOR_ROOT
    path = Path(root).resolve()
    verify_pinned_checkout(path, expected_revision=expected_revision, label="PM3 training")
    existing = sys.modules.get("evaluation")
    if existing is not None and not module_within(existing, path):
        raise RuntimeError("a different evaluation package is loaded; start PM3 in a fresh process")
    _OFFICIAL_EVALUATOR_ROOT = path


def official_module(name: str):
    """Import one module from the pinned PM3 checkout, and only from there.

    A same-named module loaded from anywhere else is refused rather than
    silently scoring training with code the run never pinned.
    """

    root = str(_OFFICIAL_EVALUATOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # import_module waits on Python's per-module initialization lock. Reading
    # sys.modules directly can expose a half-built rubric to another scorer.
    module = importlib.import_module(name)
    if not module_within(module, _OFFICIAL_EVALUATOR_ROOT):
        raise RuntimeError(f"refusing a non-official `{name}` for scoring")
    return module
