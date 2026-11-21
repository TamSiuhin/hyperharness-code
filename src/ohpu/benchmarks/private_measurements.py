"""Durable evaluator-private measurement slots, never a shared response cache.

A slot denotes ONE planned measurement (including generation/judge repeat).
Reopening it resumes that measurement; a fresh replicate needs a fresh slot.
Identity mismatches fail closed instead of silently reusing another answer.
"""
from __future__ import annotations

import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()
