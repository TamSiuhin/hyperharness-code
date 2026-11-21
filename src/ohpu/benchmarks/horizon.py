"""Reproducible active-user cohorts and causal HORIZON query inventories."""
from __future__ import annotations

import random
import statistics
from typing import Any


AMAZON_REPOSITORY = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
HORIZON_REVISION = "889e91440a4a7d49c5dfe4b6513a2953a8a5c19c"
CUTOFF_MS = 1577836800000


def role_split(user_ids: list[str], *, seed: int = 20261022) -> dict[str, Any]:
    if len(user_ids) != len(set(user_ids)) or not user_ids or len(user_ids) % 2:
        raise ValueError("User role split requires a unique even-sized cohort")
    ordered = sorted(user_ids)
    train = set(random.Random(seed).sample(ordered, len(ordered) // 2))
    return {"seed": seed, "unit": "user", "train": sorted(train), "test": sorted(set(ordered) - train)}


def distribution(values) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"count": 0, "sum": 0}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return round(values[lower] + (values[upper] - values[lower]) * (position - lower), 3)

    return {"count": len(values), "sum": sum(values), "min": values[0],
        "p25": percentile(0.25), "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 3), "p75": percentile(0.75),
        "p90": percentile(0.9), "p95": percentile(0.95), "max": values[-1]}
