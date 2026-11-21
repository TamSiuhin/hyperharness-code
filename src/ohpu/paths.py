"""Stable paths for source-checkout experiments.

The package can be imported from any working directory.  In an editable/source
checkout, the nearest parent containing ``pyproject.toml`` is the project root;
an installed wheel falls back to the caller's working directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    configured = os.environ.get("OHPU_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        *Path(__file__).resolve().parents,
        Path(sys.prefix).resolve().parent,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
