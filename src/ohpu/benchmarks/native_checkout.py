"""Shared fail-closed checks for pinned native evaluator repositories."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def module_within(module: Any, root: Path) -> bool:
    """Return whether ``module`` was loaded from ``root``."""

    source = getattr(module, "__file__", None)
    if not source:
        return False
    try:
        Path(source).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def git_revision(root: Path, *, timeout_seconds: float = 10.0) -> str | None:
    """Read a checkout's HEAD, returning ``None`` when git is unavailable."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def git_worktree_changes(
    root: Path,
    *,
    include_untracked: bool = True,
    timeout_seconds: float = 10.0,
) -> tuple[str, ...] | None:
    """Return non-ignored changes, or ``None`` when git is unavailable."""

    untracked = "all" if include_untracked else "no"
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                f"--untracked-files={untracked}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return tuple(
        line for line in result.stdout.splitlines()
        if line.strip() and not _is_bytecode_cache(line)
    )


def _is_bytecode_cache(status_line: str) -> bool:
    """Ignore CPython's own bytecode cache when judging cleanliness.

    Importing the pinned evaluator makes CPython write ``__pycache__`` beside
    its sources unless ``PYTHONDONTWRITEBYTECODE`` is set in every process that
    touches it. Those files are derived from the very sources git already
    verified, carry no source semantics, and are not what an integrity check is
    for -- but ``--untracked-files=all`` reports them, so the first process to
    import the evaluator made every later process fail with "checkout differs
    from the pinned revision". That aborted a whole 10-user launch on
    2026-08-29 after one child had imported successfully.
    """

    path = status_line[3:].strip().strip('"')
    return path.endswith((".pyc", ".pyo")) or "__pycache__/" in path


def verify_pinned_checkout(
    root: Path,
    *,
    expected_revision: str,
    label: str,
    include_untracked: bool = True,
    fetch_hint: str | None = None,
) -> str | None:
    """Verify revision and cleanliness before loading evaluator-private code.

    An empty ``expected_revision`` intentionally disables enforcement for test
    fixtures and local evaluator development, while still reporting HEAD when
    it can be read.
    """

    revision = git_revision(root)
    if not expected_revision:
        return revision

    if revision is None:
        suffix = f"; {fetch_hint}" if fetch_hint else ""
        raise RuntimeError(
            f"{label} evaluator revision cannot be verified{suffix}"
        )
    if not revision.startswith(expected_revision):
        raise RuntimeError(
            f"{label} evaluator revision mismatch: expected "
            f"{expected_revision}, got {revision}"
        )

    changes = git_worktree_changes(
        root,
        include_untracked=include_untracked,
    )
    if changes is None:
        raise RuntimeError(
            f"{label} evaluator worktree cleanliness cannot be verified"
        )
    if changes:
        raise RuntimeError(
            f"{label} evaluator checkout differs from the pinned revision: "
            + ", ".join(changes[:8])
        )
    return revision
