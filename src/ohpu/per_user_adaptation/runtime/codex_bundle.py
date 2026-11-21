from __future__ import annotations

import json
import hashlib
import logging
import os
import stat
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.models import HarnessConfig, make_minimal_config
from ..daily_memory import INITIAL_MEMORY_MD, MemoryState
from ..program import (
    DEFAULT_PROGRAM_SOURCE,
    PROGRAM_FILENAME,
    HarnessProgram,
    ProgramExecutionError,
    ProgramValidationError,
)

logger = logging.getLogger(__name__)

_INTERNAL_DIRNAME = ".memoharness"
_MEMORY_FILENAME = "memory.md"
_MEMORY_STATE_FILENAME = "memory.state.json"
_MEMORY_LOCK_FILENAME = ".memory.lock"
_ADMIN_NAMES = frozenset({
    "controller_artifacts", "controller_jobs",
    ".controller_artifacts", ".controller_jobs",
})
#: Files a bundle carried before the harness became one program file. Their
#: presence identifies a bundle on the old contract, which cannot execute under
#: this one, so the gate names them instead of reporting an unmanaged file.
_SUPERSEDED_TOP_FILES = frozenset({"AGENTS.override.md", "policy.json"})
_SUPERSEDED_INTERNAL_FILES = frozenset({"playbook.md", "user_model.json"})


@dataclass(frozen=True)
class CodexBundlePaths:
    """One program file, plus the engine's own state.

    ``harness.py`` is the whole harness and the only thing a proposer may edit.
    The base plan it patches over is an engine constant (``make_minimal_config``)
    and is not a file. ``memory.md`` is the compact user memory the engine
    compiles from causal deltas; ``memory.state.json`` is its cursor and never
    reaches a model.
    """

    root: Path
    program_path: Path
    memory_path: Path
    memory_state_path: Path

    @property
    def execution_policy_path(self) -> Path:
        return self.root / "execution_policy.json"

    def tracked_files(self) -> tuple[Path, ...]:
        return (self.program_path, self.memory_path, self.memory_state_path, *(
            (self.execution_policy_path,) if self.execution_policy_path.exists() else ()))

    def runtime_files(self) -> tuple[Path, ...]:
        """Files exposed to task execution; the causal cursor stays host-only."""

        return (self.program_path, self.memory_path, *(
            (self.execution_policy_path,) if self.execution_policy_path.exists() else ()))


def resolve_codex_bundle_paths(root: Path | str) -> CodexBundlePaths:
    bundle_root = Path(root).expanduser().resolve()
    internal_dir = bundle_root / _INTERNAL_DIRNAME
    return CodexBundlePaths(
        root=bundle_root,
        program_path=bundle_root / PROGRAM_FILENAME,
        memory_path=internal_dir / _MEMORY_FILENAME,
        memory_state_path=internal_dir / _MEMORY_STATE_FILENAME,
    )


def _default_memory_content() -> str:
    return INITIAL_MEMORY_MD


def ensure_codex_bundle(root: Path | str) -> CodexBundlePaths:
    """Materialise W0: the no-op program and the empty compact memory.

    Existing files are preserved so a resumed run keeps its state.
    """

    bundle_paths = resolve_codex_bundle_paths(root)
    bundle_paths.root.mkdir(parents=True, exist_ok=True)
    bundle_paths.memory_path.parent.mkdir(parents=True, exist_ok=True)
    if not bundle_paths.memory_path.exists():
        bundle_paths.memory_path.write_text(_default_memory_content(), encoding="utf-8")
    if not bundle_paths.program_path.exists():
        bundle_paths.program_path.write_text(
            DEFAULT_PROGRAM_SOURCE,
            encoding="utf-8",
        )
    return bundle_paths


def base_plan() -> HarnessConfig:
    """The frozen base every stage decision patches over.

    It is an engine constant rather than a bundle file on purpose: a file the
    proposer could edit would be a second, config-shaped search space next to
    the program, and a per-user harness is code. Keeping the base fixed also
    keeps "effective versus base" equal to "effective versus W0/B0".
    """

    return make_minimal_config()


def render_codex_bundle_preview(root: Path | str) -> str:
    bundle_paths = resolve_codex_bundle_paths(root)
    sections = []
    for label, path in (
        ("harness.py", bundle_paths.program_path),
        (".memoharness/memory.md", bundle_paths.memory_path),
    ):
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            sections.append(f"## {label}\n{content}")
    return "\n\n".join(sections)


def load_codex_bundle(root: Path | str) -> tuple[str, HarnessConfig]:
    """The bundle preview and the base plan it executes over."""

    return render_codex_bundle_preview(root), base_plan()


def snapshot_codex_bundle(root: Path | str) -> dict[Path, str | None]:
    bundle_paths = resolve_codex_bundle_paths(root)
    snapshot: dict[Path, str | None] = {}
    # Track the absence too: a proposer cannot create a host policy to enlarge
    # its scope. Older bundles retain their original runtime fingerprint.
    for path in (*bundle_paths.tracked_files(), bundle_paths.execution_policy_path):
        snapshot[path] = path.read_text(encoding="utf-8") if path.exists() else None
    return snapshot


def restore_codex_bundle(snapshot: dict[Path, str | None]) -> None:
    # A direct-edit controller can create helper artifacts in addition to
    # changing a tracked file.  A rejected candidate must not leave those
    # helpers live, so restore the managed bundle as an allowlisted set rather
    # than merely overwriting the paths present in the snapshot.
    roots = {path.parent for path in snapshot if path.name == PROGRAM_FILENAME}
    for root in roots:
        _remove_unmanaged_bundle_artifacts(root)
    for path, content in snapshot.items():
        # Do not ever follow a controller-created symlink while restoring a
        # rejected candidate.  The internal directory itself can also have
        # been replaced by a symlink, so repair that parent first.
        if path.parent.name == _INTERNAL_DIRNAME and (
            path.parent.is_symlink()
            or (path.parent.exists() and not path.parent.is_dir())
        ):
            _remove_path(path.parent)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            _remove_path(path)
        if content is None:
            try:
                _remove_path(path)
            except OSError:
                logger.debug("Could not remove restored Codex bundle file %s.", path, exc_info=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _remove_unmanaged_bundle_artifacts(root: Path) -> None:
    """Delete proposer-created files outside the fixed bundle manifest."""

    if not root.exists():
        return
    allowed_top = {PROGRAM_FILENAME, "execution_policy.json", _INTERNAL_DIRNAME, *_ADMIN_NAMES}
    for child in root.iterdir():
        if child.name not in allowed_top:
            _remove_path(child)
        elif child.is_symlink():
            _remove_path(child)
        elif child.name == PROGRAM_FILENAME and not child.is_file():
            _remove_path(child)
        elif child.name == _INTERNAL_DIRNAME and not child.is_dir():
            _remove_path(child)
    internal = root / _INTERNAL_DIRNAME
    if internal.exists() and internal.is_dir() and not internal.is_symlink():
        allowed_internal = {
            _MEMORY_FILENAME,
            _MEMORY_STATE_FILENAME,
            _MEMORY_LOCK_FILENAME,
        }
        for child in internal.iterdir():
            if child.name not in allowed_internal:
                _remove_path(child)


def _preserve_unmanaged_artifact(root: Path, child: Path) -> None:
    """Keep bounded forensic evidence before the normal incumbent restoration.

    Never follow an unknown symlink or let an administrative symlink redirect
    this write. Failure to collect evidence does not weaken the publish gate.
    """
    try:
        evidence = root / "controller_artifacts" / "rejected_bundle_artifacts"
        if not evidence.resolve().is_relative_to(root.resolve()):
            return
        if any(p.is_symlink() for p in (evidence, evidence.parent)):
            return
        fd = os.open(child, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            meta = os.fstat(handle.fileno())
            if not stat.S_ISREG(meta.st_mode):
                return
            contents = handle.read(1024 * 1024)
        key = hashlib.sha256(child.name.encode() + contents + str(meta.st_mtime_ns).encode()).hexdigest()
        evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
        evidence.chmod(0o700)
        snapshot = evidence / key
        snapshot.mkdir(mode=0o700, exist_ok=True)
        (snapshot / "artifact.bin").write_bytes(contents)
        (snapshot / "artifact.bin").chmod(0o600)
        (snapshot / "manifest.json").write_text(json.dumps({"name": child.name, "size": meta.st_size,
            "mtime_ns": meta.st_mtime_ns, "inode": meta.st_ino, "captured_bytes": len(contents),
            "truncated": len(contents) < meta.st_size, "captured_sha256": hashlib.sha256(contents).hexdigest(),
            "creation_cause": "undetermined"}))
        (snapshot / "manifest.json").chmod(0o600)
    except OSError:
        pass


def validate_codex_bundle(
    root: Path | str,
    *,
    require_program: bool = True,
    smoke: bool = False,
) -> HarnessProgram | None:
    """Strict server-side publish gate for a bundle edited by the controller.

    Unlike the forgiving execution loader, this never repairs or falls back.
    It rejects unknown artifacts, symlinks, a bundle on the superseded
    multi-file contract, and invalid program source, so controller retry and
    revert can actually take effect. The program's ``PROGRAM_API_VERSION`` is
    the schema marker: a bundle is a code bundle when ``harness.py`` is there
    and validates.
    """

    paths = resolve_codex_bundle_paths(root)
    if not paths.root.exists():
        raise ProgramValidationError(f"bundle root does not exist: {paths.root}")
    allowed_top = {PROGRAM_FILENAME, "execution_policy.json", _INTERNAL_DIRNAME, *_ADMIN_NAMES}
    for child in paths.root.iterdir():
        if child.name in _SUPERSEDED_TOP_FILES:
            raise ProgramValidationError(
                f"{child.name} belongs to the superseded multi-file bundle; the "
                "harness is now harness.py alone and this bundle predates it"
            )
        if child.name not in allowed_top:
            _preserve_unmanaged_artifact(paths.root, child)
            raise ProgramValidationError(
                f"unmanaged bundle artifact is not allowed: {child.name}"
            )
        if child.is_symlink():
            raise ProgramValidationError(f"bundle symlink is not allowed: {child.name}")
    read_execution_policy(paths.root)
    internal = paths.root / _INTERNAL_DIRNAME
    if internal.exists():
        if not internal.is_dir():
            raise ProgramValidationError(
                f"{_INTERNAL_DIRNAME} must be a directory"
            )
        allowed_internal = {
            _MEMORY_FILENAME,
            _MEMORY_STATE_FILENAME,
            _MEMORY_LOCK_FILENAME,
        }
        for child in internal.iterdir():
            if child.name in _SUPERSEDED_INTERNAL_FILES:
                raise ProgramValidationError(
                    f"{_INTERNAL_DIRNAME}/{child.name} belongs to the superseded "
                    "multi-file bundle; the harness is now harness.py alone and "
                    "this bundle predates it"
                )
            if child.name not in allowed_internal:
                raise ProgramValidationError(
                    "unmanaged internal bundle artifact is not allowed: "
                    f"{_INTERNAL_DIRNAME}/{child.name}"
                )
            if child.is_symlink():
                raise ProgramValidationError(
                    f"bundle symlink is not allowed: {_INTERNAL_DIRNAME}/{child.name}"
                )
            if child.name == _MEMORY_LOCK_FILENAME and (
                not child.is_file() or child.stat().st_size != 0
            ):
                raise ProgramValidationError(
                    "engine memory lock must be an empty regular file"
                )
    if not paths.memory_path.is_file() or paths.memory_path.is_symlink():
        raise ProgramValidationError(
            f"required bundle file is missing: {paths.memory_path.name}"
        )
    if paths.memory_state_path.exists():
        if not paths.memory_state_path.is_file() or paths.memory_state_path.is_symlink():
            raise ProgramValidationError("memory.state.json must be a regular file")
        try:
            state_payload = json.loads(
                paths.memory_state_path.read_text(encoding="utf-8")
            )
            if not isinstance(state_payload, dict):
                raise ValueError("state must be an object")
            MemoryState.from_dict(state_payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ProgramValidationError(
                f"memory.state.json is invalid: {type(exc).__name__}"
            ) from exc
    program = None
    if paths.program_path.exists():
        program = HarnessProgram.from_path(paths.program_path)
    elif require_program:
        raise ProgramValidationError(f"required program is missing: {PROGRAM_FILENAME}")
    policy = read_execution_policy_details(paths.root)
    if policy and policy["profile"] == "adaptive-pipeline-v1" and program is not None:
        from ..adaptive_pipeline import validate_source as validate_pipeline_source
        try:
            model = validate_pipeline_source(program.source)
            if model.get("feature_profile") != policy.get("feature_profile"):
                raise ValueError("pipeline model and execution feature profiles differ")
        except (ValueError, SyntaxError) as exc:
            raise ProgramValidationError(str(exc)) from exc
    if policy and policy["profile"] == "shared-retrieval" and program is not None:
        from ..retrieval_policy import validate_source
        try:
            validate_source(program.source)
        except (ValueError, SyntaxError) as exc:
            raise ProgramValidationError(str(exc)) from exc
    if smoke and program is not None:
        common = {
            "task_kind": "smoke",
            "query": "",
            "turn": 0,
            "blocks": [],
            "tools": [],
            "history_units": [],
            "output": "",
            "base": {},
        }
        if policy and policy["profile"] == "adaptive-pipeline-v1":
            from ..adaptive_pipeline import ACTIONS, feature_names, GROUNDED_GUIDANCE
            common.update(pipeline_features=[1.] + [0.] * (len(feature_names(policy.get("feature_profile"))) - 1),
                pipeline_candidates=[{"recall": {"memory_mode": "no_user_memory"},
                    "apply": {"prefix": GROUNDED_GUIDANCE} if a.endswith("grounded") else {}} for a in ACTIONS])
        # Every hook:stage the executor can reach is smoked here. Without a
        # call for one of them, a hook that returns keys for that stage
        # unconditionally would only fail during a paid window.
        calls = [
            ("remember", {
                **common,
                "logical_days": [],
                "events": [],
                "channels": [],
                "current_memory_sections": [],
            }),
            ("recall", {
                **common,
                "stage": "evidence",
                "available_items": 0,
                "memory_facts": {},
                "base": {"mode": "none", "top_k": 0,
                         "budget_chars": 2400, "available_items": 0},
            }),
            ("recall", {
                **common,
                "stage": "context",
                "history_units": [],
                "history_block": "",
                "history_chars": 0,
                "memory_facts": {},
            }),
            ("apply", {
                **common,
                "stage": "generation",
                "tool_count": 0,
                "observation_count": 0,
                "memory_facts": {},
            }),
            ("apply", {
                **common,
                "stage": "arguments",
                "tool_name": "",
                "arguments": {},
                "allowed_argument_names": [],
                "memory_facts": {},
            }),
            ("guard", {
                **common, "phase": "validate", "output": "",
                "memory_facts": {},
            }),
            ("guard", {
                **common, "phase": "postprocess", "output": "ok",
                "memory_facts": {},
            }),
        ]
        for hook, context in calls:
            program.call(hook, context)
        # The empty smoke above cannot reach a branch guarded by a nonempty
        # pool/memory. Exercise the actual executor input contract as well;
        # no model, retrieval, memory rewrite or benchmark query is involved.
        from ..executor import ExecutableHarness, Runner, Trace, memory_facts

        memory = paths.memory_path.read_text(encoding="utf-8")
        if not memory_facts(memory):
            memory = "## Preferences\n- Prefers concise, evidence-grounded answers.\n"
        trace = Trace(episode_id="smoke", user_id="smoke", condition="publish")
        Runner(None, ExecutableHarness(
            user_id="smoke", config=make_minimal_config(),
            memory_md=memory, program=program,
        )).retrieval_plan(
            task_kind="smoke", query="Retrieve evidence for this user's preferences.",
            mode="none", top_k=0, budget_chars=2400, available_items=3,
            trace=trace,
        )
        # Execution may fall back, but publication is fail-closed. A legal
        # no-op is still allowed: this is contract validation, not a policy
        # requirement to retrieve on every request.
        if trace.program_report.get("fallbacks"):
            raise ProgramExecutionError(
                "recall:evidence failed the nonempty causal-pool smoke check"
            )
    return program


def bundle_file_payload(root: Path | str) -> list[tuple[str, str]]:
    # Sync is a publish boundary, not a repair operation. A missing code file
    # must fail closed instead of being silently replaced with the no-op
    # default just before upload.
    validate_codex_bundle(root, require_program=True, smoke=True)
    bundle_paths = resolve_codex_bundle_paths(root)
    payload: list[tuple[str, str]] = []
    for path in bundle_paths.runtime_files():
        if path.is_symlink():
            raise ProgramValidationError(
                f"bundle artifact must not be a symlink: {path.name}"
            )
        rel_path = path.relative_to(bundle_paths.root).as_posix()
        payload.append((rel_path, path.read_text(encoding="utf-8")))
    return payload


def codex_bundle_fingerprint(
    root: Path | str,
    *,
    validate: bool = True,
    smoke: bool = False,
) -> dict[str, Any]:
    """Bind the program and the engine state it executes with.

    Code and compact memory reach execution; the engine-owned memory cursor
    determines which compact state is legal. A checkpoint is therefore
    identified by the program together with any persisted online state.
    """

    if validate:
        validate_codex_bundle(root, require_program=True, smoke=smoke)
    paths = resolve_codex_bundle_paths(root)
    files: dict[str, dict[str, Any]] = {}
    for path in paths.tracked_files():
        if path in (paths.memory_state_path, paths.execution_policy_path) and not path.exists():
            # W0 may be fingerprinted before the first user event initializes
            # its identity-bound cursor. Every online evaluated checkpoint has
            # this file and therefore binds it into the fingerprint.
            continue
        if not path.is_file() or path.is_symlink():
            raise ProgramValidationError(
                f"cannot fingerprint missing/non-regular bundle file: {path.name}"
            )
        data = path.read_bytes()
        relative = path.relative_to(paths.root).as_posix()
        files[relative] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    canonical = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def read_execution_policy(root: Path | str) -> tuple[str, ...] | None:
    policy = read_execution_policy_details(root)
    return tuple(policy["recall_task_scope"]) if policy else None


def read_execution_policy_details(root: Path | str) -> dict | None:
    path = resolve_codex_bundle_paths(root).execution_policy_path
    if not path.exists():
        return None
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("policy must be a regular file")
        policy = json.loads(path.read_text())
        if not isinstance(policy, dict):
            raise ValueError("invalid policy object")
        if policy.get("profile") == "adaptive-pipeline-v1":
            from ..adaptive_pipeline import validate_execution_policy
            return validate_execution_policy(policy)
        shared = policy.get("profile") == "shared-retrieval"
        keys = {"profile", "recall_task_scope"} | ({"adaptation_source_count"} if shared else set())
        if shared and "adaptation_task_scope" in policy:
            keys.add("adaptation_task_scope")
        if set(policy) != keys:
            raise ValueError("invalid policy fields")
        values = policy["recall_task_scope"]
        if policy["profile"] not in {"scoped-recall", "shared-retrieval"} or not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value.startswith("pm3_native::") for value in values
        ):
            raise ValueError("invalid recall task scope")
        if len(values) != len(set(values)):
            raise ValueError("duplicate task scope")
        if shared:
            count = policy["adaptation_source_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("invalid adaptation source count")
            if "adaptation_task_scope" in policy:
                subset = policy["adaptation_task_scope"]
                if (not isinstance(subset, list) or not subset
                        or any(not isinstance(x, str) for x in subset)
                        or len(set(subset)) != len(subset) or not set(subset) <= set(values)):
                    raise ValueError("adaptation scope must be a nonempty subset of shared retrieval scope")
        return policy
    except (ValueError, OSError, TypeError) as exc:
        raise ProgramValidationError("invalid host execution policy") from exc


def _write_file_command(relative_path: str, content: str) -> str:
    token = "__MEMOHARNESS_EOF__"
    mkdir = f"mkdir -p {shlex.quote(str(Path(relative_path).parent.as_posix() or '.'))}"
    return (
        f"{mkdir}\n"
        f"cat <<'{token}' > {shlex.quote(relative_path)}\n"
        f"{content}\n"
        f"{token}\n"
    )


async def sync_codex_bundle_to_environment(environment, root: Path | str) -> None:
    for relative_path, content in bundle_file_payload(root):
        result = await environment.exec(_write_file_command(relative_path, content))
        rendered = result if isinstance(result, str) else str(result)
        if "[bash error]" in rendered.lower():
            raise RuntimeError(
                f"Failed to sync Codex bundle file {relative_path}: {rendered}"
            )
