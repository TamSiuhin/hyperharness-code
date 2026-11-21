"""Validated, isolated execution for code-represented personalized harnesses.

Each per-user bundle may contain a single ``harness.py`` program.  The program
implements four personalization stages as pure functions named ``remember``,
``recall``, ``apply`` and ``guard``.  A hook receives one JSON object and
returns a JSON object describing the structural decision for that stage.  The
executor, rather than the generated program, remains responsible for model
I/O, tool legality, causal history, hard budgets, and validation.

The stages answer the four questions a personalized harness has to answer --
what to keep about this user, what this request should see of them, how to
answer for them, and when not to personalize -- rather than the six MemoHarness
dimensions, which describe agent mechanics and split each of those questions
across several hooks.  ``HarnessConfig`` is a different object with the same
four names: the engine's frozen base plan that each stage decision patches
over, grouped so a knob sits with the decision it is a default for.  It is a
constant, not a bundle file.

Programs are deliberately run in a fresh isolated Python process with a minimal,
fixed environment.  They cannot import modules or access files, and communicate only
through JSON stdin/stdout.  The AST validator is defense in depth; the process
boundary is what prevents evaluator environment variables and in-process
objects from becoming an accidental side channel.
Validated literal-empty hooks may return an empty patch without executing user
code; input bounds and a conservative initialization-step bound still apply.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

PROGRAM_API_VERSION = 1
PROGRAM_FILENAME = "harness.py"
PROGRAM_HOOKS = ("remember", "recall", "apply", "guard")
#: Sub-stages each hook is called at.  ``remember`` runs once per memory
#: rewrite and has no sub-stage; the others are consulted at more than one
#: consumption point and receive ``context["stage"]``.
#: The decision fields each ``hook:stage`` may return.  A field appears at
#: exactly one stage: the point where the executor consumes it.
STAGE_DECISION_KEYS: Mapping[str, frozenset[str]] = {
    # What the engine keeps about this user.  Compiles the persistent memory
    # document that the online engine rewrites from causal event deltas.
    "remember": frozenset({
        "keep_events", "drop_channels", "priority_channels",
        "max_chars", "guidance", "state_schema", "state_records", "memory_backend",
    }),
    # What this request gets to see of the user.
    "recall:evidence": frozenset({"mode", "top_k", "budget_chars"}),
    "recall:context": frozenset({
        "memory_sections", "memory_fact_ids", "memory_mode", "retrieval_delta", "include_history", "max_history_chars",
        "keep_history_units", "memory_query",
    }),
    # How to answer for this user.  One decision per action: the executor
    # consults this once and threads the result through context assembly,
    # action space, decoding and workflow.
    "apply:generation": frozenset({
        "block_order", "drop_blocks", "budget", "prefix", "suffix",
        "structured_instruction",
        "allow", "deny", "order",
        "temperature", "top_p", "max_tokens", "reasoning_effort",
        "workflow", "max_turns", "stop_rule", "plan_prompt", "refine_prompt",
        "max_tool_calls",
    }),
    "apply:arguments": frozenset({"defaults", "overrides", "drop_arguments"}),
    "apply:step": frozenset({
        "operation", "prompt", "selected_ids", "final_output", "state",
    }),
    # When not to personalize, and how to check the answer.
    "guard:validate": frozenset({"violations", "repair_prompt"}),
    "guard:postprocess": frozenset({"output"}),
}
PROGRAM_DECISION_KEYS: Mapping[str, frozenset[str]] = {
    hook: frozenset().union(*(
        keys for key, keys in STAGE_DECISION_KEYS.items()
        if key == hook or key.startswith(f"{hook}:")
    ))
    for hook in PROGRAM_HOOKS
}


def context_stage(hook: str, context: Mapping[str, Any]) -> str:
    """Read the sub-stage THIS hook is called at, ignoring foreign fields.

    ``guard`` is keyed by ``phase`` and the others by ``stage``.  Reading
    whichever happens to be present would let an unrelated field decide the
    contract: a ``remember`` context that carried a leftover ``phase`` would be
    validated against a guard stage and rejected for a key it never returned.
    """

    name = str(hook).strip().lower()
    if name == "guard":
        return str(context.get("phase") or "").strip().lower()
    if name == "remember":
        return ""
    return str(context.get("stage") or "").strip().lower()


def stage_key(hook: str, stage: str | None) -> str:
    """Return the ``hook`` or ``hook:stage`` key this call is governed by."""

    name = str(hook).strip().lower()
    part = str(stage or "").strip().lower()
    return f"{name}:{part}" if part else name


# The memory document's shape is the rewriter's business; the program
# influences it with one bounded clause of prose, not a schema.
# A run registers OHPU_HOST_LIMIT_SCALE to relax the length limits below; 1 keeps the historical limits.
HOST_LIMIT_SCALE = int(os.environ.get("OHPU_HOST_LIMIT_SCALE", "1"))
MAX_MEMORY_GUIDANCE_CHARS = 800 * HOST_LIMIT_SCALE
PROGRAM_OPERATIONS = ("select_memory", "analyze", "generate", "finish")
MAX_STEP_PROMPT_CHARS = 4_000
MAX_REQUEST_STATE_CHARS = 8_000 * HOST_LIMIT_SCALE
REASONING_EFFORT_DOMAIN = frozenset({"none", "low", "medium", "high"})
CHANGE_MANIFEST_NAME = "CHANGE_MANIFEST"
MAX_MANIFEST_ENTRY_CHARS = 240
MAX_PROGRAM_CHARS = 40_000
MAX_AST_NODES = 4_500
MAX_INPUT_CHARS = 256_000 * HOST_LIMIT_SCALE
MAX_OUTPUT_CHARS = 32_000 * HOST_LIMIT_SCALE
DEFAULT_TIMEOUT_SECONDS = 1.0
DEFAULT_LINE_BUDGET = 20_000
WORKER_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024

DEFAULT_PROGRAM_SOURCE = '''\
"""Personalized harness program: remember, recall, apply, guard.

Search may change USER_MODEL, helper functions, and any of the four hooks.
Every hook is a pure function: it receives JSON data and returns a JSON patch.
An empty patch keeps the engine's base plan for that stage.

The four hooks are the four questions a personalized harness has to answer:
what to keep about this user, what this request should see of them, how to
answer for them, and when not to personalize. Every hook except remember also
receives context["memory_facts"], the engine-owned compact memory parsed into
bounded per-section bullets, so a branch can condition on what has been
learned about this user AND on the request in front of it. A branch that
depends on neither produces the same decision for every request: that is a
per-user constant, which the engine reports separately from a realized
per-request decision.
"""

PROGRAM_API_VERSION = 1

# Learned information may be represented as typed constants and control flow,
# rather than only as prose in a profile or memory file.
USER_MODEL = {}

# Stages this revision changes, and the realized effect expected at the next
# window. Not a hook dependency: declaring intent is not a code change.
CHANGE_MANIFEST = {}


def remember(context):
    """What to keep about this user, from the causal event stream.

    keep_events (indices into context["events"]), drop_channels,
    priority_channels, max_chars, guidance. Compiles the persistent memory
    document the next window reads. Its layout is not yours to fix: the
    rewriter shapes the document, and guidance is how you influence it."""
    return {}


def recall(context):
    """What this request should see of the user, by context["stage"]:

    evidence -> mode, top_k, budget_chars (how much causal evidence to pull);
    context  -> memory_sections, include_history, max_history_chars,
                keep_history_units (what reaches the prompt)."""
    return {}


def apply(context):
    """How to answer for this user, by context["stage"]:

    generation -> block_order, drop_blocks, budget, prefix, suffix,
                  structured_instruction, allow, deny, order, temperature,
                  top_p, max_tokens, reasoning_effort, workflow, max_turns,
                  stop_rule, plan_prompt, refine_prompt, max_tool_calls;
    arguments  -> defaults, overrides, drop_arguments (rewrite a tool call).

    The generation stage runs ONCE per action and its decision is threaded
    through context assembly, action space, decoding and workflow, so one
    request gets one coherent plan. context["base"] reports which fields this
    boundary can actually express."""
    return {}


def guard(context):
    """When not to personalize, and how to check the answer, by
    context["phase"]: validate -> violations; postprocess -> output.

    The cheapest place to make a negative preference operational."""
    return {}
'''


class ProgramError(RuntimeError):
    """Base class for publish-time and runtime program failures."""


class ProgramValidationError(ProgramError):
    """The source does not satisfy the pure four-stage program contract."""


class ProgramExecutionError(ProgramError):
    """A validated hook failed, timed out, or returned an invalid value."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


@dataclass(frozen=True)
class ProgramFingerprint:
    api_version: int
    sha256: str
    hook_sha256: Mapping[str, str]
    # Stages this revision claims to change, with the realized effect the
    # author expects at the next window.  It is deliberately NOT a hook
    # dependency: declaring intent must never by itself look like a code
    # change, and the engine verifies the claim against observed effects.
    change_manifest: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "sha256": self.sha256,
            "hook_sha256": dict(self.hook_sha256),
            "change_manifest": dict(self.change_manifest),
        }


@dataclass(frozen=True)
class HookOutcome:
    patch: dict[str, Any]
    report: dict[str, Any]


_FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.DictComp,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.ListComp,
    ast.MatchClass,
    ast.Nonlocal,
    ast.Raise,
    ast.Set,
    ast.SetComp,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
    ast.GeneratorExp,
)
_SAFE_BUILTIN_NAMES = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "len", "list", "max", "min", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip",
})
_SAFE_METHOD_NAMES = frozenset({
    "append", "count", "endswith", "extend", "get", "index", "items",
    "join", "keys", "lower", "replace", "split", "startswith", "strip",
    "upper", "values",
})
_FORBIDDEN_CALL_NAMES = frozenset({
    "breakpoint", "compile", "delattr", "dir", "eval", "exec", "getattr",
    "globals", "hasattr", "help", "input", "locals", "memoryview", "open",
    "print", "setattr", "type", "vars", "__import__",
})
# What to write instead. A rejection that only names the banned construct
# costs a whole update opportunity: the author retries, guesses again, and
# the window closes with no proposal. These are the constructs an ordinary
# Python author reaches for first, so each one names its replacement.
FORBIDDEN_NODE_ALTERNATIVES: Mapping[str, str] = {
    "ListComp": "build the list with a for loop and .append()",
    "DictComp": "build the dict with a for loop and item assignment",
    "SetComp": "build a list with a for loop, then call set() on it",
    "GeneratorExp": "use a for loop; comprehensions of every kind are rejected",
    "Lambda": "define a module-level helper function instead",
    "While": "use a bounded for loop over range() or a sequence",
    "Try": "check the value before using it; exceptions are unavailable",
    "Raise": "return a decision instead of raising",
    "Set": "use a list literal; set literals are rejected",
    "With": "not available; hooks perform no I/O",
    "ClassDef": "use module-level literals and plain functions",
    "Import": "not available; hooks receive everything through context",
    "ImportFrom": "not available; hooks receive everything through context",
    "Global": "return the value instead of mutating module state",
    "Nonlocal": "return the value instead of mutating an outer scope",
    "Delete": "rebind the name instead of deleting it",
}


def language_subset() -> dict[str, list[str]]:
    """The exact Python subset a harness program may use.

    Rendered into the proposer's contract so the two cannot drift: every
    rejection this list prevents is one recovered update opportunity.
    """

    return {
        "safe_builtins": sorted(_SAFE_BUILTIN_NAMES),
        "safe_methods": sorted(_SAFE_METHOD_NAMES),
        "forbidden_calls": sorted(_FORBIDDEN_CALL_NAMES),
        "forbidden_syntax": sorted(node.__name__ for node in _FORBIDDEN_NODES),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


_AUDIT_DIGEST_KEY = os.urandom(32)


def _digest_json(value: Any) -> str:
    """Process-local audit token; not a cross-user enumerable content hash."""

    return hmac.new(
        _AUDIT_DIGEST_KEY,
        _canonical_json(value).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def audit_token(value: Any) -> str:
    """Return a process-local, non-enumerable token for a realized decision.

    Traces use this to compare structural effects without persisting prompts,
    histories, argument values, or model outputs.  The random process key means
    the token is useful only inside one search run and cannot be used as a
    cross-run dictionary oracle over private user data.
    """

    _validate_json_value(value, path="audit_value")
    return _digest_json(value)


def _validate_json_value(value: Any, *, path: str = "result") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ProgramExecutionError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProgramExecutionError(f"{path} contains a non-string key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ProgramExecutionError(
        f"{path} must be JSON-compatible, got {type(value).__name__}"
    )


def _validate_patch_contract(
    hook: str,
    context: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> None:
    """Reject unknown, wrongly typed, or boundary-inert decision fields."""

    if not patch:
        return
    name = str(hook).strip().lower()
    stage = context_stage(name, context)
    contract_key = stage_key(name, stage)
    keys = STAGE_DECISION_KEYS.get(contract_key)
    if keys is None:
        raise ProgramExecutionError(
            f"{name} does not support execution stage {stage!r}"
        )
    allowed = set(keys)
    if contract_key == "apply:generation":
        try:
            turn = int(context.get("turn") or 0)
        except (TypeError, ValueError, OverflowError):
            turn = 0
        # The schema-only runner consults the workflow again after an
        # observation only to decide whether its already-bounded loop should
        # stop.  Accepting a fresh workflow, turn ceiling, or prompt here
        # would make those fields look effective even though the executor
        # never consumes them.  A native benchmark owns a real external step
        # boundary, so each new action may still choose the full plan.
        if turn > 0 and str(context.get("boundary") or "") != "native benchmark":
            allowed = {"stop_rule"}
    unknown = set(patch) - allowed
    if unknown:
        raise ProgramExecutionError(
            f"{name} stage {stage or 'default'} returned unsupported decision "
            "fields; this stage accepts: " + ", ".join(sorted(allowed))
        )
    backend_key = "memory_backend" if contract_key == "remember" else "memory_query"
    if backend_key in patch:
        decision = patch[backend_key]
        if not isinstance(decision, dict) or decision.get("operation") not in {
            "schema", "upsert", "delete", "clear", "sql", "search", "model", "finish", "read_all", "understand",
            "decide", "rank"
        }:
            raise ProgramExecutionError("programmable memory requires a supported operation")
        if contract_key == "recall:context" and set(patch) != {"memory_query"}:
            raise ProgramExecutionError("programmable recall cannot mix memory selectors")

    def string_list(key: str) -> None:
        value = patch.get(key)
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
        ):
            raise ProgramExecutionError(f"{name}.{key} must be a list of strings")

    def integer(key: str) -> None:
        value = patch.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ProgramExecutionError(f"{name}.{key} must be an integer")

    def text(key: str) -> None:
        if key in patch and not isinstance(patch[key], str):
            raise ProgramExecutionError(f"{name}.{key} must be text")

    def boolean(key: str) -> None:
        if key in patch and not isinstance(patch[key], bool):
            raise ProgramExecutionError(f"{name}.{key} must be boolean")

    if contract_key == "recall:evidence":
        for key in ("top_k", "budget_chars"):
            integer(key)
        if "mode" in patch and patch["mode"] not in {
            "none", "recency", "bm25", "hybrid", "relevance"
        }:
            raise ProgramExecutionError(
                "recall.mode is outside the supported domain: "
                "none, recency, bm25, hybrid (relevance is a legacy alias for bm25)"
            )
    elif contract_key == "recall:context":
        string_list("memory_sections")
        string_list("memory_fact_ids")
        mode = patch.get("memory_mode")
        selectors = {"memory_sections", "memory_fact_ids"} & set(patch)
        if "memory_mode" in patch and (not isinstance(mode, str) or mode not in {"base", "select", "no_user_memory"}):
            raise ProgramExecutionError("invalid recall.memory_mode")
        if mode in {"base", "no_user_memory"} and selectors:
            raise ProgramExecutionError("base/no_user_memory cannot also select memory")
        if mode == "select" and len(selectors) != 1:
            raise ProgramExecutionError("select requires exactly one memory selector")
        if "retrieval_delta" in patch:
            from .retrieval_policy import validate_delta
            try:
                validate_delta(patch["retrieval_delta"])
            except ValueError as exc:
                raise ProgramExecutionError(str(exc)) from exc
            if set(patch) != {"retrieval_delta"}:
                raise ProgramExecutionError("retrieval_delta is mutually exclusive with other recall actions")
        if "memory_sections" in patch and "memory_fact_ids" in patch:
            raise ProgramExecutionError("select memory sections or fact ids, not both")
        integer("max_history_chars")
        boolean("include_history")
        values = patch.get("keep_history_units")
        if values is not None and (
            not isinstance(values, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in values
            )
        ):
            raise ProgramExecutionError(
                "recall.keep_history_units must be a list of integers"
            )
    elif contract_key == "apply:generation":
        string_list("block_order")
        string_list("drop_blocks")
        string_list("allow")
        string_list("deny")
        string_list("order")
        integer("budget")
        integer("max_tokens")
        integer("max_turns")
        integer("max_tool_calls")
        text("prefix")
        text("suffix")
        text("plan_prompt")
        text("refine_prompt")
        boolean("structured_instruction")
        for key in ("temperature", "top_p"):
            value = patch.get(key)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise ProgramExecutionError(f"apply.{key} must be numeric")
        if (
            "reasoning_effort" in patch
            and patch["reasoning_effort"] not in REASONING_EFFORT_DOMAIN
        ):
            raise ProgramExecutionError(
                "apply.reasoning_effort is outside the supported domain: "
                + ", ".join(sorted(REASONING_EFFORT_DOMAIN))
            )
        if "workflow" in patch and patch["workflow"] not in {
            "single_call", "agentic_loop", "plan_execute", "refine",
            "plan_execute_refine", "program",
        }:
            raise ProgramExecutionError(
                "apply.workflow is outside the supported domain: single_call, "
                "agentic_loop, plan_execute, refine, plan_execute_refine, program"
            )
        if "stop_rule" in patch and patch["stop_rule"] not in {
            "task_complete", "after_action", "first_action", "immediate"
        }:
            raise ProgramExecutionError(
                "apply.stop_rule is outside the supported domain: "
                "task_complete, after_action, first_action, immediate"
            )
    elif contract_key == "apply:step":
        operation = patch.get("operation")
        if not isinstance(operation, str) or operation not in PROGRAM_OPERATIONS:
            raise ProgramExecutionError("apply.step requires a supported operation")
        text("prompt")
        text("final_output")
        string_list("selected_ids")
        if "prompt" in patch and (
            operation not in {"analyze", "generate"}
            or len(patch["prompt"]) > MAX_STEP_PROMPT_CHARS
        ):
            raise ProgramExecutionError("apply.step prompt is invalid for this operation or too long")
        if "final_output" in patch and operation != "finish":
            raise ProgramExecutionError("only finish may return an output")
        if operation == "select_memory":
            if not isinstance(patch.get("selected_ids"), list):
                raise ProgramExecutionError("select_memory requires selected_ids")
        elif "selected_ids" in patch:
            raise ProgramExecutionError("only select_memory may use selected_ids")
        if "state" in patch and (
            not isinstance(patch["state"], dict)
            or len(_canonical_json(patch["state"])) > MAX_REQUEST_STATE_CHARS
        ):
            raise ProgramExecutionError("apply.step state must be a bounded JSON object")
    elif contract_key == "apply:arguments":
        string_list("drop_arguments")
        for key in ("defaults", "overrides"):
            if key in patch and not isinstance(patch[key], dict):
                raise ProgramExecutionError(f"apply.{key} must be an object")
    elif contract_key == "remember":
        if "memory_backend" in patch and ("state_schema" in patch or "state_records" in patch):
            raise ProgramExecutionError("choose programmable or compact state, not both")
        if "state_schema" in patch and "state_records" in patch:
            raise ProgramExecutionError("choose state_schema or state_records, not both")
        if "state_schema" in patch and (
            not isinstance(patch["state_schema"], dict)
            or len(_canonical_json(patch["state_schema"])) > 4_000
        ):
            raise ProgramExecutionError("remember.state_schema must be a bounded JSON schema object")
        if "state_records" in patch and (
            not isinstance(patch["state_records"], list)
            or len(patch["state_records"]) > 32
        ):
            raise ProgramExecutionError("remember.state_records must be a bounded list")
        for key in ("drop_channels", "priority_channels"):
            string_list(key)
        integer("max_chars")
        values = patch.get("keep_events")
        if values is not None and (
            not isinstance(values, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in values
            )
        ):
            raise ProgramExecutionError(
                "remember.keep_events must be a list of integers"
            )
        guidance = patch.get("guidance")
        if guidance is not None:
            if not isinstance(guidance, str):
                raise ProgramExecutionError("remember.guidance must be text")
            if len(guidance) > MAX_MEMORY_GUIDANCE_CHARS:
                raise ProgramExecutionError(
                    f"remember.guidance exceeds {MAX_MEMORY_GUIDANCE_CHARS} "
                    "characters"
                )
    elif contract_key == "guard:validate":
        string_list("violations")
        text("repair_prompt")
    elif contract_key == "guard:postprocess":
        text("output")


class _ProgramValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_names: set[str] = set()

    def validate(self, tree: ast.Module) -> None:
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES:
            raise ProgramValidationError(
                f"program has {len(nodes)} AST nodes; limit is {MAX_AST_NODES}"
            )
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if not isinstance(node.value.value, str):
                    raise ProgramValidationError("top-level expressions must be docstrings")
                continue
            if isinstance(node, ast.AnnAssign):
                # Annotation expressions execute while the module is loaded
                # and can mutate state without becoming a hook dependency.
                raise ProgramValidationError(
                    "top-level annotated assignments are not allowed"
                )
            if isinstance(node, ast.Assign):
                targets = node.targets
                if not targets or any(not isinstance(target, ast.Name) for target in targets):
                    raise ProgramValidationError(
                        "top-level state must bind literal values to simple names"
                    )
                if any(
                    target.id in _SAFE_BUILTIN_NAMES
                    or target.id in _FORBIDDEN_CALL_NAMES
                    or target.id in PROGRAM_HOOKS
                    for target in targets
                ):
                    raise ProgramValidationError(
                        "top-level state must not shadow builtins or stage hooks"
                    )
                value = node.value
                try:
                    literal = ast.literal_eval(value)
                except (ValueError, TypeError):
                    raise ProgramValidationError(
                        "top-level state must be a literal; compute values inside hooks"
                    ) from None
                try:
                    _validate_json_value(literal, path="top_level_state")
                except ProgramExecutionError:
                    raise ProgramValidationError(
                        "top-level state must contain only JSON-compatible values"
                    ) from None
                continue
            if isinstance(node, ast.FunctionDef):
                if (
                    node.name in _SAFE_BUILTIN_NAMES
                    or node.name in _FORBIDDEN_CALL_NAMES
                ):
                    raise ProgramValidationError(
                        "helper functions must not shadow safe builtins"
                    )
                if node.decorator_list:
                    raise ProgramValidationError("function decorators are not allowed")
                if (
                    node.args.defaults
                    or any(value is not None for value in node.args.kw_defaults)
                    or node.returns is not None
                    or any(
                        argument.annotation is not None
                        for argument in (
                            *node.args.posonlyargs,
                            *node.args.args,
                            *node.args.kwonlyargs,
                        )
                    )
                    or (
                        node.args.vararg is not None
                        and node.args.vararg.annotation is not None
                    )
                    or (
                        node.args.kwarg is not None
                        and node.args.kwarg.annotation is not None
                    )
                ):
                    raise ProgramValidationError(
                        "function defaults and annotations are not allowed"
                    )
                if any(
                    isinstance(child, ast.FunctionDef) and child is not node
                    for child in ast.walk(node)
                ):
                    raise ProgramValidationError(
                        "nested function definitions are not allowed"
                    )
                if node.name in self.function_names:
                    raise ProgramValidationError(
                        f"duplicate function {node.name!r} is not allowed"
                    )
                self.function_names.add(node.name)
                continue
            raise ProgramValidationError(
                f"top-level {type(node).__name__} is not allowed"
            )

        missing = set(PROGRAM_HOOKS) - self.function_names
        if missing:
            raise ProgramValidationError(
                "program must define all four stage hooks ("
                + ", ".join(PROGRAM_HOOKS)
                + "); missing "
                + ", ".join(sorted(missing))
            )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in PROGRAM_HOOKS:
                args = node.args
                if (
                    len(args.args) != 1
                    or args.posonlyargs
                    or args.kwonlyargs
                    or args.vararg is not None
                    or args.kwarg is not None
                    or args.defaults
                    or args.kw_defaults
                ):
                    raise ProgramValidationError(
                        f"{node.name} must accept exactly one context argument"
                    )
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.MatchClass):
                raise ProgramValidationError(
                    "MatchClass is not allowed in harness programs"
                )
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and (
                    node.id in self.function_names
                    or node.id in _SAFE_BUILTIN_NAMES
                )
            ):
                parent = parents.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    kind = (
                        "safe builtin names"
                        if node.id in _SAFE_BUILTIN_NAMES
                        else "function names"
                    )
                    raise ProgramValidationError(
                        f"{kind} may only be used as direct calls: {node.id!r} at line {node.lineno}; "
                        "rename any local variable that shadows this function or builtin"
                    )
            if (
                isinstance(node, ast.Attribute)
                and not node.attr.startswith("_")
                and node.attr in _SAFE_METHOD_NAMES
            ):
                parent = parents.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    raise ProgramValidationError(
                        "attributes may only be used as direct method calls"
                    )
        self.visit(tree)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _FORBIDDEN_NODES):
            name = type(node).__name__
            hint = FORBIDDEN_NODE_ALTERNATIVES.get(name)
            raise ProgramValidationError(
                f"{name} is not allowed in harness programs"
                + (f"; {hint}" if hint else "")
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_") or node.id in _FORBIDDEN_CALL_NAMES:
            raise ProgramValidationError(
                f"name {node.id!r} is not allowed; the callable names a hook "
                "may use are: " + ", ".join(sorted(_SAFE_BUILTIN_NAMES))
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr not in _SAFE_METHOD_NAMES:
            raise ProgramValidationError(
                f"attribute {node.attr!r} is not allowed; use pure JSON operations"
            )
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in PROGRAM_HOOKS:
                raise ProgramValidationError(
                    "stage hooks must not call one another"
                )
            if name not in _SAFE_BUILTIN_NAMES and name not in self.function_names:
                raise ProgramValidationError(
                    f"call to {name!r} is not allowed; callable builtins are: "
                    + ", ".join(sorted(_SAFE_BUILTIN_NAMES))
                    + " (plus helpers this program defines)"
                )
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in _SAFE_METHOD_NAMES:
                raise ProgramValidationError(
                    f"method call {node.func.attr!r} is not allowed; callable "
                    "methods are: " + ", ".join(sorted(_SAFE_METHOD_NAMES))
                )
        else:
            raise ProgramValidationError("indirect function calls are not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, str) and len(value) > 8_000:
            raise ProgramValidationError("string literal exceeds 8000 characters")
        if isinstance(value, int) and abs(value) > 1_000_000:
            raise ProgramValidationError("integer literal exceeds the safe bound")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Pow, ast.LShift, ast.RShift, ast.MatMult)):
            raise ProgramValidationError(
                f"operator {type(node.op).__name__} is not allowed"
            )
        self.generic_visit(node)


def validate_program_source(source: str) -> ProgramFingerprint:
    """Validate source and return stable whole-program and per-hook hashes."""

    if not isinstance(source, str):
        raise ProgramValidationError("program source must be text")
    if len(source) > MAX_PROGRAM_CHARS:
        raise ProgramValidationError(
            f"program has {len(source)} characters; limit is {MAX_PROGRAM_CHARS}"
        )
    try:
        tree = ast.parse(source, filename=PROGRAM_FILENAME)
    except SyntaxError as exc:
        raise ProgramValidationError(
            f"invalid {PROGRAM_FILENAME} syntax at line {exc.lineno}: {exc.msg}"
        ) from exc
    validator = _ProgramValidator()
    validator.validate(tree)

    version: int | None = None
    hook_hashes: dict[str, str] = {}
    assignments: dict[str, ast.Assign | ast.AnnAssign] = {}
    helpers: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node
        elif isinstance(node, ast.FunctionDef) and node.name not in PROGRAM_HOOKS:
            helpers[node.name] = node

    def external_names(function: ast.FunctionDef) -> set[str]:
        local = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        local.update(
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        return {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in local
            and node.id not in _SAFE_BUILTIN_NAMES
        }

    def behavior_ast(node: ast.AST) -> str:
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node, clean=False) is not None:
            node = copy.copy(node)
            node.body = node.body[1:]
        return ast.dump(node, annotate_fields=True, include_attributes=False)

    def dependency_ast(function: ast.FunctionDef) -> str:
        queue = list(external_names(function))
        seen: set[str] = set()
        dependency_nodes: dict[tuple[str, str], ast.AST] = {}
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in helpers:
                dependency_nodes[("helper", name)] = helpers[name]
                queue.extend(external_names(helpers[name]))
            elif name in assignments:
                dependency_nodes[("state", name)] = assignments[name]
        return "\n".join(
            behavior_ast(node)
            for _key, node in sorted(dependency_nodes.items())
        )

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "PROGRAM_API_VERSION"
                   for target in targets):
                try:
                    version = int(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    version = None
        if isinstance(node, ast.FunctionDef) and node.name in PROGRAM_HOOKS:
            canonical = behavior_ast(node)
            dependencies = dependency_ast(node)
            hook_hashes[node.name] = hashlib.sha256(
                (dependencies + "\n" + canonical).encode("utf-8")
            ).hexdigest()
    if version != PROGRAM_API_VERSION:
        raise ProgramValidationError(
            f"PROGRAM_API_VERSION must equal {PROGRAM_API_VERSION}, got {version!r}"
        )
    manifest: dict[str, str] = {}
    manifest_node = assignments.get(CHANGE_MANIFEST_NAME)
    if manifest_node is not None:
        try:
            raw_manifest = ast.literal_eval(manifest_node.value)
        except (ValueError, TypeError):
            raw_manifest = None
        if not isinstance(raw_manifest, dict):
            raise ProgramValidationError(
                f"{CHANGE_MANIFEST_NAME} must be a literal object mapping "
                "stage -> expected realized effect"
            )
        for key, value in raw_manifest.items():
            hook_name = str(key).strip().lower()
            if hook_name not in PROGRAM_HOOKS:
                raise ProgramValidationError(
                    f"{CHANGE_MANIFEST_NAME} keys must name a stage ("
                    + ", ".join(PROGRAM_HOOKS)
                    + f"), got {key!r}"
                )
            if not isinstance(value, str) or not value.strip():
                raise ProgramValidationError(
                    f"{CHANGE_MANIFEST_NAME}[{key!r}] must be a non-empty "
                    "description of the expected realized effect"
                )
            manifest[hook_name] = value.strip()[:MAX_MANIFEST_ENTRY_CHARS]
        declared_but_absent = sorted(set(manifest) - set(hook_hashes))
        if declared_but_absent:
            raise ProgramValidationError(
                f"{CHANGE_MANIFEST_NAME} names stages without a hook: "
                + ", ".join(declared_but_absent)
            )
    return ProgramFingerprint(
        api_version=PROGRAM_API_VERSION,
        sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        hook_sha256=hook_hashes,
        change_manifest=manifest,
    )


class HarnessProgram:
    """Immutable validated source plus a fail-closed isolated hook runner."""

    def __init__(
        self,
        source: str,
        *,
        path: Path | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        line_budget: int = DEFAULT_LINE_BUDGET,
    ) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "fingerprint", validate_program_source(source))
        tree = ast.parse(source)
        empty_hooks = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in PROGRAM_HOOKS:
                body = node.body[1:] if ast.get_docstring(node, clean=False) is not None else node.body
                if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Dict) and not body[0].value.keys:
                    empty_hooks.add(node.name)
        object.__setattr__(self, "_empty_hooks", frozenset(empty_hooks))
        object.__setattr__(self, "_empty_hook_step_bound", sum(1 for _ in ast.walk(tree)) + len(tree.body) + 3)
        object.__setattr__(
            self, "timeout_seconds", max(0.05, float(timeout_seconds))
        )
        object.__setattr__(self, "line_budget", max(100, int(line_budget)))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("HarnessProgram is immutable")
        object.__setattr__(self, name, value)

    @property
    def source(self) -> str:
        """Validated source; immutable after construction for audit integrity."""

        return self._source

    @classmethod
    def from_path(cls, path: Path | str) -> "HarnessProgram":
        unresolved = Path(path).expanduser()
        if unresolved.is_symlink():
            raise ProgramValidationError("harness.py must not be a symlink")
        resolved = unresolved.absolute()
        try:
            source = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProgramValidationError(f"could not read {resolved}: {exc}") from exc
        return cls(source, path=resolved)

    def call(
        self,
        hook: str,
        context: Mapping[str, Any],
        *,
        contract_context: Mapping[str, Any] | None = None,
    ) -> HookOutcome:
        name = str(hook).strip().lower()
        if name not in PROGRAM_HOOKS:
            raise ProgramExecutionError(
                f"unknown harness stage {hook!r}; stages are "
                + ", ".join(PROGRAM_HOOKS)
            )
        request = {
            "source": self.source,
            "hook": name,
            "context": dict(context),
            "line_budget": self.line_budget,
        }
        try:
            encoded = _canonical_json(request)
        except (TypeError, ValueError) as exc:
            raise ProgramExecutionError(f"hook input is not valid JSON: {exc}") from exc
        if len(encoded) > MAX_INPUT_CHARS:
            raise ProgramExecutionError(
                f"hook input has {len(encoded)} characters; limit is {MAX_INPUT_CHARS}"
            )
        input_digest = _digest_json(request["context"])
        started = time.monotonic()
        base_report = {
            "hook": name,
            "program_sha256": self.fingerprint.sha256,
            "input_digest": input_digest,
        }
        if self.source == DEFAULT_PROGRAM_SOURCE or (name in self._empty_hooks and self.line_budget >= self._empty_hook_step_bound):
            return HookOutcome(
                patch={},
                report={
                    **base_report,
                    "status": "ok",
                    "execution_backend": "validated_empty_patch",
                    "duration_ms": 0.0,
                    "changed": False,
                    "decision_keys": [],
                    "unknown_key_count": 0,
                    "output_digest": _digest_json({}),
                },
            )
        process_created = started
        communication_ended = started
        cleanup_ms = 0.0
        command = [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--worker"]
        try:
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(Path(os.devnull).parent),
                env={
                    "PYTHONIOENCODING": "utf-8",
                    # A code harness must replay deterministically even when it
                    # uses an allowed set/dict in a helper function.
                    "PYTHONHASHSEED": "0",
                    **({"OHPU_HOST_LIMIT_SCALE": str(HOST_LIMIT_SCALE)} if HOST_LIMIT_SCALE != 1 else {}),
                },
            ) as process:
                process_created = time.monotonic()
                try:
                    stdout, stderr = process.communicate(encoded, timeout=self.timeout_seconds)
                    communication_ended = time.monotonic()
                except subprocess.TimeoutExpired:
                    communication_ended = time.monotonic()
                    process.kill()
                    process.communicate()
                    cleanup_ms = (time.monotonic() - communication_ended) * 1000
                    raise
                completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired as exc:
            report = dict(base_report)
            report.update({
                "status": "timeout",
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "diagnostic": {"kind": "timeout", "error_type": "TimeoutError",
                    "time_budget_seconds": self.timeout_seconds,
                    "time_budget_scope": "communication_after_process_creation",
                    "timing_ms": {"process_creation": round((process_created - started) * 1000, 3),
                        "communication": round((communication_ended - process_created) * 1000, 3),
                        "cleanup": round(cleanup_ms, 3)},
                    "guidance": "Inspect process creation, communication and cleanup timings before attributing this timeout to hook logic; preserve all execution limits."},
            })
            raise ProgramExecutionError(
                f"{name} hook exceeded {self.timeout_seconds:.2f}s",
                report=report,
            ) from exc
        except OSError as exc:
            report = dict(base_report)
            report.update({
                "status": "spawn_error",
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
            })
            raise ProgramExecutionError(
                f"{name} hook worker could not start",
                report=report,
            ) from exc
        duration_ms = round((time.monotonic() - started) * 1000, 3)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "worker failed").strip()
            report = dict(base_report)
            report.update({"status": "error", "duration_ms": duration_ms})
            raise ProgramExecutionError(
                f"{name} hook failed: {detail[:500]}", report=report
            )
        if len(completed.stdout) > MAX_OUTPUT_CHARS:
            report = dict(base_report)
            report.update({"status": "oversize", "duration_ms": duration_ms})
            raise ProgramExecutionError(
                f"{name} hook output exceeds {MAX_OUTPUT_CHARS} characters",
                report=report,
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            report = dict(base_report)
            report.update({"status": "error", "duration_ms": duration_ms})
            raise ProgramExecutionError(
                f"{name} worker returned invalid JSON", report=report
            ) from exc
        if not isinstance(payload, dict) or not payload.get("ok"):
            message = str(payload.get("error") if isinstance(payload, dict) else payload)
            report = dict(base_report)
            report.update({"status": "error", "duration_ms": duration_ms})
            if isinstance(payload, dict) and isinstance(payload.get("diagnostic"), dict):
                report["diagnostic"] = payload["diagnostic"]
                if report["diagnostic"].get("kind") == "step_limit":
                    report["diagnostic"]["step_budget"] = self.line_budget
            raise ProgramExecutionError(
                f"{name} hook failed: {message[:500]}", report=report
            )
        patch = payload.get("result")
        if not isinstance(patch, dict):
            report = dict(base_report)
            report.update({"status": "invalid_result", "duration_ms": duration_ms})
            raise ProgramExecutionError(
                f"{name} hook must return a dict, got {type(patch).__name__}",
                report=report,
            )
        _validate_json_value(patch)
        try:
            validation_context = dict(request["context"])
            validation_context.update(dict(contract_context or {}))
            _validate_patch_contract(name, validation_context, patch)
        except ProgramExecutionError as exc:
            report = dict(base_report)
            report.update({"status": "invalid_result", "duration_ms": duration_ms})
            raise ProgramExecutionError(str(exc), report=report) from exc
        known_keys = PROGRAM_DECISION_KEYS[name]
        report = dict(base_report)
        report.update({
            "status": "ok",
            "duration_ms": duration_ms,
            "execution_backend": "isolated_no_site",
            "timing_ms": {"process_creation": round((process_created - started) * 1000, 3),
                "communication": round((communication_ended - process_created) * 1000, 3), "cleanup": 0.0},
            "changed": bool(patch),
            "decision_keys": sorted(key for key in patch if key in known_keys),
            # An adversarial program can derive a key from raw query/history.
            # Record only its count, never the unknown key text.
            "unknown_key_count": sum(1 for key in patch if key not in known_keys),
            "output_digest": _digest_json(patch),
        })
        return HookOutcome(patch=patch, report=report)


def load_harness_program(path: Path | str, *, required: bool = False) -> HarnessProgram | None:
    """Load ``harness.py`` without mutating the bundle.

    Legacy four-file bundles remain readable when ``required`` is false.  A v2
    publish gate should pass ``required=True`` and fail closed.
    """

    unresolved = Path(path).absolute()
    if unresolved.is_symlink():
        raise ProgramValidationError(
            f"program path must not be a symlink: {unresolved}"
        )
    if not unresolved.exists():
        if required:
            raise ProgramValidationError(f"required program is missing: {unresolved}")
        return None
    return HarnessProgram.from_path(unresolved)


def _safe_range(*args: int) -> range:
    values = [int(value) for value in args]
    result = range(*values)
    if len(result) > 10_000:
        raise RuntimeError("range exceeds 10000 iterations")
    return result


def _safe_enumerate(values: Any, start: int = 0) -> list[tuple[int, Any]]:
    result = list(enumerate(values, int(start)))
    if len(result) > 10_000:
        raise RuntimeError("enumerate exceeds 10000 items")
    return result


def _safe_reversed(values: Any) -> list[Any]:
    result = list(reversed(values))
    if len(result) > 10_000:
        raise RuntimeError("reversed exceeds 10000 items")
    return result


def _safe_zip(*values: Any) -> list[tuple[Any, ...]]:
    result = list(zip(*values))
    if len(result) > 10_000:
        raise RuntimeError("zip exceeds 10000 items")
    return result


class _DeterministicSet(set):
    def __iter__(self):
        return iter(sorted(set.__iter__(self), key=_canonical_json))

    def __repr__(self):
        return repr(list(self))

    def _operate(self, other, operation, *, inplace=False):
        result = operation(self, other)
        if result is NotImplemented:
            return NotImplemented
        checked = _safe_set(result)
        if inplace:
            set.clear(self)
            set.update(self, checked)
            return self
        return checked

    def __and__(self, other):
        return self._operate(other, set.__and__)

    def __or__(self, other):
        return self._operate(other, set.__or__)

    def __sub__(self, other):
        return self._operate(other, set.__sub__)

    def __xor__(self, other):
        return self._operate(other, set.__xor__)

    def __iand__(self, other):
        return self._operate(other, set.__and__, inplace=True)

    def __ior__(self, other):
        return self._operate(other, set.__or__, inplace=True)

    def __isub__(self, other):
        return self._operate(other, set.__sub__, inplace=True)

    def __ixor__(self, other):
        return self._operate(other, set.__xor__, inplace=True)


def _safe_set(values: Any = ()) -> _DeterministicSet:
    """Bounded sets with deterministic iteration and the existing list-style repr."""
    result = _DeterministicSet()
    for item in values:
        _validate_json_value(item, path="set_item")
        set.add(result, item)
        if len(result) > 10_000:
            raise RuntimeError("set exceeds 10000 items")
    return result


class _HookStepLimit(RuntimeError):
    pass


def _execution_diagnostic(error: BaseException) -> dict[str, Any]:
    hints = {
        "TypeError": "Check operand and argument types at this line; guard optional values before using them.",
        "ValueError": "Validate values before numeric conversion or parsing; missing data needs a fallback.",
        "KeyError": "Use .get() and handle absent keys in the supplied context or model observations.",
        "IndexError": "Check sequence length before indexing observations or selected records.",
        "AttributeError": "Check the value shape and use only methods declared in the language contract.",
        "NameError": "Define helper names and variables before using them; only declared builtins are available.",
        "ZeroDivisionError": "Handle empty collections and zero denominators before division.",
        "RecursionError": "Replace recursive helpers with bounded loops.",
        "MemoryError": "Avoid large intermediate collections; respect the isolated worker memory limit.",
    }
    name = type(error).__name__
    diagnostic = {"kind": "runtime_error", "error_type": name if name in hints else "RuntimeError",
                  "guidance": hints.get(name, "Check this hook against the bounded language and decision contracts.")}
    if isinstance(error, _HookStepLimit):
        diagnostic.update(kind="step_limit", error_type="RuntimeError",
            guidance="Reduce repeated scans; tokenize the request once and bound memory traversal. Only harness.py execution consumes the step budget.")
    frame = error.__traceback__
    while frame is not None:
        if frame.tb_frame.f_code.co_filename == PROGRAM_FILENAME:
            diagnostic["source_line"] = frame.tb_lineno
        frame = frame.tb_next
    return diagnostic


def _worker_execute(request: Mapping[str, Any]) -> dict[str, Any]:
    source = str(request.get("source") or "")
    hook = str(request.get("hook") or "").strip().lower()
    context = request.get("context")
    line_budget = max(100, int(request.get("line_budget") or DEFAULT_LINE_BUDGET))
    validate_program_source(source)
    if hook not in PROGRAM_HOOKS:
        raise ProgramExecutionError(f"unknown stage {hook!r}")
    if not isinstance(context, dict):
        raise ProgramExecutionError("context must be a JSON object")

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": _safe_enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": _safe_range,
        "reversed": _safe_reversed,
        "round": round,
        "set": _safe_set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": _safe_zip,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    code = compile(source, PROGRAM_FILENAME, "exec", dont_inherit=True, optimize=2)
    steps = 0

    def trace(frame, event, arg):
        del arg
        nonlocal steps
        if frame.f_code.co_filename != PROGRAM_FILENAME:
            return None
        if event in {"line", "call"}:
            steps += 1
            if steps > line_budget:
                raise _HookStepLimit(f"hook exceeded {line_budget} execution steps")
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        exec(code, namespace, namespace)
        result = namespace[hook](context)
    finally:
        sys.settrace(previous)
    if not isinstance(result, dict):
        raise ProgramExecutionError(
            f"{hook} hook must return a dict, got {type(result).__name__}"
        )
    _validate_json_value(result)
    rendered = _canonical_json(result)
    if len(rendered) > MAX_OUTPUT_CHARS:
        raise ProgramExecutionError(
            f"{hook} hook output exceeds {MAX_OUTPUT_CHARS} characters"
        )
    return result


def _worker_main() -> int:
    try:
        # Set limits inside the fresh worker (not preexec_fn, which is unsafe
        # when the evaluator uses threads). Wall-clock timeout remains enforced
        # by the parent; these bounds contain one-line CPU/memory expressions
        # that a line tracer cannot interrupt before allocation.
        try:
            import resource

            resource.setrlimit(
                resource.RLIMIT_AS,
                (WORKER_MEMORY_LIMIT_BYTES, WORKER_MEMORY_LIMIT_BYTES),
            )
            resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        except (ImportError, OSError, ValueError):
            pass
        raw = sys.stdin.read(MAX_INPUT_CHARS + 1)
        if len(raw) > MAX_INPUT_CHARS:
            raise ProgramExecutionError("worker input exceeds the size limit")
        request = json.loads(raw)
        result = _worker_execute(request)
        sys.stdout.write(_canonical_json({"ok": True, "result": result}))
        return 0
    except BaseException as exc:  # worker must return a bounded, typed failure
        error = f"{type(exc).__name__}: {exc}"[:1000]
        sys.stdout.write(_canonical_json({"ok": False, "error": error,
                                         "diagnostic": _execution_diagnostic(exc)}))
        return 0


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    raise SystemExit(_worker_main())
