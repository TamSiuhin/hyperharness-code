"""Fixed W0 framework knowledge, separate from the learned generation skill."""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import sys

from . import program


PROTOCOL = "w0-framework-contract-v3-facts"
FROZEN_FIELDS = frozenset({"temperature", "top_p", "max_tokens", "reasoning_effort", "max_turns", "max_tool_calls"})

HISTORY_CONTEXT_GUIDANCE = """
## Runtime history context preference

Complete history available for strategy construction and storage does not imply
complete history in a model prompt. Prefer query-relevant retrieval, compact
summaries or incremental understanding, preserving useful conflicts and exceptions.
Start with a small relevant evidence set and expand selectively when it leaves a
specific evidence gap. Apply this preference to BOTH user-understanding calls and
final answer calls: compressing the output of a full-history call does not avoid
its input cost. A weak or empty match favors alternative retrieval or explicit
uncertainty, rather than automatic full-history replay.

Full-history context remains an optional strategy, not the default or a blanket
fallback. It can be reasonable when the history is small, fits comfortably and
offers useful evidence beyond the compact alternative. Consider the complete
model input, output headroom, latency and repeated token cost; a host character
capacity is a safety ceiling, not a target or a guarantee of context-window fit.
This is a soft design preference, not a ban, new hard limit, or scoring rule.
Keep all eligible source history available and existing runtime limits unchanged.
"""

GUIDE = """# W0 Framework Contract

This is immutable framework knowledge, not a learned personalization policy.
Read framework.json for the exact language, editable fields and frozen settings.
Read framework_reference/validation.py for authoritative field types, enums,
mutual exclusions and source validation. These rules do not belong in the skill.
framework_reference/executor.py and daily_memory.py show how inputs are built and
patches are consumed; memory_backend.py, structured_memory.py and retrieval_policy.py
define the optional nested memory operations. Read only the relevant sections.
context_examples.json contains anonymous structural examples, not user cases.

## Ownership and execution
The host compiles strictly causal user observations into rolling summary memory,
then assembles context, calls the answer model, validates and postprocesses output.
Your file contains USER_MODEL, CHANGE_MANIFEST, pure helper functions and the
four hooks remember(context), recall(context), apply(context), guard(context).
summary engine with KnowledgeBase, imports, database clients or six-hook code.
You may read only the supplied files. Framework reference files are read-only;
only harness.py is editable. Model transport, evaluation data and scoring remain
outside the program. Never override registered sampling or hard call budgets.
source, including read access, by the current conservative source gate. Do not
construct those patch keys dynamically to bypass that gate.

## Stages and inputs

- remember receives events, prior memory and base memory settings during memory
  compilation. keep_events contains event indices, not movie answers. guidance
  is a string of at most 800 characters AFTER concatenation. An event's observed
  not the current answer prompt directly.
- recall uses context['stage']: evidence or context. evidence chooses retrieval
  policy; context selects the memory/history passed onward. memory_sections and
  memory_fact_ids are alternatives. memory_mode is base, select or no_user_memory;
  select requires exactly one selector. Do not combine retrieval_delta with other
  selectors or mix memory_query with other recall actions.
- apply uses context['stage']: generation, arguments or step. generation receives
  task_kind, query, memory_facts, blocks, tools, turn and base. Only use fields
  the current base['expressible'] lists AND framework.json permits. A field in
  the generic schema is not necessarily supported by this benchmark boundary.
  prefix/suffix are text; structured_instruction is BOOLEAN, not instruction
  prose. allow/deny/order are TOOL names, not movie labels. Workflow choice can
  add model calls; the registered workflow/call limits must remain unchanged.
  arguments modifies tool arguments, not classification labels. step is used
  only by program-driven workflows, not automatically for every answer.
- guard uses context['phase'], NOT stage: validate or postprocess. validate
  receives base violations and returns a string list; postprocess receives base
  output and returns text. It must not substitute a guessed semantic answer.

Inputs are JSON values, not Python objects or a toolkit. Context keys differ by
stage; use the supplied reference and stage guards rather than assuming fields
exist everywhere. memory_facts maps section names to lists of bounded STRINGS,
not dictionaries. Plain text lines and list items both become facts. Do not call
fact.get('text'); use the fact string. memory_records instead contains dictionaries
with id, section and text for record selection. Neither is raw user history.
No native evaluation target is in context.

Movie observations use channel `reaction`; `tagged_movie` is a structured event
action, not a channel. Check predicates against actual contexts.
Runtime rows may include history_context with prior_observations and cold_start.
A cold-start row had no earlier observation: its output cannot diagnose lost
memory or failure to use prior evidence. The initial memory document is nonempty
even with zero observations. Missing history_context means unknown, not warm.
prior_observations is a COUNT, not prefix membership or a history slice boundary.
Same-day and same-description exclusions still apply when reconstructing exposure.
The registered memory_policy and memory_response_cache are host-owned settings.
With tag-evidence policies, the compiler maintains classification evidence rather than
a persona. Diagnostic label coverage does not prove semantic faithfulness.

## Language and metadata

This is a restricted Python subset, NOT arbitrary Python. See the generated
language lists. No imports, classes, comprehensions, lambdas, try/raise, while,
I/O, reflection or mutable module state. Use bounded for loops and append.
Functions and safe builtin names may ONLY appear as direct calls. In particular
isinstance is NOT supported; isinstance(value, dict/list/str) is invalid. Passing
a helper as sorted(key=helper) is also invalid. JSON methods must be called,
not stored as bound method values. Do not add annotations requiring builtin
type objects. The host already guarantees the documented context value types.

CHANGE_MANIFEST is a literal dictionary mapping ONLY remember, recall, apply,
guard to NONEMPTY STRINGS of at most 240 characters. Do not use apply:generation
as a manifest key, or a list/dict as its value. Substage names belong only to
the hook dispatch contract. An empty manifest is valid for an unchanged parent.

Legal shape (illustrates the API, not a required personalization intervention):

```python
PROGRAM_API_VERSION = 1
USER_MODEL = {}
CHANGE_MANIFEST = {"remember": "Retain uncertainty in observed tagging rules."}

def remember(context):
    return {"guidance": "Preserve uncertainty in conflicting observed choices."}

def recall(context):
    return {}

def apply(context):
    return {}

def guard(context):
    return {}
```

## Validation and feedback

Run the supplied check_harness.py with the exact Python command below. This is
the explicit exception to legacy skill instructions that allow ast.parse only.
The checker uses host validation, never models, prediction scoring or private
evaluation inputs. After source validation it runs hooks in the existing isolated
worker on the published anonymous empty/populated contexts, validating returned
field types and limits. It does not execute returned plans, query a database,
rewrite memory, access a model or score an answer. Correct contract errors within
this SAME generation session. Do not run the harness outside this checker.
The host repeats validation on the final candidate; skipping the checker does
not bypass it. No new scored candidate or hidden retry is authorized.
Passing checks establishes compatibility only, not personalization quality or
coverage of all input-dependent branches. Runtime rejection/effect diagnostics
are evidence, not a fitness score. Missing diagnostics mean unknown, not success.
Fixtures demonstrate a subset of possible inputs; optional backend states and
later workflow steps are not exhaustively covered. Execution references, not
anonymous fixture values, define the actual runtime context and defaults.
Keep framework facts here; use the learned skill for evidence-based decisions
about WHICH personal behavior to change. Do not copy framework reference text
or anonymous fixture contents into a user's preference policy.
""" + HISTORY_CONTEXT_GUIDANCE


def contract(settings=None):
    return {"protocol": PROTOCOL, "api_version": program.PROGRAM_API_VERSION,
        "language": program.language_subset(), "syntax_alternatives": dict(program.FORBIDDEN_NODE_ALTERNATIVES),
        "editable_fields": {stage: sorted(keys - FROZEN_FIELDS) for stage, keys in program.STAGE_DECISION_KEYS.items()},
        "frozen_fields": sorted(FROZEN_FIELDS), "registered_settings": dict(settings or {}),
        "history_context_guidance": HISTORY_CONTEXT_GUIDANCE,
        "memory_view": {"facts": "section -> list[str]", "records": "list[{id, section, text}]",
            "text_units": "nonempty section text lines and list items; list continuations retained for selection",
            "bounded": True},
        "frozen_field_literals_forbidden_in_source": True,
        "limits": {"source_chars": program.MAX_PROGRAM_CHARS, "ast_nodes": program.MAX_AST_NODES,
            "input_chars": program.MAX_INPUT_CHARS, "output_chars": program.MAX_OUTPUT_CHARS,
            "hook_seconds": program.DEFAULT_TIMEOUT_SECONDS, "hook_line_budget": program.DEFAULT_LINE_BUDGET,
            "memory_guidance_chars": program.MAX_MEMORY_GUIDANCE_CHARS,
            "manifest_description_chars": program.MAX_MANIFEST_ENTRY_CHARS},
        "change_manifest": {"keys": list(program.PROGRAM_HOOKS), "values": "nonempty string"}}


def validate_source(source):
    fingerprint = program.validate_program_source(source)
    if any(isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FROZEN_FIELDS
            for node in ast.walk(ast.parse(source))):
        raise ValueError("W0 skill candidates cannot override frozen sampling or hard call budgets")
    return fingerprint


def validate_diagnostics(records):
    fields = {"hook", "stage", "status", "effective", "error", "applied_keys"}
    if not isinstance(records, list) or len(records) > 256:
        raise ValueError("runtime hook diagnostics must be a bounded list")
    for record in records:
        if (not isinstance(record, dict) or set(record) != fields or record["hook"] not in program.PROGRAM_HOOKS
                or not isinstance(record["stage"], str) or record["status"] not in {"ok", "error", "unknown"}
                or type(record["effective"]) is not bool or not isinstance(record["error"], str)
                or not isinstance(record["applied_keys"], list)
                or any(not isinstance(key, str) for key in record["applied_keys"])):
            raise ValueError("runtime hook diagnostics cannot contain evaluation feedback or unsupported fields")
        if len(record["error"]) > 4000 or len(record["stage"]) > 80 or len(record["applied_keys"]) > 64:
            raise ValueError("runtime hook diagnostics exceed bounded field sizes")


def validate_history_context(context):
    if (not isinstance(context, dict) or set(context) != {"prior_observations", "cold_start"}
            or type(context["prior_observations"]) is not int or context["prior_observations"] < 0
            or type(context["cold_start"]) is not bool
            or context["cold_start"] != (context["prior_observations"] == 0)):
        raise ValueError("runtime history context must contain only consistent unscored exposure counts")


def check_program(source):
    fingerprint = validate_source(source)
    instance = program.HarnessProgram(source)
    checked = []
    for example in context_examples():
        context = example["context"]
        outcome = instance.call(example["hook"], context, contract_context=example.get("contract_context"))
        forbidden = FROZEN_FIELDS.intersection(outcome.patch)
        if forbidden:
            raise ValueError("returned patch changes frozen fields: " + ", ".join(sorted(forbidden)))
        checked.append({"hook": example["hook"], "stage": program.context_stage(example["hook"], context),
            "variant": example["variant"], "patch_keys": sorted(outcome.patch)})
    return {"valid": True, "protocol": PROTOCOL, "source_sha256": fingerprint.sha256,
        "checks": checked, "coverage": "anonymous structural fixtures only; not exhaustive or an accuracy test",
        "model_calls": 0, "scored_predictions": 0}


def context_examples():
    examples = []
    for populated in (False, True):
        facts = {"Notes": ["An anonymous observation"]} if populated else {}
        records = [{"id": "note-1", "text": "An anonymous observation", "section": "Notes"}] if populated else []
        common = {"task_kind": "lamp_native::movie_tagging", "query": "An anonymous story", "memory_facts": facts}
        bases = {
            "remember": {"logical_days": ["2000-01-01"], "channels": ["reaction"] if populated else [],
                "events": [{"index": 0, "channel": "reaction", "value": "An anonymous observation", "context": "", "day": "2000-01-01"}] if populated else [],
                "current_memory_sections": list(facts), "base": {"max_chars": None, "guidance": "", "keep_events": [0] if populated else []}},
            "recall:evidence": {**common, "stage": "evidence", "candidates": [],
                "base": {"mode": "recency", "top_k": 5, "budget_chars": 3000}},
            "recall:context": {**common, "stage": "context", "memory_records": records, "history_units": [], "history_block": "", "history_chars": 0,
                "base": {"memory_sections": list(facts), "include_history": False, "history_chars": 0,
                    "memory_policy": "sliding_window", "history_keep": 12, "history_unit_count": 0}},
            "apply:generation": {**common, "stage": "generation", "turn": 0, "tool_count": 0, "observation_count": 0,
                "blocks": [{"name": "query", "chars": 18}], "tools": [],
                "base": {"expressible": ["prefix", "suffix", "structured_instruction", "block_order", "drop_blocks", "budget"],
                    "prefix": "", "suffix": "", "structured_instruction": False, "budget": 3000,
                    "fixed_sampling": True, "workflow": "single_call", "parameter_domains": {}}},
            "apply:arguments": {**common, "stage": "arguments", "tool_name": "", "arguments": {}, "base": {"arguments": {}}},
            "apply:step": {**common, "stage": "step", "step": 0, "state": {}, "base": {"operations": list(program.PROGRAM_OPERATIONS)}},
            "guard:validate": {**common, "phase": "validate", "output": "anonymous output", "base": {"violations": []}},
            "guard:postprocess": {**common, "phase": "postprocess", "output": "anonymous output", "base": {"output": "anonymous output"}},
        }
        for name, context in bases.items():
            examples.append({"hook": name.split(":")[0], "variant": "populated" if populated else "empty",
                "context": copy.deepcopy(context)})
    return examples


CHECKER = '''import json
from pathlib import Path
import sys

sys.path.insert(0, {source!r})
from ohpu.per_user_adaptation.w0_framework import check_program

try:
    path = Path("harness.py")
    if path.is_symlink() or not path.is_file():
        raise ValueError("harness.py must be a regular file")
    result = check_program(path.read_text())
except Exception as error:
    print(json.dumps({{"valid": False, "error": type(error).__name__ + ": " + str(error)}}))
    raise SystemExit(1)
print(json.dumps(result))
'''


def write_framework(directory, settings=None):
    directory = Path(directory)
    command = shlex.join([sys.executable, "-I", "-B", "check_harness.py"])
    files = {"framework.md": GUIDE + "\nContract check: " + command + "\n",
        "framework.json": json.dumps(contract(settings), indent=2, sort_keys=True) + "\n",
        "context_examples.json": json.dumps(context_examples(), indent=2, sort_keys=True) + "\n",
        "framework_reference/validation.py": inspect.getsource(program),
        "check_harness.py": CHECKER.format(source=str(Path(__file__).resolve().parents[2]))}
    for name in ("executor", "daily_memory", "memory_backend", "structured_memory", "retrieval_policy"):
        files["framework_reference/" + name + ".py"] = Path(__file__).with_name(name + ".py").read_text()
    identities = {}
    for relative, text in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        path.chmod(0o444)
        identities[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identities