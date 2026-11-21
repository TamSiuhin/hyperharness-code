from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent

from ..bank.experience import ExperienceBank
from ..core.models import HarnessConfig, make_minimal_config
from ..program import (
    PROGRAM_HOOKS,
    FORBIDDEN_NODE_ALTERNATIVES,
    HarnessProgram,
    ProgramValidationError,
    language_subset,
)
from ..runtime.codex_bundle import (
    ensure_codex_bundle,
    load_codex_bundle,
    resolve_codex_bundle_paths,
    restore_codex_bundle,
    snapshot_codex_bundle,
    validate_codex_bundle,
)
from .local_cli import CodexController

_STAGES = PROGRAM_HOOKS
# A single-hook edit was the observed failure mode: five rounds of retuning
# prompt prose while the other stages stayed empty. Breadth is required unless
# the proposal declares a CHANGE_MANIFEST that says what it is doing.
_MIN_CHANGED_STAGES = 2

# The program's own size ceiling; the validator enforces the same number.
_PROGRAM_CHAR_BUDGET = 32000

logger = logging.getLogger(__name__)


class CodexBundleController(CodexController):
    """Direct-edit controller for a Harbor Codex harness bundle."""

    def __init__(
        self,
        *,
        bundle_root: Path | str,
        **kwargs,
    ) -> None:
        self.bundle_root = Path(bundle_root).expanduser().resolve()
        self.bundle_paths = resolve_codex_bundle_paths(self.bundle_root)
        self._protected_engine_snapshot: dict[Path, str | None] = {}
        self._protected_program_hooks: dict[str, str] = {}
        ensure_codex_bundle(self.bundle_root)
        super().__init__(harness_path=self.bundle_paths.program_path, **kwargs)

    def generate_initial_harness(self, dataset: str = "") -> tuple[str, HarnessConfig]:
        fallback = self.stabilize_config(make_minimal_config())
        ensure_codex_bundle(self.bundle_root)
        if not self._direct_edit_mode_enabled():
            raise RuntimeError(
                "Codex bundle controller requires direct-edit mode. "
                "Use bypass-permission Codex CLI args so the bundle files can be updated in place."
            )
        prompt = self._build_initial_bundle_prompt(dataset or self.dataset or "terminal-bench")
        return self._run_direct_edit_harness_with_retry(
            prompt,
            fallback=fallback,
            max_retries=3,
        )

    def decide_next_harness(
        self,
        bank: ExperienceBank,
        current_code: str,
        current_config: HarnessConfig,
        iteration: int,
        min_consecutive_failures: int = 3,
        parent_iteration: int | None = None,
    ) -> tuple[str, HarnessConfig]:
        base_config = self.stabilize_config(current_config)
        ensure_codex_bundle(self.bundle_root)
        if not self._direct_edit_mode_enabled():
            raise RuntimeError(
                "Codex bundle controller requires direct-edit mode. "
                "Use bypass-permission Codex CLI args so the bundle files can be updated in place."
            )
        prompt = self._build_iteration_bundle_prompt(
            bank=bank,
            current_code=current_code,
            current_config=base_config,
            iteration=iteration,
            min_consecutive_failures=min_consecutive_failures,
        )
        if parent_iteration is not None:
            prompt += (
                f"\n\n## Search lineage\nThe live parent is W{parent_iteration}. "
                f"The latest completed TRAIN feedback is W{iteration}; it may have "
                "been rejected. Propose a change to the live parent while learning "
                "from all completed train attempts. Do not confuse observation with deployment.\n"
            )
        return self._run_direct_edit_harness_with_retry(
            prompt,
            fallback=base_config,
            max_retries=3,
        )

    def should_normalize_harness(self, code: str) -> bool:
        return False

    def normalize_harness_code(self, code: str, config: HarnessConfig) -> str:
        return code

    def _summary_path(self) -> Path | None:
        # There is no config summary file: the base plan is an engine constant
        # and the program is the whole harness.
        return None

    def _project_brief_doc(self) -> str:
        return dedent(
            """
            ## Project Brief
            MemoHarness is optimizing a per-user code-represented harness bundle. The
            executor keeps model/tool I/O safe while the learned program controls the
            four personalization stages:
            - harness.py: THE harness. Four pure code hooks; the only file you edit.
            - memory.md: bounded online persona state visible at execution time; the
              host periodically rewrites it from current memory plus new cursor deltas
            - memory.state.json: host-only causal cursor; never task-model input
            The base plan each hook patches over is an engine constant, not a file;
            every hook sees it as context["base"].

            Durable behavioral personalization must be compiled into harness.py as
            constants, helper functions, conditional branches, and workflow decisions.
            memory.md carries bounded current facts, not a chronological log and not a
            separately searched policy module. Raw history is unavailable. A proposal
            whose executable per-hook fingerprint is unchanged is rejected.
            """
        ).strip()

    def _direct_edit_targets_doc(self) -> str:
        lines = [
            "## Editable Personalized Harness",
            "Modify exactly one runtime artifact:",
            f"- {self._display_path(self.bundle_paths.program_path)}",
            "",
            "The files below are engine-owned state, readable but never edited:",
            f"- {self._display_path(self.bundle_paths.memory_path)}",
            f"- {self._display_path(self.bundle_paths.memory_state_path)}",
            "Do not edit those files or any other repository file.",
            "Keep learned structural decisions in harness.py; leave episodic facts in the engine-owned memory.md.",
            "Never reconstruct or copy raw chronological history into harness.py.",
            "Do not create helper files: all program constants and helpers belong in harness.py.",
            "",
            self._program_contract_doc(),
            "",
            self._language_subset_doc(),
        ]
        return "\n".join(lines).strip()

    def _direct_edit_finish_doc(self) -> str:
        return dedent(
            """
            ## Required Finish State
            Run a publish gate before finishing. At minimum:
            - verify `harness.py` keeps PROGRAM_API_VERSION = 1 and defines remember,
              recall, apply and guard
            - verify every hook accepts one JSON object and returns a JSON object
            - verify at least two executable stage fingerprints changed, or that a
              CHANGE_MANIFEST names exactly the one hook you changed and why
            - verify CHANGE_MANIFEST, when present, names exactly the changed hooks
            - verify every new branch conditions on USER_MODEL or context["memory_facts"],
              not on task type alone
            - verify the program has no imports, file/network access, dunder access, or mutable I/O
            - verify memory.md and memory.state.json remain byte-identical
            - keep the program concise, specific, and grounded in current causal evidence

            Do not stop at analysis. If a check fails, repair the bundle files and rerun the checks.

            Print a short final status when you are done. Do not wrap your answer in <python> or
            <config> tags in this mode because the controller will reload the bundle from disk.
            """
        ).strip()

    @staticmethod
    def _language_subset_doc() -> str:
        """The exact Python subset, rendered from the validator's own constants.

        Measured cause: across one four-user PM3 run, three of five failed
        updates were rejected for constructs the contract never mentioned
        (`type()`, a generator expression, an unlisted memory section). Each
        one burned three Codex attempts and closed the window with no
        proposal, and they all landed on the same user, whose deployed harness
        therefore stopped evolving at W2.
        """

        subset = language_subset()
        forbidden = []
        for name in subset["forbidden_syntax"]:
            hint = FORBIDDEN_NODE_ALTERNATIVES.get(name)
            forbidden.append(f"{name}" + (f" -> {hint}" if hint else ""))
        return dedent(
            """
            ## The Python Subset harness.py May Use
            This is enforced by an AST validator BEFORE the bundle is published.
            A rejected proposal costs the whole window: there is no partial credit
            and the incumbent harness is restored. Read this before writing code.

            Callable builtins (nothing else, plus helpers this program defines):
            {builtins}

            Callable methods (nothing else):
            {methods}

            Rejected syntax, and what to write instead:
            {forbidden}

            Rejected names, in a call or anywhere else:
            {calls}

            set() supports &, |, -, ^, comparisons and membership, with deterministic
            iteration and list-style string representation. Elements must be hashable
            finite JSON values; at most 10000 distinct elements are allowed. Convert
            sets to lists before returning JSON decisions. Set literals remain rejected.
            The execution-step budget counts harness.py only, not host safety helpers.
            CPU, memory and wall-clock limits still apply. Tokenize once and avoid
            repeated scans of the full request for every memory record.

            The single most common mistake is a comprehension. `[x for x in items]`,
            `{{k: v for ...}}`, `{{x for ...}}` and `(x for ...)` are ALL rejected;
            build the value with a for loop and `.append()`. Also absent: `lambda`,
            `while`, `try`/`except`, `raise`, `with`, `class`, `import` and set
            literals. No name may start with an underscore. f-strings are fine.
            """
        ).strip().format(
            builtins=", ".join(subset["safe_builtins"]),
            methods=", ".join(subset["safe_methods"]),
            forbidden="\n".join(f"- {row}" for row in forbidden),
            calls=", ".join(subset["forbidden_calls"]),
        )

    @staticmethod
    def _program_contract_doc() -> str:
        return dedent(
            f"""
            ## harness.py Program Contract
            `harness.py` is ordinary conditional Python inside a restricted pure-function
            runtime. It may define literal USER_MODEL state and helper functions. It must
            define all FOUR stage hooks; an empty dict keeps the engine's base plan for
            that stage.

            The four hooks are the four questions a personalized harness answers. Each is
            called ONCE per boundary and its decision is then applied at every point that
            reads it, so one request produces one coherent plan per stage.

            - remember(context): WHAT TO KEEP about this user. Compiles the persistent
              memory document from the causal event stream. Return `keep_events` /
              `drop_channels` / `priority_channels` to choose which observed events reach
              the rewriter, `max_chars` to bound the document, and one short `guidance`
              clause.
              You do NOT choose the document's layout. The rewriter decides what sections
              it needs and may add, drop, rename or reorder them between rewrites; the
              engine checks only that the result is a persona and not a copy of the event
              stream. `guidance` is how you influence the shape -- say what matters for
              this user, not what headings to write. `context["current_memory_sections"]`
              tells you what the document happens to look like right now.
              `max_chars` is a preference, not a guarantee: a document that outgrows it is
              kept and the audit records the budget as not applied. The engine still owns
              cursor order and commit: a dropped event still advances state. When your
              remember plan changes, the engine replays the observed prefix from an empty
              document under your new code, so a filter or guidance change really does
              change what the next window reads.
            - recall(context): WHAT THIS REQUEST SEES of the user, by context["stage"].
              * "evidence": return `mode`, `top_k`, or `budget_chars`. The engine applies
                these only to an already-causal evidence pool. Modes are "none", "recency",
                "bm25" (Okapi BM25, k1=1.2, b=0.75), and "hybrid" (BM25 plus recency).
                "relevance" is a deprecated alias for "bm25", not another scorer.
                `context["available_items"]`
                -- the canonical top-level field -- counts that pool. `base.available_items`
                is a deprecated read-only alias for older API-v1 programs, not a policy
                knob. Both are zero where no pool exists, and then this stage cannot
                move anything. The base plan
                retrieves nothing (`retrieval_mode` "none", `top_k` 0), so a task whose
                referents live in retrieved history is unanswerable until this stage
                turns retrieval on.
                Completed-case feedback reports the selected mode, limits, eligible
                count and retrieved count without revealing targets. A
                `retrieval_disabled` or `retrieval_budget_empty` failure belongs to
                recall:evidence even if recall:context successfully selected memory.
              * "context": return `memory_fact_ids` using IDs in the current
                `context["memory_records"]` to select individual evidence bullets,
                even when one section mixes topics. Records contain bounded
                id/section/text fields; do not copy IDs into constants. An empty
                or unknown ID selection falls back to unchanged memory. Choose
                either fact IDs or `memory_sections`, never both.
                `memory_mode` is base, select, or no_user_memory. Base and
                no_user_memory cannot include selectors; select needs exactly
                one. no_user_memory suppresses memory for this request, without
                deleting persistent memory or the current request. Host scoped
                experiments can restrict these actions. The shared-retrieval
                profile accepts only `retrieval_delta`, a host-validated vector
                of bounded numeric parameters; its five keys and limits are
                provided in that experiment's contract.
                Return `memory_sections` to keep a subset of the compact
                memory's sections -- name them from `context["memory_facts"]`, whose keys
                are whatever headings the document currently has -- plus
                `include_history` / `max_history_chars` to shape the
                selected evidence block, and `keep_history_units` (indices from
                context["history_units"]) to drop prior turns. The current turn is always
                retained. Input history is rendered by the causal engine and cannot be
                replaced with generated free-form text.
            - apply(context): HOW TO ANSWER for this user, by context["stage"].
              * "generation": one plan for the whole action. Context assembly:
                `block_order`, `drop_blocks`, `budget`, `prefix`, `suffix`,
                `structured_instruction`. `budget` packs the compatible prompt blocks;
                prefix/suffix are separately capped additions, so keep them short. Action
                space: `allow`, `deny`, `order`, using only names in context["tools"];
                tool names and schemas cannot be invented or rewritten. Decoding:
                `temperature`, `top_p`, `max_tokens`, `reasoning_effort`
                (none|low|medium|high). Workflow: `workflow`, `max_turns`, `stop_rule`,
                `max_tool_calls`, `plan_prompt`, `refine_prompt`. `max_tool_calls` is the
                typed form of "do not fire a batch of calls for this user" -- write it as
                a decision rather than as a sentence in a prefix.
                `base.expressible` lists exactly which of these fields THIS boundary can
                honour: without tools the action space cannot move and reasoning_effort is
                forced to "none"; a native evaluator owns loop length, so max_turns and
                stop_rule are ignored and audited there. Writing a field outside
                `base.expressible` produces a dead decision, not an error.
              * "arguments": return `defaults`, `overrides`, or `drop_arguments` to rewrite
                one tool call. Only argument names declared by the official schema apply.
                            * "step": used only with workflow "program". Return `operation` to request
                                analyze, select_memory, generate, or finish; `prompt` controls internal
                                analysis/generation, `selected_ids` selects authorized memory records,
                                `final_output` optionally supplies the finished answer, and `state` carries
                                bounded request-local data between steps. The host enforces operation limits.
                        Structured remember extensions: return `state_schema` to constrain host model
                        extraction, or `state_records` for a code-defined update. State records require
                        valid causal source references. The open-program profile documents their format.
                        Opt-in programmable memory: `remember` returns `memory_backend` operations
                        for transactional writes; `recall` at context returns `memory_query`
                        operations for read-only queries. The programmable-memory profile documents
                        collections, schemas, SQL, vector/text search, model calls and finish.
                        Each operation is executed by the host; no direct I/O is granted to hooks.
            - guard(context): WHEN NOT TO PERSONALIZE and how to check the answer, by
              context["phase"]. In "validate" return `violations`; in "postprocess" return
              `output`. This is the cheapest place to make a negative preference
              operational: turn "this user rejects hype" into a validator that fires, not
              into another sentence in the prompt.

            Hooks receive task_kind, the current public query/state, and a `base` plan.
            recall, apply and guard also receive `memory_facts`: the engine-owned compact
            memory parsed into bounded per-section bullets. That is the channel for
            conditioning code on what has been learned about the user.

            `memory_facts` is always the WHOLE compact memory, and `recall.memory_sections`
            only chooses what reaches the prompt. So a branch can act on a fact without
            spending prompt budget on it, and dropping a section from the prompt is a
            budget decision, not a claim that the fact stopped being true. guard in
            particular keeps seeing a dislike it must enforce even when recall left that
            section out of the context.

            A branch that reads neither `memory_facts` nor the request returns the same
            decision for every request. The engine reports that as `constant` rather than
            `realized`: it is per-user parameterization, which is real but weak. A stage
            whose decision varies across the window is conditioned on the request, which
            is what this search is for.

            Hooks never receive raw history; the engine may expose a bounded selected
            history block and a compact memory block, but hook code cannot replace either
            with an unbounded event log. Hooks also never receive reward, expected answer,
            split/condition name, evaluator backend, or another user's state. Prefer
            branches whose activation can be explained by observed per-user evidence. Do
            not use imports/classes/decorators, open/eval/exec/print, private attributes,
            while loops, exceptions, or global mutation.
            """
        ).strip()

    @staticmethod
    def _parse_marker(marker: str, prefix: str) -> str | None:
        return marker[len(prefix):] if marker.startswith(prefix) else None

    def _stage_coverage(self, bank: ExperienceBank) -> dict[str, dict[str, int]]:
        """Aggregate per-stage liveness across every recorded case."""

        coverage: dict[str, dict[str, int]] = {
            stage: {
                "invoked": 0, "live": 0, "changed": 0, "effective": 0,
                "cases_live": 0, "cases_effective": 0,
            }
            for stage in _STAGES
        }
        for entry in bank.entries:
            for marker in entry.trajectory.intermediate_outputs or []:
                body = self._parse_marker(str(marker), "liveness:program:")
                if body is None:
                    continue
                pieces = body.split(":")
                stage = pieces[0].strip().lower()
                if stage not in coverage:
                    continue
                fields = {
                    part.split("=", 1)[0]: part.split("=", 1)[1]
                    for part in pieces[1:]
                    if "=" in part
                }
                bucket = coverage[stage]
                for key in ("invoked", "live", "changed", "effective"):
                    try:
                        bucket[key] += int(fields.get(key, 0) or 0)
                    except ValueError:
                        continue
                if int(fields.get("live", 0) or 0) > 0:
                    bucket["cases_live"] += 1
                if int(fields.get("effective", 0) or 0) > 0:
                    bucket["cases_effective"] += 1
        return coverage

    def _latest_stage_verdicts(
        self, bank: ExperienceBank
    ) -> dict[str, str]:
        """The engine's per-stage ruling on the most recent code edit."""

        latest = max((entry.iteration for entry in bank.entries), default=None)
        if latest is None:
            return {}
        verdicts: dict[str, str] = {}
        for entry in bank.entries:
            if entry.iteration != latest:
                continue
            for marker in entry.trajectory.intermediate_outputs or []:
                body = self._parse_marker(str(marker), "verdict:program:")
                if body is None or "=" not in body:
                    continue
                stage, verdict = body.split("=", 1)
                stage = stage.strip().lower()
                verdicts[stage] = verdict
        return verdicts

    def _build_stage_coverage_doc(
        self, bank: ExperienceBank, iteration: int
    ) -> str:
        """The liveness table plus this round's required stages.

        Without this the proposer can only see which stages it already writes
        to, which is why successive rounds kept re-tuning the same one or two
        hooks. ``live`` is measured at the boundary that actually runs this
        bundle, so a stage reported dead here cannot be fixed by writing more
        code for it.
        """

        coverage = self._stage_coverage(bank)
        if not any(values["invoked"] for values in coverage.values()):
            return (
                "## Stage Coverage\n"
                "No per-stage runtime evidence recorded yet. Treat every stage "
                "as unproven and prefer a proposal that exercises several of "
                "them."
            )
        lines = [
            "## Stage Coverage",
            "Measured on the boundary that runs this bundle. `live` counts hook",
            "invocations that COULD have changed execution; `effective` counts",
            "the ones that actually did.",
            "",
            "| stage | invoked | live | returned a patch | effective | cases effective/live |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for stage in _STAGES:
            values = coverage[stage]
            lines.append(
                f"| {stage} | {values['invoked']} | {values['live']} | "
                f"{values['changed']} | {values['effective']} | "
                f"{values['cases_effective']}/{values['cases_live']} |"
            )
        dead = [
            stage for stage in _STAGES
            if coverage[stage]["invoked"] and not coverage[stage]["live"]
        ]
        debt = sorted(
            (
                stage for stage in _STAGES
                if coverage[stage]["live"] > 0
            ),
            key=lambda stage: (
                coverage[stage]["cases_effective"]
                / max(1, coverage[stage]["cases_live"]),
                stage,
            ),
        )
        inert = [
            stage for stage in _STAGES
            if coverage[stage]["changed"] > 0
            and coverage[stage]["effective"] == 0
        ]
        verdicts = self._latest_stage_verdicts(bank)
        lines.append("")
        if verdicts:
            lines.append(
                "Verdict on your last edit, measured on the window that "
                "followed it: "
                + ", ".join(
                    f"{stage}={verdict}"
                    for stage, verdict in sorted(verdicts.items())
                )
                + "."
            )
            constant = sorted(
                stage for stage, verdict in verdicts.items()
                if verdict == "constant"
            )
            if constant:
                lines.append(
                    "Stages marked `constant` DID change execution, but gave "
                    "the same decision for every case in that window: "
                    + ", ".join(constant)
                    + ". That is a per-user constant, not a per-request "
                    "decision. Condition those branches on `memory_facts` or "
                    "on the request itself."
                )
            regressed = sorted(
                stage for stage, verdict in verdicts.items()
                if verdict not in {"realized", "constant"}
            )
            if regressed:
                lines.append(
                    "You edited " + ", ".join(regressed) + " and the next "
                    "window realized no new decision there. Either make the "
                    "branch reachable on real inputs, or delete it and spend "
                    "the round elsewhere. Do not re-tune the same knob."
                )
        if dead:
            lines.append(
                "Dead at this boundary, do NOT spend a proposal there: "
                + ", ".join(dead)
                + "."
            )
        if inert:
            lines.append(
                "Returned a decision that changed nothing -- either a dead "
                "branch or a value equal to the base plan. Fix or delete: "
                + ", ".join(inert)
                + "."
            )
        if debt:
            lines.append(
                "Coverage debt, lowest first: " + ", ".join(debt) + "."
            )
            # A stage that is already effective on every case it was live
            # for has no debt to pay; naming it here would let a round satisfy
            # the requirement by re-tuning the hook it already uses.
            required = [
                stage for stage in debt
                if coverage[stage]["cases_effective"]
                < coverage[stage]["cases_live"]
            ][:2]
            if required:
                lines.append(
                    "REQUIRED this round: justify one targeted live-stage change in "
                    "CHANGE_MANIFEST. Investigate " + " or ".join(required)
                    + " if supported by the observed errors; respect frozen hooks."
                )
            else:
                lines.append(
                    "REQUIRED this round: justify the targeted change in CHANGE_MANIFEST. "
                    "Every live stage is already realizing a decision, so "
                    "improve the decisions rather than reaching for new hooks."
                )
        return "\n".join(lines)

    @staticmethod
    def _build_change_manifest_doc() -> str:
        return dedent(
            """
            ## Required CHANGE_MANIFEST
            Add or update a top-level `CHANGE_MANIFEST` literal in harness.py naming
            every stage this revision changes and the realized effect you expect at
            the next window:

                CHANGE_MANIFEST = {
                    "apply": "style-constrained asks run refine instead of single_call",
                    "guard": "reject hype adjectives this user has rejected before",
                }

            Keys are stage names: remember, recall, apply, guard. It is not a hook
            dependency, so declaring intent is not itself a code change, and the
            engine checks each claim against what the next window actually realizes.
            Declaring a hook you did not change is rejected.

            Per-hook fingerprints are dependency-aware: editing a shared constant
            such as USER_MODEL changes the fingerprint of EVERY hook that reads it.
            So the engine's changed set is often larger than your manifest. That is
            fine and is not an error -- declare the stages you deliberately changed,
            and expect the coverage report to list the rest.
            """
        ).strip()

    def _build_workspace_context(self, iteration: int) -> str:
        del iteration
        lines = [
            f"- workspace_root: {self._display_path(self.workspace_root)}",
            f"- dataset: {self.dataset or '(unspecified)'}",
            f"- model_label: {self.model}",
            f"- codex_bundle_root: {self._display_path(self.bundle_root)}",
            f"- online_memory_file: {self._display_path(self.bundle_paths.memory_path)}",
            f"- causal_cursor_file: {self._display_path(self.bundle_paths.memory_state_path)}",
            f"- harness_program_file: {self._display_path(self.bundle_paths.program_path)}",
        ]
        return "\n".join(lines)

    def _snapshot_live_harness_files(self) -> dict[Path, str | None]:
        snapshot = snapshot_codex_bundle(self.bundle_root)
        self._protected_engine_snapshot = {
            path: content
            for path, content in snapshot.items()
            if path != self.bundle_paths.program_path
        }
        previous_source = snapshot.get(self.bundle_paths.program_path)
        self._protected_program_hooks = (
            dict(HarnessProgram(previous_source).fingerprint.hook_sha256)
            if previous_source is not None
            else {}
        )
        return snapshot

    def _restore_live_harness_snapshot(self, snapshot: dict[Path, str | None]) -> None:
        restore_codex_bundle(snapshot)

    def _load_live_harness_state(self, fallback: HarnessConfig) -> tuple[str, HarnessConfig]:
        for path, expected in self._protected_engine_snapshot.items():
            try:
                current = path.read_text(encoding="utf-8")
            except OSError:
                current = None
            if current != expected:
                raise ProgramValidationError(
                    f"only harness.py is editable; protected bundle file changed: "
                    f"{path.relative_to(self.bundle_root)}"
                )
        program = validate_codex_bundle(
            self.bundle_root, require_program=True, smoke=True
        )
        if program is None:
            raise ProgramValidationError(
                "proposal must change executable stage program structure; "
                "text-only or knob-only edits are not a personalized code harness"
            )
        changed = {
            stage
            for stage in set(self._protected_program_hooks)
            | set(program.fingerprint.hook_sha256)
            if self._protected_program_hooks.get(stage)
            != program.fingerprint.hook_sha256.get(stage)
        }
        if not changed:
            raise ProgramValidationError(
                "proposal must change executable stage program structure; "
                "text-only or knob-only edits are not a personalized code harness"
            )
        manifest = dict(program.fingerprint.change_manifest)
        if manifest:
            # A false claim is what would make the next window's per-stage
            # verdict meaningless, so declaring a hook you did not touch is
            # rejected. Failing to declare one is not rejected: per-hook
            # hashes are dependency-aware, so editing a shared USER_MODEL key
            # changes the hash of every hook that reads it, and no author can
            # predict that set. The engine reports the full changed set anyway.
            unchanged = sorted(set(manifest) - changed)
            if unchanged:
                raise ProgramValidationError(
                    "CHANGE_MANIFEST names hooks this revision did not change: "
                    f"{unchanged}; declared={sorted(manifest)} "
                    f"changed={sorted(changed)}"
                )
        elif self._protected_program_hooks and len(changed) < _MIN_CHANGED_STAGES:
            # The first bundle has no incumbent to compare against, so the
            # breadth requirement starts at the first real update.
            raise ProgramValidationError(
                "a personalized harness update must change at least "
                f"{_MIN_CHANGED_STAGES} stages, or declare a "
                "CHANGE_MANIFEST explaining why one is enough; "
                f"changed={sorted(changed)}"
            )
        preview, config = load_codex_bundle(self.bundle_root)
        return preview, self.stabilize_config(config)

    def _bundle_inspection_doc(self, iteration: int) -> str:
        del iteration
        return dedent(
            f"""
            ## Inspect First
            Start with the smallest local file set:
            - the current Codex bundle files in the target section below
            - `src/memoharness/runtime/codex_bundle.py` only when you need to confirm bundle schema/sync behavior
            - expand to `src/memoharness/harbor/loop.py` only when a concrete runtime-wiring ambiguity remains

            Do not perform broad repository inventory reads unless one specific unresolved question
            requires it.
            """
        ).strip()

    def _build_initial_bundle_prompt(self, dataset: str) -> str:
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor task execution happens on Daytona cloud machines, but the Codex harness bundle
            lives in this local workspace and will be synced into the sandbox before Harbor's
            official Codex agent runs. Update the bundle files directly.

            {self._project_brief_doc()}

            ## Local Context
            {self._build_workspace_context(iteration=0)}

            {self._direct_edit_job_doc(phase='initial Codex bundle generation')}

            {self._direct_edit_targets_doc()}

            {self._bundle_inspection_doc(iteration=0)}

            {self._controller_command_scope_doc()}

            ## Objective
            Create the initial publishable Codex harness bundle for dataset "{dataset}".
            Favor focused bundle/runtime inspection, verifier-aware execution rules,
            deterministic local checks, and concise memory scaffolding.

            {self._direct_edit_finish_doc()}
            """
        ).strip()

    def _build_iteration_bundle_prompt(
        self,
        *,
        bank: ExperienceBank,
        current_code: str,
        current_config: HarnessConfig,
        iteration: int,
        min_consecutive_failures: int,
    ) -> str:
        bank_summary = self._build_iteration_bank_summary(bank, iteration)
        recent_failures = self._build_recent_failure_excerpt(bank, iteration)
        signal_summary = self._build_iteration_signal_summary(bank, iteration)
        task_history = self._build_task_history_matrix(bank, iteration)
        coverage_summary = self._build_stage_coverage_doc(bank, iteration)
        program_budget = _PROGRAM_CHAR_BUDGET
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor will still run the official Codex agent. Your job is to rewrite the local
            Codex harness bundle that Harbor syncs into the sandbox before execution.

            {self._project_brief_doc()}

            ## Local Context
            {self._build_workspace_context(iteration=iteration)}

            {self._direct_edit_job_doc(phase='iteration bundle repair and improvement')}

            {self._direct_edit_targets_doc()}

            {self._bundle_inspection_doc(iteration=iteration)}

            {self._controller_command_scope_doc()}

            ## Current Iteration
            iteration={iteration}
            dataset={self.dataset or '(unspecified)'}

            ## Current Config Summary
            {json.dumps(current_config.as_dict(), indent=2)}

            ## Per-Task History Across Iterations
            {task_history}

            ## Current Iteration Experience Summary
            {bank_summary}

            ## Recent Failure Excerpts
            {recent_failures}

            ## Change-Outcome Signals
            {signal_summary}

            {coverage_summary}

            {self._build_change_manifest_doc()}

            {self._targeted_supporting_evidence_doc()}

            {self._on_disk_state_doc()}

            ## Hard Constraints On What You May Write
            These come from measured outcomes across the iterations above, not style preference.

            1. No self-certifying markers. Never instruct the agent to print or emit a completion
               token (`..._OK`, `GATE_PASSED`, and similar). Nothing in the harness reads them.
               Every failed run recorded so far still declared itself finished, so a claim is not
               evidence. Ask for a concrete check whose output the verifier would accept instead.

            2. Stay inside the budget: harness.py is the whole harness and has a
               {program_budget}-character limit. Use that room for executable decision structure,
               not for prose copied from memory and never for a reconstructed chronological
               history. A long prose `prefix` is the cheapest thing to write and the least
               personalized thing you can ship.

            3. Prefer one targeted stage change and declare its CHANGE_MANIFEST, with a
               hypothesis grounded in completed train feedback. Change multiple hooks only
               when the mechanism requires it; more changed stages is not better. Read the
               Stage Coverage table above: it names which stages are dead at this
               boundary (never write code for those), which returned a decision that changed
               nothing (fix or delete those), and which are live but unused (that is where the
               remaining headroom is).

            4. Every branch must be user-conditioned. A branch on task type alone is a generic
               heuristic, not personalization. Condition on USER_MODEL constants you compiled
               from this user's evidence, or on the `memory_facts` view the engine passes to
               recall, apply and guard. A branch that reads neither `memory_facts` nor the
               request gives every request the same decision; the engine calls that
               `constant`, not `realized`. Prefer deleting a branch whose activation you
               cannot explain from observed per-user evidence.

            Rewrite harness.py in place. Start from the current compact memory and completed-
            window summaries below. Keep it grounded, concise, and centered on executable
            behavior. Before editing, pick the stages this round will change, write the
            CHANGE_MANIFEST first, and map each entry to the realized decision you expect the
            next window's audit to show.

            {self._direct_edit_finish_doc()}
            """
        ).strip()
