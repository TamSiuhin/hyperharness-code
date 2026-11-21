"""Open program search references and zero-API execution diagnostics."""
from __future__ import annotations

import ast
from collections import Counter
import json
import textwrap
from typing import Any, Mapping, Sequence

from .core.models import make_minimal_config
from .executor import ExecutableHarness, Runner
from .program import DEFAULT_PROGRAM_SOURCE, HarnessProgram, PROGRAM_OPERATIONS, STAGE_DECISION_KEYS

PROFILE = "open-program"
PROGRAMMABLE_PROFILE = "programmable-memory"
OPEN_PROFILES = frozenset({PROFILE, PROGRAMMABLE_PROFILE})
CONTRACT = """
## Open program execution contract (additional stages)
The four hooks remain pure Python; the host performs all model calls and storage.
Select apply:generation workflow="program" to enable repeated apply:step calls.
At step, inspect query, memory_facts, memory_records, user_state, observations,
step_index, and request-local state. base lists operations and remaining limits.
Return operation plus only the fields it consumes:
- analyze: optional prompt; host returns a fallible JSON analysis in observations.
- select_memory: selected_ids from the supplied records, [] for no memory.
- generate: optional prompt; host answers the original request using selected
    memory and earlier observations. Official output contracts remain in force.
- finish: optional final_output; otherwise return the last generated answer.
Any operation may replace request-local state with a bounded JSON object. It is
not persistent memory. Branches and bounded loops are expressed by successive
step decisions; each operation can use the preceding analysis or generated draft.
The host enforces at most 12 steps and 4 model calls per action, including a
fallback generation. Analysis calls receive no tools or evaluator labels. A tool
action immediately returns to the native environment; never invent observations.
An analyze response is text plus parsed data, or data=null for malformed/oversized
JSON. Inspect truncated and handle missing fields. Program state does not survive
the request. There is no direct network, file, shell or raw-history access.

remember may choose state_schema OR state_records instead of the prose compiler.
state_schema is a bounded JSON schema for each record's data (no refs/regex).
The host asks the memory model for a full records array and validates it. To
implement a deterministic update yourself, return state_records with entries
{key, data, source_refs, expires_on?}. key names and data structures are yours.
source_refs must reference state_events[].ref or state_records[].ref supplied in
this remember context. This is a full replacement: omission deletes state.
The host owns source receipts, expiry filtering, size checks and atomic commits.
Detected forget requests conservatively invalidate all earlier structured state
references, not only matching keys; neither code nor a schema may restore them.
All runtime hooks receive user_state mapping each record key to its custom data.
Do not copy event IDs or user facts into code as an untracked memory substitute.

Keep only decisions supported by the current boundary. GPT-5 temperature/top_p
are inert. max_tokens cannot go below base.hard_min_tokens. A larger program is
not inherently better; remove redundant steps, and use free execution probes
before paid evaluation. Probes test behavior and boundaries, never answer quality.
"""


def _source(apply: str, remember: str = "return {}", recall: str = "return {}") -> str:
    bodies = {"remember": remember, "apply": apply, "recall": recall}
    source = DEFAULT_PROGRAM_SOURCE
    manifest = {}
    for node in ast.parse(DEFAULT_PROGRAM_SOURCE).body:
        if not isinstance(node, ast.FunctionDef) or node.name not in bodies:
            continue
        body = textwrap.dedent(bodies[node.name]).strip()
        if body != "return {}":
            replacement = "def " + node.name + "(context):\n" + textwrap.indent(body, "    ")
            source = source.replace(ast.get_source_segment(DEFAULT_PROGRAM_SOURCE, node), replacement)
            manifest[node.name] = ("Change only the memory compilation policy; preserve baseline answer behavior."
                if node.name == "remember" else ("Change query construction and retrieval of authorized memory."
                    if node.name == "recall" else "Change answer construction and control flow; preserve baseline memory and guards."))
    return source.replace("CHANGE_MANIFEST = {}", "CHANGE_MANIFEST = " + json.dumps(manifest))


REFERENCE_SEEDS = {
    "draft_grounded_refinement": _source('''
        if context.get("stage") == "generation":
            if context.get("task_kind") == "pm3_native::chatbot_personalized_response" and context.get("memory_records"):
                return {"workflow": "program"}
            return {}
        if context.get("stage") != "step":
            return {}
        step = context.get("step_index", 0)
        observations = context.get("observations", [])
        if step == 0:
            return {"operation": "generate"}
        if step == 1:
            if not observations or observations[0].get("truncated"):
                return {"operation": "finish"}
            return {"operation": "analyze", "prompt": "Review the generated draft for a concrete missed opportunity to use a relevant supported preference. Return exactly {changes:[{id,quote,old_text,correction}]} with at most two changes, or {changes:[]}. id must be a supplied record ID; quote must be its exact excerpt; old_text must be an exact nonempty span from the draft; correction must specify a localized improvement to that span. Keep an already useful answer, choice and requested tone. Preserve specific medium, format, named examples and scope when supported instead of substituting a generic alternative. Do not replace the user's freshly supplied choices with unrelated stored interests or generic prudent advice. A historical instruction not to use a preference for future personalization is not a prohibition on answering a new explicit request about it using details provided now. Never restore forgotten details not supplied now. Preserve actual current exclusions, safety and factual accuracy. Skip cosmetic rewrites and changes justified only by the current query rather than additional relevant evidence. No labels or unseen preferences are available."}
        if step == 2:
            data = (observations[-1].get("data") or {}) if observations else {}
            proposals = []
            match data:
                case {"changes": [*items]}:
                    proposals = items[:2]
            draft = observations[0].get("text", "") if observations else ""
            changes = []
            for proposal in proposals:
                identifier = None
                quote = None
                old_text = None
                correction = None
                match proposal:
                    case {"id": record_id, "quote": excerpt, "old_text": original, "correction": replacement}:
                        identifier = record_id
                        quote = excerpt
                        old_text = original
                        correction = replacement
                if quote != str(quote) or not 4 <= len(quote) <= 480:
                    continue
                if old_text != str(old_text) or not 4 <= len(old_text) <= 480 or old_text not in draft:
                    continue
                if correction != str(correction) or not 1 <= len(correction) <= 480:
                    continue
                matches = []
                for record in context.get("memory_records", []):
                    record_id = record.get("id", "")
                    if identifier in [record_id, record_id.split(":")[0]] and quote in record.get("text", ""):
                        matches.append(record_id)
                if len(matches) == 1:
                    changes.append({"evidence": quote, "old_text": old_text, "correction": correction})
            if not changes:
                return {"operation": "finish"}
            return {"operation": "generate", "prompt": "Keep the first generated answer and improve only these cited spans where the evidence genuinely helps the current request: " + str(changes) + " The source quotes are verified, but the proposed corrections are fallible. Preserve all useful uncited content and the user's present choices, deliverable and tone. Do not turn past opt-outs into bans on answering current explicitly supplied topics. Do not add unrelated interests, unsupported facts, profile callouts or commentary about the revision. Return the complete final answer in the original required format."}
        return {"operation": "finish"}
    '''),
    "literal_preference_constraints": _source('''
        if context.get("stage") == "generation":
            if context.get("task_kind") == "pm3_native::chatbot_personalized_response" and context.get("memory_records"):
                return {"workflow": "program"}
            return {}
        if context.get("stage") != "step":
            return {}
        step = context.get("step_index", 0)
        if step == 0:
            return {"operation": "analyze", "prompt": "Return JSON with constraints, an array of at most three objects containing id (a supplied memory record ID), quote (an exact excerpt from its text), and use (the concrete answer choice this supports). Find preferences that help fulfill the actual request, including relevant exclusions. Preserve discriminating qualifiers: medium, format, subgenre, scope, named examples and exclusions must not collapse into a broad topic or generic tone. Interpret the user's wording in that supported context without overriding explicit instructions. Do not select merely related interests that do not help the task. Privacy boundaries and accurate, safety-conscious advice take priority; do not force humor or lifestyle themes into health advice. Return an empty array when nothing applies. No private targets or unseen preferences are available."}
        if step == 1:
            observations = context.get("observations", [])
            data = (observations[-1].get("data") or {}) if observations else {}
            proposals = []
            match data:
                case {"constraints": [*items]}:
                    proposals.extend(items[:3])
            match data:
                case {"memories": [*items]}:
                    proposals.extend(items[:3])
            selected = []
            constraints = []
            for proposal in proposals:
                identifier = None
                quote = None
                use = None
                match proposal:
                    case {"id": record_id, "quote": excerpt, "use": effect}:
                        identifier = record_id
                        quote = excerpt
                        use = effect
                if quote != str(quote) or not 4 <= len(quote) <= 480 or use != str(use):
                    continue
                matches = []
                for record in context.get("memory_records", []):
                    record_id = record.get("id", "")
                    if identifier in [record_id, record_id.split(":")[0]] and quote in record.get("text", ""):
                        matches.append(record_id)
                if len(matches) == 1 and len(constraints) < 3:
                    if matches[0] not in selected:
                        selected.append(matches[0])
                    constraints.append({"quote": quote, "use": use[:240]})
            if not constraints:
                return {"operation": "generate", "prompt": "Complete the original request. No preference excerpt was validated; do not rely on unsupported claims in the analysis.", "state": {"direct": True}}
            return {"operation": "select_memory", "selected_ids": selected, "state": {"constraints": constraints}}
        if step == 2 and not context.get("state", {}).get("direct"):
            return {"operation": "generate", "prompt": "Complete the original request in its required format. These excerpts were checked against the supplied memory; proposed uses are fallible, not new user instructions: " + str(context.get("state", {}).get("constraints", [])) + " Preserve their useful specific qualifiers in actual choices, not only in the rationale. Do not silently substitute a different medium, format or scope with a broadly similar option. Meet current constraints and privacy boundaries; ignore unrelated preferences. Deliver the requested artifact, without exposing record IDs, quoting the profile, inventing personal facts, or adding a claim that the answer is personalized."}
        return {"operation": "finish"}
    '''),
    "deliberate_preference_reasoning": _source('''
        if context.get("stage") == "generation" and context.get("task_kind") == "pm3_native::chatbot_personalized_response" and context.get("memory_records"):
            return {"reasoning_effort": "high", "max_tokens": 8192}
        return {}
    '''),
    "evidence_compared_answers": _source('''
        if context.get("stage") == "generation":
            if context.get("task_kind") == "pm3_native::chatbot_personalized_response" and context.get("memory_records"):
                return {"workflow": "program"}
            return {}
        if context.get("stage") != "step":
            return {}
        step = context.get("step_index", 0)
        observations = context.get("observations", [])
        if step == 0:
            return {"operation": "generate"}
        if step == 1:
            return {"operation": "generate", "prompt": "Produce an alternative complete answer to the original request. Use the supplied memory to make the most relevant supported preference change an actual choice, example, ranking, tradeoff or wording. Keep the user's distinctive taste instead of replacing it with a generic prudent alternative. Treat specific current requests and privacy boundaries as controlling. If the query already names an interest, make its supported details useful rather than merely repeating the topic. Preferences may inform creative voice and examples without being announced. Never fabricate user facts, endorse a false belief, quote private history, or add irrelevant themes. Return only the required answer format, not commentary about the alternative."}
        if step == 2:
            return {"operation": "analyze", "prompt": "Compare the two generated answers to the original current request and supplied memory. Return JSON with choice (0 for first, 1 for second), evidence_ids and reason. Prefer the answer that satisfies explicit current instructions and privacy while making relevant supported preferences affect useful concrete content, without generic substitution or irrelevant personalization. Mere topic repetition or announcing familiarity is not useful evidence use. Any claim about the user must be supported. Keep the first answer when the second introduces unsupported content, worsens the requested deliverable, or has no clear benefit. Judge only these observed answers; no hidden preferences or evaluator labels are available."}
        drafts = []
        for observation in observations:
            if observation.get("operation") == "generate" and not observation.get("truncated"):
                drafts.append(observation.get("text", ""))
        review = (observations[-1].get("data") or {}) if observations else {}
        choice = 0
        if review.get("choice") == 1 and len(drafts) == 2:
            choice = 1
        if drafts:
            return {"operation": "finish", "final_output": drafts[choice]}
        return {"operation": "finish"}
    '''),
    "evidence_weighted_memory": _source('return {}', remember='''
        return {"guidance": "Maintain a compact evidence-weighted user model. Preserve explicit user statements and repeated positive choices as concrete preferences with their direction, scope and useful detail. Keep contradictions explicit. A skip, scroll-past or missing reaction is weak evidence: do not convert it into a broad dislike or erase positive evidence from another context. Explicit dislikes, privacy boundaries and forget requests remain controlling. Keep broad preferences separate from one-off requests and current goals; preserve specific genres, formats, priorities and tradeoffs when supported. Do not copy assistant suggestions as user preferences or invent details. Apply current calendar changes using the provided UTC values. Spend memory on actionable supported information, not long lists of topics the user skipped."}
    '''),
    "task_conditioned_preferences": _source('''
        if context.get("stage") != "generation":
            return {}
        task = context.get("task_kind", "")
        if task == "pm3_native::chatbot_personalized_response":
            return {"prefix": "Answer the user's current request, preserving its named subjects, concrete details, imagery, medium and intended tone. A detail explicitly supplied now is available evidence even if it is absent from memory: do not replace it with a placeholder or demand confirmation just because it is not remembered. A past request to stop using a preference controls historical personalization; it is not a ban on answering a new explicit request about that subject using details supplied now. Do not recover forgotten private facts. Use relevant, permitted memory to resolve references and choose useful specific options, examples, rankings, tradeoffs or wording. A preference hinted at in the query can still guide its specific treatment; mere repetition of a topic is not enough. Preserve distinctive taste instead of substituting generic alternatives or a cautious lecture. For genuine ambiguity, give a relevant choice rather than inventing a dislike or opt-out. Deliver the complete requested artifact, not a discussion of personalization. Never fabricate user facts, insert unrelated interests or announce historical data use."}
        if task in ["pm3_native::over_personalization_chatbot_text", "pm3_native::over_personalization_context_shift"]:
            return {"prefix": "First identify the current user request, distinct from quoted prior turns. For an explicit request to forget, keep private or stop using a topic, acknowledge the boundary without repeating its subject or details, and do not invent extra limitations. For rewriting, translation or a self-contained task, deliver the requested artifact using only context appropriate to this occasion. Do not weave unrelated remembered interests into the response. Preserve the user's requested tone and output format; avoid extra commentary about a profile or memory process."}
        return {}
    '''),
    "preference_decision_plan": _source('''
        if context.get("stage") == "generation":
            if context.get("memory_records") and context.get("query"):
                return {"workflow": "program"}
            return {}
        if context.get("stage") != "step":
            return {}
        step = context["step_index"]
        observations = context["observations"]
        if step == 0:
            return {"operation": "analyze", "prompt": "Return JSON with selected_ids, current_goal, current_constraints, and decision_changes. Each change must cite a supplied record ID and name a concrete recommendation, ranking, tradeoff, style or content choice that the preference supports. Distinguish explicit current instructions, historical positive preferences, negative constraints and unknown facts. A requested topic may reinforce an existing preference; do not discard it just because the query mentions it. Casual humor or frustration is not an opt-out unless the user actually asks to avoid that topic. Preserve explicit opt-outs and privacy. If the request is self-contained or no evidence is useful, return no decision changes; never invent a detail or add unrelated personal facts."}
        if step == 1:
            data = observations[0].get("data") or {}
            if "selected_ids" not in data:
                return {"operation": "analyze", "prompt": "The previous analysis did not provide selected_ids. Return only selected_ids as a list of exact supplied record IDs and a short decision_changes list. This is internal evidence selection, not a user-facing answer. Keep relevant supporting preferences and constraints; use [] only if no supplied record applies."}
            allowed = []
            for record in context["memory_records"]:
                allowed.append(record["id"])
            selected = []
            for identifier in data.get("selected_ids", []):
                if identifier in allowed and identifier not in selected:
                    selected.append(identifier)
            return {"operation": "select_memory", "selected_ids": selected}
        if step == 2:
            return {"operation": "generate", "prompt": "Complete the user's actual task in the required output format. Make the supported decision changes concrete in the artifact: select or rank options, choose appropriate detail and style, or explain relevant tradeoffs. Do not merely claim that an answer is personalized. Keep explicit current instructions above historical preferences. Use a clearly marked placeholder for an unavailable personal fact and still provide the usable draft. Keep warnings and necessary next actions in the user-facing answer, not only in reasoning. Never expose the selection process or force personal context into a task that does not need it."}
        if step == 3:
            return {"operation": "analyze", "prompt": "Audit the draft against the original request and selected evidence. Return JSON with revise (boolean) and correction. Revise only for a concrete missed supported preference, mistaken interpretation of the current request, missing requested deliverable, unsupported personal claim, or privacy/opt-out violation. Quote the evidence ID for preference changes. Do not demand new personal details when the request is already satisfied, and do not add filler merely to look personalized."}
        if step == 4:
            review = observations[-1].get("data") or {}
            if review.get("revise") and review.get("correction"):
                return {"operation": "generate", "prompt": str(review["correction"])[:2000]}
        return {"operation": "finish"}
    '''),
    "semantic_memory_routing": _source('''
        if context.get("stage") == "generation":
            return {"workflow": "program"}
        if context.get("stage") != "step":
            return {}
        results = context["observations"]
        if context["step_index"] == 0:
            return {"operation": "analyze", "prompt": "Return selected_ids for query-relevant memory records, retaining applicable negative constraints. Use only supplied record IDs. Use [] if no evidence is relevant."}
        if context["step_index"] == 1:
            selected = (results[0].get("data") or {}).get("selected_ids", [])
            return {"operation": "select_memory", "selected_ids": selected}
        if context["step_index"] == 2:
            return {"operation": "generate", "prompt": "Answer the original request using relevant supported evidence, without inventing preferences."}
        return {"operation": "finish"}
    '''),
    "conditional_verification": _source('''
        if context.get("stage") == "generation":
            return {"workflow": "program"}
        if context.get("stage") != "step":
            return {}
        results = context["observations"]
        if len(results) == 0:
            return {"operation": "generate"}
        if len(results) == 1:
            return {"operation": "analyze", "prompt": "Check the draft against the user's applicable preferences and current instruction. Return revise (boolean) and correction (short string); do not require personalization when it is irrelevant."}
        review = results[1].get("data") or {}
        if len(results) == 2 and review.get("revise", False):
            return {"operation": "generate", "prompt": str(review.get("correction", ""))[:2000]}
        return {"operation": "finish"}
    '''),
    "structured_preference_state": _source('''
        if context.get("stage") == "generation":
            return {"workflow": "program"}
        if context.get("stage") != "step":
            return {}
        if context["step_index"] == 0:
            return {"operation": "analyze", "prompt": "Identify which state records apply to this request, including conflicting evidence, uncertainty and negative constraints. Return a JSON decision summary."}
        if context["step_index"] == 1:
            return {"operation": "generate", "prompt": "Use only supported state that applies to the current request. Current instructions override remembered preferences."}
        return {"operation": "finish"}
    ''', remember='''
        return {
            "state_schema": {
                "type": "object",
                "properties": {
                    "preference": {"type": "string"},
                    "scope": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "conflict": {"type": "boolean"}
                },
                "required": ["preference", "scope", "confidence", "conflict"],
                "additionalProperties": False
            },
            "guidance": "Track supported preferences, their applicability and uncertainty. Keep conflicting evidence explicit; neutral behavior is not a positive preference."
        }
    '''),
    "specific_preference_ledger": _source('return {}', remember='''
        return {"max_chars": 12000, "guidance": "Keep a compact ledger of distinct supported preferences, not a broad thematic biography. The supplied historical inferred_prefs are fallible past evidence, not current instructions: preserve their concrete preference wording, named works or people, formats, direction and scope when supported by positive observed engagement. Do not replace a specific taste with a broad category or merge unrelated preferences into one vague sentence. Keep one preference per bullet and retain the earlier supported specifics on later rewrites. A skipped or unexpanded item alone does not revoke a prior positive preference or establish a dislike of its whole topic. Explicit corrections, stop-personalizing instructions and forget requests control and must not be weakened; keep their exact scope, not a blanket subject ban. Distinguish an inferred preference from a verified personal fact. Do not promote assistant suggestions into user preferences. Maintain factual commitments separately with exact supplied UTC values and apply updates and cancellations. Omit repetitive weak skip lists before removing positive specifics. Never invent a preference or recover a forgotten fact."}
    '''),
}

PROGRAMMABLE_CONTRACT = """
## Programmable memory extension (opt-in, not compact memory)
All four hooks remain editable. You may replace the seeds, collection schemas,
updates, query construction, retrieval and answer policy. This is not a fixed
RAG pipeline. The host, not generated Python, owns storage and all I/O.

remember returns a memory_backend operation; recall:context returns memory_query.
The host executes it and invokes the SAME hook again with context.backend:
phase (write/read), step, state, result={data,truncated}, collection counts,
schemas, limits and model_calls_remaining. Step zero has no result. A decision
may carry state (JSON <=8000 chars). It is local to this read/write, not durable.
Return finish within 32 steps. Each read/write permits at most one model call;
read calls count toward the action's total four model calls. There is no total
wall-clock deadline for the backend read/write; elapsed time is diagnostic only.
Hook sandbox, SQL resource limits and provider request handling remain unchanged.
Only read output,
not the storage snapshot, reaches the answering model. apply and guard receive
backend_evidence={output,source_refs} after a read.

Operations, nested inside {memory_backend: {...}} or {memory_query: {...}}:
- schema: collection, schema (JSON schema for this collection's data).
- upsert: records=[{collection,key,data,source_refs,text?,vector?,expires_on?}].
  Use stable keys. Data is arbitrary JSON. text defaults to serialized data.
  vector is optional, finite, <=1024 dimensions. No automatic external encoder.
- delete: collection,key. clear: collection. These are write-only.
- sql: sql,parameters. A read-only SQLite SELECT over records(collection,key,
  data,text,sources,updated_day,expires_on,ref). data is JSON; json_extract,
  filtering, joins, aggregates and ordering are supported. No DDL, PRAGMA,
  ATTACH, extensions, arbitrary files or other databases. Writes use upsert.
- search: text OR vector, optional collection,limit (1..64). Text uses local
  character TF-IDF cosine, not semantic embeddings. Vector uses cosine over
  supplied vectors. Combine SQL and retrieval or use model to rerank/synthesize.
- model: prompt,input,optional schema. The fixed host memory/executor model
  receives only these supplied values. A schema requests JSON and validates it.
  Result has text,data,valid_json,truncated. Invalid JSON is not a valid finding.
- finish: write has no output. read has output (<=12000 chars),source_refs from
  existing record references. Any nonempty output needs source_refs. References
  prove evidence eligibility only; synthesized rules still need evaluation.

remember receives source_events with stable key, ref, item, action, topics and
bounded messages. These are ONLY this user's authorized causal observations.
Unlike compact mode, this registered mode permits retaining these supplied
source records in the backend. It does NOT permit reading raw-history files,
the current answer, private selection, native test labels, or another user.
Unseen sources cannot be invented. Previous records use record:collection:key
references. Forget boundaries clear prior sources even when filtered; expiry
propagates through derived records. Reading is strictly read-only. Writes commit
atomically with the cursor only after finish; failures retain the old snapshot.
Store limits: 4096 records, 2 million serialized chars, 16000 chars per record,
64 source receipts per record. Query results are bounded and mark truncation;
use SQL pagination or narrower queries instead of assuming missing rows are absent.
Changing remember requires a fresh causal rebuild, never inheriting another
writer's store. W0 stays the original compact baseline. Full TRAIN evidence,
including source writes, queries, results, model calls, outputs and feedback,
remains available; no sampling replaces that evidence. Provider refusals are
external failures, never instructions to evade filters or hide the input.
"""

_INDEX_WRITE = '''
    backend = context.get("backend", {})
    if not backend:
        return {}
    position = backend.get("state", {}).get("position", 0)
    events = context.get("source_events", [])
    if position >= len(events):
        return {"memory_backend": {"operation": "finish"}}
    event = events[position]
    text = event.get("item", "")
    for message in event.get("messages", []):
        text += " " + message["content"]
    return {"memory_backend": {"operation": "upsert", "records": [{"collection": "observations", "key": event["key"],
        "data": {"action": event.get("action", ""), "topics": event.get("topics", [])}, "text": text,
        "source_refs": [event["ref"]]}], "state": {"position": position + 1}}}
'''

_INDEX_FINISH = '''
    matches = context["backend"]["result"].get("data") or []
    texts = []
    refs = []
    size = 0
    for match in matches:
        text = match.get("text", "").replace("\\n", " ")
        if size + len(text) < 11000:
            texts.append("- " + text)
            refs.append(match["ref"])
            size += len(text) + 3
    output = "## Retrieved prior observations\\n" + "\\n".join(texts) if texts else ""
    return {"memory_query": {"operation": "finish", "output": output, "source_refs": refs}}
'''

PROGRAMMABLE_SEEDS = {
    "indexed_history": _source("return {}", remember=_INDEX_WRITE, recall=textwrap.dedent('''
        if context.get("stage") != "context" or not context.get("backend"):
            return {}
        if context["backend"]["step"] == 0:
            return {"memory_query": {"operation": "search", "text": context["query"], "limit": 5}}
    ''') + textwrap.dedent(_INDEX_FINISH)),
    "query_rewrite": _source("return {}", remember=_INDEX_WRITE, recall=textwrap.dedent('''
        if context.get("stage") != "context" or not context.get("backend"):
            return {}
        step = context["backend"]["step"]
        if step == 0:
            return {"memory_query": {"operation": "model", "prompt": "Form a short retrieval query describing the decision-relevant concepts in this request. Do not guess its answer or invent user facts.", "input": context["query"], "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}}
        if step == 1:
            result = context["backend"]["result"].get("data") or {}
            query = context["query"]
            if result.get("valid_json") and not result.get("truncated"):
                query = result["data"]["query"]
            return {"memory_query": {"operation": "search", "text": query, "limit": 5}}
    ''') + textwrap.dedent(_INDEX_FINISH)),
    "structured_rules": _source("return {}", remember='''
        return {"state_schema": {"type": "object", "properties": {"condition": {"type": "string"}, "choice": {"type": "string"}, "exceptions": {"type": "array", "items": {"type": "string"}}}, "required": ["condition", "choice", "exceptions"], "additionalProperties": False}, "guidance": "Represent conditional decision regularities and exceptions supported by these observations. Treat rules as fallible, not immutable facts. Preserve explicit corrections and non-use constraints. Do not memorize current answers or reinterpret an observed choice as a general liking."}
    '''),
}


SEARCH_MOVES = (
    "Reconsider evidence selection: implement a request-conditioned memory routing algorithm.",
    "Reconsider control flow: branch on intermediate semantic checks and revise only when needed.",
    "Reconsider the state representation and update algorithm, with verifiable evidence references.",
    "Compose useful parts of previous structures, removing redundant calls and unsupported rules.",
    "Simplify or replace the current algorithm where feedback shows unnecessary personalization.",
)


def reference_prompt(proposal_index: int, pool: Sequence[Mapping[str, str]], *, programmable: bool = False) -> str:
    seeds = PROGRAMMABLE_SEEDS if programmable else REFERENCE_SEEDS
    names = list(seeds)
    seed_name = names[(max(1, proposal_index) - 1) % len(names)]
    sections = [
        "## Open program search",
        "Search algorithms, not only prompt wording. These starting structures are examples, not a finite action catalog. "
        "You may add, remove or compose steps, rewrite helper functions and state schemas, or keep a simple policy. "
        "The current incumbent remains the deployment parent; archived code is a donor, not evidence of quality. "
        "Never invent user facts to increase reward. Use causal memory and completed TRAIN feedback only.",
        "Structural exploration focus: " + SEARCH_MOVES[(max(1, proposal_index) - 1) % len(SEARCH_MOVES)],
        "Reference starting structures: " + ", ".join(names),
        "Example " + seed_name + ":\n```python\n" + seeds[seed_name] + "```",
    ]
    for candidate in pool[-3:]:
        if len(candidate["source"]) <= 8_000:
            sections.append("Archived candidate " + candidate["sha256"][:12]
                + " (unranked code-only donor):\n```python\n" + candidate["source"] + "```")
    return "\n\n".join(sections)


def execution_feedback(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active = Counter()
    unused = Counter()
    operations = Counter()
    stops = Counter()
    fallbacks = 0
    errors = []
    for trace in traces:
        for event in (trace.get("program_report") or {}).get("events", []):
            hook = event.get("hook")
            stage = event.get("stage")
            allowed = STAGE_DECISION_KEYS.get(hook if hook == "remember" else f"{hook}:{stage}", ())
            for key in set(event.get("applied_keys") or []) & set(allowed):
                active[f"{hook}:{stage}.{key}"] += 1
            for key in (set(event.get("decision_keys") or []) - set(event.get("applied_keys") or [])) & set(allowed):
                unused[f"{hook}:{stage}.{key}"] += 1
            fallbacks += int(event.get("status") not in {None, "ok"})
            if event.get("status") not in {None, "ok"}:
                diagnostic = event.get("diagnostic") or {}
                errors.append({"hook": hook, "stage": stage, "status": event["status"],
                    **{key: diagnostic[key] for key in ("kind", "error_type", "source_line", "guidance",
                        "step_budget", "time_budget_seconds") if key in diagnostic}})
        flow = trace.get("program_flow") or {}
        operations.update(operation for operation in flow.get("operations", []) if operation in PROGRAM_OPERATIONS)
        if flow.get("stop_reason") in {"finished", "tool_action", "step_limit", "model_call_limit", "invalid_memory_selection", "invalid_final_output", "empty_or_invalid_step"}:
            stops[flow["stop_reason"]] += 1
    return {"active_decisions": dict(active), "requested_but_unused": dict(unused),
            "operation_counts": dict(operations), "stop_reasons": dict(stops),
            "hook_failures": fallbacks,
            "hook_errors": errors[:8],
            "model_calls": sum(int(trace.get("model_calls") or 0) for trace in traces)}


class _ProbeLLM:
    model = "gpt-5.6-luna"
    reasoning_effort = "low"

    def __init__(self, needs_revision: bool):
        self.needs_revision = needs_revision
        self.requests: list[str] = []

    def chat(self, messages, **kwargs):
        payload = {key: value for key, value in kwargs.items() if key not in {"temperature", "top_p"}}
        self.requests.append(json.dumps({"messages": messages, "options": payload}, sort_keys=True))
        content = '{"response":"probe answer"}' if kwargs.get("response_format") else "probe answer"
        if kwargs.get("response_format"):
            try:
                data = messages[-1]["content"]
                if data.startswith("BEGIN QUOTED TASK DATA\n"):
                    data = data.removeprefix("BEGIN QUOTED TASK DATA\n").split("\nEND QUOTED TASK DATA\n", 1)[0]
                evidence = json.loads(data)
            except (TypeError, ValueError):
                evidence = {}
            selected = [record["id"] for record in evidence.get("memory_records", [])[:1]]
            constraints = [{"id": record["id"], "quote": record["text"][:480],
                "use": "Keep the supported qualifier when it helps the current task."}
                for record in evidence.get("memory_records", [])[:1]]
            changes = [{"id": item["id"], "quote": item["quote"], "old_text": "probe answer",
                "correction": "Keep the supported qualifier in the proposed answer."}
                for item in constraints if any(observation.get("operation") == "generate"
                    and "probe answer" in observation.get("text", "")
                    for observation in evidence.get("observations", []))]
            content = json.dumps({"selected_ids": selected, "revise": self.needs_revision,
                "needs_revision": self.needs_revision, "supported": not self.needs_revision,
                "correction": "Check the user's applicable constraints.", "response": "probe answer",
                "constraints": constraints, "changes": changes})
        return {"content": content, "tool_calls": [],
                "usage": {"prompt": 1, "completion": 1, "reasoning": 0}}


def select_probe_queries(queries, *, focus_groups=(), limit=3):
    groups = {}
    for query in queries:
        task = str(query.get("task_kind", "")) if isinstance(query, Mapping) else ""
        group = task.removeprefix("pm3_native::")
        groups.setdefault(group, []).append(query)
    order = list(dict.fromkeys([*focus_groups, "chatbot_personalized_response", *groups]))
    selected = []
    for group in order:
        if groups.get(group) and len(selected) < limit:
            selected.append(groups[group].pop(0))
    for group in order:
        for query in groups.get(group, []):
            if len(selected) < limit:
                selected.append(query)
    return selected


def probe_candidate(
    source: str, parent_source: str, *, memory_md: str,
    queries: Sequence[str | Mapping[str, str]], user_id: str, memory_backend_enabled: bool = False,
) -> dict[str, Any]:
    candidate, parent = HarnessProgram(source), HarnessProgram(parent_source)
    memory_changed = candidate.fingerprint.hook_sha256["remember"] != parent.fingerprint.hook_sha256["remember"]
    requires_replay = memory_changed
    if memory_backend_enabled:
        from .memory_backend import MemoryStore, is_backend

        for case in queries:
            document = str(case.get("memory_md", memory_md)) if isinstance(case, Mapping) else memory_md
            if is_backend(document):
                payload = MemoryStore.payload(document)
                if payload["user_id"] != user_id:
                    raise ValueError("probe memory belongs to another user")
                requires_replay = requires_replay or payload["writer_sha256"] != candidate.fingerprint.hook_sha256["remember"]
    if memory_backend_enabled and requires_replay:
        return {"api_calls": 0, "probe_cases": 0, "valid": True, "behavior_changed": False,
            "memory_requires_causal_replay": True, "eligible": True, "checks": [], "diagnostics": [],
            "quality_measured": False, "deferred": "writer_or_snapshot_changed; actual causal rebuild required before scoring"}
    comparisons = []
    valid = True
    for case in list(queries)[:3]:
        query = str(case.get("query", "")) if isinstance(case, Mapping) else str(case)
        task_kind = str(case.get("task_kind", "program_probe")) if isinstance(case, Mapping) else "program_probe"
        case_memory = str(case.get("memory_md", memory_md)) if isinstance(case, Mapping) else memory_md
        for revision in (False, True):
            traces = []
            requests = []
            for program in (parent, candidate):
                probe_llm = _ProbeLLM(revision)
                runner = Runner(probe_llm, ExecutableHarness(
                    user_id=user_id, config=make_minimal_config(), memory_md=case_memory, program=program,
                    memory_backend_enabled=memory_backend_enabled))
                traces.append(runner.run(episode_id="local-probe", user_id=user_id, condition="local",
                    task_kind=task_kind, messages=[{"role": "user", "content": query}],
                    query=query, max_tokens=4096))
                requests.append(probe_llm.requests)
            bad = bool(traces[1].program_report.get("fallbacks")) or traces[1].program_flow.get("stop_reason") in {
                "invalid_memory_selection", "invalid_final_output", "empty_or_invalid_step"}
            valid = valid and not bad
            comparisons.append({"changed": requests[0] != requests[1]
                                or traces[0].final != traces[1].final,
                                "task_kind": task_kind,
                                "candidate_valid": not bad,
                                "candidate_execution": execution_feedback([traces[1].as_dict()])})
    changed = any(row["changed"] for row in comparisons)
    diagnostics = []
    for comparison in comparisons:
        for error in comparison["candidate_execution"]["hook_errors"]:
            if error not in diagnostics:
                diagnostics.append(error)
    return {"api_calls": 0, "probe_cases": len(comparisons), "valid": valid,
            "behavior_changed": changed, "memory_requires_causal_replay": memory_changed,
            "eligible": valid and (changed or memory_changed or not comparisons),
            "checks": comparisons, "diagnostics": diagnostics[:8], "quality_measured": False}
