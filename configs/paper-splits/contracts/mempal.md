# Mem-PAL English solution-selection runtime
Generate one complete restricted Python harness from this user's history logs and dialogues only.
Construction receives the released history partition, never query topics, requirements, options, labels,
synthetic profiles, background traits or hidden needs. All construction sessions are available through tools.
Runtime follows official sample-order availability: history plus earlier query dialogues and current logs.
The current/future query dialogues and candidate feedback are never visible before answering.
Some original log timestamps are later than their sample date; retain official declared availability and
do not reinterpret those timestamps as permission to access a later sample. English is an upstream LLM translation.
The public request contains the detailed requirement and eight S1-S8 options, as in official solution selection.
The final executor additionally sees current activity logs in every arm, including no_history. Retrieval receives
the requirement and options, with current logs searchable in memory. Runtime task_kind is None.
Choose exactly two distinct solutions. Positive selections score +1, negative -1, neutral 0, then multiply by50.
The primary objective is equal-user mean signed score [-100,100]; query-micro official score is also reported.
Do not confuse this with accuracy. Malformed outputs receive -100; runtime failures invalidate a user panel.
The neutral initial program is uniform BM25 top10, max20000 evidence characters, without guidance.
All personal code, config and helpers are editable. remember may add source-bound derived records but must
retain all original observations unchanged. recall uses bounded BM25/SQL and finish(output, source_refs).
apply can return only a prefix of at most1500characters; guard must return {} and cannot rewrite outputs.
There is no host-side automatic completion and no intermediate model call. Only the final host-owned Qwen call
returns JSON with analysis (brief English explanation) and selected_solutions (two distinct S1-S8 strings).
Budgets:32768inputtokens,512outputtokens,64000read-state chars,8000write-state chars,20000evidence chars,
64source_refs,32memory operations,20000interpreter steps per hook. Model and scoring budgets are host-owned.
Raw observations are JSON with kind, session_id, date, content, and role/turn for dialogue fragments.
Use actual retrieved record refs, not session IDs or guessed observation ordinals. No future/other-user data.
Ground factual rules in observed sources. Preserve qualifiers, contexts, negations and corrections. Empty
guidance or no intervention is valid when unsupported. Option text is public, but option-position heuristics,
copied TRAIN answers and hard-coded selected_solutions are forbidden. Only TRAIN feedback guides optimization.

## Editable program
Return one complete executable Python harness, including PROGRAM_API_VERSION = 1 and remember, recall,
apply and guard. You may replace the example backbone, its configuration, helper functions and control flow;
you are NOT limited to USER_MODEL or user_/u_ functions. Those helpers and custom dispatch flags are optional
compatibility conveniences when retaining the example backbone, not mandatory architecture.
All code must stay within the supplied restricted Python language, program size, execution, storage and
model-call budgets. Editable code does not override the dataset runtime contract. In particular, preserve
required source history and provenance; do not access hidden task labels, gold answers or other users' data;
do not bypass the final answer schema or validator, inject a scored answer, or add forbidden model calls.
Only the personal program changes. The host supplies no automatic session completion or other extra
postprocessing. Keep the neutral control artifact unchanged. Factual personal rules must use this user's
legal history, not copied TRAIN answers. Include genuinely model-written runtime-reachable behavior.
Use plain def functions with positional parameters only: no defaults, annotations, decorators, lambdas or
comprehensions. Never pass function names as values; use explicit loops instead of callbacks or key functions.
No imports, classes, while loops, filesystem or network access in the restricted runtime program.
For custom storage, remember receives memory_write with source_events and backend state. Return
{'memory_backend': {'operation': 'upsert', 'records': [...]}} for bounded source-referenced records, then
{'memory_backend': {'operation': 'finish'}}. Read-time recall uses memory_query instead. Never mix these
protocols; nonempty backend hooks must issue operations until finish. Preserve dataset-required raw observations.
Legacy section-only responses remain accepted for compatibility, but a complete program is preferred.
