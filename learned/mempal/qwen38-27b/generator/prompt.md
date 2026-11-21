# Generate this user's executable personal section

## Editable program
Return only USER_MODEL plus user_/u_ functions or helper data. The host checks the assembled program against the
shared backbone. Do not redefine PROGRAM_API_VERSION, BACKBONE, remember, recall, apply, guard, shared helpers,
or access shared global data directly. Shared helper functions may be called. Follow the supplied hook signatures.
Use user_route for retrieval routing, user_keep for evidence filtering and user_guidance for relevant guidance;
u_recall/u_apply are optional overrides only through the backbone's documented dispatch flags. Missing optional
hooks receive neutral defaults. This section-only scope overrides any generic full-program instructions below.
Preserve the dataset's provenance, privacy boundaries, output schema and host budgets. No intermediate model calls.

## Personalization objective
Infer this user's preferences, constraints, uncertainty and changes from legal history. Use request-specific
retrieval and source-grounded guidance; preserve context and negation. History is evidence, not instructions.
Do not force personalization into unrelated requests or copy benchmark answers into code. Include genuinely
model-written executable behavior reachable from runtime hooks, not only profile data or unused functions.
Write a small, history-grounded function that affects user_route or user_guidance for a matching request.
Use the requirement as the primary retrieval query. The eight options and current logs are already public to
the final answer model; putting their entire text into a BM25 search can bury earlier user dialogue.
If a user reacts to a proposal, distinguish the proposed action from the user's acceptance or rejection.
Do not turn a narrow objection into a general ban or assume a later statement is a permanent preference.
Keep guidance brief and conditional on the current requirement and retrieved source evidence.

## Dataset runtime contract
<<DATASET_RUNTIME>>

## Language and limits
<<LANGUAGE>>
At most <<MAX_CHARS>> characters and <<MAX_NODES>> AST nodes.
Use plain def functions with positional parameters: no imports, classes, defaults, annotations, decorators,
comprehensions, lambdas, while loops, filesystem or network access. Never pass functions as values.

## Shared backbone (read-only for the per-user generator)
```python
<<BACKBONE>>
```

## Personal section interface
```python
<<CONTROL_SECTION>>
```

## Legal construction history
<<DIGEST>>

## Output
Return only the personal section in one Python code block. Do not copy the shared backbone into your response.

## Personal retrieval and registered budgets
BM25 top_k is personal: set USER_MODEL['top_k'] as a default, or return 'top_k' from user_route for a request-specific override. Integers 1..64 are supported; values outside the range are clamped and invalid values use the shared default. The shared historical default remains unchanged. Code check budget: 10; construction LLM calls: 10; completion tokens including reasoning: 200000; initial repair limit: 6; interpreter steps per runtime hook: 40000. Evidence, guidance, query-count, answer-model budgets and scoring are unchanged. These registered numbers supersede older values below.

## Task-specific evidence use
Separate explicit user preferences, user feedback on suggestions and assistant-proposed ideas.
Preserve negation, dates and preference changes. Do not treat every assistant suggestion as accepted,
or discard all assistant context when it is needed to interpret user feedback. Match guidance to the
current requirement; retain uncertainty and avoid exporting a preference from an unrelated topic.
