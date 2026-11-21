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
The construction history below lists the user's own dialogue in order. A preceding assistant line is context
for the following user reply, not independent evidence that the user accepted that advice. Activity is sampled;
absence from that sample proves nothing. Read across sessions for repeated constraints and later corrections.

Write executable, user-specific behavior in user_route and user_guidance. Choose a few concrete recurring
topics supported by this user's statements. For each, use words from the public request to trigger a focused
retrieval query, and give the final model a short conditional reminder about the user's relevant constraint or
preference. Keep qualifiers: an objection to one demanding plan does not imply rejecting its whole topic.
Only state a factual preference in guidance when both the public request and retrieved evidence make it
relevant; inspect evidence.output rather than treating source_refs alone as support. The digest is for learning
what to look for, while runtime observations and current logs are the evidence available when answering.
Do not copy benchmark answers or choose option positions in code. Avoid long generic advice that repeats
the final model's task. Include model-written function logic reachable from a runtime hook.

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
