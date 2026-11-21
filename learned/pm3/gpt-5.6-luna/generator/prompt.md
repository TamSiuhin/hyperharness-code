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

## Request-conditioned retrieval for recommendations
Some public requests ask for a recommendation or ranking on behalf of "the user" rather than saying "my" or
"I". Treat an explicit request to personalize a suggestion, choose an activity, or rank a candidate slate as a
reason to use this user's history. In user_route, detect that intent from the visible request text; personal
pronouns alone are insufficient. Check this branch before any exact-personal-fact branch: a ranking task's
timestamp, "time-masked history" wording, candidate captions, and output schema do not turn it into a request
for the user's date or other exact value. A visible heading asking for a personalized recommendation or
instructions to rank items by the user's likely engagement are sufficient intent, even without a separate
"for the user" phrase. Keep this branch specific to requests that ask for personalization.

A candidate slate is the item set to rank, not the search query. Searching the whole slate, its task heading,
timestamp, output schema, or many unrelated hashtags dilutes the limited evidence window. For ranking, build
short queries from a small set of the user's positive and negative interest terms in the construction history,
and from any distinctive candidate topic that overlaps those interests. Put the selected terms in USER_MODEL so
the route reads actual user evidence rather than generic category defaults. For vague personalized suggestions,
query recent interests and any explicit constraints; avoid inventing a preference when neither is supported.
Keep current refusals and changing interests visible. Preserve user_keep and exact personal-fact behavior for
other requests. A broad slate query that retrieves plausible but unrelated posts would disconfirm this route.
For a visible social-media ranking request, the shared recall path first supplies compact recent liked and
disliked interactions with timestamps, then runs the user_route topic queries. Use those retrieved interactions
as the primary ranking evidence. A viewed post is weaker evidence than an explicit like or dislike; a
self-authored or chat mention alone is not engagement with a candidate category. Do not infer that every
candidate sharing one broad hashtag has the same appeal. The compact interaction rows already cover recent
behavior, so keep route queries focused on distinctive topics and avoid repeating a generic profile query.
For other tasks, keep retrieval for a request that needs a specific value from the user's records, even when the
value is expressed as "my" plus a noun absent from a fixed field list. Use only JSON-compatible literals in
USER_MODEL and other module-level state; lists can hold structured topic entries.

## Evidence-conditioned personal facts
The construction digest is a lossy summary for choosing behavior; it is not a source to quote as a fact in a
future answer. A request for an exact personal fact (date, address, identifier, appointment, commitment, name,
or what the user actually said or posted) needs a directly relevant runtime record. In user_route, search the
distinctive nouns and names in the user's current request, even when the request is phrased as a writing task.
Do not replace a specific request with a broad interest alias. If the fact may have changed, retrieve both the
original and later correction or cancellation. Give enough of each returned record to include the fact and its
qualification; very short per_record clips can hide the relevant field. Keep source provenance and negation.

In user_guidance or u_apply, tell the answer executor to use an exact personal fact only if the recalled evidence
states it explicitly for the same item. Include writing requests that ask to insert *my* date, address, name,
identifier, appointment detail, or other personal value, even if phrased as "write a listing" or "draft a reply"
instead of "what is my ...". A thematically related post, liked feed item, or calendar entry does not establish
a date or identifier for another item. When direct evidence is absent, use a visible placeholder or say the fact
cannot be verified, while still doing the rest of the requested task. This rule should activate on requests that
need personal facts; routine writing, recommendation, and general questions should retain their own appropriate
behavior. Store reusable decision settings in USER_MODEL and have the generated functions read them. Do not
hard-code any expected request or answer from an evaluation set.

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
