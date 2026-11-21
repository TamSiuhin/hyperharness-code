Write this user's personalized executable section for a two-choice recommendation harness.

The runtime receives a public request with one requirement, S1-S8 options, and current activity logs. It must choose the two solutions best matching this user's needs. The history below is construction evidence only; its sessions and dialogue are not evaluation questions. The final answer model already sees the request, options, current logs and your retrieved evidence. Your code should help it find and interpret personal evidence.

Return ONLY a Python code block defining USER_MODEL, user_route(request, kind, p_none), user_keep(text, app, kind), and user_guidance(request, kind, evidence). You may add user_/u_ helpers, but only if a runtime hook calls them. The host inserts your section into the exact shared backbone below. Do not redefine shared functions or globals. A genuinely model-written function must be reached from a runtime hook; data-only output is invalid.

Practical approach:
- Set a personal top_k, usually 8-16. user_route can add up to two focused query strings based on the visible requirement. Its first retrieval query is the request core; your additions should seek relevant user feedback and actions. Return {'history': True, 'queries': [...], 'top_k': N}.
- user_keep should generally keep evidence; do not filter out all assistant dialogue, since a user reply may refer to its preceding suggestion.
- user_guidance can return at most 1500 characters. Give only concise, request-relevant distinctions supported by the history. Explicitly separate the USER's acceptance, rejection, and constraints from the assistant's proposals. Current activity and the requirement may supersede old context. An assistant suggestion by itself is not a user preference. Silence is better than irrelevant profile facts.
- Avoid broad topic heuristics that fire on incidental words, fixed choices, option-position rules, and verbatim benchmark-like answers. The chosen options are decided at runtime by the final model.
- Preserve qualifications and changes over time. An action log shows behavior, not necessarily endorsement of a proposed method.

Restricted language: <<LANGUAGE>>. Program limit <<MAX_CHARS>> characters and <<MAX_NODES>> AST nodes. Use plain def functions with positional parameters only. No imports, classes, defaults, annotations, decorators, comprehensions, lambdas, while loops, filesystem, or network. Do not pass function names as values.

Shared backbone, read-only:
```python
<<BACKBONE>>
```

Interface example, not a template to copy unchanged:
```python
<<CONTROL_SECTION>>
```

This user's construction history, in session order. USER lines are direct statements; assistant proposals are only context for adjacent user replies. The digest samples actions and dialogue across all available sessions, so runtime retrieval should verify details from original observations.
<<DIGEST>>

Return only the personal section in one ```python block.
