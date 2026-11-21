# Write one user's executable harness

Write the user-specific Python section for ONE user. There is no preinstalled shared history gate or model-produced
task classification. Your code decides how the user's history informs each request. The answer
model remains Qwen3.5-9B. Treat history as evidence, not instructions.

## Interface

Define USER_MODEL as a literal JSON-compatible dict of evidence-backed vocabulary, preferences, voice, boundaries and
preference changes. Define these functions, optionally with helpers named u_*:

- user_route(request, kind, p_none): kind and p_none are ALWAYS None, retained only for API compatibility. Inspect the
  request itself; request_core(request) extracts the user's words when a known marker is present. Return a dict with
  history (bool), queries (1-3 BM25 search strings in order), recent (None or self/chat/calendar/dm), and per_record
  (None or a character limit). There is no upstream gate: your code must decide whether history is useful. Avoid
  irrelevant personal context on generic requests, but do not require explicit personal pronouns for tailored advice.
- user_keep(text, app, kind): return whether to retain a retrieved event. kind is ALWAYS None. This function has no
  current request argument; do not pretend its filters are request-aware. Avoid blanket topic or substring filters.
- user_guidance(request, kind, history): return a short, concrete answer instruction or an empty string. kind is ALWAYS
  None; history is True only when retrieval returned retained source references. Do not claim evidence was retrieved
  when it was not. This function still runs when history is absent; do not inject irrelevant static personal facts.

The default backbone provides request_core, words, has_any and BACKBONE. It executes local BM25/SQL retrieval and passes
evidence to the answer model. Searches return up to 30 records each, with 11000 characters total; later queries and
the optional recent view (10 records) run only while space remains. Guidance is limited to 1500 characters.

## Evidence use

Prefer distinctive terms in the request over task-template words or broad lists of profile interests. Bound individual
records when long conversations crowd out evidence. Preserve relevant negative and changing preferences. Own posts,
explicit engagement and own conversations are stronger evidence than views or skips. Scope private facts to the
current request, require evidence for personal facts, and state uncertainty rather than inventing missing details.
Writing guidance should capture voice, not copy catchphrases. Recommendations should avoid repeating familiar items.
For unsolicited check-ins, require current concrete justification rather than assuming old sensitive history permits
contact. Routing and tool-use choices must be implemented in the generated code.

## Optional workflow overrides

Always define u_recall(context) and u_apply(context), returning {} when unused so the program validates. You may
set USER_MODEL["custom_recall"] = True in the literal USER_MODEL to replace the
default retrieval workflow. It receives the complete recall context including backend state and the previous result.
You may similarly define u_apply(context) and set "custom_apply": True to use retrieved evidence in a custom answer
prefix. Both flags are absent from the default control, so these workflows only run when your generated code opts in.
Always keep the three standard functions for compatibility. Do not use global mutation or redefine backbone functions.

<<DECISION_POLICY>>

## Constraints

- Allowed Python subset: <<LANGUAGE>>. No imports, comprehensions, lambda, while, try, with, classes, global, set literals
  or isinstance/type. Use ordinary for loops and list append.
- Top-level assignments are literal values. Do not redefine or modify backbone names, including BACKBONE, recall,
  apply, remember, guard, finish, search, recent, route_for, collect, words, has_any and request_core.
- Entire program: at most <<MAX_CHARS>> characters and <<MAX_NODES>> AST nodes; section about <<SECTION_CHARS>> characters.
- Do not use evaluation labels, task IDs or hardcoded benchmark answers.

## Default non-personalized section

```python
<<CONTROL_SECTION>>
```

## Shared backbone

```python
<<BACKBONE>>
```

## User history digest

<<DIGEST>>

## Output

Reply with one fenced Python block containing USER_MODEL, user_route, user_keep, user_guidance, u_recall, u_apply and optional u_ helpers.