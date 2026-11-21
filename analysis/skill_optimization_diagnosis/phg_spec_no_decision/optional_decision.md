## Optional decision tool, never a preinstalled policy

Clef is available as a host tool to generated harness code. It is NOT required: zero decision calls is valid. Do not
add a fixed gate or classification pass merely because the tool exists. Your generated u_recall may use it when the
user's evidence and current request warrant it, or use local retrieval alone. The model is fixed; "learning Clef use"
means learning whether, when, what to ask and how to use its result, not training Clef weights.

The per-request decision budget is <<DECISION_CALLS>>. General-purpose model calls inside the harness are disabled.
All operations are yielded as {"memory_query": {...}} from u_recall, never direct network or Python imports.

- On the first call context["backend"]["step"] is 0. Later calls include backend["state"] and backend["result"]["data"].
- A decision request has operation "decide", situation (request/evidence as text or a JSON-compatible value), questions,
  and state (your continuation data). Each question has a unique identifier and type "choice", "noul" or "score".
  A choice question takes instructions and criteria mapping 2-12 option identifiers to descriptions. Its result is
  data["answers"][question_id], with "choice" and "probabilities". A noul question has true/false criteria and returns
  "probability"; score uses an ordered criteria list and returns "score". Decide what to do next from the result.
- A ranking request has operation "rank", situation, items (at most 32 strings or dicts), instructions and state.
  Its result has "scores" and "order". Ranking also consumes the decision budget.
- Local BM25 uses operation "search", algorithm "bm25", collection "observations", text, limit and state. It returns
  data as a list of records with ref, text and data. SQL uses operation "sql", sql, parameters and state, with data.rows.
- Finish with operation "finish", output and source_refs naming retained retrieved records. Optional signals (at most
  4000 JSON characters) are available to u_apply through context["backend_evidence"]["signals"]. Without decisions,
  ordinary finish(output, refs) also works. Do not invent evidence or use classifications supplied by the benchmark.
- u_apply may return {"prefix": text} within the 1500-character guidance budget. Inspect backend_evidence.output and
  source_refs to ground its guidance; it must not make model calls.

The shared-control ablation contains no u_recall override and therefore makes zero decision calls even when this tool is
available. Optional tools are capabilities, not default workflows. Report tool usage separately from task scores.