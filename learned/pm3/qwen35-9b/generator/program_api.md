## Learnable runtime workflow
Personalization is not restricted to query expansion or retrieval parameters. Within the dataset's registered
runtime API, learn evidence selection, task/request-specific guidance, local search/SQL sequences and stopping.
With the supplied USER_MODEL backbone, custom_recall=True activates u_recall(context) and custom_apply=True
activates u_apply(context); a nonempty helper without its dispatch flag is not executed. Enable only the hooks
you use. For guidance without retrieved evidence, custom_apply may return {'prefix': guidance} using legal
construction-history facts and the public request. Empty hooks remain valid when unnecessary, not mandatory.
Do not change the host's model-call budgets, public task visibility, output schema or data boundaries.
PM3 does not expose official task labels to runtime programs; branch only on visible request/history fields.

## Runtime tools
Only local memory retrieval and the final answer model are available. There is no decision model or decision relay. decide/rank and intermediate model calls are forbidden; their budgets are zero and cannot be enabled by generated code or generation policy.
