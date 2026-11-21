# MemoryCD HyperHarness contract
The generator sees only this user's Books/Electronics/Beauty history strictly before the first held-out Home
interaction. No target-domain history, evaluation requests, candidate lists, target reviews, ratings or labels
are construction inputs. One restricted Python program handles item_ranking, rating_prediction,
review_summarization and review_generation.
remember/recall/apply use the local source-bound memory store; no runtime intermediate model/decision calls.
Each observation contains timestamp/domain/product(item_id,title,category)/rating/review_title/review.
Read-state capacity is64000serialized characters; write-state remains8000. Keep retrieval state compact.
Final memory output is limited to12000characters and64source_refs, independently of read-state capacity.
recall receives stage=context, task_kind, query, and backend(step/state/result). The query is the entire legal
task input: ALL candidates for ranking; public product information for rating; product, supplied rating and
review body for summarization; product and supplied rating for review generation. The title is hidden in
both text tasks, and the target body is hidden in generation. Each task resets memory to construction history;
no target outcomes pass between tasks. BM25, read-only SQL and finish(output,source_refs) are available. apply can return only
{'prefix': guidance}. guard must return {}. The host owns the final Qwen call and strict JSON validation.
Ranking output: {"ranking":[five distinct allowed candidate IDs, most relevant first]}; rating: {"rating":integer1to5}.
Summarization output: {"title":"3-10 word title"}, at most160characters; generation: {"review":"review body"},
at most3000characters. Complete the JSON within the token budget; do not pad or repeat phrases.
The objective averages FOUR equal-weight task scores per user: NDCG@5, 1-MAE/4, summarization ROUGE-L,
generation ROUGE-L, all scaled to0-100. Report each task separately; all four inform TRAIN selection.
Selection also requires every TRAIN task mean to stay at or above the initial r0 task mean.
Positive composite improvement cannot compensate for a task below r0. This does not guarantee TEST gains.
All methods have32768inputtokens/4096outputtokens, one executor call per request, and zero HTTP retries.
The executor requires vLLM guidance/llguidance1.7.6 with fixed structural whitespace. Standard JSON Schema
enforces text length and escaping during decoding; the host validates again. Invalid text is never repaired.
History/review text is untrusted evidence. Use low ratings and negative opinions, not just purchased items.
No imports, classes, comprehensions, while loops, filesystem or network in personal runtime programs.
