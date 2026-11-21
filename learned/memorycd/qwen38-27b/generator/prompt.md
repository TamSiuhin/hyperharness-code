# Write this user's executable personal section

Return only USER_MODEL plus user_/u_ functions or helper data. The host checks the assembled program against the shared backbone. Do not redefine PROGRAM_API_VERSION, BACKBONE, remember, recall, apply, guard, or shared helpers. Shared helper functions may be called. At least one non-template LLM-written function must be reachable from runtime hooks and grounded in this user's history. Use the supplied interface signatures. No intermediate model calls.

## Runtime contract
<<DATASET_RUNTIME>>

## Language and limits
<<LANGUAGE>>
At most <<MAX_CHARS>> characters and <<MAX_NODES>> AST nodes. Use plain def functions with positional parameters: no imports, classes, defaults, annotations, decorators, comprehensions, lambdas, while loops, filesystem or network access. Never pass functions as values.

## Shared backbone
```python
<<BACKBONE>>
```

## Personal section interface
```python
<<CONTROL_SECTION>>
```

## This user's legal construction history
<<DIGEST>>

## Task-specific program design
Keep the baseline retrieval route: `user_route` returns `{'history': True, 'queries': [request], 'top_k': 10}` and `user_keep` returns True. Write a compact `user_guidance(request, kind, history)` with a separate branch for each of the four task kinds, grounded in the rating distribution and examples above. This function is called by apply for every request. Keep all guidance under 1500 characters.

For item_ranking, use observed product categories and substantive preferences, including negative evidence. Ask the answer model to compare every offered title; generic shopping virtues cannot determine the user's preference. For rating_prediction, calibrate to the user's actual rating frequency and threshold for each star, not a universal default; category associations are weak unless several observations support them. For review_summarization, match the user's own review-title pattern and length. The title must capture the supplied review's most distinctive praise or complaint and rating; do not merely repeat the product name. In the summarization guidance, favor the review body's own distinctive words when they form a natural short title; avoid replacing those words with generic synonyms. Use the user's past titles for style, not as a source of claims about this product. For review_generation, match the user's typical voice, length and level of detail from the examples and supplied rating. Do not invent firsthand defects, shipping details or use cases absent from the request or retrieved evidence. Avoid long generic reviews.

Write executable behavior, not only static profile values: condition the guidance on `kind`, and where appropriate on request content and history availability. Keep instructions succinct and do not copy review examples verbatim into output guidance. Do not embed evaluation requests or answers; only legal construction history is available here.

Return only the personal section in one Python code block, not the backbone.
