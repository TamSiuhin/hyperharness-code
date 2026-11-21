"""Compact, source-grounded evidence for four MemoryCD tasks."""
import collections
import json
import re
import statistics

MAX_CHARS = 33000


def clean(value, limit):
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    return text[:limit]


def spread(rows, n):
    if len(rows) <= n:
        return rows
    if n == 1:
        return [rows[-1]]
    return [rows[i * (len(rows) - 1) // (n - 1)] for i in range(n)]


def digest(events):
    records = []
    for event in events:
        detail = event.get('event') or {}
        if not detail.get('product') and event.get('text'):
            try:
                detail = json.loads(event['text'])
            except (ValueError, TypeError):
                continue
        product = detail.get('product') or {}
        if not product or not detail.get('review'):
            continue
        try:
            rating = int(round(float(detail.get('rating'))))
        except (ValueError, TypeError):
            continue
        if rating not in (1, 2, 3, 4, 5):
            continue
        records.append({'domain': clean(detail.get('domain') or event.get('app'), 60),
                        'category': clean(product.get('category'), 90),
                        'product': clean(product.get('title'), 150),
                        'rating': rating,
                        'title': clean(detail.get('review_title'), 130),
                        'review_words': len(str(detail.get('review') or '').split()),
                        'review': clean(detail.get('review'), 850)})
    counts = collections.Counter(x['rating'] for x in records)
    domains = collections.defaultdict(list)
    categories = collections.defaultdict(list)
    for x in records:
        domains[x['domain']].append(x['rating'])
        if x['category']:
            categories[x['category']].append(x['rating'])
    title_lengths = [len(x['title'].split()) for x in records if x['title']]
    review_lengths = [x['review_words'] for x in records]
    result = {'total_reviews': len(records),
              'ratings': {str(r): counts[r] for r in range(1, 6)},
              'mean_rating': round(statistics.mean(x['rating'] for x in records), 2) if records else None,
              'median_title_words': round(statistics.median(title_lengths)) if title_lengths else None,
              'median_review_words': round(statistics.median(review_lengths)) if review_lengths else None,
              'domains': {k: {'count': len(v), 'mean_rating': round(statistics.mean(v), 2)}
                          for k, v in sorted(domains.items(), key=lambda item: -len(item[1]))},
              'categories': [{'name': k, 'count': len(v), 'mean_rating': round(statistics.mean(v), 2)}
                             for k, v in sorted(categories.items(), key=lambda item: -len(item[1]))[:20] if len(v) >= 3],
              'examples_by_rating': {}}
    for rating in range(1, 6):
        group = [x for x in records if x['rating'] == rating]
        examples = []
        for x in spread(group, 9):
            examples.append({'domain': x['domain'], 'product': x['product'], 'title': x['title'],
                             'review_excerpt': x['review'][:500]})
        result['examples_by_rating'][str(rating)] = examples
    text = json.dumps(result, ensure_ascii=False)
    if len(text) > MAX_CHARS:
        for examples in result['examples_by_rating'].values():
            for x in examples:
                x['review_excerpt'] = x['review_excerpt'][:250]
        text = json.dumps(result, ensure_ascii=False)
    return text[:MAX_CHARS]
