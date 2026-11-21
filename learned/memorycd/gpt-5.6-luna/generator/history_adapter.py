"""Legal source-history examples and a rating-scale summary for construction."""
import json
from collections import Counter

MAX_EVENTS = 96
RECENT_EVENTS = 18
MAX_EVENT_CHARS = 34000


def digest(events):
    ratings = []
    for event in events:
        details = event.get('event') or {}
        if 'product' not in details or 'review' not in details:
            continue
        try:
            rating = float(details.get('rating'))
        except (TypeError, ValueError):
            continue
        if rating.is_integer() and 1 <= rating <= 5:
            ratings.append(int(rating))
    counts = Counter(ratings)
    ordered = sorted(ratings)
    rating_scale = {'count': len(ordered),
                    'counts_1_to_5': [counts.get(value, 0) for value in range(1, 6)],
                    'lower_median': ordered[(len(ordered) - 1) // 2] if ordered else None}
    recent_start = max(0, len(events) - RECENT_EVENTS)
    available = min(recent_start, MAX_EVENTS - min(len(events), RECENT_EVENTS))
    indices = set(range(recent_start, len(events)))
    for index in range(available):
        indices.add(index * (recent_start - 1) // max(1, available - 1))
    selected = []
    for index in sorted(indices):
        event = events[index]
        details = event.get('event') or {}
        if 'product' not in details or 'review' not in details:
            text = str(event.get('text') or json.dumps(details, ensure_ascii=False, sort_keys=True))
            selected.append({'id': event['id'], 'ts': event.get('ts'), 'app': str(event.get('app') or '')[:120],
                'text': text[:480] + (' ... ' + text[-130:] if len(text) > 650 else text[480:])})
            continue
        product = details.get('product') or {}
        review = str(details.get('review') or '')
        selected.append({'id': event['id'], 'ts': event.get('ts'),
            'domain': str(details.get('domain') or event.get('app') or 'unknown')[:120],
            'rating': details.get('rating'), 'review_title': str(details.get('review_title') or '')[:160],
            'product': {'title': str(product.get('title') or '')[:240],
                        'category': str(product.get('category') or '')[:160]},
            'review_excerpt': review[:480] + (' ... ' + review[-130:] if len(review) > 650 else review[480:])})
    recent_ids = {event['id'] for event in events[recent_start:]}
    while len(json.dumps(selected, ensure_ascii=False)) > MAX_EVENT_CHARS:
        older = [event for event in selected if event['id'] not in recent_ids]
        recent = [event for event in selected if event['id'] in recent_ids]
        selected = older[1::2] + recent if older else recent[1::2]
    return json.dumps({'total_history_events': len(events),
        'rating_scale_from_all_source_reviews': rating_scale,
        'selected_events': len(selected),
        'selection': 'Uniform positions plus recent18; raw observations only; no task-specific statistics',
        'events': selected}, ensure_ascii=False)
