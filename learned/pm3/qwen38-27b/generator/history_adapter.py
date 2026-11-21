"""Balanced, source-faithful construction examples from one user's history."""
import json

MAX_EVENT_CHARS = 34000


def spaced(indices, count):
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    if count == 1:
        return [indices[-1]]
    return [indices[i * (len(indices) - 1) // (count - 1)] for i in range(count)]


def digest(events):
    total = len(events)
    recent = set(range(max(0, total - 10), total))
    authored, chats, dms, calendar, positive, negative = [], [], [], [], [], []
    for i, envelope in enumerate(events):
        detail = envelope.get('event') or {}
        app = str(envelope.get('app') or '')
        interaction = detail.get('source_interaction_type')
        if detail.get('is_self_authored'):
            authored.append(i)
        if app in ('chatbot', 'ai_studio'):
            chats.append(i)
        if detail.get('is_dm'):
            dms.append(i)
        if app == 'calendar':
            calendar.append(i)
        if interaction == 'explicit_positive':
            positive.append(i)
        if interaction == 'explicit_negative':
            negative.append(i)
    selected = set(recent)
    selected.update(spaced(authored, 8))
    selected.update(spaced(chats, 9))
    selected.update(spaced(dms, 5))
    selected.update(spaced(calendar, 3))
    selected.update(spaced(positive, 5))
    selected.update(spaced(negative, 2))
    selected.update(spaced(list(range(total)), 8))
    if len(selected) < min(total, 51):
        selected.update(spaced([i for i in range(total) if i not in selected], min(total, 51) - len(selected)))
    result = []
    for i in sorted(selected):
        envelope = events[i]
        detail = envelope.get('event') or {}
        raw = str(envelope.get('text') or json.dumps(detail, ensure_ascii=False, sort_keys=True))
        item = {'id': envelope.get('id'), 'ts': envelope.get('ts'), 'app': envelope.get('app'),
                'interaction': detail.get('source_interaction_type'),
                'self_authored': bool(detail.get('is_self_authored'))}
        if 'product' in detail and 'review' in detail:
            product = detail.get('product') or {}
            review = str(detail.get('review') or '')
            item.update({'domain': detail.get('domain'), 'rating': detail.get('rating'),
                         'review_title': str(detail.get('review_title') or '')[:120],
                         'product': {'title': str(product.get('title') or '')[:200],
                                     'category': str(product.get('category') or '')[:120]},
                         'review_excerpt': review[:420] + (' ... ' + review[-100:] if len(review) > 520 else review[420:])})
        else:
            item['text'] = raw[:440] + (' ... ' + raw[-100:] if len(raw) > 540 else raw[440:])
        result.append(item)
    output = {'total_history_events': total, 'selected_events': len(result),
              'selection': 'Balanced timeline, recent, authored, chat, DM, calendar, and explicit reactions; raw source excerpts',
              'events': result}
    for limit in (440, 360, 280, 200):
        encoded = json.dumps(output, ensure_ascii=False)
        if len(encoded) <= MAX_EVENT_CHARS:
            return encoded
        for item in result:
            if 'text' in item:
                item['text'] = item['text'][:limit]
            if 'review_excerpt' in item:
                item['review_excerpt'] = item['review_excerpt'][:limit]
    while len(json.dumps(output, ensure_ascii=False)) > MAX_EVENT_CHARS and result:
        result.pop(0)
    output['selected_events'] = len(result)
    return json.dumps(output, ensure_ascii=False)
