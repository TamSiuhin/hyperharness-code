"""Compact source-labelled construction history for personal code writing."""
import json
import re
from collections import defaultdict


def short(value, limit):
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


def log_weight(text):
    text = text.lower()
    weight = 0
    for phrase, points in (
        ('the user said', 5), ('the user sent a message', 5),
        ('the user replied', 5), ('the user recorded', 3),
        ('the user purchased', 3), ('the user searched', 2),
        ('the user created', 2), ('the user cancelled', 5),
        ('the user declined', 5), ('the user rejected', 5),
        ('the user complained', 5), ('the user tried', 3),
        ('the user browsed', 1), ('the user viewed', 1),
    ):
        if phrase in text:
            weight += points
    if '"' in text or "'" in text:
        weight += 2
    return weight


def render(events, user_cap, assistant_cap, logs_per_session):
    sessions = defaultdict(list)
    for event in events:
        row = event.get('event') or {}
        sessions[str(row.get('session_id') or '?')].append((str(event.get('id') or ''), row))
    output = []
    for session, rows in sessions.items():
        dialogue = []
        logs = []
        prior_assistant = None
        for identifier, row in rows:
            kind = row.get('kind')
            content = row.get('content') or ''
            if kind == 'dialogue':
                role = row.get('role')
                if role == 'assistant':
                    prior_assistant = (identifier, content)
                elif role == 'user':
                    item = {'id': identifier, 'user': short(content, user_cap)}
                    # Replies such as "that won't work" need the preceding idea.
                    if prior_assistant and (re.search(r'\b(this|that|it|idea|suggestion|approach|method|plan|instead|but|yes|no|good|suit|work|prefer|try)\b', content.lower()) or len(content) < 125):
                        item['preceding_assistant'] = short(prior_assistant[1], assistant_cap)
                    dialogue.append(item)
                    prior_assistant = None
            elif kind == 'log':
                logs.append((log_weight(content), identifier, row))
        chosen = sorted(logs, key=lambda item: (item[0], item[2].get('date') or ''), reverse=True)[:logs_per_session]
        chosen = sorted(chosen, key=lambda item: item[2].get('date') or '')
        compact_logs = [[identifier, row.get('date') or '', short(row.get('content'), 160)] for _, identifier, row in chosen]
        output.append({'session': session, 'user_dialogue': dialogue, 'selected_activity': compact_logs})
    return {'events': len(events), 'sessions': output}


def digest(events, max_chars=155000):
    # Shrink content before dropping any user conversation.
    for user_cap, assistant_cap, logs_per_session in ((350, 180, 6), (280, 145, 5), (230, 110, 4), (180, 80, 3), (130, 55, 2)):
        result = render(events, user_cap, assistant_cap, logs_per_session)
        encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        if len(encoded) <= max_chars:
            return encoded
    for cap in (90, 70, 50, 30):
        result = render(events, cap, 0, 0)
        for session in result['sessions']:
            for turn in session['user_dialogue']:
                turn.pop('preceding_assistant', None)
        encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        if len(encoded) <= max_chars:
            return encoded
    # Keep a valid document even for an exceptionally large history.
    for session in result['sessions']:
        session['user_dialogue'] = session['user_dialogue'][-12:]
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'))
