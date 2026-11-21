PROGRAM_API_VERSION = 1
BACKBONE = {'retrieval': {'top_k': 10, 'max_chars': 20000, 'max_queries': 3, 'query_chars': 2000, 'recent_rows': 10, 'guidance_chars': 1500}, 'recent': {'self': "json_extract(data, '$.is_self_authored') = 1", 'chat': "json_extract(data, '$.app') IN ('chatbot', 'ai_studio')", 'calendar': "json_extract(data, '$.app') = 'calendar'", 'dm': "json_extract(data, '$.is_dm') = 1"}, 'core_markers': ['## Current user query', '## User query', '## User question', "## The user's actual @ai message", 'The user asked:']}

#@USER_SECTION@#
CHANGE_MANIFEST = {'recall': 'User code controls retrieval and optional workflow overrides; the default backbone makes no model decisions.', 'apply': 'User code provides request-specific guidance using actual retrieval availability.'}

def remember(context):
    return {}

def guard(context):
    return {}

def words(text):
    lowered = str(text or '').lower()
    for mark in ('.', ',', ':', ';', '?', '!', '#', '(', ')', '[', ']', '/', '-', '\n', '*', '"', "'", '_', '@'):
        lowered = lowered.replace(mark, ' ')
    return lowered.split()

def has_any(text, terms):
    lowered = str(text or '').lower()
    for term in terms:
        if str(term).lower() in lowered:
            return True
    return False

def finish(output, refs):
    return {'memory_query': {'operation': 'finish', 'output': output, 'source_refs': refs}}

def search(text, state):
    state['last'] = 'search'
    return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': text, 'limit': state['top_k'], 'state': state}}

def recent(name, state):
    state['last'] = 'sql'
    sql = "SELECT ref, text, json_extract(data, '$.app') AS app FROM records WHERE collection = 'observations' AND " + BACKBONE['recent'][name] + " ORDER BY json_extract(data, '$.ts') DESC LIMIT " + str(BACKBONE['retrieval']['recent_rows'])
    return {'memory_query': {'operation': 'sql', 'sql': sql, 'parameters': [], 'state': state}}

def balanced_query(request):
    if '\nOptions:\n' not in request:
        return request[:8000]
    header, options = request.split('\nOptions:\n', 1)
    sections = {}
    current = ''
    for line in options.split('\n'):
        if len(line) > 2 and line[0] in 'ABCDE' and (line[1] == ':'):
            current = line[0]
            sections[current] = line[2:]
        elif current and (not line.startswith('Return only')):
            sections[current] = sections[current] + ' ' + line
    result = header[:600]
    for label in 'ABCDE':
        result = result + '\n' + label + ': ' + sections.get(label, '')[:900]
    return result

def apply(context):
    if context.get('stage') != 'generation' or 'prefix' not in (context.get('base') or {}).get('expressible', []):
        return {}
    if USER_MODEL.get('custom_apply'):
        return u_apply(context)
    evidence = context.get('backend_evidence') or {'output': '', 'source_refs': []}
    if not evidence.get('source_refs'):
        return {}
    text = str(user_guidance(str(context.get('query') or ''), context.get('task_kind'), evidence) or '').strip()
    return {'prefix': text[:BACKBONE['retrieval']['guidance_chars']]} if text else {}

def collect(state, rows):
    rank = 0
    for row in rows or []:
        rank = rank + 1
        ref = row.get('ref')
        if not ref:
            continue
        text = str(row.get('text') or '')
        app = row.get('app') or (row.get('data') or {}).get('app') or ''
        if not user_keep(text, str(app), state['kind']):
            continue
        if ref not in state['candidates']:
            if len(state['candidates']) >= 32:
                continue
            state['candidates'][ref] = {'score': 0, 'text': text[:state['per_record']]}
        state['candidates'][ref]['score'] = state['candidates'][ref]['score'] + 1 / (60 + rank)

def selected(state):
    ranked = []
    for ref in state['candidates']:
        value = state['candidates'][ref]
        ranked.append([value['score'], ref, value['text']])
    texts = []
    refs = []
    size = 0
    for candidate in sorted(ranked, reverse=True):
        remaining = BACKBONE['retrieval']['max_chars'] - size - 1
        if remaining < 200:
            break
        text = candidate[2][:remaining]
        texts.append(text)
        refs.append(candidate[1])
        size = size + len(text) + 1
    return finish('\n'.join(texts), refs)

def legacy_recall(context):
    backend = context['backend']
    if backend.get('step', 0) == 0:
        return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': context.get('query', '')[:8000], 'limit': 10}}
    texts = []
    refs = []
    size = 0
    for row in backend.get('result', {}).get('data', []):
        remaining = 6000 - size - 1
        if remaining <= 0:
            break
        text = row.get('text', '')[:remaining]
        texts.append(text)
        refs.append(row['ref'])
        size = size + len(text) + 1
    return finish('\n'.join(texts), refs)

def request_core(request):
    text = str(request or '')
    if 'User query:\n' in text:
        text = text.split('User query:\n', 1)[1]
    return text.split('\nOptions:\n', 1)[0].strip()[:2000]

def route_for(request, kind):
    route = user_route(request, kind, None) or {}
    settings = BACKBONE['retrieval']
    core = request_core(request)
    queries = [core if core else request[:2000]]
    for query in route.get('queries') or []:
        text = str(query or '').strip()
        if text and text != request and ('\nOptions:\n' not in text) and (text not in queries) and (len(queries) < settings['max_queries']):
            queries.append(text[:settings['query_chars']])
    per_record = settings['max_chars']
    if route.get('per_record'):
        per_record = min(max(int(route['per_record']), 1), settings['max_chars'])
    return {'history': bool(route.get('history', True)), 'queries': queries, 'per_record': per_record, 'top_k': retrieval_top_k(route.get('top_k', USER_MODEL.get('top_k', settings['top_k'])))}

def recall(context):
    if context.get('stage') != 'context' or not context.get('backend'):
        return {}
    if USER_MODEL.get('custom_recall'):
        return u_recall(context)
    backend = context['backend']
    if backend.get('step', 0) == 0:
        route = route_for(str(context.get('query') or ''), context.get('task_kind'))
        if not route['history']:
            return finish('', [])
        state = {'queries': route['queries'][1:], 'kind': context.get('task_kind'), 'texts': [], 'refs': [], 'size': 0, 'per_record': route['per_record'], 'top_k': route['top_k']}
        return search(route['queries'][0][:8000], state)
    state = backend.get('state') or {}
    for row in (backend.get('result') or {}).get('data') or []:
        if len(state['refs']) >= 64:
            break
        remaining = BACKBONE['retrieval']['max_chars'] - state['size'] - 1
        if remaining <= 0:
            break
        ref = row.get('ref')
        text = str(row.get('text') or '')
        app = row.get('app') or (row.get('data') or {}).get('app') or ''
        if not ref or ref in state['refs'] or (not user_keep(text, str(app), state['kind'])):
            continue
        text = text[:min(remaining, state['per_record'])]
        state['texts'].append(text)
        state['refs'].append(ref)
        state['size'] = state['size'] + len(text) + 1
    if state['queries'] and state['size'] < BACKBONE['retrieval']['max_chars'] - 1:
        query = state['queries'][0]
        state['queries'] = state['queries'][1:]
        return search(query, state)
    return finish('\n'.join(state['texts']), state['refs'])

def retrieval_top_k(value):
    text = str(value).strip()
    if not text or len(text) > 9:
        return BACKBONE['retrieval']['top_k']
    for character in text:
        if character not in '0123456789':
            return BACKBONE['retrieval']['top_k']
    return min(max(int(text), 1), 64)
