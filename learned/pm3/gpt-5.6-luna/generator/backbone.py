PROGRAM_API_VERSION = 1
BACKBONE = {'retrieval': {'top_k': 30, 'max_chars': 11000, 'max_queries': 3, 'query_chars': 2000, 'recent_rows': 10, 'guidance_chars': 1500}, 'recent': {'self': "json_extract(data, '$.is_self_authored') = 1", 'chat': "json_extract(data, '$.app') IN ('chatbot', 'ai_studio')", 'calendar': "json_extract(data, '$.app') = 'calendar'", 'dm': "json_extract(data, '$.is_dm') = 1"}, 'core_markers': ['## Current user query', '## User query', '## User question', "## The user's actual @ai message", 'The user asked:']}

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

def request_core(request):
    text = str(request or '')
    for marker in BACKBONE['core_markers']:
        if marker in text:
            return text.split(marker, 1)[1].split('\n## ', 1)[0].strip()[:2000]
    return text[:2000]

def finish(output, refs):
    return {'memory_query': {'operation': 'finish', 'output': output, 'source_refs': refs}}

def search(text, state):
    state['last'] = 'search'
    return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': text, 'limit': state['top_k'], 'state': state}}

def refind_query(request):
    core = request_core(request)
    lowered = core.lower()
    if not has_any(lowered, ['find it', 'find the post', 'locate it', 'locate the post', 'which app']) or not has_any(lowered, ['post', 'video', 'item', 'clip']):
        return ''
    for phrase in ['post about ', 'video about ', 'clip about ', 'item about ', 'post on ', 'video on ', 'clip on ']:
        if phrase in lowered:
            topic = lowered.split(phrase, 1)[1].split('\n', 1)[0].split(' on one of ', 1)[0].split(' in their ', 1)[0].split(' that ', 1)[0]
            topic = topic.strip(' *#.,:;!?"\'')
            if topic and len(topic) <= 100 and len(topic.split()) <= 8:
                return topic
    return ''

def recent(name, state):
    state['last'] = 'sql'
    sql = "SELECT ref, text, json_extract(data, '$.app') AS app FROM records WHERE collection = 'observations' AND " + BACKBONE['recent'][name] + " ORDER BY json_extract(data, '$.ts') DESC LIMIT " + str(BACKBONE['retrieval']['recent_rows'])
    return {'memory_query': {'operation': 'sql', 'sql': sql, 'parameters': [], 'state': state}}

def engagement(kind, state):
    state['last'] = 'sql'
    kinds = {3: "('explicit_positive')", 2: "('implicit_positive')", 1: "('explicit_negative','implicit_negative')"}[kind]
    limit = 10 if kind > 1 else 12
    sql = ("SELECT ref, '[' || coalesce(json_extract(text, '$.source_interaction_type'), '') || ' ' || "
        "coalesce(json_extract(data, '$.ts'), '') || ' ' || "
        "coalesce(json_extract(data, '$.app'), '') || '] ' || "
        "coalesce(json_extract(text, '$.interaction_format.action_label'), '') || ' | ' || "
        "substr(coalesce(json_extract(text, '$.content.caption'), "
        "json_extract(text, '$.content.title'), json_extract(text, '$.content.text'), "
        "json_extract(text, '$.content.overall_description'), ''), 1, 210) || ' | ' || "
        "substr(coalesce(json_extract(text, '$.source_hashtags'), ''), 1, 150) AS text, "
        "json_extract(data, '$.app') AS app FROM records WHERE collection = 'observations' "
        "AND json_extract(data, '$.app') IN ('instagram','facebook','threads') "
        "AND coalesce(json_extract(data, '$.is_self_authored'), 0) = 0 "
        "AND coalesce(json_extract(data, '$.is_dm'), 0) = 0 "
        "AND json_extract(text, '$.source_interaction_type') IN " + kinds + " "
        "ORDER BY json_extract(data, '$.ts') DESC LIMIT " + str(limit))
    return {'memory_query': {'operation': 'sql', 'sql': sql, 'parameters': [], 'state': state}}

def route_for(request):
    route = user_route(request, None, None) or {}
    refind = refind_query(request)
    if refind:
        route = {'history': True, 'queries': [refind], 'top_k': 24, 'per_record': 1100}
    settings = BACKBONE['retrieval']
    queries = []
    for query in route.get('queries') or []:
        text = str(query or '').strip()
        if text and len(queries) < settings['max_queries']:
            queries.append(text[:8000] if not queries else text[:settings['query_chars']])
    if not queries:
        queries = [request[:8000]]
    view = route.get('recent')
    if view not in BACKBONE['recent']:
        view = None
    per_record = route.get('per_record')
    if per_record:
        per_record = min(max(int(per_record), 200), settings['max_chars'])
    ranking = 'personalized social-media content recommendation' in request.lower() and 'rank' in request.lower()
    return {'history': bool(route.get('history')), 'queries': queries, 'recent': view, 'per_record': per_record, 'top_k': retrieval_top_k(route.get('top_k', USER_MODEL.get('top_k', settings['top_k']))), 'engagement': ranking}

def collect(state, rows):
    for row in rows or []:
        # Leave room for each pending search instead of letting the first,
        # often broad, result set consume the entire evidence window.
        if state.get('engagement') == 3:
            ceiling = 2500
        elif state.get('engagement') == 2:
            ceiling = 4600
        elif state.get('engagement') == 1:
            ceiling = 6800
        elif state.get('ranked'):
            ceiling = BACKBONE['retrieval']['max_chars'] - 1000 * len(state['queries'])
        else:
            ceiling = BACKBONE['retrieval']['max_chars'] - 2500 * len(state['queries'])
        remaining = ceiling - state['size'] - 1
        if remaining <= 0:
            break
        ref = row.get('ref')
        if not ref or ref in state['refs']:
            continue
        text = str(row.get('text') or '')
        app = row.get('app') or (row.get('data') or {}).get('app') or ''
        if not user_keep(text, str(app), None):
            continue
        if state.get('per_record'):
            text = text[:state['per_record']]
        text = text[:remaining]
        state['texts'].append(text)
        state['refs'].append(ref)
        state['size'] = state['size'] + len(text) + 1

def recall(context):
    if context.get('stage') != 'context' or not context.get('backend'):
        return {}
    if USER_MODEL.get('custom_recall'):
        return u_recall(context)
    backend = context['backend']
    if backend.get('step', 0) == 0:
        route = route_for(str(context.get('query') or ''))
        if not route['history']:
            return finish('', [])
        state = {'queries': route['queries'], 'recent': route['recent'], 'per_record': route['per_record'], 'texts': [], 'refs': [], 'size': 0, 'top_k': route['top_k'], 'engagement': 3 if route['engagement'] else 0, 'ranked': route['engagement']}
        if state['engagement']:
            return engagement(3, state)
        query = state['queries'][0]
        state['queries'] = state['queries'][1:]
        return search(query, state)
    state = backend.get('state') or {}
    data = (backend.get('result') or {}).get('data')
    collect(state, (data or {}).get('rows') if state.get('last') == 'sql' else data)
    if state.get('engagement'):
        state['engagement'] = state['engagement'] - 1
        if state['engagement']:
            return engagement(state['engagement'], state)
    if state['queries'] and state['size'] < BACKBONE['retrieval']['max_chars'] - 1:
        query = state['queries'][0]
        state['queries'] = state['queries'][1:]
        return search(query, state)
    if state.get('recent') and state['size'] < BACKBONE['retrieval']['max_chars'] - 1:
        view = state['recent']
        state['recent'] = None
        return recent(view, state)
    return finish('\n'.join(state['texts']), state['refs'])

def apply(context):
    if context.get('stage') != 'generation' or 'prefix' not in (context.get('base') or {}).get('expressible', []):
        return {}
    if USER_MODEL.get('custom_apply'):
        return u_apply(context)
    history = bool((context.get('backend_evidence') or {}).get('source_refs'))
    text = str(user_guidance(str(context.get('query') or ''), None, history) or '').strip()
    return {'prefix': text[:BACKBONE['retrieval']['guidance_chars']]} if text else {}

def retrieval_top_k(value):
    text = str(value).strip()
    if not text or len(text) > 9:
        return BACKBONE['retrieval']['top_k']
    for character in text:
        if character not in '0123456789':
            return BACKBONE['retrieval']['top_k']
    return min(max(int(text), 1), 64)
