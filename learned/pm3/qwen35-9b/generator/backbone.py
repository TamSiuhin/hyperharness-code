PROGRAM_API_VERSION = 1
BACKBONE = {'retrieval': {'top_k': 30, 'max_chars': 11000, 'max_queries': 3, 'query_chars': 2000, 'recent_rows': 10, 'guidance_chars': 1500}, 'recent': {'self': "json_extract(data, '$.is_self_authored') = 1", 'chat': "json_extract(data, '$.app') IN ('chatbot', 'ai_studio')", 'calendar': "json_extract(data, '$.app') = 'calendar'", 'all': "1 = 1", 'dm': "json_extract(data, '$.is_dm') = 1"}, 'core_markers': ['## Current user query', '## User query', '## User question', "## The user's actual @ai message", 'The user asked:']}

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

def refind_topic(request):
    text = str(request or '')
    marker = 'vaguely remembers seeing a post about **'
    lowered = text.lower()
    if marker not in lowered:
        return ''
    position = len(lowered.split(marker, 1)[0])
    tail = text[position + len(marker):]
    return tail.split('**', 1)[0].strip()[:100]

def finish(output, refs):
    return {'memory_query': {'operation': 'finish', 'output': output, 'source_refs': refs}}

def search(text, state):
    state['last'] = 'search'
    return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': text, 'limit': state['top_k'], 'state': state}}

def recent(name, state):
    state['last'] = 'sql'
    size = state.get('per_record') or 2200
    if size > 2200:
        size = 2200
    sql = "SELECT ref, substr(text, 1, " + str(size) + ") AS text, json_extract(data, '$.app') AS app FROM records WHERE collection = 'observations' AND " + BACKBONE['recent'][name] + " ORDER BY json_extract(data, '$.ts') DESC LIMIT " + str(BACKBONE['retrieval']['recent_rows'])
    return {'memory_query': {'operation': 'sql', 'sql': sql, 'parameters': [], 'state': state}}

def refind_details(rows, state):
    refs = []
    for row in rows or []:
        ref = row.get('ref')
        if ref and ref not in refs and len(refs) < 12:
            refs.append(ref)
    if not refs:
        return finish('', [])
    state['refind_refs'] = refs
    fields = "'[source app: ' || coalesce(json_extract(data, '$.app'), '') || '] ' || "
    fields = fields + "'creator: ' || coalesce(json_extract(text, '$.content.metadata.creator_handle'), json_extract(text, '$.author_id'), '') || '; ' || "
    fields = fields + "'title: ' || coalesce(json_extract(text, '$.content.title'), '') || '; ' || "
    fields = fields + "'caption: ' || coalesce(substr(json_extract(text, '$.content.caption'), 1, 500), '') || '; ' || "
    fields = fields + "'description: ' || coalesce(substr(json_extract(text, '$.content.overall_description'), 1, 350), '') || '; ' || "
    fields = fields + "'hashtags: ' || coalesce(json_extract(text, '$.source_hashtags'), '') || '; ' || "
    fields = fields + "'interaction: ' || coalesce(json_extract(text, '$.source_interaction_type'), '')"
    sql = "SELECT ref, " + fields + " AS text, json_extract(data, '$.app') AS app FROM records WHERE collection = 'observations' AND ref IN (" + ','.join(['?'] * len(refs)) + ")"
    state['last'] = 'sql'
    return {'memory_query': {'operation': 'sql', 'sql': sql, 'parameters': refs, 'state': state}}

def route_for(request):
    route = user_route(request, None, None) or {}
    settings = BACKBONE['retrieval']
    topic = refind_topic(request)
    if request.startswith('# Task: Proactive Action Decision'):
        route = {'history': True, 'queries': ['follow up friend alert appointment message'], 'recent': ['all', 'chat', 'calendar'], 'top_k': 2, 'per_record': 750, 'include_app': True}
    elif request.startswith('# Task: recommend ONE thing for this user'):
        route['history'] = True
        route['recent'] = ['chat', 'all']
        route['per_record'] = 850
        route['top_k'] = 3
        route['include_app'] = True
    if topic:
        route = {'history': True, 'queries': [topic], 'top_k': 12, 'per_record': 1100, 'include_app': False, 'refind': True}
    queries = []
    for query in route.get('queries') or []:
        text = str(query or '').strip()
        if text and len(queries) < settings['max_queries']:
            queries.append(text[:8000] if not queries else text[:settings['query_chars']])
    if not queries:
        queries = [request[:8000]]
    view = route.get('recent')
    views = []
    if str(view).startswith('['):
        for name in view:
            if name in BACKBONE['recent'] and len(views) < 3:
                views.append(name)
    elif view in BACKBONE['recent']:
        views.append(view)
    view = views
    per_record = route.get('per_record')
    if per_record:
        per_record = min(max(int(per_record), 200), settings['max_chars'])
    return {'history': bool(route.get('history')), 'queries': queries, 'recent': view, 'per_record': per_record, 'include_app': bool(route.get('include_app')), 'refind': bool(route.get('refind')), 'top_k': retrieval_top_k(route.get('top_k', USER_MODEL.get('top_k', settings['top_k'])))}

def collect(state, rows):
    for row in rows or []:
        remaining = BACKBONE['retrieval']['max_chars'] - state['size'] - 1
        if remaining <= 0:
            break
        ref = row.get('ref')
        if not ref or ref in state['refs']:
            continue
        text = str(row.get('text') or '')
        app = row.get('app') or (row.get('data') or {}).get('app') or ''
        if not user_keep(text, str(app), None):
            continue
        if state.get('include_app') and app:
            text = '[source app: ' + str(app) + '] ' + text
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
        state = {'queries': route['queries'][1:], 'recent': route['recent'], 'per_record': route['per_record'], 'include_app': route['include_app'], 'refind': route['refind'], 'texts': [], 'refs': [], 'size': 0, 'top_k': route['top_k']}
        return search(route['queries'][0], state)
    state = backend.get('state') or {}
    data = (backend.get('result') or {}).get('data')
    if state.get('refind') and state.get('last') == 'search':
        state['refind'] = False
        return refind_details(data, state)
    rows = (data or {}).get('rows') if state.get('last') == 'sql' else data
    if state.get('refind_refs'):
        ordered = []
        for ref in state['refind_refs']:
            for row in rows or []:
                if row.get('ref') == ref:
                    ordered.append(row)
                    break
        rows = ordered
        state['refind_refs'] = []
    collect(state, rows)
    if state['queries'] and state['size'] < BACKBONE['retrieval']['max_chars'] - 1:
        query = state['queries'][0]
        state['queries'] = state['queries'][1:]
        return search(query, state)
    if state.get('recent') and state['size'] < BACKBONE['retrieval']['max_chars'] - 1:
        view = state['recent'][0]
        state['recent'] = state['recent'][1:]
        return recent(view, state)
    return finish('\n'.join(state['texts']), state['refs'])

def apply(context):
    if context.get('stage') != 'generation' or 'prefix' not in (context.get('base') or {}).get('expressible', []):
        return {}
    if USER_MODEL.get('custom_apply'):
        return u_apply(context)
    history = bool((context.get('backend_evidence') or {}).get('source_refs'))
    query = str(context.get('query') or '')
    text = str(user_guidance(query, None, history) or '').strip()
    shared = ''
    if query.startswith('# Task: personalized social-media content recommendation'):
        shared = 'Rank every candidate index exactly once, best first. Use the current slate and actual positive and negative evidence; do not treat a viewed or skipped item as a like. '
    elif 'Vague refind' in query:
        shared = 'Name a specific matching source item and its app only when the retrieved record supports both. Avoid substituting a different item with a similar topic. '
    elif 'Current user query' in query or 'Task: produce a chatbot response' in query:
        shared = 'Check factual claims and uncertain personal details independently; do not agree just because the user asks for agreement. If an exact personal detail is unsupported, say so briefly rather than inventing one. '
    combined = (shared + text)[:800]
    return {'prefix': combined} if combined else {}

def retrieval_top_k(value):
    text = str(value).strip()
    if not text or len(text) > 9:
        return BACKBONE['retrieval']['top_k']
    for character in text:
        if character not in '0123456789':
            return BACKBONE['retrieval']['top_k']
    return min(max(int(text), 1), 64)
