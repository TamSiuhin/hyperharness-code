PROGRAM_API_VERSION = 1
BACKBONE = {
    "retrieval": {"mode": "bm25", "top_k": 30, "max_chars": 11000, "max_queries": 3, "query_chars": 2000,
        "recent_rows": 10, "guidance_chars": 1500},
    "gate": {"threshold": 0.5, "situation_chars": 20000, "questions": {"context": {"type": "choice",
        "instructions": "A personal assistant must answer this request for one user. Which kind of user context should it rely on to answer well?",
        "criteria": {
            "none": "No personal history: the request is generic, or using personal details would be intrusive, sycophantic or inappropriate.",
            "evidence": "Specific past records: it needs concrete facts, events, people, messages or items from the user's own history.",
            "profile": "The user's stable preferences, tastes and habits summarized from their history, rather than individual records."}}}},
    "kind": {"questions": {"kind": {"type": "choice",
        "instructions": "What is the user mainly asking a personal assistant to do in this request?",
        "criteria": {
            "rank": "Rank, pick or order a given list of items or options.",
            "refind": "Find or recall something specific from the user's own past activity.",
            "fact": "Answer a factual question about the user's own life, plans, people or past.",
            "draft": "Write, reply to, post or send a message or post on the user's behalf.",
            "advice": "Give advice, ideas or recommendations without a given list of options.",
            "chat": "Respond conversationally to a personal remark, feeling, opinion or claim.",
            "task": "Perform a general task (summarize, plan, explain) that is not about the user's own past."}}}},
    "recent": {
        "self": "json_extract(data, '$.is_self_authored') = 1",
        "chat": "json_extract(data, '$.app') IN ('chatbot', 'ai_studio')",
        "calendar": "json_extract(data, '$.app') = 'calendar'",
        "dm": "json_extract(data, '$.is_dm') = 1"},
    "core_markers": ["## Current user query", "## User query", "## User question", "## The user's actual @ai message",
        "The user asked:"],
}
#@USER_SECTION@#
CHANGE_MANIFEST = {
    "recall": "Shared backbone: asks the decision model whether the request needs personal history (gate) and what kind "
        "of request it is (kind); the user section's user_route decides whether this user's history is used and which "
        "bm25 queries and recent-record view are run, user_keep filters the retrieved records.",
    "apply": "The user section's user_guidance returns request-specific guidance for this user, prepended to the "
        "answer prompt; no guidance when it returns an empty string."}


def remember(context):
    return {}


def guard(context):
    return {}


def words(text):
    lowered = str(text or "").lower()
    for mark in (".", ",", ":", ";", "?", "!", "#", "(", ")", "[", "]", "/", "-", "\n", "*", "\"", "'", "_", "@"):
        lowered = lowered.replace(mark, " ")
    return lowered.split()


def has_any(text, terms):
    lowered = str(text or "").lower()
    for term in terms:
        if str(term).lower() in lowered:
            return True
    return False


def request_core(request):
    text = str(request or "")
    for marker in BACKBONE["core_markers"]:
        if marker in text:
            part = text.split(marker, 1)[1].split("\n## ", 1)[0]
            return part.strip()[:2000]
    return text[:2000]


def finish(output, refs, signals):
    return {"memory_query": {"operation": "finish", "output": output, "source_refs": refs, "signals": signals}}


def decide(situation, questions, state):
    return {"memory_query": {"operation": "decide", "situation": situation, "questions": questions, "state": state}}


def search(text, state):
    settings = BACKBONE["retrieval"]
    state["last"] = "search"
    return {"memory_query": {"operation": "search", "algorithm": settings["mode"], "collection": "observations",
        "text": text, "limit": settings["top_k"], "state": state}}


def recent(name, state):
    state["last"] = "sql"
    sql = ("SELECT ref, text, json_extract(data, '$.app') AS app FROM records WHERE collection = 'observations' AND "
        + BACKBONE["recent"][name] + " ORDER BY json_extract(data, '$.ts') DESC LIMIT " + str(BACKBONE["retrieval"]["recent_rows"]))
    return {"memory_query": {"operation": "sql", "sql": sql, "parameters": [], "state": state}}


def route_for(request, kind, p_none):
    route = user_route(request, kind, p_none) or {}
    settings = BACKBONE["retrieval"]
    queries = []
    for query in route.get("queries") or []:
        text = str(query or "").strip()
        if text and len(queries) < settings["max_queries"]:
            queries.append(text[:8000] if not queries else text[:settings["query_chars"]])
    if not queries:
        queries = [request[:8000]]
    recent_view = route.get("recent")
    if recent_view not in BACKBONE["recent"]:
        recent_view = None
    per_record = route.get("per_record")
    if per_record:
        per_record = min(max(int(per_record), 200), settings["max_chars"])
    return {"history": bool(route.get("history")), "queries": queries, "recent": recent_view, "per_record": per_record}


def collect(state, rows):
    limit = BACKBONE["retrieval"]["max_chars"]
    for row in rows or []:
        remaining = limit - state["size"] - 1
        if remaining <= 0:
            break
        ref = row.get("ref")
        if not ref or ref in state["refs"]:
            continue
        text = str(row.get("text") or "")
        app = row.get("app") or (row.get("data") or {}).get("app") or ""
        if not user_keep(text, str(app), state.get("kind")):
            continue
        if state.get("per_record") and len(text) > state["per_record"]:
            text = text[:state["per_record"]]
        text = text[:remaining]
        state["texts"].append(text)
        state["refs"].append(ref)
        state["size"] = state["size"] + len(text) + 1


def recall(context):
    if context.get("stage") != "context" or not context.get("backend"):
        return {}
    gate, kinds, settings = BACKBONE["gate"], BACKBONE["kind"], BACKBONE["retrieval"]
    backend = context["backend"]
    request = str(context.get("query") or "")
    situation = {"user_request": request[:gate["situation_chars"]]}
    if backend.get("step", 0) == 0:
        return decide(situation, gate["questions"], {"phase": "gate"})
    state = backend.get("state") or {}
    data = (backend.get("result") or {}).get("data")
    if state.get("phase") == "gate":
        none = data["answers"]["context"]["probabilities"]["none"]
        if none > gate["threshold"]:
            return finish("", [], {"p_none": none, "history": False})
        return decide(situation, kinds["questions"], {"phase": "kind", "p_none": none})
    if state.get("phase") == "kind":
        kind = data["answers"]["kind"]["choice"]
        route = route_for(request, kind, state.get("p_none"))
        if not route["history"]:
            return finish("", [], {"p_none": state.get("p_none"), "kind": kind, "history": False})
        state = {"phase": "collect", "kind": kind, "p_none": state.get("p_none"), "queries": route["queries"][1:],
            "recent": route["recent"], "per_record": route["per_record"], "texts": [], "refs": [], "size": 0}
        return search(route["queries"][0], state)
    if state.get("last") == "sql":
        collect(state, (data or {}).get("rows"))
    else:
        collect(state, data)
    if state["queries"] and state["size"] < settings["max_chars"] - 1:
        query = state["queries"][0]
        state["queries"] = state["queries"][1:]
        return search(query, state)
    if state.get("recent") and state["size"] < settings["max_chars"] - 1:
        view = state["recent"]
        state["recent"] = None
        return recent(view, state)
    return finish("\n".join(state["texts"]), state["refs"], {"p_none": state.get("p_none"), "kind": state.get("kind"),
        "history": True})


def apply(context):
    if context.get("stage") != "generation" or "prefix" not in (context.get("base") or {}).get("expressible", []):
        return {}
    signals = (context.get("backend_evidence") or {}).get("signals") or {}
    text = str(user_guidance(str(context.get("query") or ""), signals.get("kind"), bool(signals.get("history"))) or "").strip()
    if not text:
        return {}
    return {"prefix": text[:BACKBONE["retrieval"]["guidance_chars"]]}
