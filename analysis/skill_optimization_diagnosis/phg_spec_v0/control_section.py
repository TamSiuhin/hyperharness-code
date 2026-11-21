USER_MODEL = {}


def user_route(request, kind, p_none):
    return {"history": kind != "task", "queries": [request]}


def user_keep(text, app, kind):
    return True


def user_guidance(request, kind, history):
    return ""
