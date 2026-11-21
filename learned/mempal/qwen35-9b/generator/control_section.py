USER_MODEL = {}

def user_route(request, kind, p_none):
    return {'history': True, 'queries': [request]}

def user_keep(text, app, kind):
    return True

def user_guidance(request, kind, evidence):
    return ''

def u_recall(context):
    return {}

def u_apply(context):
    return {}
