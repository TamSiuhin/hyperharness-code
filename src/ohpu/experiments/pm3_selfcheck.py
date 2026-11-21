"""Reward-free self-checks for personalized PM3 harnesses, built from one user's own causal history.

The host derives every probe from the history a generator already sees (events before the user's first evaluated query):
never from official queries or labels, the judge, profiles or other users. A probe at cutoff c stores events[:c]; its
label is what the user did at or after c inside that history (a later positive engagement) or a structural field of the
stored events (hashtags, interaction types). The kinds mirror the construction of official tasks with deterministic
labels:

- slate: the personalized_recommendation slate. The held-out item is the user's next positive engagement after an hour
  anchor; hard negatives are earlier items the user rejected that share a hashtag with it; fillers are earlier items
  without a shared hashtag. Scored by the official graded nDCG@5 (+2 target, +1 filler, -2 hard negative).
- slate_unseen: the same anchor with every candidate taken from the user's later events, so no candidate is in the store.
  It removes the official slate's shortcut (its held-out item is the only candidate absent from the stored history).
- refind: the vague-refind request for a hashtag topic of the stored history; gold is every stored item with that tag.
- geo: the local_recommendation_geo_shift request after a synthetic round trip appended to the history (the user's own
  social events copied with their items removed and another city's location): the answer should name the city current
  at that moment and not the one before. Scored like the official metric.
- pref: the chatbot_personalized_response request: a chat message of the user's that the history links to one of their
  preferences without naming it, with earlier evidence of that preference stored. The executor context should hold that
  preference among few others and the answer should use it.
- syco: the over_personalization_sycophancy request, built like the official builder (build_sycophancy_probes): after a
  chatbot session of the stored history, the user's next message is a fact, memory or value trap that the registered
  executor writes with the official generation prompt; the same executor judges the answer's resistance (0-10) with the
  official judge prompt. Only an executor answer scores (live kinds); no official query, label or judge is used.

Probe ids split deterministically into dev (may be shown to a generator) and audit (host only). run_probes executes a
harness through the Runner path of the official bridge: with the stub client it measures what recall exposes (no model
call); with an executor client it also scores the answer (deterministically, or with the executor as judge for live
kinds). A generator's sandbox cannot reach the executor, so its checker asks the host for live results through files
(request_live / serve_live).
"""
from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ohpu.experiments import pm3_official as official

VERSION = "pm3-selfcheck-v16"
# Seeds probe sampling: unchanged since v13, so the deterministic kinds draw the same probes as v13/v14 did.
SEED_LABEL = "pm3-selfcheck-v13"
SOCIAL_APPS = ("instagram", "facebook", "threads")
CHAT_APPS = ("chatbot", "ai_studio")
POSITIVE = frozenset({"explicit_positive", "implicit_positive"})
NEGATIVE = frozenset({"explicit_negative", "implicit_negative"})
SLATE_SIZE, HARD_NEGATIVES, MIN_HARD_NEGATIVES = 16, 7, 2
# Official personalized_recommendation anchoring (evaluation/tasks/personalized_recommendation.py).
ANCHOR_HOURS, ANCHOR_WINDOW_SECONDS, OFFICIAL_ANCHORS = (5, 8, 11, 14, 17, 20, 23), 3 * 3600, 56
APP_ORDER = {"instagram": 0, "facebook": 1, "threads": 2}
# Official requests arrive with up to about this many stored events (5 to 40 times a construction history).
SCALE_EVENTS = 4500
# Synthetic items spread over a history so that slate probes see a store of official size (padded_history).
PAD_EVENTS, PAD_SEED = 4000, 7
GRADES = {"target": 2.0, "filler": 1.0, "hard_negative": -2.0}
KINDS = ("slate", "slate_unseen", "refind", "geo", "pref", "syco")
SLATE_KINDS = ("slate", "slate_unseen")
# Kinds scored only from an executor answer judged by the executor: the host runs them (audit, or live checks a generator
# requests through files), never the stub.
LIVE_KINDS = ("syco",)
# Official build_sycophancy_probes: anchor sessions keep their last eight turns; traps of 5 words or more.
SYCO_PROBES, SYCO_TURNS, SYCO_MIN_WORDS = 12, 8, 5
# Live checks: the directory under a generator's workspace, how long a checker waits for the host, and how many
# requests the host serves per session.
LIVE_DIR, LIVE_WAIT_SECONDS, LIVE_REQUESTS = "live", 900, 12
# Official local_recommendation_geo_shift construction: a request 6 h after the first event in the new city, three
# categories per leg, city-agnostic query templates (evaluation/tasks/local_recommendation_geo_shift.py).
GEO_DELTA, GEO_SCENARIOS, GEO_SEGMENT, GEO_SEGMENT_HOURS, GEO_CATEGORIES = 6 * 3600, 3, 24, 30, 3
GEO_QUERIES = {
    "restaurant": ("where should I grab dinner tonight?", "any good restaurants you can recommend?",
        "looking for a solid place to eat \u2014 what would you suggest?"),
    "coffee": ("looking for a solid coffee spot \u2014 got any rec?", "where's a good cafe to set up with a laptop?",
        "any coffee shop recs for the morning?"),
    "activity": ("what should I do this weekend?", "any cool activities you'd suggest?",
        "thinking of something fun for saturday \u2014 ideas?"),
    "sports": ("where can I find a gym or a spot to run?", "any rec for a place to play pickup or hit a class?",
        "looking for a workout option \u2014 what's around?"),
    "entertainment": ("anywhere fun to spend friday night?", "what's good for a night out?",
        "movie / show / something to do tonight \u2014 got ideas?"),
    "bar": ("any good bars to grab drinks?", "where would you send me for a chill drink?", "wanna grab a drink \u2014 recommend somewhere?"),
    "market": ("where can I pick up groceries?", "looking for a good market \u2014 any rec?",
        "need to do a grocery run, where would you suggest?"),
    "coworking": ("looking for a coworking spot to take a call from", "where can I find a quiet place to work for a few hours?",
        "need a workspace for the afternoon \u2014 any rec?"),
    "gas": ("need to fill up \u2014 anywhere reliable around?", "where's a decent gas station to swing by?",
        "looking for a gas stop, what would you recommend?"),
}
GEO_CITIES = (
    ("Denver", "CO", 39.7392, -104.9903), ("Seattle", "WA", 47.6062, -122.3321), ("Boston", "MA", 42.3601, -71.0589),
    ("Austin", "TX", 30.2672, -97.7431), ("Minneapolis", "MN", 44.9778, -93.265), ("San Diego", "CA", 32.7157, -117.1611),
    ("Portland", "OR", 45.5152, -122.6784), ("Kansas City", "MO", 39.0997, -94.5786), ("Salt Lake City", "UT", 40.7608, -111.891),
    ("Pittsburgh", "PA", 40.4406, -79.9959), ("New Orleans", "LA", 29.9511, -90.0715), ("Albuquerque", "NM", 35.0844, -106.6504),
)
# A preference probe credits the held-out preference fully only among at most this many of the user's preferences.
PREF_CONTEXT_ITEMS = 5
# How many sample requests of unprobed task types a revision may send to the executor with a changed context.
PRESERVE_ALLOWED = 2
PREF_STOPWORDS = frozenset("enjoys enjoy content interested interest follows following related about with from that this into "
    "their they them have likes like loves love posts videos media social themed focused centered spanning including such "
    "other more less very the and for fandom culture style".split())
# A returned record exposes its source events only when it is item-level; an aggregate (a profile or topic summary over
# many events) does not identify any one of them.
ITEM_SOURCES = 4
# What a social event was about (the item), as opposed to what the user did with it and when.
ITEM_FIELDS = ("author_id", "content", "content_type", "preferences", "seed_object_id", "source_hashtags", "source_object_id",
    "is_ad", "is_trending", "_feed_react_meta")
SLATE_HEADER = ("# Task: personalized social-media content recommendation\n\nIt's {day} {hour:02d}:00 UTC. The recommendation "
    "system has proposed {count} candidate items. Rank them in the order the user is most likely to engage with, given their "
    "time-masked history below.\n\n## Candidate slate\n")
SLATE_OUTPUT = ("\n\n\n\n## Output\nRespond with ONE fenced ```json block containing the ranked indexes:\n```json\n{\n"
    "  \"ranked_indexes\": [<idx>, <idx>, ...],\n  \"reasoning\": \"<=2 sentences\"\n}\n```")
REFIND = ("# Task T11: Vague refind\n\nThe user vaguely remembers seeing a post about **{topic}** on one of their apps\nbut "
    "can't find it. Locate it in their history and tell them which app + enough\ndetail (title / caption / creator / "
    "hashtags) to identify it.\n\nSearch the user history shown below to locate the post the user is describing. Identify "
    "it precisely \u2014 app + title/caption/creator/hashtags \u2014 so they can recognize it. Do NOT just restate the topic.\n"
    "**Mode**: final-answer-only. Do NOT attempt to call any tools \u2014 none are available in this run. Use only the "
    "ground-truth context above to compose your answer.\n\n\nThis is a SEARCH task \u2014 emit the JSON response below. Do NOT "
    "post anything.\n\n\n\n\n## Output\nRespond with a single fenced ```json``` block:\n```json\n{\"response\": \"your reply "
    "or draft, as plain text\", \"reasoning\": \"<=2 sentences on why\"}\n```")
# The official chatbot-response request (chatbot_personalized_response, local_recommendation_geo_shift and other chat tasks).
CHAT = ("# Task: produce a chatbot response\n\nYou have access to the user's cross-app history (Instagram, Facebook, Threads, "
    "Chatbot, AI Studio) and calendar.\nGive personalized responses when appropriate.\n\n\n\n\n\n## Current user query\n{query}\n\n"
    "## Output\nRespond with a single fenced ```json``` block:\n```json\n{\n  \"response\": \"the assistant's reply, as plain text\",\n"
    "  \"reasoning\": \"optional short rationale (<=2 sentences)\"\n}\n```")


def _native(event: dict) -> dict:
    return event.get("event") or {}


def _tags(event: dict) -> set[str]:
    return {str(tag).lstrip("#").lower() for tag in (_native(event).get("source_hashtags") or []) if tag}


def _interaction(event: dict) -> str:
    return str(_native(event).get("source_interaction_type") or "")


def _object(event: dict) -> str:
    return str(_native(event).get("source_object_id") or event["id"])


def summary(event: dict) -> dict:
    """The official slate projection of one event (title falls back to hashtags, then the app)."""
    native = _native(event)
    content = native.get("content") if isinstance(native.get("content"), dict) else {}
    hashtags = [str(tag) for tag in (native.get("source_hashtags") or [])][:8]
    title = str(content.get("title") or content.get("caption") or "").strip()
    if not title and hashtags:
        title = " ".join(tag.lstrip("#") for tag in hashtags[:3])
    return {"title": (title or f"post on {event['app']}")[:120], "hashtags": hashtags}


def neutral_summary(event: dict, tags: int = 3) -> dict:
    """A candidate rendered without non-personal surface cues: in PM3 only engaged items carry written titles and they carry
    fewer hashtags, so every candidate shows its first ``tags`` distinct lowercase hashtags as both title and tags."""
    chosen = list(dict.fromkeys(str(tag).lstrip("#").lower() for tag in (_native(event).get("source_hashtags") or []) if tag))[:tags]
    return {"title": " ".join(chosen) or f"post on {event['app']}", "hashtags": ["#" + tag for tag in chosen]}


def slate_request(anchor_ts: int, candidates: list[dict]) -> str:
    moment = _dt.datetime.fromtimestamp(anchor_ts, tz=_dt.timezone.utc)
    lines = "\n".join(f"  [{index}] {item['title']} (hashtags: {', '.join(item['hashtags'])})" for index, item in enumerate(candidates))
    return SLATE_HEADER.format(day=moment.strftime("%Y-%m-%d"), hour=moment.hour, count=len(candidates)) + lines + SLATE_OUTPUT


def refind_request(topic: str) -> str:
    return REFIND.replace("{topic}", topic)


def chat_request(query: str) -> str:
    return CHAT.replace("{query}", query)


def _location(event: dict) -> dict | None:
    location = _native(event).get("event_location")
    return location if isinstance(location, dict) and str(location.get("city") or "").strip() else None


def _trip(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """A synthetic round trip after the end of ``events``: GEO_SCENARIOS times, GEO_SEGMENT social events in another city,
    then GEO_SEGMENT back in the home city (the last stored location). Each event copies a random stored social event's
    app, interaction and other fields with its item removed (no content, synthetic id, hashtags resampled from this
    history) and the segment's location. Returns the events and one leg per segment: the time 6 h after the segment's first
    event with the city current then and the one before."""
    located = [location for location in map(_location, events) if location]
    social = [event for event in events if event["app"] in SOCIAL_APPS]
    pool = [str(tag) for event in social for tag in (_native(event).get("source_hashtags") or []) if tag]
    if not located or not social or not pool:
        return [], []
    home = {key: located[-1].get(key) for key in ("city", "region", "country", "lat", "lon")}
    rng = random.Random(f"{SEED_LABEL}:trip:{home['city']}:{len(events)}:{events[-1]['ts']}")
    cities = [city for city in GEO_CITIES if city[0].lower() != str(home["city"]).lower() and city[1] != home.get("region")]
    added, legs = [], []
    start = int(events[-1]["ts"]) + 12 * 3600
    for scenario, (city, region, lat, lon) in enumerate(rng.sample(cities, GEO_SCENARIOS)):
        away = {"city": city, "region": region, "country": home.get("country") or "USA", "lat": lat, "lon": lon}
        for leg, (current, prior) in (("after_shift", (away, home)), ("after_return", (home, away))):
            for index in range(GEO_SEGMENT):
                template = rng.choice(social)
                native = {key: value for key, value in _native(template).items() if key not in ITEM_FIELDS + ("formatted_timestamp",)}
                count = len(_native(template).get("source_hashtags") or [])
                ts = start + index * GEO_SEGMENT_HOURS * 3600 // GEO_SEGMENT
                item = str(rng.randint(10 ** 8, 10 ** 9))
                native.update(content={}, source_object_id=item, seed_object_id=item, source_timestamp=ts,
                    source_hashtags=list(dict.fromkeys(rng.choice(pool) for _ in range(count))),
                    event_location={**current, "precision": "city"})
                added.append({**template, "id": f"{template['id']}~trip{scenario}{leg[6]}{index}", "ts": ts, "event": native,
                    "text": json.dumps(native, ensure_ascii=False)})
            legs.append({"scenario": scenario, "leg": leg, "anchor_ts": start + GEO_DELTA, "current": current, "prior": prior})
            start += (GEO_SEGMENT_HOURS + 6) * 3600
    return added, legs


def extended(events: list[dict], probes: list[dict]) -> list[dict]:
    """The history geo probes are stored over: ``events`` followed by their synthetic round trip (_trip)."""
    return list(events) + _trip(events)[0] if any(probe["kind"] == "geo" for probe in probes) else list(events)


def _named(text: str, term) -> bool:
    term = str(term or "").strip()
    return bool(term) and re.search(r"\b" + re.escape(term) + r"\b", text or "", re.I) is not None


def geo_answer(text: str, probe: dict) -> float:
    """The official geo_shift_correctness: 0 for naming the prior city, 1 for naming the current city or region, else
    0.5; a non-substantive answer scores 0."""
    if len((text or "").split()) < 3 or _named(text, probe["prior_city"]):
        return 0.0
    return 1.0 if _named(text, probe["current_city"]) or _named(text, probe["current_region"]) else 0.5


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z'-]{3,}", (text or "").lower()) if word not in PREF_STOPWORDS}


def _persona_items(events: list[dict]) -> set[str]:
    return {str(item["persona_item"]) for event in events for item in (_native(event).get("preferences") or [])
        if isinstance(item, dict) and item.get("persona_item")}


def _answer_text(answer: str) -> str:
    return str(parse_answer(answer).get("response") or answer or "")


# The official task types each probe kind covers; other sample requests should keep the base harness's behavior.
PROBED_TASKS = {"slate": ("personalized_recommendation", "hidden_persona_recommendation"),
    "slate_unseen": ("personalized_recommendation", "hidden_persona_recommendation"), "refind": ("agentic_vague_refind",),
    "geo": ("local_recommendation_geo_shift",), "pref": ("chatbot_personalized_response",),
    "syco": ("over_personalization_sycophancy",)}


def request_contexts(source: str, document: str, requests: list[dict], *, tool_model_calls: int = 0) -> dict[str, str]:
    """For each sample request ({"task_type", "request"}), the system context the executor receives over ``document``
    with the stub executor (or the runtime violation)."""
    from ohpu.experiments.lamp import SeededLuna
    from ohpu.per_user_adaptation.executor import Runner

    stub = official._StubClient()
    llm = SeededLuna(stub, 0, model=stub.model)
    contexts = {}
    for index, request in enumerate(requests):
        recorder = official._ContextRecorder(llm)
        harness = official._harness("smoke-check", source, document, editable_hooks=True, tool_model_calls=tool_model_calls)
        try:
            Runner(recorder, harness).run(episode_id=f"preserve-{index}", user_id="smoke-check", condition="preserve",
                messages=[{"role": "user", "content": request["request"]}], tools=None,
                task_kind="pm3_native::" + official.HIDDEN_TASK, max_tokens=64, query=request["request"])
            contexts[request["task_type"]] = "\n".join(recorder.systems)
        except ValueError as error:
            contexts[request["task_type"]] = f"violation: {type(error).__name__}: {error}"[:200]
    return contexts


def changed_requests(base: dict[str, str], revision: dict[str, str]) -> list[str]:
    return sorted(task for task in base if revision.get(task) != base[task])


def _extends(base: str, revised: str) -> bool:
    """Whether ``revised`` keeps every word of ``base`` in order, so text was only added."""
    words = iter(revised.split())
    return all(word in words for word in base.split())


def rewritten_requests(base: dict[str, str], revision: dict[str, str]) -> list[str]:
    """Task types whose revised context (request_contexts) removes or alters base text instead of only adding text; a
    runtime violation on either side counts as a rewrite."""
    return sorted(task for task in changed_requests(base, revision) if base[task].startswith("violation: ")
        or str(revision.get(task) or "").startswith("violation: ") or not _extends(base[task], str(revision.get(task) or "")))


def shuffled(events: list[dict], seed: int = 0) -> list[dict]:
    """Control memory for the personal gap: the same events, apps, times and interactions, but every social event's item is
    replaced by a synthetic one: hashtags resampled from this history's overall hashtag frequencies, no written content,
    a synthetic id, and the remaining item fields (author, linked preferences, flags) taken from another social event. The
    user's topics survive on average, while which items they saw, engaged with or skipped does not, so a ranking that owes
    nothing to that item-level memory scores the same over it."""
    rng = random.Random(seed)
    social = [index for index, event in enumerate(events) if event["app"] in SOCIAL_APPS]
    pool = [str(tag) for index in social for tag in (_native(events[index]).get("source_hashtags") or []) if tag]
    donors = social[:]
    rng.shuffle(donors)
    result = list(events)
    for target, source in zip(social, donors):
        native = {key: value for key, value in _native(events[target]).items() if key not in ITEM_FIELDS}
        native.update({key: value for key, value in _native(events[source]).items() if key in ITEM_FIELDS})
        count = len(_native(events[target]).get("source_hashtags") or [])
        native.update(content={}, source_object_id=f"control-{target}", seed_object_id=f"control-{target}",
            source_hashtags=list(dict.fromkeys(rng.choice(pool) for _ in range(count))) if pool else [])
        result[target] = {**events[target], "event": native, "text": json.dumps(native, ensure_ascii=False)}
    return result


def padded_history(events: list[dict], *, seed: int = PAD_SEED, extra: int = PAD_EVENTS) -> list[dict]:
    """The history with ``extra`` synthetic social items spread over its time span, so a probe's store is as large as
    official requests see without inventing any label. Each synthetic item copies a random real social event's app,
    interaction and other fields, with hashtags resampled from this history's hashtags, no content and a synthetic id: a
    ranking that relies on loose matches over a growing store degrades here as it does on official requests."""
    rng = random.Random(seed)
    social = [event for event in events if event["app"] in SOCIAL_APPS]
    pool = [str(tag) for event in social for tag in (_native(event).get("source_hashtags") or []) if tag]
    if not social or not pool:
        return list(events)
    first, last = int(events[0]["ts"]), int(events[-1]["ts"])
    added = []
    for index in range(extra):
        template = rng.choice(social)
        native = {key: value for key, value in _native(template).items() if key not in ITEM_FIELDS + ("formatted_timestamp",)}
        count = len(_native(template).get("source_hashtags") or [])
        ts = rng.randint(first, last)
        item = str(rng.randint(10 ** 8, 10 ** 9))
        native.update(content={}, source_object_id=item, seed_object_id=item, source_timestamp=ts,
            source_hashtags=list(dict.fromkeys(rng.choice(pool) for _ in range(count))))
        added.append({**template, "id": f"{template['id']}~{index}", "ts": ts, "event": native,
            "text": json.dumps(native, ensure_ascii=False)})
    return sorted(list(events) + added, key=lambda event: int(event["ts"]))


def padded_stores(user_id: str, source: str, events: list[dict], probes: list[dict], *, control: bool = False) -> dict[str, str]:
    """Each slate, geo and preference probe's store over padded_history(events) (followed by the synthetic round trip
    for geo probes) at the probe's anchor time (one replay). With ``control``, the padded history is replaced by its
    item-free control (shuffled)."""
    timeline = padded_history(events)
    timeline = shuffled(timeline) if control else timeline
    if any(probe["kind"] == "geo" for probe in probes):
        timeline += _trip(events)[0]
    cuts = {probe["id"]: official.cutoff(timeline, int(probe["anchor_ts"])) for probe in probes if probe["kind"] != "refind"}
    documents = official.documents(user_id, timeline, sorted(set(cuts.values())), source=source) if cuts else {}
    return {identifier: documents[cut] for identifier, cut in cuts.items()}


def scaled_history(events: list[dict], size: int = SCALE_EVENTS) -> list[dict]:
    """A store as large as official requests see, for load checks: the history repeated forward in time with fresh ids."""
    if not events:
        return []
    span = int(events[-1]["ts"]) - int(events[0]["ts"]) + 3600
    result, copy = [], 0
    while len(result) < size:
        for event in events:
            if len(result) >= size:
                break
            result.append({**event, "id": f"{event['id']}~{copy}", "ts": int(event["ts"]) + copy * span})
        copy += 1
    return result


def remember_digest(source: str) -> str:
    hook = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "remember")
    return hashlib.sha256(ast.dump(hook).encode()).hexdigest()


def scale_document(source: str, history: list[dict], cached: dict | None = None, *, replay: bool = False) -> str | None:
    """The program's store over scaled_history(history). A no-op remember hook stores the default observations, which is
    fast; any other hook runs once per event (minutes at this size), so it is replayed only with ``replay`` unless
    ``cached`` ({"remember": remember_digest, "document": str}) holds that hook's store. None when not computed."""
    digest = remember_digest(source)
    if cached and cached.get("remember") == digest:
        return cached["document"]
    if not official.remember_keeps_default(source) and not replay:
        return None
    scaled = scaled_history(history)
    return official.documents("smoke-check", scaled, [len(scaled)], source=source)[len(scaled)]


def scale_check(source: str, document: str, requests: list[dict], *, tool_model_calls: int = 0) -> dict:
    """Run requests ({"task_type", "request"}) over a scaled store with the stub executor. A request that breaks a host
    limit at that size (for example the memory SQL work budget) is a violation."""
    from ohpu.experiments.lamp import SeededLuna
    from ohpu.per_user_adaptation.executor import Runner

    stub = official._StubClient()
    violations = []
    for index, request in enumerate(requests):
        harness = official._harness("smoke-check", source, document, editable_hooks=True, tool_model_calls=tool_model_calls)
        try:
            Runner(SeededLuna(stub, 0, model=stub.model), harness).run(episode_id=f"scale-{index}", user_id="smoke-check",
                condition="scale", messages=[{"role": "user", "content": request["request"]}], tools=None,
                task_kind="pm3_native::" + official.HIDDEN_TASK, max_tokens=64, query=request["request"])
        except ValueError as error:
            violations.append({"task_type": request["task_type"], "error": f"{type(error).__name__}: {error}"[:300]})
    return {"events": SCALE_EVENTS, "requests": len(requests), "violations": violations}


def _slate(events: list[dict], held: int, cutoff: int, rng: random.Random, *, unseen: bool) -> dict | None:
    """One slate for the positive engagement events[held] anchored before it: candidates come from the stored prefix
    (official construction) or, when unseen, from the user's later events, which the store does not contain."""
    pool = [index for index in (range(cutoff, len(events)) if unseen else range(cutoff))
        if index != held and events[index]["app"] in SOCIAL_APPS and _object(events[index]) != _object(events[held])]
    tags = _tags(events[held])
    hard = [index for index in pool if _interaction(events[index]) in NEGATIVE and _tags(events[index]) & tags]
    rng.shuffle(hard)
    hard = hard[:HARD_NEGATIVES]
    if not unseen and len(hard) < HARD_NEGATIVES:  # the official fallback: earlier items sharing a hashtag, oldest first
        chosen = {_object(events[index]) for index in hard}
        for index in pool:
            if len(hard) >= HARD_NEGATIVES:
                break
            if _object(events[index]) not in chosen and _tags(events[index]) & tags:
                hard.append(index)
                chosen.add(_object(events[index]))
    titles = {summary(events[held])["title"].strip().lower()}
    kept = []
    for index in hard:
        title = summary(events[index])["title"].strip().lower()
        if title not in titles:
            titles.add(title)
            kept.append(index)
    if len(kept) < MIN_HARD_NEGATIVES:
        return None
    fillers, used = [], {_object(events[index]) for index in kept}
    options = [index for index in pool if not _tags(events[index]) & tags and _object(events[index]) not in used]
    rng.shuffle(options)
    for index in options:
        if len(fillers) >= SLATE_SIZE - 1 - len(kept):
            break
        title = summary(events[index])["title"].strip().lower()
        if title not in titles and _object(events[index]) not in used:
            titles.add(title)
            used.add(_object(events[index]))
            fillers.append(index)
    rows = [held] + kept + fillers
    order = list(range(len(rows)))
    rng.shuffle(order)
    positions = [rows[slot] for slot in order]
    return {"candidates": [summary(events[index]) for index in positions], "positions": positions,
        "held_out_idx": order.index(0), "hard_negative_idxs": sorted(order.index(slot + 1) for slot in range(len(kept)))}


def _support(events: list[dict], held: int, cutoff: int) -> list[int]:
    """Stored positive engagements (before ``cutoff``) sharing a hashtag with the held-out item."""
    tags = _tags(events[held])
    return [index for index in range(cutoff) if events[index]["app"] in SOCIAL_APPS
        and _interaction(events[index]) in POSITIVE and _tags(events[index]) & tags]


def _slates(events: list[dict], rng: random.Random, limit: int, min_history: int, *, unseen: bool) -> list[dict]:
    anchors = []
    for index, event in enumerate(events):
        if event["app"] in SOCIAL_APPS and _interaction(event) in POSITIVE and _tags(event):
            anchor_ts = int(event["ts"]) - int(event["ts"]) % 3600
            cutoff = official.cutoff(events, anchor_ts)
            if cutoff >= min_history:
                anchors.append((index, anchor_ts, cutoff))
    rng.shuffle(anchors)
    probes, used = [], set()
    for held, anchor_ts, cutoff in anchors:
        if len(probes) >= limit:
            break
        if cutoff in used:  # one slate per anchor hour spreads probes over the history
            continue
        available = len(_support(events, held, cutoff))
        slate = _slate(events, held, cutoff, rng, unseen=unseen) if available else None  # no stored evidence: unmeasurable
        if slate is None:
            continue
        used.add(cutoff)
        kind = "slate_unseen" if unseen else "slate"
        probes.append({"id": f"{kind}-{cutoff:05d}-{held:05d}", "kind": kind, "cutoff": cutoff, "anchor_ts": anchor_ts,
            "held_out_position": held, "support_available": available, "request": slate_request(anchor_ts, slate["candidates"]),
            **slate})
    return probes


def _official_slates(events: list[dict], rng: random.Random, limit: int, min_history: int) -> list[dict]:
    """Slates built inside this history the way the official benchmark builds personalized_recommendation instances:
    anchors at the official hours on the days with the most positive engagements (first and last such day dropped); the
    held-out item is the first positive engagement in the three hours after the anchor (apps in official order, one per
    object and day), so it is often an item the user had already seen or engaged with; hard negatives are earlier rejected
    items sharing a hashtag with it (else earlier items sharing one), fillers earlier items sharing none, titles unique.
    ``limit`` of the feasible slates are kept."""
    social = [index for index, event in enumerate(events) if event["app"] in SOCIAL_APPS]
    days: dict[str, list[int]] = {}
    for index in social:
        if _interaction(events[index]) in POSITIVE:
            day = _dt.datetime.fromtimestamp(int(events[index]["ts"]), tz=_dt.timezone.utc).strftime("%Y-%m-%d")
            days.setdefault(day, []).append(index)
    eligible = sorted(days)
    if len(eligible) >= 3:
        eligible = eligible[1:-1]
    by_volume = sorted(eligible, key=lambda day: -len(days[day]))
    anchors = [(day, hour) for hour in ANCHOR_HOURS for day in by_volume][:OFFICIAL_ANCHORS]
    built, taken = [], {}
    for day, hour in anchors:
        start = int(_dt.datetime.strptime(day, "%Y-%m-%d").replace(hour=hour, tzinfo=_dt.timezone.utc).timestamp())
        used = taken.setdefault(day, set())
        window = sorted((index for index in days[day] if start <= int(events[index]["ts"]) < start + ANCHOR_WINDOW_SECONDS
            and _object(events[index]) not in used), key=lambda index: (APP_ORDER[events[index]["app"]], index))
        if not window:
            continue
        held = window[0]
        used.add(_object(events[held]))
        cutoff = official.cutoff(events, start)
        tags = _tags(events[held])
        prior = [index for index in social if index < cutoff]
        hard = [index for index in prior if _interaction(events[index]) in NEGATIVE and _tags(events[index]) & tags]
        rng.shuffle(hard)
        hard = hard[:HARD_NEGATIVES]
        if len(hard) < HARD_NEGATIVES:
            chosen = {_object(events[index]) for index in hard} | {_object(events[held])}
            for index in prior:
                if len(hard) >= HARD_NEGATIVES:
                    break
                if _object(events[index]) not in chosen and _tags(events[index]) & tags:
                    hard.append(index)
                    chosen.add(_object(events[index]))
        if cutoff < min_history or len(hard) < MIN_HARD_NEGATIVES:
            continue
        titles = {summary(events[held])["title"].strip().lower()}
        kept = []
        for index in hard:
            title = summary(events[index])["title"].strip().lower()
            if title not in titles:
                titles.add(title)
                kept.append(index)
        used_objects = {_object(events[held])} | {_object(events[index]) for index in kept}
        pool = [index for index in prior if _object(events[index]) not in used_objects and not _tags(events[index]) & tags]
        rng.shuffle(pool)
        fillers = []
        for index in pool:
            if len(fillers) >= SLATE_SIZE - 1 - len(kept):
                break
            title = summary(events[index])["title"].strip().lower()
            if title not in titles:
                titles.add(title)
                fillers.append(index)
        rows = [held] + kept + fillers
        order = list(range(len(rows)))
        rng.shuffle(order)
        positions = [rows[slot] for slot in order]
        candidates = [summary(events[index]) for index in positions]
        built.append({"id": f"slate-{cutoff:05d}-{held:05d}", "kind": "slate", "cutoff": cutoff, "anchor_ts": start,
            "held_out_position": held, "support_available": len(_support(events, held, cutoff)),
            "request": slate_request(start, candidates), "candidates": candidates, "positions": positions,
            "held_out_idx": order.index(0), "hard_negative_idxs": sorted(order.index(slot + 1) for slot in range(len(kept)))})
    rng.shuffle(built)
    return built[:limit]


def _refinds(events: list[dict], rng: random.Random, limit: int, min_history: int) -> list[dict]:
    cutoffs = sorted({max(min_history, len(events) * share // 4) for share in (2, 3, 4)} - {0})
    cutoffs = [cutoff for cutoff in cutoffs if cutoff <= len(events)]
    probes = []
    for turn, cutoff in enumerate(cutoffs):
        stored = [index for index in range(cutoff) if events[index]["app"] in SOCIAL_APPS]
        counts = Counter(str(tag).lstrip("#") for index in stored for tag in dict.fromkeys(_native(events[index]).get("source_hashtags") or []) if tag)
        frequent = [topic for topic, count in counts.most_common(24) if count >= 2]
        rng.shuffle(frequent)
        share = limit // len(cutoffs) + (turn < limit % len(cutoffs))
        for topic in frequent[:share]:
            gold = [index for index in stored if topic.lower() in _tags(events[index])]
            digest = hashlib.sha256(topic.encode()).hexdigest()[:8]
            # The official refind target is the most recent post carrying the topic hashtag before the request.
            probes.append({"id": f"refind-{cutoff:05d}-{digest}", "kind": "refind", "cutoff": cutoff, "topic": topic,
                "gold_positions": gold, "target_position": max(gold), "request": refind_request(topic)})
    return probes


def _geos(events: list[dict], rng: random.Random) -> list[dict]:
    """GEO_CATEGORIES probes per leg of the synthetic round trip (_trip), each a different category with one of its
    city-agnostic official query templates. Cutoffs count positions in extended(events, ...)."""
    added, legs = _trip(events)
    probes = []
    for leg in legs:
        for category in rng.sample(sorted(GEO_QUERIES), GEO_CATEGORIES):
            query = rng.choice(GEO_QUERIES[category])
            probes.append({"id": f"geo-{leg['scenario']}-{leg['leg']}-{category}", "kind": "geo", "anchor_ts": leg["anchor_ts"],
                "cutoff": len(events) + sum(int(event["ts"]) < leg["anchor_ts"] for event in added), "category": category,
                "query": query, "request": chat_request(query), "current_city": leg["current"]["city"],
                "current_region": leg["current"]["region"], "prior_city": leg["prior"]["city"],
                "prior_region": leg["prior"]["region"]})
    return probes


def _prefs(events: list[dict], rng: random.Random, limit: int, min_history: int) -> list[dict]:
    """Chat messages of the user's that the history links to one of their preferences (a user turn's embeds_pref_idx)
    without naming it (no keyword of the preference in the message), when an earlier event already shows that preference.
    One probe per chat event, at most two per preference; the store holds the events before that chat."""
    seen: Counter = Counter()
    candidates = []
    for index, event in enumerate(events):
        native = _native(event)
        linked = [item for item in (native.get("preferences") or []) if isinstance(item, dict)]
        if event["app"] in CHAT_APPS and index >= min_history:
            for turn in native.get("conversation") or []:
                if not isinstance(turn, dict) or turn.get("role") != "user":
                    continue
                text = " ".join(str(turn.get("content") or "").split())
                if not 8 <= len(text) <= 600:
                    continue
                for position in turn.get("embeds_pref_idx") or []:
                    item = linked[position - 1] if isinstance(position, int) and 1 <= position <= len(linked) else {}
                    persona = str(item.get("persona_item") or "")
                    if persona and seen[persona] and not _words(persona + " " + str(item.get("category") or "")) & _words(text):
                        candidates.append((index, text, persona, str(item.get("category") or "")))
        for item in linked:
            if item.get("persona_item"):
                seen[str(item["persona_item"])] += 1
    rng.shuffle(candidates)
    probes, events_used, per_item = [], set(), Counter()
    for index, text, persona, category in candidates:
        if len(probes) >= limit or index in events_used or per_item[persona] >= 2:
            continue
        events_used.add(index)
        per_item[persona] += 1
        digest = hashlib.sha256(text.encode()).hexdigest()[:8]
        probes.append({"id": f"pref-{index:05d}-{digest}", "kind": "pref", "cutoff": index, "anchor_ts": int(events[index]["ts"]),
            "query": text, "request": chat_request(text), "persona_item": persona, "category": category})
    return probes


def _official(name: str):
    """A module of the pinned official evaluator (prompt templates only; nothing here reads official data)."""
    from ohpu.experiments import pm3_proxy

    return pm3_proxy.official_module(name)


def _json_object(text: str) -> dict:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _syco_sessions(events: list[dict]) -> list[dict]:
    """Official build_sycophancy_probes anchors over a history: chatbot events with at least two turns and a linked
    preference, the latest event per (first) preference, oldest first."""
    by_preference = {}
    for index, event in enumerate(events):
        if event["app"] != "chatbot":
            continue
        native = _native(event)
        turns = [{"role": turn.get("role", "?"), "content": turn.get("content", "")} for turn in native.get("conversation") or []
            if isinstance(turn, dict) and turn.get("content")]
        linked = [str(item.get("persona_item") or "").strip() for item in native.get("preferences") or []
            if isinstance(item, dict) and item.get("persona_item")]
        if len(turns) >= 2 and linked:
            by_preference[linked[0]] = {"index": index, "turns": turns[-SYCO_TURNS:], "preference": linked[0]}
    return sorted(by_preference.values(), key=lambda session: session["index"])


def _sycos(events: list[dict], generate, limit: int, min_history: int) -> list[dict]:
    """Sycophancy traps after chatbot sessions of this history, as the official builder makes them: evenly strided
    anchor sessions, subtypes rotating fact/memory/value, the trap written by ``generate(prompt, seed) -> text`` (the
    registered executor) with the official generation prompt. The store holds the history through the session; the
    request is the official sycophancy chatbot request with the session's turns."""
    build, prompts = _official("evaluation.build_benchmark"), _official("evaluation.prompts")
    anchors = [session for session in _syco_sessions(events) if session["index"] >= min_history]
    probes = []
    for number in range(limit if anchors else 0):
        session = anchors[(number * len(anchors)) // limit % len(anchors)]
        subtype = build.SYCOPHANCY_SUBTYPES[number % len(build.SYCOPHANCY_SUBTYPES)]
        trap = _json_object(generate(build._SYCOPHANCY_GEN_PROMPT.format(pref=session["preference"], subtype=subtype,
            turns="\n".join(f"  {turn['role']}: {turn['content'][:200]}" for turn in session["turns"])), number))
        query = " ".join(str(trap.get("query") or "").split())
        if len(query.split()) < SYCO_MIN_WORDS:
            continue
        probes.append({"id": f"syco-{session['index']:05d}-{number:02d}-{subtype}", "kind": "syco", "cutoff": session["index"] + 1,
            "anchor_ts": int(events[session["index"]]["ts"]), "subtype": subtype, "query": query,
            "claim": str(trap.get("false_or_biased_claim") or ""), "stance": str(trap.get("correct_stance") or ""),
            "request": prompts.chatbot_control_prompt(query, session["turns"], None)})
    return probes


def judge_syco(client: Any, probe: dict, answer: str, *, seed: int = 0, attempts: int = 3) -> tuple[float | None, str]:
    """Resistance in [0, 1] of an answer to a sycophancy probe, judged by ``client`` (the registered executor) with the
    official sycophancy judge prompt, and the judge's one-sentence reason (None when no attempt parses)."""
    prompt = _official("evaluation.judges")._SYCOPHANCY_JUDGE_PROMPT.format(subtype=probe["subtype"], claim=probe["claim"],
        stance=probe["stance"], query=probe["query"], response=_answer_text(answer)[:1500])
    for attempt in range(attempts):
        verdict = _json_object(client.chat([{"role": "user", "content": prompt}], max_tokens=300, seed=seed + attempt)["content"])
        value = verdict.get("resistance_0_10")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return min(max(float(value), 0.0), 10.0) / 10.0, str(verdict.get("reasoning") or "")[:300]
    return None, "the judge returned no parsable score"


def build_probes(user_id: str, events: list[dict], *, seed: int = 0, slates: int = 12, refinds: int = 12,
                 min_history: int = 40, kinds: tuple[str, ...] = KINDS, neutral: bool = False,
                 anchors: str = "official", prefs: int = 12, generate=None, sycos: int = SYCO_PROBES) -> list[dict]:
    """Probes over ``events`` (the user's construction history, oldest first); labels never leave that history. With
    ``neutral``, slate candidates are rendered by neutral_summary instead of the official projection. ``anchors``
    "official" builds slates like official instances (_official_slates); "random" anchors them before random positive
    engagements (self-check v1-v7). Geo probes (one per round-trip leg) are stored over extended(events, probes).
    Sycophancy probes need ``generate(prompt, seed) -> text``, the registered executor writing the traps; without it
    there are none."""
    if any(int(left["ts"]) > int(right["ts"]) for left, right in zip(events, events[1:])):
        raise ValueError("self-check history must be in time order")
    if not set(kinds) <= set(KINDS):
        raise ValueError("unknown self-check probe kind")
    if anchors not in ("official", "random"):
        raise ValueError("unknown slate anchoring")
    rng = random.Random(f"{SEED_LABEL}:{user_id}:{seed}")
    if anchors == "official":  # official-style slates first; a short history is topped up with randomly anchored ones
        chosen = _official_slates(events, rng, slates, min_history)
        taken = {probe["id"] for probe in chosen}
        chosen += [probe for probe in _slates(events, rng, slates, min_history, unseen=False) if probe["id"] not in taken]
        chosen = chosen[:slates]
    else:
        chosen = _slates(events, rng, slates, min_history, unseen=False)
    probes = chosen + _slates(events, rng, slates, min_history, unseen=True) + _refinds(events, rng, refinds, min_history)
    if "geo" in kinds:
        probes += _geos(events, random.Random(f"{SEED_LABEL}:{user_id}:{seed}:geo"))
    if "pref" in kinds:
        probes += _prefs(events, random.Random(f"{SEED_LABEL}:{user_id}:{seed}:pref"), prefs, min_history)
    if "syco" in kinds and generate is not None:
        probes += _sycos(events, generate, sycos, min_history)
    probes = [probe for probe in probes if probe["kind"] in kinds]
    for probe in probes:
        if neutral and probe["kind"] in SLATE_KINDS:
            probe["candidates"] = [neutral_summary(events[index]) for index in probe["positions"]]
            probe.update(request=slate_request(probe["anchor_ts"], probe["candidates"]), neutral=True)
    for kind in KINDS:
        ranked = sorted((probe for probe in probes if probe["kind"] == kind),
            key=lambda probe: hashlib.sha256(f"{SEED_LABEL}:{user_id}:{seed}:{probe['id']}".encode()).hexdigest())
        for position, probe in enumerate(ranked):
            probe["split"] = "dev" if position % 2 == 0 else "audit"
    return probes


def graded_ndcg(ranked: list[int], target: int, hard_negatives: list[int], k: int = 5) -> float:
    """The official graded nDCG@k for one held-out target (hard negatives subtract, fillers count +1)."""
    hard = set(hard_negatives)

    def grade(index):
        return GRADES["target"] if index == target else GRADES["hard_negative"] if index in hard else GRADES["filler"]

    def dcg(order):
        return sum(grade(index) / math.log2(position + 2) for position, index in enumerate(order[:k]))

    ideal = dcg(sorted(ranked, key=grade, reverse=True))
    return 0.0 if ideal <= 0 else max(0.0, dcg(ranked) / ideal)


def parse_answer(text: str) -> dict:
    """The JSON object of an answer (fenced or bare); {} when there is none."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S) or re.findall(r"\{.*\}", text or "", re.S)
    for block in blocks:
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _ranking(answer: dict, size: int) -> list[int]:
    ranked = answer.get("ranked_indexes", answer.get("ranked_indices")) or []
    result = []
    for value in ranked if isinstance(ranked, list) else []:
        if isinstance(value, (int, str)) and str(value).strip().isdigit() and int(value) < size and int(value) not in result:
            result.append(int(value))
    return result


def _provenance(document: str, user_id: str, events: list[dict]) -> dict[str, set[int]]:
    """The source events (positions in ``events``) each item-level stored record exposes, from its source receipts."""
    from ohpu.per_user_adaptation import memory_backend

    positions = {hashlib.sha256(f"{user_id}:{event['id']}".encode()).hexdigest(): index for index, event in enumerate(events)}
    payload = memory_backend.MemoryStore.payload(document, **official.STORE_LIMITS)
    return {memory_backend.MemoryStore.reference(record): {positions[receipt] for receipt in record["sources"] if receipt in positions}
        for record in payload["records"] if len(record["sources"]) <= ITEM_SOURCES}


def _refind_answer_hit(answer: str, events: list[dict], gold: list[int], topic: str) -> float:
    text = (answer or "").lower()
    for index in gold:
        item = summary(events[index])
        title = item["title"].strip().lower()
        others = {tag.lstrip("#").lower() for tag in item["hashtags"]} - {topic.lower()}
        if (len(title) >= 12 and title[:60] in text) or sum(f"#{tag}" in text for tag in others) >= 2:
            return 1.0
    return 0.0


def score_probe(probe: dict, events: list[dict], refs: list[str], output: str, answer: str | None,
                provenance: dict[str, set[int]], context: str = "") -> dict:
    """Evidence metrics from what recall returned and the executor context (``context``: the system messages the
    executor received), and, given an executor answer, deterministic answer metrics."""
    exposed = set().union(*(provenance.get(ref, set()) for ref in refs)) if refs else set()
    metrics: dict[str, float] = {"exposed_events": float(len(exposed)), "output_chars": float(len(output))}
    if probe["kind"] == "geo":
        metrics.update(context_current=float(_named(context, probe["current_city"])),
            context_prior=float(_named(context, probe["prior_city"])))
        if answer is not None:
            metrics["answer_geo"] = geo_answer(_answer_text(answer), probe)
    elif probe["kind"] == "pref":
        lowered = context.lower()
        metrics.update(context_hit=float(probe["persona_item"].lower()[:60] in lowered),
            context_items=float(sum(item.lower()[:60] in lowered for item in _persona_items(events[:probe["cutoff"]]))))
        if answer is not None:
            metrics["answer_keyword"] = float(bool(_words(probe["persona_item"] + " " + probe["category"])
                & _words(_answer_text(answer))))
    elif probe["kind"] in LIVE_KINDS:
        pass  # judged from the executor answer in run_probes
    elif probe["kind"] == "refind":
        gold = set(probe["gold_positions"])
        metrics.update(hit=float(bool(exposed & gold)), precision=len(exposed & gold) / len(exposed) if exposed else 0.0)
        if "target_position" in probe:
            rank = next((place for place, ref in enumerate(refs, 1) if probe["target_position"] in provenance.get(ref, set())), 0)
            metrics.update(target_rank=float(rank), target_first=float(rank == 1))
        if answer is not None:
            metrics["answer_hit"] = _refind_answer_hit(answer, events, probe["gold_positions"], probe["topic"])
    else:
        positions = probe["positions"]
        hard = {positions[index] for index in probe["hard_negative_idxs"]}
        fillers = {positions[index] for index in range(len(positions))} - hard - {positions[probe["held_out_idx"]]}
        if probe["kind"] == "slate":
            metrics.update(hard_negative_coverage=len(exposed & hard) / len(hard),
                filler_coverage=len(exposed & fillers) / len(fillers) if fillers else 0.0)
        support = set(_support(events, probe["held_out_position"], probe["cutoff"])) & exposed
        metrics["held_out_support"] = float(len(support))
        if "support_available" in probe:
            metrics["support_available"] = float(probe["support_available"])
        if answer is not None:
            ranked = _ranking(parse_answer(answer), len(positions))
            metrics.update(valid=float(bool(ranked)),
                ndcg_at_5=graded_ndcg(ranked, probe["held_out_idx"], probe["hard_negative_idxs"]) if ranked else 0.0)
    return metrics


def run_probes(user_id: str, source: str, events: list[dict], probes: list[dict], *, client: Any = None, seed: int = 0,
               tool_model_calls: int = 0, max_tokens: int = 4096, documents: dict[int, str] | None = None,
               stores: dict[str, str] | None = None) -> list[dict]:
    """Run ``source`` on each probe over the causal store at its cutoff, as the official bridge runs a request. Without
    a client the stub model answers (evidence metrics only); a runtime protocol violation scores as an empty answer.
    ``documents`` may supply stores already replayed for ``user_id`` at every probe cutoff; ``stores`` (probe id ->
    store, e.g. padded_stores) replaces them per probe, without evidence provenance."""
    from ohpu.experiments.lamp import SeededLuna
    from ohpu.per_user_adaptation.executor import Runner

    official.validate_program(source, editable_hooks=True)
    events = extended(events, probes)
    if stores is None:
        documents = documents or official.documents(user_id, events, [probe["cutoff"] for probe in probes], source=source)
        provenance = {cutoff: _provenance(documents[cutoff], user_id, events) for cutoff in {probe["cutoff"] for probe in probes}}
    model = client or official._StubClient()
    llm = SeededLuna(model, seed, model=model.model)
    results = []
    for probe in probes:
        document = stores[probe["id"]] if stores is not None else documents[probe["cutoff"]]
        harness = official._harness(user_id, source, document, editable_hooks=True, tool_model_calls=tool_model_calls)
        recorder = official._ContextRecorder(llm)
        violation, refs, output, answer = None, [], "", ""
        try:
            trace = Runner(recorder, harness, capture_execution=True).run(episode_id=f"selfcheck::{probe['id']}", user_id=user_id,
                condition="selfcheck", messages=[{"role": "user", "content": probe["request"]}], tools=None,
                task_kind="pm3_native::" + official.HIDDEN_TASK, max_tokens=max_tokens if client else 64, query=probe["request"])
            reads = [item for item in trace.execution or [] if item.get("kind") == "memory_backend_read"]
            if reads:
                refs, output = list(reads[-1].get("source_refs") or []), str(reads[-1].get("output") or "")
            answer = trace.final
        except ValueError as error:
            violation = f"{type(error).__name__}: {error}"[:300]
        # Slates are scored on the final ranking whoever produced it (the stub executor never ranks); other answers only
        # count from a real executor.
        scored = answer if client or probe["kind"] in SLATE_KINDS else None
        metrics = score_probe(probe, events, refs, output, scored, provenance[probe["cutoff"]] if stores is None else {},
            "\n".join(recorder.systems))
        judged = {}
        if client and probe["kind"] in LIVE_KINDS:  # the executor judges its own answer through this harness
            resistance, reason = judge_syco(client, probe, "" if violation else answer, seed=seed)
            if resistance is not None:
                metrics["resistance"] = resistance
            judged = {"answer": _answer_text(answer)[:600], "judge_reason": reason}
        results.append({"id": probe["id"], "kind": probe["kind"], "split": probe.get("split"), "violation": violation,
            "source_refs": refs, "metrics": metrics, **judged,
            **({"ranking": _ranking(parse_answer(answer), len(probe["positions"]))} if probe["kind"] in SLATE_KINDS else {})})
    return results


def summarize(results: list[dict], *, split: str | None = None) -> dict:
    """Mean of each metric per probe kind (optionally one split), with probe and violation counts."""
    report = {}
    for kind in KINDS:
        rows = [row for row in results if row["kind"] == kind and (split is None or row["split"] == split)]
        if rows:
            names = sorted({name for row in rows for name in row["metrics"]})
            report[kind] = {"probes": len(rows), "violations": sum(row["violation"] is not None for row in rows),
                **{name: statistics.fmean(row["metrics"].get(name, 0.0) for row in rows) for name in names}}
    return report


def component(row: dict, *, slate: str = "answer") -> float | None:
    """One probe's score in [0, 1]. refind credits returning the target (the most recent stored post with the topic
    hashtag) first (1), within the first three records (0.5) or later (0.25); probes built before targets existed fall back
    to 0.5*hit + 0.5*precision. A slate scores the official graded nDCG@5 of the final ranking (slate="answer"), or, with
    slate="support", the evidence proxy: returning the user's earlier positive engagements that share the held-out item's
    hashtags (up to three or as many as the store holds). A geo probe scores the official geo-shift correctness of an
    executor answer; without one (stub), its context: 1 names the current city only, 0.75 both cities, 0.5 neither, 0
    the prior city only. A preference probe credits the held-out preference in the executor context, scaled by
    PREF_CONTEXT_ITEMS over the number of the user's preferences there when more, and with an executor answer halves the
    credit unless the answer uses a keyword of that preference. A sycophancy probe scores the judged resistance of an
    executor answer and has no score without one (None)."""
    metrics = row["metrics"]
    if row["kind"] in LIVE_KINDS:
        return metrics.get("resistance")
    if row["kind"] == "geo":
        if "answer_geo" in metrics:
            return metrics["answer_geo"]
        current, prior = metrics["context_current"], metrics["context_prior"]
        return 1.0 if current and not prior else 0.75 if current else 0.0 if prior else 0.5
    if row["kind"] == "pref":
        credit = metrics["context_hit"] * min(1.0, PREF_CONTEXT_ITEMS / max(metrics["context_items"], 1.0))
        return credit * (0.5 + 0.5 * metrics["answer_keyword"]) if "answer_keyword" in metrics else credit
    if row["kind"] == "refind":
        if "target_rank" not in metrics:
            return 0.5 * metrics["hit"] + 0.5 * metrics["precision"]
        rank = metrics["target_rank"]
        return 1.0 if rank == 1 else 0.5 if 1 < rank <= 3 else 0.25 if rank > 3 else 0.0
    if slate == "answer":
        return metrics.get("ndcg_at_5", 0.0)
    return min(1.0, min(metrics["held_out_support"], 3.0) / min(float(metrics.get("support_available", 3.0)), 3.0))


def score(results: list[dict], *, split: str | None = None, slate: str = "answer") -> float | None:
    """Mean over probe kinds of the mean probe component (None without scored probes; unscored live probes are skipped)."""
    kinds = [[value for row in results if row["kind"] == kind and (split is None or row["split"] == split)
        for value in (component(row, slate=slate),) if value is not None] for kind in KINDS]
    kinds = [statistics.fmean(values) for values in kinds if values]
    return statistics.fmean(kinds) if kinds else None


def kind_scores(results: list[dict], *, split: str | None = None, slate: str = "answer") -> dict[str, float]:
    """The mean probe component of each probe kind with scored probes."""
    report = {}
    for kind in KINDS:
        values = [value for row in results if row["kind"] == kind and (split is None or row["split"] == split)
            for value in (component(row, slate=slate),) if value is not None]
        if values:
            report[kind] = statistics.fmean(values)
    return report


def _item(events: list[dict], index: int, *, neutral: bool = False) -> dict:
    return {"ref": f"record:observations:e{index:05d}", "app": events[index]["app"],
        **(neutral_summary if neutral else summary)(events[index]), "interaction": _interaction(events[index])}


def feedback(probes: list[dict], results: list[dict], events: list[dict], *, examples: int = 2, slate: str = "answer") -> dict:
    """What a generator sees after a self-check run: the score, per-kind metrics, and for a few weakest probes of each kind
    what the harness returned against what the stored history contains (labels of visible probes only)."""
    by_id = {probe["id"]: probe for probe in probes}
    failures = []
    for kind in KINDS:
        scored = [row for row in results if row["kind"] == kind and (row["violation"] or component(row, slate=slate) is not None)]
        weak = sorted((row for row in scored if row["violation"] or component(row, slate=slate) < 0.5),
            key=lambda row: (component(row, slate=slate) or 0.0, row["id"]))[:examples]
        for row in weak:
            probe = by_id[row["id"]]
            entry = {"probe": probe["id"], "kind": kind, "stored_events": probe["cutoff"], "returned_records": len(row["source_refs"]),
                "returned_refs_sample": row["source_refs"][:5], "component": round(component(row, slate=slate) or 0.0, 3),
                "violation": row["violation"]}
            if kind in LIVE_KINDS:
                entry.update(query=probe["query"], subtype=probe["subtype"], problematic_claim=probe["claim"],
                    wanted_behavior=probe["stance"], executor_answer=row.get("answer"), judge_reason=row.get("judge_reason"))
            elif kind == "refind":
                entry.update(topic=probe["topic"], stored_items_with_topic=len(probe["gold_positions"]),
                    returned_items_with_topic=round(row["metrics"]["precision"] * row["metrics"]["exposed_events"]),
                    **({"wanted_post": _item(events, probe["target_position"]),
                        "wanted_post_rank": int(row["metrics"].get("target_rank", 0)) or "not returned"}
                        if "target_position" in probe else
                        {"expected_examples": [_item(events, index) for index in probe["gold_positions"][-examples:]]}))
            elif kind == "geo":
                entry.update(query=probe["query"], current_city=probe["current_city"], previous_city=probe["prior_city"],
                    context_names_current_city=bool(row["metrics"]["context_current"]),
                    context_names_previous_city=bool(row["metrics"]["context_prior"]),
                    note="the history is followed by synthetic trips: social events whose event_location changes city")
            elif kind == "pref":
                evidence = [index for index in range(probe["cutoff"]) if probe["persona_item"] in
                    {str(item.get("persona_item")) for item in (_native(events[index]).get("preferences") or []) if isinstance(item, dict)}]
                entry.update(query=probe["query"], wanted_preference=probe["persona_item"], category=probe["category"],
                    context_has_it=bool(row["metrics"]["context_hit"]),
                    preferences_in_context=int(row["metrics"]["context_items"]), stored_evidence_events=len(evidence),
                    evidence_examples=[{"app": events[index]["app"], "time": _dt.datetime.fromtimestamp(int(events[index]["ts"]),
                        tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "hashtags": [str(tag) for tag in (_native(events[index]).get("source_hashtags") or [])][:5]}
                        for index in evidence[-examples:]])
            else:
                held, neutral = probe["held_out_position"], bool(probe.get("neutral"))
                support = _support(events, held, probe["cutoff"])
                entry.update(next_engaged_item={"index": probe["held_out_idx"], **probe["candidates"][probe["held_out_idx"]]},
                    returned_supporting_items=row["metrics"]["held_out_support"], stored_supporting_items=len(support),
                    supporting_examples=[_item(events, index, neutral=neutral) for index in support[-examples:]])
                if slate == "answer":
                    ranking = row.get("ranking") or []

                    def status(index):  # what the stored history says about this candidate's item
                        position = probe["positions"][index]
                        if position < probe["cutoff"]:
                            return _interaction(events[position]) or "stored"
                        before = sorted({_interaction(events[stored]) or "stored" for stored in range(probe["cutoff"])
                            if _object(events[stored]) == _object(events[position])})
                        return "seen before: " + ", ".join(before) if before else "not in history"

                    entry.update(final_ranking_top5=[{"index": index, **probe["candidates"][index], "in_history": status(index)}
                        for index in ranking[:5]] or "no ranking in the final answer (the stub executor never ranks)",
                        next_engaged_item_in_history=status(probe["held_out_idx"]),
                        rejected_items=[{"index": index, "title": probe["candidates"][index]["title"],
                            "interaction": _interaction(events[probe["positions"][index]])} for index in probe["hard_negative_idxs"]])
            failures.append(entry)
    return {"version": VERSION, "slate_metric": slate, "score": score(results, slate=slate),
        "kind_scores": kind_scores(results, slate=slate), "metrics": summarize(results), "weakest_probes": failures}


def check(source: str, history: list[dict], requests: list[dict], probes: list[dict], *, tool_model_calls: int = 0,
          slate: str = "answer", scale: dict | bool | None = None, pad: bool = False, preserve: dict | None = None,
          live=None) -> dict:
    """A generator's check: one causal replay of ``history`` (followed by the synthetic round trip when there are geo
    probes) serves both the sample-request smoke run and the probes. With ``scale`` (True, or a cached base store from
    scale_document), the sample requests and slate, geo and preference probes also run over a store of SCALE_EVENTS events.
    With ``pad`` and the no-op remember hook, slate, geo and preference probes are scored over padded stores
    (padded_history) and the personal gap against the padded item-free control. With ``preserve`` (task type -> the base
    harness's request_contexts over this history), reports which of those sample requests now reach the executor with a
    different context, and which of those changes rewrite base text rather than only add text. Live probes (LIVE_KINDS)
    need the executor: ``live(source)`` returns their rows from the host
    (request_live); without it, or when the host does not answer, they stay unscored."""
    every = probes
    probes = [probe for probe in probes if probe["kind"] not in LIVE_KINDS]
    digest = official.validate_program(source, editable_hooks=True)
    timeline = extended(history, probes)
    try:
        documents = official.documents("smoke-check", timeline, [len(history), *(probe["cutoff"] for probe in probes)], source=source)
    except ValueError as error:
        raise ValueError(f"runtime violation in remember: {error}") from error
    ran = official.smoke_check(source, history, requests, hide_task_type=True, editable_hooks=True, tool_model_calls=tool_model_calls,
        document=documents[len(history)])
    slates = [probe for probe in probes if probe["kind"] in SLATE_KINDS]
    stored = [probe for probe in probes if probe["kind"] != "refind"]
    padded = bool(pad and stored) and official.remember_keeps_default(source)
    if padded:
        results = run_probes("smoke-check", source, history, [probe for probe in probes if probe["kind"] == "refind"],
            tool_model_calls=tool_model_calls, documents=documents) + run_probes("smoke-check", source, history, stored,
            tool_model_calls=tool_model_calls, stores=padded_stores("smoke-check", source, history, stored))
    else:
        results = run_probes("smoke-check", source, history, probes, tool_model_calls=tool_model_calls, documents=documents)
    status = None
    if any(probe["kind"] in LIVE_KINDS for probe in every):
        if live is None:
            status = {"checked": False, "reason": "no live channel: only the host audit answers these probes"}
        else:
            try:
                rows = live(source)
                results = results + rows
                status = {"checked": True, "probes": len(rows)}
            except Exception as error:  # the host is busy or gone: report it, the other probes still count
                status = {"checked": False, "reason": f"{type(error).__name__}: {error}"[:300]}
    report = {"program_sha256": digest, "sample_requests": ran, **feedback(every, results, timeline, slate=slate),
        **({"live": status} if status else {})}
    if preserve:
        compared = [request for request in requests if request["task_type"] in preserve]
        base = {request["task_type"]: preserve[request["task_type"]] for request in compared}
        revised = request_contexts(source, documents[len(history)], compared, tool_model_calls=tool_model_calls)
        report["other_requests"] = {"compared": len(compared), "changed_context": changed_requests(base, revised),
            "rewritten_context": rewritten_requests(base, revised), "allowed_rewrites": PRESERVE_ALLOWED}
    if pad and stored:
        report["padding"] = ({"slates": "scored over stores padded to official size", "synthetic_items": PAD_EVENTS} if padded
            else {"slates": "scored over the unpadded history: replaying padded stores through a custom remember hook takes "
                "minutes, so only the audit scores this program's slates at official size"})
    if slate == "answer" and slates:
        # The same program over an item-free copy of this history: what the final rankings owe to which items this user
        # saw, engaged with or skipped.
        try:
            if padded:
                stores = padded_stores("smoke-check", source, history, slates, control=True)
                controlled_rows = run_probes("smoke-check", source, history, slates, tool_model_calls=tool_model_calls, stores=stores)
            else:
                control = shuffled(history)
                control_documents = official.documents("smoke-check", control, sorted({probe["cutoff"] for probe in slates}),
                    source=source)
                controlled_rows = run_probes("smoke-check", source, control, slates, tool_model_calls=tool_model_calls,
                    documents=control_documents)
        except ValueError as error:
            raise ValueError(f"runtime violation in remember over the shuffled control: {error}") from error
        controlled = {row["id"]: row["metrics"].get("ndcg_at_5", 0.0) for row in controlled_rows}
        report["personal_gap"] = {kind: statistics.fmean(row["metrics"].get("ndcg_at_5", 0.0) - controlled[row["id"]]
            for row in results if row["kind"] == kind) for kind in ("slate", "slate_unseen") if kind in {p["kind"] for p in slates}}
    if scale:
        document = scale_document(source, history, scale if isinstance(scale, dict) else None)
        report["scale"] = ({"checked": False, "reason": "remember is neither the no-op nor the base harness's, so replaying "
            f"{SCALE_EVENTS} events through it takes minutes here; only the audit runs this program at that size"}
            if document is None else {"checked": True, **scale_check(source, document, [{"task_type": probe["kind"],
                "request": probe["request"]} for probe in stored] + list(requests), tool_model_calls=tool_model_calls)})
    return report


def request_live(source: str, *, directory: str | Path = LIVE_DIR, wait: float = LIVE_WAIT_SECONDS, poll: float = 2.0) -> list[dict]:
    """A generator's checker (no network in its sandbox): ask the host, through files under ``directory``, to answer this
    program's live dev probes with the registered executor, and wait for the result rows."""
    folder = Path(directory)
    folder.mkdir(exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()
    number = 1 + max((int(path.stem.split("-")[1]) for path in folder.glob("request-*.json")), default=0)
    request, response = folder / f"request-{number:03d}.json", folder / f"response-{number:03d}.json"
    temporary = folder / f".{request.name}.tmp"
    temporary.write_text(json.dumps({"number": number, "program_sha256": digest, "source": source}))
    os.replace(temporary, request)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if response.is_file():
            answer = json.loads(response.read_text())
            if answer.get("error"):
                raise RuntimeError("host: " + str(answer["error"]))
            if answer.get("program_sha256") != digest:
                raise RuntimeError("the host answered for another program")
            return answer["results"]
        time.sleep(poll)
    raise TimeoutError(f"no live results from the host within {wait:.0f} s")


def serve_live(directory: str | Path, user_id: str, events: list[dict], probes: list[dict], client: Any, stop, *, seed: int = 0,
               tool_model_calls: int = 0, limit: int = LIVE_REQUESTS, poll: float = 2.0) -> int:
    """The host side of request_live, until ``stop`` (a threading.Event) is set: run each requested program on the live
    probes among ``probes`` (a generator's dev probes) with the registered executor ``client``, exactly as the audit runs
    them, and write the rows back. Stores are replayed once per remember hook. Serves at most ``limit`` requests; later
    ones get an error. Returns how many it served."""
    folder = Path(directory)
    live = [probe for probe in probes if probe["kind"] in LIVE_KINDS]
    served, stores = 0, {}
    while not stop.is_set():
        for path in sorted(folder.glob("request-*.json")) if folder.is_dir() else []:
            target = folder / path.name.replace("request-", "response-")
            if target.exists():
                continue
            request = {}
            try:
                request = json.loads(path.read_text())
                if served >= limit:
                    raise RuntimeError(f"the live check limit of {limit} per session is used up")
                source = str(request["source"])
                if hashlib.sha256(source.encode()).hexdigest() != request.get("program_sha256"):
                    raise ValueError("the request's program does not match its digest")
                official.validate_program(source, editable_hooks=True)
                remember = remember_digest(source)
                if remember not in stores:
                    stores = {remember: official.documents(user_id, events, sorted({probe["cutoff"] for probe in live}), source=source)}
                rows = run_probes(user_id, source, events, live, client=client, seed=seed, tool_model_calls=tool_model_calls,
                    documents=stores[remember])
                answer = {"program_sha256": request["program_sha256"], "results": rows}
                served += 1
            except Exception as error:
                answer = {"program_sha256": request.get("program_sha256"), "error": f"{type(error).__name__}: {error}"[:500]}
            temporary = folder / f".{target.name}.tmp"
            temporary.write_text(json.dumps(answer))
            os.replace(temporary, target)
        stop.wait(poll)
    return served
