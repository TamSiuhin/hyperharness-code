"""PersonaMem-v3 -> canonical IR.

Source: HF `bowen-upenn/PersonaMem-v3`, `backend/{uid}/` for the first public
cohort of 100 users (the project overview describes 200 source users).
Data CC-BY-NC-4.0 (inherited from facebook/gistbench); code MIT.

Files used per user:
  profile.json                        evaluator-private oracle; never agent-visible
  test.json                           ~157 held-out queries, each with its own oracle
  {facebook,instagram,threads,chatbot,ai_studio}.json   timestamped history events
  calendar.json                       CRUD modification stream

Three properties earn this corpus its slot:

1. ROLLING T. Every event has a unix `source_timestamp` and every query has `ts`,
   and the two interleave -- only ~8 of 190 queries fall after all history. So a
   single global T would throw away 95% of the eval set. We assign each event to
   the episode boundary it precedes, which makes fixed-T the special case of
   rolling-T rather than a competing design.

2. IMPLICIT SIGNAL IS LABELLED, so the implicit-only headline subset (proposal
   6.7b, defense 3) is a filter rather than a research project.
   `source_interaction_type` in {explicit_positive, implicit_positive,
   implicit_negative, explicit_negative, feed_visible}.

3. A NEGATIVE CLASS. `expected_behavior` in {avoid_overpersonalization,
   resist_sycophancy, abstain, restrain} marks 4,887 queries where restraint is
   the correct answer. That is the over-imposition column of the section 7.2
   confusion matrix, purpose-built -- and it is what OP-Bench was supposed to
   supply before it turned out not to exist publicly.

HISTORY VISIBILITY. PM3's official evaluator exposes the five time-masked app
event files, including safe per-event inferred preference annotations, while
withholding ``profile.json`` and scoring-only preference fields. With
``strip_leaks=True`` (the default), OHPU mirrors that firewall rather than
inventing a stricter raw-behavior projection. ``HistoryItem.reveals`` retains
the normalized preference lineage for evaluator-side audits but is never the
source of model-visible content.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from .. import pm3_history
from ...ir import Checkpoint, Corpus, Episode, HistoryItem, Preference, User

DATASET = "personamem3"
STREAMS = ["facebook", "instagram", "threads", "chatbot", "ai_studio"]
IMPLICIT = {"implicit_positive", "implicit_negative", "feed_visible"}
EXPLICIT = {"explicit_positive", "explicit_negative"}
# expected_behavior values where doing MORE personalization is the error
RESTRAINT = {"avoid_overpersonalization", "resist_sycophancy", "abstain", "restrain"}
_PREF_TS = re.compile(r"^\s*(\d{2}:\d{2}),\s*(\d{2}/\d{2}/\d{4})\s*:\s*(.*)$", re.S)


def _load(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_pref(s: str) -> tuple[str, str]:
    m = _PREF_TS.match(s or "")
    return (f"{m.group(2)} {m.group(1)}", m.group(3).strip()) if m else ("", (s or "").strip())


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())[:160]


def load_user(root: str, uid: str, strip_leaks: bool = True) -> Optional[User]:
    base = os.path.join(root, "backend", uid)
    prof = _load(os.path.join(base, "profile.json"))
    test = _load(os.path.join(base, "test.json"))
    if prof is None or test is None:
        return None

    # ---- oracle preferences -------------------------------------------------
    prefs: dict[str, Preference] = {}
    for raw in prof.get("preferences") or []:
        ts, text = _parse_pref(raw)
        key = _norm(text)
        prefs[key] = Preference(pref_id=f"{uid}::p{len(prefs)}", user_id=uid,
                                text=text, category="stated_preference",
                                status_history=[{"ts": ts}])
    # hidden_personas are the deepest oracle layer: inferred, never surfaced,
    # and by construction never stated outright.
    for i, hp in enumerate(prof.get("hidden_personas") or []):
        text = hp.get("label") or hp.get("description") or ""
        key = _norm(text)
        if key and key not in prefs:
            prefs[key] = Preference(pref_id=f"{uid}::h{i}", user_id=uid, text=text,
                                    category=f"hidden::{hp.get('type', '')}",
                                    explicit=False,
                                    status_history=[{"first_seen_ts": hp.get("first_seen_ts"),
                                                     "last_seen_ts": hp.get("last_seen_ts"),
                                                     "evidence_rows": hp.get("evidence_rows")}])

    # ---- history events, merged and time-sorted ----------------------------
    events: list[tuple[int, str, dict]] = []
    for s in STREAMS:
        for ev in _load(os.path.join(base, f"{s}.json")) or []:
            events.append((int(ev.get("source_timestamp") or 0), s, ev))
    cal = _load(os.path.join(base, "calendar.json")) or {}
    for m in (cal.get("modifications") if isinstance(cal, dict) else []) or []:
        events.append((int(m.get("ts") or 0), "calendar", m))
    events.sort(key=lambda x: x[0])

    # Which events evidence which preference, and was it ever stated outright?
    seen_explicit: dict[str, bool] = {}
    for _ts, _s, ev in events:
        kind = ev.get("source_interaction_type") or ""
        for pr in ev.get("preferences") or []:
            k = _norm(pr.get("persona_item") or "")
            if k:
                seen_explicit[k] = seen_explicit.get(k, False) or (kind in EXPLICIT)
    for k, p in prefs.items():
        if p.explicit is None and k in seen_explicit:
            p.explicit = seen_explicit[k]

    # ---- episodes: one per held-out query, rolling T ------------------------
    queries = sorted(test, key=lambda q: int(q.get("ts") or 0))
    eps: list[Episode] = []
    cursor = 0
    for order, q in enumerate(queries, start=1):
        qts = int(q.get("ts") or 0)
        delta: list[HistoryItem] = []
        while cursor < len(events) and events[cursor][0] < qts:
            ts, stream, ev = events[cursor]
            reveals = [_norm(pr.get("persona_item") or "")
                       for pr in (ev.get("preferences") or [])]
            payload = (
                pm3_history.wrap_event(ev, sanitize=strip_leaks)
                if stream in STREAMS
                else ev
            )
            delta.append(HistoryItem(
                item_id=f"{uid}:{order}:{stream}:{cursor}", user_id=uid, order=order,
                date=str(ts), kind=stream, content=payload,
                is_noise=(ev.get("source_interaction_type") == "feed_visible"),
                reveals=[r for r in reveals if r],
            ))
            cursor += 1

        gt = q.get("groundtruth_preference") or ""
        pid = (prefs.get(_norm(gt)) or Preference("", "", "")).pref_id or None
        eb = q.get("expected_behavior") or ""
        qid = str(q.get("query_id") or f"{uid}:{order}")
        cps = [Checkpoint(
            cp_id=f"{qid}::cp",
            episode_id=qid, user_id=uid,
            text=gt or f"expected_behavior={eb}",
            pref_id=pid,
            # 14,320/15,791 queries ship a typed inferior_response, so
            # pick-the-better is exact-match scoring with no judge in the loop.
            decidable="program" if q.get("inferior_response") or q.get("tool_call") else "judge",
            predicate={"expected_behavior": eb,
                       "restraint_expected": eb in RESTRAINT,
                       "rubric_tags": q.get("rubric_tags"),
                       "has_pairwise": bool(q.get("inferior_response")),
                       "tool_call": q.get("tool_call"),
                       "distractor_preferences": q.get("distractor_preferences")},
            applicable=True,
            source="test.json",
        )]

        eps.append(Episode(
            episode_id=qid, user_id=uid, order=order,
            timestamp=q.get("ts_iso") or "", domain=q.get("task_family") or "",
            instruction=q.get("user_query") or "",
            message_history=q.get("prior_conversation") or [],
            gold_actions=[q["tool_call"]] if q.get("tool_call") else [],
            checkpoints=cps, history_delta=delta,
            skills_tested=[q.get("task_type") or ""],
            meta={"task_family": q.get("task_family"), "task_type": q.get("task_type"),
                  "expected_behavior": eb, "query_kind": q.get("query_kind"),
                  "example_response": q.get("example_response"),
                  "inferior_response": q.get("inferior_response"),
                  # Ranking items answer "Ranked indexes: [...]" over this
                  # pool. Only the pool is kept; held_out_idx,
                  # hard_negative_idxs and groundtruth_preference name the
                  # answer and never leave instance_full.
                  "candidates": [
                      {k: c.get(k) for k in ("title", "caption", "hashtags")}
                      for c in ((q.get("instance_full") or {}).get("candidates") or [])
                      if isinstance(c, dict)
                  ]},
        ))

    return User(user_id=uid, dataset=DATASET,
                profile={k: prof.get(k) for k in
                         ("name", "gender", "career", "education", "mbti", "big_five", "bio")},
                preferences=list(prefs.values()), episodes=eps)


def load(root: str, limit: Optional[int] = None, strip_leaks: bool = True) -> Corpus:
    ids = sorted((d for d in os.listdir(os.path.join(root, "backend")) if d.isdigit()),
                 key=int)
    if limit:
        ids = ids[:limit]
    users = [u for u in (load_user(root, i, strip_leaks) for i in ids) if u]
    return Corpus(dataset=DATASET, users=users, notes={
        "source": "hf://bowen-upenn/PersonaMem-v3",
        "license": "CC-BY-NC-4.0 (data), MIT (code)",
        "language": "en",
        "split_semantics": "rolling T per query; fixed-T is the special case",
        "leaks_stripped": strip_leaks,
        "agent_visible_history": pm3_history.HISTORY_CONTRACT,
    })
