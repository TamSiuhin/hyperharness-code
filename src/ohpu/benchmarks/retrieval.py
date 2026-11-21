"""Runtime history retrieval.

This is the departure from proposal 4.6 ("运行时不发送历史"): the harness now
gets to look at what this user did before the current episode, and recall's
`retrieval_mode` / `top_k` -- knobs that were inert while there was nothing to
retrieve -- become live coordinates the search can actually move.

TWO PROPERTIES THIS MODULE MUST HAVE OR THE WHOLE COMPARISON IS WORTHLESS.

1. NO FUTURE LEAKAGE. Only items revealed before the current task are eligible.
   Most corpora attach history that is already visible at an episode boundary,
   so source order ``<=`` target order is correct. UserToolBench instead attaches
   a COMPLETED session to that session's episode; it becomes visible only to a
   later session, so its boundary is strict ``<``. `audit()` re-checks the
   corpus-specific boundary per call, because a silent leak here would look
   exactly like a large personalization win.

2. RETRIEVAL IS DETERMINISTIC AND FREE. Relevance scoring is Okapi BM25,
   fitted only to the eligible causal pool. It requires no embedding model or
   model call, and stable tie-breaking keeps a cached run reproducible.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

from . import pm3_history
from ..experiments import pm3_proxy
from ..ir import Episode, HistoryItem, User

# `relevance` is an input alias for saved API-v1 harnesses, not another scorer.
MODES = ["none", "recency", "bm25", "hybrid", "relevance"]
BM25_K1 = 1.2
BM25_B = 0.75

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]+")
_PRIOR_SESSION_POLICY = "prior_sessions_only"
_CAUSAL_PROJECTION_MARKER = "_ohpu_observed_history_projection_v1"
_PROJECTED_CURSOR = "_ohpu_source_event_cursor"
_PROXY_EVIDENCE_ID = "_ohpu_proxy_evidence_id"
_CURSOR_FIELDS = ("episode_order", "episode_rank", "item_rank", "event_rank")


# --------------------------------------------------------------- eligibility

def _requires_strict_source_order(ep: Episode, corpus: str) -> bool:
    """Whether a source episode becomes visible only after it completes."""
    return (
        corpus == "usertoolbench"
        or ep.meta.get("history_visibility") == _PRIOR_SESSION_POLICY
    )


def _cursor_key(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Mapping):
        return None
    parts: list[int] = []
    for field in _CURSOR_FIELDS:
        part = value.get(field)
        if isinstance(part, bool) or not isinstance(part, int):
            return None
        parts.append(part)
    return (parts[0], parts[1], parts[2], parts[3])


def _is_strict_prior_projection(item: HistoryItem, ep: Episode) -> bool:
    """Whether an adapter-issued atomic event is provably before ``ep``.

    Native UserToolBench history items represent completed sessions and must
    remain invisible at the same source order. PM3 proxy adaptation instead
    rebuilds one sanitized event per carrier and embeds its full causal cursor.
    Only that narrower representation may cross the session boundary, and only
    when its cursor is strictly earlier than the proxy decision point.
    """
    content = item.content
    if not isinstance(content, Mapping) or content.get(
        _CAUSAL_PROJECTION_MARKER
    ) is not True:
        return False
    record = content.get("observed_event")
    if not isinstance(record, Mapping):
        record = content
    item_cursor = _cursor_key(record.get(_PROJECTED_CURSOR))
    public = ep.meta.get(pm3_proxy.PUBLIC_META_KEY)
    target_cursor = _cursor_key(
        public.get("decision_cursor") if isinstance(public, Mapping) else None
    )
    return bool(
        item_cursor is not None
        and target_cursor is not None
        and item_cursor < target_cursor
    )


def history_before(u: User, ep: Episode, corpus: str) -> list[tuple[int, HistoryItem]]:
    """Everything this user revealed BEFORE `ep`, and nothing after.

    Returns (source episode order, item) pairs. Carrying the SOURCE is what
    makes the audit a structural guarantee rather than a proxy: `HistoryItem.order`
    records the episode index in the ORIGINAL corpus numbering, while the driver
    renumbers episodes 1..N after dropping unusable ones, so comparing the two
    scales reports leaks that are not there (and would hide ones that are).
    """
    excluded = {str(item_id) for item_id in ep.meta.get("exclude_history_item_ids", [])}
    allowed_raw = ep.meta.get("allowed_history_item_ids")
    allowed = ({str(item_id) for item_id in allowed_raw}
               if isinstance(allowed_raw, list) else None)

    def visible(item: HistoryItem) -> bool:
        return item.item_id not in excluded and (allowed is None or item.item_id in allowed)
    strict_source_order = _requires_strict_source_order(ep, corpus)
    out: list[tuple[int, HistoryItem]] = []
    for e in u.episodes:
        for h in e.history_delta:
            source_is_visible = (
                e.order < ep.order
                or (
                    e.order == ep.order
                    and (
                        not strict_source_order
                        or _is_strict_prior_projection(h, ep)
                    )
                )
            )
            if source_is_visible and visible(h):
                out.append((e.order, h))
    return out


def audit(pairs: Iterable[tuple[int, HistoryItem]], ep: Episode,
          corpus: str) -> dict[str, Any]:
    """Per-call proof that no item violates the corpus's causal boundary."""
    pairs = list(pairs)
    same_session = 0
    same_session = sum(1 for src, _ in pairs if src == ep.order)
    if _requires_strict_source_order(ep, corpus):
        # A completed UTB session contains the oracle assistant reply for
        # every turn. Same-session retrieval is therefore future leakage unless
        # an atomic adapter projection proves it precedes the withheld target.
        bad = sum(
            1
            for src, item in pairs
            if src > ep.order
            or (src == ep.order and not _is_strict_prior_projection(item, ep))
        )
    else:
        bad = sum(1 for src, _ in pairs if src > ep.order)
    return {
        "n": len(pairs),
        "future_items": bad,
        "same_session_items": same_session,
        # The source order actually used. `future_items` is computed with the
        # predicate COMPLEMENTARY to the one `history_before` filtered on, so it
        # is zero by construction and cannot catch a caller that hands in the
        # wrong cursor. A consumer that wants a real causal check asserts
        # `max_source_order < cursor` on this instead.
        "max_source_order": max((src for src, _ in pairs), default=-1),
    }


# ------------------------------------------------------------------ rendering

def _clip(s: Any, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def _render_pm(h: HistoryItem) -> str:
    c = h.content if isinstance(h.content, dict) else {}
    evidence_id = str(c.get(_PROXY_EVIDENCE_ID) or "").strip()
    evidence_prefix = f"[evidence_id={evidence_id}] " if evidence_id else ""
    visible = pm3_history.unwrap_event(c)
    if visible is not None:
        # This is the same compact field view used by PM3's official
        # long-context and textual-memory modes. The loader has already removed
        # scoring internals, and retrieval still applies the per-query cutoff.
        return evidence_prefix + pm3_history.render_compact_event(
            visible, app=h.kind
        )
    if h.kind == "calendar":
        # PM3's agent snapshot exposes the time-masked CRUD stream. Keep the
        # modification structured so retrieval can recover updates/removals.
        return evidence_prefix + json.dumps(c, ensure_ascii=False, sort_keys=True)
    fmt = c.get("interaction_format") or {}
    app = fmt.get("app") or h.kind
    act = fmt.get("action") or ""
    sign = {"explicit_positive": "+", "implicit_positive": "+?",
            "explicit_negative": "−", "implicit_negative": "−?"}.get(
        str(c.get("source_interaction_type") or ""), "")
    tags = " ".join(str(t) for t in (c.get("source_hashtags") or [])[:6])
    body = c.get("source_text") or c.get("text") or c.get("content") or ""
    if isinstance(body, dict):
        body = " / ".join(str(body.get(k)) for k in ("title", "caption", "text", "body")
                          if body.get(k))
    line = f"[{app} {act} {sign}] {tags}".strip()
    rendered = f"{line} {_clip(body, 110)}".strip() if body else line
    return evidence_prefix + rendered


def _render_utb(h: HistoryItem) -> str:
    if isinstance(h.content, dict) and isinstance(h.content.get("messages"), list):
        items = h.content["messages"]
    else:
        items = h.content if isinstance(h.content, list) else [h.content]
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        if item.get("tool_calls"):
            calls = []
            for raw_call in item["tool_calls"]:
                if not isinstance(raw_call, Mapping):
                    continue
                function = raw_call.get("function")
                if not isinstance(function, Mapping):
                    continue
                calls.append(
                    f"{function.get('name')}({function.get('arguments')})"
                )
            body = "tool_calls=" + ", ".join(calls)
        else:
            body = _clip(item.get("content"), 300)
        if body:
            out.append(f"{role}: {body}")
    return "[completed trajectory] " + " | ".join(out)


def _render_review(h: HistoryItem) -> str:
    content = h.content if isinstance(h.content, dict) else {}
    review = content.get("review") if isinstance(content.get("review"), dict) else content
    product = (content.get("product_info")
               if isinstance(content.get("product_info"), dict) else content)
    title = product.get("title") or product.get("description") or "product"
    rating = review.get("rating") or review.get("overall") or content.get("overall") or ""
    text = review.get("text") or review.get("reviewText") or ""
    return f"[review rating={rating}] {_clip(title, 80)} :: {_clip(text, 180)}"


_RENDER = {
    "personamem3": _render_pm,
    "usertoolbench": _render_utb,
    "personalwab": _render_review,
}


def render(h: HistoryItem, corpus: str) -> str:
    try:
        return _RENDER.get(corpus, lambda x: _clip(x.content, 160))(h)
    except Exception:
        return _clip(h.content, 160)


# ------------------------------------------------------------------- scoring

def _tokenize(s: str) -> list[str]:
    """Preserve term frequency; CJK bigrams stay within contiguous spans."""
    s = (s or "").lower()
    out = _WORD.findall(s)
    for span in _CJK.findall(s):
        if len(span) == 1:
            out.append(span)
        else:
            out.extend(span[i:i + 2] for i in range(len(span) - 1))
    return out


def bm25_scores(query: str, texts: Sequence[str]) -> list[float]:
    """Okapi BM25 with k1=1.2, b=0.75 and non-negative Robertson IDF.

    Each distinct query term contributes
        log(1 + (N - df + .5) / (df + .5))
        * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)).
    The IDF convention and default parameters match Lucene's BM25Similarity:
    https://lucene.apache.org/core/9_12_3/core/org/apache/lucene/search/similarities/BM25Similarity.html

    There is no persistent/global index: N, df and avgdl are recomputed from
    these texts only. The retrieval caller filters future and excluded history
    BEFORE this function, so even corpus statistics cannot leak future data.
    """
    terms = sorted(set(_tokenize(query)))
    if not texts or not terms:
        return [0.0] * len(texts)
    docs = [Counter(_tokenize(text)) for text in texts]
    lengths = [sum(doc.values()) for doc in docs]
    avgdl = sum(lengths) / len(docs)
    if not avgdl:
        return [0.0] * len(docs)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(doc.keys())
    idf = {
        term: math.log1p((len(docs) - df[term] + 0.5) / (df[term] + 0.5))
        for term in terms if term in df
    }
    scores = []
    for doc, length in zip(docs, lengths):
        norm = BM25_K1 * (1 - BM25_B + BM25_B * length / avgdl)
        scores.append(sum(
            weight * doc[term] * (BM25_K1 + 1) / (doc[term] + norm)
            for term, weight in idf.items() if doc[term]
        ))
    return scores


def bm25_rank(query: str, texts: Sequence[str]) -> list[int]:
    """Best-first BM25 indices; ties retain the input's causal order."""
    scores = bm25_scores(query, texts)
    return sorted(range(len(texts)), key=lambda i: (-scores[i], i))


# ---------------------------------------------------------------- the entry

_BLOCK_HEADER = "这位用户在此之前的行为与对话记录（按时间先后排列，检索自其历史）："


def _empty_stats(*, degraded: str = "", ranker: str = "none") -> dict[str, Any]:
    return {"n_eligible": 0, "n_used": 0, "chars": 0, "future_items": 0,
            "max_source_order": -1, "degraded": degraded, "ranker": ranker}


def _budget_order(mode: str, n: int, texts: list[str], ep: Episode,
                  top_k: int) -> list[int]:
    """Indices in the order they compete for the character budget.

    Selection order and RENDER order are deliberately different. The channel
    picks by BM25 or recency but emits chronologically: the block carries
    no dates, so time order is the only cue the model has for deciding that a
    later preference supersedes an earlier one. Ranking here rather than in the
    emit loop is what makes a binding budget drop the least useful items
    instead of whichever ones happen to sit at the end of the timeline.
    """
    if mode == "recency":
        return list(range(n))[-top_k:][::-1]
    scored = bm25_rank(ep.instruction, texts)
    if mode != "hybrid":
        return scored[:top_k]
    # Alternate score and recency instead of unioning two fixed-size slices.
    # The union deduplicated down to FEWER than top_k items whenever a recent
    # item also scored well, and at top_k <= 2 it left no relevance slot at all
    # (half = max(1, top_k // 2) made `scored[:top_k - half]` empty).
    recent = list(range(n))[::-1]
    merged: list[int] = []
    for pair in zip(scored, recent):
        merged.extend(pair)
    return list(dict.fromkeys(merged))[:top_k]


def retrieve(u: User, ep: Episode, corpus: str, *, mode: str = "bm25",
             top_k: int = 8, budget_chars: int = 2400) -> tuple[str, dict[str, Any]]:
    """-> (rendered block, stats). Empty block when mode == 'none'."""
    if mode == "relevance":
        mode = "bm25"  # compatibility only; the old overlap scorer was removed
    if mode not in MODES:
        # An out-of-domain value used to fall through to full "relevance"
        # scoring, so a proposer typo ("NONE", "semantic", "embeddings") turned
        # this user's history ON while the controller believed it had turned it
        # off. The one knob that decides what the model sees about the user
        # fails closed, and the degradation is reported rather than silent.
        return "", _empty_stats(degraded=str(mode)[:60])
    if mode == "none" or top_k <= 0:
        return "", _empty_stats()

    ranker = "bm25" if mode in {"bm25", "hybrid"} else "recency"
    pairs = history_before(u, ep, corpus)
    a = audit(pairs, ep, corpus)
    items = [h for _, h in pairs]
    if not items:
        return "", _empty_stats(ranker=ranker)

    texts = [render(h, corpus) for h in items]
    order = _budget_order(mode, len(items), texts, ep, top_k)

    # Charge what the block actually costs -- the header, the "- " prefix and
    # the joining newline -- so a full block cannot overshoot budget_chars by
    # three characters per item plus the header.
    kept: list[int] = []
    used = len(_BLOCK_HEADER)
    for i in order:
        cost = len(texts[i]) + 3
        if used + cost > budget_chars:
            # `continue`, not `break`: a single oversized item used to discard
            # every item ranked after it, and because the emit list was in time
            # order the discarded ones were systematically the most recent.
            continue
        kept.append(i)
        used += cost
    if not kept:
        return "", {"n_eligible": a["n"], "n_used": 0, "chars": 0,
                    "future_items": a["future_items"],
                    "max_source_order": a["max_source_order"], "degraded": "",
                    "ranker": ranker}
    block = _BLOCK_HEADER + "\n" + "\n".join("- " + texts[i] for i in sorted(kept))
    return block, {"n_eligible": a["n"], "n_used": len(kept), "chars": len(block),
                   "future_items": a["future_items"],
                   "max_source_order": a["max_source_order"], "degraded": "",
                   "ranker": ranker}
