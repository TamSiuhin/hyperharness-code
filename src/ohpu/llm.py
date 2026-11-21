"""Shared model-call primitives.

This is the layer MemoHarness keeps in `harbor/agent.py`: build the client,
speak the tool protocol, retry, and account for cost. Nothing here knows about
personalization -- that lives in MemoHarness's bundle and the OHPU executor
adapter (ohpu.per_user_adaptation.executor).

Two properties this module must have for the experiment to be trustworthy:

1. DETERMINISM WHERE CLAIMED. Every call is keyed by a hash of (model, messages,
   tools, decoding params) and cached on disk. Re-running an experiment with one
   condition changed re-issues only the calls that condition changed. This also
   means a crashed sweep resumes for free.
2. COST IS OBSERVABLE. Token counts are accumulated per (condition, corpus) so
   E5 (cost accounting) is a byproduct of running anything at all, not a
   separate instrumented pass.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)

# USD per 1M tokens, as published for the models this project actually calls.
PRICES = {
    "gpt-4.1":       (2.00, 8.00),
    "gpt-4.1-mini":  (0.40, 1.60),
    "gpt-4.1-nano":  (0.10, 0.40),
    "gpt-5.4":       (1.25, 10.00),
    "gpt-5.4-mini":  (0.25, 2.00),
    "gpt-5.6-luna":  (0.20, 1.20),
}
# Reservations deliberately ignore provider prompt-cache discounts and price
# every input token as a cache write. Settlement uses the provider's observed
# cached/cache-write breakdown when present; missing metadata stays at this
# conservative upper bound. Input is also padded for tokenizer/framing
# uncertainty, and Luna requests whose bound crosses the long-context tier use
# its worst-case input rate for the hard budget.
INPUT_BUDGET_MULTIPLIER = 1.25
CACHED_INPUT_MULTIPLIER = 0.10
CACHE_WRITE_INPUT_MULTIPLIER = 1.25
LUNA_LONG_CONTEXT_INPUT_TOKENS = 272_000
LUNA_LONG_CONTEXT_INPUT_MULTIPLIER = 2.0
LUNA_LONG_CONTEXT_OUTPUT_MULTIPLIER = 1.5
GLOBAL_BUDGET_DB_ENV = "OHPU_GLOBAL_BUDGET_DB"
GLOBAL_BUDGET_USD_ENV = "OHPU_GLOBAL_BUDGET_USD"
# Models whose per-token price this file does not know. Their tokens are still
# counted; the USD figure is reported as unpriced rather than silently as 0.00,
# because a $0.00 next to 4M tokens reads as "free" instead of "unknown".
UNPRICED: set[str] = set()

# Sentinel for "send neither knob and take the provider's own default". The
# pinned PM3 evaluator's OpenAI branch passes only model and messages, so its
# judge is never truncated by a caller-chosen budget and never has its
# reasoning effort overridden. A caller that must match the official judge
# passes max_tokens=None and reasoning_effort=PROVIDER_DEFAULT.
PROVIDER_DEFAULT = "__provider_default__"
# An uncapped request still needs a finite cost bound for the budget guard.
# The provider caps output at the model's own limit; this bound only sizes the
# reservation, which settles to the observed cost immediately afterwards.
UNCAPPED_OUTPUT_RESERVATION_TOKENS = 32_768
# Registered Qwen3.5-9B non-thinking sampling. The low-temperature profile keeps
# the model-card nucleus and presence penalty but samples near-greedily to reduce
# answer variance; pure greedy decoding risks repetition in long memory outputs.
QWEN_SAMPLING_PROFILES = {
    "qwen35-instruct-v1": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "top_k": 20, "min_p": 0.0},
    "qwen35-instruct-lowtemp-v1": {"temperature": 0.1, "top_p": 0.8, "presence_penalty": 1.5, "top_k": 20, "min_p": 0.0},
}


class APIBudgetExceeded(RuntimeError):
    """Raised before a request whose conservative reservation exceeds budget."""


def _price(model: str, prompt_tokens: int) -> tuple[float, float]:
    prices = PRICES.get(model)
    if prices is None:
        return 0.0, 0.0
    pin, pout = prices
    if model == "gpt-5.6-luna" and prompt_tokens > LUNA_LONG_CONTEXT_INPUT_TOKENS:
        pin *= LUNA_LONG_CONTEXT_INPUT_MULTIPLIER
        pout *= LUNA_LONG_CONTEXT_OUTPUT_MULTIPLIER
    return pin, pout


def _budget_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Conservative cost for reservations and the global cap."""
    if model not in PRICES:
        return 0.0
    pin, pout = _price(model, prompt_tokens)
    return (
        prompt_tokens / 1e6 * pin * INPUT_BUDGET_MULTIPLIER
        + completion_tokens / 1e6 * pout
    )


def _metered_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Official realized cost from the provider's input-token breakdown."""

    if model not in PRICES:
        return 0.0
    cached_tokens = int(cached_tokens)
    cache_write_tokens = int(cache_write_tokens)
    ordinary_tokens = int(prompt_tokens) - cached_tokens - cache_write_tokens
    if min(ordinary_tokens, cached_tokens, cache_write_tokens) < 0:
        raise ValueError("invalid prompt token detail breakdown")
    pin, pout = _price(model, int(prompt_tokens))
    billed_input = (
        ordinary_tokens
        + CACHED_INPUT_MULTIPLIER * cached_tokens
        + CACHE_WRITE_INPUT_MULTIPLIER * cache_write_tokens
    )
    return billed_input / 1e6 * pin + completion_tokens / 1e6 * pout


def _is_clearly_unbilled(exc: BaseException) -> bool:
    """Errors that are local/auth/precondition failures, safe to release."""
    text = str(exc).lower()
    return any(marker in text for marker in (
        "invalid_api_key", "incorrect api key", "authentication",
        "permission denied", "invalid_request_error", "invalid_request",
        "bad request", "status code: 400", "http 400",
    ))


_SQLITE_BUSY_TIMEOUT_MS = 60_000
_SQLITE_BUSY_CODES = {
    int(getattr(sqlite3, "SQLITE_BUSY", 5)),
    int(getattr(sqlite3, "SQLITE_LOCKED", 6)),
}


def _is_sqlite_busy(exc: sqlite3.OperationalError) -> bool:
    """Return whether *exc* is SQLite's transient lock/busy condition."""

    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) in _SQLITE_BUSY_CODES:
        return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def _retry_sqlite_busy(operation, *, timeout_seconds: float = 60.0):
    """Retry one idempotent SQLite initialization operation while it is busy.

    ``sqlite3.connect(timeout=...)`` normally installs a busy handler, but
    concurrent ``PRAGMA journal_mode=WAL`` calls can still fail immediately on
    some SQLite builds.  Startup operations are idempotent, so retry them until
    the same bounded deadline instead of making a parallel worker fail.
    """

    deadline = time.monotonic() + timeout_seconds
    delay = 0.01
    while True:
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_busy(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def _open_sqlite_wal(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open a process-local connection and establish WAL without a race."""

    conn = sqlite3.connect(os.fspath(path), timeout=60.0)
    try:
        # Install SQLite's busy handler before the operation that needs the
        # exclusive journal-mode transition.  The explicit retry covers builds
        # where that PRAGMA bypasses the handler during simultaneous startup.
        conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        _retry_sqlite_busy(
            lambda: conn.execute("PRAGMA journal_mode=WAL").fetchone()
        )
    except BaseException:
        conn.close()
        raise
    return conn


class SQLiteBudgetLedger:
    """Atomic cross-process API budget ledger.

    Reservations intentionally survive a process crash (fail closed): a
    resumed sweep cannot spend through a request that may have reached the
    provider before its process died.
    """

    def __init__(self, path: str | os.PathLike[str], limit_usd: float, *, reservation_wait_seconds: float = 0.):
        self.path = os.fspath(path)
        self.limit_usd = float(limit_usd)
        self.reservation_wait_seconds = float(reservation_wait_seconds)
        if not 0 <= self.reservation_wait_seconds <= 55:
            raise ValueError("budget reservation wait must be between 0 and 55 seconds")
        if self.limit_usd <= 0:
            raise ValueError("global API budget must be positive")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._local = threading.local()
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = _open_sqlite_wal(self.path)
            self._local.conn = conn
        return conn

    def _init(self) -> None:
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS api_budget ("
            "id INTEGER PRIMARY KEY CHECK (id=1), limit_usd REAL NOT NULL, "
            "spent_usd REAL NOT NULL DEFAULT 0, reserved_usd REAL NOT NULL DEFAULT 0, "
            "calls INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS api_budget_reservations ("
            "ticket_id TEXT PRIMARY KEY, amount_usd REAL NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'open')"
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT limit_usd FROM api_budget WHERE id=1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO api_budget(id, limit_usd) VALUES (1, ?)",
                (self.limit_usd,),
            )
        elif abs(float(row[0]) - self.limit_usd) > 1e-9:
            conn.rollback()
            raise ValueError(
                f"global budget ledger already has limit ${float(row[0]):.2f}; "
                f"requested ${self.limit_usd:.2f}"
            )
        conn.commit()

    def reserve_ticket(self, amount_usd: float) -> str | None:
        if self.reservation_wait_seconds:
            from ohpu.budget_admission import reserve_with_backpressure
            return reserve_with_backpressure(self._reserve_ticket_once, self.snapshot, amount_usd,
                budget_error=APIBudgetExceeded, timeout_seconds=self.reservation_wait_seconds)
        return self._reserve_ticket_once(amount_usd)

    def _reserve_ticket_once(self, amount_usd: float) -> str | None:
        amount_usd = float(amount_usd)
        if amount_usd <= 0:
            return None
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT limit_usd, spent_usd, reserved_usd FROM api_budget WHERE id=1"
        ).fetchone()
        if row is None:
            conn.rollback()
            raise RuntimeError("global API budget ledger is not initialized")
        limit, spent, reserved = map(float, row)
        if spent + reserved + amount_usd > limit + 1e-12:
            conn.rollback()
            raise APIBudgetExceeded(
                "global API budget guard refused the next request: "
                f"spent=${spent:.4f}, reserved=${reserved:.4f}, "
                f"next_upper_bound=${amount_usd:.4f}, limit=${limit:.2f}"
            )
        conn.execute(
            "UPDATE api_budget SET reserved_usd=reserved_usd+? WHERE id=1",
            (amount_usd,),
        )
        ticket_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO api_budget_reservations(ticket_id, amount_usd) VALUES (?, ?)",
            (ticket_id, amount_usd),
        )
        conn.commit()
        return ticket_id

    def reserve(self, amount_usd: float) -> None:
        """Legacy reserve API; callers needing settlement identity use tickets."""
        self.reserve_ticket(amount_usd)

    def _resolve_ticket(
        self, conn: sqlite3.Connection, reservation_usd: float,
        ticket_id: str | None,
    ) -> str:
        if ticket_id:
            row = conn.execute(
                "SELECT amount_usd FROM api_budget_reservations "
                "WHERE ticket_id=? AND status='open'", (ticket_id,)
            ).fetchone()
            if row is None or abs(float(row[0]) - reservation_usd) > 1e-9:
                raise RuntimeError("global API budget reservation ticket is invalid")
            return ticket_id
        # Compatibility for direct users of the old API. Refuse ambiguity so
        # one process can never settle another process's equal-sized ticket.
        rows = conn.execute(
            "SELECT ticket_id FROM api_budget_reservations "
            "WHERE status='open' AND abs(amount_usd-?) <= 1e-9",
            (reservation_usd,),
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError("global API budget reservation ticket is required")
        return str(rows[0][0])

    def settle(
        self, reservation_usd: float, budget_usd: float,
        ticket_id: str | None = None,
    ) -> None:
        reservation_usd = float(reservation_usd)
        budget_usd = float(budget_usd)
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            ticket_id = self._resolve_ticket(conn, reservation_usd, ticket_id)
        except Exception:
            conn.rollback()
            raise
        row = conn.execute(
            "SELECT limit_usd, spent_usd, reserved_usd FROM api_budget WHERE id=1"
        ).fetchone()
        if row is None or float(row[2]) + 1e-9 < reservation_usd:
            conn.rollback()
            raise RuntimeError("global API budget reservation is missing")
        if budget_usd > reservation_usd + 1e-12:
            conn.rollback()
            raise APIBudgetExceeded(
                "global API budget settlement exceeded its request reservation: "
                f"reservation=${reservation_usd:.4f}, actual=${budget_usd:.4f}"
            )
        if float(row[1]) + budget_usd > float(row[0]) + 1e-12:
            conn.rollback()
            raise APIBudgetExceeded(
                "global API budget settle exceeded the hard limit: "
                f"spent=${float(row[1]):.4f}, actual=${budget_usd:.4f}, "
                f"limit=${float(row[0]):.2f}"
            )
        conn.execute(
            "UPDATE api_budget SET reserved_usd=reserved_usd-?, "
            "spent_usd=spent_usd+?, calls=calls+1 WHERE id=1",
            (reservation_usd, budget_usd),
        )
        conn.execute(
            "DELETE FROM api_budget_reservations WHERE ticket_id=?",
            (ticket_id,),
        )
        conn.commit()

    def release(self, reservation_usd: float, ticket_id: str | None = None) -> None:
        reservation_usd = float(reservation_usd)
        if reservation_usd <= 0:
            return
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            ticket_id = self._resolve_ticket(conn, reservation_usd, ticket_id)
        except Exception:
            conn.rollback()
            raise
        row = conn.execute(
            "SELECT reserved_usd FROM api_budget WHERE id=1"
        ).fetchone()
        if row is None or float(row[0]) + 1e-9 < reservation_usd:
            conn.rollback()
            raise RuntimeError("global API budget reservation is missing")
        conn.execute(
            "UPDATE api_budget SET reserved_usd=reserved_usd-? WHERE id=1",
            (reservation_usd,),
        )
        conn.execute(
            "DELETE FROM api_budget_reservations WHERE ticket_id=?",
            (ticket_id,),
        )
        conn.commit()

    def snapshot(self) -> dict[str, float | int]:
        row = self._conn().execute(
            "SELECT limit_usd, spent_usd, reserved_usd, calls FROM api_budget WHERE id=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("global API budget ledger is not initialized")
        return {
            "limit_usd": float(row[0]), "spent_usd": float(row[1]),
            "reserved_usd": float(row[2]), "calls": int(row[3]),
        }


class _Reservation(float):
    """A numeric upper bound carrying this request's unique ledger identity."""

    def __new__(cls, amount: float):
        value = super().__new__(cls, amount)
        value.ticket_id = None
        value.preserved = False
        return value


@dataclass
class Usage:
    calls: int = 0
    failed_calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    prompt_token_detail_calls: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    usd: float = 0.0
    budget_usd: float = 0.0
    max_usd: float | None = None
    ledger: SQLiteBudgetLedger | None = field(default=None, repr=False)
    reserved_usd: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.ledger is not None:
            return
        db_path = os.environ.get(GLOBAL_BUDGET_DB_ENV)
        limit = os.environ.get(GLOBAL_BUDGET_USD_ENV)
        if db_path and limit:
            self.ledger = SQLiteBudgetLedger(db_path, float(limit))

    @staticmethod
    def _cost(model: str, pt: int, ct: int) -> float:
        if model not in PRICES:
            return 0.0
        pin, pout = _price(model, pt)
        return pt / 1e6 * pin + ct / 1e6 * pout

    @staticmethod
    def _budget_cost(model: str, pt: int, ct: int) -> float:
        return _budget_cost(model, pt, ct)

    def reserve(self, model: str, pt_upper: int, ct_upper: int) -> float:
        """Reserve a conservative request upper bound across concurrent calls."""

        if self.max_usd is None and self.ledger is None:
            return 0.0
        if model not in PRICES:
            raise APIBudgetExceeded(
                f"cannot enforce API budget for unpriced model {model!r}"
            )
        reservation = _Reservation(self._budget_cost(model, pt_upper, ct_upper))
        with _LOCK:
            projected = self.budget_usd + self.reserved_usd + reservation
            if self.max_usd is not None and projected > self.max_usd:
                raise APIBudgetExceeded(
                    "API budget guard refused the next request: "
                    f"spent=${self.budget_usd:.4f}, reserved=${self.reserved_usd:.4f}, "
                    f"next_upper_bound=${reservation:.4f}, "
                    f"limit=${self.max_usd:.2f}"
                )
            if self.max_usd is not None:
                self.reserved_usd += reservation
        try:
            if self.ledger is not None:
                reservation.ticket_id = self.ledger.reserve_ticket(reservation)
        except Exception:
            with _LOCK:
                if self.max_usd is not None:
                    self.reserved_usd = max(0.0, self.reserved_usd - reservation)
            raise
        return reservation

    def release(self, reservation: float) -> None:
        if reservation <= 0:
            return
        # A provider response was received but ledger settlement failed (for
        # example because the observed usage exceeded its conservative bound).
        # Keep that reservation stranded so a retry/resume cannot spend past a
        # hard cap. This is the intentional fail-closed crash semantics.
        if getattr(reservation, "preserved", False):
            return
        if self.ledger is not None:
            ticket_id = getattr(reservation, "ticket_id", None)
            if ticket_id is None:
                raise RuntimeError("global API budget reservation ticket is missing")
            self.ledger.release(reservation, ticket_id=ticket_id)
        with _LOCK:
            if self.max_usd is not None:
                self.reserved_usd = max(0.0, self.reserved_usd - reservation)

    def settle(
        self, model: str, pt: int, ct: int, reservation: float,
        *, budget_cost: float | None = None,
        failed: bool = False,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        prompt_token_details_available: bool = False,
    ) -> None:
        cached_tokens = int(cached_tokens or 0)
        cache_write_tokens = int(cache_write_tokens or 0)
        details_valid = (
            prompt_token_details_available
            and cached_tokens >= 0
            and cache_write_tokens >= 0
            and cached_tokens + cache_write_tokens <= pt
        )
        if not details_valid:
            cached_tokens = 0
            cache_write_tokens = 0
        realized_cost = (
            _metered_cost(
                model,
                pt,
                ct,
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            if details_valid else self._cost(model, pt, ct)
        )
        if budget_cost is None:
            budget_cost = (
                realized_cost
                if details_valid
                else self._budget_cost(model, pt, ct)
            )
        else:
            budget_cost = float(budget_cost)
        if self.ledger is not None:
            ticket_id = getattr(reservation, "ticket_id", None)
            if ticket_id is None and reservation > 0:
                raise RuntimeError("global API budget reservation ticket is missing")
            try:
                self.ledger.settle(reservation, budget_cost, ticket_id=ticket_id)
            except APIBudgetExceeded:
                if isinstance(reservation, _Reservation):
                    reservation.preserved = True
                raise
            except Exception as exc:
                if isinstance(reservation, _Reservation):
                    reservation.preserved = True
                raise APIBudgetExceeded(
                    "global API budget settlement failed; reservation preserved"
                ) from exc
        with _LOCK:
            if self.max_usd is not None:
                self.reserved_usd = max(0.0, self.reserved_usd - reservation)
            self.calls += 1
            self.failed_calls += int(failed)
            self.prompt_tokens += pt
            self.cached_tokens += cached_tokens
            self.cache_write_tokens += cache_write_tokens
            self.prompt_token_detail_calls += int(details_valid)
            self.completion_tokens += ct
            if model in PRICES:
                self.usd += realized_cost
                self.budget_usd += budget_cost
            else:
                UNPRICED.add(model)

    def add(self, model: str, pt: int, ct: int) -> None:
        reservation = self.reserve(model, pt, ct) if self.ledger is not None else 0.0
        self.settle(model, pt, ct, reservation)

    def settle_reserved(self, model: str, reservation: float,
                        pt_upper: int, ct_upper: int) -> None:
        """Consume a failed/ambiguous attempt at its full reserved bound."""
        self.settle(
            model, pt_upper, ct_upper, reservation,
            budget_cost=reservation,
            failed=True,
        )

    def hit(self) -> None:
        with _LOCK:
            self.cached += 1

    def as_dict(self) -> dict[str, Any]:
        d = {"calls": self.calls, "cache_hits": self.cached,
             "prompt_tokens": self.prompt_tokens,
             "cached_tokens": self.cached_tokens,
             "cache_write_tokens": self.cache_write_tokens,
             "prompt_token_detail_calls": self.prompt_token_detail_calls,
             "completion_tokens": self.completion_tokens,
             "failed_calls": self.failed_calls,
             "reasoning_tokens": self.reasoning_tokens,
             "usd": round(self.usd, 4),
             "budget_usd": round(self.budget_usd, 4)}
        if UNPRICED:
            d["usd_note"] = "excludes " + ",".join(sorted(UNPRICED)) + " (price unknown)"
        return d


class Cache:
    """SQLite because a sweep runs many threads and one file beats 50k files."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = _open_sqlite_wal(self.path)
            self._local.c = c
        return c

    def _init(self) -> None:
        c = self._conn()

        def initialize() -> None:
            try:
                c.execute(
                    "CREATE TABLE IF NOT EXISTS kv "
                    "(k TEXT PRIMARY KEY, v TEXT)"
                )
                c.commit()
            except BaseException:
                c.rollback()
                raise

        _retry_sqlite_busy(initialize)

    def get(self, k: str) -> Optional[dict]:
        r = self._conn().execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k: str, v: dict) -> None:
        c = self._conn()
        c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)",
                  (k, json.dumps(v, ensure_ascii=False)))
        c.commit()


def model_api_destination(expected: str | None = None) -> str:
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    endpoint = azure_endpoint or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("model API endpoint must be an HTTPS URL without credentials or query parameters")
    if azure_endpoint:
        if not parsed.hostname.endswith(".openai.azure.com") or parsed.path not in {"", "/"}:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be an Azure OpenAI resource root URL")
    elif parsed.hostname != "api.openai.com" or parsed.path not in {"", "/", "/v1", "/v1/"}:
        raise ValueError("unexpected model API destination")
    destination = f"https://{parsed.netloc.lower()}"
    if expected is not None and destination != expected.rstrip("/"):
        raise ValueError("model API destination does not match the run authorization")
    return destination


@dataclass
class LLM:
    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "low"      # gpt-5 family only; forced to "none" with tools
    cache: Optional[Cache] = None
    usage: Usage = field(default_factory=Usage)
    max_retries: int = 6
    _client: Any = None
    request_timeout_seconds: Optional[float] = 180.0
    connection_keepalive: bool = False
    enable_thinking: Optional[bool] = None
    repetition_penalty: Optional[float] = None
    sampling_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if self.request_timeout_seconds is not None and (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not 0 < self.request_timeout_seconds < float("inf")
        ):
            raise ValueError("request_timeout_seconds must be positive and finite, or None")
        if type(self.connection_keepalive) is not bool:
            raise ValueError("connection_keepalive must be a boolean")
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean or None")
        if self.repetition_penalty is not None and (
                type(self.repetition_penalty) not in (int, float) or not 0 < self.repetition_penalty < float("inf")):
            raise ValueError("repetition_penalty must be positive and finite, or None")
        if self.sampling_profile is not None and (
                self.sampling_profile not in QWEN_SAMPLING_PROFILES or self.model != "Qwen3.5-9B"
                or self.enable_thinking is not False or self.repetition_penalty not in (None, 1.0)):
            raise ValueError("Qwen sampling profiles (" + ", ".join(QWEN_SAMPLING_PROFILES)
                + ") require Qwen3.5-9B, thinking disabled and repetition penalty 1.0")
        if self._client is None:
            client_options = {}
            if self.connection_keepalive:
                import socket
                from openai import DefaultHttpxClient, _base_client

                http_module = getattr(_base_client, "httpx2", None) or _base_client.httpx
                socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
                for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4)):
                    if hasattr(socket, name):
                        socket_options.append((socket.IPPROTO_TCP, getattr(socket, name), value))
                client_options["http_client"] = DefaultHttpxClient(
                    transport=http_module.HTTPTransport(retries=0, socket_options=socket_options),
                    timeout=self.request_timeout_seconds,
                )
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if endpoint:
                model_api_destination()
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_version=os.environ.get(
                        "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
                    ),
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    timeout=self.request_timeout_seconds,
                    max_retries=0,
                    **client_options,
                )
            else:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"], timeout=self.request_timeout_seconds, max_retries=0,
                    **client_options,
                )

    # -- the single entry point -------------------------------------------
    def chat(self, messages: list[dict], *, model: Optional[str] = None,
             tools: Optional[list[dict]] = None, temperature: float = 0.0,
             top_p: float = 1.0, max_tokens: Optional[int] = 1024,
             tool_choice: Any = None, seed: Optional[int] = 7,
             reasoning_effort: Optional[str] = None,
             response_format: Any = None, max_retries: Optional[int] = None) -> dict:
        attempts = self.max_retries if max_retries is None else max_retries
        if type(attempts) is not int or attempts < 1:
            raise ValueError("max_retries must be a positive attempt count")
        m = model or self.model
        if self.sampling_profile is not None and m != self.model:
            raise ValueError("sampling profile cannot change its registered model")
        five = m.startswith(("gpt-5", "gpt-6", "o3", "o4"))
        payload: dict[str, Any] = {"model": m, "messages": messages}
        if seed is not None:
            payload["seed"] = seed

        if five:
            # Measured against gpt-5.6-luna, 2026-08:
            #  * max_tokens is rejected outright; the budget is max_completion_tokens
            #    and REASONING TOKENS COME OUT OF IT (31-105 on a one-word answer),
            #    so a budget sized for gpt-4.1-mini returns empty content;
            #  * temperature and top_p are not merely ignored, they 400 -- even at
            #    the default value. The sampling knobs are therefore NOT
            #    EXPRESSIBLE on this model at all. That is the capability-degradation
            #    problem in 5.7/E7 with a name on it;
            #  * function tools require reasoning_effort="none" on
            #    /v1/chat/completions, so the runner downgrades and records it
            #    rather than failing the call.
            # max_tokens=None omits the budget entirely, and PROVIDER_DEFAULT
            # omits the effort, so a caller can reproduce a request that sends
            # neither -- the shape the official evaluator's judge uses.
            if max_tokens is not None:
                payload["max_completion_tokens"] = max_tokens if m.startswith("gpt-6") else max(max_tokens, 256)
            eff = (
                self.reasoning_effort if reasoning_effort is None
                else reasoning_effort
            )
            if tools:
                eff = "none"
            if eff != PROVIDER_DEFAULT:
                payload["reasoning_effort"] = eff
        else:
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
            payload["top_p"] = top_p
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        if self.enable_thinking is not None:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if self.repetition_penalty is not None:
            payload.setdefault("extra_body", {})["repetition_penalty"] = self.repetition_penalty
        if self.sampling_profile in QWEN_SAMPLING_PROFILES:
            profile = QWEN_SAMPLING_PROFILES[self.sampling_profile]
            payload.update(temperature=profile["temperature"], top_p=profile["top_p"],
                presence_penalty=profile["presence_penalty"])
            payload.setdefault("extra_body", {}).update(top_k=profile["top_k"], min_p=profile["min_p"], repetition_penalty=1.0)

        cache_request = {
            "payload": payload,
            "endpoint": str(getattr(self._client, "base_url", "")),
            "api_version": getattr(self._client, "default_query", {}).get("api-version"),
        }
        key = hashlib.sha256(
            json.dumps(cache_request, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.usage.hit()
                # Per-subsystem accounting (for example the daily memory
                # updater) cannot infer cache hits from token usage: a cached
                # response legitimately reports the original token counts.
                # Mark the returned copy without persisting the marker.
                cached = dict(hit)
                cached["_cache_hit"] = True
                return cached

        serialized_bytes = len(
            json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        )
        # BPE token count cannot exceed UTF-8 bytes for ordinary text. The
        # fixed allowance covers chat framing and provider-side message tokens.
        input_token_upper = serialized_bytes + 2048
        output_token_upper = int(
            payload.get("max_completion_tokens")
            or payload.get("max_tokens")
            or UNCAPPED_OUTPUT_RESERVATION_TOKENS
        )
        last: Exception | None = None
        for attempt in range(attempts):
            reservation = self.usage.reserve(
                m, input_token_upper, output_token_upper
            )
            try:
                try:
                    r = self._client.chat.completions.create(**payload)
                    msg = r.choices[0].message
                    prompt_details = getattr(
                        r.usage, "prompt_tokens_details", None
                    )
                    fields_set = getattr(prompt_details, "model_fields_set", None)

                    def detail_field(name: str) -> tuple[bool, int]:
                        if prompt_details is None:
                            return False, 0
                        if isinstance(prompt_details, dict):
                            present = name in prompt_details
                            value = prompt_details.get(name)
                        else:
                            present = (
                                name in fields_set
                                if fields_set is not None
                                else hasattr(prompt_details, name)
                            )
                            value = getattr(prompt_details, name, None)
                        if not present or value is None:
                            return False, 0
                        try:
                            return True, int(value)
                        except (TypeError, ValueError):
                            # Token metadata is accounting evidence, not a
                            # reason to repeat a provider call whose response
                            # was already received. Malformed metadata falls
                            # back to the all-cache-write budget settlement.
                            return False, 0

                    cached_present, cached_tokens = detail_field("cached_tokens")
                    write_present, cache_write_tokens = detail_field(
                        "cache_write_tokens"
                    )
                    token_details_available = cached_present or write_present
                    if (
                        cached_tokens < 0
                        or cache_write_tokens < 0
                        or cached_tokens + cache_write_tokens
                        > int(r.usage.prompt_tokens)
                    ):
                        token_details_available = False
                        cached_tokens = 0
                        cache_write_tokens = 0
                    out = {
                        "content": msg.content or "",
                        "tool_calls": [
                            {"name": tc.function.name, "arguments": tc.function.arguments}
                            for tc in (msg.tool_calls or [])
                        ],
                        "finish_reason": r.choices[0].finish_reason,
                        "model": m,
                        "usage": {"prompt": r.usage.prompt_tokens,
                                  "completion": r.usage.completion_tokens,
                                  "cached_tokens": cached_tokens,
                                  "cache_write_tokens": cache_write_tokens,
                                  "prompt_token_details_available": (
                                      token_details_available
                                  )},
                    }
                    rt = getattr(getattr(r.usage, "completion_tokens_details", None),
                                 "reasoning_tokens", 0) or 0
                    out["usage"]["reasoning"] = rt
                    self.usage.settle(
                        m,
                        r.usage.prompt_tokens,
                        r.usage.completion_tokens,
                        reservation,
                        cached_tokens=cached_tokens,
                        cache_write_tokens=cache_write_tokens,
                        prompt_token_details_available=token_details_available,
                    )
                    with _LOCK:
                        self.usage.reasoning_tokens += rt
                    if self.cache:
                        try:
                            self.cache.put(key, out)
                        except Exception as exc:  # cache is only an optimization
                            _LOG.warning("LLM response cache write failed: %s", exc)
                    out["_cache_hit"] = False
                    return out
                except APIBudgetExceeded:
                    raise
                except Exception as e:  # noqa: BLE001 - retry on anything transient
                    last = e
                    if _is_clearly_unbilled(e):
                        self.usage.release(reservation)
                        raise
                    self.usage.settle_reserved(
                        m, reservation, input_token_upper, output_token_upper
                    )
                    if attempt + 1 < attempts:
                        time.sleep(min(2 ** attempt, 30) * (0.7 + 0.6 * random.random()))
            except APIBudgetExceeded:
                raise
        raise RuntimeError(f"LLM call failed after {attempts} tries: {last}") from last


def load_env(path: str = ".env") -> None:
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
