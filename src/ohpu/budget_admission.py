"""Wait for temporary reservations without changing any request or budget."""
from __future__ import annotations

import math
import time


def reserve_with_backpressure(reserve, snapshot, amount, *, budget_error,
                              timeout_seconds=55., poll_seconds=.05,
                              monotonic=time.monotonic, sleep=time.sleep):
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("invalid budget admission wait")
    deadline = monotonic() + timeout_seconds
    while True:
        try:
            return reserve(amount)
        except budget_error:
            state = snapshot()
            # Waiting cannot make settled spending disappear. Never shrink the
            # upper bound, reset the ledger, or release another request's ticket.
            if (state["spent_usd"] + amount > state["limit_usd"] + 1e-12
                    or monotonic() >= deadline):
                raise
            # The failed atomic transaction has already rolled back. Sleeping
            # outside it lets active requests settle using other connections.
            # If the reservation cleared before snapshot(), retry normally too.
            sleep(min(poll_seconds, max(0., deadline - monotonic())))
