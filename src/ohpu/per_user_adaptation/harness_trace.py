"""Compact host-side record of the memory operations a personal harness issues for one request.

Diagnostics only: the recorder observes ``memory_backend.execute`` callbacks and never changes program inputs.
"""
from __future__ import annotations

import json
import time

TEXT_CHARS = 400
SQL_CHARS = 1200
PARAMETERS = 16
REFERENCES = 64
STATE_KEYS = 32
COLUMNS = 24


def clip(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f"...[{len(text)} chars]"


def result_rows(value):
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return value["rows"], bool(value.get("truncated"))
    if isinstance(value, list):
        return value, False
    return None, False


class Recorder:
    def __init__(self):
        self.started = self.previous = time.monotonic()
        self.steps = []

    def record(self, kind, **details):
        now = time.monotonic()
        decision = details.get("decision") if isinstance(details.get("decision"), dict) else {}
        operation = decision.get("operation")
        # Time since the previous operation: the hook call that chose this step plus the backend operation.
        entry = {"step": details.get("step"), "operation": operation, "ms": round(1000 * (now - self.previous), 1)}
        self.previous = now
        if operation == "search":
            entry.update(collection=decision.get("collection"), algorithm=decision.get("algorithm", "tfidf"),
                limit=decision.get("limit", 5), text=clip(decision.get("text") or "", TEXT_CHARS))
        elif operation == "sql":
            parameters = decision.get("parameters") or []
            entry.update(sql=clip(decision.get("sql") or "", SQL_CHARS),
                parameters=[clip(item, TEXT_CHARS) for item in parameters[:PARAMETERS]]
                    if isinstance(parameters, list) else clip(parameters, TEXT_CHARS))
        state = decision.get("state")
        if isinstance(state, dict):
            entry.update(state_keys=sorted(str(key) for key in state)[:STATE_KEYS],
                state_chars=len(json.dumps(state, ensure_ascii=False, default=str)))
        if kind == "memory_backend_error":
            entry["error_type"] = details.get("error_type")
        result = details.get("result")
        if operation in ("finish", "read_all", "understand") and isinstance(result, dict):
            refs = result.get("source_refs") if isinstance(result.get("source_refs"), list) else []
            entry.update(output_chars=len(result.get("output") or ""), reference_count=len(refs),
                source_refs=[str(ref) for ref in refs[:REFERENCES]])
        else:
            rows, truncated = result_rows(result)
            if rows is not None:
                shown = [row for row in rows[:REFERENCES] if isinstance(row, dict)]
                entry.update(rows=len(rows), truncated=truncated,
                    refs=[str(row["ref"]) for row in shown if row.get("ref") is not None],
                    columns=sorted(str(key) for key in shown[0])[:COLUMNS] if shown else [])
                scores = [round(row["score"], 4) for row in shown
                          if type(row.get("score")) in (int, float)]
                if scores:
                    entry["scores"] = scores
        self.steps.append(entry)

    def export(self):
        return {"version": 1, "retrieval": bool(self.steps), "operations": [step["operation"] for step in self.steps],
            "elapsed_ms": round(1000 * (self.previous - self.started), 1), "steps": self.steps}
