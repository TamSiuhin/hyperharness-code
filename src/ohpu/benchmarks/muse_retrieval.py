"""Source-bound span retrieval for MUSE, without changing other benchmark stores."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

from ohpu.per_user_adaptation.memory_backend import MemoryStore


def query_parts(request):
    if "\nOptions:\n" not in request:
        return [request.strip()]
    header, choices = request.split("\nOptions:\n", 1)
    question = header.split("\nUser query:\n", 1)[-1]
    choices = choices.rsplit("\nReturn only", 1)[0]
    options = [option.strip() for option in re.split(r"(?m)^[ABCDE]:\s*", choices)[1:]]
    return [question.strip(), *sorted(set(options))]


class MuseStore(MemoryStore):
    def build_index(self):
        self.index = sqlite3.connect(":memory:")
        self.index.execute("CREATE VIRTUAL TABLE spans USING fts5(text, tokenize='porter unicode61')")
        self.spans = []
        self.by_reference = {}
        for record in self.records.values():
            if record["collection"] != "observations":
                continue
            reference = self.reference(record)
            self.by_reference[reference] = record
            payload = json.loads(record["text"])
            messages = payload["messages"]
            for position, message in enumerate(messages):
                content = message["content"]
                for offset in range(0, len(content), 650):
                    snippet = content[offset:offset + 900]
                    neighbor = messages[position - 1] if position and message["role"] == "assistant" else None
                    self.spans.append({"ref": reference, "date": payload["date"], "session_id": payload["session_id"],
                        "role": message["role"], "turn": message["turn_index"],
                        "offset": message["character_offset"] + offset, "text": snippet,
                        "neighbor": (neighbor["role"] + ": " + neighbor["content"][-160:]) if neighbor else ""})
        self.index.executemany("INSERT INTO spans(rowid,text) VALUES (?,?)",
            [(index + 1, span["text"]) for index, span in enumerate(self.spans)])

    def search(self, *, collection=None, text=None, vector=None, limit=5, algorithm="tfidf"):
        if getattr(self, "task_kind", None) != "apply-user-multi":
            return super().search(collection=collection, text=text, vector=vector, limit=limit, algorithm=algorithm)
        if algorithm != "bm25" or vector is not None or collection not in (None, "observations"):
            return super().search(collection=collection, text=text, vector=vector, limit=limit, algorithm=algorithm)
        if not isinstance(text, str) or type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("MUSE search requires text and a bounded result limit")
        if not hasattr(self, "index"):
            self.build_index()
        request = getattr(self, "request", text)
        parts = query_parts(request)
        if text not in (request, *parts) and "\nOptions:\n" not in text and len(re.findall(r"(?m)^[ABCDE]:", text)) < 2:
            parts.append(text)
        ranks = defaultdict(float)
        matches_by_span = {}
        self.last_search_queries = []
        for part in sorted(set(parts)):
            terms = list(dict.fromkeys(re.findall(r"\w+", part.casefold())))
            if not terms:
                continue
            if len(terms) > 512:
                terms = [terms[index * (len(terms) - 1) // 511] for index in range(512)]
            groups = [terms[offset:offset + 128] for offset in range(0, len(terms), 128)]
            for group in groups:
                expression = " OR ".join('"' + term + '"' for term in group)
                self.last_search_queries.append(tuple(group))
                matches = self.index.execute("SELECT rowid,snippet(spans,0,'','',' ... ',64) FROM spans "
                    "WHERE spans MATCH ? ORDER BY bm25(spans),rowid LIMIT ?",
                    (expression, min(64, max(24, limit * 2)))).fetchall()
                for rank, (identifier, excerpt) in enumerate(matches, 1):
                    ranks[identifier - 1] += 1 / ((60 + rank) * len(groups))
                    previous = matches_by_span.get(identifier - 1)
                    if previous is None or rank < previous[0]:
                        matches_by_span[identifier - 1] = (rank, excerpt)
        grouped = defaultdict(list)
        for index, score in ranks.items():
            grouped[self.spans[index]["ref"]].append((score, index))
        selected = sorted(grouped, key=lambda reference: (-max(score for score, index in grouped[reference]), reference))[:limit]
        results = []
        for reference in selected:
            candidates = sorted(grouped[reference], key=lambda item: (-item[0], item[1]))
            excerpts, positions = [], set()
            for score, index in candidates:
                span = self.spans[index]
                identity = (span["turn"], span["offset"] // 1300)
                if identity in positions:
                    continue
                positions.add(identity)
                excerpts.append(f"{span['role']} turn={span['turn']} offset={span['offset']}: {matches_by_span[index][1]}")
                if len(excerpts) == 2:
                    break
            first = self.spans[candidates[0][1]]
            prefix = f"Session {first['session_id']} Date {first['date']}\n"
            if first["neighbor"]:
                prefix += first["neighbor"] + "\n"
            results.append({**self.view(self.by_reference[reference]), "text": (prefix + "\n".join(excerpts))[:1400],
                "score": candidates[0][0]})
        return results

    def close(self):
        if hasattr(self, "index"):
            self.index.close()
            del self.index

    def __del__(self):
        self.close()