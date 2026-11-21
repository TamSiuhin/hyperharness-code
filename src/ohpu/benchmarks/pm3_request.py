"""Public PM3 request contract shared by history training and native generation.

This module never receives an evaluator instance, verifier, preference label or
score. Generation retains the complete task; retrieval uses its public stimulus.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

PROFILE = "pm3-public-stimulus-v2"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prepare(prompt):
    text = str(prompt).strip()
    # Only top-level template headings delimit sections. A user's ordinary
    # Markdown headings are retained inside the corresponding stimulus.
    headings = list(re.finditer(
        r"(?m)^## (?:Current user query|User query|User question[^\n]*|"
        r"Prior conversation turns|Prior [^\n]*|Original [^\n]*|"
        r"User context at current moment|The user's actual[^\n]*|"
        r"Candidate[^\n]*|Your job|Output)\s*\n", text))
    pieces, output = [], ""
    for i, match in enumerate(headings):
        title = match.group().strip()[3:]
        body = text[match.end():headings[i + 1].start() if i + 1 < len(headings) else len(text)].strip()
        if title == "Output":
            output = body
        elif title != "Your job":
            pieces.append(title + "\n" + body)
    # Ranking includes public browsing context before its first section.
    if headings:
        preamble = text[:headings[0].start()]
        context = re.search(r"(?m)^User is currently browsing .+$", preamble)
        if context:
            pieces.insert(0, context.group())
    focus = "\n\n".join(pieces) if pieces else text
    properties = {}
    # Constrain only the object shape already requested in the public Output
    # block. This does not enforce a correct ordering or synthesize an answer.
    fields = set(re.findall(r'"(\w+)"\s*:', output))
    allowed = {"response", "reasoning", "ranked_indices", "ranked_indexes",
               "stopped_due_to", "recommendation", "hashtags"}
    answer_fields = {"response", "ranked_indices", "ranked_indexes", "recommendation"}
    if fields and fields <= allowed and fields & answer_fields:
        for field in sorted(fields):
            if field in {"ranked_indices", "ranked_indexes", "hashtags"}:
                item_type = "string" if field == "hashtags" else "integer"
                properties[field] = {"type": "array", "items": {"type": item_type}}
            else:
                properties[field] = {"type": "string"}
    response_format = ({"type": "json_schema", "json_schema": {
        "name": "pm3_public_response", "strict": True,
        "schema": {"type": "object", "properties": properties,
                   "required": list(properties), "additionalProperties": False}}} if properties else None)
    return {"query": focus, "response_format": response_format, "audit": {
        "request_profile": PROFILE, "stimulus_extracted": bool(pieces),
        "generation_characters": len(text), "retrieval_characters": len(focus),
        "structured_output": bool(response_format)}}
