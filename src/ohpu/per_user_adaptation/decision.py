"""Host-owned decision-model access for harness programs (Jev/SystemOne API, e.g. Cloudflare/clef-flash)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request
import uuid
from typing import Any, Mapping

DEFAULT_URL = "http://127.0.0.1:8795/v1/systemone"
DEFAULT_MODEL = "Cloudflare/clef-flash"
MAX_QUESTIONS = 16
MAX_OPTIONS = 12
MAX_RANK_ITEMS = 32
MAX_STATE_CHARS = 24_000
MAX_INSTRUCTION_CHARS = 1_000
MAX_CRITERION_CHARS = 400
QUESTION_TYPES = ("choice", "noul", "score")
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,47}")


def _text(value: Any, limit: int, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"decision {name} must be a nonempty string of at most {limit} characters")
    return value


def validate_questions(questions: Any) -> dict:
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"decision questions must be a mapping of 1 to {MAX_QUESTIONS} questions")
    clean = {}
    for key, question in questions.items():
        if not isinstance(key, str) or not _ID.fullmatch(key) or not isinstance(question, dict):
            raise ValueError("decision question ids must be identifiers mapped to question objects")
        if set(question) - {"type", "instructions", "criteria"} or question.get("type") not in QUESTION_TYPES:
            raise ValueError("decision questions take only type (choice|noul|score), instructions and criteria")
        item = {"type": question["type"]}
        if "instructions" in question:
            item["instructions"] = _text(question["instructions"], MAX_INSTRUCTION_CHARS, "instructions")
        criteria = question.get("criteria")
        if question["type"] == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS or any(
                    not isinstance(option, str) or not _ID.fullmatch(option) for option in criteria):
                raise ValueError(f"choice questions need 2 to {MAX_OPTIONS} identifier options")
            item["criteria"] = {option: _text(text, MAX_CRITERION_CHARS, "criterion") for option, text in criteria.items()}
        elif question["type"] == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_OPTIONS:
                raise ValueError(f"score questions need 2 to {MAX_OPTIONS} ordered levels")
            item["criteria"] = [_text(text, MAX_CRITERION_CHARS, "criterion") for text in criteria]
        elif criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"} or not criteria:
                raise ValueError("noul criteria may only describe true and false")
            item["criteria"] = {option: _text(text, MAX_CRITERION_CHARS, "criterion") for option, text in criteria.items()}
        clean[key] = item
    return clean


def encode_state(state: Any) -> Any:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if not text or len(text) > MAX_STATE_CHARS:
        raise ValueError(f"decision state must be nonempty and at most {MAX_STATE_CHARS} characters")
    return state


def normalize_answers(questions: Mapping[str, dict], answers: Any) -> dict:
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("decision model did not answer exactly the asked questions")
    result = {}
    for key, question in questions.items():
        probabilities = answers[key].get("probabilities") if isinstance(answers[key], dict) else None
        if question["type"] == "noul":
            expected = {"true", "false"}
            if isinstance(probabilities, dict) and set(map(str, probabilities)) == expected:
                probabilities = {str(option): value for option, value in probabilities.items()}
            else:
                true = next((answers[key].get(field) for field in ("noul", "probability") if isinstance(answers[key].get(field), (int, float))), None)
                if true is not None and 0 <= true <= 1:
                    probabilities = {"true": float(true), "false": 1.0 - float(true)}
        elif question["type"] == "choice":
            expected = set(question["criteria"])
        else:
            expected = {str(index) for index in range(len(question["criteria"]))}
            if isinstance(probabilities, list):
                probabilities = {str(index): value for index, value in enumerate(probabilities)}
            elif isinstance(probabilities, dict):
                probabilities = {str(option): value for option, value in probabilities.items()}
        if (not isinstance(probabilities, dict) or set(probabilities) != expected
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in probabilities.values())):
            raise ValueError("decision model returned invalid probabilities for " + key)
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("decision model returned zero probability mass for " + key)
        probabilities = {option: round(value / total, 6) for option, value in probabilities.items()}
        item = {"type": question["type"], "probabilities": probabilities}
        if question["type"] == "noul":
            item["probability"] = probabilities["true"]
        elif question["type"] == "choice":
            item["choice"] = max(sorted(probabilities), key=probabilities.get)
        else:
            item["score"] = round(sum(int(level) * value for level, value in probabilities.items()), 6)
        result[key] = item
    return result


def rank_questions(instructions: str, items: list) -> dict:
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_RANK_ITEMS:
        raise ValueError(f"rank needs 1 to {MAX_RANK_ITEMS} items")
    _text(instructions, MAX_INSTRUCTION_CHARS, "instructions")
    return {f"item_{index}": {"type": "noul", "instructions": f"{instructions} Item {index}: "
        + (item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True))[:MAX_CRITERION_CHARS]}
        for index, item in enumerate(items)}


class SystemOneClient:
    """Deterministic typed decisions from a Jev/SystemOne-compatible server."""

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, *, timeout_seconds: float = 120.0):
        self.url, self.model, self.timeout_seconds = url, model, timeout_seconds
        self.calls = 0
        self.input_tokens = 0

    def decide(self, state: Any, questions: Mapping[str, Any]) -> dict:
        clean = validate_questions(dict(questions))
        return normalize_answers(clean, self._answers(encode_state(state), clean))

    def _answers(self, state: Any, questions: dict) -> Any:
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())
        self.calls += 1
        self.input_tokens += int((payload.get("usage") or {}).get("input_tokens") or 0)
        return payload.get("answers")

    def rank(self, state: Any, items: list, instructions: str) -> dict:
        answers = self.decide(state, rank_questions(instructions, items))
        scores = [answers[f"item_{index}"]["probability"] for index in range(len(items))]
        return {"scores": scores, "order": sorted(range(len(items)), key=lambda index: (-scores[index], index))}


class RelayDecisionClient(SystemOneClient):
    """Sandbox side of a file relay: a generator's checker has no network, so it writes validated requests under
    ``directory`` and waits for the host's ``serve_relay`` to answer with the registered decision model."""

    def __init__(self, directory: str | Path, *, wait_seconds: float = 120.0, poll_seconds: float = 0.02):
        super().__init__(url="relay", model="relay")
        self.directory, self.wait_seconds, self.poll_seconds = Path(directory), wait_seconds, poll_seconds

    def _answers(self, state: Any, questions: dict) -> Any:
        self.directory.mkdir(exist_ok=True)
        name = uuid.uuid4().hex
        request, response = self.directory / f"request-{name}.json", self.directory / f"response-{name}.json"
        temporary = self.directory / f".request-{name}.tmp"
        temporary.write_text(json.dumps({"state": state, "questions": questions}))
        os.replace(temporary, request)
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            if response.is_file():
                answer = json.loads(response.read_text())
                if answer.get("error"):
                    raise RuntimeError("decision relay: " + str(answer["error"]))
                self.calls += 1
                return answer["answers"]
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"no decision-model answer from the host within {self.wait_seconds:.0f} s")


def serve_relay(directory: str | Path, client: SystemOneClient, stop, *, limit: int, log: str | Path | None = None,
                poll_seconds: float = 0.02) -> int:
    """Host side of ``RelayDecisionClient`` until ``stop`` (a threading.Event) is set. Re-validates every request,
    answers at most ``limit`` with ``client`` (later ones get an error) and optionally appends each exchange to ``log``.
    Returns the number of answered requests."""
    folder, served = Path(directory), 0
    while not stop.is_set():
        for path in sorted(folder.glob("request-*.json")) if folder.is_dir() else []:
            target = folder / path.name.replace("request-", "response-", 1)
            if target.exists():
                continue
            entry = {"request": path.name}
            try:
                request = json.loads(path.read_text())
                if served >= limit:
                    raise RuntimeError(f"the decision relay limit of {limit} calls per session is used up")
                questions = validate_questions(request.get("questions"))
                state = encode_state(request.get("state"))
                answers = client._answers(state, questions)
                normalize_answers(questions, answers)
                answer = {"answers": answers}
                served += 1
                entry.update(questions=sorted(questions), served=served)
            except Exception as error:
                answer = {"error": f"{type(error).__name__}: {error}"[:500]}
                entry.update(error=answer["error"])
            temporary = folder / f".{target.name}.tmp"
            temporary.write_text(json.dumps(answer))
            os.replace(temporary, target)
            if log is not None:
                with open(log, "a") as stream:
                    stream.write(json.dumps(entry) + "\n")
        stop.wait(poll_seconds)
    return served
