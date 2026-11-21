"""Shared Codex invocation helpers used by the supplied dataset hosts."""

import hashlib
import importlib.util
from pathlib import Path
import re
import statistics
import sys

BASE_PATH = Path(__file__).with_name("base_runner.py")
if not BASE_PATH.exists():
    BASE_PATH = Path(__file__).with_name("run_lamp_skill.py")
SPEC = importlib.util.spec_from_file_location("hyperharness_base_runner", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

SKILL = "history-to-harness"
GATE_SAMPLES = 4000
read, write, now, load_credentials = base.read, base.write, base.now, base.load_credentials
RATE_LIMIT_LINE = re.compile(r"^ERROR: (?:Reconnecting\.\.\. \d+/\d+|(?:Error running remote compact task: )?"
    r"stream disconnected before completion: .*exceeded (?:token )?rate limit.*)$", re.IGNORECASE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(out, bundle, prompt, *, timeout, editable=(), retries=0, reasoning_effort=None):
    source = str(Path(out) / "source_snapshot/src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from ohpu.per_user_adaptation import memory_evolution as evolution

    proposer = evolution.NativeCodexProposer(bundle, model=read(out / "plan.json")["optimizer_model"], timeout_seconds=timeout,
        reasoning_effort=reasoning_effort)
    for path in bundle.rglob("*"):
        relative = str(path.relative_to(bundle))
        if path.is_file() and relative not in editable and not path.is_relative_to(bundle / "skill_draft"):
            path.chmod(0o444)
            proposer.evidence[relative] = sha(path)
    if (bundle / "skill_draft" / SKILL / "assets/scaffold.py").exists():
        path = bundle / "skill_draft" / SKILL / "assets/scaffold.py"
        proposer.evidence[str(path.relative_to(bundle))] = sha(path)
    proposer._verify_evidence()
    result = {"outcome": "completed"}
    try:
        if not proposer.session(prompt, retries=retries) and "harness.py" in editable:
            result = {"outcome": "empty", "error": "Codex left harness.py unchanged"}
    except RuntimeError as error:
        if not base.timed_out(bundle):
            raise
        result = {"outcome": "timeout", "error": str(error)}
    if proposer.empty_attempts:
        result["empty_attempts"] = proposer.empty_attempts
    proposer._verify_evidence()
    return result


def gate_statistics(gains, *, level, samples=GATE_SAMPLES, seed=0):
    """Equal-user mean gain and its one-sided user-cluster bootstrap lower bound at ``level``."""
    import random

    values = [gains[user] for user in sorted(gains)]
    generator = random.Random(seed)
    draws = sorted(statistics.fmean(generator.choices(values, k=len(values))) for _ in range(samples))
    return {"mean_gain": statistics.fmean(values), "lower_bound": draws[int(samples * level)], "level": level,
        "users_better": sum(value > 0 for value in values), "users_worse": sum(value < 0 for value in values)}


def failed_only_on_rate_limit(directory):
    failed = False
    for path in sorted(directory.glob("workspace/controller_artifacts/controller/call-*.command.json")):
        command = read(path)
        if command.get("returncode") is None or command.get("timed_out"):
            return False
        if command["returncode"] == 0:
            continue
        output = path.with_name(path.name.replace(".command.json", ".output.log")).read_text(errors="replace")
        errors = [line.strip() for line in output.splitlines() if line.startswith("ERROR:")]
        if not errors or not re.search(r"exceeded (?:token )?rate limit", output, re.IGNORECASE) \
                or any(not RATE_LIMIT_LINE.fullmatch(line) for line in errors):
            return False
        failed = True
    return failed
