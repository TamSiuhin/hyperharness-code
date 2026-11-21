"""Credential, result, and executor helpers for frozen HyperHarness hosts."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

CREDENTIALS = os.environ.get("HYPERHARNESS_EXECUTOR_CREDENTIALS", str(Path.home() / ".config/personalized_harness/azure.env"))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from ohpu.experiments.lamp import write_once

    write_once(Path(path), value)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_credentials():
    from ohpu.llm import load_env

    load_env(CREDENTIALS)
    missing = [name for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OHPU_EXECUTOR_API_KEY") if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing credentials in " + CREDENTIALS + ": " + ", ".join(missing))


def executor_client(out, manifest):
    import socket
    from openai import DefaultHttpxClient, OpenAI, _base_client
    from ohpu.llm import LLM, Cache

    http = getattr(_base_client, "httpx2", None) or _base_client.httpx
    keepalive = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)] + [(socket.IPPROTO_TCP, getattr(socket, name), value)
        for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4)) if hasattr(socket, name)]
    timeout = manifest.get("model_request_timeout_seconds", 180.0)
    sdk = OpenAI(base_url=manifest["executor_base_url"], api_key=os.environ["OHPU_EXECUTOR_API_KEY"], timeout=timeout, max_retries=0,
        http_client=DefaultHttpxClient(transport=http.HTTPTransport(retries=0,
            socket_options=keepalive if manifest.get("model_connection_keepalive") else []), timeout=timeout))
    return LLM(model=manifest["executor_model"], reasoning_effort="low", cache=Cache(str(Path(out) / "generation.sqlite")),
        _client=sdk, enable_thinking=manifest.get("executor_thinking"), repetition_penalty=manifest.get("executor_repetition_penalty"),
        sampling_profile=manifest.get("executor_sampling_profile"), request_timeout_seconds=timeout,
        connection_keepalive=manifest.get("model_connection_keepalive", False))


def timed_out(workspace):
    commands = sorted((Path(workspace) / "controller_artifacts/controller").glob("*.command.json"))
    command = read(commands[-1]) if commands else {}
    return bool(command.get("timed_out")) and command.get("returncode") is not None
