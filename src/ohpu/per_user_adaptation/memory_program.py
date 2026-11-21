"""Native Python memory programs compatible with the Mstar module interface.

Import validation is a compatibility check, not a security sandbox. Generated
modules must run in a data-scoped worker, never in the optimizer's process.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Callable, Iterator


ALLOWED_IMPORTS = frozenset({
    "json", "re", "math", "hashlib", "collections", "dataclasses", "typing",
    "datetime", "textwrap", "sqlite3",
})
INSTRUCTION_NAMES = (
    "INSTRUCTION_KNOWLEDGE_ITEM", "INSTRUCTION_QUERY", "INSTRUCTION_RESPONSE",
    "ALWAYS_ON_KNOWLEDGE",
)


class MemoryProgramError(ValueError):
    pass


class MemoryInfrastructureError(RuntimeError):
    pass


def _native_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".", 1)[0] not in ALLOWED_IMPORTS:
        raise ImportError("memory program import is not allowed: " + name)
    return builtins.__import__(name, globals, locals, fromlist, level)


@dataclass(frozen=True)
class CompiledMemoryProgram:
    knowledge_item: type
    query: type
    knowledge_base: type
    instructions: dict[str, str]
    parse_knowledge_item: Callable | None = None
    parse_query: Callable | None = None


@dataclass(frozen=True)
class MemoryProgram:
    source: str

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise MemoryProgramError("memory program source must be nonempty text")
        if self.source.startswith("STRATEGY_PROTOCOL = "):
            from .memory_strategy import validate_strategy

            validate_strategy(self.source)
            return
        try:
            tree = ast.parse(self.source, filename="harness.py")
        except SyntaxError as error:
            raise MemoryProgramError(f"invalid memory program syntax: {error.msg}") from error
        for node in ast.walk(tree):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name.split(".", 1)[0] not in ALLOWED_IMPORTS for name in names) or (
                    isinstance(node, ast.ImportFrom) and node.level):
                raise MemoryProgramError("memory program import is not allowed")

    @property
    def sha256(self):
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @contextmanager
    def compile(self) -> Iterator[CompiledMemoryProgram]:
        if self.source.startswith("STRATEGY_PROTOCOL = "):
            from .memory_strategy import compile_strategy

            yield compile_strategy(self.source)
            return
        module = types.ModuleType("_ohpu_memory_" + uuid.uuid4().hex)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": _native_import}
        sys.modules[module.__name__] = module
        try:
            exec(compile(self.source, "harness.py", "exec"), module.__dict__)
            if not all(isinstance(getattr(module, name, None), type) for name in ("KnowledgeItem", "Query", "KnowledgeBase")):
                raise MemoryProgramError("memory program requires KnowledgeItem, Query and KnowledgeBase")
            instructions = {name: getattr(module, name, None) for name in INSTRUCTION_NAMES}
            if any(not isinstance(value, str) for value in instructions.values()):
                raise MemoryProgramError("memory program requires four string instruction constants")
            knowledge_item, query = module.KnowledgeItem, module.Query
            if not is_dataclass(knowledge_item) or not is_dataclass(query):
                raise MemoryProgramError("KnowledgeItem and Query must be dataclasses")
            knowledge_base = module.KnowledgeBase
            if not all(callable(getattr(knowledge_base, name, None)) for name in ("write", "read")):
                raise MemoryProgramError("KnowledgeBase requires write and read methods")
            parser = getattr(module, "parse_knowledge_item", None)
            if parser is not None and not callable(parser):
                raise MemoryProgramError("parse_knowledge_item must be callable when defined")
            query_parser = getattr(module, "parse_query", None)
            if query_parser is not None and not callable(query_parser):
                raise MemoryProgramError("parse_query must be callable when defined")
            yield CompiledMemoryProgram(knowledge_item, query, knowledge_base, instructions, parser, query_parser)
        finally:
            sys.modules.pop(module.__name__, None)


class NativeMemorySession:
    """No-deadline worker session; only explicitly supplied data crosses RPC."""

    def __init__(self, program, *, complete, embed, record, llm_call_budget=1,
                 memory_limit_mb=2048, max_read_chars=3000, write_llm_call_budget=0, worker_factory=None):
        if sys.version_info < (3, 12):
            raise MemoryInfrastructureError("mstar-native requires Python 3.12 or later")
        if write_llm_call_budget is not None and (type(write_llm_call_budget) is not int or write_llm_call_budget < 0):
            raise ValueError("write_llm_call_budget must be a nonnegative integer or None")
        self.complete, self.embed, self.record = complete, embed, record
        self.llm_call_budget = llm_call_budget
        self.write_llm_call_budget = llm_call_budget if write_llm_call_budget is None else write_llm_call_budget
        self.operation = "init"
        self.model_calls = 0
        self.closed = False
        self.worker_factory = worker_factory
        self.factory_acquired = False
        self.directory = tempfile.TemporaryDirectory(prefix="ohpu-native-")
        self.stderr = ((Path(self.directory.name) / "worker.stderr").open("w+t", encoding="utf-8")
            if worker_factory is not None else tempfile.TemporaryFile(mode="w+t", encoding="utf-8"))
        self.stderr.seek(0, os.SEEK_END)
        self.stderr_start = self.stderr.tell()
        environment = {"PATH": os.defpath, "HOME": self.directory.name, "TMPDIR": self.directory.name,
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "ANONYMIZED_TELEMETRY": "False"}
        root = str(Path(__file__).resolve().parents[2])
        bootstrap = "import sys;sys.path.insert(0," + repr(root) + ");from ohpu.per_user_adaptation.memory_worker import main;main()"
        self.process = worker_factory.process if worker_factory is not None else subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", bootstrap], cwd=self.directory.name, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding="utf-8",
            close_fds=True, start_new_session=True)
        try:
            initialization = {"source": program.source, "llm_call_budget": llm_call_budget,
                "memory_limit_mb": memory_limit_mb, "max_read_chars": max_read_chars,
                "write_llm_call_budget": self.write_llm_call_budget}
            if worker_factory is None:
                self._send(initialization)
            else:
                worker_factory.acquire({**initialization, "directory": self.directory.name})
                self.factory_acquired = True
            response = self._receive()
            if not response.get("ready"):
                raise MemoryProgramError("native worker did not initialize")
            self.schemas, self.instructions = response["schemas"], response["instructions"]
            self.has_local_query_parser = response.get("local_query_parser", False)
        except BaseException:
            self.close()
            raise

    def _send(self, value):
        self.process.stdin.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def _receive(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                self.process.wait()
                raise MemoryInfrastructureError(f"native worker exited unexpectedly ({self.process.returncode}); no candidate score may be assigned")
            response = json.loads(line)
            if "worker_finished" in response and self.worker_factory is not None:
                self.worker_factory.finished(response)
                raise MemoryInfrastructureError(f"native memory child exited unexpectedly ({response['exitcode']})")
            if "event" in response:
                self.record(response["event"], **{key: value for key, value in response.items() if key != "event"})
                continue
            if "error" in response:
                self.record("memory_error", **response)
                if response.get("infrastructure"):
                    raise MemoryInfrastructureError(response["error"])
                raise MemoryProgramError(response["error"])
            if "request" not in response:
                return response
            operation = response["request"]
            self.record("memory_host_request", **response)
            try:
                if operation == "complete":
                    budget = self.llm_call_budget if self.operation == "read" else self.write_llm_call_budget
                    if budget == 0 and self.operation != "read":
                        raise MemoryProgramError("LLM calls are forbidden during memory initialization, parsing and writes")
                    if self.model_calls >= budget:
                        raise MemoryProgramError("memory toolkit model-call budget exhausted")
                    self.model_calls += 1
                    options = response.get("options", {})
                    if not isinstance(options, dict) or set(options) - {"temperature", "max_tokens", "max_completion_tokens", "response_format", "reasoning_effort", "top_p"}:
                        raise MemoryProgramError("unsupported toolkit model option")
                    value = self.complete(response["messages"], **options)
                elif operation == "embed":
                    texts = response["texts"]
                    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                        raise MemoryProgramError("embedding input must be a list of text")
                    value = self.embed(texts)
                    value = value.tolist() if hasattr(value, "tolist") else [row.tolist() if hasattr(row, "tolist") else list(row) for row in value]
                else:
                    raise MemoryProgramError("unknown host request")
            except BaseException:
                if self.worker_factory is not None:
                    self.worker_factory.release(abort=True)
                else:
                    if self.process.poll() is None:
                        self.process.kill()
                    self.process.wait()
                raise
            self.record("memory_host_response", request_id=response["request_id"], value=value)
            self._send({"request_id": response["request_id"], "value": value})

    def command(self, operation, **values):
        self.operation = operation
        self.model_calls = 0
        self.record("memory_operation", operation=operation, **values)
        self._send({"operation": operation, **values})
        result = self._receive()["result"]
        self.record("memory_result", operation=operation, value=result)
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.worker_factory is not None:
            if self.factory_acquired:
                self.worker_factory.release(abort=True)
        else:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            for channel in (self.process.stdin, self.process.stdout):
                if channel:
                    channel.close()
        self.stderr.seek(self.stderr_start)
        chunk_index = 0
        while output := self.stderr.read(64 * 1024):
            self.record("memory_worker_stderr", chunk_index=chunk_index, output=output)
            chunk_index += 1
        self.stderr.close()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
