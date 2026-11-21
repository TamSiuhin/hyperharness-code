"""Hosted GPT execution with the existing benchmark budgets and score contracts."""

import os
from pathlib import Path

from ohpu.llm import LLM


MODEL = 'gpt-5.6-luna'


def load_credentials(path=None):
    import shlex
    path = Path(path or os.environ.get('HYPERHARNESS_EXECUTOR_CREDENTIALS',
                                      Path.home() / '.config/personalized_harness/azure.env')).expanduser()
    for line in path.read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == 'export':
            parts = parts[1:]
        if not parts:
            continue
        key, separator, value = parts[0].partition('=')
        if separator and key in ('AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_API_VERSION'):
            os.environ[key] = value


class LunaTokenizer:
    def __init__(self):
        import tiktoken
        self.encoding = tiktoken.get_encoding('o200k_base')

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens, **kwargs):
        return self.encoding.decode(tokens)

    def apply_chat_template(self, messages, **kwargs):
        identifiers = [0] * 3
        for message in messages:
            identifiers.extend([0] * 4)
            identifiers.extend(self.encode(str(message.get('role', ''))))
            identifiers.extend(self.encode(str(message.get('content', ''))))
        return identifiers


class LunaClient:
    MODEL = MODEL

    @staticmethod
    def create_client(model):
        if not os.environ.get('HYPERHARNESS_PROVIDER'):
            load_credentials()
        return LLM(model=model, reasoning_effort='none', cache=None,
                   request_timeout_seconds=180, max_retries=1)

    def __init__(self, *, base_url=None, model=MODEL, client=None, recovery_module=None,
                 recovery_log=None, recovery_extra_attempts=2):
        if not os.environ.get('HYPERHARNESS_PROVIDER') and model != self.MODEL:
            raise ValueError('Luna executor model is fixed by registration')
        self.model = model
        self.reasoning_effort = 'none'
        if client is None:
            client = self.create_client(model)
        self.client = client
        self._chat = client.chat
        if recovery_module is not None:
            if recovery_log is None:
                raise ValueError('Executor recovery requires an append-only attempt log')
            import importlib.util
            specification = importlib.util.spec_from_file_location('luna_executor_recovery', recovery_module)
            recovery = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(recovery)
            self._chat = recovery.recover_call(client.chat, recovery_log, extra_attempts=recovery_extra_attempts)

    def chat(self, messages, *, max_tokens=1024, seed=7, response_format=None, **options):
        if options.get('model', self.model) != self.model or options.get('tools'):
            raise ValueError('Unregistered model or extra executor tools')
        response = self._chat(messages, max_tokens=max_tokens, seed=seed,
                                    response_format=response_format, reasoning_effort='none')
        usage = response.get('usage') or {}
        usage = {**usage, 'prompt_tokens': usage.get('prompt_tokens', usage.get('prompt', 0)),
                 'completion_tokens': usage.get('completion_tokens', usage.get('completion', 0))}
        if usage['prompt_tokens'] > 32768 or usage['completion_tokens'] > max_tokens:
            raise ValueError('Provider-reported usage exceeds registered executor budget')
        return {**response, 'usage': usage}
