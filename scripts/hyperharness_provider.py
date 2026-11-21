"""Configure direct model APIs for a frozen HyperHarness experiment."""

import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from scripts import hyperharness_runtime as shared


PROVIDERS = ("openai", "azure")
OPENAI_URL = "https://api.openai.com/v1"
AZURE_KEYS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION")
OPTIMIZER_PHASE = Path("src/ohpu/per_user_adaptation/phase_a.py")
REASONING_MODEL = 'm.startswith(("gpt-5", "gpt-6", "o3", "o4"))'


def validate_environment(configuration, *, local_pm3=False):
    provider = configuration["provider"]
    keys = ("OPENAI_API_KEY",) if provider == "openai" else AZURE_KEYS
    if local_pm3 and configuration["answer_model"] is None:
        keys += ("OHPU_EXECUTOR_API_KEY",)
    missing = [key for key in keys if not os.environ.get(key, "").strip()]
    if missing:
        raise ValueError(f"Missing {provider} run settings: {', '.join(missing)}")
    if provider == "azure":
        endpoint = urlsplit(os.environ["AZURE_OPENAI_ENDPOINT"])
        if (endpoint.scheme != "https" or not re.fullmatch(r"[a-z0-9-]+\.openai\.azure\.com", endpoint.netloc)
                or endpoint.path not in ("", "/") or endpoint.query or endpoint.fragment):
            raise ValueError("AZURE_OPENAI_ENDPOINT must be an HTTPS Azure OpenAI resource root")


def configure_environment(configuration, *, local_pm3=False):
    validate_environment(configuration, local_pm3=local_pm3)
    provider = configuration["provider"]
    if provider == "openai":
        for key in (*AZURE_KEYS, "AZURE_OPENAI_RESPONSES_API_VERSION"):
            os.environ.pop(key, None)
        os.environ["OPENAI_BASE_URL"] = OPENAI_URL
    elif not os.environ.get("AZURE_OPENAI_RESPONSES_API_VERSION"):
        os.environ["AZURE_OPENAI_RESPONSES_API_VERSION"] = "2025-04-01-preview"
    os.environ["HYPERHARNESS_PROVIDER"] = provider


def install(out, configuration, phase_source):
    """Patch the copied runtime without replacing unrelated frozen code."""
    snapshot = Path("source_snapshot/src/ohpu")
    patches = []

    def patch(relative, replacements):
        path = out / relative
        before = shared.sha(path)
        source = path.read_text()
        for old, new in replacements:
            if source.count(old) != 1:
                raise ValueError(f"Missing or ambiguous model API patch in {relative}")
            source = source.replace(old, new, 1)
        compile(source, str(path), "exec")
        path.write_text(source)
        patches.append({"file": str(relative), "before_sha256": before, "after_sha256": shared.sha(path)})

    for relative, name, arguments in (
        ("pm3_generator_harness.py", "load_generator_credentials", "plan"),
        ("pm3_mechanism_experience.py", "load_optimizer_credentials", "plan"),
        ("base_runner.py", "load_credentials", ""),
    ):
        old = shared.function_source((out / relative).read_text(), name)
        new = (f"def {name}({arguments}):\n"
               f'    if os.environ.get("HYPERHARNESS_PROVIDER") != {configuration["provider"]!r}:\n'
               '        raise RuntimeError("Launch the registered model API through run_hyperharness_reference.py")')
        patch(relative, [(old, new)])

    if configuration["provider"] == "azure":
        patch(snapshot / "llm.py", [
            ("five = " + REASONING_MODEL,
             "five = " + REASONING_MODEL + "\n"
             "        if not five:\n"
             "            from openai import AzureOpenAI\n"
             "            five = isinstance(self._client, AzureOpenAI)"),
        ])

    if configuration["provider"] == "openai":
        phase = snapshot / OPTIMIZER_PHASE.relative_to("src/ohpu")
        old_phase = (out / phase).read_text()
        if "def _codex_api_environment(" not in old_phase:
            old_args = shared.function_source(old_phase, "_safe_codex_args")
            new_args = shared.function_source(phase_source, "_safe_codex_args")
            auth = shared.function_source(phase_source, "_codex_api_environment")
            patch(phase, [(old_args, new_args + "\n\n\n" + auth)])
            patch(snapshot / "per_user_adaptation/memory_evolution.py", [
                ("from .phase_a import _safe_codex_args",
                 "from .phase_a import _codex_api_environment, _safe_codex_args"),
                ('        environment = {"AZURE_OPENAI_API_KEY": os.environ["AZURE_OPENAI_API_KEY"]} '
                 'if os.environ.get("AZURE_OPENAI_ENDPOINT") else None\n', ""),
                ("model=model, env=environment,", "model=model, env=_codex_api_environment(),"),
            ])
            patch(snapshot / "per_user_adaptation/controllers/local_cli.py", [
                ('\n                merged_env.pop("OPENAI_API_KEY", None)',
                 '\n                if not (self.env and self.env.get("HYPERHARNESS_PROVIDER") == "openai"\n'
                 '                        and self.env.get("OPENAI_API_KEY")):\n'
                 '                    merged_env.pop("OPENAI_API_KEY", None)'),
            ])
    return patches
