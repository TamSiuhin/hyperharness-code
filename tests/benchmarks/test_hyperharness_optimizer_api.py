import io
import json
from types import SimpleNamespace

import pytest

from ohpu.per_user_adaptation.controllers import local_cli
from ohpu.per_user_adaptation.memory_evolution import NativeCodexProposer


@pytest.mark.parametrize("provider", [None, "openai", "azure"])
def test_optimizer_cli_auth_and_shell_isolation(tmp_path, monkeypatch, provider):
    monkeypatch.delenv("HYPERHARNESS_PROVIDER", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    if provider:
        monkeypatch.setenv("HYPERHARNESS_PROVIDER", provider)
    if provider == "azure":
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_RESPONSES_API_VERSION", "2025-04-01-preview")

    def fake_popen(command, *, env, stdout, **options):
        assert env.get("OPENAI_API_KEY") == ("test-openai-key" if provider == "openai" else None)
        assert env.get("AZURE_OPENAI_API_KEY") == ("test-azure-key" if provider == "azure" else None)
        assert env["CODEX_HOME"] == str(tmp_path / "codex-home")
        assert 'shell_environment_policy.inherit="none"' in command
        assert command[command.index("--sandbox") + 1] == "workspace-write"
        assert command[command.index("-m") + 1] == "test-optimizer-model"
        if provider == "openai":
            assert 'model_provider="hyperharness_openai"' in command
            assert 'model_providers.hyperharness_openai.base_url="https://api.openai.com/v1"' in command
            assert 'model_providers.hyperharness_openai.env_key="OPENAI_API_KEY"' in command
            assert 'model_providers.hyperharness_openai.wire_api="responses"' in command
        elif provider == "azure":
            assert 'model_provider="azure"' in command
            assert 'model_providers.azure.base_url="https://test-resource.openai.azure.com/openai"' in command
            assert 'model_providers.azure.env_http_headers={"api-key" = "AZURE_OPENAI_API_KEY"}' in command
        else:
            assert not any(arg.startswith("model_provider=") for arg in command)
        stdout.write("Optimizer response")
        return SimpleNamespace(stdin=io.StringIO(), poll=lambda: 0, returncode=0)

    monkeypatch.setattr(local_cli.subprocess, "Popen", fake_popen)
    proposer = NativeCodexProposer(tmp_path / "bundle", model="test-optimizer-model")
    assert proposer.controller._run_controller_command("Optimize") == "Optimizer response"
    record = next((tmp_path / "bundle/controller_artifacts/controller").glob("*.command.json"))
    command = json.dumps(json.loads(record.read_text())["command"])
    assert "test-openai-key" not in command
    assert "test-azure-key" not in command


def test_optimizer_openai_opt_in_requires_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERHARNESS_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        NativeCodexProposer(tmp_path)
