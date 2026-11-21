import json

import pytest

from scripts import hyperharness_executor as executor


@pytest.mark.parametrize('dataset', ['pm3', 'mempal', 'memorycd'])
def test_executor_swap_preserves_scoring_and_generation_contract(tmp_path, dataset):
    plan = {'executor_model': 'Qwen3.5-9B', 'seed': 20261022, 'input_tokens': 32768,
        'output_tokens': 4096, 'judge_model': 'gpt-5.5', 'optimizer_model': 'gpt-6-sol',
        'phg': {'generator': {'model': 'gpt-6-sol'}}}
    manifest = {'executor_model': 'Qwen3.5-9B', 'executor_base_url': 'http://127.0.0.1:8001/v1',
        'executor_thinking': False, 'executor_sampling_profile': 'qwen35-instruct-lowtemp-v1'}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    runtime = tmp_path / 'source_snapshot/src/ohpu/llm.py'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('def check(self):\n    return self.model != "Qwen3.5-9B"\n')
    if dataset == 'mempal':
        mempal_runtime = tmp_path / 'source_snapshot/src/ohpu/benchmarks/mempal_runtime.py'
        mempal_runtime.parent.mkdir(parents=True)
        mempal_runtime.write_text('RESPONSE_FORMAT = {"properties": {"analysis": {"type": "string", "maxLength": 1200, "pattern": ".*"}, "selected_solutions": {"enum": ["S1", "S2"]}}}\n')
    executor.install(tmp_path, dataset, plan)
    assert plan['executor_model'] == 'Qwen3.8-27B' and plan['executor_url'].endswith(':8002/v1')
    assert plan['seed'] == 20261022 and plan['input_tokens'] == 32768 and plan['output_tokens'] == 4096
    assert plan['judge_model'] == 'gpt-5.5' and plan['optimizer_model'] == plan['phg']['generator']['model'] == 'gpt-6-sol'
    assert plan['executor_experiment']['profile'] == executor.PROFILE
    assert plan['executor_experiment']['original_executor_model'] == 'Qwen3.5-9B'
    if dataset == 'pm3':
        modified = json.loads((tmp_path / 'manifest.json').read_text())
        assert modified == {**manifest, 'executor_model': executor.MODEL, 'executor_base_url': executor.URL}
        assert 'Qwen3.8-27B' in runtime.read_text()
    else:
        assert plan['tokenizer_model'] == executor.TOKENIZER and plan['tokenizer_revision'] == executor.REVISION
        if dataset == 'mempal':
            namespace = {}
            exec(mempal_runtime.read_text(), namespace)
            properties = namespace['RESPONSE_FORMAT']['properties']
            assert properties['analysis']['maxLength'] == 240
            assert properties['analysis']['pattern'] == "^[A-Za-z0-9 .,;:!?()'/-]*$"
            assert properties['selected_solutions'] == {'enum': ['S1', 'S2']}


def test_luna_executor_preserves_schema_seed_and_budget():
    from ohpu.benchmarks.luna_executor import LunaClient
    class Client:
        def chat(self, messages, **options):
            self.options = options
            return {'content': '{}', 'finish_reason': 'stop', 'usage': {'prompt': 20, 'completion': 5}}
    client = Client()
    result = LunaClient(client=client).chat([{'role': 'user', 'content': 'task'}],
        max_tokens=512, seed=17, response_format={'type': 'json_object'})
    assert client.options == {'max_tokens': 512, 'seed': 17,
        'response_format': {'type': 'json_object'}, 'reasoning_effort': 'none'}
    assert result['usage']['prompt_tokens'] == 20 and result['usage']['completion_tokens'] == 5


def test_luna_executor_rejects_usage_over_budget():
    from ohpu.benchmarks.luna_executor import LunaClient
    class Client:
        def chat(self, messages, **options):
            return {'usage': {'prompt': 32769, 'completion': 10}}
    with pytest.raises(ValueError, match='budget'):
        LunaClient(client=Client()).chat([], max_tokens=512, seed=17)


def test_luna_recovery_requires_log_and_is_opt_in(tmp_path):
    from ohpu.benchmarks.luna_executor import LunaClient
    class Client:
        def chat(self, *args, **kwargs):
            raise TimeoutError('request timed out')
    with pytest.raises(ValueError, match='attempt log'):
        LunaClient(client=Client(), recovery_module=tmp_path / 'unused.py')
    with pytest.raises(TimeoutError):
        LunaClient(client=Client()).chat([], max_tokens=512, seed=17)


@pytest.mark.parametrize(('provider', 'model'), [
    ('openai', 'gpt-5.6-luna'), ('azure', 'paper-answer-deployment'),
])
def test_explicit_hosted_executor_uses_registered_model(monkeypatch, provider, model):
    from ohpu.benchmarks import luna_executor

    monkeypatch.setenv('HYPERHARNESS_PROVIDER', provider)
    def legacy_credentials():
        pytest.fail('Explicit provider must not load legacy Azure credentials')
    monkeypatch.setattr(luna_executor, 'load_credentials', legacy_credentials)

    class Client:
        def __init__(self, *, model, **options):
            self.model = model

        def chat(self, messages, **options):
            return {'content': self.model}

    monkeypatch.setattr(luna_executor, 'LLM', Client)
    executor_client = luna_executor.LunaClient(model=model)
    assert executor_client.chat([])['content'] == model
    with pytest.raises(ValueError, match='Unregistered model'):
        executor_client.chat([], model='unexpected-model')


@pytest.mark.parametrize('dataset', ['pm3', 'mempal', 'memorycd'])
@pytest.mark.parametrize('model', [None, 'paper-answer-deployment'])
def test_luna_installation_patches_owned_client_only(tmp_path, dataset, model):
    snapshot = tmp_path / 'source_snapshot/src/ohpu/benchmarks'
    snapshot.mkdir(parents=True)
    plan = {'executor_model': 'Qwen3.5-9B', 'input_tokens': 32768, 'output_tokens': 512,
            'seed': 17, 'judge_model': 'gpt-5.5', 'optimizer_model': 'gpt-6-sol'}
    (tmp_path / 'manifest.json').write_text('{}')
    (tmp_path / 'pm3_native_skillopt.py').write_text('def executor(out):\n    return "old"\n\ndef judge(plan):\n    return "unchanged"\n')
    (tmp_path / (dataset + '_phg.py')).write_text('def score():\n    from transformers import AutoTokenizer\n    return AutoTokenizer\n')
    runtime = snapshot / (dataset + '_runtime.py')
    runtime.write_text('RESPONSE_FORMAT = {"properties": {"analysis": {"maxLength": 1200}, "selected_solutions": {}}}\n')
    executor.install_luna(tmp_path, dataset, plan, model=model)
    assert plan['executor_model'] == (model or 'gpt-5.6-luna')
    assert plan['executor_url'] == ('provider-configured' if model else 'azure-configured')
    assert ('credential_file' in plan['executor_experiment']) == (model is None)
    assert (plan['input_tokens'], plan['output_tokens'], plan['seed']) == (32768, 512, 17)
    assert plan['judge_model'] == 'gpt-5.5' and plan['optimizer_model'] == 'gpt-6-sol'
    assert plan['score_workers'] == 2
    assert (snapshot / 'luna_executor.py').exists()
    if dataset == 'pm3':
        assert 'return "unchanged"' in (tmp_path / 'pm3_native_skillopt.py').read_text()
        assert json.loads((tmp_path / 'manifest.json').read_text())['executor_model'] == plan['executor_model']
        arguments = f'model={model!r}' if model else ''
        assert f'return LunaClient({arguments})' in (tmp_path / 'pm3_native_skillopt.py').read_text()
    else:
        assert 'LunaClient as QwenClient' in runtime.read_text()
        assert 'LunaTokenizer as AutoTokenizer' in (tmp_path / (dataset + '_phg.py')).read_text()
