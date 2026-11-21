"""Executor-only registration changes for the existing HyperHarness pipeline."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

PROFILE = 'qwen38-27b'
MODEL = 'Qwen3.8-27B'
TOKENIZER = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
URL = 'http://127.0.0.1:8002/v1'
LUNA_PROFILE = 'gpt56-luna'
LUNA_MODULE = Path(__file__).resolve().parents[1] / 'src/ohpu/benchmarks/luna_executor.py'


def install_luna(out, dataset, plan, *, model=None):
    direct_provider = model is not None
    model = model if direct_provider else 'gpt-5.6-luna'
    original_model = plan.get('executor_model')
    snapshot = out / 'source_snapshot/src/ohpu'
    shutil.copyfile(LUNA_MODULE, snapshot / 'benchmarks/luna_executor.py')
    patches = []
    def patch(path, source):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        compile(source, str(path), 'exec')
        path.write_text(source)
        patches.append({'file': str(path.relative_to(out)), 'before_sha256': before,
                        'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plan.update(executor_model=model,
                executor_url='provider-configured' if direct_provider else 'azure-configured',
                score_workers=2, answer_workers=2, judge_workers=2)
    plan.setdefault('concurrency', {}).update(executor=2, answer=2, judge=2)
    if dataset == 'pm3':
        manifest_path = out / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(executor_model=model, executor_base_url=None,
                        executor_sampling_profile=None, executor_thinking=None,
                        executor_repetition_penalty=None)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        path = out / 'pm3_native_skillopt.py'
        source = path.read_text()
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'executor')
        original = ast.get_source_segment(source, node)
        arguments = f'model={model!r}' if direct_provider else ''
        patch(path, source.replace(original, 'def executor(out):\n'
              '    from ohpu.benchmarks.luna_executor import LunaClient\n'
              f'    return LunaClient({arguments})'))
    else:
        runtime = snapshot / ('benchmarks/' + dataset + '_runtime.py')
        source = runtime.read_text()
        if dataset == 'mempal':
            source = mempal_output_schema(source)
        patch(runtime, source + '\nfrom ohpu.benchmarks.luna_executor import LunaClient as QwenClient\n')
        driver = out / (dataset + '_phg.py')
        source = driver.read_text()
        if 'from transformers import AutoTokenizer' not in source:
            raise ValueError('Missing tokenizer construction anchor')
        patch(driver, source.replace('from transformers import AutoTokenizer',
              'from ohpu.benchmarks.luna_executor import LunaTokenizer as AutoTokenizer'))
        plan.update(tokenizer_model='o200k_base', tokenizer_revision='tiktoken-installed')
    plan['executor_experiment'] = {'profile': LUNA_PROFILE, 'model': model,
        'original_executor_model': original_model, 'reasoning_effort': 'none',
        'input_budget_validation': 'o200k_base chat estimate plus provider-reported usage guard',
        'patches': patches}
    if not direct_provider:
        plan['executor_experiment']['credential_file'] = '/root/.config/personalized_harness/azure.env'
    return plan['executor_experiment']


def mempal_output_schema(source):
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'RESPONSE_FORMAT' for target in node.targets))
    original = ast.get_source_segment(source, assignment)
    properties = next(node for node in ast.walk(assignment) if isinstance(node, ast.Dict)
        and {'analysis', 'selected_solutions'} <= {key.value for key in node.keys if isinstance(key, ast.Constant)})
    analysis = next(value for key, value in zip(properties.keys, properties.values)
                    if isinstance(key, ast.Constant) and key.value == 'analysis')
    assert isinstance(analysis, ast.Dict)
    for index, key in enumerate(analysis.keys):
        if isinstance(key, ast.Constant) and key.value == 'maxLength':
            analysis.values[index] = ast.Constant(240)
        elif isinstance(key, ast.Constant) and key.value == 'pattern':
            analysis.values[index] = ast.Constant("^[A-Za-z0-9 .,;:!?()'/-]*$")
    result = source.replace(original, ast.unparse(ast.fix_missing_locations(assignment)), 1)
    compile(result, 'mempal_output_schema', 'exec')
    return result


def install(out, dataset, plan):
    original_model = plan.get('executor_model')
    original_tokens = {key: plan.get(key) for key in ('input_tokens', 'output_tokens', 'seed')}
    plan['executor_model'] = MODEL
    plan['executor_url'] = URL
    if isinstance(plan.get('executor'), dict):
        plan['executor']['model'] = MODEL
    patches = []
    if dataset == 'pm3':
        manifest_path = out / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(executor_model=MODEL, executor_base_url=URL)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        runtime = out / 'source_snapshot/src/ohpu/llm.py'
        source = runtime.read_text()
        before = hashlib.sha256(runtime.read_bytes()).hexdigest()
        old = 'self.model != "Qwen3.5-9B"'
        if source.count(old) != 1:
            raise ValueError('Unexpected sampling-profile model validation')
        source = source.replace(old, 'self.model != "Qwen3.8-27B"').replace(
            ') require Qwen3.5-9B, thinking disabled', ') require Qwen3.8-27B, thinking disabled')
        compile(source, str(runtime), 'exec')
        runtime.write_text(source)
        patches.append({'file': str(runtime.relative_to(out)), 'before_sha256': before,
            'after_sha256': hashlib.sha256(runtime.read_bytes()).hexdigest()})
    else:
        plan.update(tokenizer_model=TOKENIZER, tokenizer_revision=REVISION,
            model_cache='/opt/model-apis/huggingface/hub')
        if dataset == 'mempal':
            runtime = out / 'source_snapshot/src/ohpu/benchmarks/mempal_runtime.py'
            before = hashlib.sha256(runtime.read_bytes()).hexdigest()
            runtime.write_text(mempal_output_schema(runtime.read_text()))
            patches.append({'file': str(runtime.relative_to(out)), 'before_sha256': before,
                'after_sha256': hashlib.sha256(runtime.read_bytes()).hexdigest()})
            plan['executor_output_policy'] = {'protocol': 'mempal-short-ascii-analysis-v2',
                'analysis_max_chars': 240, 'output_token_budget_unchanged': True,
                'applies_from_initial_registration': True}
    plan['executor_experiment'] = {'profile': PROFILE, 'model': MODEL, 'url': URL,
        'original_executor_model': original_model, 'tokenizer': TOKENIZER, 'tokenizer_revision': REVISION,
        'patches': patches}
    assert original_tokens == {key: plan.get(key) for key in original_tokens}
    return plan['executor_experiment']
