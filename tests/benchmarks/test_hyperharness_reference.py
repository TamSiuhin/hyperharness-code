import json
import sys
from pathlib import Path

import pytest

from scripts import run_hyperharness_reference as pipeline

REPO = Path(__file__).resolve().parents[2]


def test_transport_reopens_after_timeout_without_changing_request(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ohpu import llm
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as host
    clients, calls, closed = [], [], []

    class Client:
        def __init__(self, **options):
            assert options['connection_keepalive'] is True
            assert options['max_retries'] == 1
            assert options['reasoning_effort'] == 'medium'
            assert options['request_timeout_seconds'] == 600
            clients.append(self)
            self._client = SimpleNamespace(close=lambda: closed.append(self))

        def chat(self, messages, **options):
            calls.append((messages, options))
            if self is clients[0]:
                raise TimeoutError('stale connection')
            return {'content': 'program', 'usage': {'completion': 5}}

    monkeypatch.setattr(llm, 'LLM', Client)
    log = tmp_path / 'attempts.jsonl'
    call = host.luna(retries=1, request_timeout_seconds=600, retry_log=log, backoff_seconds=0,
                     connection_keepalive=True, fresh_client_on_retry=True)
    result = call([{'role': 'user', 'content': 'same request'}], max_tokens=16000, effort='medium')
    assert result['transport_attempts'] == 2
    assert len(clients) == 2 and closed == [clients[0]]
    assert calls[0] == calls[1]
    with log.open() as stream:
        attempts = [json.loads(line) for line in stream]
    assert attempts[0]['usage_unknown'] and attempts[0]['connection_closed']
    assert attempts[1]['outcome'] == 'completed'


def test_fresh_transport_does_not_add_unregistered_attempts(monkeypatch):
    from types import SimpleNamespace
    from ohpu import llm
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as host
    clients = []

    def fail(*args, **kwargs):
        raise TimeoutError('provider unavailable')

    def factory(**options):
        clients.append(options)
        return SimpleNamespace(chat=fail, _client=SimpleNamespace(close=lambda: None))

    monkeypatch.setattr(llm, 'LLM', factory)
    call = host.luna(retries=1, backoff_seconds=0, connection_keepalive=True, fresh_client_on_retry=True)
    with pytest.raises(TimeoutError):
        call([], max_tokens=16000, effort='medium')
    assert len(clients) == 2


def test_recall_reference_cap_applies_across_multiple_search_batches():
    source = '''def recall(context):
    state = context['state']
    for row in context['rows']:
        state['texts'].append(row['text'])
        state['refs'].append(row['ref'])
    return state
'''
    namespace = {}
    exec(pipeline.bound_recall_references(source), namespace)
    state = {'texts': [], 'refs': []}
    for batch in range(3):
        rows = [{'text': str(index), 'ref': str(index)} for index in range(batch * 24, (batch + 1) * 24)]
        state = namespace['recall']({'state': state, 'rows': rows})
    assert state['refs'] == [str(index) for index in range(64)]
    assert state['texts'] == state['refs']


def test_task_history_preserves_rating_conditioned_evidence_and_sources():
    from analysis.skill_optimization_diagnosis.phg_spec_v0.memorycd_history_adapter import digest
    events = [{'id': str(index), 'ts': index, 'app': 'shop', 'event': {
        'domain': 'tools', 'rating': 1 if index < 3 else 5,
        'product': {'title': 'portable lamp', 'category': 'lighting'},
        'review_title': 'Too dim' if index < 3 else 'Useful light',
        'review': 'dim' if index < 3 else 'bright and portable'}} for index in range(6)]
    result = json.loads(digest(events))
    stats = result['history_statistics']
    assert stats['rating_counts'] == {'1.0': 3, '5.0': 3}
    assert stats['category_ratings'][0]['mean_rating'] == 3
    assert [row['review_words_median'] for row in stats['review_style_by_rating']] == [1, 3]
    assert {row['rating'] for row in result['rated_product_examples']} == {1, 5}
    assert {row['id'] for row in result['events']} == {str(index) for index in range(6)}
    assert digest(events) == digest(events)


def test_simple_history_has_only_bounded_raw_samples():
    import copy
    from analysis.skill_optimization_diagnosis.phg_spec_v0.memorycd_simple_history_adapter import digest
    events = [{'id': str(index), 'ts': index, 'event': {'rating': index % 5 + 1,
        'product': {'title': 'item ' + str(index), 'category': 'tools'},
        'review_title': 'title', 'review': 'text'}} for index in range(300)]
    original = copy.deepcopy(events)
    result = json.loads(digest(events))
    assert set(result) == {'total_history_events', 'selected_events', 'selection', 'events'}
    assert result['selected_events'] == 96
    ids = [row['id'] for row in result['events']]
    assert ids[-18:] == [str(index) for index in range(282, 300)]
    assert ids[0] == '0' and len(set(ids)) == len(ids)
    assert events == original and digest(events) == digest(events)
    assert json.loads(digest([]))['events'] == []
    events[0]['event']['review'] = 'x' * 100000
    assert len(json.dumps(json.loads(digest(events))['events'], ensure_ascii=False)) <= 34000
    for event in events:
        event['event']['review'] = 'x' * 100000
    bounded = json.loads(digest(events))['events']
    assert len(json.dumps(bounded, ensure_ascii=False)) <= 34000
    assert [event['id'] for event in bounded[-18:]] == [str(index) for index in range(282, 300)]


def test_simple_init_changes_only_prompt_and_history_adapter(tmp_path):
    source = reference(tmp_path, 'memorycd')
    strong, simple = tmp_path / 'strong', tmp_path / 'simple'
    pipeline.register_train(strong, 'memorycd', source, variant='task-aware')
    pipeline.register_train(simple, 'memorycd', source, variant='simple-init')
    left = strong / 'memorycd/train/generator/v0'
    right = simple / 'memorycd/train/generator/v0'
    changed = {path.name for path in left.iterdir() if path.read_bytes() != (right / path.name).read_bytes()}
    assert changed == {'prompt.md', 'history_adapter.py'}
    prompt = (right / 'prompt.md').read_text()
    assert 'Task-specific evidence use' not in prompt
    assert 'exactly one query' in prompt and 'outer optimizer may' in prompt
    assert (right / 'history_adapter.py').read_bytes() == pipeline.SIMPLE_HISTORY_ADAPTER.read_bytes()
    before = pipeline.read(strong / 'memorycd/train/plan.json')
    after = pipeline.read(simple / 'memorycd/train/plan.json')
    assert before['phg']['limits'] == after['phg']['limits']
    assert before['phg']['generator'] == after['phg']['generator']
    assert before['test_users'] == after['test_users']
    with pytest.raises(ValueError, match='MemoryCD only'):
        pipeline.register_train(tmp_path / 'invalid', 'pm3', variant='simple-init')


def test_task_history_empty_and_oversized_record_are_bounded():
    from analysis.skill_optimization_diagnosis.phg_spec_v0.memorycd_history_adapter import digest
    assert json.loads(digest([]))['history_statistics']['rating_mean'] is None
    event = {'id': 'source', 'event': {'rating': 5, 'review': 'prefix' + 'x' * 100000 + 'suffix',
        'review_title': 'x' * 100000, 'product': {'title': 'x' * 100000, 'category': 'x' * 100000}}}
    result = json.loads(digest([event]))
    assert len(json.dumps(result['events'])) < 34000
    assert result['events'][0]['review_excerpt'].endswith('suffix')
    assert result['events'][0]['id'] == 'source'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reference(tmp_path, dataset):
    root = tmp_path / 'reference'
    train, test = root / 'train', root / 'test'
    native = root / 'native'
    plan = {'dataset': dataset, 'split': 'TRAIN', 'test_users': ['train_user'],
        'heldout_users': ['test_user'], 'arms': ['no_history', 'phg_r0', 'phg_r1'],
        'active_arms': ['no_history', 'phg_r0', 'phg_r1'],
        'external_baselines': {'arms': ['no_history'], 'files': {}},
        'comparisons': {'comparison': ['phg_r0', 'no_history']},
        'study_design': {'baseline_groups': {'reference': ['no_history']}},
        'executor_model': 'Qwen3.5-9B', 'prepared': str(root / 'prepared'),
        'concurrency': {'optimizer': 1}, 'seed': 42, 'output_tokens': 512,
        'queue': {'obsolete_dependency': True}, 'host_limit_scale': 1, 'input_tokens': 32768,
        'phg': {'arms': {'phg_r0': 0, 'phg_r1': 1}, 'rounds': 11, 'replicates': 1,
            'limits': {'llm_calls': 6, 'checks': 6, 'completion_tokens': 120000},
            'generator': {'model': 'gpt-6-sol', 'request_timeout_seconds': 120},
            'optimizer': {'reasoning_effort': 'medium'}, 'selection_policy': 'source_specific'},
        'workflow_experiment': {'native_test_reference': {'root': str(native), 'files': {}}}}
    write(train / 'plan.json', plan)
    if dataset == 'memorycd':
        plan['task_guard'] = {'numerical_tolerance': 1e-9}
        write(train / 'plan.json', plan)
    driver = train / 'pm3_generator_harness.py'
    driver.write_text('''GENERATOR = {'model': 'gpt-6-sol', 'efforts': ['none', 'low'], 'default_effort': 'low'}

def configured_luna(plan, retry_log):
    return luna(plan['phg']['generator']['model'])

def run_generator(spec_dir, events, **options):
    return {}

def framework(plan, native):
    return "Historical task and CI gates"

def round_summary(out, version):
    plan = read(out / "plan.json")
    parent = read(out / "rounds/parent.json")
    scores = arm_scores(out, plan, round_arm(plan, version))
    decision = {"adopted": True}
    summary = {"parent_decision": decision}
    if decision["adopted"]:
        parent["version"] = version
    replace(out / "rounds/parent.json", parent)
    return summary

def diagnose(out, version, *, client=None):
    DIAGNOSED.append(("host", version))
''')
    if dataset != 'pm3':
        (train / (dataset + '_phg.py')).write_text("BASELINES = ('no_history',)\n"
            'def round_summary(out, version):\n    raise ValueError("old gate")\n'
            '\ndef framework(plan, native):\n    return "dataset framework"\n'
            '\ndef diagnose(out, version):\n    DIAGNOSED.append(("dataset", version))\n'
            '\ndef tokenizer():\n    from transformers import AutoTokenizer\n    return AutoTokenizer\n')
    test_arm = 'phg_test_r0' if dataset == 'pm3' else 'test_initial'
    test_plan = {**plan, 'split': 'TEST', 'test_users': ['test_user'],
        'arms': ['no_history', test_arm], 'splits': {'test_user': {'evaluated': ['q1']}}}
    write(test / 'plan.json', test_plan)
    if dataset == 'pm3':
        write(native / 'plan.json', test_plan)
        write(native / 'private/users/test_user.json', {'rows': [{'query_id': 'q1'}, {'query_id': 'q2'}]})
        write(native / 'private/requests/test_user.json', {'q1': 'TEST request', 'q2': 'excluded request'})
    for out, users, initial in [(train, ['train_user'], 'phg_r0'), (test, ['test_user'], test_arm)]:
        for user in users:
            for arm in ['no_history', initial]:
                write(out / 'scores' / arm / user / 'summary.json',
                      {'evaluation_complete': True, 'runtime_violations': [], 'accuracy_pct_micro': 40})
            write(out / 'harness' / initial / user / 'harness.json', {'generation': {'valid': True}})
        write(out / 'registration.json', {'files': {}})
    for name in ['source_snapshot', 'inputs', 'private', 'generator/v0', 'generator/v1', 'diagnostics/round0']:
        write(train / name / 'fixture.json', {'original': name})
    (train / 'generator/v0/generator.py').write_text("EFFORT = 'low'\n")
    (train / 'generator/v0/program_api.md').write_text('Dataset runtime fixture: ' + dataset)
    runtime = train / 'source_snapshot/src/ohpu/per_user_adaptation/program.py'
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text('DEFAULT_LINE_BUDGET = 20_000\n')
    (runtime.parent.parent / 'llm.py').write_text('def check(self):\n    return self.model != "Qwen3.5-9B"\n')
    (train / 'pm3_native_skillopt.py').write_text('def executor(out):\n    return "original"\n')
    write(train / 'manifest.json', {'executor_model': 'Qwen3.5-9B', 'executor_base_url': 'http://127.0.0.1:8001/v1'})
    benchmarks = runtime.parent.parent / 'benchmarks'
    benchmarks.mkdir(parents=True, exist_ok=True)
    for name in ('horizon_runtime', 'muse_runtime', 'memorycd_runtime', 'mempal_runtime'):
        (benchmarks / (name + '.py')).write_bytes((REPO / 'src/ohpu/benchmarks' / (name + '.py')).read_bytes())
    (train / 'generator/v0/backbone.py').write_text('''PROGRAM_API_VERSION = 1
BACKBONE = {'retrieval': {'top_k': 10}}
#@USER_SECTION@#
def remember(context):
    return {}
def guard(context):
    return {}
def apply(context):
    return {}
def route_for(request, kind):
    settings = BACKBONE['retrieval']
    route = user_route(request, kind, None)
    return {'queries': [request], 'history': True}
def search(text, state):
    return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': text, 'limit': BACKBONE['retrieval']['top_k'], 'state': state}}
def recall(context):
    route = route_for(context['query'], context['task_kind'])
    state = {}
    return search(context['query'], state)
''')
    (train / 'generator/v0/control_section.py').write_text('USER_MODEL = {}\ndef user_route(request, kind, extra):\n    return {}\n')
    write(train / 'generator/v0.json', {'version': 0, 'spec_sha256': 'original'})
    decision = {'version': 0, 'adopted': True}
    initial = {'parent_decision': decision}
    write(train / 'rounds/round0.json', initial)
    write(train / 'optimizer/v1/workspace/diagnostics/summary.json',
          {'rounds': [initial], 'parent': {'version': 0, 'history': [decision]}})
    write(train / 'selection.json', {'version': 1})
    return root


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
def test_reference_register_preserves_policy_and_excludes_later_feedback(tmp_path, dataset):
    source = reference(tmp_path, dataset)
    before = (source / 'train/plan.json').read_bytes()
    root = tmp_path / 'unified'
    pipeline.register_train(root, dataset, source)
    out = root / dataset / 'train'
    plan = pipeline.read(out / 'plan.json')
    original = json.loads(before)
    assert plan['phg'] == {**original['phg'], 'start_round': 0, 'selection_policy': 'mean_gain',
        'diagnostic_reference': 'phg_r0',
        'limits': {'llm_calls': 10, 'checks': 10, 'completion_tokens': 200000},
        'generator': {**original['phg']['generator'], 'model': 'gpt-6-sol', 'reasoning_effort': 'medium',
            'efforts': ['none', 'low', 'medium'], 'default_effort': 'medium',
            'effort_note': 'Registered construction calls are host-fixed to gpt-6-sol medium.'},
        'optimizer': {**original['phg']['optimizer'], 'model': 'gpt-6-sol', 'reasoning_effort': 'medium'},
        'spec_v0_sha256': pipeline.spec_sha(out / 'generator/v0'), 'seed_reuse': None}
    assert plan['selection_rule'] == pipeline.SELECTION_RULE
    assert plan['optimizer_model'] == 'gpt-6-sol'
    assert plan['reference_pipeline']['selection_patches']
    if dataset == 'memorycd':
        assert plan['task_guard'] is None
    assert plan['host_limit_scale'] == 1 and plan['input_tokens'] == 32768
    assert 'queue' not in plan
    assert not (out / 'generator/v1').exists()
    assert not (out / 'optimizer').exists()
    assert not (out / 'selection.json').exists()
    assert not (root / dataset / 'test').exists()
    assert (out / 'generator/v0/fixture.json').read_bytes() == (source / 'train/generator/v0/fixture.json').read_bytes()
    assert not (out / 'scores/phg_r0').exists()
    assert not (out / 'harness/phg_r0').exists()
    assert not (out / 'rounds').exists()
    assert not (out / 'diagnostics').exists()
    assert plan['arms'] == plan['active_arms'] == list(plan['phg']['arms'])
    assert plan['external_baselines'] == {'arms': [], 'files': {}}
    assert plan['comparisons'] == {} and 'study_design' not in plan
    assert not (out / 'scores/no_history').exists()
    if dataset != 'pm3':
        driver = load_module(out / f'{dataset}_phg.py', 'registered_' + dataset)
        assert driver.BASELINES == ()
    assert "EFFORT = 'medium'" in (out / 'generator/v0/generator.py').read_text()
    assert (out / 'generator/v0/generator.py').read_text() == pipeline.generation_source()
    assert (out / 'generator/v0/prompt.md').read_text() == pipeline.GENERATION_PROMPT + '\n' + pipeline.TASK_APPENDICES[dataset]
    assert plan['program_edit_scope'] == 'personal-section-v1'
    assert (out / 'harness_permission.py').exists()
    assert pipeline.read(out / 'generator/v0.json')['spec_sha256'] == pipeline.spec_sha(out / 'generator/v0')
    assert (source / 'train/plan.json').read_bytes() == before
    pipeline.shared.verify(out)
    with pytest.raises(ValueError, match='overwrite'):
        pipeline.register_train(root, dataset, source)


def test_registered_memorycd_installs_transport_without_keepalive(tmp_path):
    import inspect
    source = reference(tmp_path, 'memorycd')
    driver = source / 'train/pm3_generator_harness.py'
    driver.write_text(driver.read_text() + '\ndef luna(model, retries=0):\n    return None\n')
    root = tmp_path / 'unified'
    pipeline.register_train(root, 'memorycd', source)
    out = root / 'memorycd/train'
    namespace = {}
    exec(compile((out / 'pm3_generator_harness.py').read_text(), 'registered-host', 'exec'), namespace)
    assert 'fresh_client_on_retry' in inspect.signature(namespace['luna']).parameters
    seen = []

    def factory(model, **options):
        seen.append(options)
        return lambda *args, **kwargs: {'content': 'ok'}

    namespace['luna'] = factory
    namespace['configured_luna'](pipeline.read(out / 'plan.json'), None)
    assert 'connection_keepalive' not in seen[0]
    assert 'fresh_client_on_retry' not in seen[0]
    assert seen[0]['retries'] == 0


def test_missing_failover_hosts_leaves_study_root_available(tmp_path, monkeypatch):
    monkeypatch.delenv('HYPERHARNESS_AZURE_FAILOVER_HOSTS', raising=False)
    root = tmp_path / 'study'
    credentials = tmp_path / 'outside-registration.env'
    with pytest.raises(ValueError, match='HYPERHARNESS_AZURE_FAILOVER_HOSTS'):
        pipeline.prepare(root, datasets=['pm3'], failover_credentials=credentials)
    assert not root.exists()


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
def test_fresh_failover_registration_freezes_model_and_concurrency(tmp_path, dataset, monkeypatch):
    hosts = [f'test-{name}.openai.azure.com' for name in ('one', 'two', 'three')]
    monkeypatch.setenv('HYPERHARNESS_AZURE_FAILOVER_HOSTS', ','.join(hosts))
    source = reference(tmp_path, dataset)
    host = source / 'train/pm3_generator_harness.py'
    loader = '''
def load_generator_credentials(plan):
    configuration = plan['phg']['generator']
    if "swedencentral" not in os.environ.get("AZURE_OPENAI_ENDPOINT", ""):
        raise ValueError('Unexpected endpoint')
'''
    host.write_text(host.read_text() + loader)
    (source / 'train/pm3_mechanism_experience.py').write_text(
        loader.replace('load_generator_credentials', 'load_optimizer_credentials'))
    for relative in ('llm.py', 'per_user_adaptation/phase_a.py'):
        path = source / 'train/source_snapshot/src/ohpu' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / 'src/ohpu' / relative).read_bytes())
    credential = tmp_path / 'outside-registration.env'
    credential.write_text('private-token-not-for-snapshot')
    root = tmp_path / 'study'
    pipeline.register_train(root, dataset, source, failover_credentials=credential, generation_workers=1)
    out = root / dataset / 'train'
    plan = pipeline.read(out / 'plan.json')
    assert plan['optimizer_model'] == plan['phg']['optimizer']['model'] == plan['phg']['generator']['model'] == 'gpt-6-sol'
    assert plan['phg']['generator']['transport_retries'] == 0
    assert plan['phg']['generator']['request_timeout_seconds'] == 930
    assert plan['phg']['luna_workers'] == plan['concurrency']['generation'] == 1
    assert plan['endpoint_failover']['endpoints'] == hosts
    assert plan['endpoint_failover']['max_endpoint_attempts'] == 6
    assert len(plan['endpoint_failover']['patches']) == 4
    assert plan['optimizer']['credential_file'] == str(credential)
    assert 'OHPU_AZURE_FAILOVER_URL' in (out / 'pm3_mechanism_experience.py').read_text()
    assert 'stream_max_retries' in (out / 'source_snapshot/src/ohpu/per_user_adaptation/phase_a.py').read_text()
    assert not list(out.rglob('*.env'))
    assert not (out / 'scores/phg_r0').exists()
    pipeline.shared.verify(out)


def load_module(path, name):
    import importlib.util
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def traced_context(module):
    from types import SimpleNamespace
    seen = []

    class Program:
        fingerprint = SimpleNamespace(hook_sha256={'remember': '0' * 64})

        def call(self, name, values):
            seen.append((name, json.dumps(values, sort_keys=True, default=str)))
            if name != 'recall':
                return SimpleNamespace(patch={'prefix': 'guidance'})
            if values['backend']['step'] == 0:
                return SimpleNamespace(patch={'memory_query': {'operation': 'search', 'algorithm': 'bm25',
                    'collection': 'observations', 'text': 'portable lamp', 'limit': 7, 'state': {'seen': 1}}})
            return SimpleNamespace(patch={'memory_query': {'operation': 'finish', 'output': '', 'source_refs': []}})

    state = object.__new__(module.PersonalState)
    state.user_id, state.program, state.events, state.document = 'user', Program(), [], ''
    return state, state.context('item_ranking', 'find a lamp'), seen


@pytest.mark.parametrize('dataset', ['memorycd', 'mempal'])
def test_registered_runtime_traces_operations_without_changing_hook_inputs(tmp_path, dataset):
    source = reference(tmp_path, dataset)
    pipeline.register_train(tmp_path / 'unified', dataset, source)
    out = tmp_path / 'unified' / dataset / 'train'
    snapshot = out / pipeline.SNAPSHOT
    assert (snapshot / 'per_user_adaptation/harness_trace.py').read_bytes() == pipeline.TRACE_MODULE.read_bytes()
    for name in pipeline.STATE_RUNTIMES:
        text = (snapshot / name).read_text()
        assert pipeline.SILENT_RECORD not in text and 'record=self.harness_trace.record' in text
    row_file, _, row_patch = pipeline.ROW_TRACE_PATCHES[dataset]
    assert row_patch in (snapshot / row_file).read_text()
    binding = pipeline.read(out / 'plan.json')['reference_pipeline']['diagnostic_traces']
    assert binding['runtime_trace'] and not binding['scoring_changed']
    original = load_module(REPO / 'src/ohpu/benchmarks/horizon_runtime.py', 'original_horizon_' + dataset)
    patched = load_module(snapshot / 'benchmarks/horizon_runtime.py', 'patched_horizon_' + dataset)
    _, before, before_inputs = traced_context(original)
    state, after, after_inputs = traced_context(patched)
    assert before == after and before_inputs == after_inputs
    trace = state.harness_trace.export()
    assert trace['retrieval'] and trace['operations'] == ['search', 'finish']
    assert trace['steps'][0] | {'ms': 0} == {'step': 0, 'operation': 'search', 'ms': 0, 'collection': 'observations',
        'algorithm': 'bm25', 'limit': 7, 'text': 'portable lamp', 'state_keys': ['seen'], 'state_chars': 11,
        'rows': 0, 'truncated': False, 'refs': [], 'columns': []}
    pipeline.shared.verify(out)


def test_trace_recorder_summarizes_rows_sql_finish_and_errors():
    from ohpu.per_user_adaptation.harness_trace import Recorder
    recorder = Recorder()
    rows = [{'ref': 'observations:e' + str(index), 'text': 'x', 'score': index / 3} for index in range(70)]
    recorder.record('memory_backend_operation', step=0, decision={'operation': 'search', 'text': 'q' * 900,
        'limit': 64}, result=rows)
    recorder.record('memory_backend_operation', step=1, decision={'operation': 'sql', 'sql': 'SELECT ref FROM records',
        'parameters': ['p'] * 20}, result={'rows': [{'ref': 'observations:e1', 'rating': 5}], 'truncated': True})
    recorder.record('memory_backend_error', step=2, decision={'operation': 'sql', 'sql': 'bad'}, error_type='ValueError')
    recorder.record('memory_backend_operation', step=3, decision={'operation': 'finish', 'output': 'evidence',
        'source_refs': ['observations:e1']}, result={'output': 'evidence', 'source_refs': ['observations:e1']})
    trace = recorder.export()
    search, sql, error, finish = trace['steps']
    assert search['rows'] == 70 and len(search['refs']) == 64 and search['scores'][3] == 1.0
    assert search['text'].endswith('...[900 chars]')
    assert sql['truncated'] and sql['refs'] == ['observations:e1'] and sql['columns'] == ['rating', 'ref']
    assert len(sql['parameters']) == 16 and error['error_type'] == 'ValueError'
    assert finish == {**finish, 'output_chars': 8, 'reference_count': 1, 'source_refs': ['observations:e1']}
    assert trace['operations'] == ['search', 'sql', 'sql', 'finish']
    json.dumps(trace)


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
def test_diagnose_exposes_round_generation_protocols(tmp_path, dataset):
    import shutil
    from types import SimpleNamespace
    source = reference(tmp_path, dataset)
    pipeline.register_train(tmp_path / 'unified', dataset, source)
    out = tmp_path / 'unified' / dataset / 'train'
    if dataset == 'pm3':
        assert not pipeline.read(out / 'plan.json')['reference_pipeline']['diagnostic_traces']['runtime_trace']
    diagnosed = []
    host = {'DIAGNOSED': diagnosed, 'read': pipeline.read, 'write': pipeline.write, 'shutil': shutil,
            'now': lambda: 'now', 'round_arm': lambda plan, version: 'phg_r' + str(version)}
    exec(compile((out / 'pm3_generator_harness.py').read_text(), 'registered-host', 'exec'), host)
    assert 'harness_trace' in host['framework'](pipeline.read(out / 'plan.json'), None)
    diagnose = host['diagnose']
    if dataset != 'pm3':
        driver = {'DIAGNOSED': diagnosed, 'PHG': SimpleNamespace(expose_generation=host['expose_generation'])}
        exec(compile((out / (dataset + '_phg.py')).read_text(), 'registered-driver', 'exec'), driver)
        diagnose = driver['diagnose']
    slot = out / 'harness/phg_r0/train_user'
    write(slot / 'harness.json', {'source': 'program', 'generation': {'valid': False, 'error': 'repair limit'}})
    (slot / 'generator').mkdir()
    (slot / 'generator/protocol.jsonl').write_text('{"op": "llm"}\n{"op": "check"}\n')
    diagnose(out, 0)
    target = out / 'diagnostics/round0/generation'
    assert diagnosed == [('host' if dataset == 'pm3' else 'dataset', 0)]
    assert (target / 'train_user.jsonl').read_bytes() == (slot / 'generator/protocol.jsonl').read_bytes()
    assert pipeline.read(target / 'train_user.json') == {'valid': False, 'error': 'repair limit'}
    assert pipeline.read(target / 'done.json')['users'] == ['train_user']
    diagnose(out, 0)
    plan = pipeline.read(out / 'plan.json')
    write(out / 'plan.json', {**plan, 'split': 'TEST'})
    host['expose_generation'](out, 1)
    assert not (out / 'diagnostics/round1').exists()

def test_reference_rejects_v2_or_population_source(tmp_path):
    source = reference(tmp_path, 'mempal')
    plan = pipeline.read(source / 'train/plan.json')
    plan['harness_contract'] = {'version': 'v2'}
    write(source / 'train/plan.json', plan)
    with pytest.raises(ValueError, match='predate'):
        pipeline.register_train(tmp_path / 'unified', 'mempal', source)
    assert not (tmp_path / 'unified').exists()


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
def test_context_ablation_changes_only_memorycd_adapter(tmp_path, dataset):
    source = reference(tmp_path, dataset)
    control, treatment = tmp_path / 'control', tmp_path / 'treatment'
    pipeline.register_train(control, dataset, source, variant='reference')
    pipeline.register_train(treatment, dataset, source, variant='context')
    left = control / dataset / 'train/generator/v0'
    right = treatment / dataset / 'train/generator/v0'
    names = {path.name for path in left.iterdir()} | {path.name for path in right.iterdir()}
    changed = {name for name in names if not (left / name).exists() or not (right / name).exists()
               or (left / name).read_bytes() != (right / name).read_bytes()}
    assert changed == ({'history_adapter.py'} if dataset == 'memorycd' else set())
    pipeline.shared.verify(treatment / dataset / 'train')


def test_prepare_can_select_one_explicit_reference(tmp_path, monkeypatch):
    source = reference(tmp_path, 'memorycd')
    registered = []

    def execute(command, **options):
        dataset = command[command.index('--dataset') + 1]
        reference_path = Path(command[command.index('--reference') + 1])
        registered.append((dataset, reference_path))
        pipeline.register_train(Path(command[command.index('--root') + 1]), dataset, reference_path)

    monkeypatch.setattr(pipeline.subprocess, 'run', execute)
    root = tmp_path / 'study'
    monkeypatch.setattr(pipeline.sys, 'argv', ['runner', 'prepare', '--root', str(root),
        '--datasets', 'memorycd', '--memorycd-reference', str(source)])
    pipeline.main()
    study = pipeline.read(root / 'study.json')
    assert registered == [('memorycd', source)] and study['dataset_order'] == ['memorycd']
    assert study['references'] == {'memorycd': str(source)}
    assert list(study['registrations']) == ['memorycd/train/registration.json']
    assert (root / 'code' / pipeline.TRACE_MODULE.relative_to(pipeline.REPO)).read_bytes() == pipeline.TRACE_MODULE.read_bytes()
    with pytest.raises(ValueError, match='at least one'):
        pipeline.prepare(tmp_path / 'empty', datasets=[])


def test_reference_registration_ignores_removed_baseline_outputs(tmp_path):
    source = reference(tmp_path, 'memorycd')
    pipeline.shared.freeze(source / 'train')
    pipeline.shared.freeze(source / 'test')
    (source / 'train/scores/no_history/train_user/summary.json').unlink()
    (source / 'test/scores/no_history/test_user/summary.json').unlink()
    root = tmp_path / 'unified'
    pipeline.register_train(root, 'memorycd', source)
    assert not (root / 'memorycd/train/scores').exists()


@pytest.mark.parametrize('relative', ['train/inputs/fixture.json', 'test/plan.json'])
def test_reference_registration_rejects_changed_frozen_inputs(tmp_path, relative):
    source = reference(tmp_path, 'mempal')
    pipeline.shared.freeze(source / 'train')
    pipeline.shared.freeze(source / 'test')
    (source / relative).write_text('{}')
    with pytest.raises(ValueError, match='Registered file changed'):
        pipeline.register_train(tmp_path / 'unified', 'mempal', source)


def test_reference_stage_dispatches_original_driver(tmp_path, monkeypatch):
    source = reference(tmp_path, 'mempal')
    root = tmp_path / 'unified'
    pipeline.register_train(root, 'mempal', source)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    pipeline.run_stage(root, 'mempal', 'train')
    command, options = calls[0]
    assert Path(command[3]) == root / 'mempal/train/mempal_phg.py'
    assert command[4:] == ['run', '--out', str(root / 'mempal/train')]
    assert options['check'] is True


@pytest.mark.parametrize('scores,parent,expected', [
    ({'a': 0, 'b': 10}, {'a': 4, 'b': 4}, True),
    ({'a': 5, 'b': 3}, {'a': 4, 'b': 4}, False),
    ({'a': 2, 'b': 3}, {'a': 4, 'b': 4}, False),
    ({'a': 100}, {'a': 4, 'b': 4}, False),
    ({'a': 10, 'b': 10}, {'a': 4}, False),
    ({'a': float('nan'), 'b': 10}, {'a': 4, 'b': 4}, False),
])
def test_mean_gain_accepts_only_complete_strict_improvement(scores, parent, expected):
    assert pipeline.mean_gain_accept(scores, parent, ['a', 'b']) is expected


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
def test_installed_selection_really_bypasses_historical_gates(tmp_path, dataset):
    from types import SimpleNamespace
    source = reference(tmp_path, dataset)
    out = source / 'train'
    pipeline.install_selection(out, dataset)
    state = {'version': 0}
    scores = {'phg_r0': {'a': 4, 'b': 4}, 'phg_r1': {'a': 0, 'b': 10}}
    namespace = {'read': lambda path: {'split': 'TRAIN', 'test_users': ['a', 'b']} if path.name == 'plan.json' else state,
        'arm_scores': lambda root, plan, arm: scores[arm], 'round_arm': lambda plan, version: f'phg_r{version}',
        'replace': lambda path, value: None}
    exec(compile((out / 'pm3_generator_harness.py').read_text(), 'driver', 'exec'), namespace)
    api = SimpleNamespace(**namespace)
    if dataset == 'pm3':
        function = namespace['round_summary']
    else:
        adapter = {'PHG': api}
        exec(compile((out / (dataset + '_phg.py')).read_text(), 'adapter', 'exec'), adapter)
        function = adapter['round_summary']
        assert pipeline.SELECTION_RULE in adapter['framework']({}, None)
        assert pipeline.MODULAR_OPTIMIZATION in adapter['framework']({}, None)
    assert function(out, 1)['parent_decision']['adopted']
    assert state['version'] == 1
    assert pipeline.SELECTION_RULE in api.framework({}, None)


def test_conservative_gain_rejects_noisy_and_incomplete_candidates():
    users = [f'user_{index}' for index in range(10)]
    parent = {user: 30.0 for user in users}
    noisy = {user: 30.1 + (5 if index % 2 else -5) for index, user in enumerate(users)}
    assert pipeline.mean_gain_accept(noisy, parent, users)
    assert not pipeline.conservative_gain_decision(noisy, parent, users)['adopted']
    assert pipeline.conservative_gain_decision({user: 31.0 for user in users}, parent, users)['adopted']
    assert not pipeline.conservative_gain_decision({users[0]: 99.0}, parent, users)['complete']
    assert not pipeline.conservative_gain_decision(parent, parent, users)['adopted']


def test_conservative_selection_is_installed_before_fresh_optimization(tmp_path):
    import shutil
    source = reference(tmp_path, 'memorycd')
    out = source / 'train'
    shutil.rmtree(out / 'optimizer')
    (out / 'selection.json').unlink()
    pipeline.install_selection(out, 'memorycd')
    plan = pipeline.read(out / 'plan.json')
    plan.update(split='TRAIN', dataset='memorycd', executor_model='Qwen3.5-9B', seed=20261022)
    plan.setdefault('phg', {})['rounds'] = 11
    pipeline.shared.write(out / 'plan.json', plan)
    pipeline.install_conservative_selection(out)
    namespace = {}
    exec(compile((out / 'pm3_generator_harness.py').read_text(), 'driver', 'exec'), namespace)
    users = [f'user_{index}' for index in range(10)]
    parent = {user: 30.0 for user in users}
    noisy = {user: 30.1 + (5 if index % 2 else -5) for index, user in enumerate(users)}
    assert not namespace['mean_gain_accept'](noisy, parent, users)
    assert namespace['mean_gain_accept']({user: 31.0 for user in users}, parent, users)
    assert 'Conservative adoption override' in namespace['framework']({}, None)
    assert pipeline.MODULAR_OPTIMIZATION in namespace['framework']({}, None)
    pipeline.shared.verify(out)
    (out / 'optimizer').mkdir()
    with pytest.raises(ValueError, match='before optimization'):
        pipeline.install_conservative_selection(out)


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
@pytest.mark.parametrize('requested', ['low', 'medium', 'high'])
def test_frozen_generator_client_forces_medium(tmp_path, dataset, requested):
    source = reference(tmp_path, dataset)
    out = source / 'train'
    pipeline.install_generation(out)
    seen = []

    def factory(model, **configuration):
        assert model == 'gpt-6-sol'
        assert configuration['retries'] == 3

        def call(messages, **options):
            seen.append(options)
            return {'content': 'code', 'usage': {}}
        return call

    namespace = {'luna': factory}
    exec(compile((out / 'pm3_generator_harness.py').read_text(), 'driver', 'exec'), namespace)
    plan = {'phg': {'generator': {'model': 'gpt-6-sol', 'reasoning_effort': 'medium',
                                  'transport_retries': 3, 'request_timeout_seconds': 120}}}
    reply = namespace['configured_luna'](plan, None)([], max_tokens=123, effort=requested)
    assert seen == [{'max_tokens': 123, 'effort': 'medium'}]
    assert reply['effective_effort'] == 'medium'
    assert reply['requested_effort'] == requested
    plan['phg']['generator']['reasoning_effort'] = 'low'
    with pytest.raises(ValueError, match='requires gpt-6-sol medium'):
        namespace['configured_luna'](plan, None)


def test_luna_prepare_copies_executor_module(tmp_path, monkeypatch):
    root = tmp_path / 'luna'

    def register(command, check):
        assert check
        assert command[command.index('--executor-profile') + 1] == 'gpt56-luna'
        module = root / 'code/src/ohpu/benchmarks/luna_executor.py'
        assert module.read_bytes() == pipeline.executor_profile.LUNA_MODULE.read_bytes()
        write(root / 'mempal/train/registration.json', {'files': {}})

    monkeypatch.setattr(pipeline.subprocess, 'run', register)
    pipeline.prepare(root, datasets=['mempal'], executor='gpt56-luna')
    assert pipeline.read(root / 'study.json')['executor_profile'] == 'gpt56-luna'


@pytest.mark.parametrize('provider', ['openai', 'azure'])
def test_direct_api_is_frozen_for_train_and_test(tmp_path, monkeypatch, provider):
    import os

    source = reference(tmp_path, 'pm3')
    train_source = source / 'train'
    host = train_source / 'pm3_generator_harness.py'
    host.write_text(host.read_text() + '\ndef load_generator_credentials(plan):\n    raise RuntimeError("legacy")\n')
    source_files = {
        'base_runner.py': REPO / 'scripts/run_lamp_skill.py',
        'pm3_mechanism_experience.py': pipeline.PERMISSION_MODULE.with_name('pm3_mechanism_experience.py'),
        pipeline.SNAPSHOT + 'llm.py': REPO / 'src/ohpu/llm.py',
    }
    for target, original in source_files.items():
        (train_source / target).write_bytes(original.read_bytes())
    optimizer = train_source / pipeline.SNAPSHOT / 'per_user_adaptation'
    (optimizer / 'controllers').mkdir()
    (optimizer / 'phase_a.py').write_text('def _safe_codex_args(bundle_root, model=None, reasoning_effort=None):\n    pass\n')
    (optimizer / 'memory_evolution.py').write_text(
        'class NativeCodexProposer:\n'
        '    def __init__(self, model):\n'
        '        from .phase_a import _safe_codex_args\n'
        '        environment = {"AZURE_OPENAI_API_KEY": os.environ["AZURE_OPENAI_API_KEY"]} '
        'if os.environ.get("AZURE_OPENAI_ENDPOINT") else None\n'
        '        self.controller = Codex(model=model, env=environment, effort="medium")\n')
    (optimizer / 'controllers/local_cli.py').write_text(
        'class Controller:\n    def launch(self):\n        if True:\n'
        '            if True:\n                merged_env.pop("OPENAI_API_KEY", None)\n')

    api = {'provider': provider, 'generator_model': 'gpt-6-sol', 'optimizer_model': 'gpt-6-sol',
           'answer_model': 'gpt-5.6-luna', 'judge_model': 'gpt-5.5'}
    if provider == 'azure':
        api.update(generator_model='construct', optimizer_model='optimize', answer_model='answer', judge_model='judge')
    root = tmp_path / 'unified'
    pipeline.register_train(root, 'pm3', source, api=api)
    train = root / 'pm3/train'
    write(train / 'selection.json', {'version': 0})
    write(train / 'scores/phg_r0/train_user/summary.json', {'evaluation_complete': True})
    monkeypatch.setattr(sys, 'path', list(sys.path))
    pipeline.register_test(root, 'pm3')
    for name, value in {
        'OPENAI_API_KEY': 'fake-public-key',
        'OPENAI_BASE_URL': 'https://other.example/v1',
        'AZURE_OPENAI_ENDPOINT': 'https://example-resource.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'fake-azure-key',
        'AZURE_OPENAI_API_VERSION': '2024-12-01-preview',
        'AZURE_OPENAI_RESPONSES_API_VERSION': '',
        'HYPERHARNESS_PROVIDER': 'stale',
    }.items():
        monkeypatch.setenv(name, value)

    def run(out, *, experiment_key):
        text = (out / 'plan.json').read_text()
        assert 'fake-public-key' not in text and 'fake-azure-key' not in text
        plan = json.loads(text)
        assert plan[experiment_key]['model_api'] == api
        assert plan['phg']['generator']['model'] == api['generator_model']
        assert plan['phg']['optimizer']['model'] == plan['optimizer_model'] == api['optimizer_model']
        assert plan['executor_model'] == api['answer_model']
        assert plan['judge_model'] == api['judge_model']
        assert os.environ['HYPERHARNESS_PROVIDER'] == provider
        if provider == 'openai':
            assert 'AZURE_OPENAI_ENDPOINT' not in os.environ
            assert os.environ['OPENAI_BASE_URL'] == 'https://api.openai.com/v1'
        else:
            assert os.environ['AZURE_OPENAI_RESPONSES_API_VERSION'] == '2025-04-01-preview'
        for file, loader, args in (
            ('pm3_generator_harness.py', 'load_generator_credentials', (plan,)),
            ('pm3_mechanism_experience.py', 'load_optimizer_credentials', (plan,)),
            ('base_runner.py', 'load_credentials', ()),
        ):
            scope = {'os': os}
            exec(pipeline.shared.function_source((out / file).read_text(), loader), scope)
            scope[loader](*args)

    monkeypatch.setattr(pipeline.shared, 'run_pm3', run)
    for split in ('train', 'test'):
        pipeline.run_stage(root, 'pm3', split)
    if provider == 'azure':
        from types import SimpleNamespace
        from openai import AzureOpenAI, OpenAI

        llm_path = train / pipeline.SNAPSHOT / 'llm.py'
        scope = {'__name__': __name__}
        exec(compile(llm_path.read_text(), str(llm_path), 'exec'), scope)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='ok', tool_calls=None), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=1))
        with (AzureOpenAI(api_version=os.environ['AZURE_OPENAI_API_VERSION']) as azure,
              OpenAI(base_url='http://127.0.0.1:8001/v1', api_key='local') as qwen):
            for sdk, model, uses_reasoning in [(azure, 'construct', True), (qwen, 'Qwen3.5-9B', False)]:
                requests = []
                monkeypatch.setattr(sdk.chat.completions, 'create', lambda **payload: requests.append(payload) or response)
                llm = scope['LLM'](model=model, _client=sdk, max_retries=1)
                assert llm.chat([], max_tokens=1024, seed=None)['content'] == 'ok'
                request, = requests
                assert request['model'] == model
                assert ('max_completion_tokens' in request) == uses_reasoning
                assert ('temperature' in request) != uses_reasoning


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
@pytest.mark.parametrize('selected', [0, 1])
@pytest.mark.parametrize('executor', [None, 'qwen38-27b', 'gpt56-luna'])
def test_medium_test_registration_uses_only_fresh_personal_arms(tmp_path, monkeypatch, dataset, selected, executor):
    from types import SimpleNamespace

    source = reference(tmp_path, dataset)
    root = tmp_path / 'unified'
    for split, user in [('train', 'train_user'), ('test', 'test_user')]:
        write(source / split / 'harness/no_history' / user / 'harness.json', {'source': 'archived baseline'})
    pipeline.register_train(root, dataset, source, executor=executor)
    train = root / dataset / 'train'
    trained = pipeline.read(train / 'plan.json')
    expected_model = {None: 'Qwen3.5-9B', 'qwen38-27b': 'Qwen3.8-27B', 'gpt56-luna': 'gpt-5.6-luna'}[executor]
    assert trained['executor_model'] == expected_model
    assert not (train / 'harness/no_history').exists()
    assert not (train / 'scores/no_history').exists()
    write(train / 'selection.json', {'version': selected})
    write(train / 'scores' / f'phg_r{selected}' / 'train_user/summary.json',
          {'evaluation_complete': True, 'runtime_violations': [], 'accuracy_pct_micro': 50})
    if selected:
        write(train / 'generator/v1/fixture.json', {'selected': True})
        write(train / 'generator/v1.json', {'version': 1})

    if dataset != 'pm3':
        original = pipeline.read(source / 'test/plan.json')

        def register_mock(out, prepared, host, *, reference):
            assert host == reference == train and prepared == Path(trained['prepared'])
            assert (host / f'{dataset}_phg.py').read_text().startswith('BASELINES = ()')
            testing = {**original, 'executor_model': trained['executor_model'],
                'phg': {**original['phg'], 'arms': {'test_initial': 0, 'test_optimized': selected}}}
            write(out / 'plan.json', testing)

        driver = SimpleNamespace(register=register_mock)
        monkeypatch.setattr(pipeline.importlib.util, 'spec_from_file_location',
                            lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)))
        monkeypatch.setattr(pipeline.importlib.util, 'module_from_spec', lambda spec: driver)

    pipeline.register_test(root, dataset)
    testing = root / dataset / 'test'
    plan = pipeline.read(testing / 'plan.json')
    assert plan['arms'] == list(plan['phg']['arms'])
    assert plan['external_baselines'] == {'arms': [], 'files': {}}
    assert not (testing / 'harness').exists()
    assert not (testing / 'scores').exists()
    assert plan['executor_model'] == expected_model
    assert pipeline.read(testing / 'reference/test_reuse.json') == {
        'personal_r0_reused': False, 'generation_effort': 'medium', 'selected_r0_alias': selected == 0}
    if dataset == 'pm3':
        names = ['phg_test_r0', 'phg_test_r1'] if selected else ['phg_test_r0']
        assert plan['arms'] == names
        assert pipeline.read(testing / 'private/requests/test_user.json') == {'q1': 'TEST request'}
    else:
        assert plan['phg']['arms'] == {'test_initial': 0, 'test_optimized': selected}
    if executor == 'gpt56-luna':
        assert plan['executor_url'] == 'azure-configured'
        assert plan['score_workers'] == plan['answer_workers'] == plan['judge_workers'] == 2
        assert plan['concurrency']['executor'] == 2
        if dataset != 'pm3':
            assert plan['tokenizer_model'] == 'o200k_base'
            assert plan['tokenizer_revision'] == 'tiktoken-installed'
    pipeline.shared.verify(testing)


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
@pytest.mark.skipif(sys.platform != 'linux', reason='Generator subprocess requires Linux RLIMIT_AS support')
def test_common_generation_runs_personal_section_and_six_repairs(tmp_path, monkeypatch, dataset):
    import ast
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as phg
    from ohpu.experiments import pm3_official as official
    protocol = pipeline.generation_protocol(Path(phg.__file__).read_text())
    configuration = next(ast.literal_eval(node.value) for node in ast.parse(protocol).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'GENERATOR' for target in node.targets))
    monkeypatch.setattr(phg, 'GENERATOR', configuration)
    spec = tmp_path / 'spec'
    phg.initial_spec(spec, full_program=True)
    (spec / 'generator.py').write_text(pipeline.generation_source())
    (spec / 'prompt.md').write_text(pipeline.GENERATION_PROMPT)
    (spec / 'program_api.md').write_text('runtime for ' + dataset)
    (spec / 'history_adapter.py').write_text('def digest(events):\n    return events[0]["text"]\n')
    policy_path = spec / 'generation_policy.json'
    policy = pipeline.read(policy_path) if policy_path.exists() else {}
    policy.update({key: pipeline.BUDGETS[key] for key in ('llm_calls', 'checks', 'completion_tokens')})
    pipeline.shared.write(policy_path, policy)
    output = '''USER_MODEL = {}

def user_route(request, kind, previous):
    return {'history': True, 'queries': [request]}

def user_keep(text, app, kind):
    return True

def user_guidance(request, kind, evidence):
    return 'User prefers concise answers.'

def u_recall(context):
    return {}

def u_apply(context):
    return {}
'''
    calls = []
    checks = []

    def llm(messages, **options):
        calls.append((messages, options))
        return {'content': '```python\n' + output + '```', 'usage': {}}

    def check(program):
        checks.append(program)
        return 'repair required' if len(checks) <= 6 else None

    result = phg.run_generator(spec, [{'id': 'h0', 'text': 'legal user history', 'event': {}, 'ts': 1}],
        llm=llm, check=check, sandbox=False, workdir=tmp_path / 'generation',
        open_contract=True, require_model_code=True)
    assert result['error'] is None
    from analysis.skill_optimization_diagnosis import harness_permission
    backbone = (spec / 'backbone.py').read_text()
    assert result['program'] == backbone.replace('#@USER_SECTION@#', output.strip() + '\n')
    harness_permission.validate_program(result['program'], backbone, 'section')
    official.validate_program(result['program'], editable_hooks=True)
    assert len(calls) == 7
    assert all(options['effort'] == 'medium' and options['max_tokens'] == 16000 for _, options in calls)
    assert 'runtime for ' + dataset in calls[0][0][0]['content']
    assert 'legal user history' in calls[0][0][0]['content']
    assert '<<DATASET_RUNTIME>>' not in calls[0][0][0]['content']
    assert 'repair required' in calls[-1][0][-1]['content']


def test_common_generator_keeps_section_fallback(tmp_path):
    namespace = {'__file__': str(tmp_path / 'generator.py'), '__name__': 'probe'}
    exec(compile(pipeline.generation_source(), 'generator', 'exec'), namespace)
    output = namespace['submitted_program']('```python\nUSER_MODEL = {}\n```',
                                            'PROGRAM_API_VERSION = 1\n#@USER_SECTION@#')
    assert output == 'PROGRAM_API_VERSION = 1\nUSER_MODEL = {}\n'


def test_common_generator_extracts_pm3_runtime_without_changing_backbone(tmp_path):
    source = reference(tmp_path, 'pm3')
    spec = source / 'train/generator/v0'
    (spec / 'program_api.md').unlink()
    (spec / 'prompt.md').write_text('header\n## Dataset runtime contract\nPM3 runtime\n## Language and limits\nlimits')
    (spec / 'backbone.py').write_text('original backbone')
    pipeline.install_generation(source / 'train')
    assert (spec / 'program_api.md').read_text() == 'PM3 runtime\n'
    assert (spec / 'backbone.py').read_text() == 'original backbone'


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
@pytest.mark.parametrize('change', ['entrypoint', 'constant', 'none'])
def test_frozen_section_guard_cannot_be_bypassed_by_generator(tmp_path, dataset, change):
    import ast
    import hashlib
    source = reference(tmp_path, dataset)
    out = source / 'train'
    pipeline.install_generation(out)
    spec = out / 'generator/v0'
    backbone = 'PROGRAM_API_VERSION = 1\nBACKBONE = {}\n#@USER_SECTION@#\ndef guard(context):\n    return {}\n'
    (spec / 'backbone.py').write_text(backbone)
    (spec / 'control_section.py').write_text('USER_MODEL = {}\n')
    program = backbone.replace('#@USER_SECTION@#', 'USER_MODEL = {}')
    if change == 'entrypoint':
        program = program.replace('return {}', "return {'prefix': 'changed'}")
    elif change == 'constant':
        program += '\ndef user_mutate():\n    BACKBONE["changed"] = True\n'
    namespace = {'HERE': out, 'Path': Path, 'hashlib': hashlib, 'ast': ast}
    exec(compile((out / 'pm3_generator_harness.py').read_text(), 'frozen', 'exec'), namespace)
    namespace['unrestricted_run_generator'] = lambda *args, **kwargs: {'program': program, 'error': None}
    namespace['recall_stress_check'] = lambda *args: None
    result = namespace['run_generator'](spec, [], check=lambda program: None)
    assert bool(result['error']) is (change != 'none')


def test_host_completes_only_missing_called_personal_hooks(tmp_path):
    from analysis.skill_optimization_diagnosis import harness_permission
    backbone = 'PROGRAM_API_VERSION = 1\n#@USER_SECTION@#\ndef recall(context):\n    return u_recall(context)\n'
    (tmp_path / 'backbone.py').write_text(backbone)
    (tmp_path / 'control_section.py').write_text('USER_MODEL = {}\ndef u_recall(context):\n    return {}\n')
    program = pipeline.section_program(backbone.replace('#@USER_SECTION@#', 'USER_MODEL = {}'), tmp_path, harness_permission)
    assert 'def u_recall(context)' in program
    harness_permission.validate_program(program, backbone, 'section')


@pytest.mark.parametrize('dataset', list(pipeline.REFERENCES))
@pytest.mark.parametrize('setting,expected', [(47, 47), (999, 64), (0, 1), ('bad', 10), (True, 10)])
def test_personal_top_k_reaches_search(tmp_path, dataset, setting, expected):
    from ohpu.per_user_adaptation.program import HarnessProgram
    source = reference(tmp_path, dataset)
    out = source / 'train'
    pipeline.install_budgets(out)
    spec = out / 'generator/v0'
    backbone = (spec / 'backbone.py').read_text()
    personal = ('USER_MODEL = {"top_k": ' + repr(setting) + '}\n'
                'def user_route(request, kind, extra):\n    return {}\n')
    program = HarnessProgram(backbone.replace('#@USER_SECTION@#', personal))
    result = program.call('recall', {'stage': 'context', 'query': 'example', 'task_kind': None}).patch
    assert result['memory_query']['limit'] == expected
    assert result['memory_query']['state']['top_k'] == expected
    override = personal.replace('return {}', "return {'top_k': 23}")
    program = HarnessProgram(backbone.replace('#@USER_SECTION@#', override))
    result = program.call('recall', {'stage': 'context', 'query': 'example', 'task_kind': None}).patch
    assert result['memory_query']['limit'] == 23
    assert 'DEFAULT_LINE_BUDGET = 40_000' in (out / 'source_snapshot/src/ohpu/per_user_adaptation/program.py').read_text()
    assert pipeline.read(spec / 'generation_policy.json')['checks'] == 10


def stress_program(body):
    return ('PROGRAM_API_VERSION = 1\n\ndef remember(context):\n    return {}\n\n'
            'def apply(context):\n    return {}\n\ndef guard(context):\n    return {}\n\n'
            'def recall(context):\n' + body)


def test_stress_check_reports_real_hook_step_limit():
    source = stress_program('''    total = 0
    for outer in range(300):
        for inner in range(300):
            total = total + inner
    return {'memory_query': {'operation': 'finish', 'output': '', 'source_refs': []}}
''')
    with pytest.raises(Exception, match='execution steps'):
        pipeline.recall_stress_check(source, [])


def test_stress_check_rejects_empty_callback_and_unfinished_loop():
    empty = stress_program('''    if context['backend']['step'] == 0:
        return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': 'history', 'limit': 1}}
    return {}
''')
    with pytest.raises(ValueError, match='never.*mid-retrieval'):
        pipeline.recall_stress_check(empty, [])
    endless = stress_program("    return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': 'history', 'limit': 1}}\n")
    with pytest.raises(ValueError, match='did not finish'):
        pipeline.recall_stress_check(endless, [])


def test_stress_check_respects_limit_and_defers_sql():
    source = stress_program('''    if context['backend']['step'] == 0:
        return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': 'history', 'limit': 1}}
    if len(context['backend']['result']['data']) > 1:
        return {}
    return {'memory_query': {'operation': 'finish', 'output': '', 'source_refs': []}}
''')
    pipeline.recall_stress_check(source, [{'text': 'history'}] * 10)
    sql = stress_program("    return {'memory_query': {'operation': 'sql', 'sql': 'SELECT text AS custom_column FROM records LIMIT 1'}}\n")
    pipeline.recall_stress_check(sql, [{'text': 'history'}])


def test_stress_check_catches_cumulative_reference_overflow():
    source = stress_program('''    backend = context['backend']
    state = backend.get('state') or {'refs': [], 'texts': []}
    for row in backend['result'].get('data') or []:
        state['texts'].append(row['text'])
        state['refs'].append(row['ref'])
    if backend['step'] >= 3:
        return {'memory_query': {'operation': 'finish', 'output': '\\n'.join(state['texts']), 'source_refs': state['refs']}}
    return {'memory_query': {'operation': 'search', 'algorithm': 'bm25', 'collection': 'observations', 'text': 'history', 'limit': 24, 'state': state}}
''')
    events = [{'text': 'tiny'}] * 64
    with pytest.raises(ValueError, match='cumulative source reference limit'):
        pipeline.recall_stress_check(source, events)
    pipeline.recall_stress_check(pipeline.bound_recall_references(source), events)


def test_stress_failure_is_fed_back_and_cannot_bypass_final_submission(tmp_path):
    import ast
    import hashlib
    source = reference(tmp_path, 'pm3')
    out = source / 'train'
    pipeline.install_generation(out)
    spec = out / 'generator/v0'
    backbone = stress_program('    return u_recall(context)\n').replace('PROGRAM_API_VERSION = 1',
        'PROGRAM_API_VERSION = 1\n#@USER_SECTION@#')
    (spec / 'backbone.py').write_text(backbone)
    (spec / 'control_section.py').write_text('USER_MODEL = {}\n')
    bad = backbone.replace('#@USER_SECTION@#', '''USER_MODEL = {}
def u_recall(context):
    total = 0
    for outer in range(300):
        for inner in range(300):
            total = total + inner
    return {}
''')
    good = backbone.replace('#@USER_SECTION@#', '''USER_MODEL = {}
def u_recall(context):
    return {'memory_query': {'operation': 'finish', 'output': '', 'source_refs': []}}
''')
    namespace = {'HERE': out, 'Path': Path, 'hashlib': hashlib, 'ast': ast}
    exec(compile((out / 'pm3_generator_harness.py').read_text(), 'frozen', 'exec'), namespace)

    def generate(spec_dir, events, **options):
        assert 'execution steps' in options['check'](bad)
        assert options['check'](good) is None
        return {'program': good, 'error': None}
    namespace['unrestricted_run_generator'] = generate
    assert namespace['run_generator'](spec, [], check=lambda program: None)['error'] is None
    namespace['unrestricted_run_generator'] = lambda *args, **kwargs: {'program': bad, 'error': None}
    assert 'execution steps' in namespace['run_generator'](spec, [], check=lambda program: None)['error']
