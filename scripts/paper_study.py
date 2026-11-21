"""Register a fresh native benchmark study after checking the released paper split."""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from paper_splits import LOCK, ROOT, digest, memorycd, mempal, read, verify


HOST = {'pm3_native_skillopt': 'run_pm3_native_skillopt', 'pm3_skillopt': 'run_pm3_skillopt',
        'pm3_harness': 'run_pm3_harness', 'pm3_official_driver': 'run_pm3_official',
        'runner': 'run_lamp_seed_harness', 'base_runner': 'run_lamp_skill'}
INITIAL_SPEC = {
    'pm3': '48a6ab4c8c5674924e7c87c99dfbafcf9a1f03219f3e67d09d7eb1e2aa9d464f',
    'mempal': 'f88a0ab8e9800a1c23c9ac79f6690afd653bcf1995511e69b5831af62b7547d9',
    'memorycd': 'f9195ab591e738334519a311e90f587a36ed7189af78d475aa00c60d710b8a52',
}


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)


def check_panel(actual, expected):
    if set(actual) != set(expected) or any(
        any(actual[uid].get(key) != value for key, value in entry.items()) for uid, entry in expected.items()
    ):
        raise ValueError('Native registration differs from the paper user/query/cutoff lock')


def luna_train_panel(dataset, panel):
    if dataset != 'mempal':
        return panel
    user, excluded = 'mempal_0000', '0000_sample30_topic-3'
    ids = panel[user]['evaluated']
    if ids.count(excluded) != 1:
        raise ValueError('Fixed Luna TRAIN exclusion is not in the original panel')
    return {**panel, user: {**panel[user], 'evaluated': [qid for qid in ids if qid != excluded]}}


def install_luna_hooks(train, dataset, plan, pipeline):
    pipeline.executor_profile.install_luna(train, dataset, plan)
    if dataset == 'mempal':
        uid = 'mempal_0000'
        native_path = train / 'private/native' / (uid + '.json')
        native = read(native_path)
        original = plan['splits'][uid]['evaluated']
        plan['splits'] = luna_train_panel(dataset, plan['splits'])
        retained = set(plan['splits'][uid]['evaluated'])
        native.update(cases=[case for case in native['cases'] if case['case_id'] in retained],
                      evaluation_mask={'protocol': 'mempal-fixed-content-filter-mask-v1', 'original_query_ids': original,
                          'excluded_query_ids': ['0000_sample30_topic-3'],
                          'mask_sha256': '1efc908912ec6cbc1a9e3443a5c3e3c04fc0af42809811b920a5fe86f6211641'})
        pipeline.shared.write(native_path, native)
        path = train / 'perma_phg.py'
        path.write_text(pipeline.replace_function(path.read_text(), 'preflight', '''def preflight(out):
    from ohpu.benchmarks.luna_executor import LunaTokenizer, load_credentials
    load_credentials()
    LunaTokenizer()
    PHG.replace(out / 'operations/preflight.json', {'status': 'ready', 'executor': 'gpt-5.6-luna'})'''))
        return
    from scripts import paper_luna_selection as targeted
    snapshot = train / 'source_snapshot/src/ohpu/benchmarks'
    shutil.copyfile(Path(targeted.__file__), snapshot / 'targeted_optimization.py')
    plan['targeted_optimization'] = {'policy': targeted.POLICY, 'groups': targeted.groups(plan['test_users'], plan['seed'])}
    driver = train / ('pm3_generator_harness.py' if dataset == 'pm3' else 'memorycd_phg.py')
    hooks = "sys.path.insert(0, str(HERE / 'source_snapshot/src'))\n"
    if dataset == 'memorycd':
        policy = read(LOCK / 'luna-memorycd-filter.json')
        for uid, qids in policy['users'].items():
            if not set(qids) <= set(plan['splits'][uid]['evaluated']):
                raise ValueError('Unknown prior MemoryCD filter')
        pipeline.shared.write(train / 'inputs/filter_exclusions.json', policy)
        plan['filter_successor'] = {'protocol': policy['protocol']}
        hooks += 'from ohpu.benchmarks.memorycd_public_filter import install_hooks\ninstall_hooks(globals())\n'
    hooks += f'from ohpu.benchmarks.targeted_optimization import install_namespace\ninstall_namespace(globals(), {dataset!r})\n'
    source = driver.read_text()
    anchor = 'if __name__ == "__main__":'
    if source.count(anchor) != 1:
        raise ValueError('Unexpected driver entrypoint')
    driver.write_text(source.replace(anchor, hooks + '\n' + anchor))


def configured_construction(plan, retry_log, transport_factory):
    configuration = plan['phg']['generator']
    if configuration['model'] != 'gpt-6-sol' or configuration['reasoning_effort'] not in ('low', 'medium'):
        raise ValueError('Unregistered construction model or effort')
    transport = transport_factory(configuration['model'], retries=configuration.get('transport_retries', 0),
        request_timeout_seconds=configuration['request_timeout_seconds'], retry_log=retry_log,
        **{key: configuration[key] for key in ('connection_keepalive', 'fresh_client_on_retry') if key in configuration})

    def call(messages, *, max_tokens, effort):
        actual = configuration['reasoning_effort']
        result = transport(messages, max_tokens=max_tokens, effort=actual)
        return {**result, 'requested_effort': effort, 'effective_effort': actual}
    return call

def install_paper_profile(train, dataset, executor='9b'):
    sys.path.insert(0, str(ROOT))
    from scripts import run_hyperharness_reference as pipeline
    import inspect
    plan = read(train / 'plan.json')
    for template in ('memorycd_history_adapter.py', 'memorycd_simple_history_adapter.py'):
        (train / 'generator/v0' / template).unlink(missing_ok=True)
    pipeline.install_selection(train, dataset)
    pipeline.install_budgets(train)
    pipeline.install_generation(train)
    pipeline.install_evidence(train, dataset, 'task-aware')
    shutil.copyfile(LOCK / 'contracts' / (dataset + '.md'), train / 'generator/v0/program_api.md')
    host = train / 'pm3_generator_harness.py'
    host.write_text(pipeline.replace_function(host.read_text(), 'medium_luna', inspect.getsource(
        configured_construction).replace('def configured_construction(', 'def medium_luna(', 1)))
    expected_hash = INITIAL_SPEC[dataset]
    if executor in ('27b', 'luna'):
        if executor == '27b':
            pipeline.executor_profile.install(train, dataset, plan)
        else:
            install_luna_hooks(train, dataset, plan, pipeline)
        if (dataset, executor) == ('pm3', 'luna'):
            (train / 'generator/v0/prompt.md').write_text(pipeline.GENERATION_PROMPT)
            expected_hash = 'e613446d5ec5db308173048fc5a7d61520290774d27fd15dbf0aa2c8e872e4f3'
        elif dataset != 'mempal':
            seed = train / 'generator/v0'
            shutil.copyfile(pipeline.SIMPLE_HISTORY_ADAPTER, seed / 'history_adapter.py')
            (seed / 'prompt.md').write_text(pipeline.GENERATION_PROMPT + '\n' + pipeline.SIMPLE_INITIALIZATION.replace('top_k=30', 'top_k=10'))
            expected_hash = {'pm3': '63fcf01f7f47b096ced16cf05aacc0e515eaff2faa4f59c79950ea9bf899f6c5',
                             'memorycd': '007721179285d2af147358d1f21ca43dd68a8fd4c4ebe3f4ddc3b8ab8ce81652'}[dataset]
    effort = 'low' if (dataset, executor) == ('memorycd', '9b') else 'medium'
    plan['phg']['generator'].update(model='gpt-6-sol', reasoning_effort=effort, default_effort=effort)
    plan['phg']['optimizer'].update(model='gpt-6-sol', reasoning_effort='medium')
    plan['phg']['limits'].update({key: pipeline.BUDGETS[key] for key in ('llm_calls', 'checks', 'completion_tokens')})
    plan['phg'].update(selection_policy='mean_gain', spec_v0_sha256=pipeline.spec_sha(train / 'generator/v0'))
    if plan['phg']['spec_v0_sha256'] != expected_hash:
        raise ValueError(f'{dataset}: initial generator differs from the paper')
    receipt = train / 'generator/v0.json'
    pipeline.shared.write(receipt, {**read(receipt), 'spec_sha256': expected_hash})
    plan['program_edit_scope'] = 'personal-section-v1'
    plan['selection_rule'] = pipeline.SELECTION_RULE
    if 'host_contract' in plan:
        plan['host_contract']['hook_steps'] = pipeline.BUDGETS['hook_steps']
    if 'task_guard' in plan:
        plan['task_guard'] = None
    credential = os.environ.get('HYPERHARNESS_SOL_ENV', Path.home() / '.config/personalized_harness/sol-swedencentral.env')
    for config in (plan['phg']['generator'], plan['optimizer']):
        config['credential_file'] = str(Path(credential).resolve())
    if 'model_cache' in plan:
        plan['model_cache'] = str(Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub')
    plan['public_transport'] = 'Direct provider; no historical multi-endpoint recovery gateway.'
    pipeline.shared.write(train / 'plan.json', plan)
    pipeline.shared.freeze(train)


def run_study(work, dataset):
    sys.path.insert(0, str(ROOT))
    from scripts import hyperharness_runtime as shared
    train, test = work / 'train', work / 'test'
    if test.exists():
        raise ValueError('Refusing to overwrite an existing TEST run')
    lock = read(LOCK / f'{dataset}.json')['panels']
    shared.verify(train)
    plan = read(train / 'plan.json')
    luna = plan['executor_model'] == 'gpt-5.6-luna'
    check_panel(plan['splits'], luna_train_panel(dataset, lock['train']) if luna else lock['train'])
    if dataset == 'pm3':
        from paper_pm3 import study
        study(work)
        return
    driver = f'{dataset}_phg.py'
    def invoke(stage, *args):
        subprocess.run([sys.executable, '-I', '-B', str(stage / driver), *map(str, args)], check=True)
    invoke(train, 'run', '--out', train)
    selected = read(train / 'selection.json')['version']
    if not shared.complete(train, f'phg_r{selected}', plan['test_users']):
        raise ValueError('TEST requires a complete selected TRAIN panel')
    invoke(train, 'register-test', '--out', test, '--reference', train, '--host-source', train,
           '--prepared', plan['prepared'])
    check_panel(read(test / 'plan.json')['splits'], lock['test'])
    if luna and dataset == 'memorycd':
        shared.write(test / 'inputs/filter_exclusions.json', {'protocol': plan['filter_successor']['protocol'], 'users': {}})
        shared.write(test / 'plan.json', {**read(test / 'plan.json'), 'filter_successor': plan['filter_successor'],
                     'targeted_optimization': plan['targeted_optimization']})
    shutil.copyfile(train / 'harness_permission.py', test / 'harness_permission.py')
    shared.freeze(test)
    invoke(test, 'run', '--out', test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('pm3', 'memorycd', 'mempal'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--run', action='store_true', help='Execute an already registered TRAIN-to-TEST study')
    parser.add_argument('--executor', choices=('9b', '27b', 'luna'), default='9b', help='Registration profile; execution uses the frozen plan')
    parser.add_argument('--work', required=True, type=Path, help='Fresh node-local directory')
    parser.add_argument('--catalog', type=Path, help='MemoryCD Amazon item title parquet')
    parser.add_argument('--evaluator', type=Path, help='Pinned PM3 official evaluator checkout')
    args = parser.parse_args()
    work = args.work.resolve()
    if args.run:
        run_study(work, args.dataset)
        return
    if not args.source:
        parser.error('Registration requires --source')
    source = args.source.resolve()
    panels = verify(args.dataset, source)
    if args.dataset == 'memorycd' and not args.catalog:
        parser.error('MemoryCD requires --catalog; see docs/public_reproduction.md')
    if args.dataset == 'pm3':
        if not args.evaluator or subprocess.check_output(['git', '-C', str(args.evaluator), 'rev-parse', 'HEAD'], text=True).strip() != 'fcbc03ec3514cca7db32e81c3a844d3875659404':
            parser.error('PM3 requires the pinned evaluator at fcbc03ec3514cca7db32e81c3a844d3875659404')
    work.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    if args.dataset == 'pm3':
        write(work / 'plan.json', {'roles': {uid: role for role, panel in panels.items() for uid in panel}})
        write(work / 'manifest.json', {'executor_base_url': 'http://127.0.0.1:8001/v1', 'executor_model': 'Qwen3.5-9B',
              'executor_thinking': False, 'executor_sampling_profile': 'qwen35-instruct-lowtemp-v1',
              'model_request_timeout_seconds': 600, 'model_connection_keepalive': True})
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_pm3_native_skillopt.py'), 'register',
                        '--out', str(work / 'native'), '--data-root', str(source), '--evaluator-root',
                        str(args.evaluator.resolve()), '--manifest-from', str(work), '--roles-from', str(work),
                        '--query-scope', 'later_half'], env=env, check=True)
        check_panel(read(work / 'native/plan.json')['splits'], {**panels['train'], **panels['test']})
        runner = str(ROOT / 'scripts/paper_pm3.py')
        subprocess.run([sys.executable, runner, 'register', '--work', str(work), '--executor', args.executor], env=env, check=True)
        print(shlex.join([sys.executable, runner, 'study', '--work', str(work)]))
        return
    prepared = work / 'prepared'
    if args.dataset == 'memorycd':
        registration = read(LOCK / 'memorycd.json')['source']
        staging = work / 'release'
        staging.mkdir()
        for name in registration['files']:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source / name)
        write(staging / 'source.json', {'repository': memorycd.DATASET_REPOSITORY, 'revision': memorycd.DATASET_REVISION,
              'files': {name: {'sha256': value, 'bytes': (source / name).stat().st_size}
                        for name, value in registration['files'].items()}})
        memorycd.prepare_dataset(staging, prepared, metadata_catalog=args.catalog.resolve())
        if digest(prepared / 'users.jsonl') != 'a53b67118bb5caa196cc79418617a435038bd548bb4339f6c1e1576fbe7a71a4':
            raise ValueError('MemoryCD native users or item metadata differ from the paper')
    else:
        mempal.prepare_dataset(source, prepared)
    host = work / 'host'
    host.mkdir()
    for name, original in HOST.items():
        shutil.copyfile(ROOT / 'scripts' / (original + '.py'), host / (name + '.py'))
    driver = args.dataset + '_phg.py'
    subprocess.run([sys.executable, str(ROOT / 'analysis/skill_optimization_diagnosis' / driver),
                    'register', '--prepared', str(prepared), '--host-source', str(host), '--out', str(work / 'train')],
                   env=env, check=True)
    check_panel(read(work / 'train/plan.json')['splits'], panels['train'])
    install_paper_profile(work / 'train', args.dataset, args.executor)
    print('Registered TRAIN. To start model execution:')
    print(shlex.join([sys.executable, str(Path(__file__).resolve()), args.dataset, '--run', '--work', str(work)]))


if __name__ == '__main__':
    main()