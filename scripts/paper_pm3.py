"""Start the PM3 shared-generator protocol from freshly registered native inputs."""
import argparse
import os
import shutil
import sys
from pathlib import Path

from paper_splits import LOCK, ROOT, read

sys.path.insert(0, str(ROOT))
from scripts import hyperharness_runtime as shared


def register(work, executor='9b'):
    from scripts import run_pm3_native_skillopt as native
    from ohpu.experiments.pm3_official import uniform_programs
    drivers = ROOT / 'analysis/skill_optimization_diagnosis'
    sys.path.insert(0, str(drivers))
    import pm3_generator_harness as phg

    source, train = work / 'native', work / 'train'
    base = read(source / 'plan.json')
    lock = read(LOCK / 'pm3.json')['panels']
    for role in ('train', 'test'):
        if set(base[role + '_users']) != set(lock[role]) or any(
            any(base['splits'][uid][key] != value for key, value in entry.items())
            for uid, entry in lock[role].items()
        ):
            raise ValueError(f'Native registration differs from the paper {role} panel')
    train.mkdir()
    shutil.copytree(source / 'source_snapshot', train / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    for name in phg.DRV.HOST_FILES:
        shutil.copyfile(source / name, train / name)
    for name in (*phg.DRV.CODE_FILES, 'pm3_decision_harness.py', 'pm3_generator_harness.py', 'harness_permission.py'):
        shutil.copyfile(drivers / name, train / name)
    shutil.copyfile(source / 'manifest.json', train / 'manifest.json')
    (train / 'inputs').mkdir()
    (train / 'inputs/scaffold.py').write_text(native.PERSONALIZED_SCAFFOLD)
    users = base['train_users']
    baselines = ('no_history', 'bm25_10')
    programs = uniform_programs()
    for uid in users:
        query_ids = base['splits'][uid]['evaluated']
        private = read(source / 'private/users' / f'{uid}.json')
        rows = {row['query_id']: row for row in private['rows']}
        requests = read(source / 'private/requests' / f'{uid}.json')
        shared.write(train / 'private/users' / f'{uid}.json', {**private, 'rows': [rows[key] for key in query_ids]})
        shared.write(train / 'private/requests' / f'{uid}.json', {key: requests[key] for key in query_ids})
        for arm in baselines:
            shared.write(train / 'harness' / arm / uid / 'harness.json', {'source': programs[arm], 'generation': {'valid': True}})
    seed = train / 'generator/v0'
    phg.initial_spec(seed)
    phg.write(seed / 'generation_policy.json', {})
    phg.write(train / 'generator/v0.json', {'version': 0, 'parent': None, 'outcome': 'accepted', 'error': None,
              'spec_sha256': phg.spec_sha(seed)})
    methods = {f'phg_r{index}': index for index in range(11)}
    arms = [*baselines, *methods]
    plan = {key: base[key] for key in ('data_root', 'evaluator_root', 'judge_model', 'seed', 'optimizer_model')}
    plan['executor_model'] = read(source / 'manifest.json')['executor_model']
    credential = os.environ.get('HYPERHARNESS_SOL_ENV', phg.GENERATOR['credential_file'])
    plan.update(protocol=phg.PROTOCOL, split='TRAIN', dataset='pm3', two_stage=True, host_limit_scale=4,
                test_users=users, heldout_users=base['test_users'], splits={uid: base['splits'][uid] for uid in users},
                optimizer={'model': plan['optimizer_model'], 'credential_file': credential, 'api_version': '2024-12-01-preview'},
                active_arms=arms, arms=arms, arm_roles={arm: 'generated' if arm in methods else 'fixed' for arm in arms},
                answer_workers=6, judge_workers=4, score_workers=6, concurrency={'optimizer': 1, 'generation': 2},
                comparisons={}, external_baselines={'arms': [], 'files': {}},
                decision_model={'arms': [], 'calls_per_request': 0, 'host_limits': {'stress_request_chars': 30000,
                    'state_chars': 32000, 'request_chars': 32000, 'loop_steps': 32, 'hook_steps': 20000}},
                program_edit_scope=phg.PROGRAM_EDIT_SCOPE, performance={'memory_checkpoints': True},
                public_study={'native_test_reference': {'root': str(source), 'files': read(source / 'registration.json')['files']}},
                phg={'arms': methods, 'start_round': 0, 'rounds': 11, 'luna_workers': 2, 'open_generation_contract': True,
                     'require_model_generated_code': True, 'decision_model_enabled': False, 'decision_policy': 'disabled',
                     'generator': {**phg.GENERATOR, 'credential_file': credential}, 'limits': phg.LIMITS,
                     'optimizer': {**phg.OPTIMIZER, 'model': plan['optimizer_model'], 'reasoning_effort': 'medium',
                                   'sessions': 10, 'prompt': phg.OPEN_OPTIMIZER_PROMPT}, 'selection_policy': 'mean_gain',
                     'diagnostic_reference': 'bm25_10', 'spec_v0_sha256': phg.spec_sha(seed)}, upper_bounds={})
    plan['phg']['dev_users'] = phg.dev_users(train, plan)
    shared.write(train / 'plan.json', plan)
    from paper_study import install_paper_profile
    install_paper_profile(train, 'pm3', executor)


def study(work):
    train, test = work / 'train', work / 'test'
    shared.run_pm3(train, experiment_key='public_study')
    shared.register_pm3_test(train, test, experiment_key='public_study')
    from paper_study import check_panel
    check_panel(read(test / 'plan.json')['splits'], read(LOCK / 'pm3.json')['panels']['test'])
    shared.run_pm3(test, experiment_key='public_study')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'study'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--executor', choices=('9b', '27b', 'luna'), default='9b')
    args = parser.parse_args()
    if args.command == 'register':
        register(args.work.resolve(), args.executor)
    else:
        study(args.work.resolve())