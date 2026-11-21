"""Frozen Luna TRAIN-only selection hooks for the public benchmark runner."""
import hashlib
import json
import math
from pathlib import Path
import statistics


POLICY = 'targeted-balanced-train-v1'


TASKS = ('item_ranking', 'rating_prediction', 'review_summarization', 'review_generation')


COMMON_GUIDANCE = '''
## Registered targeted optimization experiment
The answer executor is gpt-5.6-luna with reasoning none, NOT Qwen. The construction
and outer optimization model is gpt-6-sol medium. The r0 generator is immutable.
Only TRAIN feedback may guide your changes; never read TEST artifacts or labels.
Propose one falsifiable mechanism per round. Explain the evidence, predicted gains,
likely regressions, and the request/history conditions in which it should activate.
Do not add generic personalized instructions indiscriminately. Inspect both harmed
and improved users, and distinguish retrieval failures from evidence-use failures.
The host requires a complete panel, positive equal-user paired improvement, and
improvement in at least three of five fixed balanced TRAIN user groups. This is
an adaptive stability heuristic, not an independent holdout or significance test.
This registered policy supersedes any earlier mean-only adoption instructions.
Keep model-call/token/time budgets, scoring and causal history boundaries unchanged.
Do not add a quality-feedback tool, hidden-task labels, or an extra runtime LLM.
'''


GUIDANCE = {
    'memorycd': '''
## MemoryCD: targeted task branches
Choose ONE of ranking, rating, title summarization or review generation per proposal.
Use the legal task_kind to isolate the behavior; preserve unrelated task branches.
Avoid broad history compression or generic guidance changes that affect all tasks.
Ranking: compare all candidates symmetrically; retrieve transferable attribute
preferences and negative evidence rather than treating every purchase as approval.
Rating: distinguish the user's rating scale from product-specific evidence; avoid
overconfident preference estimates when similar source-history evidence is sparse.
Title summarization: the supplied review body owns facts and sentiment. Past history
may inform concise style but must not inject different product facts or preferences.
Review generation: separate writing style from product facts. Ground preferences in
legal history and facts in the allowed current product/rating, not hidden reviews.
No TRAIN task may regress from r0 by more than one paired-user standard error on
common evaluable samples. Empty/missing tasks are not filled with zero.
''',
    'pm3': '''
## PM3: temporal evidence and conditional retrieval
Choose one mechanism per proposal: temporal conflict resolution, request-conditioned
retrieval, or evidence-conditioned personalization. Preserve all unrelated behavior.
Separate explicit current constraints from inferred preferences, obsolete statements,
retractions and one-off plans. Keep timestamps and source references with evidence.
Use only public request text and causally available user history to choose retrieval;
hidden evaluator task labels are unavailable and must never be inferred from IDs.
Prefer diverse, relevant evidence to a large generic user profile. Explicit facts
and current constraints outrank weak inferences. Without relevant evidence, leave
the answer neutral rather than inventing preferences. Do not copy stale facts into
new requests. Respect consent, correction and deletion semantics in the task API.
Inspect lifecycle pairs together and diagnose retrieval versus answer-use failures.
''',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)


def groups(users, seed):
    ordered = sorted(users, key=lambda user: hashlib.sha256(f'{seed}:{user}'.encode()).hexdigest())
    return [ordered[index::5] for index in range(5)]


def balanced_gain(scores, parent, users, seed):
    expected = set(users)
    if len(expected) < 5 or set(scores) != expected or any(not math.isfinite(value) for value in scores.values()):
        return False
    if parent is None:
        return True
    if set(parent) != expected or any(not math.isfinite(value) for value in parent.values()):
        return False
    deltas = {user: scores[user] - parent[user] for user in users}
    return statistics.fmean(deltas.values()) > 0 and sum(
        statistics.fmean(deltas[user] for user in group) > 0 for group in groups(users, seed)) >= 3


def task_gate(deltas, expected_users):
    result = {}
    for task in TASKS:
        values = deltas.get(task, [])
        if len(values) < max(2, math.ceil(0.8 * expected_users)) or any(not math.isfinite(value) for value in values):
            result[task] = {'passed': False, 'users': len(values), 'reason': 'insufficient_common_task_coverage'}
            continue
        mean = statistics.fmean(values)
        error = statistics.stdev(values) / math.sqrt(len(values))
        result[task] = {'passed': mean >= -error, 'users': len(values), 'mean_delta': mean, 'paired_standard_error': error}
    return {'passed': all(row['passed'] for row in result.values()), 'tasks': result}


def memorycd_task_gate(out, arm, users):
    deltas = {task: [] for task in TASKS}
    for user in users:
        candidate = out / 'scores' / arm / user
        reference = out / 'scores/phg_r0' / user
        if not (candidate / 'summary.json').exists():
            return {'passed': False, 'reason': 'incomplete_candidate'}
        with (candidate / 'records.jsonl').open() as stream:
            left = {row['query_id']: row for row in (json.loads(line) for line in stream if line.strip())}
        with (reference / 'records.jsonl').open() as stream:
            right = {row['query_id']: row for row in (json.loads(line) for line in stream if line.strip())}
        for task in TASKS:
            common = [query for query in left.keys() & right.keys() if left[query]['task_type'] == task
                and left[query].get('accuracy') is not None and right[query].get('accuracy') is not None
                and not left[query].get('filtered') and not right[query].get('filtered')]
            if common:
                deltas[task].append(statistics.fmean(left[query]['accuracy'] - right[query]['accuracy'] for query in common))
    return task_gate(deltas, len(users))


def install(host, dataset):
    original_framework = host.framework
    original_summary = host.round_summary
    original_optimizer = host.optimize

    def framework(plan, native):
        text = original_framework(plan, native)
        return text.replace('final Qwen call', 'final gpt-5.6-luna call').replace(
            'answer model is Qwen3.5-9B', 'answer model is gpt-5.6-luna') + COMMON_GUIDANCE + GUIDANCE[dataset]

    def round_summary(out, version):
        plan = read(out / 'plan.json')
        if plan['split'] != 'TRAIN':
            raise ValueError('Targeted adoption is TRAIN-only')
        guard = memorycd_task_gate(out, host.round_arm(plan, version), plan['test_users']) if dataset == 'memorycd' and version else {'passed': True}
        previous_accept = host.mean_gain_accept
        host.mean_gain_accept = lambda scores, parent, users: guard['passed'] and balanced_gain(scores, parent, users, plan['seed'])
        try:
            result = original_summary(out, version)
        finally:
            host.mean_gain_accept = previous_accept
        result['targeted_selection'] = {'policy': POLICY, 'groups': groups(plan['test_users'], plan['seed']), 'task_guard': guard}
        write(out / 'rounds' / f'round{version}.json', result)
        return result

    def optimize(out, version, **options):
        parent = out / 'rounds/parent.json'
        if not parent.exists() or read(parent).get('version') is None:
            raise ValueError('No complete TRAIN parent; do not consume an optimizer slot')
        return original_optimizer(out, version, **options)

    host.framework = framework
    host.round_summary = round_summary
    host.optimize = optimize


class GlobalNamespace:
    def __init__(self, values):
        object.__setattr__(self, '_values', values)

    def __getattr__(self, name):
        return self._values[name]

    def __setattr__(self, name, value):
        self._values[name] = value


def install_namespace(namespace, dataset):
    install(namespace['PHG'] if dataset == 'memorycd' else GlobalNamespace(namespace), dataset)
