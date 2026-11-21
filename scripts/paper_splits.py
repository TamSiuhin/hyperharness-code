"""Fetch pinned public releases and verify the paper's original query panels."""
import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen
import shutil

from ohpu.benchmarks import memorycd, mempal


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'configs/paper-splits'
SIZES = {'pm3': ((25, 1860), (75, 5700)), 'mempal': ((50, 418), (50, 408)),
         'memorycd': ((100, 1200), (223, 2676))}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_files(source, registration, fetch):
    for relative, expected in registration['files'].items():
        path = source / relative
        if not path.exists() and fetch:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                scratch = Path(stream.name)
            try:
                with urlopen(registration['url'] + '/' + relative, timeout=180) as response, scratch.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
                if digest(scratch) != expected:
                    raise ValueError(f'Download checksum mismatch: {relative}')
                scratch.replace(path)
            finally:
                scratch.unlink(missing_ok=True)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or changed public data: {path}')


def verify(dataset, source, fetch=False):
    lock = read(LOCK / f'{dataset}.json')
    check_files(source, lock['source'], fetch)
    panels = lock['panels']
    users = set(panels['train']) | set(panels['test'])
    if set(panels['train']) & set(panels['test']):
        raise ValueError('TRAIN and TEST overlap')
    if dataset == 'mempal':
        raw = read(source / mempal.DATASET_FILE)
        roles = mempal.roles(['mempal_' + uid for uid in raw])
    elif dataset == 'memorycd':
        with gzip.open(source / 'users/cross_domain_users_sampled.jsonl.gz', 'rt') as stream:
            raw = {row['user_id']: row for row in map(json.loads, stream)}
        roles = memorycd.roles(list(raw))
    else:
        if users != {path.name for path in (source / 'backend').iterdir() if path.is_dir()}:
            raise ValueError('Unexpected PM3 user roster')
    for (role, panel), expected in zip(((role, panels[role]) for role in ('train', 'test')), SIZES[dataset]):
        if dataset != 'pm3' and set(roles[role]) != set(panel):
            raise ValueError(f'{role}: user IDs differ')
        for uid, entry in panel.items():
            queries = entry['evaluated']
            if dataset == 'pm3':
                rows = read(source / 'backend' / uid / 'test.json')
                wanted = set(queries)
                selected = [row for row in rows if row['query_id'] in wanted]
                by_id = {row['query_id']: row for row in selected}
                valid = (len(by_id) == len(selected) == len(queries) and
                         int(by_id[queries[0]]['ts']) == entry['construction_timestamp'])
            elif dataset == 'mempal':
                user = mempal.prepare_user(uid.removeprefix('mempal_'), raw[uid.removeprefix('mempal_')])
                valid = ([case['case_id'] for case in user['cases']] == queries and
                         user['construction_cutoff'] == entry['construction_cutoff'])
            else:
                interactions = raw[uid]['interactions']
                targets = sorted(interactions[memorycd.TARGET_DOMAIN], key=lambda row: row['timestamp'])[-3:]
                cutoff = targets[0]['timestamp']
                count = sum(row['timestamp'] < cutoff for domain in memorycd.DOMAINS
                            if domain != memorycd.TARGET_DOMAIN for row in interactions.get(domain, []))
                valid = (count == entry['construction_cutoff'] and queries == [
                    f'{uid}:{index}:{task}' for task in memorycd.TASKS for index in range(len(targets))])
            if not valid or len(queries) != len(set(queries)):
                raise ValueError(f'{dataset}/{role}/{uid}: query or cutoff mismatch')
        actual = len(panel), sum(len(entry['evaluated']) for entry in panel.values())
        if actual != expected:
            raise ValueError(f'{dataset}/{role}: {actual} != {expected}')
        print(f'{dataset}/{role}: {actual[0]} users, {actual[1]} queries')
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=SIZES)
    parser.add_argument('--source', required=True, type=Path, help='Directory containing the public release')
    parser.add_argument('--fetch', action='store_true', help='Download missing files; never repair mismatches')
    args = parser.parse_args()
    verify(args.dataset, args.source, args.fetch)


if __name__ == '__main__':
    main()