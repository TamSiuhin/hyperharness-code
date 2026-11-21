"""Maintainer-only: export public split metadata, never scores, histories or answers."""
import argparse
import json
from pathlib import Path

from paper_splits import SIZES, digest, memorycd, mempal, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=SIZES)
    parser.add_argument('stages', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    panels = {}
    for role in ('train', 'test'):
        plan = read(args.stages / role / 'plan.json')
        if plan['split'].lower() != role or set(plan['test_users']) != set(plan['splits']):
            raise ValueError(f'Unexpected registration: {role}')
        panels[role] = {uid: {key: entry[key] for key in ('evaluated', 'construction_cutoff',
                        'construction_timestamp') if key in entry} for uid, entry in plan['splits'].items()}
    if args.dataset == 'pm3':
        url = 'https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3/resolve/7b00a090b35b7293e6efeeb19494207f32b5a9ee'
        files = {path.relative_to(args.source).as_posix(): digest(path)
                 for path in sorted((args.source / 'backend').glob('*/*.json'))}
    elif args.dataset == 'mempal':
        url = 'https://raw.githubusercontent.com/hzp3517/Mem-PAL/' + mempal.DATASET_REVISION
        files = {mempal.DATASET_FILE: plan['source']['sha256']}
    else:
        url = f'https://huggingface.co/datasets/{memorycd.DATASET_REPOSITORY}/resolve/{memorycd.DATASET_REVISION}'
        files = {name: info['sha256'] for name, info in read(args.source / 'source.json')['files'].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump({'source': {'url': url, 'files': files}, 'panels': panels}, stream, indent=1, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()