"""Check or locate the TRAIN-selected HyperHarness generators reported in the paper."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ROOT / 'learned'
INITIAL = {
    ('pm3', 'qwen35-9b'): '48a6ab4c8c5674924e7c87c99dfbafcf9a1f03219f3e67d09d7eb1e2aa9d464f',
    ('pm3', 'qwen38-27b'): '63fcf01f7f47b096ced16cf05aacc0e515eaff2faa4f59c79950ea9bf899f6c5',
    ('pm3', 'gpt-5.6-luna'): 'e613446d5ec5db308173048fc5a7d61520290774d27fd15dbf0aa2c8e872e4f3',
    ('mempal', 'qwen35-9b'): 'f88a0ab8e9800a1c23c9ac79f6690afd653bcf1995511e69b5831af62b7547d9',
    ('mempal', 'qwen38-27b'): 'f88a0ab8e9800a1c23c9ac79f6690afd653bcf1995511e69b5831af62b7547d9',
    ('mempal', 'gpt-5.6-luna'): 'f88a0ab8e9800a1c23c9ac79f6690afd653bcf1995511e69b5831af62b7547d9',
    ('memorycd', 'qwen35-9b'): 'f9195ab591e738334519a311e90f587a36ed7189af78d475aa00c60d710b8a52',
    ('memorycd', 'qwen38-27b'): '007721179285d2af147358d1f21ca43dd68a8fd4c4ebe3f4ddc3b8ab8ce81652',
    ('memorycd', 'gpt-5.6-luna'): '007721179285d2af147358d1f21ca43dd68a8fd4c4ebe3f4ddc3b8ab8ce81652',
}


def spec_sha(directory):
    # Same digest as the training pipeline's spec_sha.
    files = {str(path.relative_to(directory)): path.read_text(encoding='utf-8')
             for path in sorted(directory.rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def verify(root=LEARNED):
    found = {}
    for manifest_path in sorted(root.glob('*/*/manifest.json')):
        key = (manifest_path.parent.parent.name, manifest_path.parent.name)
        manifest = json.loads(manifest_path.read_text())
        if spec_sha(manifest_path.parent / 'generator') != manifest['spec_sha256']:
            raise ValueError(f'{key}: generator files differ from the selected checkpoint')
        if manifest['initial_spec_sha256'] != INITIAL.get(key):
            raise ValueError(f'{key}: initial generator differs from the paper registration')
        found[key] = manifest
    if set(found) != set(INITIAL):
        raise ValueError(f'Missing learned generators: {sorted(set(INITIAL) - set(found))}')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify')
    show = commands.add_parser('show')
    show.add_argument('dataset', choices=sorted({key[0] for key in INITIAL}))
    show.add_argument('executor', choices=sorted({key[1] for key in INITIAL}))
    args = parser.parse_args()
    found = verify()
    if args.command == 'verify':
        for (dataset, executor), manifest in sorted(found.items()):
            print(f"{dataset:9s} {executor:13s} r{manifest['selected_round']:<3d} {manifest['spec_sha256'][:12]}")
        return
    manifest = found[(args.dataset, args.executor)]
    json.dump({**manifest, 'generator': str(LEARNED / args.dataset / args.executor / 'generator')}, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
