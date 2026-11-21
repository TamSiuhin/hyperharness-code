import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper_splits import LOCK, SIZES, check_files, digest
from paper_study import check_panel, luna_train_panel


@pytest.mark.parametrize('dataset', SIZES)
def test_published_split(dataset):
    lock = json.loads((LOCK / (dataset + '.json')).read_text())
    panels = lock['panels']
    assert set(panels) == {'train', 'test'}
    assert not (set(panels['train']) & set(panels['test']))
    for role, expected in zip(('train', 'test'), SIZES[dataset]):
        panel = panels[role]
        assert (len(panel), sum(len(entry['evaluated']) for entry in panel.values())) == expected
        for entry in panel.values():
            assert len(set(entry['evaluated'])) == len(entry['evaluated'])
            assert set(entry) <= {'evaluated', 'construction_cutoff', 'construction_timestamp'}


def test_changed_release_is_not_overwritten(tmp_path):
    path = tmp_path / 'release'
    path.write_bytes(b'canonical')
    registration = {'files': {'release': digest(path)}}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed public data'):
        check_files(tmp_path, registration, fetch=True)
    assert path.read_bytes() == b'changed'


def test_reordered_or_reassigned_queries_fail():
    expected = {'user': {'construction_cutoff': 5, 'evaluated': ['first', 'second']}}
    for panel in ({'other': expected['user']}, {'user': {**expected['user'], 'evaluated': ['second', 'first']}},
                  {'user': {**expected['user'], 'construction_cutoff': 6}}):
        with pytest.raises(ValueError, match='registration differs'):
            check_panel(panel, expected)


def test_luna_masks_only_the_registered_training_query():
    panels = json.loads((LOCK / 'mempal.json').read_text())['panels']
    effective = luna_train_panel('mempal', panels['train'])
    assert set(effective) == set(panels['train'])
    assert sum(len(row['evaluated']) for row in effective.values()) == 417
    assert sum(len(row['evaluated']) for row in panels['train'].values()) == 418
    assert sum(len(row['evaluated']) for row in panels['test'].values()) == 408
    filtered = json.loads((LOCK / 'luna-memorycd-filter.json').read_text())['users']
    train = json.loads((LOCK / 'memorycd.json').read_text())['panels']['train']
    assert sum(map(len, filtered.values())) == 7
    assert all(set(ids) <= set(train[uid]['evaluated']) for uid, ids in filtered.items())