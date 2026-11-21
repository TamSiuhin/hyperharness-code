import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('learned_generator', ROOT / 'scripts/learned_generator.py')
learned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learned)


def test_released_generators_match_selected_checkpoints():
    found = learned.verify()
    assert set(found) == set(learned.INITIAL)
    assert {key: manifest['selected_round'] for key, manifest in found.items()} == {
        ('pm3', 'qwen35-9b'): 9, ('pm3', 'qwen38-27b'): 9, ('pm3', 'gpt-5.6-luna'): 5,
        ('mempal', 'qwen35-9b'): 1, ('mempal', 'qwen38-27b'): 2, ('mempal', 'gpt-5.6-luna'): 10,
        ('memorycd', 'qwen35-9b'): 6, ('memorycd', 'qwen38-27b'): 10, ('memorycd', 'gpt-5.6-luna'): 5}


def test_initial_hashes_match_paper_registration():
    source = (ROOT / 'scripts/paper_study.py').read_text()
    for (dataset, executor), digest in learned.INITIAL.items():
        assert digest in source, (dataset, executor)


def test_edited_generator_is_rejected(tmp_path):
    copy = tmp_path / 'learned'
    shutil.copytree(learned.LEARNED, copy)
    prompt = copy / 'mempal/qwen35-9b/generator/prompt.md'
    prompt.write_text(prompt.read_text() + '\n')
    with pytest.raises(ValueError, match='differ from the selected checkpoint'):
        learned.verify(copy)
