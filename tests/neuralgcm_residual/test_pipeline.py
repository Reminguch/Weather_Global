import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import pytest

from src.models.neuralgcm_residual.io import read_json, write_json, sha256

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[2] /
                                             'scripts/experiments/launch_neuralgcm_2p8_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_submission_records_intent_and_never_duplicates(tmp_path, monkeypatch):
    plan = {'sbatch': ['sbatch'], 'script': '#!/bin/bash\ntrue\n'}
    def fake(*a, **kw):
        assert read_json(tmp_path / 'pipeline/jobs/example.json')['submission_intent']
        return SimpleNamespace(stdout='12345;della\n')
    monkeypatch.setattr(subprocess, 'run', fake)
    assert pipeline.once(tmp_path, 'example', lambda: plan) == '12345'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('duplicate submission'))
    assert pipeline.once(tmp_path, 'example', lambda: plan) == '12345'
    write_json(tmp_path / 'pipeline/jobs/ambiguous.json', {'submission_intent': 1})
    with pytest.raises(RuntimeError, match='ambiguous'):
        pipeline.once(tmp_path, 'ambiguous', lambda: plan)


def test_training_sizing_includes_full_budget_validation_and_resume_slices():
    origins = [str(t) for t in np.arange(np.datetime64('2015-01-02T00'), np.datetime64('2022-01-01T00'), np.timedelta64(6, 'h'))]
    cache = {'shards': [{'split': 'train', 'origins': origins}, {'split': 'val', 'origins': ['t'] * 1455}]}
    profile = dict(cached_update_seconds=[80, 90, 100], live20_seconds=[120, 130, 140],
                   validation20_seconds=dict(cold=20, warm=25), one_step_validation_seconds=0.1,
                   peak_cpu_rss_bytes=20 * 2**30)
    resources = pipeline.training_resources(profile, cache)
    assert resources['updates_per_arm'] == 8480
    assert resources['records_per_pass'] == 10176
    assert resources['pretrain_estimated_seconds'] > 8480 * 100
    assert resources['pretrain_hours'] == 24
    assert resources['memory_gb'] == 40


def test_resume_uses_highest_committed_valid_checkpoint(tmp_path):
    for update in (0, 200, 400):
        path = tmp_path / f'checkpoint_update_{update:08d}.pkl'
        path.write_bytes(str(update).encode())
        write_json(path.with_suffix('.json'), dict(stage='pretrain', cursor={'update': update}, sha256=sha256(path)))
    (tmp_path / 'checkpoint_update_00000400.pkl').write_bytes(b'corrupt')
    (tmp_path / 'checkpoint_update_00000600.pkl').write_bytes(b'uncommitted')
    update, path = pipeline.latest_checkpoint(tmp_path, 'pretrain')
    assert update == 200
    assert path.name == 'checkpoint_update_00000200.pkl'
