"""Production starts independently; continuation retains its own dependencies."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name, filename):
    path = Path(__file__).resolve().parents[2] / 'scripts/experiments' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


early = load('feedback_early', 'run_neuralgcm_feedback_early_start.py')
frozen = load('feedback_frozen', 'restart_neuralgcm_feedback.py')


@pytest.mark.parametrize('after', [(), ('123',)])
def test_submission_uses_operational_runner_and_only_requested_dependencies(tmp_path, after):
    policy = tmp_path / 'pipeline/early_start_v1/policy.json'
    policy.parent.mkdir(parents=True)
    policy.write_text('{}')
    backend = SimpleNamespace(once=lambda root, key, factory: (key, factory()))
    manifest = dict(source_root='/frozen/source', source_id='source', python='/env/python')
    key, plan = early.submit_one(tmp_path, manifest, frozen, backend, 'run', after=after)
    assert key == 'early-pretrain-run-000'
    assert plan['afterok'] == list(after)
    assert ('--dependency' in plan['sbatch']) == bool(after)
    if after:
        assert plan['sbatch'][plan['sbatch'].index('--dependency') + 1] == 'afterok:123'
    assert plan['sbatch'][plan['sbatch'].index('--qos') + 1] == 'gpu-short'
    assert '--partition' not in plan['sbatch']
    assert plan['script'].splitlines()[-1].startswith('exec /env/python -u ' + early.__file__)
    assert 'cd /frozen/source' in plan['script']
    assert 'restart_neuralgcm_feedback.py' not in plan['script']


def test_authorization_binds_source_runner_and_independent_smoke(tmp_path):
    pilot = tmp_path / 'pilot/outputs/report.json'
    pilot.parent.mkdir(parents=True)
    pilot.write_text('{"passed": true}')
    path = tmp_path / 'pipeline/early_start_v1/policy.json'
    path.parent.mkdir(parents=True)
    manifest = dict(source_id='source', pilot_root=str(tmp_path / 'pilot'))
    value = dict(source_id='source', experiment_root=str(tmp_path),
        runner=dict(path=str(Path(early.__file__).resolve()), sha256=early.sha(early.__file__)),
        wait_for_full_smokes=False, authorization_text='user requested independent production',
        independent_smoke_sha256=early.sha(pilot))
    path.write_text(json.dumps(value))
    assert early.policy(tmp_path, manifest) == value
    with pytest.raises(ValueError, match='authorization identity'):
        early.policy(tmp_path, dict(manifest, source_id='different'))
    pilot.write_text('{"passed": false}')
    with pytest.raises(ValueError, match='Independent smoke evidence changed'):
        early.policy(tmp_path, manifest)
