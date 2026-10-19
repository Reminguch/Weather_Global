"""A native MGDA roundtrip must preserve complete state and provenance."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments import run_v24_mgda as campaign
from src.models.mamba.v24_Ilya import checkpoint


def native():
    return SimpleNamespace(
        residual_params={'layer': {'w': np.array([0., 1.], np.float32)}},
        optimizer_state={'count': np.array(100, np.int32), 'v': np.array([1., 2.], np.float32)},
        residual_state={'layer': {'s': np.array([2., 3.], np.float32)}},
        rng_key=np.array([1, 2], np.uint32), completed_step=100,
        training_cursor={'segment': 3, 'offset': 4}, resolved_training_config={'seed': 22},
        baseline_checkpoint_path='baseline.npz', baseline_checkpoint_fingerprint='basehash',
        anchor_manifest_fingerprint='anchorhash', parameter_overlay_metadata={'copied': 4})


def install(monkeypatch, left, right):
    monkeypatch.setattr(checkpoint, 'load_v24_Ilya_training_checkpoint',
                        lambda path: left if Path(path).name == 'left.pkl' else right)


def test_identical_native_roundtrip(monkeypatch):
    left = native()
    install(monkeypatch, left, copy.deepcopy(left))
    campaign.compare_native('left.pkl', 'right.pkl')


@pytest.mark.parametrize('field', ['residual_params', 'optimizer_state', 'residual_state', 'rng_key'])
def test_native_roundtrip_rejects_changed_numerical_state(monkeypatch, field):
    left, right = native(), native()
    if field == 'rng_key':
        right.rng_key[0] += 1
    elif field == 'optimizer_state':
        right.optimizer_state['v'][0] += 1
    else:
        next(iter(getattr(right, field)['layer'].values()))[0] += 1
    install(monkeypatch, left, right)
    with pytest.raises((ValueError, AssertionError)):
        campaign.compare_native('left.pkl', 'right.pkl')


@pytest.mark.parametrize('field,new_value', [
    ('completed_step', 99), ('training_cursor', {'segment': 4, 'offset': 4}),
    ('resolved_training_config', {'seed': 23}),
    ('baseline_checkpoint_path', 'other.npz'), ('baseline_checkpoint_fingerprint', 'otherhash'),
    ('anchor_manifest_fingerprint', 'otheranchors'), ('parameter_overlay_metadata', {'copied': 5}),
])
def test_native_roundtrip_rejects_changed_provenance(monkeypatch, field, new_value):
    left, right = native(), native()
    setattr(right, field, new_value)
    install(monkeypatch, left, right)
    with pytest.raises((ValueError, AssertionError)):
        campaign.compare_native('left.pkl', 'right.pkl')


@pytest.mark.parametrize('change', ['signed_zero', 'dtype', 'nonfinite'])
def test_native_roundtrip_requires_bitwise_finite_arrays(monkeypatch, change):
    left, right = native(), native()
    if change == 'signed_zero':
        right.residual_params['layer']['w'][0] = -0.
    elif change == 'dtype':
        right.rng_key = right.rng_key.astype(np.uint64)
    else:
        left.residual_state['layer']['s'][0] = np.nan
        right.residual_state['layer']['s'][0] = np.nan
    install(monkeypatch, left, right)
    with pytest.raises((ValueError, AssertionError)):
        campaign.compare_native('left.pkl', 'right.pkl')


def test_advisory_replay_records_float_drift_without_losing_integrity(monkeypatch):
    left, right = native(), native()
    right.residual_params['layer']['w'][1] += .01
    right.optimizer_state['v'][0] += .02
    right.residual_state['layer']['s'][1] += .03
    install(monkeypatch, left, right)
    result = campaign.compare_native('left.pkl', 'right.pkl', policy='advisory')
    assert result['integrity_passed'] and result['status'] == 'complete_with_warnings'
    assert result['measurements']['residual_state']['relative_l2'] > 0


@pytest.mark.parametrize('change', ['rng', 'count', 'nonfinite', 'shape', 'cursor'])
def test_advisory_replay_still_rejects_integrity_errors(monkeypatch, change):
    left, right = native(), native()
    if change == 'rng':
        right.rng_key[0] += 1
    elif change == 'count':
        right.optimizer_state['count'] += 1
    elif change == 'nonfinite':
        right.residual_state['layer']['s'][0] = np.nan
    elif change == 'shape':
        right.residual_params['layer']['w'] = np.zeros(3, np.float32)
    else:
        right.training_cursor['segment'] += 1
    install(monkeypatch, left, right)
    with pytest.raises(ValueError):
        campaign.compare_native('left.pkl', 'right.pkl', policy='advisory')


def proof_fixture(tmp_path, *, smoke=True):
    manifest = {'record_hash': 'manifesthash', 'source_digest': 'sourcehash'}
    paths = {
        'config': tmp_path/'configs'/f'mgda{"_smoke" if smoke else ""}.json',
        'resolved': tmp_path/'resolved.json',
        'checkpoint': tmp_path/('smoke' if smoke else 'runs')/'mgda/checkpoints'/
                      ('checkpoint_step00000100.pkl' if smoke else 'checkpoint_step00002000.pkl'),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    proof = {'passed': True, 'arm': 'mgda', 'completed_steps': 100 if smoke else 2000,
             'manifest_hash': manifest['record_hash'], 'source_digest': manifest['source_digest'],
             'resume_checked': smoke, **{key: campaign.pinned(path) for key, path in paths.items()}}
    return manifest, proof, tmp_path/'checks'/f'mgda_{"smoke" if smoke else "train"}.json'


@pytest.mark.parametrize('smoke', [False, True])
def test_completion_proof_binds_all_expected_artifacts(tmp_path, smoke):
    manifest, proof, path = proof_fixture(tmp_path, smoke=smoke)
    campaign.seal(path, proof)
    actual = campaign.validate_completed_proof(tmp_path, manifest, 'mgda', smoke=smoke)
    assert actual['checkpoint'] == proof['checkpoint']


def test_advisory_completion_requires_integrity_but_accepts_warning(tmp_path):
    manifest, proof, path = proof_fixture(tmp_path)
    manifest['numerical_policy'] = 'advisory'
    proof.update(integrity_passed=True, serialization_checked=True, status='complete_with_warnings')
    campaign.seal(path, proof)
    assert campaign.validate_completed_proof(tmp_path, manifest, 'mgda', smoke=True)['passed']
    for key in ('integrity_passed', 'serialization_checked'):
        path.unlink()
        campaign.seal(path, {**proof, key: False})
        with pytest.raises(ValueError, match='serialization/integrity'):
            campaign.validate_completed_proof(tmp_path, manifest, 'mgda', smoke=True)


@pytest.mark.parametrize('key,new_value', [
    ('passed', False), ('arm', 'mse'), ('completed_steps', 99),
    ('manifest_hash', 'othermanifest'), ('source_digest', 'othersource'), ('resume_checked', False),
])
def test_completion_proof_rejects_other_task_even_if_hash_valid(tmp_path, key, new_value):
    manifest, proof, path = proof_fixture(tmp_path)
    proof[key] = new_value
    campaign.seal(path, proof)
    with pytest.raises(ValueError, match='proof'):
        campaign.validate_completed_proof(tmp_path, manifest, 'mgda', smoke=True)


@pytest.mark.parametrize('key', ['config', 'resolved', 'checkpoint'])
@pytest.mark.parametrize('change', ['content', 'path'])
def test_completion_proof_rejects_changed_or_substituted_artifacts(tmp_path, key, change):
    manifest, proof, path = proof_fixture(tmp_path)
    if change == 'path':
        replacement = tmp_path/'other'
        replacement.write_text('original')
        proof[key] = campaign.pinned(replacement)
    campaign.seal(path, proof)
    if change == 'content':
        Path(proof[key]['path']).write_text('changed')
    with pytest.raises(ValueError):
        campaign.validate_completed_proof(tmp_path, manifest, 'mgda', smoke=True)
