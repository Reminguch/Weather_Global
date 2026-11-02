"""The twenty-year replacement reuses controls and keeps dataset bounds explicit."""
import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import fresh_data_scale_cache as aligned
from scripts.experiments import fresh_data_scale_evaluation as evaluation
from scripts.experiments import fresh_data_scale_training as training
from scripts.experiments import run_v24_fresh_data_scale as campaign
from src.models.mamba.v24_Ilya.training.cached_checkpoint import digest_json, scientific_config


def twenty_manifest(tmp_path):
    value = dict(root=str(tmp_path), workspace=str(tmp_path), source=str(tmp_path / 'source'),
                 workflow_id='twenty', wait_hours=143, cache_shards=20,
                 new_arms=['w128_20y', 'w512_20y'], reuse_qualification=True,
                 arms={f'w{w}_{y}y': dict(width=w, years=y, reused=y == 7)
                       for w in (128, 512) for y in (7, 20)})
    campaign.write(tmp_path / 'state.json', dict(automation_enabled=True, jobs={}))
    return value


def test_only_new_twenty_year_arms_start_without_redundant_gpu_qualification(tmp_path):
    m = twenty_manifest(tmp_path)
    jobs = {key: (spec, deps) for key, spec, deps in campaign.initial_jobs(m)}
    assert len(jobs) == 6
    assert not any('7y' in key or '40y' in key or 'bridge' in key for key in jobs)
    assert jobs['cache_generate'][0]['array'] == 20
    assert all(spec['gpu'] is None for spec, _ in jobs.values())
    for arm in m['new_arms']:
        spec, dependencies = jobs[f'qualify_{arm}']
        assert spec['stage'] == 'startup' and dependencies == ['cache_verify']


def test_wait_can_depend_on_producer_gate_and_only_two_new_finals_are_required(tmp_path, monkeypatch):
    m = twenty_manifest(tmp_path)
    planned = campaign.submit(m, dry=True, producer_dependency='1234')['initial_jobs']
    assert planned[0][0] == 'prepared_wait' and planned[0][2] == ['1234']
    with pytest.raises(ValueError, match='numeric'):
        campaign.submit(m, dry=True, producer_dependency='1234:afterany:5678')
    campaign.write(tmp_path / 'state.json', {'jobs': {
        'eval2022_w128_20y': {'job_id': '100'}, 'eval2022_w512_20y': {'job_id': '101'}}})
    calls = []
    monkeypatch.setattr(campaign, 'submit_job', lambda *args: calls.append(args))
    campaign.maybe_freeze_finals(m)
    assert calls[0][3] == ['100', '101']


def test_twenty_year_alignment_preserves_all_original_windows():
    clock = lambda start: np.arange(np.datetime64(f'{start}-01-01T00'), np.datetime64('2023-01-01T00'),
                                   np.timedelta64(6, 'h'))
    full, old = clock(2002), clock(2015)
    raw = np.arange(2, len(old) - 2)
    train = np.flatnonzero(old[raw] < np.datetime64('2022-01-01'))
    val = np.flatnonzero(old[raw] >= np.datetime64('2022-01-01'))
    original = dict(anchor_indices=raw, anchor_times=old[raw], split_train=train, split_val=val)
    arrays, proof = aligned.aligned_anchor_arrays(full, old, original)
    assert proof['historical_segments'] == 197
    assert proof['historical_anchors'] == 197 * 96
    assert proof['original_train_anchors'] == len(train)
    for split in ('split_train', 'split_val'):
        positions = arrays[split][proof['historical_anchors']:] if split == 'split_train' else arrays[split]
        np.testing.assert_array_equal(full[arrays['anchor_indices'][positions]], old[raw[original[split]]])


def evidence_fixture():
    previous = dict(data=dict(prepared_root='/old', anchor_manifest_root='/oldanchors'),
                    optimizer=dict(learning_rate=1e-4, learning_rate_schedule='constant'),
                    architecture=dict(residual_width=512))
    current = copy.deepcopy(previous)
    current['data'].update(prepared_root='/new20', anchor_manifest_root='/aligned20')
    provenance = dict(scientific_config_sha256=digest_json(scientific_config(previous)),
                      shared_init_sha256='oldinit', environment={'jax': 'same'},
                      code_sha256={'src/models/mamba/v24_Ilya/training/cached_runner.py': 'same'})
    evidence = dict(kind='cached_integrity', numerical_policy='advisory', integrity_passed=True,
                    checks={'serialization': True}, provenance=copy.deepcopy(provenance),
                    numerical_warnings=['Finite restart drift'])
    return evidence, previous, current, provenance


def test_reusing_qualification_accepts_new_dataset_and_retains_advisory_finite_drift():
    evidence, previous, current, provenance = evidence_fixture()
    evidence['passed'] = False  # An old strict numerical diagnostic is not integrity.
    assert all(training.reusable_evidence_checks(evidence, previous, current, 'oldinit', provenance).values())


@pytest.mark.parametrize('damage', ['learning_rate', 'init', 'code', 'environment', 'integrity'])
def test_reuse_rejects_changed_scientific_settings_or_corrupt_evidence(damage):
    evidence, previous, current, provenance = evidence_fixture()
    oldinit = 'oldinit'
    if damage == 'learning_rate':
        current['optimizer']['learning_rate'] *= 2
    elif damage == 'init':
        oldinit = 'different'
    elif damage == 'code':
        provenance['code_sha256'][next(iter(provenance['code_sha256']))] = 'changed'
    elif damage == 'environment':
        provenance['environment']['jax'] = 'changed'
    else:
        evidence['checks']['serialization'] = False
    with pytest.raises(ValueError, match='cannot be reused'):
        training.reusable_evidence_checks(evidence, previous, current, oldinit, provenance)


def test_comparison_uses_twenty_year_names_and_keeps_finite_worse_results(tmp_path):
    dates = [str(i) for i in range(122)]
    baseline = np.ones((122, 40))
    identity = dict(baseline={}, statistics=[], data_metadata={}, initializations={})
    reports = {}
    keys = {(w, y): f'w{w}_{y}y' for w in (128, 512) for y in (7, 20)}
    for (width, years), name in keys.items():
        reports[name] = dict(status='complete', initialization_times=dates,
                             baseline_loss_matrix=baseline.tolist(),
                             full_loss_matrix=(baseline * (0.9 if years == 7 else 1.1)).tolist(),
                             identity={**identity, 'checkpoint': {'path': name, 'sha256': name}})
    result = evaluation.comparison_report(reports, tmp_path, arm_keys=keys, bootstrap_samples=100)
    assert result['training_year_counts'] == [7, 20]
    assert result['primary_data_gain']['128']['gain_percentage_points'] == pytest.approx(-20)
    assert result['status'] == 'complete'
    text = (tmp_path / 'README.md').read_text()
    assert '20 versus seven' in text and '40 training years' not in text
