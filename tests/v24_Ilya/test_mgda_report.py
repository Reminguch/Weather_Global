"""Paired MGDA outcomes and report plumbing, without model inference."""
from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analyze_models import report_v24_mgda as reporting
from src.models.mamba.v24_Ilya import checkpoint


def record(loss=1., power=10., rmse=1., coherence=.8):
    return {'rows': [{'feature': feature, 'day': day, 'rmse': rmse,
                      'mean_relative_power_error_pct': power, 'mean_coherence': coherence}
                     for feature in reporting.evaluation.FEATURES for day in reporting.evaluation.DAYS],
            'result': {'original_graphcast_loss': {'full_rollout': loss, 'baseline_rollout': 2.}},
            'stacked': {'truth_power': np.ones((1, 1))}, 'sources': [], 'times': ['2023-01-15T00:00:00Z']}


@pytest.fixture(autouse=True)
def feature_rows(monkeypatch):
    monkeypatch.setattr(reporting.evaluation, 'feature_rows', lambda item, branch='full': item['rows'])


def interval(low=.8, high=.95):
    return {'status': 'complete', 'percentile95': {key: [low, high] for key in
             ('exact_gc_loss_ratio', 'mean_power_error_ratio', 'mean_rmse_ratio')}}


def test_distinguishes_initialization_improvement_from_matched_control():
    candidate = record(.8, 8.)
    comparisons = {'initialization': reporting.compare(candidate, record()),
                   'matched_control': reporting.compare(candidate, record(.7, 9.))}
    result = reporting.outcomes(comparisons, {'initialization': interval(), 'matched_control': interval(.9, 1.2)})
    assert result['both_improved_vs_initialization']
    assert not result['power_improved_gc_preserved_vs_matched_control']
    assert not result['qualified_point_estimate']
    assert result['supported_vs_initialization_95']
    assert not result['supported_vs_matched_control_95']


def test_no_mean_hides_a_single_field_harm_or_coherence_loss():
    candidate = record(.8, 8., .9, .7)
    candidate['rows'][0]['rmse'] = 1.021
    comparison = reporting.compare(candidate, record())
    assert comparison['mean_rmse_ratio'] < 1
    assert not comparison['safety_guard_passed']
    assert len(comparison['safety_guard_failures']) == 1
    assert comparison['mean_coherence_change'] == pytest.approx(-.1)
    candidate['rows'][0]['rmse'] = 1.02
    assert reporting.compare(candidate, record())['safety_guard_passed']


def test_zero_reference_error_is_explicit_and_not_a_fake_finite_percentage():
    comparison = reporting.compare(record(power=1.), record(power=0.))
    assert comparison['mean_power_error_ratio'] is None
    assert comparison['mean_power_error_difference_pct'] == 1.
    assert reporting.ratio(0, 0) == 1


@pytest.mark.parametrize('role,expected_stages', [('development', ['exact', 'spectral']), ('confirmation', ['spectral'])])
def test_report_uses_paired_checkpoint_and_never_selects(tmp_path, monkeypatch, role, expected_stages):
    tasks = [{'id': 'initial', 'label': 'Mature MSE'},
             {'id': 'mse_swa', 'label': 'MSE late SWA', 'paired_mse_id': 'mse_swa'},
             {'id': 'mgda_swa', 'label': 'MGDA late SWA', 'paired_mse_id': 'mse_swa'}]
    manifest = {'task_specs': tasks, 'canonical_mse_id': 'initial', 'evaluation_role': role,
                'bootstrap_grouping': [], 'selection_policy': reporting.evaluation.POLICY,
                'output_dir': str(tmp_path)}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(reporting.evaluation, 'load_manifest', lambda _: manifest)
    data = [record(), record(.9, 9.), record(.8, 8.)]
    calls = []

    def resolve(_manifest, index, stage):
        calls.append((index, stage))
        return tmp_path/str(index), {}, data[index]['result'], False

    monkeypatch.setattr(reporting.evaluation, 'resolve_result', resolve)
    monkeypatch.setattr(reporting.evaluation, 'frozen_file', lambda p: {'path': str(p), 'sha256': 'test'})
    monkeypatch.setattr(reporting.evaluation, 'load_spectral_result', lambda p, s, r: data[int(p.name)])
    monkeypatch.setattr(reporting.evaluation, 'bootstrap_joint', lambda *args: interval())
    monkeypatch.setattr(reporting.evaluation, 'rank_finalists', lambda *args: pytest.fail('MGDA report must not select'))
    report = reporting.build_report(path, image_dir=tmp_path/'images')
    assert [stage for index, stage in calls if index == 0] == expected_stages
    assert report['finalists'] == [] and not report['selection_allowed']
    mgda = report['records'][2]
    assert mgda['comparisons']['matched_control']['reference_id'] == 'mse_swa'
    assert mgda['comparisons']['matched_control']['exact_gc_loss_ratio'] == pytest.approx(.8/.9)
    assert mgda['outcomes']['qualified_point_estimate']
    figures = json.loads((tmp_path/'images/figures_manifest.json').read_text())['figures']
    assert {row['name'] for row in figures} == {
        'joint_rmse_power', 'exact_graphcast_loss', 'feature_physical_rmse', 'feature_power_error', 'feature_coherence'}
    assert all(Path(row['files']['png']).is_file() for row in figures)
    assert next(row for row in figures if row['name'] == 'feature_coherence')['reference'] == 'matched_control'
    tasks[2].pop('paired_mse_id')
    with pytest.raises(ValueError, match='Explicit matched MSE'):
        reporting.build_report(path)


def test_optimizer_plot_records_missing_diagnostics_instead_of_fabricating(tmp_path):
    from scripts.analyze_models import plot_v24_amse_regime as plotting
    metrics = tmp_path/'metrics.jsonl'
    metrics.write_text('\n'.join(json.dumps({'step': step, 'gc_loss': 1./step, 'power_loss': 2./step}) for step in (1, 2)))
    manifest = {'figures': []}
    reporting.optimizer_diagnostics({'MGDA': metrics}, tmp_path, manifest, plotting)
    assert 'mgda_alpha' in manifest['figures'][0]['missing_metrics']['MGDA']
    metrics.write_text('{"step": 1}\n{"step": 1}\n')
    with pytest.raises(ValueError, match='Duplicate'):
        reporting.optimizer_diagnostics({'MGDA': metrics}, tmp_path, {'figures': []}, plotting)


def test_swa_zero_policy_omits_recurrent_state_and_preserves_default(tmp_path, monkeypatch):
    inputs = [tmp_path/f'step{step}.pkl' for step in (1100, 1200)]
    values = {}
    for path, step, value in zip(inputs, (1100, 1200), (1., 3.), strict=True):
        path.write_bytes(pickle.dumps({'checkpoint_format': checkpoint.TRAINING_CHECKPOINT_FORMAT}))
        values[path] = SimpleNamespace(completed_step=step,
            residual_params={'module': {'w': np.full(2, value, np.float32)}},
            residual_state={'module': {'s': np.full(2, 7., np.float32)}},
            resolved_training_config={}, baseline_checkpoint_path='baseline',
            baseline_checkpoint_fingerprint='baseline-hash', anchor_manifest_fingerprint='anchors-hash')
    monkeypatch.setattr(checkpoint, 'load_v24_Ilya_training_checkpoint', lambda path: values[path])
    default = checkpoint.build_v24_Ilya_swa(inputs, [1100, 1200], tmp_path/'default.pkl')
    zero = checkpoint.build_v24_Ilya_swa(inputs, [1100, 1200], tmp_path/'zero.pkl', residual_state_policy='zero')
    assert default['residual_state'] is not None
    assert 'residual_state_policy' not in default
    assert zero['residual_state'] is None and zero['residual_state_policy'] == 'zero'
    np.testing.assert_array_equal(zero['residual_params']['module']['w'], [2., 2.])
    assert not checkpoint.load_v24_Ilya_checkpoint(tmp_path/'zero.pkl').has_residual_state
    with pytest.raises(ValueError, match='residual_state_policy'):
        checkpoint.build_v24_Ilya_swa(inputs, [1100, 1200], tmp_path/'bad.pkl', residual_state_policy='average')
