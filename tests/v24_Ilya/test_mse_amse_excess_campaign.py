"""Scientific identity, actual-loss evidence and bounded pilot submission DAG."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import run_v24_mse_amse_excess_pilot as campaign


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def manifest(tmp_path):
    return {'workspace': str(tmp_path), 'source_root': str(tmp_path / 'source'),
        'source_digest': 'source', 'record_hash': 'manifest',
        'initialization': {'checkpoint': {'path': '/init/parent.pkl'},
            'source_run_dir': '/init/parent', 'architecture': {'width': 512, 'temporal_d_inner': 16}},
        'resources': {'qualification': {'memory': 320, 'minutes': 240}, 'historical_peak_rss_gib': 234.71}}


def test_all_arms_parse_same_pair_rms_and_constant_cap(tmp_path):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    template = campaign.read(campaign.ROOT / campaign.INITIAL_ROOT / 'configs/di16_joint.json')
    original = copy.deepcopy(template)
    comparable = []
    for arm in campaign.ARMS:
        value = campaign.make_config(template, tmp_path, arm)
        config = load_training_config(put(tmp_path / f'{arm}.json', value))
        assert config.optimizer_kind == 'mgda_rms' and config.mgda_direction == arm
        assert config.objective_pair == 'mse_amse_excess'
        assert config.loss_kind == 'amse' and config.amse_lambda == 1 and config.amse_lmax == 179
        assert config.max_steps == 200 and config.checkpoint_every == 50
        assert config.ar_tail_k == 20
        assert config.learning_rate == config.end_learning_rate == config.warmup_start_learning_rate == 1e-6
        assert config.feedback_mode == 'closed_loop_sg' and config.bptt_steps == 24 and config.segment_steps == 96
        assert config.distributed.per_device_batch_size == 4
        assert config.power_calibration is None and config.amse_split_calibration is None
        value.pop('output'); value['optimizer'].pop('mgda_direction')
        comparable.append(value)
    assert comparable[0] == comparable[1] == comparable[2]
    assert template == original


def test_qualification_changes_only_operational_output(tmp_path):
    template = campaign.read(campaign.ROOT / campaign.INITIAL_ROOT / 'configs/di16_joint.json')
    production = campaign.make_config(template, tmp_path, 'mgda')
    qualification = campaign.make_config(template, tmp_path, 'mgda', qualification=True)
    assert production.pop('output') != qualification.pop('output')
    assert production == qualification


def test_exactly_parent_and_three_native_final_tasks(tmp_path):
    tasks = campaign.tasks_for(manifest(tmp_path), tmp_path)
    assert [row['id'] for row in tasks] == ['parent', 'mgda_step00200', 'mse_control_step00200', 'fixed_sum_step00200']
    assert tasks[0]['role'] == 'mse_control'
    for row in tasks[1:]:
        assert row['step'] == 200 and row['endpoint'] == 'primary'
        assert row['paired_initialization_id'] == 'parent'
        assert row['checkpoint'].endswith('/checkpoints/checkpoint_step00000200.pkl')
        assert 'swa' not in row['checkpoint']


def test_training_always_fresh_starts_same_weights_and_zero_memory(tmp_path):
    for arm in campaign.ARMS:
        args = list(map(str, campaign.training_command(tmp_path, manifest(tmp_path), arm)))
        assert args[-3:] == ['--init-from', '/init/parent.pkl', '--zero-state-on-init']
        assert '--resume' not in args and '--stop-after-steps' not in args


def test_measured_budget_and_hard_limits():
    qualified = {'passed': True, 'batches': 4, 'peak_rss_gib': 240., 'steady_p90_seconds': 180.}
    assert campaign.resource_budget(qualified) == {'memory': 336, 'minutes': 870}
    for change in ({'passed': False}, {'batches': 3}, {'peak_rss_gib': float('nan')},
                   {'peak_rss_gib': 400.}, {'steady_p90_seconds': 700.}):
        with pytest.raises(ValueError):
            campaign.resource_budget({**qualified, **change})


def test_sbatch_frozen_gpu_and_unthrottled_arrays(tmp_path):
    args = campaign.sbatch(tmp_path, manifest(tmp_path), 'evaluate-exact', memory=16,
                           minutes=62, gpu=True, array='0-3', dependency='100:101')
    assert '--array=0-3' in args and '--dependency=afterok:100:101' in args
    assert '--kill-on-invalid-dep=yes' in args and '--constraint=gpu80' in args
    assert '--qos=gpu-short' in args and '--time=62' in args
    assert args[-3:] == [str(tmp_path / 'source' / campaign.WRAPPER), 'evaluate-exact', str(tmp_path)]
    for change in ({'array': '0-3%2'}, {'minutes': 2881}, {'memory': 513}):
        with pytest.raises(ValueError):
            campaign.sbatch(tmp_path, manifest(tmp_path), 'train', **{'memory': 320, 'minutes': 1000, **change})


def test_dry_run_never_submits_or_mutates(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: manifest(tmp_path))
    monkeypatch.setattr(campaign, 'submit_once', lambda *a, **k: pytest.fail('dry run submitted'))
    before = sorted(tmp_path.rglob('*'))
    result = campaign.dry_run(tmp_path)
    assert sorted(tmp_path.rglob('*')) == before
    nodes = {row['id']: row for row in result['nodes']}
    assert nodes['release']['depends_on'] == ['qualify']
    assert nodes['train']['array'] == '0-2' and nodes['train']['updates_per_arm'] == 200
    assert nodes['evaluate_exact']['depends_on'] == nodes['evaluate_spectral']['depends_on'] == ['train']
    assert nodes['report']['depends_on'] == ['evaluate_exact', 'evaluate_spectral']
    assert not result['submitted'] and not result['automatic_extension']
    assert not any('2000' in key or 'di32' in key for key in nodes)


def test_submit_only_qualification_and_afterok_release(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: manifest(tmp_path))
    calls = []
    def submit(_root, name, command):
        calls.append((name, command))
        return str(100 + len(calls))
    monkeypatch.setattr(campaign, 'submit_once', submit)
    campaign.submit(tmp_path)
    assert [name for name, _ in calls] == ['qualify', 'release']
    assert '--dependency=afterok:101' in calls[1][1]


def test_failed_qualification_cannot_release_any_job(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: manifest(tmp_path))
    monkeypatch.setattr(campaign, 'qualification_proof', lambda *a: (_ for _ in ()).throw(ValueError('not qualified')))
    monkeypatch.setattr(campaign, 'submit_once', lambda *a: pytest.fail('failed qualification released jobs'))
    with pytest.raises(ValueError, match='not qualified'):
        campaign.release(tmp_path)


def test_release_dag_waits_for_all_arms_and_evaluation_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: manifest(tmp_path))
    monkeypatch.setattr(campaign, 'qualification_proof', lambda *a: {
        'passed': True, 'batches': 4, 'peak_rss_gib': 235., 'steady_p90_seconds': 100.})
    put(tmp_path / 'checks/qualification.json', {'proof': 'pinned'})
    calls = []
    def submit(_root, name, command):
        calls.append((name, command))
        return str(100 + len(calls))
    monkeypatch.setattr(campaign, 'submit_once', submit)
    monkeypatch.setenv('SLURM_JOB_ID', '99')
    campaign.release(tmp_path)
    values = dict(calls)
    assert '--dependency=afterok:99' in values['train'] and '--array=0-2' in values['train']
    assert '--dependency=afterok:101' in values['evaluate_exact']
    assert '--dependency=afterok:101' in values['evaluate_spectral']
    assert '--dependency=afterok:102:103' in values['report']


def accepted(arm='mgda'):
    return {'objective_pair': 'mse_amse_excess', 'mgda_direction': arm,
        'finite': True, 'descent_valid': True, 'backtracking_accepted': True,
        'gc_loss': 3., 'amse_excess_loss': 1., 'update_norm': .001, 'learning_rate': 1e-6,
        'objective_1_before': 3., 'objective_2_before': 1.,
        'objective_1_after': 2.99, 'objective_2_after': .99,
        'acceptance_threshold_1': 1e-5, 'acceptance_threshold_2': 1e-5,
        'repeatability_span_1': 1e-7, 'repeatability_span_2': 1e-7,
        'backtracking_trials': 2, 'backtracking_halvings': 1, 'backtracking_baseline_repeats': 3}


@pytest.mark.parametrize('arm', campaign.ARMS)
def test_actual_loss_acceptance_proof_for_each_arm(arm):
    campaign.validate_update(accepted(arm), arm)
    worse_second = {**accepted(arm), 'objective_2_after': 1.001}
    if arm == 'mgda':
        with pytest.raises(ValueError, match='criterion'):
            campaign.validate_update(worse_second, arm)
    else:
        campaign.validate_update(worse_second, arm)
    if arm == 'fixed_sum':
        campaign.validate_update({**accepted(arm), 'objective_1_after': 3.001}, arm)


@pytest.mark.parametrize('changes', [
    {'objective_pair': 'amse_split'}, {'backtracking_accepted': False}, {'finite': False},
    {'learning_rate': 1.1e-6}, {'update_norm': 0.}, {'gc_loss': float('nan')},
    {'objective_1_after': 3.}, {'objective_2_after': 1.},
    {'backtracking_trials': 10, 'backtracking_halvings': 9}, {'backtracking_baseline_repeats': 2},
])
def test_rejects_fake_success_nonfinite_or_subnoise_updates(changes):
    with pytest.raises(ValueError):
        campaign.validate_update({**accepted(), **changes}, 'mgda')


def acceptance_history():
    from src.models.mamba.v24_Ilya.training.mgda_acceptance import (
        backtrack_losses, calibrate_loss_tolerances)
    tolerances = calibrate_loss_tolerances([[3., 1.]] * 3)
    first_lr = float(np.float32(1e-6))
    result = backtrack_losses([3., 1.],
        lambda lr: [3.01, 1.01] if lr == first_lr else [2.99, .99],
        mode='mgda', tolerances=tolerances, initial_learning_rate=first_lr)
    row = {**accepted(), 'learning_rate': result.learning_rate,
        'acceptance_threshold_1': result.thresholds[0],
        'acceptance_threshold_2': result.thresholds[1]}
    return {'acceptance': result.to_dict(), 'repeatability': tolerances.to_dict()}, row


def test_candidate_history_accepts_real_fp32_schedule_and_pins_each_trial():
    history, row = acceptance_history()
    assert history['acceptance']['trials'][0]['learning_rate'] != 1e-6
    campaign.validate_update(row, 'mgda')
    campaign.validate_acceptance_history(history, row)


@pytest.mark.parametrize('tamper', ['before', 'losses', 'accepted_twice', 'schedule', 'repeats', 'final_lr'])
def test_candidate_history_rejects_inconsistent_saved_evidence(tamper):
    history, row = acceptance_history()
    value = history['acceptance']
    if tamper == 'before':
        value['before_losses'][0] += .1
    elif tamper == 'losses':
        value['trials'][-1]['losses'][0] += .1
    elif tamper == 'accepted_twice':
        value['trials'][0]['accepted'] = True
    elif tamper == 'schedule':
        value['trials'][0]['learning_rate'] = 1e-5
    elif tamper == 'repeats':
        history['repeatability']['repeats'] = 2
    else:
        value['learning_rate'] *= .5
    with pytest.raises(ValueError):
        campaign.validate_acceptance_history(history, row)


def test_both_protocols_use_identical_eight_seasonal2022_starts(tmp_path, monkeypatch):
    import xarray as xr
    from src.data_operations.loaders import zarr_metadata
    times = np.arange(np.datetime64('2022-01-01'), np.datetime64('2023-01-01'), np.timedelta64(6, 'h'))
    monkeypatch.setattr(xr, 'open_zarr', lambda *a, **k: xr.Dataset(coords={'time': times}))
    metadata_path = tmp_path / 'data/.zmetadata'
    metadata_path.parent.mkdir(); metadata_path.write_text('test')
    metadata = campaign.pinned(metadata_path)
    monkeypatch.setattr(zarr_metadata, 'verify_frozen_zarr_metadata', lambda *a: metadata)
    prototype = {'data_path': str(tmp_path / 'data'), 'data_metadata': metadata,
        'evaluation_year': 2022, 'n_samples': 32, 'warmup_steps': 24, 'target_steps': 40,
        'chosen_idx': [1] * 32, 'forecast': {'eval_mode': 'cold_full'},
        'initialization_times': ['2023-01-15T00:00:00Z']}
    for stage in ('exact', 'spectral'):
        put(tmp_path / campaign.REFERENCE / 'eval' / stage / 'mse_fp32/swa_step00500-02000/specification.json',
            {**prototype, 'stage': stage})
    experiment = tmp_path / 'experiment'; experiment.mkdir()
    for arm in campaign.ARMS:
        put(experiment / 'configs' / f'{arm}.json', {'arm': arm})
    data, coverage = campaign.prepare_evaluation(manifest(tmp_path), experiment)
    frozen = campaign.read(experiment / 'evaluation/manifest.json')
    assert coverage['passed'] and coverage['samples'] == 8
    assert frozen['canonical_mse_id'] == 'parent'
    assert len(frozen['task_specs']) == 4
    for stage, protocol in frozen['protocols'].items():
        assert protocol['evaluation_year'] == 2022 and protocol['n_samples'] == 8
        assert protocol['initialization_times'] == list(campaign.DATES)
        assert protocol['target_steps'] == 40 and protocol['warmup_steps'] == 0
        assert protocol['data_path'] == data['root']
    exact = frozen['protocols']['exact']
    bounded = times[times >= np.datetime64('2022-01-14T18:00:00')]
    assert exact['chosen_idx'] == [int(np.flatnonzero(bounded == np.datetime64(stamp.replace('Z', '')))[0]) for stamp in campaign.DATES]
    assert exact['chosen_idx'][0] == 1
    # Exercise the evaluator's actual manifest resolver on its bounded dataset.
    import pandas as pd
    from src.models.mamba.v24_Ilya.data import EvalDataset, resolve_initialization_manifest
    bounded = bounded[bounded <= np.datetime64('2022-11-25T00:00:00')]
    resolved = resolve_initialization_manifest(
        EvalDataset(xr.Dataset(coords={'time': bounded}), pd.Timedelta('6h'), 2),
        Path(exact['initialization_manifest']['path']), target_steps=40)
    assert [row['dataset_index'] for row in resolved] == exact['chosen_idx']


def test_terminal_report_uses_single_canonical_parent_gc_and_retains_drift(tmp_path, monkeypatch):
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    m = manifest(tmp_path)
    tasks = campaign.tasks_for(m, tmp_path)
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: m)
    monkeypatch.setattr(campaign, 'qualification_proof', lambda *a: {'passed': True})
    monkeypatch.setattr(campaign, 'train_proof', lambda *a: {'passed': True})
    put(tmp_path / 'checks/qualification.json', {'passed': True})
    monkeypatch.setattr(evaluator, 'load_manifest', lambda *a: {'task_specs': tasks})
    def resolve(_path, index, stage):
        baseline = np.arange(1, 41, dtype=float) * (1 + index * .01)
        full = np.full(40, 10. + index)
        result = {'original_graphcast_loss': {
            'baseline_per_step': baseline.tolist(), 'full_per_step': full.tolist(),
            'baseline_rollout': float(baseline.mean()), 'full_rollout': float(full.mean()),
            'improvement_pct_rollout': float(100 * (1 - full.mean() / baseline.mean()))}}
        return tmp_path, {}, result, False
    monkeypatch.setattr(evaluator, 'resolve_result', resolve)
    monkeypatch.setattr(evaluator, 'load_spectral_result', lambda *a: {})
    monkeypatch.setattr(evaluator, 'feature_rows', lambda *a: [
        {'feature': 'T2m', 'day': 1, 'mean_relative_power_error_pct': 10., 'mean_coherence': .9}])
    campaign.report(tmp_path)
    result = campaign.read(tmp_path / 'evaluation/report/pilot_report.json')
    summary = result['summary']
    assert summary[1]['exact_gc_gain_pct'] == pytest.approx(100 * (1 - 11 / 20.5))
    assert summary[1]['own_baseline_gain_pct'] != summary[1]['exact_gc_gain_pct']
    assert summary[1]['loss_change_vs_parent_pct'] == pytest.approx(10.)
    assert summary[1]['baseline_max_abs_drift'] == pytest.approx(.4)
    assert not result['selection_allowed'] and not result['automatic_extension']
    assert campaign.unseal(tmp_path / 'complete.json')['status'] == 'complete'
