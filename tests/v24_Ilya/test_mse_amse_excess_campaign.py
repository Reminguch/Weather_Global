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


def observed_manifest(tmp_path):
    value = manifest(tmp_path)
    value['policy'] = {'feedback_mode': 'closed_loop', 'resource_stage': 'observe',
                       'observation_updates': 8, 'observation_advances_production': True}
    value['resources'].pop('qualification')
    value['resources'].update(observe={'memory': 320, 'minutes': 240},
                             measurement_role='first_eight_production_updates')
    return value


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
        assert config.mgda_backtracking['repeatability_policy'] == 'warn'
        assert config.mgda_backtracking['decrease_policy'] == 'warn'
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


def test_full_feedback_changes_only_shared_gradient_mode_and_backend(tmp_path):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    template = campaign.read(campaign.ROOT / campaign.INITIAL_ROOT / 'configs/di16_joint.json')
    for arm in campaign.ARMS:
        stopped = campaign.make_config(template, tmp_path, arm)
        full = campaign.make_config(template, tmp_path, arm, feedback_mode='closed_loop')
        config = load_training_config(put(tmp_path / f'{arm}.json', full))
        assert config.feedback_mode == 'closed_loop' and config.bptt_steps == 24
        assert config.ar_tail_k == 20 and config.segment_steps == 96
        assert config.max_steps == 200 and config.checkpoint_every == 50
        assert config.distributed.global_batch_size == 4
        stopped['sequence']['feedback_mode'] = 'closed_loop'
        stopped['memory']['bptt_backend'] = 'explicit_weather_reverse_vjp'
        assert stopped == full


def test_full_feedback_cannot_reuse_old_source_overlay_or_qualification(tmp_path):
    for kwargs in ({'recovery_from': tmp_path / 'old'}, {'reuse_qualification': True}):
        with pytest.raises(ValueError, match='fresh workspace snapshot'):
            campaign.prepare(tmp_path / 'new', tmp_path, feedback_mode='closed_loop', **kwargs)
    assert not (tmp_path / 'new').exists()


def test_fresh_snapshot_includes_all_new_runtime_and_historical_loader(tmp_path, monkeypatch):
    workspace, legacy, destination = tmp_path / 'workspace', tmp_path / 'legacy', tmp_path / 'frozen'
    source_files = {
        campaign.PREPARED_LOADER: 'new loader',
        'src/models/mamba/v24_Ilya/training/weather_feedback_step.py': 'new engine',
        'src/models/mamba/v24_Ilya/training/runner.py': 'new runner',
        'src/models/mamba/v24_Ilya/training/memory_contract.py': 'new metadata',
        'third_party/graphcast/graphcast/graphcast.py': 'graphcast',
        campaign.WRAPPER: 'observe wrapper', campaign.DRIVER: 'observe driver'}
    for relative, value in source_files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    legacy.mkdir(); (legacy / 'git_commit.txt').write_text('parent-revision')
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda *a, **k: b'historical loader')
    source, provenance = campaign.freeze_runtime(workspace, destination, legacy)
    assert set(source['files']) == set(source_files)
    assert (destination / campaign.PREPARED_LOADER).read_text() == 'historical loader'
    for relative, value in source_files.items():
        if relative != campaign.PREPARED_LOADER:
            assert (destination / relative).read_text() == value
    assert provenance['other_source'] == 'current workspace frozen at preparation'
    assert 'recovery_source' not in provenance


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


def test_observation_and_continuation_commands_preserve_total_budget(tmp_path):
    m = observed_manifest(tmp_path)
    initial = list(map(str, campaign.training_command(tmp_path, m, 'mgda', stop_after_steps=8)))
    assert initial[-2:] == ['--stop-after-steps', '8']
    assert '--init-from' in initial and '--zero-state-on-init' in initial and '--resume' not in initial
    continuation = list(map(str, campaign.training_command(tmp_path, m, 'mgda', resume='/run/step8.pkl')))
    assert continuation[-2:] == ['--resume', '/run/step8.pkl']
    assert '--init-from' not in continuation and '--zero-state-on-init' not in continuation
    assert '--stop-after-steps' not in continuation


def test_measured_budget_and_hard_limits():
    qualified = {'passed': True, 'batches': 4, 'peak_rss_gib': 240., 'steady_p90_seconds': 180.}
    assert campaign.resource_budget(qualified) == {'memory': 336, 'minutes': 870}
    for change in ({'passed': False}, {'batches': 3}, {'peak_rss_gib': float('nan')},
                   {'peak_rss_gib': 400.}, {'steady_p90_seconds': 700.}):
        with pytest.raises(ValueError):
            campaign.resource_budget({**qualified, **change})


def test_production_resource_budget_uses_eight_real_updates_and_compile_measurement():
    observed = {'passed': True, 'measurement_kind': 'production_observation', 'completed_steps': 8,
                'peak_rss_gib': 240., 'steady_p90_seconds': 180., 'startup_compile_seconds': 1800.}
    assert campaign.resource_budget(observed) == {'memory': 336, 'minutes': 900}
    for change in ({'completed_steps': 4}, {'passed': False}, {'startup_compile_seconds': float('nan')},
                   {'peak_rss_gib': 400.}, {'steady_p90_seconds': 700.}):
        with pytest.raises(ValueError):
            campaign.resource_budget({**observed, **change})


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


def test_full_feedback_submit_observes_production_and_dry_run_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: observed_manifest(tmp_path))
    monkeypatch.setattr(campaign, 'qualification_proof', lambda *a: pytest.fail('discarded qualification used'))
    calls = []
    def submit(_root, name, command):
        calls.append((name, command))
        return str(100 + len(calls))
    monkeypatch.setattr(campaign, 'submit_once', submit)
    campaign.submit(tmp_path)
    assert [name for name, _ in calls] == ['observe', 'release']
    assert '--dependency=afterok:101' in calls[1][1]
    assert '--mem=320G' in calls[0][1] and '--time=240' in calls[0][1]
    graph = {row['id']: row for row in campaign.dry_run(tmp_path)['nodes']}
    assert 'qualify' not in graph and graph['release']['depends_on'] == ['observe']
    assert graph['observe']['production_updates'] == 8 and graph['observe']['max_steps'] == 200
    assert graph['train']['updates_this_stage'] == {'mgda': 192, 'mse_control': 200, 'fixed_sum': 200}
    assert graph['train']['resources']['measured_from'] == 'checks/observation.json'
    assert graph['evaluate_exact']['depends_on'] == graph['evaluate_spectral']['depends_on'] == ['train']
    outputs = campaign.status(tmp_path)['missing_outputs']
    assert 'checks/observation.json' in outputs and 'checks/qualification.json' not in outputs


def test_full_feedback_release_uses_observation_and_all_final_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: observed_manifest(tmp_path))
    monkeypatch.setattr(campaign, 'qualification_proof', lambda *a: pytest.fail('old qualification used'))
    monkeypatch.setattr(campaign, 'observation_proof', lambda *a: {
        'passed': True, 'measurement_kind': 'production_observation', 'completed_steps': 8,
        'peak_rss_gib': 235., 'steady_p90_seconds': 100., 'startup_compile_seconds': 1200.})
    put(tmp_path / 'checks/observation.json', {'production_updates': 8})
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
    resources = campaign.unseal(tmp_path / 'resources/production.json')
    assert resources['updates_remaining'] == {'mgda': 192, 'mse_control': 200, 'fixed_sum': 200}
    assert resources['measurement_role'] == 'first_eight_production_updates'
    assert resources['measured_from'] == campaign.pinned(tmp_path / 'checks/observation.json')


def test_observation_failure_cannot_release(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: observed_manifest(tmp_path))
    monkeypatch.setattr(campaign, 'observation_proof', lambda *a: (_ for _ in ()).throw(ValueError('corrupt checkpoint')))
    monkeypatch.setattr(campaign, 'submit_once', lambda *a: pytest.fail('invalid observation released jobs'))
    with pytest.raises(ValueError, match='corrupt checkpoint'):
        campaign.release(tmp_path)


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
        mode='mgda', tolerances=tolerances, initial_learning_rate=first_lr, decrease_policy='error')
    row = {**accepted(), 'learning_rate': result.learning_rate,
        'acceptance_threshold_1': result.thresholds[0],
        'acceptance_threshold_2': result.thresholds[1]}
    return {'acceptance': result.to_dict(), 'repeatability': tolerances.to_dict()}, row


def test_candidate_history_accepts_real_fp32_schedule_and_pins_each_trial():
    history, row = acceptance_history()
    assert history['acceptance']['trials'][0]['learning_rate'] != 1e-6
    campaign.validate_update(row, 'mgda')
    campaign.validate_acceptance_history(history, row)


def test_saved_large_variance_warning_does_not_block_campaign_validation():
    history, row = acceptance_history()
    history['repeatability'].update(warning=True, repeatability_span=[.1, .1],
        noise_exceeded=[True, True], accepted_decrease_above_noise=False)
    row.update(repeatability_warning=True, repeatability_span_1=.1, repeatability_span_2=.1,
               accepted_decrease_above_noise=False)
    campaign.validate_update(row, 'mgda')
    campaign.validate_acceptance_history(history, row)


def fake_production(tmp_path, monkeypatch):
    """Exercise campaign observation/resume artifacts without GPU training."""
    from types import SimpleNamespace
    m = observed_manifest(tmp_path)
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: m)
    monkeypatch.setattr(campaign, 'gpu_required', lambda: None)
    monkeypatch.setattr(campaign, 'serialization_check', lambda p: {'passed': True})
    monkeypatch.setattr(campaign.resource, 'getrusage', lambda *a: SimpleNamespace(ru_maxrss=240 * 1024**2))
    monkeypatch.setattr(campaign, 'qualification_proof', lambda *a: pytest.fail('qualification used'))
    calls = []
    for arm in campaign.ARMS:
        put(tmp_path / 'configs' / f'{arm}.json', {'max_steps': 200, 'feedback_mode': 'closed_loop'})
    def execute(args):
        args = list(map(str, args)); calls.append(args)
        arm = Path(args[args.index('--config') + 1]).stem
        run = tmp_path / 'runs' / arm
        stop = int(args[args.index('--stop-after-steps') + 1]) if '--stop-after-steps' in args else 200
        history, base = acceptance_history()
        rows = [{**base, 'step': step, 'loss': 3., 'step_seconds': 100.,
                 'end_to_end_seconds': 101., 'device_memory': {'max_peak_bytes_in_use': 1024**3}}
                for step in range(1, stop + 1)]
        run.mkdir(parents=True, exist_ok=True)
        (run / 'train_metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (run / 'backtracking_trials.jsonl').write_text(''.join(
            json.dumps({**history, 'step': step}) + '\n' for step in range(1, stop + 1)))
        put(run / 'run_config.json', {'derived': {'startup_seconds': 500.}})
        checkpoint = put(run / 'checkpoints' / f'checkpoint_step{stop:08d}.pkl', {'step': stop})
        put(run / 'checkpoint_metrics.jsonl', {'step': stop})
        put(run / 'latest_checkpoint.json', {'completed_step': stop, 'checkpoint': str(checkpoint)})
    def validate(experiment, arm, *, expected_steps=200):
        run = experiment / 'runs' / arm
        rows = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(1, expected_steps + 1))
        return rows, run / 'checkpoints' / f'checkpoint_step{expected_steps:08d}.pkl'
    monkeypatch.setattr(campaign, 'call', execute)
    monkeypatch.setattr(campaign, 'validate_run', validate)
    return m, calls


def test_observation_commits_eight_then_exact_resume_reaches_200_without_reinitialization(tmp_path, monkeypatch):
    m, calls = fake_production(tmp_path, monkeypatch)
    proof = campaign.observe(tmp_path)
    assert proof['production_updates'] == proof['completed_steps'] == 8
    assert proof['remaining_mgda_updates'] == 192 and proof['steady_p90_seconds'] == 101.
    assert proof['peak_rss_gib'] == 240. and proof['gpu_memory']['jax_peak_bytes_in_use'] == 1024**3
    assert proof['config'] == campaign.pinned(tmp_path / 'configs/mgda.json')
    with pytest.raises(FileExistsError):
        campaign.observe(tmp_path)
    result = campaign.train(tmp_path, 0)
    assert result['completed_steps'] == 200 and result['updates_this_stage'] == 192
    assert result['resumed_from_observation'] is True
    assert '--stop-after-steps' in calls[0] and '--init-from' in calls[0]
    assert '--resume' in calls[1] and calls[1][-1] == proof['checkpoint']['path']
    assert not {'--init-from', '--zero-state-on-init', '--stop-after-steps'}.intersection(calls[1])
    # The immutable observation remains valid after live files advance to 200.
    assert campaign.observation_proof(tmp_path, m)['production_updates'] == 8
    assert campaign.train_proof(tmp_path, m, 'mgda')['completed_steps'] == 200
    for task in (1, 2):
        result = campaign.train(tmp_path, task)
        assert result['updates_this_stage'] == 200 and not result['resumed_from_observation']
        assert '--init-from' in calls[-1] and '--resume' not in calls[-1]
    with pytest.raises(ValueError, match='changed since observation'):
        campaign.train(tmp_path, 0)
    with pytest.raises(FileExistsError):
        campaign.train(tmp_path, 1)


@pytest.mark.parametrize('tamper', ['checkpoint', 'metrics', 'config', 'source', 'failed_update'])
def test_continuation_rejects_changed_observation_without_launch(tmp_path, monkeypatch, tamper):
    m, calls = fake_production(tmp_path, monkeypatch)
    proof = campaign.observe(tmp_path)
    if tamper == 'checkpoint':
        Path(proof['checkpoint']['path']).write_text('corrupt')
    elif tamper == 'metrics':
        (tmp_path / 'runs/mgda/train_metrics.jsonl').write_text('changed')
    elif tamper == 'config':
        (tmp_path / 'configs/mgda.json').write_text('changed')
    elif tamper == 'source':
        m['source_digest'] = 'different'
    else:
        put(tmp_path / 'runs/mgda/failed_update.json', {'attempted_step': 9})
    with pytest.raises(ValueError):
        campaign.train(tmp_path, 0)
    assert len(calls) == 1


def test_gpu_observation_snapshots_complete_finite_monitor_samples(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    log = tmp_path / 'logs/observe_123_0_gpu.csv'; log.parent.mkdir()
    log.write_text('2026/10/28, 0, A100, 30000, 81920, 95\n'
                   '2026/10/28, 0, A100, 40000, 81920, 99\n'
                   '2026/10/28, 0, A100, N/A, 81920, 99\npartial')
    value = campaign.gpu_observation(tmp_path, [])
    assert value['sampled_peak_used_mib'] == 40000
    assert len(value['samples']) == 2 and value['warning'] is None
    assert campaign.gpu_observation(tmp_path / 'absent', [])['warning'] is not None


@pytest.mark.parametrize('expected_steps', [8, 200])
def test_run_validation_requires_the_declared_endpoint_and_full_budget_config(tmp_path, monkeypatch, expected_steps):
    from collections import namedtuple
    from types import SimpleNamespace
    from src.models.mamba.v24_Ilya import checkpoint as checkpoint_module
    from src.models.mamba.v24_Ilya.training import config as config_module
    run = tmp_path / 'runs/mgda'; run.mkdir(parents=True)
    history, base = acceptance_history()
    rows = [{**base, 'step': step, 'loss': 3., 'step_seconds': 100.}
            for step in range(1, expected_steps + 1)]
    (run / 'train_metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (run / 'backtracking_trials.jsonl').write_text(''.join(
        json.dumps({**history, 'step': step}) + '\n' for step in range(1, expected_steps + 1)))
    baseline = tmp_path / 'baseline'; baseline.write_text('frozen GC')
    config_value = {'optimizer': {'max_steps': 200}, 'sequence': {'feedback_mode': 'closed_loop'}}
    config = SimpleNamespace(to_dict=lambda: config_value, baseline_checkpoint=baseline)
    checkpoint = SimpleNamespace(completed_step=expected_steps, resolved_training_config=config_value,
        optimizer_state=namedtuple('RMSState', 'count')(np.asarray(expected_steps)),
        residual_params={'w': np.asarray(1.)}, residual_state={'m': np.asarray(0.)},
        baseline_checkpoint_fingerprint=campaign.file_sha256(baseline))
    monkeypatch.setattr(config_module, 'load_training_config', lambda p: config)
    monkeypatch.setattr(checkpoint_module, 'load_v24_Ilya_training_checkpoint', lambda p: checkpoint)
    result, path = campaign.validate_run(tmp_path, 'mgda', expected_steps=expected_steps)
    assert len(result) == expected_steps and path.name == f'checkpoint_step{expected_steps:08d}.pkl'
    if expected_steps == 8:
        with pytest.raises(ValueError, match='exactly200'):
            campaign.validate_run(tmp_path, 'mgda')
    checkpoint.optimizer_state = checkpoint.optimizer_state._replace(count=np.asarray(expected_steps - 1))
    with pytest.raises(ValueError, match='moment count'):
        campaign.validate_run(tmp_path, 'mgda', expected_steps=expected_steps)


@pytest.mark.parametrize('arm', campaign.ARMS)
def test_advisory_loss_increase_reaches_validation_without_false_descent_claim(arm):
    from src.models.mamba.v24_Ilya.training.mgda_acceptance import backtrack_losses, calibrate_loss_tolerances
    tol = calibrate_loss_tolerances([[3., 1.]] * 3)
    with pytest.warns(RuntimeWarning):
        result = backtrack_losses([3., 1.], lambda lr: [3.01, 1.01], mode=arm,
                                  tolerances=tol, initial_learning_rate=float(np.float32(1e-6)))
    row = {**accepted(arm), 'decrease_policy': 'warn', 'measured_decrease': False,
        'decrease_warning': True, 'descent_valid': False, 'update_norm': 0.,
        'objective_1_after': 3.01, 'objective_2_after': 1.01,
        'acceptance_threshold_1': result.thresholds[0], 'acceptance_threshold_2': result.thresholds[1],
        'learning_rate': result.learning_rate, 'backtracking_trials': 1, 'backtracking_halvings': 0}
    history = {'acceptance': result.to_dict(), 'repeatability': tol.to_dict()}
    campaign.validate_update(row, arm)
    campaign.validate_acceptance_history(history, row)
    with pytest.raises(ValueError, match='diagnostics'):
        campaign.validate_update({**row, 'measured_decrease': True}, arm)


def test_reused_qualification_is_resource_only_and_does_not_submit_another_gpu_gate(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    m['resources']['qualification_reference'] = {'path': '/old/qualification.json'}
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: m)
    monkeypatch.setattr(campaign, 'qualification_proof', lambda *a: {'passed': True})
    calls = []
    monkeypatch.setattr(campaign, 'submit_once', lambda root, name, command: calls.append((name, command)))
    campaign.submit(tmp_path)
    assert len(calls) == 1 and calls[0][0] == 'release'
    assert not any('dependency' in arg or 'gres=gpu' in arg for arg in calls[0][1])
    graph = campaign.dry_run(tmp_path)
    assert graph['nodes'][0]['id'] == 'release' and graph['nodes'][0]['depends_on'] == []
    assert not any(node['id'] == 'qualify' for node in graph['nodes'])
    assert 'checks/qualification.json' not in campaign.status(tmp_path)['missing_outputs']


def test_reused_qualification_verifies_parent_hashes_and_never_claims_current_proof(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old_values = {**manifest(old), 'source_digest': 'old-source'}
    old_values.pop('record_hash')
    old_manifest = campaign.seal(old / 'manifest.json', old_values)
    raw = put(old / 'raw.json', {'passed': True})
    campaign.seal(old / 'checks/qualification.json', {
        'passed': True, 'batches': 4, 'production_updates': 0,
        'manifest_hash': old_manifest['record_hash'], 'source_digest': 'old-source',
        'raw_qualification': campaign.pinned(raw)})
    m = manifest(new)
    m['resources']['qualification_reference'] = campaign.pinned(old / 'checks/qualification.json')
    m['recovery'] = {'parent_manifest': campaign.pinned(old / 'manifest.json')}
    result = campaign.qualification_proof(new, m)
    assert result['source_digest'] == 'old-source'
    assert not (new / 'checks/qualification.json').exists()
    raw.write_text('{}')
    with pytest.raises(ValueError):
        campaign.qualification_proof(new, m)


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
