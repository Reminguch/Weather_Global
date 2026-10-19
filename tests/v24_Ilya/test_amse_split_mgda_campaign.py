"""Bounded split-AMSE workflow, restart isolation, and submission provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments import run_v24_amse_split_mgda as campaign


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def manifest(tmp_path):
    return {'workspace': str(tmp_path), 'source_root': str(tmp_path / 'source'),
            'source_digest': 'source', 'record_hash': 'manifest',
            'initializations': {arm: {'checkpoint': {'path': f'/init/{arm}.pkl'},
                'source_run_dir': f'/init/{arm}', 'architecture': {'temporal_d_inner': int(arm[2:])}}
                for arm in campaign.ARMS},
            'resources': {'historical_peak_rss_gib': 234.225,
                'calibration': {'di16': {'memory': 128, 'minutes': 90}, 'di32': {'memory': 160, 'minutes': 90}},
                'qualification': {'di16': {'memory': 320, 'minutes': 210}, 'di32': {'memory': 384, 'minutes': 240}}}}


def test_configs_keep_full_horizon_and_one_scientific_identity(tmp_path):
    template = campaign.read(campaign.ROOT / campaign.INITIAL_ROOT / 'configs/di16_joint.json')
    before = copy.deepcopy(template)
    scale = {'version': 1, 'objective_pair': 'amse_split', 'common_scale': 4.}
    canonical = campaign.make_config(template, tmp_path, 'di16', scale)
    shadow = campaign.make_config(template, tmp_path, 'di16', scale, shadow=True)
    for config in (canonical, shadow):
        assert config['optimizer']['max_steps'] == 2000
        assert config['optimizer']['checkpoint_every'] == 50
        assert config['optimizer']['weight_decay'] == 0
        assert config['optimizer']['optimizer_kind'] == 'mgda_rms'
        assert config['objective']['amse_split_calibration'] == scale
        assert config['objective']['amse_lambda'] == 1
        assert config['sequence']['ar_tail_k'] == 20
        assert config['architecture']['temporal_d_inner'] == 16
    assert canonical.pop('output') != shadow.pop('output')
    assert canonical == shadow and template == before
    bootstrap = campaign.make_config(template, tmp_path, 'di16', bootstrap=True)
    assert 'amse_split_calibration' not in bootstrap['objective']
    assert bootstrap['optimizer']['optimizer_kind'] == 'adamw'


def test_six_native_tasks_pair_each_width_to_its_initialization(tmp_path):
    tasks = campaign.tasks_for(manifest(tmp_path), tmp_path)
    assert [row['id'] for row in tasks] == [f'{arm}_step{step:05d}' for arm in campaign.ARMS for step in (0, 500, 2000)]
    assert len(tasks) == 6
    for row in tasks:
        assert row['paired_initialization_id'] == row['family'] + '_step00000'
        assert row['architecture']['temporal_d_inner'] == int(row['family'][2:])
        assert row['endpoint'] == ('primary' if row['step'] == 2000 else 'initialization' if not row['step'] else 'diagnostic')
        assert 'swa' not in row['checkpoint']


def test_training_commands_promote_canonical100_without_schedule_reset(tmp_path):
    m = manifest(tmp_path)
    qualification = list(map(str, campaign.training_command(tmp_path, m, 'di32', stop_after=100)))
    assert qualification[-2:] == ['--stop-after-steps', '100']
    assert '--zero-state-on-init' in qualification
    canonical = tmp_path / 'runs/di32/checkpoints/checkpoint_step00000100.pkl'
    continuation = list(map(str, campaign.training_command(tmp_path, m, 'di32', resume=canonical)))
    assert continuation[-2:] == ['--resume', str(canonical)]
    assert '--init-from' not in continuation and '--stop-after-steps' not in continuation
    midpoint = tmp_path / 'shadow/di32/checkpoints/checkpoint_step00000050.pkl'
    shadow = list(map(str, campaign.training_command(tmp_path, m, 'di32', resume=midpoint, stop_after=100, shadow=True)))
    assert str(tmp_path / 'configs/di32_shadow.json') in shadow
    assert str(midpoint) in shadow


def test_budget_counts_only_remaining1900_updates():
    report = {'passed': True, 'completed_steps': 100, 'peak_rss_gib': 240., 'steady_p90_seconds': 50.}
    assert campaign.resource_budget(report) == {'memory_gib': 336, 'minutes': 2190}
    for change in ({'passed': False}, {'completed_steps': 99}, {'peak_rss_gib': float('nan')},
                   {'peak_rss_gib': 400}, {'steady_p90_seconds': 200.}):
        with pytest.raises(ValueError):
            campaign.resource_budget({**report, **change})


def test_dry_run_dag_has_no500_skill_gate_or_swa(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    put(tmp_path / 'manifest.json', m)
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: m)
    monkeypatch.setattr(campaign, 'submit_once', lambda *a, **k: pytest.fail('dry run submitted'))
    before = sorted(tmp_path.rglob('*'))
    result = campaign.dry_run(tmp_path)
    assert sorted(tmp_path.rglob('*')) == before
    nodes = {row['id']: row for row in result['nodes']}
    assert nodes['combine']['depends_on'] == ['calibrate_di16', 'calibrate_di32']
    assert nodes['release']['depends_on'] == ['qualify_di16', 'qualify_di32']
    assert nodes['train_di32']['remaining_updates'] == 1900
    assert nodes['evaluate_exact']['depends_on'] == ['train_di16', 'train_di32']
    assert '--array=0-5' in nodes['evaluate_spectral']['command']
    assert nodes['report']['depends_on'] == ['evaluate_exact', 'evaluate_spectral']
    assert not any('500' in name or 'swa' in name for name in nodes)
    assert not result['submitted']


def test_sbatch_pins_environment_dependencies_and_limits(tmp_path):
    args = campaign.sbatch(tmp_path, manifest(tmp_path), 'evaluate-spectral', memory=16, minutes=60,
                           gpu=True, array='0-5', dependency='100:101')
    assert '--kill-on-invalid-dep=yes' in args
    assert '--dependency=afterok:100:101' in args
    assert '--array=0-5' in args and '--qos=gpu-short' in args
    assert args[-3:] == [str(tmp_path / 'source' / campaign.WRAPPER), 'evaluate-spectral', str(tmp_path)]
    for kw in ({'array': '0-5%2'}, {'minutes': 4321}, {'memory': 513}):
        with pytest.raises(ValueError):
            campaign.sbatch(tmp_path, manifest(tmp_path), 'train', **{'memory': 320, 'minutes': 1920, **kw})


def test_release_requires_both_proofs_before_any_submission(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: m)
    calls = []
    def proof(*args, **kwargs):
        if args[2] == 'di32':
            raise ValueError('di32 qualification missing')
        return {'passed': True, 'completed_steps': 100, 'peak_rss_gib': 230., 'steady_p90_seconds': 42.}
    monkeypatch.setattr(campaign, 'proof', proof)
    monkeypatch.setattr(campaign, 'submit_once', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='di32'):
        campaign.release(tmp_path)
    assert not calls


def test_submission_graph_records_stage_dependencies(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: m)
    calls = []
    def submit(_experiment, name, args):
        calls.append((name, args))
        return str(100 + len(calls))
    monkeypatch.setattr(campaign, 'submit_once', submit)
    campaign.submit(tmp_path)
    by_name = dict(calls)
    assert '--dependency=afterok:101:102' in by_name['combine']
    assert '--dependency=afterok:103' in by_name['qualify_di32']
    assert '--dependency=afterok:104:105' in by_name['release']


def test_relocation_changes_only_output_metadata(tmp_path):
    import pickle
    source = tmp_path / 'canonical.pkl'
    payload = {'resolved_training_config': {'output': {'output_root': '/canonical', 'run_name': 'di16'},
               'objective': {'objective_pair': 'amse_split'}},
               'state': np.asarray([3., 4.], np.float32), 'count': 50}
    source.write_bytes(pickle.dumps(payload))
    original = source.read_bytes()
    destination = tmp_path / 'shadow.pkl'
    campaign.relocated_checkpoint(source, destination, {'output_root': '/shadow', 'run_name': 'di16'})
    assert source.read_bytes() == original
    loaded = pickle.loads(destination.read_bytes())
    assert loaded['resolved_training_config']['output']['output_root'] == '/shadow'
    assert loaded['resolved_training_config']['objective'] == payload['resolved_training_config']['objective']
    np.testing.assert_array_equal(loaded['state'], payload['state'])
    assert loaded['count'] == 50


def row(step):
    value = {key: 1. for key in ('loss', 'gradient_norm', 'step_seconds', 'gc_loss', 'amplitude_loss',
        'pattern_loss', 'amplitude_loss_normalized', 'pattern_loss_normalized', 'amplitude_gradient_norm',
        'pattern_gradient_norm', 'combined_gradient_norm', 'direction_norm', 'update_norm',
        'amplitude_direction_dot', 'pattern_direction_dot', 'amplitude_displacement_dot',
        'pattern_displacement_dot', 'metric_amplitude_amplitude', 'metric_amplitude_pattern', 'metric_pattern_pattern')}
    return {**value, 'step': step, 'objective_pair': 'amse_split', 'finite': True,
            'descent_valid': True, 'clip_factor': 1., 'mgda_alpha': .5, 'gradient_cosine': -.3}


def test_training_validation_has_semantic_loss_names_and_all_updates():
    campaign.validate_records([row(1), row(2)], 2)
    for changes in ({'power_loss': 1.}, {'objective_pair': 'gc_power'}, {'descent_valid': False},
                    {'amplitude_loss': float('nan')}, {'pattern_displacement_dot': -.1}):
        with pytest.raises(ValueError):
            campaign.validate_records([{**row(1), **changes}], 1)
    with pytest.raises(ValueError):
        campaign.validate_records([row(1), row(1)], 2)


def test_small_and_large_finite_replay_drift_are_advisory():
    canonical = [row(step) for step in range(1, 101)]
    replay = copy.deepcopy(canonical[50:])
    replay[0]['amplitude_loss'] += .001
    result = campaign.compare_trajectory(canonical, replay, campaign.NUMERICAL_ENVELOPES)
    assert result['amplitude_loss']['within_envelope']
    replay[0]['amplitude_loss'] += 100
    with pytest.warns(RuntimeWarning, match='warning threshold'):
        result = campaign.compare_trajectory(canonical, replay, campaign.NUMERICAL_ENVELOPES)
    assert not result['amplitude_loss']['within_envelope']
    assert result['amplitude_loss']['warning']
    replay[0]['amplitude_loss'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        campaign.compare_trajectory(canonical, replay, campaign.NUMERICAL_ENVELOPES)


def test_timings_include_checkpoint_io_and_data_wait(tmp_path):
    rows = [{**row(i), 'end_to_end_seconds': 40.} for i in range(1, 7)]
    (tmp_path / 'checkpoint_metrics.jsonl').write_text(json.dumps({'step': 5, 'checkpoint_seconds': 3.}) + '\n')
    assert campaign.steady_timings(tmp_path, rows) == [43., 40.]


def test_coordinate_coverage_rejects_missing_leads(monkeypatch):
    import xarray as xr
    times = np.arange(np.datetime64('2023-01-01'), np.datetime64('2023-02-01'), np.timedelta64(6, 'h'))
    dataset = xr.Dataset(coords={'time': times})
    monkeypatch.setattr(xr, 'open_zarr', lambda *a, **k: dataset)
    spec = {'data_path': 'unused', 'stage': 'spectral', 'target_steps': 40,
            'initialization_times': ['2023-01-15T00:00:00Z']}
    assert campaign.coordinate_coverage(spec)['passed']
    with pytest.raises(ValueError, match='coverage'):
        campaign.coordinate_coverage({**spec, 'initialization_times': ['2023-01-30T00:00:00Z']})


def native_configs(tmp_path):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.models.mamba.v24_Ilya.training.amse_split_loss import (
        STAT_FILES, combine_amse_split_calibrations, make_amse_split_calibration_record)
    records, templates = [], {}
    for arm in campaign.ARMS:
        template = campaign.read(campaign.ROOT / campaign.INITIAL_ROOT / f'configs/{arm}_joint.json')
        templates[arm] = template
        path = put(tmp_path / 'configs' / f'calibration_{arm}.json',
                   campaign.make_config(template, tmp_path, arm, bootstrap=True))
        config = load_training_config(path)
        records.append(make_amse_split_calibration_record([[1., 3.]] * 32,
            [{'index': index} for index in range(32)],
            init_from_sha256=campaign.INITIAL_HASHES[arm], source_config_sha256=campaign.file_sha256(path),
            source_config=config.to_dict(), baseline_checkpoint_fingerprint='4' * 64,
            anchor_manifest_fingerprint='5' * 64, prepared_metadata_sha256='6' * 64,
            stats_sha256={name: '7' * 64 for name in STAT_FILES}))
    calibration = combine_amse_split_calibrations(records)
    configs = {}
    for arm in campaign.ARMS:
        for shadow in (False, True):
            path = put(tmp_path / 'configs' / f'{arm}{"_shadow" if shadow else ""}.json',
                campaign.make_config(templates[arm], tmp_path, arm, calibration, shadow=shadow))
            configs[arm, shadow] = path, load_training_config(path)
    return configs, records


def test_real_config_parse_and_calibration_signatures_match_both_widths(tmp_path):
    from src.models.mamba.v24_Ilya.training.amse_split_loss import calibration_signature
    configs, records = native_configs(tmp_path)
    for arm, record in zip(campaign.ARMS, records, strict=True):
        for shadow in (False, True):
            _, config = configs[arm, shadow]
            assert config.optimizer_kind == 'mgda_rms'
            assert config.max_steps == 2000 and config.checkpoint_every == 50
            assert calibration_signature(record['source_config']) == calibration_signature(config.to_dict())


def native_checkpoint(tmp_path):
    from src.models.mamba.v24_Ilya.checkpoint import training_checkpoint_payload, atomic_pickle_dump
    from src.models.mamba.v24_Ilya.training.mgda_optimizer import RMSState
    configs, _ = native_configs(tmp_path)
    path, config = configs['di16', False]
    shadow_path, shadow_config = configs['di16', True]
    canonical = tmp_path / 'runs/di16/checkpoints/checkpoint_step00000100.pkl'
    payload = training_checkpoint_payload(completed_step=100,
        residual_params={'layer': {'w': np.array([1., 2.], np.float32)}},
        residual_state={'layer': {'s': np.array([2., 3.], np.float32)}},
        optimizer_state=RMSState(np.array(100, np.int32), {'layer': {'w': np.array([1e-4, 2e-4], np.float32)}}),
        rng_key=np.array([1, 2], np.uint32), training_cursor={'epoch': 1, 'segment_index': 3, 'segment_offset': 4},
        resolved_training_config=config.to_dict(), baseline_checkpoint_path='baseline.npz',
        baseline_checkpoint_fingerprint='basehash', anchor_manifest_fingerprint='anchorhash',
        parameter_overlay_metadata={'init_from_sha256': campaign.INITIAL_HASHES['di16']})
    canonical.parent.mkdir(parents=True)
    atomic_pickle_dump(payload, canonical)
    shadow = tmp_path / 'shadow/di16/checkpoints/checkpoint_step00000100.pkl'
    campaign.relocated_checkpoint(canonical, shadow, shadow_config.to_dict()['output'])
    return canonical, shadow, path, shadow_path


def test_native_shadow_roundtrip_preserves_all_state_and_canonical_hash(tmp_path):
    canonical, shadow, config, shadow_config = native_checkpoint(tmp_path)
    before = campaign.file_sha256(canonical)
    result = campaign.compare_shadow(canonical, shadow, config, shadow_config, campaign.NUMERICAL_ENVELOPES)
    assert result['bitwise_equal'] and result['output_relocation_only']
    assert result['gross_comparisons']['second_moment']['relative_l2'] == 0
    assert campaign.file_sha256(canonical) == before


@pytest.mark.parametrize('change', ['integer', 'nonfinite', 'configuration'])
def test_native_shadow_rejects_integrity_failures(tmp_path, change):
    import pickle
    canonical, shadow, config, shadow_config = native_checkpoint(tmp_path)
    payload = pickle.loads(shadow.read_bytes())
    if change == 'integer':
        payload['rng_key'][0] += 1
    elif change == 'nonfinite':
        payload['optimizer_state'].second_moment['layer']['w'][0] = np.nan
    else:
        payload['resolved_training_config']['optimizer']['seed'] = 23
    shadow.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError):
        campaign.compare_shadow(canonical, shadow, config, shadow_config, campaign.NUMERICAL_ENVELOPES)


def test_large_second_moment_drift_warns_and_preserves_integrity(tmp_path):
    import pickle
    canonical, shadow, config, shadow_config = native_checkpoint(tmp_path)
    payload = pickle.loads(shadow.read_bytes())
    payload['optimizer_state'].second_moment['layer']['w'][0] += .1
    shadow.write_bytes(pickle.dumps(payload))
    with pytest.warns(RuntimeWarning, match='second_moment'):
        result = campaign.compare_shadow(canonical, shadow, config, shadow_config, campaign.NUMERICAL_ENVELOPES)
    assert result['integrity_passed']
    assert result['status'] == 'complete_with_warnings'
    assert not result['gross_comparisons']['second_moment']['within_envelope']
    assert result['warnings']


def test_actual_runner_shadow50_to100_preserves_canonical_and_resumes_to2000(tmp_path, monkeypatch):
    """Exercise real directory/resume orchestration with a tiny synthetic model."""
    from dataclasses import replace
    from test_mgda_runner import mock_run
    from src.models.mamba.v24_Ilya.training import runner
    from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainInvocation
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    config = replace(mock_run(tmp_path, monkeypatch), max_steps=2000, checkpoint_every=50)
    config_path = put(tmp_path / 'canonical.json', config.to_dict())
    canonical = runner.run_training(V24IlyaTrainInvocation(config, config_path, stop_after_steps=100))
    before = campaign.file_sha256(canonical)
    shadow_config = replace(config, output_root=tmp_path / 'shadow')
    shadow_path = put(tmp_path / 'shadow.json', shadow_config.to_dict())
    midpoint = config.run_dir / 'checkpoints/checkpoint_step00000050.pkl'
    shadow50 = campaign.prepare_shadow_run(config.run_dir, shadow_config.run_dir, midpoint, shadow_path)
    assert (shadow_config.run_dir / 'run_config.json').is_file()
    shadow100 = runner.run_training(V24IlyaTrainInvocation(shadow_config, shadow_path, resume=shadow50, stop_after_steps=100))
    result = campaign.compare_shadow(canonical, shadow100, config_path, shadow_path, campaign.NUMERICAL_ENVELOPES)
    assert result['bitwise_equal']
    assert campaign.file_sha256(canonical) == before
    assert [json.loads(line)['step'] for line in (shadow_config.run_dir / 'train_metrics.jsonl').read_text().splitlines()] == list(range(51, 101))
    # A short post-qualification continuation proves the real cursor/count path;
    # the same max_steps=2000 and schedule remain embedded throughout.
    resumed = runner.run_training(V24IlyaTrainInvocation(config, config_path, resume=canonical, stop_after_steps=102))
    saved = load_v24_Ilya_training_checkpoint(resumed)
    assert saved.completed_step == int(saved.optimizer_state.count) == 102
    assert saved.resolved_training_config['optimizer']['max_steps'] == 2000
    assert [json.loads(line)['step'] for line in (config.run_dir / 'train_metrics.jsonl').read_text().splitlines()] == list(range(1, 103))
