"""Contract tests for the full-GC horizon experiment; no GPU jobs are launched."""
from copy import deepcopy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import xarray as xr

from scripts.experiments import run_v24_full_gc_ar as campaign
from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_checkpoint, training_checkpoint_payload
from src.models.mamba.v24_Ilya.training.config import load_training_config


def template():
    return json.loads((campaign.ROOT / campaign.PARENT / 'configs' / f'{campaign.K20_ID}.json').read_text())


@pytest.mark.parametrize('horizon', [8, 12, 16])
def test_horizon_configuration_changes_only_feedback_tail_and_output(tmp_path, horizon):
    original = template()
    before = deepcopy(original)
    result = campaign.make_config(original, tmp_path, horizon)
    expected = deepcopy(before)
    expected['sequence']['ar_tail_k'] = horizon
    expected['output'] = result['output']
    assert result == expected
    assert original == before
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(result))
    config = load_training_config(path)
    assert config.truth_prefix_steps == 24 - horizon
    assert config.max_steps == 2500 and config.learning_rate == 3e-7
    assert config.train_order_manifest == Path(before['data']['train_order_manifest'])
    assert config.model_family == 'full_gc' and config.spectral_excess_weight == 0


def test_all_horizons_have_five_raw_checkpoints_and_one_swa():
    assert {campaign.task_index('evaluate-early', i) for i in range(8)} == {
        (k, step) for k in campaign.HORIZONS for step in (500, 1000)}
    assert {campaign.task_index('evaluate-late', i) for i in range(16)} == {
        (k, step) for k in campaign.HORIZONS for step in (1500, 2000, 2500, 'swa')}
    with pytest.raises(ValueError):
        campaign.task_index('evaluate-early', 8)


def test_spherical_observer_keeps_truth_baseline_and_each_forecast():
    class Operator:
        def spectral_chunk(self, forecast, truth):
            return {'truth_power': truth[:, :2] ** 2,
                    'prediction_power': forecast[:, :2] ** 2,
                    'error_power': (forecast[:, :2] - truth[:, :2]) ** 2,
                    'cross_power': forecast[:, :2] * truth[:, :2]}
    observer = campaign.RawSphericalObserver.__new__(campaign.RawSphericalObserver)
    observer.base = SimpleNamespace(layout=SimpleNamespace(channels=tuple(range(3))),
                                    lmax=1, chunk_channels=2, operator=Operator(), records=[])
    truth = np.array([[1., 2.], [3., 4.], [5., 6.]])
    observer._record((truth + 1, truth + 2), truth, 0)
    record = observer.base.records[0]
    np.testing.assert_array_equal(record['truth_power'], truth ** 2)
    np.testing.assert_array_equal(record['prediction_power'][0], (truth + 1) ** 2)
    np.testing.assert_array_equal(record['prediction_power'][1], (truth + 2) ** 2)
    np.testing.assert_array_equal(record['error_power'][1], np.full((3, 2), 4.))
    with pytest.raises(ValueError, match='Nonfinite'):
        observer._record((truth, truth * np.nan), truth, 0)


def test_spherical_observer_real_transform_writes_complete_powers(tmp_path):
    latitude = np.linspace(-90, 90, 9)
    longitude = np.arange(16) * 22.5
    field = np.sin(np.deg2rad(latitude))[:, None] * np.ones((1, 16))
    truth = xr.Dataset({'2m_temperature': (('lat', 'lon'), field)},
                       coords={'lat': latitude, 'lon': longitude})
    scales = xr.Dataset({'2m_temperature': xr.DataArray(1.)})
    observer = campaign.RawSphericalObserver(tmp_path, scales, {'campaign_hash': 'test'})
    observer.base.target_steps = 1
    observer.base.lmax = 5
    observer.base.knots = (1, 2, 3, 5)
    observer.base.expected_channels = 1
    observer.begin_sample({'initialization_id': '20220101T000000Z',
                           'initialization_time': '2022-01-01T00:00:00Z', 'lead_hours': [6]})
    observer.update_step(0, truth, truth * 1.2, truth * .8)
    observer.finish_sample()
    with np.load(tmp_path / '20220101T000000Z.npz') as data:
        assert data['truth_power'].shape == (1, 1, 6)
        assert data['prediction_power'].shape == (1, 2, 1, 6)
        for branch in (0, 1):
            np.testing.assert_allclose(data['error_power'][:, branch],
                data['truth_power'] + data['prediction_power'][:, branch] - 2 * data['cross_power'][:, branch],
                rtol=1e-6, atol=1e-10)
        assert json.loads(str(data['metadata']))['status'] == 'complete'


def test_actual_swa_builder_preserves_full_gc_family_and_averages_weights(tmp_path):
    cfg_path = tmp_path / 'config.json'
    cfg_path.write_text(json.dumps(template()))
    config = load_training_config(cfg_path).to_dict()
    run = tmp_path / 'original_run'
    directory = run / 'checkpoints'
    directory.mkdir(parents=True)
    for step in campaign.SWA_STEPS:
        payload = training_checkpoint_payload(completed_step=step,
            residual_params={'graphcast': {'w': np.array([step, 2 * step], dtype=np.float32)}},
            residual_state={}, optimizer_state={}, rng_key=jax.random.PRNGKey(22),
            training_cursor={'epoch': 0, 'segment_index': 0, 'segment_offset': 0},
            resolved_training_config=config, baseline_checkpoint_path='public.npz',
            baseline_checkpoint_fingerprint='baseline', anchor_manifest_fingerprint='anchors',
            parameter_overlay_metadata={})
        (directory / f'checkpoint_step{step:08d}.pkl').write_bytes(pickle.dumps(payload))
    root = tmp_path / 'campaign'
    (root / 'swa').mkdir(parents=True)
    (root / 'checks').mkdir()
    manifest = {'k20_run': str(run), 'record_hash': 'test'}
    proof = campaign.build_swa(root, manifest, 20)
    result = load_v24_Ilya_checkpoint(Path(proof['checkpoint']['path']))
    assert result.model_family == 'full_gc' and not result.has_residual_state
    np.testing.assert_array_equal(result.residual_params['graphcast']['w'], [1750., 3500.])
    assert proof['source_steps'] == [1000, 1500, 2000, 2500]
    with pytest.raises(FileExistsError):
        campaign.build_swa(root, manifest, 20)


def test_submission_plan_respects_dependencies_and_has_no_throttle(tmp_path):
    manifest = {'workspace': '/workspace', 'source_root': '/snapshot'}
    nodes = campaign.submission_plan(tmp_path, manifest)['nodes']
    by_id = {node['id']: node['command'] for node in nodes}
    assert '--array=0-2' in by_id['train0']
    assert '--dependency=aftercorr:<train0>' in by_id['train1']
    assert '--dependency=afterok:<train0>' in by_id['evaluate-early']
    assert '--array=0-7' in by_id['evaluate-early']
    assert '--array=0-15' in by_id['evaluate-late']
    assert '--dependency=afterok:<evaluate-early>:<evaluate-late>' in by_id['report']
    assert all('%' not in arg for cmd in by_id.values() for arg in cmd if arg.startswith('--array='))


def test_existing_k20_state_checks_live_queue_before_accounting(monkeypatch):
    calls = []
    def pending(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='PENDING\n')
    monkeypatch.setattr(campaign.subprocess, 'run', pending)
    assert campaign.k20_job_state('17049163_0') == 'PENDING'
    assert len(calls) == 1 and calls[0][0] == 'squeue'

    def completed(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='' if args[0] == 'squeue' else 'COMPLETED|\n')
    calls.clear()
    monkeypatch.setattr(campaign.subprocess, 'run', completed)
    assert campaign.k20_job_state('17049163_0') == 'COMPLETED'
    assert [args[0] for args in calls] == ['squeue', 'sacct']


def test_report_produces_all_checkpoint_comparisons_and_keeps_baseline_drift(tmp_path):
    root = tmp_path / 'campaign'
    (root / 'report').mkdir(parents=True)
    checkpoint = root / 'checkpoint.pkl'
    checkpoint.write_bytes(b'fixture')
    metrics = root / 'metrics.json'
    metrics.write_text('{}')
    layout = {'channels': [['2m_temperature', None], ['geopotential', 500],
                           ['u_component_of_wind', 850]]}
    truth = np.ones((40, 3, 180))
    forecasts = np.ones((40, 2, 3, 180))
    for horizon in campaign.HORIZONS:
        for point in campaign.EARLY + campaign.LATE:
            label = f'step{point:05d}' if isinstance(point, int) else 'swa'
            dest = root / 'evaluation' / f'k{horizon:02d}' / label
            dest.mkdir(parents=True)
            spectra = dest / 'pooled_spectra.npz'
            np.savez_compressed(spectra, truth_power=truth, prediction_power=forecasts,
                                error_power=forecasts, cross_power=forecasts,
                                metadata=np.asarray(json.dumps({'layout': layout})))
            campaign.seal(dest / 'complete.json', {
                'horizon': horizon, 'point': point, 'manifest_hash': 'fixture',
                'checkpoint': campaign.pinned(checkpoint), 'evaluation': campaign.pinned(metrics),
                'pooled_spectra': campaign.pinned(spectra),
                'baseline_rollout': 10. + horizon * 1e-5, 'full_rollout': 9.,
                'baseline_day10': 20., 'full_day10': 18.})
    result = campaign.report(root, {'record_hash': 'fixture'})
    assert len(result['records']) == 24
    assert len(result['figures']) == 6 and all(Path(p).is_file() for p in result['figures'])
    assert len(result['common_gc_baseline']['comparisons']) == 24
    assert result['baseline_repeatability']['max_abs_difference'] == pytest.approx(0.00012)
    assert all(x['magnitude_exceeds_baseline_spread'] for x in result['baseline_repeatability']['loss_differences'])
