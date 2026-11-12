from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_models import report_v24_smoothing_sweep as report


def test_moments_reconstruct_grid_and_physical_power():
    rng = np.random.default_rng(21)
    # Two channels with unequal weights ensure exact-GC normalization is tested.
    forecast, truth, delta = rng.normal(size=(3, 4, 2, 2, 7))
    moments = {
        'grid_mse': np.mean((forecast - truth)**2, axis=-1),
        'delta_grid_cross': np.mean((forecast - truth) * delta, axis=-1)[:, :, None],
        'delta_grid_norm': np.mean(delta**2, axis=-1)[:, :, None],
    }
    # A complex transform gives independent power and cross-power evidence;
    # the native grid norm above remains the source of the GC loss.
    cf, ct, cd = (np.fft.rfft(value, axis=-1) for value in (forecast, truth, delta))
    moments.update(prediction_power=np.abs(cf)**2, cross_power=(cf * ct.conj()).real,
                   delta_power=(np.abs(cd)**2)[:, :, None],
                   forecast_delta_cross=(cf * cd.conj()).real[:, :, None],
                   delta_truth_cross=(cd * ct.conj()).real[:, :, None])
    layout = {'scales': [2., 5.], 'weights': [.3, 1.7]}
    for alpha in (0., .1, .7, 1.):
        mse, power, cross = report.reconstruct(moments, 0, alpha)
        adjusted = forecast + alpha * delta
        expected_mse = np.mean((adjusted - truth)**2, axis=-1)
        coefficients = np.fft.rfft(adjusted, axis=-1)
        np.testing.assert_allclose(mse, expected_mse, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(power, np.abs(coefficients)**2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cross, (coefficients * ct.conj()).real, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(report.exact_loss(mse, layout),
                                   expected_mse[..., 0] * .3 / 4 + expected_mse[..., 1] * 1.7 / 25)
    with pytest.raises(ValueError, match='Nonfinite'):
        report.reconstruct(moments, 0, float('nan'))


def test_candidate_endpoints_and_undefined_bins():
    from src.analysis.smoothing_sweep_observer import BASE_NAMES
    assert report.BASE_NAMES == list(BASE_NAMES)
    candidates = report.candidates()
    assert len(candidates) == 30
    assert [c['width_degrees'] for c in candidates[:10]] == [0., .25, .5, .75, 1., 1.5, 2., 3., 4., 8.]
    for series in ('fitted_gaussian', 'fitted_band'):
        assert [c['alpha'] for c in candidates if c['series'] == series] == [i / 10 for i in range(1, 11)]
    value = report.ratio(np.array([0., 1., 3.]), np.array([0., 0., 2.]))
    assert np.isnan(value[:2]).all()
    assert value[2] == 1.5
    assert report.json_finite({'undefined': float('nan')}) == {'undefined': None}


def test_report_pools_before_ratios_and_retains_warnings(tmp_path, monkeypatch):
    from scripts.experiments import run_v24_smoothing as campaign
    from scripts.experiments.run_v24_mgda import seal
    from src.analysis.smoothing_observer import atomic_npz
    from src.analysis.spherical_smoothing import json_digest

    layout = {'channels': [['temperature', 850.]], 'scales': [2.], 'weights': [3.]}
    frozen = seal(tmp_path / 'filters.json', {'layout': layout, 'campaign_hash': 'test'})
    paths = []
    for index in range(2):
        truth = np.full((40, 1, 3), 1. if index == 0 else 4.)
        prediction = np.broadcast_to(truth[:, None], (40, 2, 1, 3)).copy()
        prediction *= 2 if index == 0 else 1
        arrays = {'grid_mse': np.ones((40, 2, 1)), 'truth_power': truth,
                  'prediction_power': prediction, 'cross_power': prediction * .25,
                  'delta_grid_cross': np.zeros((40, 2, 11, 1)),
                  'delta_grid_norm': np.zeros((40, 2, 11, 1)),
                  'delta_power': np.zeros((40, 2, 11, 1, 3)),
                  'forecast_delta_cross': np.zeros((40, 2, 11, 1, 3)),
                  'delta_truth_cross': np.zeros((40, 2, 11, 1, 3)),
                  'mean_drift': np.ones((40, 2, 11, 1)) * 2e-8,
                  'exact_gc_loss': np.full((40, 2, 1), .75)}
        path = tmp_path / f'2023010{1 + 3 * index}T000000Z.npz'
        atomic_npz(path, arrays, {'campaign_hash': 'test', 'mode': 'evaluate',
            'status': 'complete', 'initialization_id': path.stem, 'layout': layout,
            'base_names': report.BASE_NAMES, 'branches': ['GC', 'M1'], 'lmax': 2,
            'base_width_degrees': [*report.WIDTHS, None, None],
            'schema': 'v24_smoothing_sweep_statistics_v1',
            'source_digest': 'source', 'checkpoint_sha256': 'checkpoint',
            'filter_digest': json_digest(frozen)})
        paths.append(path)
    monkeypatch.setattr(campaign, 'statistics_paths', lambda *args: paths)
    monkeypatch.setattr(report, 'plot_results', lambda *args: None)
    manifest = {'record_hash': 'test', 'model_id': 'M1', 'source_digest': 'source',
                'checkpoint': {'sha256': 'checkpoint'}, 'workspace': str(tmp_path), 'target_steps': 40}
    result = report.report(tmp_path, manifest, name='sweep-test')
    assert result['status'] == 'complete' and result['count'] == 2
    assert result['numerical_diagnostics']['warnings']
    assert result['scientific_improvement_required'] is False
    for row in result['summary']:
        # (2+4)/(1+4)=1.2, not average(2,1)=1.5.
        assert row['rollout_mean_relative_power_error_pct'] == pytest.approx(20.)
        assert row['exact_gc_rollout_loss'] == pytest.approx(.75)
        assert row['rollout_mean_relative_error_power'] == pytest.approx(1.6)
        assert row['rollout_mean_spectral_correlation'] == pytest.approx(.75 / np.sqrt(7.5))
    with np.load(tmp_path / 'annual_pooled.npz') as archive:
        assert archive['per_initialization_exact_gc_loss'].shape == (2, 40, 2, 30)
        np.testing.assert_allclose(archive['prediction_power'] / archive['truth_power'][:, None], 1.2)
    assert Path(result['data_dir'], 'summary.csv').exists()


def test_repeated_endpoint_drift_is_advisory(tmp_path):
    from scripts.experiments.run_v24_mgda import pinned, seal
    from src.analysis.smoothing_observer import atomic_npz
    parent = tmp_path / 'parent'
    parent.mkdir()
    manifest = seal(parent / 'manifest.json', {'model_id': 'M1'})
    atomic_npz(parent / 'annual_pooled.npz', {'exact_gc_loss': np.ones((40, 2, 3))},
               {'initialization_ids': ['date'], 'branches': ['GC', 'M1'],
                'variants': ['raw', 'gaussian', 'band']})
    seal(parent / 'report.json', {'campaign_hash': manifest['record_hash'], 'status': 'complete',
                                  'pooled_statistics': pinned(parent / 'annual_pooled.npz')})
    observed = np.ones((40, 2, 30))
    observed[:, :, 19] += .01
    result = report.compare_parent_endpoints({'parent_campaign': pinned(parent / 'manifest.json'),
                                              'model_id': 'M1'}, ['date'], observed)
    assert result['exceeds_reference'] is True
    assert result['max_reference_bound_ratio'] == pytest.approx(1000.)
    assert result['max_absolute_difference'] == pytest.approx(.01)
    assert result['current_rollout_loss'][0][1] == pytest.approx(1.01)
