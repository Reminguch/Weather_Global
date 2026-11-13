import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from scripts.analyze_models import report_v24_optimal_smoothing as report
from scripts.experiments.run_v24_mgda import pinned, seal


def fixture_campaign(tmp_path, monkeypatch):
    from scripts.experiments import run_v24_smoothing as original
    from src.analysis.smoothing_observer import atomic_npz
    from src.analysis.spherical_smoothing import json_digest

    layout = {'channels': [['temperature', 850.]], 'scales': [2.], 'weights': [3.]}
    manuscript = tmp_path / 'v4.tex'
    manuscript.write_text('Fixed v4 reference')
    frozen = seal(tmp_path / 'filters.json', {'campaign_hash': 'test', 'layout': layout})
    paths, canonical_records = [], []
    factors = np.asarray([[1.01, .95, .8], [.8, .7, .6]])
    for index in range(6):
        stamp = datetime(2023, 1, 1) + timedelta(days=3 * index)
        identity = stamp.strftime('%Y%m%dT%H%M%SZ')
        # Unequal date and lead losses reveal accidental averaging of ratios.
        canonical = (index + 1.) * np.arange(1, 41)
        losses = canonical[:, None, None] * factors
        truth = np.full((40, 1, 3), index + 1.)
        prediction = np.broadcast_to(truth[:, None, None], (40, 2, 3, 1, 3)).copy()
        prediction *= 2 if index == 0 else 1
        cross = .1 * prediction
        error = prediction + truth[:, None, None] - 2 * cross
        arrays = {'exact_gc_loss': losses, 'grid_mse': losses[..., None] / .75,
                  'spectral_weighted_gc_loss': error.sum(axis=(-1, -2)) * .75,
                  'truth_power': truth, 'prediction_power': prediction,
                  'cross_power': cross, 'error_power': error,
                  'forecast_projection_residual_mse': np.full((40, 2, 1), .1),
                  'truth_projection_residual_mse': np.full((40, 1), .05),
                  'mean_drift': np.full((40, 2, 3, 1), .001),
                  'prediction_mean': np.ones((40, 2, 3, 1)), 'truth_mean': np.ones((40, 1)),
                  'negative_fraction': np.zeros((40, 2, 3, 1)),
                  'positive_field_minimum': np.ones((40, 2, 3, 1))}
        path = tmp_path / f'{identity}.npz'
        atomic_npz(path, arrays, {'campaign_hash': 'test', 'mode': 'evaluate',
            'status': 'complete', 'initialization_id': identity, 'layout': layout,
            'branches': ['GC', 'M1'], 'variants': ['raw', 'projected', 'optimal'],
            'filter_digest': json_digest(frozen), 'source_digest': 'source',
            'checkpoint_sha256': 'm1'})
        paths.append(path)
        canonical_records.append({'initialization_id': identity, 'lead_hours': list(range(6, 241, 6)),
                                  'baseline_per_step': canonical.tolist(), 'full_per_step': (.8 * canonical).tolist()})
    references = []
    # Reverse date order and scatter between shards: identity, not list order,
    # must define pairing.
    for index in range(2):
        path = tmp_path / f'M1_{index:02d}.json'
        path.write_text(json.dumps({'model_id': 'M1', 'checkpoint_sha256': 'm1',
            'baseline_checkpoint_sha256': 'gc',
            'original_graphcast_loss_per_initialization': canonical_records[::-1][index::2]}))
        references.append(pinned(path))
    manifest = {'format': 'v24_optimal_smoothing_campaign_v1', 'model_id': 'M1',
        'record_hash': 'test', 'source_digest': 'source', 'checkpoint': {'sha256': 'm1'},
        'baseline': {'checkpoint_sha256': 'gc'}, 'target_steps': 40,
        'policy': {'lmax': 2, 'fit_year': 2022, 'evaluation_year': 2023},
        'manuscript_reference': pinned(manuscript), 'v4_reference_metrics': references}
    monkeypatch.setattr(original, 'statistics_paths', lambda *args: paths)
    monkeypatch.setattr(report, 'plot_summary', lambda *args: None)
    monkeypatch.setattr(report, 'BOOTSTRAP', {'replicates': 100, 'block_length': 5, 'seed': 20260916})
    return manifest, paths


def test_report_uses_canonical_denominator_and_preserves_repeated_drift(tmp_path, monkeypatch):
    manifest, paths = fixture_campaign(tmp_path, monkeypatch)
    result = report.report(tmp_path, manifest)
    rows = {(row['model'], row['variant']): row for row in result['summary']}
    # Current raw GC is 1% higher than the historical v4 GC. Canonical reporting
    # must show -1%, rather than silently renormalizing the repeat to zero.
    assert rows['GC', 'raw']['rollout_improvement_pct'] == pytest.approx(-1.)
    assert rows['M1', 'optimal']['rollout_improvement_pct'] == pytest.approx(40.)
    assert result['filtered_m1_vs_filtered_gc']['rollout_reduction_pct'] == pytest.approx(25.)
    assert rows['M1', 'optimal']['rollout_improvement_ci95_lower'] == pytest.approx(40.)
    assert rows['M1', 'optimal']['rollout_improvement_ci95_upper'] == pytest.approx(40.)
    assert result['numerical_diagnostics']['repeated_v4_raw']['exceeds_reference']
    assert result['numerical_diagnostics']['warnings']
    assert result['scientific_improvement_required'] is False
    # Pooled ratio=(2+2+3+4+5+6)/(1+2+3+4+5+6), not mean(2,1,1,1,1,1).
    assert rows['GC', 'raw']['rollout_mean_relative_power_error_pct'] == pytest.approx(100 / 21.)
    with np.load(tmp_path / 'annual_pooled.npz', allow_pickle=False) as archive:
        canonical = archive['canonical_v4_raw_loss_per_initialization']
        assert canonical.shape == (6, 40, 2)
        assert canonical[0, 0, 0] == 1.
        assert canonical[-1, -1, 0] == 240.
        np.testing.assert_allclose(archive['physical_channel_spectral_mse'], archive['error_power'].sum(axis=-1))
        np.testing.assert_allclose(archive['raw_repeat_difference_per_initialization'][..., 0], .01 * canonical[..., 0])
        np.testing.assert_allclose(archive['raw_repeat_reference_bound_per_initialization'], 1e-5 * np.abs(canonical))
    for name in ('summary.csv', 'perlead.csv', 'physical_channel_spectral.csv',
                 'per_initialization.csv', 'REPORT.md', 'report.json'):
        assert (tmp_path / name).is_file()
    # Recovering after report output creation is deterministic and does not
    # replace immutable pooled data or change the sealed report.
    assert report.report(tmp_path, manifest) == result


def test_reference_identity_and_cohort_are_strict(tmp_path, monkeypatch):
    manifest, paths = fixture_campaign(tmp_path, monkeypatch)
    ids = [path.stem for path in paths]
    with pytest.raises(ValueError, match='cohorts differ'):
        report.load_v4_reference(manifest, ids[:-1])
    manifest['checkpoint']['sha256'] = 'wrong-checkpoint'
    with pytest.raises(ValueError, match='checkpoint identity'):
        report.load_v4_reference(manifest, ids)


def test_nonfinite_saved_statistics_remain_errors(tmp_path, monkeypatch):
    manifest, paths = fixture_campaign(tmp_path, monkeypatch)
    with np.load(paths[0], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['exact_gc_loss'][0, 0, 0] = np.nan
    np.savez_compressed(paths[0], **arrays)
    with pytest.raises(ValueError, match='Nonfinite saved statistics'):
        report.report(tmp_path, manifest)
