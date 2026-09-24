"""Independent reductions for the offline normalization audit (CPU only)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('loss_alignment_audit',
    Path(__file__).resolve().parents[2] / 'scripts/diagnostics/audit_neuralgcm_loss_alignment.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('per_level', [False, True])
@pytest.mark.parametrize('latitude_weights', [np.ones(6), np.array([1, 2, 3, 3, 2, 1])])
def test_streamed_scales_include_between_sample_and_level_variation(per_level, latitude_weights):
    x = np.random.default_rng(97).normal(size=(5, 3, 4, 6))
    x += np.arange(3)[None, :, None, None] * 20 + np.arange(5)[:, None, None, None] * 7
    weights = np.broadcast_to(latitude_weights, x.shape).astype(float)
    if per_level:
        expected = []
        for level in range(3):
            mean = np.average(x[:, level], weights=weights[:, level])
            expected.append(np.sqrt(np.average((x[:, level] - mean) ** 2, weights=weights[:, level])))
    else:
        mean = np.average(x, weights=weights)
        expected = np.sqrt(np.average((x - mean) ** 2, weights=weights))
    moments = audit.ChangeMoments(3, latitude_weights, per_level)
    for sample in x:
        moments.add(sample)
    np.testing.assert_allclose(moments.std(), expected, rtol=1e-13)


def test_rescoring_aggregates_matches_direct_normalized_field_errors():
    rng = np.random.default_rng(102)
    levels = np.array([2., 500., 850.])
    area = np.polynomial.legendre.leggauss(4)[1] / 2
    errors = {model: {field: rng.normal(size=(5, 3, 8, 4)) * (j + 1)
                      for j, field in enumerate(audit.FIELDS)} for model in ('baseline', 'residual')}
    sigma = {field: np.array([1., 2., 3.]) if field == 'specific_humidity' else np.array([2.])
             for field in audit.FIELDS}
    historical = {'loss_scales': {f: np.repeat(2., 3) for f in audit.FIELDS}}
    rows = []
    for field in audit.FIELDS:
        for k, pressure in enumerate(levels):
            row = dict(run_id='toy', field=field, pressure_hpa=pressure)
            for model in errors:
                row[f'{model}_rmse'] = np.sqrt(np.mean(np.sum(errors[model][field][:, k] ** 2 * area, axis=-1)))
            rows.append(row)
    legacy_scores = {}
    expected = {}
    for model in errors:
        legacy_scores[model + '_loss'] = 0.
        expected[model] = 0.
        for field in audit.FIELDS:
            effective = np.float32(2 / audit.AMPLITUDES[field])
            e2 = (errors[model][field] / effective) ** 2
            legacy_scores[model + '_loss'] += np.mean(np.sum(
                np.sum(e2 * area, axis=-1) * (levels / levels.sum())[None, :, None], axis=1)) / 7
            normalized = errors[model][field] * audit.AMPLITUDES[field] / sigma[field][None, :, None, None]
            expected[model] += np.mean(np.sum(normalized ** 2 * area, axis=-1)) / 1.25
    statistics = dict(pressure_hpa=levels, scales={key: sigma for key in
                      ('6h_gaussian', '24h_gaussian', '24h_uniform')})
    _, totals, _ = audit.rescore(rows, statistics, historical, {'runs': {'toy': legacy_scores}})
    result = next(r for r in totals if r['recipe'] == '24h_uniform_stats_equal_levels')
    np.testing.assert_allclose([result['baseline_score'], result['residual_score']],
                               [expected['baseline'], expected['residual']], rtol=1e-13)


def test_degenerate_scale_is_rejected_instead_of_silently_floored():
    moments = audit.ChangeMoments(3, np.ones(4))
    moments.add(np.zeros((3, 8, 4)))
    with pytest.raises(ValueError, match='Degenerate scales'):
        moments.std()
