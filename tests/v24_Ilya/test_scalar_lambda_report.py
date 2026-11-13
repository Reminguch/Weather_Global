import json

import numpy as np
import pytest

from scripts.analyze_models.report_v24_scalar_lambda import build_if_complete, pooled_power_score
from scripts.analyze_models import report_v24_scalar_lambda as report_module
from scripts.experiments.run_v24_mgda import pinned, seal
from scripts.analyze_models.comparison_policy import POLICY, PRESENTATION_SUFFIX


def test_macro_pooling_uses_truth_mask_and_valid_zero_forecast():
    truth = np.asarray([[[1., 2.], [0., 4.]], [[2., 0.], [3., 4.]]])
    prediction = np.stack([truth, .5 * truth, np.zeros_like(truth)], axis=1)
    np.testing.assert_allclose(pooled_power_score(prediction, truth), [0., .5, 1.])


def test_incomplete_sweep_is_not_published(tmp_path):
    result = build_if_complete(tmp_path)
    assert result['status'] == 'pending'
    assert not (tmp_path / 'report/complete.json').exists()


@pytest.mark.parametrize('exclude_rejected', [False, True])
def test_complete_synthetic_sweep_pools_and_compares_all_arms(tmp_path, monkeypatch, exclude_rejected):
    from datetime import datetime, timedelta
    if exclude_rejected:
        tmp_path = tmp_path / POLICY['campaign']
        tmp_path.mkdir()
        (tmp_path / 'report').mkdir()
        seal(tmp_path / 'report/complete.json', {'status': 'complete', 'archived': True})
        archived = (tmp_path / 'report/complete.json').read_bytes()
    tasks = []
    dates = [(datetime(2023, 1, 1) + timedelta(days=3 * i)).strftime('%Y%m%dT%H%M%SZ') for i in range(122)]
    for family, prefix in [('full_gc', 'gcft'), ('gc_mamba', 'gcm')]:
        for weight in (0., .25, .5, .75, 1.):
            name = f'{prefix}_s22_l{round(weight * 100):03d}'
            task = {'id': name, 'logical_id': name, 'model_family': family, 'lambda': weight}
            if exclude_rejected and family == 'full_gc':
                task['id'] = name + '_profile' if weight == .5 else name
                task['learning_rate'] = 1e-5 if weight else 3e-7
            tasks.append(task)
            if exclude_rejected and family == 'full_gc' and weight:
                # Excluded arms must neither block publication nor need source files.
                continue
            directory = tmp_path / 'smoothing' / name
            directory.mkdir(parents=True)
            losses = np.ones((122, 40, 2, 3))
            losses[:, :, 1] = .9 if family == 'full_gc' else .8
            powers = np.ones((40, 2, 3, 2, 4))
            powers[:, 1] = .75
            path = directory / 'annual_pooled.npz'
            np.savez_compressed(path, metadata=json.dumps({'initialization_ids': dates, 'layout': {'channels': ['a', 'b']}}),
                                per_initialization_exact_gc_loss=losses,
                                truth_power=np.ones((40, 2, 4)), prediction_power=powers)
            seal(directory / 'report.json', {'pooled_statistics': pinned(path)})
    seal(tmp_path / 'selected_arms.json', {'tasks': tasks})
    monkeypatch.setattr(report_module, 'cadence_block_counts', lambda times, **kwargs: np.ones((20, len(times))))
    monkeypatch.setattr(report_module, '_plot', lambda *args: None)
    result = build_if_complete(tmp_path)
    assert result['status'] == 'complete'
    assert len(result['records']) == (21 if exclude_rejected else 33)
    assert len(result['contrasts']) == (3 if exclude_rejected else 15)
    np.testing.assert_allclose([c['mamba_advantage_percentage_points'] for c in result['contrasts']], 10.)
    np.testing.assert_allclose([r['spherical_power_error'] for r in result['records'][3:]], .25)
    report_name = 'report' + (PRESENTATION_SUFFIX if exclude_rejected else '')
    assert (tmp_path / report_name / 'tradeoff.csv').exists()
    if exclude_rejected:
        assert len(result['excluded_models']) == 4
        assert (tmp_path / 'report/complete.json').read_bytes() == archived
