import copy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import fresh_data_scale_evaluation as module


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def losses(b, f):
    return {'baseline_per_step': b.mean(axis=0).tolist(),
            'full_per_step': f.mean(axis=0).tolist(),
            'baseline_rollout': float(b.mean()), 'full_rollout': float(f.mean()),
            'improvement_pct_rollout': float(100 * (1 - f.sum() / b.sum()))}


def fixture(tmp_path, shards=8):
    origin = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    times = [(origin + dt.timedelta(days=3*i)).isoformat().replace('+00:00', 'Z') for i in range(122)]
    manifest = save(tmp_path / 'initializations.json', {'schema_version': 1, 'initializations': times})
    identity = {name: {'path': name, 'sha256': name} for name in
                ('config', 'checkpoint', 'baseline', 'data_metadata')}
    identity.update(statistics=[], year=2023, initializations=module.pinned(manifest))
    b = (np.arange(122)[:, None] + 1) * np.ones((122, 40))
    f = b * .9
    results = []
    for i in range(shards):
        indices = list(range(i, 122, shards))
        results.append({'ckpt': 'checkpoint', 'ckpt_in': 'baseline',
            'evaluation_status': 'complete', 'target_steps': 40,
            'eval_mode': 'cold_full', 'eval_feedback': 'full', 'residual_state_init': 'zero',
            'anchor_shard_index': i, 'anchor_shard_count': shards,
            'evaluated_samples': len(indices), 'n_samples': 122,
            'initialization_manifest_sha256': module.pinned(manifest)['sha256'],
            'global_initialization_times': times,
            'original_graphcast_loss': losses(b[indices], f[indices]),
            'original_graphcast_loss_per_initialization': [
                {'initialization_time': times[j], 'baseline_per_step': b[j].tolist(),
                 'full_per_step': f[j].tolist()} for j in indices],
            'fresh_data_scale_identity': identity, 'integrity': {'passed': True}})
    return manifest, results, b, f


def test_legacy_preserves_matched_indices(tmp_path):
    _, results, b, f = fixture(tmp_path)
    result = results[0]
    result.update(evaluated_samples=32, n_samples=32, anchor_shard_count=1,
                  chosen_idx=module.EXACT32_INDICES,
                  original_graphcast_loss=losses(b[:32], f[:32]))
    assert module.validate_result(result, checkpoint='checkpoint', baseline_checkpoint='baseline', year=2022)['passed']
    result['chosen_idx'] = list(reversed(module.EXACT32_INDICES))
    with pytest.raises(ValueError, match='origins'):
        module.validate_result(result, checkpoint='checkpoint', baseline_checkpoint='baseline', year=2022)


def test_merge_weights_initializations_and_requires_all_shards(tmp_path):
    manifest, results, b, f = fixture(tmp_path)
    # Unequal shard sizes and changing losses expose a naive average of shard means.
    for result in results:
        for row in result['original_graphcast_loss_per_initialization']:
            row['full_per_step'] = [x ** .5 for x in row['baseline_per_step']]
        records = result['original_graphcast_loss_per_initialization']
        result['original_graphcast_loss'] = losses(
            np.array([r['baseline_per_step'] for r in records]),
            np.array([r['full_per_step'] for r in records]))
    paths = [save(tmp_path / f'shard_{i:02d}.json', r) for i, r in enumerate(results)]
    report = module.merge(paths, manifest)
    assert report['full_rollout'] == pytest.approx(np.sqrt(b).mean())
    assert report['loss_reduction_pct_own_baseline'] == pytest.approx(100 * (1 - np.sqrt(b).sum() / b.sum()))
    with pytest.raises(ValueError, match='Missing or duplicated'):
        module.merge(paths[:-1], manifest)
    with pytest.raises(ValueError, match='Missing or duplicated'):
        module.merge(paths[:-1] + [paths[0]], manifest)


@pytest.mark.parametrize('damage', ['nonfinite', 'order', 'aggregate', 'checkpoint', 'status', 'identity'])
def test_shard_rejects_invalid_evidence(tmp_path, damage):
    manifest, results, _, _ = fixture(tmp_path)
    result = results[0]
    if damage == 'nonfinite':
        result['original_graphcast_loss_per_initialization'][0]['full_per_step'][0] = float('nan')
    elif damage == 'order':
        result['original_graphcast_loss_per_initialization'].reverse()
    elif damage == 'aggregate':
        result['original_graphcast_loss']['full_rollout'] *= .5
    elif damage == 'checkpoint':
        result['ckpt'] = 'wrong'
    elif damage == 'status':
        result['evaluation_status'] = 'partial'
    else:
        result['initialization_manifest_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        module.validate_result(result, checkpoint='checkpoint', baseline_checkpoint='baseline',
                               year=2023, initialization_manifest=manifest, index=0, shards=8)


def arm_reports(tmp_path):
    manifest, results, b, _ = fixture(tmp_path, shards=1)
    path = save(tmp_path / 'shard.json', results[0])
    base = module.merge([path], manifest)
    arms = {}
    for name, reduction in [('w128_7y', .1), ('w128_40y', .12), ('w512_7y', .15), ('w512_40y', .20)]:
        report = copy.deepcopy(base)
        report['full_loss_matrix'] = (b * (1 - reduction)).tolist()
        report['identity']['checkpoint'] = {'path': name, 'sha256': name}
        arms[name] = report
    return arms, b


def test_paired_primary_gain_and_canonical_baseline_drift(tmp_path):
    arms, b = arm_reports(tmp_path)
    arms['w512_40y']['baseline_loss_matrix'] = (b * 1.1).tolist()
    report = module.comparison_report(arms, tmp_path / 'report', bootstrap_samples=1000)
    assert report['primary_data_gain']['128']['gain_percentage_points'] == pytest.approx(2.)
    assert report['primary_data_gain']['512']['gain_percentage_points'] == pytest.approx(5.)
    assert report['width512_minus_width128_data_gain']['gain_percentage_points'] == pytest.approx(3.)
    assert report['arms']['w512_40y']['loss_reduction_pct'] == pytest.approx(20.)
    assert report['baseline_drift_advisory']['w512_40y']['pooled_pct'] == pytest.approx(10.)
    assert report['primary_data_gain']['128']['simultaneous_95_interval'] == pytest.approx([2., 2.])
    assert (tmp_path / 'report/README.md').is_file()


@pytest.mark.parametrize('damage', ['missing', 'date', 'baseline', 'nonfinite'])
def test_comparison_rejects_unmatched_arms(tmp_path, damage):
    arms, _ = arm_reports(tmp_path)
    if damage == 'missing':
        arms.pop('w512_40y')
    elif damage == 'date':
        arms['w512_40y']['initialization_times'].reverse()
    elif damage == 'baseline':
        arms['w512_40y']['identity']['baseline']['sha256'] = 'different'
    else:
        arms['w512_40y']['full_loss_matrix'][0][0] = float('inf')
    with pytest.raises(ValueError):
        module.comparison_report(arms, tmp_path / 'report', bootstrap_samples=100)


def test_manifest_rejects_duplicate_dates_and_overwrite(tmp_path):
    manifest, _, _, _ = fixture(tmp_path)
    value = module.read(manifest)
    value['initializations'][1] = value['initializations'][0]
    save(manifest, value)
    with pytest.raises(ValueError, match='unique'):
        module.manifest_times(manifest)
    with pytest.raises(FileExistsError):
        module.write(manifest, {})


@pytest.mark.parametrize('version', [2, 3])
def test_evaluation_inputs_pin_selected_zarr_metadata_and_detect_format_change(tmp_path, version):
    data_path = tmp_path / 'weather.zarr'
    data_path.mkdir()
    v2 = {'zarr_consolidated_format': 1, 'metadata': {'lat/.zarray': {}}}
    v3 = {'zarr_format': 3, 'consolidated_metadata': {'kind': 'inline', 'metadata': {'lat': {}}}}
    save(data_path / ('.zmetadata' if version == 2 else 'zarr.json'), v2 if version == 2 else v3)
    for name in ('config', 'checkpoint', 'baseline', 'mean_by_level.nc', 'stddev_by_level.nc', 'diffs_stddev_by_level.nc'):
        (tmp_path / name).write_bytes(b'fixed input')
    source = {'data': {'baseline_checkpoint': str(tmp_path / 'baseline'), 'stats_dir': str(tmp_path)}}
    identity = module.freeze_inputs(tmp_path / 'config', tmp_path / 'checkpoint', source, data_path, 2022)
    assert Path(identity['data_metadata']['path']).name == ('.zmetadata' if version == 2 else 'zarr.json')
    module.verify_inputs(identity, data_path)
    if version == 2:
        save(data_path / 'zarr.json', v3)
    else:
        v3['changed'] = True
        save(data_path / 'zarr.json', v3)
    with pytest.raises(ValueError, match='metadata changed'):
        module.verify_inputs(identity, data_path)
