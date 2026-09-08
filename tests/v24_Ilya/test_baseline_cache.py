from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya import baseline_cache as cache


def frame(u, f=None):
    variables = {'u': (('batch', 'time', 'lat', 'lon'), np.full((1, 1, 1, 2), u, np.float32))}
    if f is not None:
        variables['f'] = (('batch', 'time', 'lat', 'lon'), np.full((1, 1, 1, 2), f, np.float32))
    return xr.Dataset(variables, coords={'batch': [0], 'time': np.array([6], dtype='timedelta64[h]'), 'lat': [0.], 'lon': [0., 1.]})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = SimpleNamespace(segment_steps=12, bptt_steps=6, truth_prefix_steps=2)
    times = np.arange(50) * np.timedelta64(6, 'h') + np.datetime64('2020-01-01T00')
    data = SimpleNamespace(
        store=SimpleNamespace(time=SimpleNamespace(values=times)),
        time_step=pd.Timedelta('6h'), anchor_indices=np.arange(1, 41),
        train_split=np.arange(25), val_split=np.arange(25, 40),
        segments=(np.arange(12), np.arange(12, 24)),
        validation_segments=(np.arange(25, 37),),
    )
    chunks, excluded = cache.enumerate_chunks(data, config)
    compatibility = {'bptt_steps': 6, 'truth_prefix_steps': 2}
    manifest = {
        'format': cache.FORMAT, 'compatibility': compatibility,
        'compatibility_sha256': cache.digest(compatibility),
        'variables': {'u': {'dims': ['lat', 'lon'], 'shape': [1, 2], 'dtype': 'float32'}},
        'coordinates': {'lat': [0.], 'lon': [0., 1.]},
        'chunks': cache.select_chunks(chunks, None, 1),
        'num_shards': 1, 'excluded_tail_anchors': excluded,
        'full_chunk_count': len(chunks), 'partial': False,
    }
    manifest['manifest_sha256'] = cache.digest(manifest)

    def load(context, item):
        start = item['raw_anchor_indices'][0]
        return SimpleNamespace(
            input_frames=tuple(frame(start + i, start + i + 100) for i in range(3)),
            static_inputs=xr.Dataset({'land': (('batch', 'lat', 'lon'), np.ones((1, 1, 2), np.float32))}),
            truths=tuple(frame(start + i + 2) for i in range(6)),
            forcings=tuple(frame(0, start + i + 102).drop_vars('u') for i in range(6)),
            raw_anchor_indices=np.array(item['raw_anchor_indices']),
        )

    def predict(inputs, truth, forcing):
        result = truth.copy(deep=True)
        result['u'].data[:] = inputs['u'].isel(time=-1).values[:, None] + 10
        return result
    monkeypatch.setattr(cache, 'make_predictor', lambda context: predict)
    monkeypatch.setattr(cache, 'load_chunk', load)
    context = (config, None, None, None, data, None)
    return tmp_path, context, manifest, load, predict


def test_enumeration_and_representative_shards(setup):
    _, context, manifest, _, _ = setup
    assert manifest['excluded_tail_anchors']['train']['count'] == 1
    assert manifest['excluded_tail_anchors']['val']['count'] == 3
    assert len(manifest['chunks']) == 6
    chosen = cache.select_chunks(manifest['chunks'], 4, 2)
    assert [c['id'] for c in chosen] == [0, 3, 4, 5]
    assert [c['shard'] for c in chosen] == [0, 0, 1, 1]
    raw = manifest['chunks'][0]['raw_anchor_indices']
    expected = context[4].store.time.values[raw[0] + 1]
    assert np.datetime64(manifest['chunks'][0]['target_timestamps'][0]) == expected


def test_transition_forcing_alignment_and_chunk_restart(setup):
    _, context, manifest, load, predict = setup
    for item in manifest['chunks'][:2]:
        chunk = load(context, item)
        rows = list(cache.iter_trajectory(chunk, predict, 2, context[4].time_step))
        start = item['raw_anchor_indices'][0]
        assert [float(inputs.u.isel(time=-1).mean()) for _, inputs, _ in rows] == [start + 1, start + 2, start + 12, start + 22, start + 32, start + 42]
        for i, inputs, _ in rows:
            assert float(inputs.f.isel(time=-1).mean()) == start + i + 101
        np.testing.assert_array_equal(rows[2][1].time.values, np.array([0, 6], dtype='timedelta64[h]'))


def test_write_read_resume_and_corruption(setup, monkeypatch):
    root, context, manifest, load, predict = setup
    cache.generate_shard(context, manifest, root, 0)
    ready = cache.verify_cache(root)
    assert ready['predictions'] == 36
    reader = cache.BaselineCacheReader(root, manifest['compatibility_sha256'])
    for item in manifest['chunks']:
        chunk = load(context, item)
        online = list(cache.iter_trajectory(chunk, predict, 2, context[4].time_step))
        cached = list(reader.iter_chunk(item['id'], chunk, context[4].time_step))
        for (_, inputs, baseline), (saved_inputs, saved_baseline, target, forcing) in zip(online, cached):
            xr.testing.assert_equal(inputs, saved_inputs)
            xr.testing.assert_equal(baseline, saved_baseline)
        for i, (_, baseline, target, _) in enumerate(cached):
            xr.testing.assert_equal(target, chunk.truths[i] - baseline)
    monkeypatch.setattr(cache, 'make_predictor', lambda _: pytest.fail('resume initialized model'))
    cache.generate_shard(context, manifest, root, 0, resume=True)
    with pytest.raises(ValueError, match='Incompatible'):
        cache.BaselineCacheReader(root, 'wrong')
    file = root / 'shard_000/chunk_000000/u.npy'
    arr = np.load(file, mmap_mode='r+')
    arr[0] += 1
    arr.flush()
    with pytest.raises(ValueError, match='Checksum'):
        cache.verify_cache(root)
    with pytest.raises(ValueError, match='Checksum'):
        cache.generate_shard(context, manifest, root, 0, resume=True)


def test_interrupted_shard_is_rebuilt_only_on_resume(setup):
    root, context, manifest, _, _ = setup
    incomplete = root / '.shard_000.incomplete'
    incomplete.mkdir()
    (incomplete / 'partial.npy').write_bytes(b'broken')
    with pytest.raises(FileExistsError, match='resume'):
        cache.generate_shard(context, manifest, root, 0)
    cache.generate_shard(context, manifest, root, 0, resume=True)
    assert not incomplete.exists()
    assert not (root / 'shard_000/partial.npy').exists()
    cache.verify_cache(root)


def test_incompatible_manifest_and_missing_chunk(setup):
    root, context, manifest, _, _ = setup
    cache.ensure_manifest(root, manifest)
    changed = {**manifest, 'partial': True}
    with pytest.raises(ValueError, match='Incompatible'):
        cache.ensure_manifest(root, changed)
    with pytest.raises(ValueError, match='fingerprint'):
        cache.validate_manifest(changed)
    with pytest.raises(FileNotFoundError):
        cache.verify_cache(root)


def test_reject_corrected_feedback_before_loading_data(tmp_path):
    source = Path('configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json')
    config = json.loads(source.read_text())
    config['sequence']['feedback_mode'] = 'closed_loop_sg'
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='feedback_mode'):
        cache.load_context(path)


def test_pilot_online_parity_and_recovery(setup, monkeypatch):
    from graphcast import xarray_jax
    from src.models.mamba.v24_Ilya.training import endpoint_step
    root, context, full_manifest, load, predict = setup
    config, _, _, _, data, _ = context
    config.precision = 'fp32'
    manifest = {**full_manifest, 'partial': True,
                'chunks': cache.select_chunks(full_manifest['chunks'], 4, 1)}
    manifest.pop('manifest_sha256')
    manifest['manifest_sha256'] = cache.digest(manifest)

    def batch(*, indices, **kwargs):
        start = indices[0]
        inputs = xr.concat([frame(start, start + 100), frame(start + 1, start + 101)], dim='time')
        inputs = inputs.assign_coords(time=np.array([-6, 0], dtype='timedelta64[h]'))
        inputs['land'] = (('batch', 'lat', 'lon'), np.ones((1, 1, 2), np.float32))
        return inputs, frame(start + 2), frame(0, start + 102).drop_vars('u')
    data.store.build_batch_from_indices = batch

    class Transform:
        def init(self, *args):
            return {}, {}

        def apply(self, params, state, key, inputs, truth, forcing):
            values = xarray_jax.unwrap_data(inputs.u)[:, -1:, :, :] + 10
            ds = xr.Dataset({'u': xarray_jax.DataArray(values, dims=truth.u.dims, coords=truth.u.coords)})
            return ds, {}
    monkeypatch.setattr(endpoint_step, 'build_training_transforms',
                        lambda *args: SimpleNamespace(baseline_predict=Transform()))
    stats = {'stddev_by_level': xr.Dataset({'u': 1., 'f': 1., 'land': 1.}),
             'diffs_stddev_by_level': xr.Dataset({'u': 1.})}
    context = (config, SimpleNamespace(params={}), None, stats, data, None)
    cache.generate_shard(context, manifest, root, 0, parity=True)
    report = json.loads((root / 'pilot_report.json').read_text())
    assert report['passed']
    assert report['chunks'] == 4
    assert all(v == 0 for v in report['max_normalized_errors'].values())
    with pytest.raises(ValueError, match='partial|subset'):
        cache.verify_cache(root)
    with pytest.raises(FileNotFoundError):
        cache.BaselineCacheReader(root, manifest['compatibility_sha256'])
    # Crash after shard publication but before parity-report publication.
    (root / 'pilot_report.json').unlink()
    cache.generate_shard(context, manifest, root, 0, resume=True, parity=True)
    assert json.loads((root / 'pilot_report.json').read_text())['passed']


def test_production_budget_and_slurm_submission():
    from scripts.preprocessing.submit_v24_baseline_production import memory_gib, production_budget
    assert memory_gib('33554432K') == 32
    report = {'peak_host_gib': 20, 'initialization_seconds': 10,
              'compilation_seconds_estimate': 100, 'steady_chunk_seconds': 50}
    assert production_budget(report, 61) == (32, 90)
    assert production_budget(report, 61, accounting_peak=40) == (64, 90)


def test_submission_chain_is_gated_and_idempotent(setup, monkeypatch):
    from scripts.preprocessing import submit_v24_baseline_production as launcher
    root, context, manifest, _, _ = setup
    monkeypatch.chdir(root)
    pilot = root / 'pilot'
    pilot.mkdir()
    report = {'passed': True, 'chunks': 4, 'max_normalized_errors': {'inputs': 0., 'predictions': 0., 'residuals': 0.},
              'compatibility_sha256': manifest['compatibility_sha256'],
              'manifest_sha256': manifest['manifest_sha256'],
              'peak_host_gib': 20, 'initialization_seconds': 10,
              'compilation_seconds_estimate': 100, 'steady_chunk_seconds': 50}
    cache.atomic_json(pilot / 'manifest.json', manifest)
    cache.atomic_json(pilot / 'pilot_report.json', report)
    manifest['prediction_bytes'] = 1000
    monkeypatch.setattr(launcher, 'load_context', lambda _: context)
    monkeypatch.setattr(launcher, 'build_manifest', lambda *args: manifest)
    cache.atomic_json(pilot / 'manifest.json', manifest)
    monkeypatch.setattr(launcher, 'verify_cache', lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='123|COMPLETED|\n123.batch|COMPLETED|33554432K\n'))
    calls = []
    monkeypatch.setattr(launcher, 'submit', lambda args: calls.append(args) or str(200 + len(calls)))
    monkeypatch.setattr('sys.argv', ['submit', '--config', 'config.json', '--output-root', str(root / 'production'), '--pilot-root', str(pilot), '--pilot-job-id', '123'])
    launcher.main()
    assert '--array=0-7' in calls[0]
    assert '--mem=48G' in calls[0]
    assert '--dependency=afterok:201' in calls[1]
    launcher.main()
    assert len(calls) == 2  # No duplicate production or verification submissions.
    report['passed'] = False
    cache.atomic_json(pilot / 'pilot_report.json', report)
    with pytest.raises(ValueError, match='did not pass'):
        launcher.main()
    assert len(calls) == 2
