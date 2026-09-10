"""Behavior checks for the batch/LR continuation experiment."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.config import (
    V24IlyaDistributedConfig, V24IlyaTrainConfig, load_training_config,
)
from src.models.mamba.v24_Ilya.training.data import (
    TrainingCursor, V24IlyaTrainingData, advance_cursor, concatenate_batch,
)
from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer, is_mamba_parameter
from src.models.mamba.v24_Ilya.training.frame_data import FrameDataReport


def config(**kwargs):
    return V24IlyaTrainConfig(
        prepared_root=Path('prepared'), anchor_manifest_root=Path('anchors'),
        baseline_checkpoint=Path('baseline'), output_root=Path('output'), run_name='batch-lr',
        architecture=V24IlyaArchitectureConfig(), segment_steps=8, bptt_steps=4,
        ar_tail_k=2, warmup_steps=0, **kwargs,
    )


@pytest.mark.parametrize('spatial,mamba', [(1., .3), (0., 1.)])
def test_group_updates_and_exact_freeze(spatial, mamba):
    cfg = config(spatial_lr_multiplier=spatial, mamba_lr_multiplier=mamba, weight_decay=.1)
    params = {'mesh_interleaved_temporal_r0_s0/~_run_sequence/mamba_block_0': {'w': jnp.ones(2)},
              'mesh_interleaved_temporal_r0_s0/~_run_sequence/layer_norm_0': {'scale': jnp.ones(2)},
              'temporal_residual_head': {'w': jnp.ones(2)}}
    optimizer, _ = build_optimizer(cfg)
    state = optimizer.init(params)
    before = params
    for _ in range(3):
        gradients = jax.tree_util.tree_map(jnp.ones_like, params)
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
    module = 'mesh_interleaved_temporal_r0_s0/~_run_sequence/mamba_block_0'
    delta_m = np.asarray(before[module]['w'] - params[module]['w'])
    delta_s = np.asarray(before['temporal_residual_head']['w'] - params['temporal_residual_head']['w'])
    assert np.all(delta_m > 0)
    if not spatial:
        np.testing.assert_array_equal(delta_s, 0)
    else:
        np.testing.assert_allclose(delta_m / delta_s, mamba / spatial, rtol=.002)
    assert is_mamba_parameter('mesh_interleaved_temporal_r0_s0/~_run_sequence/layer_norm_0')
    assert not is_mamba_parameter('temporal_residual_head')


def test_batch_config_roundtrip_and_group_cursor(tmp_path):
    cfg = config(distributed=V24IlyaDistributedConfig(per_device_batch_size=4),
                 mamba_lr_multiplier=.3)
    p = tmp_path / 'config.json'
    p.write_text(json.dumps(cfg.to_dict()))
    assert load_training_config(p) == cfg
    assert advance_cursor(TrainingCursor(0, 0, 0), cfg, 10) == TrainingCursor(0, 0, 4)
    assert advance_cursor(TrainingCursor(0, 0, 4), cfg, 10) == TrainingCursor(0, 4, 0)
    assert advance_cursor(TrainingCursor(0, 4, 4), cfg, 10) == TrainingCursor(1, 0, 0)


def test_batch_loader_keeps_lanes_and_times_separate():
    cfg = config(distributed=V24IlyaDistributedConfig(per_device_batch_size=4))
    def dataset(value):
        return xr.Dataset({'x': (('batch', 'time', 'lat'), np.full((1, 1, 2), value, dtype=np.float32))},
                          coords={'batch': [0], 'time': [np.timedelta64(6, 'h')], 'lat': [0, 1]})
    def load(segment, offset, config, task, workspace_slot=0):
        value = int(segment[0]) + offset
        return SimpleNamespace(input_frames=(dataset(value), dataset(value+1), dataset(value+2)),
                               static_inputs=dataset(1), truths=(dataset(value+3),),
                               forcings=(dataset(value+4),), raw_anchor_indices=np.arange(value, value+4),
                               data_report=FrameDataReport(2, 3, 4, 1, 128))
    data = SimpleNamespace(segments=tuple(np.arange(i*8, i*8+8) for i in range(10)),
                           validate_cursor=lambda *args: None, load_segment_chunk=load)
    chunk = V24IlyaTrainingData.build_chunk(data, TrainingCursor(), cfg, None)
    np.testing.assert_array_equal(chunk.input_frames[0]['x'][:, 0, 0], [0, 8, 16, 24])
    np.testing.assert_array_equal(chunk.input_frames[0].batch, [0, 1, 2, 3])
    assert chunk.input_frames[0].sizes['time'] == 1
    assert chunk.raw_anchor_indices.shape == (4, 4)
    assert chunk.data_report.bytes_loaded == 512
    chunk.truths[0]['x'].values[0] = -1
    assert np.all(chunk.truths[0]['x'].values[1:] > 0)


def test_batched_bptt_update_equals_mean_of_independent_lane_gradients():
    from graphcast import xarray_jax
    from src.models.mamba.v24_Ilya.training.endpoint_step import (
        V24IlyaTrainingTransforms, make_train_step,
    )
    class Baseline:
        def apply(self, params, state, key, inputs, template, forcing):
            return jax.tree_util.tree_map(jnp.zeros_like, template), state
    class Residual:
        def apply(self, params, state, key, inputs, template, forcing):
            x = xarray_jax.unwrap_data(inputs['x']).mean(axis=1)
            s = .7 * state['s'] + params['a'] * x
            prediction = xr.Dataset({'x': xr.DataArray(params['b'] * s[:, None],
                                      dims=('batch', 'time'), coords=template.coords)})
            return prediction, {'s': s}
    class Loss:
        def apply(self, params, state, key, inputs, target, forcing):
            pred, state = Residual().apply(params, state, key, inputs, target, forcing)
            error = xarray_jax.unwrap_data(pred['x']) - xarray_jax.unwrap_data(target['x'])
            loss = xr.DataArray((error**2).mean(axis=1), dims=('batch',), coords={'batch': target.batch})
            return ((loss, {}), pred), state
    def dataset(values, name='x', hour=0):
        return xr.Dataset({name: (('batch', 'time'), np.asarray(values, np.float32)[:, None])},
                          coords={'batch': np.arange(len(values)), 'time': [np.timedelta64(hour, 'h')]})
    cfg = config(loss_mode='all_steps', distributed=V24IlyaDistributedConfig(per_device_batch_size=4))
    transforms = V24IlyaTrainingTransforms(Baseline(), Residual(), Loss(), Loss())
    optimizer = optax.sgd(.01)
    step = make_train_step(transforms=transforms, optimizer=optimizer, baseline_params={}, baseline_state={},
                           config=cfg, time_step=pd.Timedelta('6h'), input_steps=2)
    params = {'a': jnp.array(.1), 'b': jnp.array(.2)}
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    def evaluate(lanes):
        values = np.asarray(lanes, np.float32)
        frames = tuple(xr.merge([dataset(values+i), dataset(values*0, 'forcing')]) for i in range(3))
        truths = tuple(dataset(values+i+2, hour=6) for i in range(4))
        forcings = tuple(dataset(values*0, 'forcing', 6) for i in range(4))
        return step(params, {'s': jnp.zeros(len(lanes))}, optimizer.init(params), keys,
                    frames, xr.Dataset(), truths, forcings)
    batch = evaluate([1, 2, 3, 4])
    singles = [evaluate([lane]) for lane in [1, 2, 3, 4]]
    for name in params:
        np.testing.assert_allclose(batch[0][name], np.mean([x[0][name] for x in singles]), rtol=1e-5)
    np.testing.assert_allclose(batch[3], np.mean([x[3] for x in singles]), rtol=1e-5)
    np.testing.assert_allclose(batch[1]['s'], np.concatenate([x[1]['s'] for x in singles]), rtol=1e-5)
