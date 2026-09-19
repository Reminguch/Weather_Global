"""Full-field loss routing and recurrent VJP parity, independent of SHT numerics."""
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.config import (
    V24IlyaDistributedConfig, V24IlyaTrainConfig, load_training_config,
)
from src.models.mamba.v24_Ilya.training.endpoint_step import (
    V24IlyaTrainingTransforms, make_bptt_objective, make_train_step,
    make_validation_step, make_data_parallel_train_step, supervised_step,
    supervision_target,
)
from src.models.mamba.v24_Ilya.training.data_parallel import replicate_tree


def config(kind='amse', batch=4, mode='all_steps'):
    extra = dict(supervised_horizons=(1, 3), supervised_weights=(1., 3.)) if mode == 'sparse_steps' else {}
    return V24IlyaTrainConfig(
        prepared_root=Path('prepared'), anchor_manifest_root=Path('anchors'),
        baseline_checkpoint=Path('baseline'), output_root=Path('output'), run_name='amse-test',
        architecture=V24IlyaArchitectureConfig(width=4), segment_steps=4,
        bptt_steps=4, ar_tail_k=2, feedback_mode='closed_loop_sg', loss_mode=mode,
        loss_kind=kind, distributed=V24IlyaDistributedConfig(per_device_batch_size=batch), **extra,
    )


def dataset(value, batch=4, name='x', hour=0):
    values = jnp.broadcast_to(jnp.asarray(value, jnp.float32), (batch, 1, 2))
    return xr.Dataset({name: (('batch', 'time', 'mode'), values)},
                      coords={'batch': np.arange(batch), 'time': [np.timedelta64(hour, 'h')], 'mode': [0, 1]})


def arguments(cfg):
    n = cfg.distributed.per_device_batch_size
    frames = tuple(xr.merge([dataset([v, .5*v], n), dataset([0., 0.], n, 'forcing')]) for v in (1., 2., 3.))
    truths = tuple(dataset([i+3., .5*(i+3.)], n, hour=6) for i in cfg.supervised_step_indices)
    return (jax.random.split(jax.random.PRNGKey(0), 4), frames, xr.Dataset(), truths,
            tuple(dataset([0., 0.], n, 'forcing', 6) for _ in range(4)))


class Baseline:
    def __init__(self):
        self.calls = 0

    def apply(self, params, state, key, inputs, template, forcing):
        self.calls += 1
        prediction = jax.tree_util.tree_map(lambda t: jnp.broadcast_to(jnp.asarray([1.25, -.5]), t.shape), template)
        return prediction, state


class Residual:
    def apply(self, params, state, key, inputs, template, forcing):
        value = jnp.mean(xarray_jax.unwrap_data(inputs['x']), axis=(1, 2))
        next_state = {'s': .7 * state['s'] + params['a'] * value}
        prediction = jax.tree_util.tree_map(lambda t: jnp.ones_like(t) * params['b'] * next_state['s'][:, None, None], template)
        return prediction, next_state


class ForbiddenResidualLoss:
    def apply(self, *args):
        raise AssertionError('Full-field scoring must bypass the residual-only loss wrapper')


def coefficient_amse(f, y):
    """AMSE for two real coefficients at one degree, separately per batch."""
    pf, py = jnp.mean(f**2, axis=-1) + 1e-6, jnp.mean(y**2, axis=-1) + 1e-6
    cross = jnp.mean(f*y, axis=-1) + 1e-6
    coherence = cross / jnp.sqrt(pf*py)
    return (jnp.sqrt(pf)-jnp.sqrt(py))**2 + 2*jnp.maximum(pf, py)*(1-coherence)


def full_loss(prediction, truth):
    f, y = (xarray_jax.unwrap_data(d['x'])[:, 0, :] for d in (prediction, truth))
    assert f.dtype == y.dtype == jnp.float32
    return coefficient_amse(f, y), {}


def transforms():
    residual, legacy = Residual(), ForbiddenResidualLoss()
    return V24IlyaTrainingTransforms(Baseline(), residual, legacy, legacy, full_loss)


def reference(params, initial_state, cfg, args):
    _, input_frames, _, truths, _ = args
    frames = [xarray_jax.unwrap_data(frame['x'])[:, 0, :] for frame in input_frames]
    state = initial_state['s']
    components = []
    positions = {step: pos for pos, step in enumerate(cfg.supervised_step_indices)}
    for i in range(4):
        state = .7*state + params['a'] * jnp.mean(jnp.stack([frames[i], frames[i+1]]), axis=(0, 2))
        residual = params['b'] * state[:, None] * jnp.ones((state.shape[0], 2))
        prediction = jnp.asarray([1.25, -.5]) + residual
        if i in positions:
            truth = xarray_jax.unwrap_data(truths[positions[i]]['x'])[:, 0, :]
            components.append(jnp.mean(coefficient_amse(prediction, truth)))
        if i >= cfg.truth_prefix_steps - 1 and i < 3:
            frames.append(jax.lax.stop_gradient(prediction))
    return jnp.dot(jnp.stack(components), jnp.asarray(cfg.normalized_supervised_weights)), {'s': state}


def assert_trees_close(actual, expected):
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('kind', ['mse', 'mse_fp32', 'amse'])
def test_loss_config_roundtrip(tmp_path, kind):
    cfg = config(kind)
    if kind == 'amse':
        cfg = replace(cfg, amse_lmax=17, amse_power_epsilon=1e-10)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg.to_dict()))
    assert load_training_config(path) == cfg
    if kind == 'mse':
        assert 'loss_kind' not in cfg.to_dict()['objective']


def test_cached_full_field_loss_fails_explicitly():
    from src.models.mamba.v24_Ilya.training.cached_config import CachedExecutionConfig, CachedTrainingConfig
    with pytest.raises(ValueError, match='only supports loss_kind'):
        CachedTrainingConfig(config(), CachedExecutionConfig(Path('cache')), Path('cfg.json'))


@pytest.mark.parametrize('mode', ['last_step', 'all_steps', 'sparse_steps'])
def test_full_field_recurrent_reverse_and_streaming_match_direct_ad(mode):
    cfg, model = config(mode=mode), transforms()
    args = arguments(cfg)
    params, state = {'a': jnp.asarray(.11), 'b': jnp.asarray(.23)}, {'s': jnp.arange(4, dtype=jnp.float32)/10}
    kwargs = dict(transforms=model, baseline_params={}, baseline_state={}, config=cfg,
                  time_step=pd.Timedelta('6h'), input_steps=2)
    objective = make_bptt_objective(**kwargs)
    actual, grad = jax.value_and_grad(lambda p: objective(p, state, *args), has_aux=True)(params)
    expected, expected_grad = jax.value_and_grad(lambda p: reference(p, state, cfg, args), has_aux=True)(params)
    assert_trees_close(actual, expected)
    assert_trees_close(grad, expected_grad)
    # Baseline is evaluated in forward only; the recurrent reverse rematerializes just residuals.
    assert model.baseline_predict.calls == len(set(cfg.supervised_step_indices) | {1, 2})
    validation = make_validation_step(**kwargs)
    value, last_state, _ = validation(params, state, *args)
    assert_trees_close((value, last_state), expected)
    optimizer = optax.sgd(.01)
    train = make_train_step(**kwargs, optimizer=optimizer)
    updated, last_state, _, value, _, _ = train(params, state, optimizer.init(params), *args)
    expected_params = jax.tree_util.tree_map(lambda p, g: p-.01*g, params, expected_grad)
    assert_trees_close((updated, last_state, value), (expected_params, expected[1], expected[0]))


def test_full_field_inputs_are_truth_and_baseline_plus_residual():
    model = transforms()
    args = arguments(config())
    _, frames, _, truths, forcings = args
    baseline = dataset([1.25, -.5], hour=6)
    target = supervision_target(model, truths[0], baseline)
    params, state = {'a': jnp.asarray(.1), 'b': jnp.asarray(.2)}, {'s': jnp.ones(4)}
    value, residual, _ = supervised_step(model, params, state, args[0][0], frames[0], target, forcings[0])
    f = xarray_jax.unwrap_data(residual['x'])[:, 0, :] + jnp.asarray([1.25, -.5])
    y = xarray_jax.unwrap_data(truths[0]['x'])[:, 0, :]
    np.testing.assert_allclose(value, jnp.mean(coefficient_amse(f, y)), rtol=1e-6)
    wrong = jnp.mean(coefficient_amse(f-jnp.asarray([1.25, -.5]), y-jnp.asarray([1.25, -.5])))
    assert not np.isclose(value, wrong)


def test_missing_full_field_hook_is_rejected():
    with pytest.raises(ValueError, match='loss_kind and full_field_loss'):
        make_bptt_objective(transforms=replace(transforms(), full_field_loss=None),
                            baseline_params={}, baseline_state={}, config=config(),
                            time_step=pd.Timedelta('6h'), input_steps=2)


def test_single_replica_fused_forward_and_reverse_match_direct_ad():
    # Executes the exact pmap forward/tape path even when only one CPU is available.
    cfg, model = config(batch=1), transforms()
    args = arguments(cfg)
    params, state = {'a': jnp.asarray(.11), 'b': jnp.asarray(.23)}, {'s': jnp.asarray([.3])}
    (expected_loss, expected_state), grad = jax.value_and_grad(lambda p: reference(p, state, cfg, args), has_aux=True)(params)
    devices = (jax.local_devices()[0],)
    opt = optax.sgd(.01)
    train = make_data_parallel_train_step(transforms=model, optimizer=opt, baseline_params={},
        baseline_state={}, config=cfg, time_step=pd.Timedelta('6h'), input_steps=2, devices=devices)
    keys, frames, static, truths, forcings = args
    result = train(replicate_tree(params, devices), replicate_tree(state, devices),
                   replicate_tree(opt.init(params), devices), keys[None, ...], (frames,), (static,), (truths,), (forcings,))
    expected_params = jax.tree_util.tree_map(lambda p, g: p-.01*g, params, grad)
    assert_trees_close(result[0], replicate_tree(expected_params, devices))
    assert_trees_close(result[1], replicate_tree(expected_state, devices))
    np.testing.assert_allclose(result[3], expected_loss, rtol=3e-5)
