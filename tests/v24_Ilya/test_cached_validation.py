from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr
from graphcast import losses, normalization, xarray_jax

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.metrics import ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS, V24IlyaMetricAccumulator
from src.models.mamba.v24_Ilya.training.cached_data import CachedValidationStep
from src.models.mamba.v24_Ilya.training.cached_stepwise import CachedResidualTransforms, CachedStepwiseTrainer
from src.models.mamba.v24_Ilya.training.cached_validation import make_online_validation_callback, run_cached_validation
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig
from src.models.mamba.v24_Ilya.training.validation import validation_step_keys


def _weather(value):
    # Unequal errors at poles/levels make simplified weighting fail this test.
    surface = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2) / 3 + value
    pressure = np.arange(12, dtype=np.float32).reshape(1, 1, 2, 3, 2) / 5 + value
    return xr.Dataset({
        "2m_temperature": (("batch", "time", "lat", "lon"), surface),
        "mean_sea_level_pressure": (("batch", "time", "lat", "lon"), surface * 2),
        "temperature": (("batch", "time", "level", "lat", "lon"), pressure),
    }, coords={"batch": [0], "time": np.array([6], dtype="timedelta64[h]"),
               "lat": [-90., 0., 90.], "lon": [0., 180.], "level": [500, 1000]})


@pytest.fixture
def validation_case():
    config = V24IlyaTrainConfig(
        prepared_root=Path("prepared"), anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline.npz"), output_root=Path("output"), run_name="val",
        architecture=V24IlyaArchitectureConfig(), segment_steps=8, bptt_steps=4,
        ar_tail_k=2, loss_mode="all_steps", precision="bf16",
    )
    scales = xr.Dataset({"2m_temperature": 2., "mean_sea_level_pressure": 10.,
                         "temperature": (("level",), np.array([4., 8.], np.float32))},
                        coords={"level": [500, 1000]}).astype(np.float32)

    def predict(inputs, target, forcing):
        del forcing
        gain = hk.get_parameter("gain", (), jnp.float32, init=hk.initializers.Constant(.15))
        state = hk.get_state("memory", (), jnp.float32, init=jnp.zeros)
        hk.set_state("memory", state + 1)
        value = gain * jnp.mean(xarray_jax.unwrap_data(inputs["2m_temperature"])) + .1 * state
        return xr.Dataset({name: xarray_jax.DataArray(jnp.ones(var.shape, jnp.float32) * value,
                            dims=var.dims, coords=var.coords) for name, var in target.data_vars.items()})

    def supervised(inputs, target, forcing):
        residual = predict(inputs, target, forcing)
        pred_normalized = normalization.normalize(residual, scales, None).astype(jnp.bfloat16)
        target_normalized = normalization.normalize(target, scales, None).astype(jnp.bfloat16)
        loss, diagnostics = losses.weighted_mse_per_level(
            pred_normalized, target_normalized,
            per_variable_weights={name: weight for name, weight in ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS.items()
                                  if name in target.data_vars},
        )
        return (loss.astype(jnp.float32), diagnostics), residual

    transforms = CachedResidualTransforms(hk.transform_with_state(predict), hk.transform_with_state(supervised))
    params, state = transforms.residual_loss_and_predictions.init(jax.random.PRNGKey(12), _weather(1.), _weather(2.), xr.Dataset())
    trainer = CachedStepwiseTrainer(config=config, optimizer=optax.sgd(.1), transforms=transforms)
    closed = []

    def rows(segment_id, offset):
        try:
            for index in range(4):
                value = 1 + segment_id + offset / 4 + index / 4
                truth, baseline = _weather(value + 3), _weather(value)
                yield CachedValidationStep(index, _weather(value), truth - baseline, xr.Dataset(), truth, baseline)
        finally:
            closed.append((segment_id, offset))

    training_data = SimpleNamespace(
        validation_segments=(np.arange(8), np.arange(8, 16)),
        fingerprint_validation_subset=lambda ids: "segments:" + ",".join(str(int(i)) for i in ids),
    )
    cached = SimpleNamespace(
        common=config, training_data=training_data,
        context=(config, None, None, {"diffs_stddev_by_level": scales}, training_data, None),
        manifest={"coordinates": {"lat": [-90., 0., 90.]}, "manifest_sha256": "fixed-test-cache"},
        iter_validation_steps=rows,
        load_validation_chunk=lambda *_args, **_kwargs: pytest.fail("Validation retained a whole owned chunk"),
    )
    return SimpleNamespace(config=config, cached=cached, trainer=trainer, params=params, state=state, closed=closed, scales=scales)


def _run(case, **overrides):
    arguments = dict(train_step=case.trainer, cached_data=case.cached, config=case.config,
                     params=case.params, zero_state=case.state, step=500)
    arguments.update(overrides)
    return run_cached_validation(**arguments)


def test_streamed_loss_matches_chunk_evaluation_and_exact_physical_metric(validation_case):
    case = validation_case
    result = _run(case)
    expected = V24IlyaMetricAccumulator(4, np.array([-90., 0., 90.]),
        diffs_stddev_by_level=case.scales, store_spatial_bias=False)
    chunk_losses, component_losses = [], []
    for segment_id in range(2):
        state = jax.tree_util.tree_map(jnp.zeros_like, case.state)
        for chunk_index, offset in enumerate((0, 4)):
            rows = list(case.cached.iter_validation_steps(segment_id, offset))
            keys = validation_step_keys(seed=case.config.seed, segment_id=segment_id,
                                        chunk_index=chunk_index, bptt_steps=4)
            loss, state, components, predictions = case.trainer.evaluate(
                case.params, state, keys, tuple(row.inputs for row in rows),
                tuple(row.target for row in rows), tuple(row.forcing for row in rows),
            )
            chunk_losses.append(float(loss))
            component_losses.append(np.asarray(components))
            for row, prediction in zip(rows, predictions):
                pred_host = xr.Dataset({name: (variable.dims, np.asarray(xarray_jax.unwrap_data(variable)))
                                       for name, variable in prediction.data_vars.items()}, coords=prediction.coords)
                expected.update_step(row.step_index, row.truth, row.baseline, row.baseline + pred_host)
    assert result["loss"] == pytest.approx(np.mean(chunk_losses), rel=1e-7)
    np.testing.assert_allclose(list(result["loss_by_horizon"].values()), np.mean(component_losses, axis=0), rtol=1e-7)
    assert result["original_graphcast_loss"] == expected.finalize()["original_graphcast_loss"]
    assert result["num_chunks"] == 4 and result["num_anchors"] == 16
    assert result["segment_ids"] == [0, 1]
    # Legacy BF16 objective and raw-field FP32 metric are intentionally reported separately.
    assert abs(result["loss"] - result["original_graphcast_loss"]["full_rollout"]) > 1e-6


def test_validation_resets_segments_carries_chunks_and_preserves_rng(validation_case):
    case = validation_case
    state_before = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), case.state)
    params_before = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), case.params)
    rng = jax.random.PRNGKey(99)
    rng_before = np.array(rng)
    incoming, keys_seen = [], []
    evaluate = case.trainer.evaluate_step

    def recording(params, state, key, *args):
        incoming.append(float(next(iter(jax.tree_util.tree_leaves(state)))))
        keys_seen.append(np.array(key))
        return evaluate(params, state, key, *args)

    case.trainer.evaluate_step = recording
    first = _run(case)
    assert incoming == list(range(8)) * 2
    assert case.closed == [(0, 0), (0, 4), (1, 0), (1, 4)]
    expected_keys = [key for segment in range(2) for chunk in range(2)
                     for key in np.asarray(validation_step_keys(seed=case.config.seed, segment_id=segment,
                                                                 chunk_index=chunk, bptt_steps=4))]
    np.testing.assert_array_equal(keys_seen, expected_keys)
    second = _run(case, step=1000)
    assert first["loss"] == second["loss"]
    assert first["original_graphcast_loss"] == second["original_graphcast_loss"]
    np.testing.assert_array_equal(rng, rng_before)
    for before, after in zip(jax.tree_util.tree_leaves(state_before), jax.tree_util.tree_leaves(case.state)):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(jax.tree_util.tree_leaves(params_before), jax.tree_util.tree_leaves(case.params)):
        np.testing.assert_array_equal(before, after)


def test_online_callback_uses_same_evaluator_and_rejects_subset_drift(validation_case):
    case = validation_case
    callback = make_online_validation_callback(train_step=case.trainer, cached_data=case.cached)
    arguments = dict(residual_params=case.params, zero_residual_state=case.state, config=case.config,
                     segment_ids=np.arange(2), step=500, role="fixed_checkpoint",
                     training_data=case.cached.training_data)
    result = callback(**arguments)
    direct = _run(case)
    for field in ("loss", "loss_by_horizon", "original_graphcast_loss", "subset_fingerprint", "validation_protocol"):
        assert result[field] == direct[field]
    with pytest.raises(ValueError, match="all cached"):
        callback(**{**arguments, "segment_ids": np.array([0])})
    with pytest.raises(ValueError, match="seed differ"):
        callback(**{**arguments, "config": dataclasses.replace(case.config, seed=123)})


def test_nonfinite_prediction_closes_stream_without_mutating_state(validation_case):
    case = validation_case
    original = case.trainer.evaluate_step

    def bad_prediction(*args):
        loss, state, prediction = original(*args)
        return loss, state, prediction * np.float32(np.nan)

    case.trainer.evaluate_step = bad_prediction
    with pytest.raises(ValueError, match="Non-finite original GraphCast"):
        _run(case)
    assert case.closed == [(0, 0)]
