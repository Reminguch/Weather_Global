from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
for path in (ROOT, GRAPHCAST_LOCAL):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.models.graphcast.training.core.eval import run_eval  # noqa: E402
from src.models.graphcast.training.core.eval_selection import (  # noqa: E402
    EVAL_SUBSET_STRATIFIED_ROTATING,
    select_eval_subset,
)
from src.models.graphcast.training.core.segments import (  # noqa: E402
    _stop_gradient_dataset,
    iter_eval_segment_chunks,
    run_eval_segments,
)
from src.models.graphcast.training.core import model as core_model  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    FinalStepLossAutoregressivePredictor,
    build_zero_residual_inputs,
    reset_residual_input_lanes,
)
from src.models.graphcast.evaluation.device_resolution_eval import (  # noqa: E402
    _accumulate_step as accumulate_device_step,
    _empty_device_accumulator,
    add_device_accumulator_to_host,
    build_metric_spec,
)
from src.models.mamba.residual_mamba.training import model as residual_training_model  # noqa: E402
from src.models.mamba.residual_mamba.training.model import (  # noqa: E402
    augment_run_config,
    build_loss_prediction_transform as build_residual_loss_prediction_transform,
    residual_autoregressive_final_horizon,
    run_residual_eval,
    should_checkpoint_residual_ar_step,
)
from src.models.mamba.residual_mamba.runtime import _residual_output_head_enabled  # noqa: E402
from scripts.analyze_models.unified_resolution_eval import (  # noqa: E402
    NYC_LAT,
    NYC_LON,
    NYC_POINT_VARIABLE,
    _accumulate_metrics,
    _empty_metric_accumulator,
)
from scripts.analyze_models.legacy.analysis_metrics import normalized_weighted_mse_allvars  # noqa: E402
from graphcast import losses as gc_losses  # noqa: E402
from graphcast import xarray_jax  # noqa: E402


def _task_cfg() -> SimpleNamespace:
    return SimpleNamespace()


def _batch_builder(target_scale: float):
    def build_batch(
        _ds,
        *,
        indices,
        input_steps: int,
        target_steps: int,
        task_cfg,
        dt: pd.Timedelta,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        del input_steps, target_steps, task_cfg, dt
        arr = np.asarray(indices, dtype=np.float32)
        batch = np.arange(arr.size, dtype=np.int64)
        input_time = np.arange(2, dtype=np.int64)
        target_time = np.asarray([2], dtype=np.int64)
        inputs = xr.Dataset(
            {"x": (("batch", "time"), np.repeat(arr[:, None], input_time.size, axis=1))},
            coords={"batch": batch, "time": input_time},
        )
        targets = xr.Dataset(
            {"x": (("batch", "time"), (arr * target_scale)[:, None])},
            coords={"batch": batch, "time": target_time},
        )
        forcings = xr.Dataset(coords={"batch": batch, "time": target_time})
        return inputs, targets, forcings

    return build_batch


def _stateful_loss_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del inputs, forcings, is_training
        x = xarray_jax.unwrap_data(targets["x"])
        target_signal = jnp.mean(x, axis=tuple(range(1, x.ndim))) if x.ndim > 1 else x
        prev = hk.get_state("toy_ssm_state", shape=(target_signal.shape[0],), dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("toy_ssm_state", prev + target_signal)
        loss = xr.DataArray(xarray_jax.wrap(prev), dims=("batch",), coords={"batch": targets.coords["batch"]})
        return (loss, {})

    return hk.transform_with_state(forward)


def _target_length_stateful_loss_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del inputs, forcings, is_training
        batch_size = targets.sizes["batch"]
        prev = hk.get_state("toy_ssm_state", shape=(batch_size,), dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("toy_ssm_state", prev + jnp.asarray(targets.sizes["time"], dtype=jnp.float32))
        loss = xr.DataArray(xarray_jax.wrap(prev), dims=("batch",), coords={"batch": targets.coords["batch"]})
        return (loss, {})

    return hk.transform_with_state(forward)


def _stateful_loss_prediction_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del inputs, forcings, is_training
        x = xarray_jax.unwrap_data(targets["x"])
        target_signal = jnp.mean(x, axis=tuple(range(1, x.ndim))) if x.ndim > 1 else x
        prev = hk.get_state("toy_ssm_state", shape=(target_signal.shape[0],), dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("toy_ssm_state", prev + target_signal)
        loss = xr.DataArray(xarray_jax.wrap(prev), dims=("batch",), coords={"batch": targets.coords["batch"]})
        predictions = xr.Dataset({"x": xr.zeros_like(targets["x"])})
        return (loss, {}), predictions

    return hk.transform_with_state(forward)


def _counting_loss_prediction_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        batch_size = targets.sizes["batch"]
        prev = hk.get_state("toy_ssm_state", shape=(batch_size,), dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("toy_ssm_state", prev + 1.0)
        loss = xr.DataArray(xarray_jax.wrap(prev), dims=("batch",), coords={"batch": targets.coords["batch"]})
        pred = (inputs["x"].isel(time=-1, drop=True) + 1.0).expand_dims(
            time=targets.coords["time"]
        ).transpose(*targets["x"].dims)
        return (loss, xr.Dataset()), xr.Dataset({"x": pred})

    return hk.transform_with_state(forward)


def _input_feedback_loss_prediction_transform(increment: float = 10.0) -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        last = inputs["x"].isel(time=-1, drop=True)
        loss = xr.DataArray(
            xarray_jax.wrap(xarray_jax.unwrap_data(last)),
            dims=("batch",),
            coords={"batch": targets.coords["batch"]},
        )
        pred = (last + increment).expand_dims(time=targets.coords["time"]).transpose(*targets["x"].dims)
        return (loss, xr.Dataset()), xr.Dataset({"x": pred})

    return hk.transform_with_state(forward)


def _baseline_predict_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        last_input = inputs["x"].isel(time=-1, drop=True)
        pred = last_input.expand_dims(time=targets.coords["time"]).transpose(*targets["x"].dims)
        return xr.Dataset({"x": pred})

    return hk.transform_with_state(forward)


def test_stopped_baseline_predictions_cut_only_baseline_gradient() -> None:
    coords = {"batch": np.asarray([0]), "time": np.asarray([0])}

    def dataset_from(data):
        return xr.Dataset(
            {
                "x": xr.DataArray(
                    xarray_jax.wrap(data),
                    dims=("batch", "time"),
                    coords=coords,
                )
            }
        )

    def loss_fn(baseline_scale, residual_weight, *, stop_baseline: bool):
        target = dataset_from(jnp.ones((1, 1), dtype=jnp.float32))
        baseline = dataset_from(jnp.ones((1, 1), dtype=jnp.float32) * baseline_scale)
        if stop_baseline:
            baseline = _stop_gradient_dataset(baseline)
        residual_target = target - baseline
        residual_pred = dataset_from(xarray_jax.unwrap_data(residual_target["x"]) * residual_weight)
        error = xarray_jax.unwrap_data(residual_pred["x"] - residual_target["x"])
        return jnp.sum(error**2)

    grad_baseline_live, grad_residual_live = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.asarray(2.0),
        jnp.asarray(0.0),
        stop_baseline=False,
    )
    grad_baseline_stopped, grad_residual_stopped = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.asarray(2.0),
        jnp.asarray(0.0),
        stop_baseline=True,
    )

    assert not np.isclose(float(grad_baseline_live), 0.0)
    np.testing.assert_allclose(float(grad_baseline_stopped), 0.0)
    np.testing.assert_allclose(float(grad_residual_stopped), float(grad_residual_live))
    assert not np.isclose(float(grad_residual_stopped), 0.0)


class _IncrementPredictor:
    def __call__(self, inputs, targets_template, forcings):
        del forcings
        last = inputs["x"].isel(time=-1, drop=True)
        pred = (last + 1.0).expand_dims(time=targets_template.coords["time"]).transpose(*targets_template["x"].dims)
        return xr.Dataset({"x": pred})

    def loss(self, inputs, targets, forcings):
        (loss_and_diag, _predictions) = self.loss_and_predictions(inputs, targets, forcings)
        return loss_and_diag

    def loss_and_predictions(self, inputs, targets, forcings):
        predictions = self(inputs, targets, forcings)
        loss = ((predictions["x"] - targets["x"]) ** 2).mean("time", skipna=False)
        return (loss, xr.Dataset()), predictions


def _multi_step_batch_builder(target_values: np.ndarray):
    def build_batch(
        _ds,
        *,
        indices,
        input_steps: int,
        target_steps: int,
        task_cfg,
        dt: pd.Timedelta,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        del input_steps, task_cfg, dt
        arr = np.asarray(indices, dtype=np.float32)
        batch = np.arange(arr.size, dtype=np.int64)
        input_time = np.arange(2, dtype=np.int64)
        target_time = np.arange(2, 2 + target_steps, dtype=np.int64)
        values = np.asarray(target_values[:target_steps], dtype=np.float32)
        inputs = xr.Dataset(
            {"x": (("batch", "time"), np.repeat(arr[:, None], input_time.size, axis=1))},
            coords={"batch": batch, "time": input_time},
        )
        targets = xr.Dataset(
            {"x": (("batch", "time"), np.repeat(values[None, :], arr.size, axis=0))},
            coords={"batch": batch, "time": target_time},
        )
        forcings = xr.Dataset(coords={"batch": batch, "time": target_time})
        return inputs, targets, forcings

    return build_batch


def _constant_target_batch_builder(target_value: float):
    def build_batch(
        _ds,
        *,
        indices,
        input_steps: int,
        target_steps: int,
        task_cfg,
        dt: pd.Timedelta,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        del task_cfg, dt
        arr = np.asarray(indices, dtype=np.float32)
        batch = np.arange(arr.size, dtype=np.int64)
        input_time = np.arange(input_steps, dtype=np.int64)
        target_time = np.arange(input_steps, input_steps + target_steps, dtype=np.int64)
        inputs = xr.Dataset(
            {"x": (("batch", "time"), np.repeat(arr[:, None], input_steps, axis=1))},
            coords={"batch": batch, "time": input_time},
        )
        targets = xr.Dataset(
            {"x": (("batch", "time"), np.full((arr.size, target_steps), target_value, dtype=np.float32))},
            coords={"batch": batch, "time": target_time},
        )
        forcings = xr.Dataset(coords={"batch": batch, "time": target_time})
        return inputs, targets, forcings

    return build_batch


def test_final_step_autoregressive_loss_ignores_intermediate_losses() -> None:
    def forward(inputs, targets, forcings):
        predictor = FinalStepLossAutoregressivePredictor(_IncrementPredictor())
        return predictor.loss(inputs, targets, forcings)

    transformed = hk.transform_with_state(forward)
    inputs = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[0.0, 0.0]], dtype=np.float32))},
        coords={"batch": np.asarray([0]), "time": np.asarray([0, 1])},
    )
    targets = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[1.0, 99.0, 3.0]], dtype=np.float32))},
        coords={"batch": np.asarray([0]), "time": np.asarray([2, 3, 4])},
    )
    forcings = xr.Dataset(coords={"batch": np.asarray([0]), "time": np.asarray([2, 3, 4])})
    rng = jax.random.PRNGKey(0)
    params, state = transformed.init(rng, inputs, targets, forcings)

    loss_and_diag, _state = transformed.apply(params, state, rng, inputs, targets, forcings)

    np.testing.assert_allclose(np.asarray(xarray_jax.unwrap_data(loss_and_diag[0])), np.asarray([0.0]))


def test_iter_eval_segment_chunks_keeps_order_and_underfilled_final_group() -> None:
    segments = [
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.asarray([10, 11, 12, 13], dtype=np.int64),
        np.asarray([20, 21, 22, 23], dtype=np.int64),
    ]

    chunks = list(iter_eval_segment_chunks(segments, batch_size=2, bptt_steps=2))

    assert len(chunks) == 4
    np.testing.assert_array_equal(chunks[0][0][0], np.asarray([0, 10], dtype=np.int64))
    np.testing.assert_array_equal(chunks[0][0][1], np.asarray([1, 11], dtype=np.int64))
    np.testing.assert_array_equal(chunks[0][1], np.asarray([True, True]))
    np.testing.assert_array_equal(chunks[1][0][0], np.asarray([2, 12], dtype=np.int64))
    np.testing.assert_array_equal(chunks[1][0][1], np.asarray([3, 13], dtype=np.int64))
    np.testing.assert_array_equal(chunks[1][1], np.asarray([False, False]))
    np.testing.assert_array_equal(chunks[2][0][0], np.asarray([20], dtype=np.int64))
    np.testing.assert_array_equal(chunks[2][0][1], np.asarray([21], dtype=np.int64))
    np.testing.assert_array_equal(chunks[2][1], np.asarray([True]))
    np.testing.assert_array_equal(chunks[3][0][0], np.asarray([22], dtype=np.int64))
    np.testing.assert_array_equal(chunks[3][0][1], np.asarray([23], dtype=np.int64))
    np.testing.assert_array_equal(chunks[3][1], np.asarray([False]))


def test_stratified_eval_subset_covers_year_instead_of_first_segments() -> None:
    segment_ids = np.arange(48, dtype=np.int64)
    times = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range("2022-01-01", periods=12, freq="7D").values,
                pd.date_range("2022-04-01", periods=12, freq="7D").values,
                pd.date_range("2022-07-01", periods=12, freq="7D").values,
                pd.date_range("2022-10-01", periods=12, freq="7D").values,
            ]
        )
    )

    selection = select_eval_subset(segment_ids, 16, times=times)

    assert selection.positions.size == 16
    assert selection.item_ids.tolist() != list(range(16))
    selected_quarters = pd.Series(times[selection.positions]).dt.quarter.value_counts().sort_index().to_dict()
    assert selected_quarters == {1: 4, 2: 4, 3: 4, 4: 4}


def test_rotating_eval_subset_keeps_quarter_coverage_but_changes_ids() -> None:
    segment_ids = np.arange(48, dtype=np.int64)
    times = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range("2022-01-01", periods=12, freq="7D").values,
                pd.date_range("2022-04-01", periods=12, freq="7D").values,
                pd.date_range("2022-07-01", periods=12, freq="7D").values,
                pd.date_range("2022-10-01", periods=12, freq="7D").values,
            ]
        )
    )

    fold0 = select_eval_subset(
        segment_ids,
        16,
        times=times,
        policy=EVAL_SUBSET_STRATIFIED_ROTATING,
        role="rotating_diagnostic",
        fold=0,
    )
    fold1 = select_eval_subset(
        segment_ids,
        16,
        times=times,
        policy=EVAL_SUBSET_STRATIFIED_ROTATING,
        role="rotating_diagnostic",
        fold=1,
    )

    assert fold0.positions.size == fold1.positions.size == 16
    assert fold0.item_ids.tolist() != fold1.item_ids.tolist()
    assert pd.Series(times[fold0.positions]).dt.quarter.value_counts().sort_index().to_dict() == {
        1: 4,
        2: 4,
        3: 4,
        4: 4,
    }
    assert pd.Series(times[fold1.positions]).dt.quarter.value_counts().sort_index().to_dict() == {
        1: 4,
        2: 4,
        3: 4,
        4: 4,
    }


def test_eval_subset_fallbacks_for_small_or_nondatetime_inputs() -> None:
    all_items = select_eval_subset(np.arange(3), 8, times=["bad", "times", "here"])
    capped = select_eval_subset(np.arange(10), 4, times=["bad"] * 10)

    assert all_items.policy == "all"
    assert all_items.item_ids.tolist() == [0, 1, 2]
    assert capped.item_ids.tolist() != [0, 1, 2, 3]
    assert capped.item_ids.size == 4


def test_reset_residual_input_lanes_resets_only_target_variables() -> None:
    inputs = xr.Dataset(
        {
            "x": (("batch", "time"), np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)),
            "constant": (("batch",), np.asarray([10.0, 11.0, 12.0], dtype=np.float32)),
        },
        coords={"batch": np.arange(3), "time": np.arange(2)},
    )
    inputs["x"].attrs["units"] = "toy"
    targets = xr.Dataset({"x": (("batch",), np.zeros(3, dtype=np.float32))}, coords={"batch": np.arange(3)})

    reset = reset_residual_input_lanes(inputs, targets, jnp.asarray([True, False, True]))

    np.testing.assert_allclose(
        np.asarray(xarray_jax.unwrap_data(reset["x"])),
        np.asarray([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(np.asarray(xarray_jax.unwrap_data(reset["constant"])), np.asarray([10.0, 11.0, 12.0]))
    assert reset["x"].dims == ("batch", "time")
    assert reset["x"].attrs["units"] == "toy"


def test_reset_residual_input_lanes_handles_jax_wrapped_inputs_inside_jit() -> None:
    inputs = xr.Dataset(
        {
            "x": (("batch", "time"), xarray_jax.wrap(jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32))),
            "constant": (("batch",), xarray_jax.wrap(jnp.asarray([10.0, 11.0], dtype=jnp.float32))),
        },
        coords={"batch": np.arange(2), "time": np.arange(2)},
    )
    targets = xr.Dataset({"x": (("batch",), np.zeros(2, dtype=np.float32))}, coords={"batch": np.arange(2)})

    @jax.jit
    def reset_fn(batch_inputs, mask):
        return reset_residual_input_lanes(batch_inputs, targets, mask)

    reset = reset_fn(inputs, jnp.asarray([False, True]))

    np.testing.assert_allclose(
        np.asarray(xarray_jax.unwrap_data(reset["x"])),
        np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(np.asarray(xarray_jax.unwrap_data(reset["constant"])), np.asarray([10.0, 11.0]))


def _tiny_residual_rollout_case():
    def baseline_forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        pred = (inputs["x"].isel(time=-1, drop=True) + 10.0).expand_dims(
            time=targets.coords["time"]
        ).transpose(*targets["x"].dims)
        return xr.Dataset({"x": pred})

    def residual_forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        pred = (inputs["x"].isel(time=-1, drop=True) + 1.0).expand_dims(
            time=targets.coords["time"]
        ).transpose(*targets["x"].dims)
        predictions = xr.Dataset({"x": pred})
        loss = ((predictions["x"] - targets["x"]) ** 2).mean("time", skipna=False)
        return (loss, xr.Dataset()), predictions

    baseline_transform = hk.transform_with_state(baseline_forward)
    residual_transform = hk.transform_with_state(residual_forward)
    batch = np.asarray([0], dtype=np.int64)
    inputs = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[5.0, 9.0]], dtype=np.float32))},
        coords={"batch": batch, "time": np.asarray([0, 1])},
    )
    targets = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[20.0, 40.0]], dtype=np.float32))},
        coords={"batch": batch, "time": np.asarray([2, 3])},
    )
    forcings = xr.Dataset(coords={"batch": batch, "time": targets.coords["time"]})
    residual_inputs = build_zero_residual_inputs(inputs, targets)
    rng = jax.random.PRNGKey(0)
    baseline_params, baseline_state = baseline_transform.init(
        rng,
        inputs,
        targets.isel(time=slice(0, 1)),
        forcings.isel(time=slice(0, 1)),
        False,
    )
    residual_params, residual_state = residual_transform.init(
        rng,
        residual_inputs,
        targets.isel(time=slice(0, 1)),
        forcings.isel(time=slice(0, 1)),
        True,
    )
    return {
        "baseline_transform": baseline_transform,
        "residual_transform": residual_transform,
        "baseline_params": baseline_params,
        "baseline_state": baseline_state,
        "residual_params": residual_params,
        "residual_state": residual_state,
        "rng": rng,
        "inputs": inputs,
        "targets": targets,
        "forcings": forcings,
        "residual_inputs": residual_inputs,
    }


def test_residual_autoregressive_rollout_uses_predicted_full_feedback_and_teacher_carry() -> None:
    case = _tiny_residual_rollout_case()
    loss_by_lane, _carry_state, teacher_carry = residual_autoregressive_final_horizon(
        case["residual_transform"],
        case["baseline_transform"],
        residual_params=case["residual_params"],
        baseline_params=case["baseline_params"],
        residual_state=case["residual_state"],
        baseline_state=case["baseline_state"],
        rng_key=case["rng"],
        inputs=case["inputs"],
        targets=case["targets"],
        forcings=case["forcings"],
        residual_inputs=case["residual_inputs"],
        is_training=True,
        residual_ar_feedback="baseline_plus_residual",
    )

    np.testing.assert_allclose(np.asarray(jax.device_get(loss_by_lane)), np.asarray([64.0]))
    np.testing.assert_allclose(
        np.asarray(xarray_jax.unwrap_data(teacher_carry["x"])),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )


def test_residual_autoregressive_checkpoint_step_equivalence() -> None:
    case = _tiny_residual_rollout_case()

    def run(checkpoint_step: bool):
        return residual_autoregressive_final_horizon(
            case["residual_transform"],
            case["baseline_transform"],
            residual_params=case["residual_params"],
            baseline_params=case["baseline_params"],
            residual_state=case["residual_state"],
            baseline_state=case["baseline_state"],
            rng_key=case["rng"],
            inputs=case["inputs"],
            targets=case["targets"],
            forcings=case["forcings"],
            residual_inputs=case["residual_inputs"],
            is_training=True,
            checkpoint_step=checkpoint_step,
        )

    loss_plain, state_plain, teacher_plain = run(False)
    loss_ckpt, state_ckpt, teacher_ckpt = run(True)

    np.testing.assert_allclose(np.asarray(jax.device_get(loss_ckpt)), np.asarray(jax.device_get(loss_plain)))
    leaves_plain = jax.tree_util.tree_leaves(state_plain)
    leaves_ckpt = jax.tree_util.tree_leaves(state_ckpt)
    assert len(leaves_ckpt) == len(leaves_plain)
    for ckpt_leaf, plain_leaf in zip(leaves_ckpt, leaves_plain, strict=True):
        np.testing.assert_allclose(np.asarray(jax.device_get(ckpt_leaf)), np.asarray(jax.device_get(plain_leaf)))
    np.testing.assert_allclose(
        np.asarray(xarray_jax.unwrap_data(teacher_ckpt["x"])),
        np.asarray(xarray_jax.unwrap_data(teacher_plain["x"])),
    )


def test_residual_step_checkpoint_modes_request_checkpoint(monkeypatch) -> None:
    checkpoint_calls = 0

    def fake_checkpoint(fn, *args, **kwargs):
        nonlocal checkpoint_calls
        del args, kwargs
        checkpoint_calls += 1
        return fn

    monkeypatch.setattr(residual_training_model.jax, "checkpoint", fake_checkpoint)
    case = _tiny_residual_rollout_case()

    for mode, expected_calls in (
        ("standard", 0),
        ("conservative", 1),
        ("optimal", 1),
    ):
        checkpoint_calls = 0
        residual_autoregressive_final_horizon(
            case["residual_transform"],
            case["baseline_transform"],
            residual_params=case["residual_params"],
            baseline_params=case["baseline_params"],
            residual_state=case["residual_state"],
            baseline_state=case["baseline_state"],
            rng_key=case["rng"],
            inputs=case["inputs"],
            targets=case["targets"],
            forcings=case["forcings"],
            residual_inputs=case["residual_inputs"],
            is_training=True,
            checkpoint_step=should_checkpoint_residual_ar_step(mode),
        )
        assert checkpoint_calls == expected_calls


def test_residual_loss_prediction_transform_uses_one_step_predictor(monkeypatch) -> None:
    class FakeGraphCast:
        def __init__(self, model_cfg, task_cfg):
            del model_cfg, task_cfg

        def loss_and_predictions(self, inputs, targets, forcings):
            del inputs, forcings
            predictions = xr.zeros_like(targets)
            loss = ((predictions["x"] - targets["x"]) ** 2).mean("time", skipna=False)
            return (loss, xr.Dataset()), predictions

    monkeypatch.setattr(core_model.gc, "GraphCast", FakeGraphCast)
    cfg = SimpleNamespace(
        precision="fp32",
        temporal_backbone="mamba",
        temporal_location="mesh_processor_interleaved",
        temporal_d_inner=4,
        temporal_d_state=2,
        temporal_d_conv=4,
        temporal_dt_rank="auto",
        temporal_bias=False,
        temporal_conv_bias=True,
        temporal_layers=1,
        temporal_dropout=0.0,
        temporal_stateful=True,
        temporal_insert_count=1,
    )
    stats = {
        "stddev_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(1.0, dtype=np.float32))}),
        "mean_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(0.0, dtype=np.float32))}),
        "diffs_stddev_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(1.0, dtype=np.float32))}),
    }
    transform = build_residual_loss_prediction_transform(
        SimpleNamespace(),
        SimpleNamespace(),
        stats,
        cfg,
        gradient_checkpointing=True,
    )
    batch = np.asarray([0], dtype=np.int64)
    inputs = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[0.0, 0.0]], dtype=np.float32))},
        coords={"batch": batch, "time": np.asarray([0, 1])},
    )
    targets = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[2.0]], dtype=np.float32))},
        coords={"batch": batch, "time": np.asarray([2])},
    )
    forcings = xr.Dataset(coords={"batch": batch, "time": targets.coords["time"]})
    residual_inputs = build_zero_residual_inputs(inputs, targets)

    rng = jax.random.PRNGKey(0)
    params, state = transform.init(rng, residual_inputs, targets, forcings, True)
    (loss_and_diag, predictions), _ = transform.apply(
        params,
        state,
        rng,
        residual_inputs,
        targets,
        forcings,
        True,
    )

    np.testing.assert_allclose(np.asarray(jax.device_get(xarray_jax.unwrap_data(loss_and_diag[0]))), [4.0])
    np.testing.assert_allclose(np.asarray(jax.device_get(xarray_jax.unwrap_data(predictions["x"]))), [[0.0]])


def test_residual_loss_prediction_transform_enables_output_head(monkeypatch) -> None:
    class FakeGraphCast:
        def __init__(self, model_cfg, task_cfg):
            del model_cfg, task_cfg
            self._residual_output_head_enabled = False

        def loss_and_predictions(self, inputs, targets, forcings):
            del inputs, forcings
            upstream_predictions = xr.ones_like(targets) * 7.0
            predictions = xr.zeros_like(upstream_predictions) if self._residual_output_head_enabled else upstream_predictions
            loss = ((predictions["x"] - targets["x"]) ** 2).mean("time", skipna=False)
            return (loss, xr.Dataset()), predictions

    monkeypatch.setattr(core_model.gc, "GraphCast", FakeGraphCast)
    cfg = SimpleNamespace(
        precision="fp32",
        temporal_backbone="mamba",
        temporal_location="mesh_processor_interleaved",
        temporal_d_inner=4,
        temporal_d_state=2,
        temporal_d_conv=4,
        temporal_dt_rank="auto",
        temporal_bias=False,
        temporal_conv_bias=True,
        temporal_layers=1,
        temporal_dropout=0.0,
        temporal_stateful=True,
        temporal_insert_count=1,
        residual_output_head=True,
    )
    stats = {
        "stddev_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(1.0, dtype=np.float32))}),
        "mean_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(0.0, dtype=np.float32))}),
        "diffs_stddev_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(1.0, dtype=np.float32))}),
    }
    transform = build_residual_loss_prediction_transform(
        SimpleNamespace(),
        SimpleNamespace(),
        stats,
        cfg,
        gradient_checkpointing=True,
    )
    batch = np.asarray([0], dtype=np.int64)
    inputs = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[0.0, 0.0]], dtype=np.float32))},
        coords={"batch": batch, "time": np.asarray([0, 1])},
    )
    targets = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[2.0]], dtype=np.float32))},
        coords={"batch": batch, "time": np.asarray([2])},
    )
    forcings = xr.Dataset(coords={"batch": batch, "time": targets.coords["time"]})
    residual_inputs = build_zero_residual_inputs(inputs, targets)

    rng = jax.random.PRNGKey(0)
    params, state = transform.init(rng, residual_inputs, targets, forcings, True)
    (loss_and_diag, predictions), _ = transform.apply(
        params,
        state,
        rng,
        residual_inputs,
        targets,
        forcings,
        True,
    )

    np.testing.assert_allclose(np.asarray(jax.device_get(xarray_jax.unwrap_data(loss_and_diag[0]))), [4.0])
    np.testing.assert_allclose(np.asarray(jax.device_get(xarray_jax.unwrap_data(predictions["x"]))), [[0.0]])


def test_residual_output_head_run_config_defaults_to_disabled_for_old_checkpoints() -> None:
    assert not _residual_output_head_enabled({})
    assert not _residual_output_head_enabled({"residual_training": {"enabled": True}})
    assert _residual_output_head_enabled(
        {"residual_training": {"output_head": {"enabled": True}}}
    )


def test_residual_augment_run_config_records_output_head(tmp_path, monkeypatch) -> None:
    def fake_write_run_config(out_dir, *args, **kwargs):
        del args, kwargs
        (out_dir / "run_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "src.models.mamba.residual_mamba.training.model._write_run_config",
        fake_write_run_config,
    )
    base_cfg = SimpleNamespace(
        target_steps=1,
        temporal_backbone="mamba",
        residual_output_head=True,
    )
    segment_cfg = SimpleNamespace(
        base_cfg=base_cfg,
        len_segment=30,
        bptt_steps=6,
        chunk_load_workers=2,
        eval_num_segments=16,
        final_eval_num_segments=None,
        eval_subset_policy="stratified_fixed",
        eval_rotating_diagnostics=True,
        training_target="residual",
        baseline_ckpt="baseline.npz",
        resume_ckpt=None,
        residual_output_head_mode="auto",
    )

    augment_run_config(
        tmp_path,
        segment_cfg=segment_cfg,
        model_cfg=SimpleNamespace(),
        task_cfg=SimpleNamespace(),
        numpy_cache_active=False,
        train_cache_estimate_gib=None,
        effective_train_batch_builder="direct",
        effective_eval_batch_builder="direct",
    )

    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert payload["residual_training"]["output_head"] == {
        "enabled": True,
        "kind": "linear_channels",
        "position": "after_mesh2grid_before_residual_unnormalize",
        "init": "zero",
        "mode": "auto",
    }


def test_residual_loss_prediction_transform_init_uses_one_step_template(monkeypatch) -> None:
    class OneStepOnlyGraphCast:
        def __init__(self, model_cfg, task_cfg):
            del model_cfg, task_cfg

        def loss_and_predictions(self, inputs, targets, forcings):
            del inputs, forcings
            if targets.sizes["time"] != 1:
                raise ValueError(f"expected one target step, got {targets.sizes['time']}")
            predictions = xr.zeros_like(targets)
            loss = ((predictions["x"] - targets["x"]) ** 2).mean("time", skipna=False)
            return (loss, xr.Dataset()), predictions

    monkeypatch.setattr(core_model.gc, "GraphCast", OneStepOnlyGraphCast)
    cfg = SimpleNamespace(
        precision="fp32",
        temporal_backbone="mamba",
        temporal_location="mesh_processor_interleaved",
        temporal_d_inner=4,
        temporal_d_state=2,
        temporal_d_conv=4,
        temporal_dt_rank="auto",
        temporal_bias=False,
        temporal_conv_bias=True,
        temporal_layers=1,
        temporal_dropout=0.0,
        temporal_stateful=True,
        temporal_insert_count=1,
    )
    stats = {
        "stddev_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(1.0, dtype=np.float32))}),
        "mean_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(0.0, dtype=np.float32))}),
        "diffs_stddev_by_level": xr.Dataset({"x": xr.DataArray(np.asarray(1.0, dtype=np.float32))}),
    }
    transform = build_residual_loss_prediction_transform(
        SimpleNamespace(),
        SimpleNamespace(),
        stats,
        cfg,
        gradient_checkpointing=True,
    )
    inputs, targets, forcings = _multi_step_batch_builder(np.asarray([2.0, 3.0, 4.0, 5.0]))(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=4,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    init_targets = targets.isel(time=slice(0, 1))
    init_forcings = forcings.isel(time=slice(0, 1))
    residual_inputs = build_zero_residual_inputs(inputs, init_targets)

    rng = jax.random.PRNGKey(0)
    params, state = transform.init(rng, residual_inputs, init_targets, init_forcings, True)
    (loss_and_diag, predictions), _ = transform.apply(
        params,
        state,
        rng,
        residual_inputs,
        init_targets,
        init_forcings,
        True,
    )

    np.testing.assert_allclose(np.asarray(jax.device_get(xarray_jax.unwrap_data(loss_and_diag[0]))), [4.0])
    np.testing.assert_allclose(np.asarray(jax.device_get(xarray_jax.unwrap_data(predictions["x"]))), [[0.0]])


def test_run_eval_segments_preserves_state_within_segment_and_resets_between_segments() -> None:
    transformed = _stateful_loss_transform()

    sample_inputs, sample_targets, sample_forcings = _batch_builder(1.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    params, _ = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, False)

    metrics = run_eval_segments(
        transformed,
        params,
        rng,
        eval_ds=xr.Dataset(),
        eval_indices=np.arange(8, dtype=np.int64),
        eval_batch_size=1,
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=4,
        bptt_steps=2,
        progress_label="test",
        batch_builder=_batch_builder(1.0),
        chunk_load_workers=1,
    )

    assert "total" in metrics
    np.testing.assert_allclose(metrics["total"], 4.0)


def test_run_eval_segments_carries_one_step_state_for_multi_step_targets() -> None:
    transformed = _target_length_stateful_loss_transform()
    sample_inputs, sample_targets, sample_forcings = _multi_step_batch_builder(np.asarray([1.0, 2.0, 3.0]))(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=3,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    params, _ = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, False)

    metrics = run_eval_segments(
        transformed,
        params,
        rng,
        eval_ds=xr.Dataset(),
        eval_indices=np.arange(8, dtype=np.int64),
        eval_batch_size=1,
        input_steps=2,
        target_steps=3,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=4,
        bptt_steps=2,
        progress_label="test",
        batch_builder=_multi_step_batch_builder(np.asarray([1.0, 2.0, 3.0])),
        chunk_load_workers=1,
    )

    np.testing.assert_allclose(metrics["total"], 1.5)


def test_run_eval_segments_rolling_ar_scores_uniform_chunk_tail() -> None:
    transformed = _counting_loss_prediction_transform()
    sample_inputs, sample_targets, sample_forcings = _constant_target_batch_builder(0.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    params, _ = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, False)

    metrics = run_eval_segments(
        transformed,
        params,
        rng,
        eval_ds=xr.Dataset(),
        eval_indices=np.arange(6, dtype=np.int64),
        eval_batch_size=1,
        input_steps=2,
        target_steps=4,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=6,
        bptt_steps=6,
        progress_label="test",
        batch_builder=_constant_target_batch_builder(0.0),
        chunk_load_workers=1,
        rolling_ar=True,
    )

    np.testing.assert_allclose(metrics["total"], 3.5)


def test_run_eval_segments_rolling_ar_first_tail_input_uses_previous_prediction() -> None:
    transformed = _input_feedback_loss_prediction_transform(increment=10.0)
    sample_inputs, sample_targets, sample_forcings = _constant_target_batch_builder(0.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    params, _ = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, False)

    metrics = run_eval_segments(
        transformed,
        params,
        rng,
        eval_ds=xr.Dataset(),
        eval_indices=np.arange(4, dtype=np.int64),
        eval_batch_size=1,
        input_steps=2,
        target_steps=2,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=4,
        bptt_steps=4,
        progress_label="test",
        batch_builder=_constant_target_batch_builder(0.0),
        chunk_load_workers=1,
        rolling_ar=True,
    )

    np.testing.assert_allclose(metrics["total"], 16.0)


def test_run_eval_segments_rolling_ar_resets_physical_inputs_each_chunk_but_carries_state() -> None:
    def forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        batch_size = targets.sizes["batch"]
        last = inputs["x"].isel(time=-1, drop=True)
        prev = hk.get_state("toy_ssm_state", shape=(batch_size,), dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("toy_ssm_state", prev + 100.0)
        loss_data = xarray_jax.unwrap_data(last) + prev
        loss = xr.DataArray(xarray_jax.wrap(loss_data), dims=("batch",), coords={"batch": targets.coords["batch"]})
        pred = (last + 10.0).expand_dims(time=targets.coords["time"]).transpose(*targets["x"].dims)
        return (loss, xr.Dataset()), xr.Dataset({"x": pred})

    transformed = hk.transform_with_state(forward)
    sample_inputs, sample_targets, sample_forcings = _constant_target_batch_builder(0.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    params, _ = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, False)

    metrics = run_eval_segments(
        transformed,
        params,
        rng,
        eval_ds=xr.Dataset(),
        eval_indices=np.arange(8, dtype=np.int64),
        eval_batch_size=1,
        input_steps=2,
        target_steps=2,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=8,
        bptt_steps=4,
        progress_label="test",
        batch_builder=_constant_target_batch_builder(0.0),
        chunk_load_workers=1,
        rolling_ar=True,
    )

    np.testing.assert_allclose(metrics["total"], 468.0)


def test_run_eval_segments_uses_stratified_subset_metadata() -> None:
    transformed = _stateful_loss_transform()
    sample_inputs, sample_targets, sample_forcings = _batch_builder(1.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    params, _ = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, False)
    eval_ds = xr.Dataset(coords={"time": pd.date_range("2022-01-01", periods=194, freq="6h")})

    metrics = run_eval_segments(
        transformed,
        params,
        rng,
        eval_ds=eval_ds,
        eval_indices=np.arange(2, 194, dtype=np.int64),
        eval_batch_size=4,
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=4,
        bptt_steps=2,
        progress_label="test",
        batch_builder=_batch_builder(1.0),
        chunk_load_workers=1,
        max_segments=16,
    )

    assert metrics["segments"] == 16.0
    assert metrics["eval_subset_policy"] == "stratified_fixed"
    assert metrics["eval_subset_available_segments"] == 48
    assert metrics["eval_subset_selected_segments"] == 16
    assert metrics["eval_subset_segments_ids"] != list(range(16))


def test_batch_eval_uses_stratified_windows_when_capped() -> None:
    transformed = _stateful_loss_transform()
    sample_inputs, sample_targets, sample_forcings = _batch_builder(1.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    params, state = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, False)
    eval_ds = xr.Dataset(coords={"time": pd.date_range("2022-01-01", periods=96, freq="6h")})

    metrics = run_eval(
        transformed,
        params,
        state,
        rng,
        eval_ds=eval_ds,
        eval_indices=np.arange(2, 96, dtype=np.int64),
        eval_batch_size=4,
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        progress_label="test",
        batch_builder=_batch_builder(1.0),
        prefetch_workers=1,
        prefetch_depth=1,
        max_batches=4,
    )

    assert metrics["batches"] == 4.0
    assert metrics["eval_subset_selected_windows"] == 16
    assert metrics["eval_subset_windows_ids"] != list(range(2, 18))


def test_run_residual_eval_preserves_residual_state_within_segment() -> None:
    residual_transform = _stateful_loss_prediction_transform()
    baseline_transform = _baseline_predict_transform()

    sample_inputs, sample_targets, sample_forcings = _batch_builder(2.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    residual_params, _ = residual_transform.init(rng, sample_inputs, sample_targets, sample_forcings, False)
    baseline_params, _ = baseline_transform.init(rng, sample_inputs, sample_targets, sample_forcings, False)

    metrics = run_residual_eval(
        residual_transform,
        baseline_transform,
        residual_params,
        baseline_params,
        rng,
        eval_ds=xr.Dataset(),
        eval_indices=np.arange(8, dtype=np.int64),
        eval_batch_size=1,
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=4,
        bptt_steps=2,
        progress_label="test",
        batch_builder=_batch_builder(2.0),
        chunk_load_workers=1,
    )

    assert "total" in metrics
    np.testing.assert_allclose(metrics["total"], 32.0 / 6.0)


def _run_constant_target_residual_eval(*, residual_ar_feedback: str | None = None) -> dict[str, float]:
    def baseline_forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        pred = (inputs["x"].isel(time=-1, drop=True) + 10.0).expand_dims(
            time=targets.coords["time"]
        ).transpose(*targets["x"].dims)
        return xr.Dataset({"x": pred})

    def residual_forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        pred = (inputs["x"].isel(time=-1, drop=True) + 1.0).expand_dims(
            time=targets.coords["time"]
        ).transpose(*targets["x"].dims)
        predictions = xr.Dataset({"x": pred})
        loss = ((predictions["x"] - targets["x"]) ** 2).mean("time", skipna=False)
        return (loss, xr.Dataset()), predictions

    residual_transform = hk.transform_with_state(residual_forward)
    baseline_transform = hk.transform_with_state(baseline_forward)
    sample_inputs, sample_targets, sample_forcings = _constant_target_batch_builder(100.0)(
        None,
        indices=np.asarray([0], dtype=np.int64),
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    rng = jax.random.PRNGKey(0)
    residual_inputs = build_zero_residual_inputs(sample_inputs, sample_targets)
    residual_params, _ = residual_transform.init(rng, residual_inputs, sample_targets, sample_forcings, False)
    baseline_params, _ = baseline_transform.init(rng, sample_inputs, sample_targets, sample_forcings, False)

    kwargs = {}
    if residual_ar_feedback is not None:
        kwargs["residual_ar_feedback"] = residual_ar_feedback
    return run_residual_eval(
        residual_transform,
        baseline_transform,
        residual_params,
        baseline_params,
        rng,
        eval_ds=xr.Dataset(),
        eval_indices=np.arange(4, dtype=np.int64),
        eval_batch_size=1,
        input_steps=2,
        target_steps=2,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
        len_segment=4,
        bptt_steps=4,
        progress_label="test",
        batch_builder=_constant_target_batch_builder(100.0),
        chunk_load_workers=1,
        **kwargs,
    )


def test_run_residual_eval_rolling_ar_uses_baseline_plus_residual_feedback() -> None:
    metrics = _run_constant_target_residual_eval(residual_ar_feedback="baseline_plus_residual")
    np.testing.assert_allclose(metrics["total"], 25806.5)


def test_run_residual_eval_rolling_ar_can_use_baseline_only_feedback() -> None:
    metrics = _run_constant_target_residual_eval(residual_ar_feedback="baseline")
    np.testing.assert_allclose(metrics["total"], 302.5)


def test_device_metric_accumulator_matches_host_xarray_metrics() -> None:
    batch = np.arange(2)
    time = np.arange(2)
    level = np.asarray([100, 300], dtype=np.int32)
    lat = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    lon = np.asarray([0.0, 90.0, 180.0], dtype=np.float32)

    surface_target = np.arange(batch.size * time.size * lat.size * lon.size, dtype=np.float32).reshape(
        batch.size, time.size, lat.size, lon.size
    )
    surface_pred = surface_target + 1.5
    level_target = np.arange(batch.size * time.size * level.size * lat.size * lon.size, dtype=np.float32).reshape(
        batch.size, time.size, level.size, lat.size, lon.size
    )
    level_pred = level_target - 2.0

    coords = {"batch": batch, "time": time, "level": level, "lat": lat, "lon": lon}
    targets = xr.Dataset(
        {
            "2m_temperature": (("batch", "time", "lat", "lon"), surface_target),
            "temperature": (("batch", "time", "level", "lat", "lon"), level_target),
        },
        coords=coords,
    )
    preds = xr.Dataset(
        {
            "2m_temperature": (("batch", "time", "lat", "lon"), surface_pred),
            "temperature": (("batch", "time", "level", "lat", "lon"), level_pred),
        },
        coords=coords,
    )
    stats = {
        "diffs_stddev_by_level": xr.Dataset(
            {
                "2m_temperature": ((), np.asarray(2.0, dtype=np.float32)),
                "temperature": (("level",), np.asarray([2.0, 4.0, 6.0], dtype=np.float32)),
            },
            coords={"level": np.asarray([100, 200, 300], dtype=np.int32)},
        )
    }
    res_grid_lats = xr.DataArray(np.asarray([-10.0, 10.0], dtype=np.float32), dims=("lat",))
    res_grid_lons = xr.DataArray(np.asarray([0.0, 180.0], dtype=np.float32), dims=("lon",))

    host_acc = _empty_metric_accumulator(max_lead_steps=2)
    _accumulate_metrics(
        host_acc,
        preds,
        targets,
        res_grid_lats=res_grid_lats,
        res_grid_lons=res_grid_lons,
        stats=stats,
    )

    metric_spec = build_metric_spec(
        targets,
        stats=stats,
        res_grid_lats=res_grid_lats,
        res_grid_lons=res_grid_lons,
        per_variable_weights={"2m_temperature": 1.0, "temperature": 1.0},
        max_lead_steps=2,
        nyc_lat=NYC_LAT,
        nyc_lon=NYC_LON,
        nyc_output_name=NYC_POINT_VARIABLE,
    )
    device_acc = _empty_device_accumulator(metric_spec)
    for lead_i in range(2):
        device_acc = accumulate_device_step(
            device_acc,
            preds.isel(time=slice(lead_i, lead_i + 1)),
            targets.isel(time=slice(lead_i, lead_i + 1)),
            lead_i=lead_i,
            metric_spec=metric_spec,
        )
    device_host_acc = _empty_metric_accumulator(max_lead_steps=2)
    add_device_accumulator_to_host(
        device_host_acc,
        device_acc,
        variable_names=tuple(var.output_name for var in metric_spec.variables),
    )

    np.testing.assert_allclose(device_host_acc["weighted_sum"], host_acc["weighted_sum"], rtol=1e-6)
    np.testing.assert_array_equal(device_host_acc["weighted_count"], host_acc["weighted_count"])
    for name in host_acc["per_variable_sum"]:
        np.testing.assert_allclose(
            device_host_acc["per_variable_sum"][name],
            host_acc["per_variable_sum"][name],
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            device_host_acc["per_variable_count"][name],
            host_acc["per_variable_count"][name],
        )


def test_resolution_weighted_allvars_matches_graphcast_loss_on_projected_grid() -> None:
    batch = np.arange(2)
    time = np.asarray([0], dtype=np.int64)
    level = np.asarray([100, 300], dtype=np.int32)
    lat = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    lon = np.asarray([0.0, 90.0, 180.0], dtype=np.float32)

    surface_target = np.arange(batch.size * time.size * lat.size * lon.size, dtype=np.float32).reshape(
        batch.size, time.size, lat.size, lon.size
    )
    level_target = np.arange(batch.size * time.size * level.size * lat.size * lon.size, dtype=np.float32).reshape(
        batch.size, time.size, level.size, lat.size, lon.size
    )
    targets = xr.Dataset(
        {
            "2m_temperature": (("batch", "time", "lat", "lon"), surface_target),
            "temperature": (("batch", "time", "level", "lat", "lon"), level_target),
        },
        coords={"batch": batch, "time": time, "level": level, "lat": lat, "lon": lon},
    )
    preds = xr.Dataset(
        {
            "2m_temperature": targets["2m_temperature"] + 2.0,
            "temperature": targets["temperature"] - 3.0,
        }
    )
    stats = {
        "diffs_stddev_by_level": xr.Dataset(
            {
                "2m_temperature": ((), np.asarray(4.0, dtype=np.float32)),
                "temperature": (("level",), np.asarray([2.0, 6.0], dtype=np.float32)),
            },
            coords={"level": level},
        )
    }
    res_grid_lats = xr.DataArray(np.asarray([-10.0, 10.0], dtype=np.float32), dims=("lat",))
    res_grid_lons = xr.DataArray(np.asarray([0.0, 180.0], dtype=np.float32), dims=("lon",))
    grid_preds = preds.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")
    grid_targets = targets.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")

    actual = normalized_weighted_mse_allvars(
        grid_preds,
        grid_targets,
        per_variable_weights={"2m_temperature": 1.0, "temperature": 0.25},
        use_latitude_weights=True,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )

    norm_errors = xr.Dataset()
    zero_targets = xr.Dataset()
    for name in grid_targets.data_vars:
        scale = stats["diffs_stddev_by_level"][name].astype(grid_preds[name].dtype)
        norm_errors[name] = (grid_preds[name] - grid_targets[name]) / scale
        zero_targets[name] = xr.zeros_like(norm_errors[name])
    expected, _diagnostics = gc_losses.weighted_mse_per_level(
        norm_errors,
        zero_targets,
        per_variable_weights={"2m_temperature": 1.0, "temperature": 0.25},
    )

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-6)
