from __future__ import annotations

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
from src.models.graphcast.training.core.segments import iter_eval_segment_chunks, run_eval_segments  # noqa: E402
from src.models.graphcast.training.core.model import reset_residual_input_lanes  # noqa: E402
from src.models.graphcast.evaluation.device_resolution_eval import (  # noqa: E402
    _accumulate_step as accumulate_device_step,
    _empty_device_accumulator,
    add_device_accumulator_to_host,
    build_metric_spec,
)
from src.models.mamba.residual_mamba.training.model import run_residual_eval  # noqa: E402
from scripts.analyze_models.unified_resolution_eval import (  # noqa: E402
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


def _baseline_predict_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del forcings, is_training
        last_input = inputs["x"].isel(time=-1, drop=True)
        pred = last_input.expand_dims(time=targets.coords["time"]).transpose(*targets["x"].dims)
        return xr.Dataset({"x": pred})

    return hk.transform_with_state(forward)


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
    residual_transform = _stateful_loss_transform()
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
        variable_names=tuple(var.name for var in metric_spec.variables),
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
