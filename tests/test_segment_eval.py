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

from src.models.graphcast.training.core.segments import iter_eval_segment_chunks, run_eval_segments  # noqa: E402
from src.models.mamba.residual_mamba.training.model import run_residual_eval  # noqa: E402
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
        inputs = xr.Dataset({"x": (("batch",), arr)}, coords={"batch": batch})
        targets = xr.Dataset({"x": (("batch",), arr * target_scale)}, coords={"batch": batch})
        forcings = xr.Dataset(coords={"batch": batch})
        return inputs, targets, forcings

    return build_batch


def _stateful_loss_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del inputs, forcings, is_training
        x = xarray_jax.unwrap_data(targets["x"])
        prev = hk.get_state("toy_ssm_state", shape=(x.shape[0],), dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("toy_ssm_state", prev + x)
        loss = xr.DataArray(jnp.mean(prev))
        return (loss, {})

    return hk.transform_with_state(forward)


def _baseline_predict_transform() -> hk.TransformedWithState:
    def forward(inputs, targets, forcings, is_training):
        del targets, forcings, is_training
        return xr.Dataset({"x": inputs["x"]})

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

    assert metrics.keys() == {"total"}
    np.testing.assert_allclose(metrics["total"], 4.0)


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

    assert metrics.keys() == {"total"}
    np.testing.assert_allclose(metrics["total"], 4.0)
