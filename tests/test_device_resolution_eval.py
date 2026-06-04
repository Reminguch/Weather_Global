from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
for path in (ROOT, GRAPHCAST_LOCAL):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from graphcast import xarray_jax
from src.models.graphcast.evaluation.device_resolution_eval import _ModelBundle, _model_step_residual
from src.models.graphcast.training.core.model import build_zero_residual_inputs
from src.models.mamba.residual_mamba import runtime as residual_runtime


def _array_values(dataset: xr.Dataset, name: str) -> np.ndarray:
    return np.asarray(xarray_jax.unwrap_data(dataset[name]))


def _write_residual_run_config(tmp_path: Path) -> Path:
    ckpt_path = tmp_path / "ckpt_best.npz"
    (tmp_path / "run_config.json").write_text(
        json.dumps({"residual_training": {"enabled": True}}),
        encoding="utf-8",
    )
    return ckpt_path


def _fake_residual_bundles(baseline_seen: list[xr.Dataset], residual_seen: list[xr.Dataset]):
    def baseline_step(*, rng, inputs, targets_template, forcings, state=None):
        del rng, forcings
        baseline_seen.append(inputs.copy(deep=True))
        pred = xr.zeros_like(targets_template["x"]) + 10.0
        return xr.Dataset({"x": pred}), state

    def residual_step(*, rng, inputs, targets_template, forcings, state=None):
        del rng, forcings
        residual_seen.append(inputs.copy(deep=True))
        pred = xr.zeros_like(targets_template["x"]) + float(len(residual_seen))
        return xr.Dataset({"x": pred}), state

    common = {
        "task_cfg": SimpleNamespace(),
        "model_cfg": SimpleNamespace(resolution=2.0),
        "run_cfg": {"residual_training": {"enabled": True}},
        "params": {},
        "transformed": SimpleNamespace(),
    }
    return (
        {**common, "step": residual_step},
        {**common, "step": baseline_step},
    )


def test_residual_device_step_uses_residual_history_inputs() -> None:
    batch = np.asarray([0], dtype=np.int64)
    input_time = np.asarray([0, 1], dtype=np.int64)
    target_time = np.asarray([2], dtype=np.int64)
    rolling_inputs = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[5.0, 9.0]], dtype=np.float32))},
        coords={"batch": batch, "time": input_time},
    )
    constant_inputs = xr.Dataset(
        {"constant": (("batch",), np.asarray([7.0], dtype=np.float32))},
        coords={"batch": batch},
    )
    target_step_1 = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[14.0]], dtype=np.float32))},
        coords={"batch": batch, "time": target_time},
    )
    forcings_step = xr.Dataset(coords={"batch": batch, "time": target_time})
    residual_inputs = build_zero_residual_inputs(xr.merge([constant_inputs, rolling_inputs]), target_step_1)

    baseline_seen: list[xr.Dataset] = []
    residual_seen: list[xr.Dataset] = []

    def baseline_apply(params, state, rng, inputs, targets_template, forcings):
        del params, state, rng, forcings
        baseline_seen.append(inputs.copy(deep=True))
        pred = xr.zeros_like(targets_template["x"]) + 10.0
        return xr.Dataset({"x": pred}), {"baseline": len(baseline_seen)}

    def residual_apply(params, state, rng, inputs, targets_template, forcings):
        del params, state, rng, forcings
        residual_seen.append(inputs.copy(deep=True))
        pred = xr.zeros_like(targets_template["x"]) + float(len(residual_seen))
        return xr.Dataset({"x": pred}), {"residual": len(residual_seen)}

    baseline = _ModelBundle(params={}, transformed=SimpleNamespace(apply=baseline_apply))
    residual = _ModelBundle(params={}, transformed=SimpleNamespace(apply=residual_apply))

    pred_1, residual_state, baseline_state, residual_inputs = _model_step_residual(
        residual,
        baseline,
        residual_params={},
        baseline_params={},
        residual_state={},
        baseline_state={},
        rng=(jax.random.PRNGKey(0), jax.random.PRNGKey(1)),
        rolling_inputs=rolling_inputs,
        constant_inputs=constant_inputs,
        residual_inputs=residual_inputs,
        target_step=target_step_1,
        forcings_step=forcings_step,
    )

    np.testing.assert_allclose(_array_values(pred_1, "x"), np.asarray([[11.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(baseline_seen[0], "x"), np.asarray([[5.0, 9.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[0], "x"), np.asarray([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[0], "constant"), np.asarray([7.0], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_inputs, "x"), np.asarray([[0.0, 4.0]], dtype=np.float32))
    assert residual_state == {"residual": 1}
    assert baseline_state == {"baseline": 1}

    target_step_2 = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[16.0]], dtype=np.float32))},
        coords={"batch": batch, "time": target_time},
    )
    pred_2, _residual_state, _baseline_state, residual_inputs = _model_step_residual(
        residual,
        baseline,
        residual_params={},
        baseline_params={},
        residual_state=residual_state,
        baseline_state=baseline_state,
        rng=(jax.random.PRNGKey(2), jax.random.PRNGKey(3)),
        rolling_inputs=rolling_inputs,
        constant_inputs=constant_inputs,
        residual_inputs=residual_inputs,
        target_step=target_step_2,
        forcings_step=forcings_step,
    )

    np.testing.assert_allclose(_array_values(pred_2, "x"), np.asarray([[12.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[1], "x"), np.asarray([[0.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_inputs, "x"), np.asarray([[4.0, 6.0]], dtype=np.float32))


def test_residual_training_equivalent_cold_runner_teacher_forces_history(monkeypatch, tmp_path) -> None:
    baseline_seen: list[xr.Dataset] = []
    residual_seen: list[xr.Dataset] = []
    monkeypatch.setattr(
        residual_runtime,
        "_build_residual_rollout_bundle",
        lambda ckpt_obj, stats, ckpt_path: _fake_residual_bundles(baseline_seen, residual_seen),
    )

    batch = np.asarray([0], dtype=np.int64)
    input_time = np.asarray([0, 1], dtype=np.int64)
    target_time = np.asarray([2, 3], dtype=np.int64)
    inputs = xr.Dataset(
        {
            "x": (("batch", "time"), np.asarray([[5.0, 9.0]], dtype=np.float32)),
            "constant": (("batch",), np.asarray([7.0], dtype=np.float32)),
        },
        coords={"batch": batch, "time": input_time},
    )
    targets = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[14.0, 16.0]], dtype=np.float32))},
        coords={"batch": batch, "time": target_time},
    )
    forcings = xr.Dataset(coords={"batch": batch, "time": target_time})

    run_jitted, _task_cfg, _model_cfg, _run_cfg = residual_runtime.build_training_equivalent_run_jitted(
        SimpleNamespace(), {}, _write_residual_run_config(tmp_path)
    )
    pred = run_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets,
        forcings=forcings,
    )

    np.testing.assert_allclose(_array_values(pred, "x"), np.asarray([[11.0, 12.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(baseline_seen[0], "x"), np.asarray([[5.0, 9.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(baseline_seen[1], "x"), np.asarray([[9.0, 14.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[0], "x"), np.asarray([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[1], "x"), np.asarray([[0.0, 4.0]], dtype=np.float32))


def test_residual_training_equivalent_warm_branch_teacher_forces_history(monkeypatch, tmp_path) -> None:
    baseline_seen: list[xr.Dataset] = []
    residual_seen: list[xr.Dataset] = []
    monkeypatch.setattr(
        residual_runtime,
        "_build_residual_rollout_bundle",
        lambda ckpt_obj, stats, ckpt_path: _fake_residual_bundles(baseline_seen, residual_seen),
    )

    batch = np.asarray([0], dtype=np.int64)
    input_time = np.asarray([0, 1], dtype=np.int64)
    target_time = np.asarray([2, 3, 4], dtype=np.int64)
    inputs = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[5.0, 9.0]], dtype=np.float32))},
        coords={"batch": batch, "time": input_time},
    )
    targets = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[14.0, 16.0, 18.0]], dtype=np.float32))},
        coords={"batch": batch, "time": target_time},
    )
    forcings = xr.Dataset(coords={"batch": batch, "time": target_time})

    runner = residual_runtime.build_training_equivalent_truth_anchored_residual_runner(
        SimpleNamespace(), {}, _write_residual_run_config(tmp_path)
    )
    context = runner["initialize_context"](
        inputs=inputs,
        targets_template=targets.isel(time=slice(0, 1)),
        forcings=forcings.isel(time=slice(0, 1)),
    )
    _truth_pred, context = runner["truth_step"](
        rng=(jax.random.PRNGKey(0), jax.random.PRNGKey(1)),
        context=context,
        target_step=targets.isel(time=slice(0, 1)),
        forcings_step=forcings.isel(time=slice(0, 1)),
    )
    branch_pred = runner["branch_rollout"](
        rng=jax.random.PRNGKey(2),
        context=context,
        targets_template=targets.isel(time=slice(1, 3)),
        forcings=forcings.isel(time=slice(1, 3)),
    )

    np.testing.assert_allclose(_array_values(branch_pred, "x"), np.asarray([[12.0, 13.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(baseline_seen[1], "x"), np.asarray([[9.0, 14.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(baseline_seen[2], "x"), np.asarray([[14.0, 16.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[1], "x"), np.asarray([[0.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[2], "x"), np.asarray([[4.0, 6.0]], dtype=np.float32))


def test_residual_rollout_warm_context_truth_then_branch_predictions(monkeypatch, tmp_path) -> None:
    baseline_seen: list[xr.Dataset] = []
    residual_seen: list[xr.Dataset] = []
    monkeypatch.setattr(
        residual_runtime,
        "_build_residual_rollout_bundle",
        lambda ckpt_obj, stats, ckpt_path: _fake_residual_bundles(baseline_seen, residual_seen),
    )

    batch = np.asarray([0], dtype=np.int64)
    input_time = np.asarray([0, 1], dtype=np.int64)
    target_time = np.asarray([2, 3, 4], dtype=np.int64)
    inputs = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[5.0, 9.0]], dtype=np.float32))},
        coords={"batch": batch, "time": input_time},
    )
    targets = xr.Dataset(
        {"x": (("batch", "time"), np.asarray([[14.0, 16.0, 18.0]], dtype=np.float32))},
        coords={"batch": batch, "time": target_time},
    )
    forcings = xr.Dataset(coords={"batch": batch, "time": target_time})

    runner = residual_runtime.build_truth_anchored_residual_runner(
        SimpleNamespace(), {}, _write_residual_run_config(tmp_path)
    )
    context = runner["initialize_context"](
        inputs=inputs,
        targets_template=targets.isel(time=slice(0, 1)),
        forcings=forcings.isel(time=slice(0, 1)),
    )
    _truth_pred, context = runner["truth_step"](
        rng=(jax.random.PRNGKey(0), jax.random.PRNGKey(1)),
        context=context,
        target_step=targets.isel(time=slice(0, 1)),
        forcings_step=forcings.isel(time=slice(0, 1)),
    )
    branch_pred = runner["branch_rollout"](
        rng=jax.random.PRNGKey(2),
        context=context,
        targets_template=targets.isel(time=slice(1, 3)),
        forcings=forcings.isel(time=slice(1, 3)),
    )

    np.testing.assert_allclose(_array_values(branch_pred, "x"), np.asarray([[12.0, 13.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(baseline_seen[1], "x"), np.asarray([[9.0, 14.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(baseline_seen[2], "x"), np.asarray([[14.0, 12.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[1], "x"), np.asarray([[0.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_array_values(residual_seen[2], "x"), np.asarray([[4.0, 2.0]], dtype=np.float32))
