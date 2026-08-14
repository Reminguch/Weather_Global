from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import xarray as xr

from src.models.mamba.v23_Ilya.rollout import (
    run_v23_Ilya_rollout,
    shift_inputs_with_field,
)


TIME_STEP = pd.Timedelta("6h")


def _dataset(values: list[float], start: str = "2022-01-01T00:00") -> xr.Dataset:
    times = np.asarray(
        [np.datetime64(start) + index * np.timedelta64(6, "h") for index in range(len(values))]
    )
    array = np.asarray(values, dtype=np.float32)[None, :, None, None]
    return xr.Dataset(
        {"x": (("batch", "time", "lat", "lon"), array)},
        coords={"batch": [0], "time": times, "lat": [0.0], "lon": [0.0]},
    )


def _forcings(length: int, start: str = "2022-01-01T00:00") -> xr.Dataset:
    times = np.asarray(
        [np.datetime64(start) + index * np.timedelta64(6, "h") for index in range(length)]
    )
    array = np.arange(length, dtype=np.float32)[None, :, None, None]
    return xr.Dataset(
        {"forcing": (("batch", "time", "lat", "lon"), array)},
        coords={"batch": [0], "time": times, "lat": [0.0], "lon": [0.0]},
    )


def _inputs() -> xr.Dataset:
    inputs = _dataset([-1.0, 0.0], start="2021-12-31T12:00")
    inputs["forcing"] = inputs["x"] * 0
    return inputs


def _baseline_step(params, state, key, inputs, targets, forcings):
    del params, key, forcings
    last = inputs["x"].isel(time=-1, drop=True)
    prediction = xr.Dataset({"x": xr.zeros_like(targets["x"]) + last + 1.0})
    return prediction, {"count": state["count"] + 1}


def _residual_step(params, state, key, inputs, targets, forcings):
    del params, key, inputs, forcings
    prediction = xr.Dataset({"x": xr.ones_like(targets["x"]) * 2.0})
    return prediction, {"count": state["count"] + 1}


def _run(
    *,
    full_feedback: bool,
    warmup_steps: int = 0,
    reset: bool = False,
    reset_every_step: bool = False,
):
    target_values = [10.0] * warmup_steps + [0.0, 0.0, 0.0]
    return run_v23_Ilya_rollout(
        rng=jax.random.PRNGKey(0),
        inputs=_inputs(),
        all_targets=_dataset(target_values),
        all_forcings=_forcings(len(target_values)),
        baseline_step=_baseline_step,
        residual_step=_residual_step,
        baseline_params={},
        residual_params={},
        baseline_state_init={"count": 0},
        residual_state_init={"count": 0},
        time_step=TIME_STEP,
        input_steps=2,
        warmup_steps=warmup_steps,
        target_steps=3 if warmup_steps == 0 else 1,
        full_feedback=full_feedback,
        reset_state_after_warmup=reset,
        residual_alpha=1.0,
        reset_state_every_step=reset_every_step,
    )


def _values(dataset: xr.Dataset) -> np.ndarray:
    return np.asarray(dataset["x"]).reshape(-1)


def test_baseline_feedback_keeps_single_baseline_trajectory() -> None:
    result = _run(full_feedback=False)
    np.testing.assert_allclose(_values(result.baseline_prediction), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(_values(result.full_prediction), [3.0, 4.0, 5.0])


def test_full_feedback_uses_two_diverging_trajectories() -> None:
    result = _run(full_feedback=True)
    np.testing.assert_allclose(_values(result.baseline_prediction), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(_values(result.full_prediction), [3.0, 6.0, 9.0])


def test_truth_warmup_and_state_reset() -> None:
    continued = _run(full_feedback=False, warmup_steps=1, reset=False)
    reset = _run(full_feedback=False, warmup_steps=1, reset=True)
    np.testing.assert_allclose(_values(continued.baseline_prediction), [11.0])
    assert continued.residual_state["count"] == 2
    assert reset.residual_state["count"] == 1


def test_reset_state_every_step_prevents_temporal_carry() -> None:
    continued = _run(full_feedback=True, reset_every_step=False)
    reset = _run(full_feedback=True, reset_every_step=True)
    assert continued.residual_state["count"] == 3
    assert reset.residual_state["count"] == 1


def test_shift_inputs_retains_window_and_matching_forcing() -> None:
    inputs = _inputs()
    new_field = _dataset([7.0], start="2022-01-01T00:00")
    forcing = _forcings(1)
    shifted = shift_inputs_with_field(
        inputs,
        new_field,
        forcing,
        time_step=TIME_STEP,
        input_steps=2,
    )
    np.testing.assert_allclose(np.asarray(shifted["x"]).reshape(-1), [0.0, 7.0])
    np.testing.assert_allclose(np.asarray(shifted["forcing"]).reshape(-1), [0.0, 0.0])
