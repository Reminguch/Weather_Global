"""Device-side rollout and metric accumulation for resolution eval."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from graphcast import graphcast as gc, xarray_jax
try:
    from graphcast import losses as gc_losses
except Exception:  # pragma: no cover - graphcast env only.
    gc_losses = None

from src.models.graphcast.runtime import (
    _constant_inputs,
    _update_inputs,
    infer_family,
    load_run_config,
)
from src.models.graphcast.runtime import _build_one_step_bundle as build_graphcast_bundle
from src.models.graphcast.training.core.model import advance_residual_inputs, build_zero_residual_inputs
from src.models.mamba.gc_mamba.legacy_runtime import (
    _build_one_step_bundle as build_legacy_gc_mamba_bundle,
    is_legacy_gc_mamba_checkpoint,
)
from src.models.mamba.gc_mamba.runtime import _build_one_step_bundle as build_gc_mamba_bundle
from src.models.mamba.residual_mamba.runtime import _build_residual_rollout_bundle


@dataclasses.dataclass(frozen=True)
class _ModelBundle:
    params: hk.Params
    transformed: hk.TransformedWithState


@dataclasses.dataclass(frozen=True)
class DeviceRolloutBundle:
    family: str
    task_cfg: gc.TaskConfig
    model_cfg: gc.ModelConfig
    run_cfg: dict[str, Any]
    primary: _ModelBundle
    baseline: _ModelBundle | None = None


@dataclasses.dataclass(frozen=True)
class _VariableMetricSpec:
    name: str
    output_name: str
    dims: tuple[str, ...]
    lat_indices: jax.Array | None
    lon_indices: jax.Array | None
    lat_weights: jax.Array | None
    level_weights: jax.Array | None
    scale: jax.Array | None
    scale_dims: tuple[str, ...]
    variable_weight: float
    include_in_weighted: bool = True


@dataclasses.dataclass(frozen=True)
class DeviceMetricSpec:
    variables: tuple[_VariableMetricSpec, ...]
    max_lead_steps: int
    total_variable_weight: float


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.abs(source[None, :] - target[:, None]).argmin(axis=1).astype(np.int32)


def _latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    latitudes = np.asarray(latitudes, dtype=np.float64)
    dummy = xr.DataArray(np.zeros((latitudes.size,), dtype=np.float32), coords={"lat": latitudes}, dims=("lat",))
    if gc_losses is not None:
        try:
            return np.asarray(gc_losses.normalized_latitude_weights(dummy).values, dtype=np.float32)
        except Exception:
            pass
    if latitudes.size == 1:
        return np.ones((1,), dtype=np.float32)
    edges = np.empty(latitudes.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (latitudes[:-1] + latitudes[1:])
    edges[0] = latitudes[0] - (latitudes[1] - latitudes[0]) / 2.0
    edges[-1] = latitudes[-1] + (latitudes[-1] - latitudes[-2]) / 2.0
    edges = np.clip(edges, -90.0, 90.0)
    weights = np.abs(np.sin(np.deg2rad(edges[:-1])) - np.sin(np.deg2rad(edges[1:])))
    weights = weights.astype(np.float32)
    return weights / np.mean(weights)


def _select_scale_grid(
    scale: xr.DataArray,
    *,
    lat_indices: np.ndarray | None,
    lon_indices: np.ndarray | None,
    level_indices: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(scale.values, dtype=np.float32)
    dims = tuple(scale.dims)
    if level_indices is not None and "level" in dims:
        values = np.take(values, level_indices, axis=dims.index("level"))
    if lat_indices is not None and "lat" in dims:
        values = np.take(values, lat_indices, axis=dims.index("lat"))
    if lon_indices is not None and "lon" in dims:
        values = np.take(values, lon_indices, axis=dims.index("lon"))
    return values, dims


def build_metric_spec(
    targets: xr.Dataset,
    *,
    stats: dict[str, xr.Dataset],
    res_grid_lats: xr.DataArray,
    res_grid_lons: xr.DataArray,
    per_variable_weights: dict[str, float],
    max_lead_steps: int,
    nyc_lat: float | None = None,
    nyc_lon: float | None = None,
    nyc_output_name: str | None = None,
) -> DeviceMetricSpec:
    variables: list[_VariableMetricSpec] = []
    target_names = tuple(targets.data_vars)
    total_variable_weight = float(sum(float(per_variable_weights.get(name, 1.0)) for name in target_names))
    if total_variable_weight <= 0.0:
        raise ValueError("Total variable weight must be positive.")

    for name in target_names:
        target = targets[name]
        dims = tuple(target.dims)
        lat_indices_np: np.ndarray | None = None
        lon_indices_np: np.ndarray | None = None
        lat_weights_np: np.ndarray | None = None
        level_weights_np: np.ndarray | None = None
        if "lat" in dims:
            lat_indices_np = _nearest_indices(np.asarray(target.coords["lat"].values), np.asarray(res_grid_lats.values))
            selected_lats = np.asarray(target.coords["lat"].values)[lat_indices_np]
            lat_weights_np = _latitude_weights(selected_lats)
        if "lon" in dims:
            lon_indices_np = _nearest_indices(np.asarray(target.coords["lon"].values), np.asarray(res_grid_lons.values))
        if "level" in dims:
            levels_np = np.asarray(target.coords["level"].values, dtype=np.float32)
            level_weights_np = levels_np / np.mean(levels_np)

        scale = None
        scale_dims: tuple[str, ...] = ()
        if name in stats["diffs_stddev_by_level"]:
            level_indices_np = None
            scale_da = stats["diffs_stddev_by_level"][name]
            if "level" in dims and "level" in scale_da.dims:
                level_indices_np = _nearest_indices(
                    np.asarray(scale_da.coords["level"].values),
                    np.asarray(target.coords["level"].values),
                )
            scale_np, scale_dims = _select_scale_grid(
                scale_da,
                lat_indices=lat_indices_np,
                lon_indices=lon_indices_np,
                level_indices=level_indices_np,
            )
            scale = jnp.asarray(scale_np, dtype=jnp.float32)

        variables.append(
            _VariableMetricSpec(
                name=name,
                output_name=name,
                dims=dims,
                lat_indices=None if lat_indices_np is None else jnp.asarray(lat_indices_np, dtype=jnp.int32),
                lon_indices=None if lon_indices_np is None else jnp.asarray(lon_indices_np, dtype=jnp.int32),
                lat_weights=None if lat_weights_np is None else jnp.asarray(lat_weights_np, dtype=jnp.float32),
                level_weights=None if level_weights_np is None else jnp.asarray(level_weights_np, dtype=jnp.float32),
                scale=scale,
                scale_dims=scale_dims,
                variable_weight=float(per_variable_weights.get(name, 1.0)),
            )
        )

    if nyc_lat is not None and nyc_lon is not None and nyc_output_name and "2m_temperature" in targets:
        target = targets["2m_temperature"]
        dims = tuple(target.dims)
        if "lat" in dims and "lon" in dims:
            metric_lats = np.asarray(res_grid_lats.values, dtype=np.float64)
            metric_lons = np.asarray(res_grid_lons.values, dtype=np.float64)
            point_lat = metric_lats[np.abs(metric_lats - float(nyc_lat)).argmin()]
            point_lon = metric_lons[np.abs(metric_lons - float(nyc_lon)).argmin()]
            lat_indices_np = _nearest_indices(np.asarray(target.coords["lat"].values), np.asarray([point_lat]))
            lon_indices_np = _nearest_indices(np.asarray(target.coords["lon"].values), np.asarray([point_lon]))
            scale = None
            scale_dims: tuple[str, ...] = ()
            if "2m_temperature" in stats["diffs_stddev_by_level"]:
                scale_np, scale_dims = _select_scale_grid(
                    stats["diffs_stddev_by_level"]["2m_temperature"],
                    lat_indices=lat_indices_np,
                    lon_indices=lon_indices_np,
                    level_indices=None,
                )
                scale = jnp.asarray(scale_np, dtype=jnp.float32)
            variables.append(
                _VariableMetricSpec(
                    name="2m_temperature",
                    output_name=nyc_output_name,
                    dims=dims,
                    lat_indices=jnp.asarray(lat_indices_np, dtype=jnp.int32),
                    lon_indices=jnp.asarray(lon_indices_np, dtype=jnp.int32),
                    lat_weights=None,
                    level_weights=None,
                    scale=scale,
                    scale_dims=scale_dims,
                    variable_weight=0.0,
                    include_in_weighted=False,
                )
            )

    return DeviceMetricSpec(
        variables=tuple(variables),
        max_lead_steps=int(max_lead_steps),
        total_variable_weight=total_variable_weight,
    )


def _reshape_scale(scale: jax.Array, scale_dims: tuple[str, ...], dims: list[str]) -> jax.Array:
    shape: list[int] = []
    for dim in dims:
        if dim in scale_dims:
            shape.append(int(scale.shape[scale_dims.index(dim)]))
        else:
            shape.append(1)
    return jnp.reshape(scale, tuple(shape))


def _per_variable_loss(pred: xr.Dataset, target: xr.Dataset, spec: _VariableMetricSpec) -> jax.Array:
    pred_values = jnp.asarray(xarray_jax.unwrap(pred[spec.name].data), dtype=jnp.float32)
    target_values = jnp.asarray(xarray_jax.unwrap(target[spec.name].data), dtype=jnp.float32)
    loss = (pred_values - target_values) ** 2
    dims = list(spec.dims)

    if spec.lat_indices is not None and "lat" in dims:
        axis = dims.index("lat")
        loss = jnp.take(loss, spec.lat_indices, axis=axis)
    if spec.lon_indices is not None and "lon" in dims:
        axis = dims.index("lon")
        loss = jnp.take(loss, spec.lon_indices, axis=axis)

    if spec.scale is not None:
        loss = loss / (_reshape_scale(spec.scale, spec.scale_dims, dims) ** 2)

    if spec.lat_weights is not None and "lat" in dims:
        axis = dims.index("lat")
        weight_shape = [1] * loss.ndim
        weight_shape[axis] = int(spec.lat_weights.shape[0])
        weights = jnp.reshape(spec.lat_weights, tuple(weight_shape))
        loss = loss * weights

    if spec.level_weights is not None and "level" in dims:
        axis = dims.index("level")
        weight_shape = [1] * loss.ndim
        weight_shape[axis] = int(spec.level_weights.shape[0])
        weights = jnp.reshape(spec.level_weights, tuple(weight_shape))
        loss = loss * weights

    for dim in tuple(dims):
        if dim not in ("batch", "time"):
            axis = dims.index(dim)
            loss = jnp.mean(loss, axis=axis)
            dims.pop(axis)

    if dims == ["time", "batch"]:
        loss = jnp.swapaxes(loss, 0, 1)
        dims = ["batch", "time"]
    if dims != ["batch", "time"]:
        raise ValueError(f"Metric loss for {spec.name} reduced to unexpected dims {dims}.")
    return loss


def _per_variable_physical_mse(pred: xr.Dataset, target: xr.Dataset, spec: _VariableMetricSpec) -> jax.Array:
    pred_values = jnp.asarray(xarray_jax.unwrap(pred[spec.name].data), dtype=jnp.float32)
    target_values = jnp.asarray(xarray_jax.unwrap(target[spec.name].data), dtype=jnp.float32)
    loss = (pred_values - target_values) ** 2
    dims = list(spec.dims)

    if spec.lat_indices is not None and "lat" in dims:
        axis = dims.index("lat")
        loss = jnp.take(loss, spec.lat_indices, axis=axis)
    if spec.lon_indices is not None and "lon" in dims:
        axis = dims.index("lon")
        loss = jnp.take(loss, spec.lon_indices, axis=axis)

    if spec.lat_weights is not None and "lat" in dims:
        axis = dims.index("lat")
        weight_shape = [1] * loss.ndim
        weight_shape[axis] = int(spec.lat_weights.shape[0])
        weights = jnp.reshape(spec.lat_weights, tuple(weight_shape))
        loss = loss * weights

    for dim in tuple(dims):
        if dim not in ("batch", "time"):
            axis = dims.index(dim)
            loss = jnp.mean(loss, axis=axis)
            dims.pop(axis)

    if dims == ["time", "batch"]:
        loss = jnp.swapaxes(loss, 0, 1)
        dims = ["batch", "time"]
    if dims != ["batch", "time"]:
        raise ValueError(f"Physical MSE for {spec.name} reduced to unexpected dims {dims}.")
    return loss


def _empty_device_accumulator(
    metric_spec: DeviceMetricSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n_vars = len(metric_spec.variables)
    max_lead_steps = metric_spec.max_lead_steps
    return (
        jnp.zeros((max_lead_steps,), dtype=jnp.float32),
        jnp.zeros((max_lead_steps,), dtype=jnp.int32),
        jnp.zeros((n_vars, max_lead_steps), dtype=jnp.float32),
        jnp.zeros((n_vars, max_lead_steps), dtype=jnp.int32),
        jnp.zeros((n_vars, max_lead_steps), dtype=jnp.float32),
        jnp.zeros((n_vars, max_lead_steps), dtype=jnp.int32),
    )


def _accumulate_step(
    acc: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    pred_step: xr.Dataset,
    target_step: xr.Dataset,
    *,
    lead_i: int,
    metric_spec: DeviceMetricSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    weighted_sum, weighted_count, per_var_sum, per_var_count, physical_mse_sum, physical_mse_count = acc
    weighted_batch = jnp.zeros_like(_per_variable_loss(pred_step, target_step, metric_spec.variables[0]))
    for var_i, var_spec in enumerate(metric_spec.variables):
        loss_bt = _per_variable_loss(pred_step, target_step, var_spec)
        physical_mse_bt = _per_variable_physical_mse(pred_step, target_step, var_spec)
        if var_spec.include_in_weighted:
            weighted_batch = weighted_batch + loss_bt * var_spec.variable_weight
        per_var_sum = per_var_sum.at[var_i, lead_i].add(jnp.sum(loss_bt))
        per_var_count = per_var_count.at[var_i, lead_i].add(jnp.asarray(loss_bt.size, dtype=jnp.int32))
        physical_mse_sum = physical_mse_sum.at[var_i, lead_i].add(jnp.sum(physical_mse_bt))
        physical_mse_count = physical_mse_count.at[var_i, lead_i].add(
            jnp.asarray(physical_mse_bt.size, dtype=jnp.int32)
        )

    weighted_sum = weighted_sum.at[lead_i].add(jnp.sum(weighted_batch))
    weighted_count = weighted_count.at[lead_i].add(jnp.asarray(weighted_batch.size, dtype=jnp.int32))
    return weighted_sum, weighted_count, per_var_sum, per_var_count, physical_mse_sum, physical_mse_count


def _model_step_single(
    model: _ModelBundle,
    *,
    params: hk.Params,
    state: hk.State,
    rng: jax.Array,
    rolling_inputs: xr.Dataset,
    constant_inputs: xr.Dataset,
    target_step: xr.Dataset,
    forcings_step: xr.Dataset,
) -> tuple[xr.Dataset, hk.State]:
    all_inputs = xr.merge([constant_inputs, rolling_inputs])
    return model.transformed.apply(params, state, rng, all_inputs, target_step, forcings_step)


def _model_step_residual(
    residual: _ModelBundle,
    baseline: _ModelBundle,
    *,
    residual_params: hk.Params,
    baseline_params: hk.Params,
    residual_state: hk.State,
    baseline_state: hk.State,
    rng: tuple[jax.Array, jax.Array],
    rolling_inputs: xr.Dataset,
    constant_inputs: xr.Dataset,
    residual_inputs: xr.Dataset,
    target_step: xr.Dataset,
    forcings_step: xr.Dataset,
    teacher_forced: bool = True,
) -> tuple[xr.Dataset, hk.State, hk.State, xr.Dataset]:
    all_inputs = xr.merge([constant_inputs, rolling_inputs])
    baseline_pred, baseline_next = baseline.transformed.apply(
        baseline_params,
        baseline_state,
        rng[0],
        all_inputs,
        target_step,
        forcings_step,
    )
    residual_pred, residual_next = residual.transformed.apply(
        residual_params,
        residual_state,
        rng[1],
        residual_inputs,
        target_step,
        forcings_step,
    )
    residual_feedback = target_step - baseline_pred if teacher_forced else residual_pred
    return (
        baseline_pred + residual_pred,
        residual_next,
        baseline_next,
        advance_residual_inputs(residual_inputs, residual_feedback),
    )


def _build_bundle_from_checkpoint(ckpt_obj: gc.CheckPoint, stats: dict[str, xr.Dataset], ckpt_path: Path) -> DeviceRolloutBundle:
    run_cfg = load_run_config(ckpt_path)
    family = infer_family(run_cfg)
    if family == "graphcast":
        bundle = build_graphcast_bundle(ckpt_obj, stats, ckpt_path)
        return DeviceRolloutBundle(
            family=family,
            task_cfg=bundle["task_cfg"],
            model_cfg=bundle["model_cfg"],
            run_cfg=bundle["run_cfg"],
            primary=_ModelBundle(params=bundle["params"], transformed=bundle["transformed"]),
        )
    if family == "gc_mamba":
        builder = build_legacy_gc_mamba_bundle if is_legacy_gc_mamba_checkpoint(ckpt_obj) else build_gc_mamba_bundle
        bundle = builder(ckpt_obj, stats, ckpt_path)
        return DeviceRolloutBundle(
            family=family,
            task_cfg=bundle["task_cfg"],
            model_cfg=bundle["model_cfg"],
            run_cfg=bundle["run_cfg"],
            primary=_ModelBundle(params=bundle["params"], transformed=bundle["transformed"]),
        )
    if family == "residual_mamba":
        residual_bundle, baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)
        return DeviceRolloutBundle(
            family=family,
            task_cfg=residual_bundle["task_cfg"],
            model_cfg=residual_bundle["model_cfg"],
            run_cfg=residual_bundle["run_cfg"],
            primary=_ModelBundle(params=residual_bundle["params"], transformed=residual_bundle["transformed"]),
            baseline=_ModelBundle(params=baseline_bundle["params"], transformed=baseline_bundle["transformed"]),
        )
    raise ValueError(f"Unsupported device eval family: {family}")


class PreparedDeviceResolutionEvaluator:
    def __init__(
        self,
        ckpt_obj: gc.CheckPoint,
        stats: dict[str, xr.Dataset],
        ckpt_path: Path,
        *,
        res_grid_lats: xr.DataArray,
        res_grid_lons: xr.DataArray,
        per_variable_weights: dict[str, float],
        max_lead_steps: int,
        nyc_lat: float | None = None,
        nyc_lon: float | None = None,
        nyc_output_name: str | None = None,
        residual_eval_semantics: str = "teacher_forced_training_equivalent",
    ) -> None:
        self.bundle = _build_bundle_from_checkpoint(ckpt_obj, stats, ckpt_path)
        self._stats = stats
        self._res_grid_lats = res_grid_lats
        self._res_grid_lons = res_grid_lons
        self._per_variable_weights = per_variable_weights
        self._max_lead_steps = int(max_lead_steps)
        self._nyc_lat = nyc_lat
        self._nyc_lon = nyc_lon
        self._nyc_output_name = nyc_output_name
        self._residual_branch_teacher_forced = residual_eval_semantics == "teacher_forced_training_equivalent"
        self._metric_spec: DeviceMetricSpec | None = None
        self._state_cache: dict[tuple[str, int], Any] = {}
        self._cold_fn = None
        self._truth_step_fn = None
        self._branch_metric_fn = None

    @property
    def task_cfg(self) -> gc.TaskConfig:
        return self.bundle.task_cfg

    @property
    def run_cfg(self) -> dict[str, Any]:
        return self.bundle.run_cfg

    @property
    def metric_variable_names(self) -> tuple[str, ...]:
        if self._metric_spec is None:
            raise RuntimeError("Metric spec has not been initialized yet.")
        return tuple(var.output_name for var in self._metric_spec.variables)

    def _ensure_metric_spec(self, targets: xr.Dataset) -> DeviceMetricSpec:
        if self._metric_spec is None:
            self._metric_spec = build_metric_spec(
                targets,
                stats=self._stats,
                res_grid_lats=self._res_grid_lats,
                res_grid_lons=self._res_grid_lons,
                per_variable_weights=self._per_variable_weights,
                max_lead_steps=self._max_lead_steps,
                nyc_lat=self._nyc_lat,
                nyc_lon=self._nyc_lon,
                nyc_output_name=self._nyc_output_name,
            )
            self._cold_fn = self._build_cold_fn(self._metric_spec)
            self._truth_step_fn = self._build_truth_step_fn()
            self._branch_metric_fn = self._build_branch_metric_fn(self._metric_spec)
        return self._metric_spec

    def _initial_states(self, rng: jax.Array, inputs: xr.Dataset, targets: xr.Dataset, forcings: xr.Dataset):
        batch_size = int(inputs.sizes["batch"])
        key = (self.bundle.family, batch_size)
        if key in self._state_cache:
            return self._state_cache[key]
        target_step = targets.isel(time=slice(0, 1))
        forcings_step = forcings.isel(time=slice(0, 1))
        if self.bundle.baseline is None:
            _params, state = self.bundle.primary.transformed.init(rng, inputs, target_step, forcings_step)
            self._state_cache[key] = state
            return state
        residual_key, baseline_key = jax.random.split(rng)
        residual_inputs = build_zero_residual_inputs(inputs, target_step)
        _params, residual_state = self.bundle.primary.transformed.init(
            residual_key, residual_inputs, target_step, forcings_step
        )
        _params, baseline_state = self.bundle.baseline.transformed.init(baseline_key, inputs, target_step, forcings_step)
        self._state_cache[key] = (residual_state, baseline_state)
        return self._state_cache[key]

    def _build_cold_fn(self, metric_spec: DeviceMetricSpec):
        bundle = self.bundle

        @jax.jit
        def cold_fn(rng, inputs, targets, forcings, initial_states):
            constant_inputs = _constant_inputs(inputs, targets, forcings)
            rolling_inputs = inputs.drop_vars(constant_inputs.keys())
            acc = _empty_device_accumulator(metric_spec)
            if bundle.baseline is None:
                state = initial_states
                step_keys = jax.random.split(rng, metric_spec.max_lead_steps)
                for lead_i in range(metric_spec.max_lead_steps):
                    target_step = targets.isel(time=slice(lead_i, lead_i + 1))
                    forcings_step = forcings.isel(time=slice(lead_i, lead_i + 1))
                    pred_step, state = _model_step_single(
                        bundle.primary,
                        params=bundle.primary.params,
                        state=state,
                        rng=step_keys[lead_i],
                        rolling_inputs=rolling_inputs,
                        constant_inputs=constant_inputs,
                        target_step=target_step,
                        forcings_step=forcings_step,
                    )
                    acc = _accumulate_step(acc, pred_step, target_step, lead_i=lead_i, metric_spec=metric_spec)
                    rolling_inputs = _update_inputs(rolling_inputs, xr.merge([pred_step, forcings_step]))
                return acc

            residual_state, baseline_state = initial_states
            residual_inputs = build_zero_residual_inputs(inputs, targets.isel(time=slice(0, 1)))
            step_keys = jax.random.split(rng, metric_spec.max_lead_steps * 2)
            assert bundle.baseline is not None
            for lead_i in range(metric_spec.max_lead_steps):
                target_step = targets.isel(time=slice(lead_i, lead_i + 1))
                forcings_step = forcings.isel(time=slice(lead_i, lead_i + 1))
                pred_step, residual_state, baseline_state, residual_inputs = _model_step_residual(
                    bundle.primary,
                    bundle.baseline,
                    residual_params=bundle.primary.params,
                    baseline_params=bundle.baseline.params,
                    residual_state=residual_state,
                    baseline_state=baseline_state,
                    rng=(step_keys[2 * lead_i], step_keys[2 * lead_i + 1]),
                    rolling_inputs=rolling_inputs,
                    constant_inputs=constant_inputs,
                    residual_inputs=residual_inputs,
                    target_step=target_step,
                    forcings_step=forcings_step,
                    teacher_forced=self._residual_branch_teacher_forced,
                )
                acc = _accumulate_step(acc, pred_step, target_step, lead_i=lead_i, metric_spec=metric_spec)
                feedback_step = target_step if self._residual_branch_teacher_forced else pred_step
                rolling_inputs = _update_inputs(rolling_inputs, xr.merge([feedback_step, forcings_step]))
            return acc

        return cold_fn

    def _build_truth_step_fn(self):
        bundle = self.bundle

        @jax.jit
        def truth_step_fn(rng, rolling_inputs, residual_inputs, constant_inputs, target_step, forcings_step, states):
            if bundle.baseline is None:
                _pred_step, state = _model_step_single(
                    bundle.primary,
                    params=bundle.primary.params,
                    state=states,
                    rng=rng,
                    rolling_inputs=rolling_inputs,
                    constant_inputs=constant_inputs,
                    target_step=target_step,
                    forcings_step=forcings_step,
                )
                return _update_inputs(rolling_inputs, xr.merge([target_step, forcings_step])), residual_inputs, state

            assert bundle.baseline is not None
            residual_state, baseline_state = states
            _pred_step, residual_state, baseline_state, residual_inputs = _model_step_residual(
                bundle.primary,
                bundle.baseline,
                residual_params=bundle.primary.params,
                baseline_params=bundle.baseline.params,
                residual_state=residual_state,
                baseline_state=baseline_state,
                rng=rng,
                rolling_inputs=rolling_inputs,
                constant_inputs=constant_inputs,
                residual_inputs=residual_inputs,
                target_step=target_step,
                forcings_step=forcings_step,
                # Warm truth/context steps should always feed the true residual history.
                # The residual eval semantic only controls the scored branch below.
                teacher_forced=True,
            )
            return _update_inputs(rolling_inputs, xr.merge([target_step, forcings_step])), residual_inputs, (
                residual_state,
                baseline_state,
            )

        return truth_step_fn

    def _build_branch_metric_fn(self, metric_spec: DeviceMetricSpec):
        bundle = self.bundle

        @jax.jit
        def branch_metric_fn(rng, rolling_inputs, residual_inputs, constant_inputs, branch_targets, branch_forcings, states):
            acc = _empty_device_accumulator(metric_spec)
            if bundle.baseline is None:
                branch_rolling = rolling_inputs.copy(deep=False)
                branch_state = jax.tree_util.tree_map(lambda x: x, states)
                branch_keys = jax.random.split(rng, metric_spec.max_lead_steps)
                for lead_i in range(metric_spec.max_lead_steps):
                    target_step = branch_targets.isel(time=slice(lead_i, lead_i + 1))
                    forcings_step = branch_forcings.isel(time=slice(lead_i, lead_i + 1))
                    pred_step, branch_state = _model_step_single(
                        bundle.primary,
                        params=bundle.primary.params,
                        state=branch_state,
                        rng=branch_keys[lead_i],
                        rolling_inputs=branch_rolling,
                        constant_inputs=constant_inputs,
                        target_step=target_step,
                        forcings_step=forcings_step,
                    )
                    acc = _accumulate_step(acc, pred_step, target_step, lead_i=lead_i, metric_spec=metric_spec)
                    branch_rolling = _update_inputs(branch_rolling, xr.merge([pred_step, forcings_step]))
                return acc

            assert bundle.baseline is not None
            residual_state, baseline_state = states
            branch_rolling = rolling_inputs.copy(deep=False)
            branch_residual_inputs = residual_inputs.copy(deep=False)
            branch_residual_state = jax.tree_util.tree_map(lambda x: x, residual_state)
            branch_baseline_state = jax.tree_util.tree_map(lambda x: x, baseline_state)
            branch_keys = jax.random.split(rng, metric_spec.max_lead_steps * 2)
            for lead_i in range(metric_spec.max_lead_steps):
                target_step = branch_targets.isel(time=slice(lead_i, lead_i + 1))
                forcings_step = branch_forcings.isel(time=slice(lead_i, lead_i + 1))
                pred_step, branch_residual_state, branch_baseline_state, branch_residual_inputs = _model_step_residual(
                    bundle.primary,
                    bundle.baseline,
                    residual_params=bundle.primary.params,
                    baseline_params=bundle.baseline.params,
                    residual_state=branch_residual_state,
                    baseline_state=branch_baseline_state,
                    rng=(branch_keys[2 * lead_i], branch_keys[2 * lead_i + 1]),
                    rolling_inputs=branch_rolling,
                    constant_inputs=constant_inputs,
                    residual_inputs=branch_residual_inputs,
                    target_step=target_step,
                    forcings_step=forcings_step,
                    teacher_forced=self._residual_branch_teacher_forced,
                )
                acc = _accumulate_step(acc, pred_step, target_step, lead_i=lead_i, metric_spec=metric_spec)
                feedback_step = target_step if self._residual_branch_teacher_forced else pred_step
                branch_rolling = _update_inputs(branch_rolling, xr.merge([feedback_step, forcings_step]))
            return acc

        return branch_metric_fn

    def evaluate_cold_batch(self, rng: jax.Array, inputs: xr.Dataset, targets: xr.Dataset, forcings: xr.Dataset):
        metric_spec = self._ensure_metric_spec(targets)
        del metric_spec
        initial_states = self._initial_states(rng, inputs, targets, forcings)
        assert self._cold_fn is not None
        return self._cold_fn(rng, inputs, targets, forcings, initial_states)

    def evaluate_warm_chunk(
        self,
        rng: jax.Array,
        inputs: xr.Dataset,
        targets_by_anchor: tuple[xr.Dataset, ...],
        forcings_by_anchor: tuple[xr.Dataset, ...],
        *,
        warmup_steps: int,
        trunk_steps: int,
    ):
        metric_spec = self._ensure_metric_spec(targets_by_anchor[0])
        states = self._initial_states(rng, inputs, targets_by_anchor[0], forcings_by_anchor[0])
        constant_inputs = _constant_inputs(inputs, targets_by_anchor[0], forcings_by_anchor[0])
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        residual_inputs = (
            build_zero_residual_inputs(inputs, targets_by_anchor[0].isel(time=slice(0, 1)))
            if self.bundle.baseline is not None
            else None
        )
        acc = _empty_device_accumulator(metric_spec)
        assert self._truth_step_fn is not None
        assert self._branch_metric_fn is not None

        step_keys = jax.random.split(
            rng,
            2 * int(warmup_steps) + int(trunk_steps) * (2 + metric_spec.max_lead_steps * 2),
        )
        key_i = 0
        for step_i in range(int(warmup_steps)):
            target_step = targets_by_anchor[step_i].isel(time=slice(0, 1))
            forcings_step = forcings_by_anchor[step_i].isel(time=slice(0, 1))
            truth_rng = step_keys[key_i]
            if self.bundle.baseline is not None:
                truth_rng = (step_keys[key_i], step_keys[key_i + 1])
            rolling_inputs, residual_inputs, states = self._truth_step_fn(
                truth_rng,
                rolling_inputs,
                residual_inputs,
                constant_inputs,
                target_step,
                forcings_step,
                states,
            )
            key_i += 2

        for anchor_i in range(int(trunk_steps)):
            branch_targets = targets_by_anchor[int(warmup_steps) + anchor_i]
            branch_forcings = forcings_by_anchor[int(warmup_steps) + anchor_i]
            branch_acc = self._branch_metric_fn(
                step_keys[key_i],
                rolling_inputs,
                residual_inputs,
                constant_inputs,
                branch_targets,
                branch_forcings,
                states,
            )
            acc = tuple(lhs + rhs for lhs, rhs in zip(acc, branch_acc))
            key_i += 1

            truth_target = branch_targets.isel(time=slice(0, 1))
            truth_forcings = branch_forcings.isel(time=slice(0, 1))
            truth_rng = step_keys[key_i]
            if self.bundle.baseline is not None:
                truth_rng = (step_keys[key_i], step_keys[key_i + 1])
            rolling_inputs, residual_inputs, states = self._truth_step_fn(
                truth_rng,
                rolling_inputs,
                residual_inputs,
                constant_inputs,
                truth_target,
                truth_forcings,
                states,
            )
            key_i += 2
        return acc


def add_device_accumulator_to_host(
    host_acc: dict[str, object],
    device_acc,
    *,
    variable_names: tuple[str, ...],
) -> None:
    weighted_sum, weighted_count, per_var_sum, per_var_count, physical_mse_sum, physical_mse_count = jax.device_get(
        device_acc
    )
    host_acc["weighted_sum"] += np.asarray(weighted_sum, dtype=float)
    host_acc["weighted_count"] += np.asarray(weighted_count, dtype=int)
    per_variable_sum = host_acc["per_variable_sum"]
    per_variable_count = host_acc["per_variable_count"]
    physical_sum = host_acc["physical_mse_sum"]
    physical_count = host_acc["physical_mse_count"]
    for var_i, name in enumerate(variable_names):
        per_variable_sum.setdefault(name, np.zeros_like(host_acc["weighted_sum"]))
        per_variable_count.setdefault(name, np.zeros_like(host_acc["weighted_count"]))
        per_variable_sum[name] += np.asarray(per_var_sum[var_i], dtype=float)
        per_variable_count[name] += np.asarray(per_var_count[var_i], dtype=int)
        physical_sum.setdefault(name, np.zeros_like(host_acc["weighted_sum"]))
        physical_count.setdefault(name, np.zeros_like(host_acc["weighted_count"]))
        physical_sum[name] += np.asarray(physical_mse_sum[var_i], dtype=float)
        physical_count[name] += np.asarray(physical_mse_count[var_i], dtype=int)


__all__ = [
    "DeviceMetricSpec",
    "DeviceRolloutBundle",
    "PreparedDeviceResolutionEvaluator",
    "add_device_accumulator_to_host",
    "build_metric_spec",
]
