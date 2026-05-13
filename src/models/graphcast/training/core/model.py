from __future__ import annotations

import warnings
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from . import bootstrap as _bootstrap  # noqa: F401


def _require_graphcast() -> None:
    try:
        import graphcast  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "graphcast is required. Activate env via scripts/graphcast_env.sh or install graphcast+jax+haiku."
        ) from exc



_require_graphcast()
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast as gc,
    losses as gc_losses,
    normalization,
    predictor_base,
    xarray_tree,
    xarray_jax,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"graphcast\.autoregressive",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The return type of `Dataset\.dims` will be changed.*",
)

_ORIG_NORMALIZED_LATITUDE_WEIGHTS = gc_losses.normalized_latitude_weights


def _fallback_normalized_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """Area weights for any uniformly spaced latitude vector (with or without poles)."""
    latitude = data.coords["lat"]
    lat_vals = np.asarray(latitude.values, dtype=np.float64)
    if lat_vals.ndim != 1 or lat_vals.size < 2:
        raise ValueError(f"Expected 1D latitude with at least 2 points; got shape={lat_vals.shape}")

    diffs = np.diff(lat_vals)
    if not np.all(np.isclose(diffs, diffs[0], atol=1e-6)):
        raise ValueError(f"Latitude vector is not uniformly spaced: {latitude}")
    delta = float(diffs[0])

    edges = np.empty(lat_vals.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lat_vals[:-1] + lat_vals[1:])
    edges[0] = lat_vals[0] - (delta / 2.0)
    edges[-1] = lat_vals[-1] + (delta / 2.0)
    edges = np.clip(edges, -90.0, 90.0)

    weights_np = np.abs(np.sin(np.deg2rad(edges[:-1])) - np.sin(np.deg2rad(edges[1:])))
    weights = xr.DataArray(weights_np, coords=latitude.coords, dims=latitude.dims).astype(np.float32)
    return weights / weights.mean(skipna=False)


def _normalized_latitude_weights_with_fallback(data: xr.DataArray) -> xr.DataArray:
    try:
        return _ORIG_NORMALIZED_LATITUDE_WEIGHTS(data)
    except ValueError as exc:
        if "does not start/end" not in str(exc):
            raise
        return _fallback_normalized_latitude_weights(data)


def load_stats(stats_dir: Path) -> dict[str, xr.Dataset]:
    def open_nc(name: str) -> xr.Dataset:
        return xr.open_dataset(stats_dir / f"{name}.nc")

    return {
        "stddev_by_level": open_nc("stddev_by_level"),
        "mean_by_level": open_nc("mean_by_level"),
        "diffs_stddev_by_level": open_nc("diffs_stddev_by_level"),
    }


def load_graphcast_checkpoint(path: Path) -> gc.CheckPoint:
    with path.open("rb") as f:
        return checkpoint.load(f, gc.CheckPoint)


class DirectResidualNormalizer(predictor_base.Predictor):
    """Normalizes inputs normally and target corrections as direct residual fields."""

    def __init__(
        self,
        predictor: predictor_base.Predictor,
        *,
        stddev_by_level: xr.Dataset,
        mean_by_level: xr.Dataset,
        diffs_stddev_by_level: xr.Dataset,
    ):
        self._predictor = predictor
        self._scales = stddev_by_level
        self._locations = mean_by_level
        self._residual_scales = diffs_stddev_by_level
        self._residual_locations = None

    def _normalize_inputs(self, inputs: xr.Dataset) -> xr.Dataset:
        residual_vars = [name for name in inputs.data_vars if name in self._residual_scales.data_vars]
        normal_vars = [name for name in inputs.data_vars if name not in residual_vars]
        normalized_parts = []
        if normal_vars:
            normalized_parts.append(normalization.normalize(inputs[normal_vars], self._scales, self._locations))
        if residual_vars:
            normalized_parts.append(
                normalization.normalize(inputs[residual_vars], self._residual_scales, self._residual_locations)
            )
        if not normalized_parts:
            return xr.Dataset(coords=inputs.coords)
        return xr.merge(normalized_parts)

    def __call__(
        self,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
        **kwargs,
    ) -> xr.Dataset:
        norm_inputs = self._normalize_inputs(inputs)
        norm_forcings = normalization.normalize(forcings, self._scales, self._locations)
        norm_predictions = self._predictor(
            norm_inputs,
            targets_template,
            forcings=norm_forcings,
            **kwargs,
        )
        return xarray_tree.map_structure(
            lambda pred: normalization.unnormalize(pred, self._residual_scales, self._residual_locations),
            norm_predictions,
        )

    def loss(
        self,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        forcings: xr.Dataset,
        **kwargs,
    ) -> predictor_base.LossAndDiagnostics:
        norm_inputs = self._normalize_inputs(inputs)
        norm_forcings = normalization.normalize(forcings, self._scales, self._locations)
        norm_targets = normalization.normalize(targets, self._residual_scales, self._residual_locations)
        return self._predictor.loss(norm_inputs, norm_targets, forcings=norm_forcings, **kwargs)

    def loss_and_predictions(
        self,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        forcings: xr.Dataset,
        **kwargs,
    ) -> predictor_base.LossAndDiagnostics:
        norm_inputs = self._normalize_inputs(inputs)
        norm_forcings = normalization.normalize(forcings, self._scales, self._locations)
        norm_targets = normalization.normalize(targets, self._residual_scales, self._residual_locations)
        (loss, scalars), norm_predictions = self._predictor.loss_and_predictions(
            norm_inputs,
            norm_targets,
            forcings=norm_forcings,
            **kwargs,
        )
        predictions = xarray_tree.map_structure(
            lambda pred: normalization.unnormalize(pred, self._residual_scales, self._residual_locations),
            norm_predictions,
        )
        return (loss, scalars), predictions


def build_zero_residual_inputs(inputs: xr.Dataset, targets_template: xr.Dataset) -> xr.Dataset:
    """Return an input-shaped dataset whose prognostic variables are residual-valued zeros."""
    residual_vars = [name for name in inputs.data_vars if name in targets_template.data_vars]
    constant_vars = [name for name in inputs.data_vars if name not in residual_vars]
    pieces = []
    if constant_vars:
        pieces.append(inputs[constant_vars])
    if residual_vars:
        pieces.append(xr.Dataset({name: xr.zeros_like(inputs[name]) for name in residual_vars}))
    return xr.merge(pieces) if pieces else xr.Dataset(coords=inputs.coords)


def reset_residual_input_lanes(
    residual_inputs: xr.Dataset,
    targets_template: xr.Dataset,
    reset_mask: jax.Array,
) -> xr.Dataset:
    """Zero residual input history for lanes that start a new segment."""
    residual_vars = [name for name in residual_inputs.data_vars if name in targets_template.data_vars]
    if not residual_vars:
        return residual_inputs
    pieces = [residual_inputs.drop_vars(residual_vars)]
    reset_mask = jnp.asarray(reset_mask, dtype=bool)
    reset_vars = {}
    for name in residual_vars:
        data_array = residual_inputs[name]
        if "batch" not in data_array.dims:
            raise ValueError(f"Residual input variable {name!r} must have a 'batch' dimension.")
        batch_axis = data_array.dims.index("batch")
        data = xarray_jax.unwrap_data(data_array)
        mask_shape = [1] * data.ndim
        mask_shape[batch_axis] = reset_mask.shape[0]
        reset_by_batch = jnp.reshape(reset_mask, mask_shape)
        reset_data = jnp.where(reset_by_batch, jnp.zeros_like(data), data)
        reset_vars[name] = xr.DataArray(
            xarray_jax.wrap(reset_data),
            dims=data_array.dims,
            coords=data_array.coords,
            attrs=data_array.attrs,
            name=data_array.name,
        )
    pieces.append(xr.Dataset(reset_vars))
    return xr.merge(pieces)


def advance_residual_inputs(residual_inputs: xr.Dataset, residual_targets: xr.Dataset) -> xr.Dataset:
    """Shift residual target predictions into the next input window."""
    residual_vars = [name for name in residual_inputs.data_vars if name in residual_targets.data_vars]
    if not residual_vars:
        return residual_inputs
    num_input_times = residual_inputs.sizes["time"]
    updated = (
        xr.concat([residual_inputs[residual_vars], residual_targets[residual_vars]], dim="time")
        .tail(time=num_input_times)
        .assign_coords(time=residual_inputs.coords["time"])
    )
    return xr.merge([residual_inputs.drop_vars(residual_vars), updated])


def build_predictor(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    stats: dict[str, xr.Dataset],
    *,
    use_bf16: bool,
    gradient_checkpointing: bool,
    temporal_backbone: str,
    temporal_location: str,
    temporal_d_inner: int | None,
    temporal_d_state: int,
    temporal_d_conv: int,
    temporal_dt_rank: str,
    temporal_bias: bool,
    temporal_conv_bias: bool,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool = False,
    temporal_insert_count: int | None = None,
    zero_init_temporal_out: bool = False,
):
    predictor = gc.GraphCast(model_cfg, task_cfg)
    if hasattr(predictor, "_temporal_backbone"):
        predictor._temporal_backbone = temporal_backbone
        predictor._temporal_location = temporal_location
        predictor._temporal_stateful = temporal_stateful
        predictor._temporal_d_inner = temporal_d_inner
        predictor._temporal_d_state = temporal_d_state
        predictor._temporal_d_conv = temporal_d_conv
        predictor._temporal_dt_rank = temporal_dt_rank
        predictor._temporal_bias = temporal_bias
        predictor._temporal_conv_bias = temporal_conv_bias
        predictor._temporal_layers = temporal_layers
        predictor._temporal_dropout = temporal_dropout
        predictor._temporal_insert_count = temporal_insert_count
        predictor._temporal_zero_init_out = zero_init_temporal_out
    if use_bf16:
        predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=gradient_checkpointing)
    return predictor


def build_residual_correction_predictor(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    stats: dict[str, xr.Dataset],
    *,
    use_bf16: bool,
    gradient_checkpointing: bool,
    temporal_backbone: str,
    temporal_location: str,
    temporal_d_inner: int | None,
    temporal_d_state: int,
    temporal_d_conv: int,
    temporal_dt_rank: str,
    temporal_bias: bool,
    temporal_conv_bias: bool,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool = False,
    temporal_insert_count: int | None = None,
    zero_init_temporal_out: bool = False,
):
    predictor = gc.GraphCast(model_cfg, task_cfg)
    if hasattr(predictor, "_temporal_backbone"):
        predictor._temporal_backbone = temporal_backbone
        predictor._temporal_location = temporal_location
        predictor._temporal_stateful = temporal_stateful
        predictor._temporal_d_inner = temporal_d_inner
        predictor._temporal_d_state = temporal_d_state
        predictor._temporal_d_conv = temporal_d_conv
        predictor._temporal_dt_rank = temporal_dt_rank
        predictor._temporal_bias = temporal_bias
        predictor._temporal_conv_bias = temporal_conv_bias
        predictor._temporal_layers = temporal_layers
        predictor._temporal_dropout = temporal_dropout
        predictor._temporal_insert_count = temporal_insert_count
        predictor._temporal_zero_init_out = zero_init_temporal_out
    if use_bf16:
        predictor = casting.Bfloat16Cast(predictor)
    predictor = DirectResidualNormalizer(
        predictor,
        stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=gradient_checkpointing)
    return predictor


def build_loss_transform(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    stats: dict[str, xr.Dataset],
    *,
    use_bf16: bool,
    gradient_checkpointing: bool,
    temporal_backbone: str,
    temporal_location: str,
    temporal_d_inner: int | None,
    temporal_d_state: int,
    temporal_d_conv: int,
    temporal_dt_rank: str,
    temporal_bias: bool,
    temporal_conv_bias: bool,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool = False,
    temporal_insert_count: int | None = None,
    zero_init_temporal_out: bool = False,
) -> hk.TransformedWithState:
    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            stats,
            use_bf16=use_bf16,
            gradient_checkpointing=gradient_checkpointing,
            temporal_backbone=temporal_backbone,
            temporal_location=temporal_location,
            temporal_d_inner=temporal_d_inner,
            temporal_d_state=temporal_d_state,
            temporal_d_conv=temporal_d_conv,
            temporal_dt_rank=temporal_dt_rank,
            temporal_bias=temporal_bias,
            temporal_conv_bias=temporal_conv_bias,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            temporal_stateful=temporal_stateful,
            temporal_insert_count=temporal_insert_count,
            zero_init_temporal_out=zero_init_temporal_out,
        )
        return predictor.loss(inputs, targets, forcings)

    return hk.transform_with_state(forward_fn)


def build_prediction_transform(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    stats: dict[str, xr.Dataset],
    *,
    use_bf16: bool,
    gradient_checkpointing: bool,
    temporal_backbone: str,
    temporal_location: str,
    temporal_d_inner: int | None,
    temporal_d_state: int,
    temporal_d_conv: int,
    temporal_dt_rank: str,
    temporal_bias: bool,
    temporal_conv_bias: bool,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool = False,
    temporal_insert_count: int | None = None,
    zero_init_temporal_out: bool = False,
) -> hk.TransformedWithState:
    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            stats,
            use_bf16=use_bf16,
            gradient_checkpointing=gradient_checkpointing,
            temporal_backbone=temporal_backbone,
            temporal_location=temporal_location,
            temporal_d_inner=temporal_d_inner,
            temporal_d_state=temporal_d_state,
            temporal_d_conv=temporal_d_conv,
            temporal_dt_rank=temporal_dt_rank,
            temporal_bias=temporal_bias,
            temporal_conv_bias=temporal_conv_bias,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            temporal_stateful=temporal_stateful,
            temporal_insert_count=temporal_insert_count,
            zero_init_temporal_out=zero_init_temporal_out,
        )
        return predictor(inputs, targets, forcings)

    return hk.transform_with_state(forward_fn)


def validate_stats_coverage(task_cfg: gc.TaskConfig, stats: dict[str, xr.Dataset]) -> None:
    required_inputs = set(task_cfg.input_variables) | set(task_cfg.forcing_variables)
    required_targets = set(task_cfg.target_variables)

    stddev_vars = set(stats["stddev_by_level"].data_vars)
    mean_vars = set(stats["mean_by_level"].data_vars)
    diffs_vars = set(stats["diffs_stddev_by_level"].data_vars)

    missing_stddev = sorted(required_inputs - stddev_vars)
    missing_mean = sorted(required_inputs - mean_vars)
    missing_diffs = sorted(required_targets - diffs_vars)

    if missing_stddev or missing_mean or missing_diffs:
        raise ValueError(
            "Normalization stats missing required variables: "
            f"stddev_missing={missing_stddev}, "
            f"mean_missing={missing_mean}, "
            f"diffs_stddev_missing={missing_diffs}"
        )


def scalarize_loss(loss_da: xr.DataArray) -> jax.Array:
    return jnp.mean(xarray_jax.unwrap_data(loss_da))



gc_losses.normalized_latitude_weights = _normalized_latitude_weights_with_fallback
