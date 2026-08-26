"""Legacy-compatible evaluation metrics for v24_Ilya."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
from graphcast import losses as graphcast_losses
from graphcast import normalization as graphcast_normalization


# Per-variable weights used by DeepMind GraphCast's original
# ``GraphCast.loss_and_predictions`` objective. Variables not listed here
# have unit weight.
ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}

METRIC_MERGE_STATE_FORMAT = "v24_Ilya_metric_merge_state_v1"

_MERGE_STATE_ARRAY_ATTRIBUTES = (
    "_graphcast_loss_sum_baseline",
    "_graphcast_loss_sum_full",
    "_graphcast_loss_count",
    "sum_sq_b",
    "sum_sq_f",
    "sum_abs_b",
    "sum_abs_f",
    "sum_dot_re",
    "sum_norm_r_sq",
    "sum_norm_e_sq",
    "n_per_var",
    "sum_sq_b_pl",
    "sum_sq_f_pl",
    "sum_abs_b_pl",
    "sum_abs_f_pl",
    "n_per_pl",
    "sum_err_cell_b",
    "sum_err_cell_f",
    "sum_err_cell_b_pl",
    "sum_err_cell_f_pl",
)


class V24IlyaMetricAccumulator:
    """Accumulate the metric schema emitted by v24_Ilya evaluation."""

    def __init__(
        self,
        target_steps: int,
        latitudes: np.ndarray,
        *,
        diffs_stddev_by_level: xr.Dataset | None = None,
        store_spatial_bias: bool = True,
    ):
        if target_steps <= 0:
            raise ValueError(f"target_steps must be positive, got {target_steps}")
        self.target_steps = target_steps
        self._latitudes = np.asarray(latitudes).copy()
        cos_lat = np.cos(np.deg2rad(self._latitudes))
        if cos_lat.ndim != 1 or cos_lat.size == 0:
            raise ValueError(f"Expected a non-empty 1-D latitude coordinate, got {cos_lat.shape}")
        self._cos_lat = cos_lat / cos_lat.mean()
        self._cos_lat_da = xr.DataArray(self._cos_lat, dims="lat")
        self._diffs_stddev_by_level = diffs_stddev_by_level
        self._has_original_graphcast_loss = diffs_stddev_by_level is not None
        self._store_spatial_bias = bool(store_spatial_bias)

        self._graphcast_loss_sum_baseline = np.zeros(
            target_steps,
            dtype=np.float64,
        )
        self._graphcast_loss_sum_full = np.zeros(target_steps, dtype=np.float64)
        self._graphcast_loss_count = np.zeros(target_steps, dtype=np.int64)

        self.sum_sq_b: dict[str, np.ndarray] = {}
        self.sum_sq_f: dict[str, np.ndarray] = {}
        self.sum_abs_b: dict[str, np.ndarray] = {}
        self.sum_abs_f: dict[str, np.ndarray] = {}
        self.sum_dot_re: dict[str, np.ndarray] = {}
        self.sum_norm_r_sq: dict[str, np.ndarray] = {}
        self.sum_norm_e_sq: dict[str, np.ndarray] = {}
        self.n_per_var: dict[str, np.ndarray] = {}

        self.sum_sq_b_pl: dict[str, dict[int, np.ndarray]] = {}
        self.sum_sq_f_pl: dict[str, dict[int, np.ndarray]] = {}
        self.sum_abs_b_pl: dict[str, dict[int, np.ndarray]] = {}
        self.sum_abs_f_pl: dict[str, dict[int, np.ndarray]] = {}
        self.n_per_pl: dict[str, dict[int, np.ndarray]] = {}

        self.sum_err_cell_b: dict[str, np.ndarray] = {}
        self.sum_err_cell_f: dict[str, np.ndarray] = {}
        self.sum_err_cell_b_pl: dict[str, dict[int, np.ndarray]] = {}
        self.sum_err_cell_f_pl: dict[str, dict[int, np.ndarray]] = {}

    def update(
        self,
        truth_dataset: xr.Dataset,
        baseline_dataset: xr.Dataset,
        full_dataset: xr.Dataset,
    ) -> None:
        expected_steps = self.target_steps
        for label, dataset in (
            ("truth", truth_dataset),
            ("baseline", baseline_dataset),
            ("full", full_dataset),
        ):
            if dataset.sizes.get("time", 0) != expected_steps:
                raise ValueError(
                    f"{label} has {dataset.sizes.get('time', 0)} time steps; "
                    f"expected {expected_steps}"
                )

        for step_index in range(self.target_steps):
            selection = {"time": slice(step_index, step_index + 1)}
            self.update_step(
                step_index,
                truth_dataset.isel(**selection),
                baseline_dataset.isel(**selection),
                full_dataset.isel(**selection),
            )

    def update_step(
        self,
        step_index: int,
        truth_dataset: xr.Dataset,
        baseline_dataset: xr.Dataset,
        full_dataset: xr.Dataset,
    ) -> None:
        """Accumulate one lead without retaining a complete rollout."""

        if step_index < 0 or step_index >= self.target_steps:
            raise ValueError(
                f"step_index={step_index} is outside 0..{self.target_steps - 1}"
            )
        for label, dataset in (
            ("truth", truth_dataset),
            ("baseline", baseline_dataset),
            ("full", full_dataset),
        ):
            if dataset.sizes.get("time", 0) != 1:
                raise ValueError(
                    f"{label} has {dataset.sizes.get('time', 0)} time steps; expected 1"
                )

        if self._diffs_stddev_by_level is not None:
            self._update_original_graphcast_loss_step(
                step_index,
                truth_dataset.astype("float32"),
                baseline_dataset.astype("float32"),
                full_dataset.astype("float32"),
            )

        for variable in truth_dataset.data_vars:
            if variable not in baseline_dataset or variable not in full_dataset:
                raise ValueError(f"Predictions are missing target variable {variable!r}")
            truth = truth_dataset[variable].isel(time=0).astype("float32", copy=False)
            baseline = baseline_dataset[variable].isel(time=0).astype("float32", copy=False)
            full = full_dataset[variable].isel(time=0).astype("float32", copy=False)
            self._ensure_variable(variable)

            baseline_error = baseline - truth
            full_error = full - truth
            residual = full - baseline
            pre_error = -baseline_error
            self.sum_sq_b[variable][step_index] += self._weighted_mean(baseline_error**2)
            self.sum_sq_f[variable][step_index] += self._weighted_mean(full_error**2)
            self.sum_abs_b[variable][step_index] += self._weighted_mean(np.abs(baseline_error))
            self.sum_abs_f[variable][step_index] += self._weighted_mean(np.abs(full_error))
            self.sum_dot_re[variable][step_index] += self._weighted_mean(residual * pre_error)
            self.sum_norm_r_sq[variable][step_index] += self._weighted_mean(residual * residual)
            self.sum_norm_e_sq[variable][step_index] += self._weighted_mean(pre_error * pre_error)

            if self._store_spatial_bias and "level" not in truth.dims:
                baseline_error_cells = self._lat_lon(baseline_error)
                full_error_cells = self._lat_lon(full_error)
                if variable not in self.sum_err_cell_b:
                    shape = (self.target_steps, *baseline_error_cells.shape)
                    self.sum_err_cell_b[variable] = np.zeros(shape, dtype=np.float32)
                    self.sum_err_cell_f[variable] = np.zeros(shape, dtype=np.float32)
                self.sum_err_cell_b[variable][step_index] += baseline_error_cells
                self.sum_err_cell_f[variable][step_index] += full_error_cells

            self.n_per_var[variable][step_index] += 1
            if "level" in truth.dims:
                self._update_pressure_levels_step(
                    variable,
                    step_index,
                    truth,
                    baseline,
                    full,
                )

    def _ensure_variable(self, variable: str) -> None:
        if variable in self.n_per_var:
            return
        zeros = lambda: np.zeros(self.target_steps)
        self.sum_sq_b[variable] = zeros()
        self.sum_sq_f[variable] = zeros()
        self.sum_abs_b[variable] = zeros()
        self.sum_abs_f[variable] = zeros()
        self.sum_dot_re[variable] = zeros()
        self.sum_norm_r_sq[variable] = zeros()
        self.sum_norm_e_sq[variable] = zeros()
        self.n_per_var[variable] = np.zeros(self.target_steps, dtype=np.int64)

    def _update_pressure_levels_step(
        self,
        variable: str,
        step_index: int,
        truth: xr.DataArray,
        baseline: xr.DataArray,
        full: xr.DataArray,
    ) -> None:
        if variable not in self.sum_sq_b_pl:
            self.sum_sq_b_pl[variable] = {}
            self.sum_sq_f_pl[variable] = {}
            self.sum_abs_b_pl[variable] = {}
            self.sum_abs_f_pl[variable] = {}
            self.n_per_pl[variable] = {}
            self.sum_err_cell_b_pl[variable] = {}
            self.sum_err_cell_f_pl[variable] = {}

        for level_index, level_value in enumerate(truth["level"].values):
            level = int(level_value)
            if level not in self.n_per_pl[variable]:
                self.sum_sq_b_pl[variable][level] = np.zeros(self.target_steps)
                self.sum_sq_f_pl[variable][level] = np.zeros(self.target_steps)
                self.sum_abs_b_pl[variable][level] = np.zeros(self.target_steps)
                self.sum_abs_f_pl[variable][level] = np.zeros(self.target_steps)
                self.n_per_pl[variable][level] = np.zeros(
                    self.target_steps,
                    dtype=np.int64,
                )

            truth_level = truth.isel(level=level_index)
            baseline_level = baseline.isel(level=level_index)
            full_level = full.isel(level=level_index)
            baseline_error = baseline_level - truth_level
            full_error = full_level - truth_level
            self.sum_sq_b_pl[variable][level][step_index] += self._weighted_mean(
                baseline_error**2
            )
            self.sum_sq_f_pl[variable][level][step_index] += self._weighted_mean(
                full_error**2
            )
            self.sum_abs_b_pl[variable][level][step_index] += self._weighted_mean(
                np.abs(baseline_error)
            )
            self.sum_abs_f_pl[variable][level][step_index] += self._weighted_mean(
                np.abs(full_error)
            )

            if self._store_spatial_bias:
                baseline_error_cells = self._lat_lon(baseline_error)
                full_error_cells = self._lat_lon(full_error)
                if level not in self.sum_err_cell_b_pl[variable]:
                    shape = (self.target_steps, *baseline_error_cells.shape)
                    self.sum_err_cell_b_pl[variable][level] = np.zeros(
                        shape, dtype=np.float32
                    )
                    self.sum_err_cell_f_pl[variable][level] = np.zeros(
                        shape, dtype=np.float32
                    )
                self.sum_err_cell_b_pl[variable][level][step_index] += (
                    baseline_error_cells
                )
                self.sum_err_cell_f_pl[variable][level][step_index] += full_error_cells
            self.n_per_pl[variable][level][step_index] += 1

    def finalize(
        self,
        *,
        include_rms_bias: bool | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return accumulated metrics.

        ``include_rms_bias=False`` avoids scanning the large spatial-bias
        accumulators.  It is intended for inexpensive progress snapshots;
        final evaluation outputs retain the full legacy RMS-bias schema.
        """

        if include_rms_bias is None:
            include_rms_bias = self._store_spatial_bias
        if include_rms_bias and not self._store_spatial_bias:
            raise ValueError(
                "RMS-bias metrics were requested without spatial bias accumulation"
            )
        if not self.n_per_var:
            raise ValueError("Cannot finalize v24_Ilya metrics without samples")
        output: dict[str, dict[str, Any]] = {
            "per_variable_per_step": {},
            "per_channel_per_step": {},
            "residual_diagnostics_per_variable": {},
        }

        if self._has_original_graphcast_loss:
            output["original_graphcast_loss"] = (
                self._finalize_original_graphcast_loss()
            )

        for variable in sorted(self.n_per_var):
            count = self.n_per_var[variable]
            self._require_complete_counts(variable, count)
            baseline_rmse = np.sqrt(self.sum_sq_b[variable] / count)
            full_rmse = np.sqrt(self.sum_sq_f[variable] / count)
            baseline_mae = self.sum_abs_b[variable] / count
            full_mae = self.sum_abs_f[variable] / count
            improvement_rmse = -(
                (full_rmse - baseline_rmse) / np.maximum(baseline_rmse, 1e-12) * 100
            )
            improvement_mae = -(
                (full_mae - baseline_mae) / np.maximum(baseline_mae, 1e-12) * 100
            )
            entry: dict[str, Any] = {
                "rmse_baseline": baseline_rmse.tolist(),
                "rmse_full": full_rmse.tolist(),
                "mae_baseline": baseline_mae.tolist(),
                "mae_full": full_mae.tolist(),
                "improvement_pct": improvement_rmse.tolist(),
                "improvement_pct_rmse": improvement_rmse.tolist(),
                "improvement_pct_mae": improvement_mae.tolist(),
            }
            if include_rms_bias and variable in self.sum_err_cell_b:
                denominator = count[:, None, None]
                baseline_rmsb = self._lat_weighted_rms(
                    self.sum_err_cell_b[variable] / denominator
                )
                full_rmsb = self._lat_weighted_rms(
                    self.sum_err_cell_f[variable] / denominator
                )
                improvement_rmsb = -(
                    (full_rmsb - baseline_rmsb) / np.maximum(baseline_rmsb, 1e-12) * 100
                )
                entry.update(
                    rmsb_baseline=baseline_rmsb.tolist(),
                    rmsb_full=full_rmsb.tolist(),
                    improvement_pct_rmsb=improvement_rmsb.tolist(),
                )
            output["per_variable_per_step"][variable] = entry
            output["residual_diagnostics_per_variable"][variable] = self._residual_diagnostics(
                variable,
                count,
            )

            if variable in self.sum_sq_b_pl:
                for level in sorted(self.sum_sq_b_pl[variable]):
                    output["per_channel_per_step"][f"{variable}_level{level}"] = (
                        self._pressure_level_entry(
                            variable,
                            level,
                            include_rms_bias=include_rms_bias,
                        )
                    )
            else:
                output["per_channel_per_step"][variable] = entry

        return output

    def export_merge_state(self) -> dict[str, Any]:
        """Return host sufficient statistics for exact cross-shard merging.

        The spatial bias arrays are intentionally included: without their
        signed per-grid-cell sums, RMS bias cannot be reconstructed exactly
        from independently finalized shard JSON files.
        """

        if not self.n_per_var:
            raise ValueError("Cannot export merge state without samples")
        return {
            "format": METRIC_MERGE_STATE_FORMAT,
            "target_steps": self.target_steps,
            "latitudes": self._latitudes,
            "has_original_graphcast_loss": self._has_original_graphcast_loss,
            "stores_spatial_bias": self._store_spatial_bias,
            "arrays": {
                name: getattr(self, name)
                for name in _MERGE_STATE_ARRAY_ATTRIBUTES
            },
        }

    @classmethod
    def from_merge_state(cls, state: Mapping[str, Any]) -> "V24IlyaMetricAccumulator":
        """Restore one exported state without copying its large host arrays."""

        cls._validate_merge_state_header(state)
        accumulator = cls(
            target_steps=int(state["target_steps"]),
            latitudes=np.asarray(state["latitudes"]),
            store_spatial_bias=bool(state.get("stores_spatial_bias", True)),
        )
        accumulator._has_original_graphcast_loss = bool(
            state["has_original_graphcast_loss"]
        )
        arrays = state["arrays"]
        for name in _MERGE_STATE_ARRAY_ATTRIBUTES:
            setattr(accumulator, name, arrays[name])
        accumulator._validate_restored_merge_arrays()
        return accumulator

    def merge_exported_state(self, state: Mapping[str, Any]) -> None:
        """Add another compatible shard's sufficient statistics in place."""

        other = self.from_merge_state(state)
        if other.target_steps != self.target_steps:
            raise ValueError(
                "Metric merge target_steps mismatch: "
                f"{other.target_steps} != {self.target_steps}"
            )
        if not np.array_equal(other._latitudes, self._latitudes):
            raise ValueError("Metric merge latitude coordinates differ")
        if other._has_original_graphcast_loss != self._has_original_graphcast_loss:
            raise ValueError("Metric merge GraphCast-loss availability differs")
        if other._store_spatial_bias != self._store_spatial_bias:
            raise ValueError("Metric merge spatial-bias policies differ")
        for name in _MERGE_STATE_ARRAY_ATTRIBUTES:
            _add_array_tree_in_place(
                getattr(self, name),
                getattr(other, name),
                path=name,
            )

    @staticmethod
    def _validate_merge_state_header(state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Metric merge state must be a mapping")
        if state.get("format") != METRIC_MERGE_STATE_FORMAT:
            raise ValueError(
                f"Unsupported metric merge state format {state.get('format')!r}"
            )
        target_steps = state.get("target_steps")
        if not isinstance(target_steps, int) or target_steps <= 0:
            raise ValueError("Metric merge state has invalid target_steps")
        latitudes = np.asarray(state.get("latitudes"))
        if latitudes.ndim != 1 or latitudes.size == 0:
            raise ValueError("Metric merge state has invalid latitudes")
        if not isinstance(state.get("has_original_graphcast_loss"), (bool, np.bool_)):
            raise ValueError("Metric merge state has invalid GraphCast-loss flag")
        if not isinstance(state.get("stores_spatial_bias", True), (bool, np.bool_)):
            raise ValueError("Metric merge state has invalid spatial-bias flag")
        arrays = state.get("arrays")
        if not isinstance(arrays, Mapping):
            raise ValueError("Metric merge state has no array mapping")
        missing = sorted(set(_MERGE_STATE_ARRAY_ATTRIBUTES) - set(arrays))
        extra = sorted(set(arrays) - set(_MERGE_STATE_ARRAY_ATTRIBUTES))
        if missing or extra:
            raise ValueError(
                f"Metric merge state array keys differ: missing={missing}, extra={extra}"
            )

    def _validate_restored_merge_arrays(self) -> None:
        if not self.n_per_var:
            raise ValueError("Metric merge state contains no variables")
        for name in _MERGE_STATE_ARRAY_ATTRIBUTES:
            _validate_array_tree(
                getattr(self, name),
                target_steps=self.target_steps,
                path=name,
            )

    @property
    def stores_spatial_bias(self) -> bool:
        return self._store_spatial_bias

    def _update_original_graphcast_loss_step(
        self,
        step_index: int,
        truth: xr.Dataset,
        baseline: xr.Dataset,
        full: xr.Dataset,
    ) -> None:
        """Accumulate the exact original GraphCast objective for one lead."""

        assert self._diffs_stddev_by_level is not None
        missing = sorted(
            set(truth.data_vars) - set(self._diffs_stddev_by_level.data_vars)
        )
        if missing:
            raise ValueError(
                "Original GraphCast loss is missing difference scales for "
                f"target variables: {missing}"
            )

        scales = self._diffs_stddev_by_level
        losses_by_branch: dict[str, np.ndarray] = {}
        for branch, prediction in (("baseline", baseline), ("full", full)):
            normalized_error = graphcast_normalization.normalize(
                prediction - truth,
                scales,
                None,
            )
            loss, _diagnostics = graphcast_losses.weighted_mse_per_level(
                normalized_error,
                xr.zeros_like(normalized_error),
                per_variable_weights={
                    variable: weight
                    for variable, weight in ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS.items()
                    if variable in normalized_error
                },
            )
            values = np.asarray(loss.values, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Non-finite original GraphCast {branch} loss at step "
                    f"{step_index}"
                )
            losses_by_branch[branch] = values

        if losses_by_branch["baseline"].shape != losses_by_branch["full"].shape:
            raise ValueError("Baseline and full original GraphCast loss shapes differ")
        count = losses_by_branch["baseline"].size
        self._graphcast_loss_sum_baseline[step_index] += float(
            losses_by_branch["baseline"].sum()
        )
        self._graphcast_loss_sum_full[step_index] += float(
            losses_by_branch["full"].sum()
        )
        self._graphcast_loss_count[step_index] += count

    def _finalize_original_graphcast_loss(self) -> dict[str, Any]:
        if np.any(self._graphcast_loss_count <= 0):
            raise ValueError(
                "Original GraphCast loss has no samples for one or more leads"
            )
        baseline = self._graphcast_loss_sum_baseline / self._graphcast_loss_count
        full = self._graphcast_loss_sum_full / self._graphcast_loss_count
        if np.any(baseline <= 0.0):
            raise ValueError("Original GraphCast baseline loss must be positive")
        improvement = 100.0 * (1.0 - full / baseline)
        baseline_rollout = float(np.mean(baseline))
        full_rollout = float(np.mean(full))
        return {
            "definition": (
                "DeepMind GraphCast normalized latitude- and "
                "pressure-level-weighted MSE"
            ),
            "baseline_per_step": baseline.tolist(),
            "full_per_step": full.tolist(),
            "improvement_pct_per_step": improvement.tolist(),
            "baseline_rollout": baseline_rollout,
            "full_rollout": full_rollout,
            "improvement_pct_rollout": 100.0
            * (1.0 - full_rollout / baseline_rollout),
        }

    def _pressure_level_entry(
        self,
        variable: str,
        level: int,
        *,
        include_rms_bias: bool = True,
    ) -> dict[str, Any]:
        count = self.n_per_pl[variable][level]
        self._require_complete_counts(f"{variable}_level{level}", count)
        baseline_rmse = np.sqrt(self.sum_sq_b_pl[variable][level] / count)
        full_rmse = np.sqrt(self.sum_sq_f_pl[variable][level] / count)
        baseline_mae = self.sum_abs_b_pl[variable][level] / count
        full_mae = self.sum_abs_f_pl[variable][level] / count
        improvement_rmse = -(
            (full_rmse - baseline_rmse) / np.maximum(baseline_rmse, 1e-12) * 100
        )
        improvement_mae = -(
            (full_mae - baseline_mae) / np.maximum(baseline_mae, 1e-12) * 100
        )
        entry: dict[str, Any] = {
            "rmse_baseline": baseline_rmse.tolist(),
            "rmse_full": full_rmse.tolist(),
            "mae_baseline": baseline_mae.tolist(),
            "mae_full": full_mae.tolist(),
            "improvement_pct": improvement_rmse.tolist(),
            "improvement_pct_rmse": improvement_rmse.tolist(),
            "improvement_pct_mae": improvement_mae.tolist(),
        }
        if include_rms_bias and level in self.sum_err_cell_b_pl[variable]:
            baseline_rmsb = self._lat_weighted_rms(
                self.sum_err_cell_b_pl[variable][level] / count[:, None, None]
            )
            full_rmsb = self._lat_weighted_rms(
                self.sum_err_cell_f_pl[variable][level] / count[:, None, None]
            )
            improvement_rmsb = -(
                (full_rmsb - baseline_rmsb) / np.maximum(baseline_rmsb, 1e-12) * 100
            )
            entry.update(
                rmsb_baseline=baseline_rmsb.tolist(),
                rmsb_full=full_rmsb.tolist(),
                improvement_pct_rmsb=improvement_rmsb.tolist(),
            )
        return entry

    def _residual_diagnostics(self, variable: str, count: np.ndarray) -> dict[str, Any]:
        dot_product = self.sum_dot_re[variable] / count
        residual_norm_sq = self.sum_norm_r_sq[variable] / count
        baseline_error_norm_sq = self.sum_norm_e_sq[variable] / count
        residual_norm = np.sqrt(np.maximum(residual_norm_sq, 0.0))
        baseline_error_norm = np.sqrt(np.maximum(baseline_error_norm_sq, 0.0))
        cosine = dot_product / np.maximum(residual_norm * baseline_error_norm, 1e-12)
        gain = residual_norm / np.maximum(baseline_error_norm, 1e-12)
        delta_mse = 2.0 * dot_product - residual_norm_sq
        return {
            "residual_cosine_by_lead": cosine.tolist(),
            "residual_gain_by_lead": gain.tolist(),
            "residual_norm_by_lead": residual_norm.tolist(),
            "baseline_err_norm_by_lead": baseline_error_norm.tolist(),
            "dot_r_epre_by_lead": dot_product.tolist(),
            "delta_mse_by_lead": delta_mse.tolist(),
            "E_pre_mse_by_lead": baseline_error_norm_sq.tolist(),
            "E_post_mse_by_lead": (baseline_error_norm_sq - delta_mse).tolist(),
        }

    def _weighted_mean(self, value: xr.DataArray) -> float:
        return float((value * self._cos_lat_da).mean().values)

    @staticmethod
    def _lat_lon(value: xr.DataArray) -> np.ndarray:
        array = np.asarray(
            value.transpose(..., "lat", "lon").values,
            dtype=np.float32,
        )
        while array.ndim > 2:
            array = array.squeeze(0)
        return array

    @staticmethod
    def _require_complete_counts(label: str, count: np.ndarray) -> None:
        missing = np.flatnonzero(count <= 0)
        if missing.size:
            raise ValueError(
                f"Cannot finalize {label}: no observations for lead indices "
                f"{missing.tolist()}"
            )

    def _lat_weighted_rms(self, bias_cells: np.ndarray) -> np.ndarray:
        weighted_mean = (bias_cells**2 * self._cos_lat[None, :, None]).mean(axis=(1, 2))
        return np.sqrt(weighted_mean)


def _validate_array_tree(value: Any, *, target_steps: int, path: str) -> None:
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != target_steps:
            raise ValueError(
                f"Metric merge array {path} has shape {value.shape}; "
                f"expected leading dimension {target_steps}"
            )
        if value.dtype.kind not in "fiu":
            raise ValueError(
                f"Metric merge array {path} has unsupported dtype {value.dtype}"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_array_tree(
                child,
                target_steps=target_steps,
                path=f"{path}.{key}",
            )
        return
    raise ValueError(
        f"Metric merge value {path} must be an array or mapping, got "
        f"{type(value).__name__}"
    )


def _add_array_tree_in_place(destination: Any, source: Any, *, path: str) -> None:
    if isinstance(destination, np.ndarray):
        if not isinstance(source, np.ndarray):
            raise ValueError(f"Metric merge type mismatch at {path}")
        if destination.shape != source.shape:
            raise ValueError(
                f"Metric merge shape mismatch at {path}: "
                f"{source.shape} != {destination.shape}"
            )
        if destination.dtype != source.dtype:
            raise ValueError(
                f"Metric merge dtype mismatch at {path}: "
                f"{source.dtype} != {destination.dtype}"
            )
        np.add(destination, source, out=destination)
        return
    if isinstance(destination, Mapping):
        if not isinstance(source, Mapping):
            raise ValueError(f"Metric merge type mismatch at {path}")
        if set(destination) != set(source):
            raise ValueError(
                f"Metric merge keys differ at {path}: "
                f"destination={sorted(destination, key=str)}, "
                f"source={sorted(source, key=str)}"
            )
        for key in destination:
            _add_array_tree_in_place(
                destination[key],
                source[key],
                path=f"{path}.{key}",
            )
        return
    raise ValueError(
        f"Metric merge value {path} must be an array or mapping, got "
        f"{type(destination).__name__}"
    )
