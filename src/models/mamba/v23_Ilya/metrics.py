"""Legacy-compatible evaluation metrics for v23_Ilya."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


class V23IlyaMetricAccumulator:
    """Accumulate the metric schema emitted by v23_Ilya evaluation."""

    def __init__(self, target_steps: int, latitudes: np.ndarray):
        if target_steps <= 0:
            raise ValueError(f"target_steps must be positive, got {target_steps}")
        self.target_steps = target_steps
        cos_lat = np.cos(np.deg2rad(np.asarray(latitudes)))
        if cos_lat.ndim != 1 or cos_lat.size == 0:
            raise ValueError(f"Expected a non-empty 1-D latitude coordinate, got {cos_lat.shape}")
        self._cos_lat = cos_lat / cos_lat.mean()
        self._cos_lat_da = xr.DataArray(self._cos_lat, dims="lat")

        self.sum_sq_b: dict[str, np.ndarray] = {}
        self.sum_sq_f: dict[str, np.ndarray] = {}
        self.sum_abs_b: dict[str, np.ndarray] = {}
        self.sum_abs_f: dict[str, np.ndarray] = {}
        self.sum_dot_re: dict[str, np.ndarray] = {}
        self.sum_norm_r_sq: dict[str, np.ndarray] = {}
        self.sum_norm_e_sq: dict[str, np.ndarray] = {}
        self.n_per_var: dict[str, int] = {}

        self.sum_sq_b_pl: dict[str, dict[int, np.ndarray]] = {}
        self.sum_sq_f_pl: dict[str, dict[int, np.ndarray]] = {}
        self.sum_abs_b_pl: dict[str, dict[int, np.ndarray]] = {}
        self.sum_abs_f_pl: dict[str, dict[int, np.ndarray]] = {}
        self.n_per_pl: dict[str, dict[int, int]] = {}

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

        for variable in truth_dataset.data_vars:
            if variable not in baseline_dataset or variable not in full_dataset:
                raise ValueError(f"Predictions are missing target variable {variable!r}")
            truth = truth_dataset[variable].astype("float32")
            baseline = baseline_dataset[variable].astype("float32")
            full = full_dataset[variable].astype("float32")
            self._ensure_variable(variable)

            for step_index in range(self.target_steps):
                baseline_error = baseline.isel(time=step_index) - truth.isel(time=step_index)
                full_error = full.isel(time=step_index) - truth.isel(time=step_index)
                residual = full.isel(time=step_index) - baseline.isel(time=step_index)
                pre_error = -baseline_error
                self.sum_sq_b[variable][step_index] += self._weighted_mean(baseline_error**2)
                self.sum_sq_f[variable][step_index] += self._weighted_mean(full_error**2)
                self.sum_abs_b[variable][step_index] += self._weighted_mean(np.abs(baseline_error))
                self.sum_abs_f[variable][step_index] += self._weighted_mean(np.abs(full_error))
                self.sum_dot_re[variable][step_index] += self._weighted_mean(residual * pre_error)
                self.sum_norm_r_sq[variable][step_index] += self._weighted_mean(residual * residual)
                self.sum_norm_e_sq[variable][step_index] += self._weighted_mean(pre_error * pre_error)

            if "level" not in truth.dims:
                baseline_error_cells = self._time_lat_lon(baseline - truth)
                full_error_cells = self._time_lat_lon(full - truth)
                self.sum_err_cell_b.setdefault(variable, np.zeros_like(baseline_error_cells))
                self.sum_err_cell_f.setdefault(variable, np.zeros_like(full_error_cells))
                self.sum_err_cell_b[variable] += baseline_error_cells
                self.sum_err_cell_f[variable] += full_error_cells

            self.n_per_var[variable] += 1
            if "level" in truth.dims:
                self._update_pressure_levels(variable, truth, baseline, full)

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
        self.n_per_var[variable] = 0

    def _update_pressure_levels(
        self,
        variable: str,
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
                self.n_per_pl[variable][level] = 0

            truth_level = truth.isel(level=level_index)
            baseline_level = baseline.isel(level=level_index)
            full_level = full.isel(level=level_index)
            for step_index in range(self.target_steps):
                baseline_error = baseline_level.isel(time=step_index) - truth_level.isel(time=step_index)
                full_error = full_level.isel(time=step_index) - truth_level.isel(time=step_index)
                self.sum_sq_b_pl[variable][level][step_index] += self._weighted_mean(baseline_error**2)
                self.sum_sq_f_pl[variable][level][step_index] += self._weighted_mean(full_error**2)
                self.sum_abs_b_pl[variable][level][step_index] += self._weighted_mean(
                    np.abs(baseline_error)
                )
                self.sum_abs_f_pl[variable][level][step_index] += self._weighted_mean(
                    np.abs(full_error)
                )

            baseline_error_cells = self._time_lat_lon(baseline_level - truth_level)
            full_error_cells = self._time_lat_lon(full_level - truth_level)
            self.sum_err_cell_b_pl[variable].setdefault(level, np.zeros_like(baseline_error_cells))
            self.sum_err_cell_f_pl[variable].setdefault(level, np.zeros_like(full_error_cells))
            self.sum_err_cell_b_pl[variable][level] += baseline_error_cells
            self.sum_err_cell_f_pl[variable][level] += full_error_cells
            self.n_per_pl[variable][level] += 1

    def finalize(self) -> dict[str, dict[str, Any]]:
        if not self.n_per_var:
            raise ValueError("Cannot finalize v23_Ilya metrics without samples")
        output: dict[str, dict[str, Any]] = {
            "per_variable_per_step": {},
            "per_channel_per_step": {},
            "residual_diagnostics_per_variable": {},
        }

        for variable in sorted(self.n_per_var):
            count = self.n_per_var[variable]
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
            if variable in self.sum_err_cell_b:
                baseline_rmsb = self._lat_weighted_rms(self.sum_err_cell_b[variable] / count)
                full_rmsb = self._lat_weighted_rms(self.sum_err_cell_f[variable] / count)
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
                        self._pressure_level_entry(variable, level)
                    )
            else:
                output["per_channel_per_step"][variable] = entry

        return output

    def _pressure_level_entry(self, variable: str, level: int) -> dict[str, Any]:
        count = self.n_per_pl[variable][level]
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
        if level in self.sum_err_cell_b_pl[variable]:
            baseline_rmsb = self._lat_weighted_rms(
                self.sum_err_cell_b_pl[variable][level] / count
            )
            full_rmsb = self._lat_weighted_rms(self.sum_err_cell_f_pl[variable][level] / count)
            improvement_rmsb = -(
                (full_rmsb - baseline_rmsb) / np.maximum(baseline_rmsb, 1e-12) * 100
            )
            entry.update(
                rmsb_baseline=baseline_rmsb.tolist(),
                rmsb_full=full_rmsb.tolist(),
                improvement_pct_rmsb=improvement_rmsb.tolist(),
            )
        return entry

    def _residual_diagnostics(self, variable: str, count: int) -> dict[str, Any]:
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
    def _time_lat_lon(value: xr.DataArray) -> np.ndarray:
        array = value.transpose(..., "time", "lat", "lon").values.astype(np.float32)
        while array.ndim > 3:
            array = array.squeeze(0)
        return array

    def _lat_weighted_rms(self, bias_cells: np.ndarray) -> np.ndarray:
        weighted_mean = (bias_cells**2 * self._cos_lat[None, :, None]).mean(axis=(1, 2))
        return np.sqrt(weighted_mean)
