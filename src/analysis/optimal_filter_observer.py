"""Streaming per-lead statistics for GraphCast-style optimal spectral filtering.

The rollout evaluator owns all feedback. This observer only reads its fields.
The spectral reference uses the represented coefficients directly; native-grid
scores separately evaluate raw, projected, and optimally filtered forecasts.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .optimal_spherical_filter import OptimalSphericalFilter, spectral_moments
from .smoothing_observer import atomic_npz
from .spherical_smoothing import ChannelLayout, json_digest


VARIANTS = ("raw", "projected", "optimal")
STATISTICS_SCHEMA = "v24_optimal_filter_statistics_v1"


class OptimalFilterObserver:
    """Collect independent channel/lead fits and score frozen spectral gains.

    Layout group indices are retained only for compatibility with established
    channel provenance. They never tie winds or pool leads in this observer.
    """

    def __init__(self, output_dir, diffs_stddev, identity, *, mode,
                 filters=None, branches=("GC", "M1"), target_steps=40,
                 lmax=179, chunk_channels=8, expected_channels=83):
        if mode not in ("calibrate", "evaluate"):
            raise ValueError("Unknown optimal-filter observer mode")
        if mode == "evaluate" and filters is None:
            raise ValueError("Evaluation requires frozen development filters")
        if len(branches) != 2 or len(set(branches)) != 2:
            raise ValueError("Expected distinct baseline and full-model branches")
        if target_steps < 1 or chunk_channels < 1:
            raise ValueError("Positive lead and channel-chunk counts are required")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_stddev, self.identity = diffs_stddev, dict(identity)
        self.mode, self.filters, self.branches = mode, filters, tuple(branches)
        self.target_steps, self.lmax = target_steps, lmax
        self.chunk_channels, self.expected_channels = chunk_channels, expected_channels
        self.layout = self.operator = None
        self.sample = None
        self.step_seconds = []

    def begin_sample(self, metadata):
        if self.sample is not None:
            raise ValueError("Previous optimal-filter initialization is incomplete")
        if metadata["lead_hours"] != list(range(6, 6 * self.target_steps + 1, 6)):
            raise ValueError("Unexpected optimal-filter forecast lead schedule")
        self.path = self.output_dir / f"{metadata['initialization_id']}.npz"
        if self.path.exists():
            raise FileExistsError(self.path)
        self.sample = {**metadata, **self.identity}
        self.steps = 0
        self.accumulated = None
        self.records = []

    def _setup(self, truth, baseline, full):
        latitude, longitude = np.asarray(truth.lat), np.asarray(truth.lon)
        for dataset in (baseline, full):
            if not np.array_equal(dataset.lat, latitude) or not np.array_equal(dataset.lon, longitude):
                raise ValueError("Forecast and truth grids differ")
        if self.layout is None:
            self.layout = ChannelLayout.from_dataset(
                truth, self.diffs_stddev, expected_channels=self.expected_channels)
            self.operator = OptimalSphericalFilter(latitude, longitude, lmax=self.lmax)
            self.positive_indices = [i for i, (name, _) in enumerate(self.layout.channels)
                                     if name in ("specific_humidity", "total_precipitation_6hr")]
            if self.filters is not None:
                if self.filters["layout"] != self.layout.metadata():
                    raise ValueError("Filter layout/normalization differs from evaluation")
                if set(self.filters["branches"]) != set(self.branches):
                    raise ValueError("Filter predictor set differs")
                if self.filters.get("lmax", self.lmax) != self.lmax:
                    raise ValueError("Filter spherical truncation differs")
                if self.filters.get("lead_hours", list(range(6, 6*self.target_steps+1, 6))) != list(range(6, 6*self.target_steps+1, 6)):
                    raise ValueError("Filter lead schedule differs")
                expected = (self.target_steps, len(self.layout.channels), self.lmax + 1)
                self.gains = {}
                for branch in self.branches:
                    gains = np.asarray(self.filters["branches"][branch]["gains"], dtype=np.float64)
                    if gains.shape != expected or not np.all(np.isfinite(gains)):
                        raise ValueError("Invalid per-lead/channel/degree filter gains")
                    self.gains[branch] = gains
        elif not np.array_equal(self.operator.latitude, latitude) or not np.array_equal(self.operator.longitude, longitude):
            raise ValueError("Optimal-filter grid changed")

    def update_step(self, step_index, truth, baseline, full):
        if self.sample is None or step_index != self.steps or step_index >= self.target_steps:
            raise ValueError("Missing or out-of-order optimal-filter forecast step")
        started = time.monotonic()
        self._setup(truth, baseline, full)
        target = self.layout.pack(truth)
        predictions = (self.layout.pack(baseline), self.layout.pack(full))
        if self.mode == "calibrate":
            self._calibrate(predictions, target, step_index)
        else:
            self._evaluate(predictions, target, step_index)
        self.steps += 1
        elapsed = time.monotonic() - started
        self.step_seconds.append(elapsed)
        if step_index in (0, 3, 19, 39):
            print(f"[optimal-filter] {self.mode} {self.sample['initialization_time']} "
                  f"lead={(step_index+1)*6}h observer_seconds={elapsed:.3f}", flush=True)

    def _calibrate(self, predictions, target, lead):
        channels = len(self.layout.channels)
        if self.accumulated is None:
            prefix = (2, self.target_steps, channels)
            self.accumulated = {key: np.zeros(prefix + (self.lmax + 1,))
                                for key in ("prediction_power", "cross_power", "error_power")}
            self.accumulated["truth_power"] = np.zeros((self.target_steps, channels, self.lmax + 1))
            self.accumulated["e0"] = np.zeros(prefix)
        for start in range(0, channels, self.chunk_channels):
            selection = slice(start, min(start + self.chunk_channels, channels))
            target_coefficients = self.operator.analyze(target[selection])
            for branch, prediction in enumerate(predictions):
                coefficients = self.operator.analyze(prediction[selection])
                statistics = spectral_moments(coefficients, target_coefficients)
                for key in ("prediction_power", "cross_power", "error_power"):
                    self.accumulated[key][branch, lead, selection] = statistics[key]
                if branch == 0:
                    self.accumulated["truth_power"][lead, selection] = statistics["truth_power"]
                error = (prediction[selection] - target[selection]) / self.layout.scales[selection, None, None]
                self.accumulated["e0"][branch, lead, selection] = self.operator.mean(error**2)
        if not all(np.all(np.isfinite(values)) for values in self.accumulated.values()):
            raise ValueError("Nonfinite optimal-filter calibration statistics")

    def _evaluate(self, predictions, target, lead):
        channels = len(self.layout.channels)
        record = {key: np.empty((2, len(VARIANTS), channels, self.lmax + 1))
                  for key in ("prediction_power", "cross_power", "error_power")}
        record.update({key: np.empty((2, len(VARIANTS), channels))
                       for key in ("grid_mse", "prediction_mean", "mean_drift")})
        record["truth_power"] = np.empty((channels, self.lmax + 1))
        record["truth_mean"] = self.operator.mean(target)
        record["negative_fraction"] = np.zeros((2, len(VARIANTS), len(self.positive_indices)))
        record["positive_field_minimum"] = np.zeros((2, len(VARIANTS), len(self.positive_indices)))
        record["forecast_projection_residual_mse"] = np.empty((2, channels))
        record["truth_projection_residual_mse"] = np.empty(channels)
        for start in range(0, channels, self.chunk_channels):
            stop = min(start + self.chunk_channels, channels)
            selection = slice(start, stop)
            target_coefficients = self.operator.analyze(target[selection])
            target_projection = self.operator.synthesize(target_coefficients)
            record["truth_projection_residual_mse"][selection] = self.operator.mean(
                (target[selection] - target_projection)**2)
            for bi, (branch, prediction) in enumerate(zip(self.branches, predictions)):
                raw = prediction[selection]
                coefficients = self.operator.analyze(raw)
                filtered_coefficients = coefficients * self.gains[branch][lead, selection, :, None]
                projected = self.operator.synthesize(coefficients)
                filtered = self.operator.synthesize(filtered_coefficients)
                record["forecast_projection_residual_mse"][bi, selection] = self.operator.mean((raw - projected)**2)
                raw_mean = self.operator.mean(raw)
                for vi, (field, represented) in enumerate(zip(
                        (raw, projected, filtered), (coefficients, coefficients, filtered_coefficients))):
                    statistics = spectral_moments(represented, target_coefficients)
                    for key in ("prediction_power", "cross_power", "error_power"):
                        record[key][bi, vi, selection] = statistics[key]
                    if bi == 0 and vi == 0:
                        record["truth_power"][selection] = statistics["truth_power"]
                    record["grid_mse"][bi, vi, selection] = self.operator.mean((field - target[selection])**2)
                    record["prediction_mean"][bi, vi, selection] = self.operator.mean(field)
                    record["mean_drift"][bi, vi, selection] = record["prediction_mean"][bi, vi, selection] - raw_mean
                    for pi, ci in enumerate(self.positive_indices):
                        if start <= ci < stop:
                            values = field[ci - start]
                            record["negative_fraction"][bi, vi, pi] = np.mean(values < 0)
                            record["positive_field_minimum"][bi, vi, pi] = values.min()
        weights = self.layout.weights / self.layout.scales**2
        record["exact_gc_loss"] = np.sum(record["grid_mse"] * weights[None, None], axis=-1)
        record["spectral_weighted_gc_loss"] = np.sum(record["error_power"].sum(axis=-1) * weights[None, None], axis=-1)
        if not all(np.all(np.isfinite(values)) for values in record.values()):
            raise ValueError("Nonfinite optimal-filter evaluation statistic")
        self.records.append(record)

    def finish_sample(self):
        if self.sample is None or self.steps != self.target_steps:
            raise ValueError("Incomplete optimal-filter initialization")
        if self.mode == "calibrate":
            arrays = self.accumulated
            axes = {"e0": "branch,lead,channel", "prediction_power": "branch,lead,channel,degree",
                    "cross_power": "branch,lead,channel,degree", "error_power": "branch,lead,channel,degree",
                    "truth_power": "lead,channel,degree"}
        else:
            arrays = {key: np.stack([r[key] for r in self.records]) for key in self.records[0]}
            axes = {"exact_gc_loss": "lead,branch,variant", "grid_mse": "lead,branch,variant,channel",
                    "prediction_power": "lead,branch,variant,channel,degree",
                    "cross_power": "lead,branch,variant,channel,degree",
                    "error_power": "lead,branch,variant,channel,degree", "truth_power": "lead,channel,degree"}
        metadata = {**self.sample, "schema": STATISTICS_SCHEMA, "status": "complete", "mode": self.mode,
                    "layout": self.layout.metadata(), "branches": list(self.branches), "variants": list(VARIANTS),
                    "lmax": self.lmax, "filter_digest": json_digest(self.filters) if self.filters else None,
                    "axes": axes, "fit_grouping": "independent physical channel and lead; no wind tying or lead pooling",
                    "operator": "raw=f; projected=S[A f]; optimal=S[b A f]; no residual or mean correction",
                    "spectral_metric": "continuous coefficient-domain MSE; raw/projected use A f, optimal uses b A f",
                    "grid_mse": "FP64 physical error versus native truth, exact pole-cell areas; GC weights and difference scales",
                    "feedback": "unaltered evaluator fields; filtered outputs never feed subsequent steps",
                    "positive_field_channels": [list(self.layout.channels[i]) for i in self.positive_indices],
                    "negative_fraction_definition": "Unweighted native grid-cell fraction; no clipping",
                    "step_seconds": self.step_seconds[-self.target_steps:]}
        atomic_npz(self.path, arrays, metadata)
        self.sample = self.accumulated = None
        self.records = []
