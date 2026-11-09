"""Streaming calibration/evaluation for frozen spherical output filters."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from .spherical_smoothing import ChannelLayout, SphericalAttenuator, json_digest, lead_group
from src.models.mamba.v24_Ilya.spectral_diagnostics import zonal_spectral_statistics
from src.models.mamba.v24_Ilya.spectral_observer import FIELD_SPECS


def atomic_npz(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".partial.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class SmoothingObserver:
    """The evaluator owns rollout; this observer never mutates its datasets.

    ``identity`` records checkpoint/protocol identities. ``branches`` names
    the evaluator's baseline/full outputs, defaulting to public GC and M1.
    Fitting artifacts are independent of evaluation data and immutable.
    """
    def __init__(self, output_dir, diffs_stddev, identity, *, mode,
                 filters=None, branches=("GC", "M1"), target_steps=40,
                 lmax=179, knots=(1, 10, 40, 179), chunk_channels=8,
                 expected_channels=83, compute_zonal=True):
        if mode not in ("calibrate", "evaluate"):
            raise ValueError("Unknown smoothing observer mode")
        if mode == "evaluate" and filters is None:
            raise ValueError("Evaluation requires frozen development filters")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_stddev, self.identity = diffs_stddev, dict(identity)
        self.mode, self.filters, self.branches = mode, filters, tuple(branches)
        self.target_steps, self.lmax, self.knots = target_steps, lmax, knots
        self.chunk_channels, self.expected_channels = chunk_channels, expected_channels
        self.compute_zonal = compute_zonal
        self.layout = self.operator = None
        self.sample = None
        self.step_seconds = []

    def begin_sample(self, metadata):
        if self.sample is not None:
            raise ValueError("Previous smoothing initialization is incomplete")
        if metadata["lead_hours"] != list(range(6, 6 * self.target_steps + 1, 6)):
            raise ValueError("Unexpected smoothing forecast lead schedule")
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
            self.layout = ChannelLayout.from_dataset(truth, self.diffs_stddev,
                                                    expected_channels=self.expected_channels)
            self.operator = SphericalAttenuator(latitude, longitude, lmax=self.lmax,
                                               knots=self.knots, chunk_channels=self.chunk_channels)
            if self.filters is not None:
                if self.filters["layout"] != self.layout.metadata():
                    raise ValueError("Filter layout/normalization differs from evaluation")
                if tuple(self.filters["branches"]) != self.branches:
                    # JSON dictionary ordering is not an identity condition.
                    if set(self.filters["branches"]) != set(self.branches):
                        raise ValueError("Filter predictor set differs")
            self.positive_indices = [i for i, (name, _) in enumerate(self.layout.channels)
                                     if name in ("specific_humidity", "total_precipitation_6hr")]
            self.zonal_indices = []
            if self.compute_zonal:
                self.zonal_indices = [self.layout.channels.index((name, level)) for _, name, level, _ in FIELD_SPECS]
        elif not np.array_equal(self.operator.latitude, latitude) or not np.array_equal(self.operator.longitude, longitude):
            raise ValueError("Smoothing grid changed")

    def update_step(self, step_index, truth, baseline, full):
        if self.sample is None or step_index != self.steps or step_index >= self.target_steps:
            raise ValueError("Missing or out-of-order smoothing forecast step")
        started = time.monotonic()
        self._setup(truth, baseline, full)
        target = self.layout.pack(truth)
        predictions = (self.layout.pack(baseline), self.layout.pack(full))
        group = lead_group((step_index + 1) * 6)
        if self.mode == "calibrate":
            self._calibrate(predictions, target, group)
        else:
            self._evaluate(predictions, target, group)
        self.steps += 1
        elapsed = time.monotonic() - started
        self.step_seconds.append(elapsed)
        if step_index in (0, 3, 19, 39):
            print(f"[smoothing] {self.mode} {self.sample['initialization_time']} "
                  f"lead={(step_index+1)*6}h observer_seconds={elapsed:.3f}", flush=True)

    def _calibrate(self, predictions, target, group):
        channels = len(self.layout.channels)
        if self.accumulated is None:
            prefix = (2, 4, channels)
            self.accumulated = {"e0": np.zeros(prefix), "cross": np.zeros(prefix + (4,)),
                                "gram": np.zeros(prefix + (4, 4)), "gaussian": np.zeros(prefix + (7,))}
        for branch, prediction in enumerate(predictions):
            for start in range(0, channels, self.chunk_channels):
                selection = slice(start, min(start + self.chunk_channels, channels))
                stats = self.operator.calibration_chunk(prediction[selection], target[selection],
                                                        self.layout.scales[selection])
                for key, values in stats.items():
                    if not np.all(np.isfinite(values)):
                        raise ValueError("Nonfinite calibration statistics")
                    self.accumulated[key][branch, group, selection] += values

    def _evaluate(self, predictions, target, group):
        channels = len(self.layout.channels)
        scalar_keys = ("grid_mse", "projected_grid_mse", "unresolved_grid_mse")
        power_keys = ("prediction_power", "cross_power", "error_power")
        record = {key: np.empty((2, 3, channels, self.lmax + 1)) for key in power_keys}
        record.update({key: np.empty((2, 3, channels)) for key in scalar_keys})
        record["truth_power"] = np.empty((channels, self.lmax + 1))
        record["negative_fraction"] = np.zeros((2, 3, len(self.positive_indices)))
        record["positive_field_minimum"] = np.zeros((2, 3, len(self.positive_indices)))
        record["mean_drift"] = np.zeros((2, 3, channels))
        zonal_fields = np.empty((2, 3, len(self.zonal_indices), *target.shape[-2:]))
        for branch_index, (branch, prediction) in enumerate(zip(self.branches, predictions)):
            recipe = self.filters["branches"][branch]
            gaussian = np.asarray(recipe["gaussian_indices"], dtype=int)[group]
            bands = np.asarray(recipe["band_gains"], dtype=np.float64)[group]
            for start in range(0, channels, self.chunk_channels):
                stop = min(start + self.chunk_channels, channels)
                selection = slice(start, stop)
                pair = self.operator.filtered_pair(prediction[selection], self.layout.scales[selection],
                                                   gaussian[selection], bands[selection])
                fields = (prediction[selection], pair[0], pair[1])
                for variant, field in enumerate(fields):
                    stats = self.operator.spectral_chunk(field, target[selection])
                    for key in power_keys + scalar_keys:
                        record[key][branch_index, variant, selection] = stats[key]
                    if branch_index == 0 and variant == 0:
                        record["truth_power"][selection] = stats["truth_power"]
                    record["mean_drift"][branch_index, variant, selection] = (
                        self.operator.mean(field) - self.operator.mean(prediction[selection]))
                    for pi, ci in enumerate(self.positive_indices):
                        if start <= ci < stop:
                            values = field[ci-start]
                            record["negative_fraction"][branch_index, variant, pi] = np.mean(values < 0)
                            record["positive_field_minimum"][branch_index, variant, pi] = values.min()
                    for zi, ci in enumerate(self.zonal_indices):
                        if start <= ci < stop:
                            zonal_fields[branch_index, variant, zi] = field[ci-start]
        # Recompute physical error in FP64, including every native-grid channel.
        record["exact_gc_loss"] = np.sum(record["grid_mse"] *
            (self.layout.weights / self.layout.scales**2)[None, None], axis=-1)
        for key, values in record.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite smoothing evaluation statistic: {key}")
        if self.compute_zonal:
            for bi in range(2):
                for variant in range(3):
                    values = zonal_spectral_statistics(target[self.zonal_indices], zonal_fields[bi, variant],
                                                       self.operator.latitude)
                    for key in ("power_forecast", "cross_power", "error_power"):
                        output_key = "zonal_" + key
                        if output_key not in record:
                            record[output_key] = np.empty((2, 3, len(self.zonal_indices), values[key].shape[-1]))
                        record[output_key][bi, variant] = values[key]
                    record["zonal_truth_power"] = values["power_truth"]
        self.records.append(record)

    def finish_sample(self):
        if self.sample is None or self.steps != self.target_steps:
            raise ValueError("Incomplete smoothing initialization")
        if self.mode == "calibrate":
            arrays = self.accumulated
        else:
            arrays = {key: np.stack([r[key] for r in self.records]) for key in self.records[0]}
        metadata = {**self.sample, "status": "complete", "mode": self.mode,
                    "layout": self.layout.metadata(), "branches": list(self.branches),
                    "variants": ["raw", "gaussian", "band"], "lmax": self.lmax,
                    "knots": list(self.knots), "filter_digest": json_digest(self.filters) if self.filters else None,
                    "operator": "f + S[(g-1)Af] - mean_GC(S[(g-1)Af]); unrepresented residual retained",
                    "grid_mse": "FP64 physical error, exact pole-cell areas; GC weights and difference scales",
                    "spherical": "Physical continuous fitted norm; distinct from exact discrete grid MSE",
                    "positive_field_channels": [list(self.layout.channels[i]) for i in self.positive_indices],
                    "negative_fraction_definition": "Unweighted native grid-cell fraction; no clipping",
                    "zonal_variables": [row[0] for row in FIELD_SPECS] if self.compute_zonal else [],
                    "step_seconds": self.step_seconds[-self.target_steps:]}
        atomic_npz(self.path, arrays, metadata)
        self.sample = self.accumulated = None
        self.records = []

