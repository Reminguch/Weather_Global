"""Read-only raw forecast spectra using the frozen optimal-filter transform.

Reuse its sufficient-statistics routine without fitting or applying any filter.
The forecast evaluator alone owns reset policy and autoregressive feedback.
"""
from __future__ import annotations

import time

import numpy as np

from .optimal_filter_observer import OptimalFilterObserver
from .smoothing_observer import atomic_npz


STATISTICS_SCHEMA = "v24_raw_spectral_statistics_v1"


class RawSpectralObserver(OptimalFilterObserver):
    def __init__(self, output_dir, diffs_stddev, identity, *,
                 branches=("GC", "R_matched"), target_steps=40, lmax=179,
                 chunk_channels=8, expected_channels=83):
        # The parent calibration routine collects unmodified raw moments; it
        # does not estimate gains. No parent filtering/evaluation path is used.
        super().__init__(output_dir, diffs_stddev, identity, mode="calibrate",
                         branches=branches, target_steps=target_steps, lmax=lmax,
                         chunk_channels=chunk_channels, expected_channels=expected_channels)
        self.mode = "evaluate"

    def update_step(self, step_index, truth, baseline, full):
        if self.sample is None or step_index != self.steps or step_index >= self.target_steps:
            raise ValueError("Missing or out-of-order raw spectral step")
        started = time.monotonic()
        self._setup(truth, baseline, full)
        target = self.layout.pack(truth)
        predictions = (self.layout.pack(baseline), self.layout.pack(full))
        self._calibrate(predictions, target, step_index)
        self.steps += 1
        elapsed = time.monotonic() - started
        self.step_seconds.append(elapsed)
        if step_index in (0, 3, 19, 39):
            print(f"[raw-spectra] {self.sample['initialization_time']} "
                  f"lead={(step_index+1)*6}h observer_seconds={elapsed:.3f}", flush=True)

    def finish_sample(self):
        if self.sample is None or self.steps != self.target_steps:
            raise ValueError("Incomplete raw spectral initialization")
        arrays = {key: self.accumulated[key].swapaxes(0, 1)
                  for key in ("prediction_power", "cross_power", "error_power")}
        arrays["truth_power"] = self.accumulated["truth_power"]
        arrays["normalized_grid_mse"] = self.accumulated["e0"].swapaxes(0, 1)
        arrays["exact_gc_loss"] = np.sum(
            arrays["normalized_grid_mse"] * self.layout.weights[None, None], axis=-1)
        arrays["spectral_weighted_gc_loss"] = np.sum(
            arrays["error_power"].sum(axis=-1) *
            (self.layout.weights / self.layout.scales**2)[None, None], axis=-1)
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("Nonfinite raw spectral statistics")
        metadata = {**self.sample, "schema": STATISTICS_SCHEMA, "status": "complete",
                    "mode": "evaluate", "layout": self.layout.metadata(),
                    "branches": list(self.branches), "variants": ["raw"], "lmax": self.lmax,
                    "axes": {"prediction_power": "lead,branch,channel,degree",
                             "cross_power": "lead,branch,channel,degree",
                             "error_power": "lead,branch,channel,degree",
                             "truth_power": "lead,channel,degree",
                             "normalized_grid_mse": "lead,branch,channel",
                             "exact_gc_loss": "lead,branch",
                             "spectral_weighted_gc_loss": "lead,branch"},
                    "operator": "raw forecast; same normalized T179 analysis as optimal-filter comparison",
                    "spectral_metric": "physical continuous coefficient-domain moments, no climatology subtraction",
                    "exact_gc_loss_definition": "FP64 native-grid normalized squared errors, exact pole-cell areas, original variable/pressure weights",
                    "feedback": "unaltered evaluator fields; no filtering or observer feedback",
                    "filter_fitting": False, "step_seconds": self.step_seconds[-self.target_steps:]}
        atomic_npz(self.path, arrays, metadata)
        self.sample = self.accumulated = None
        self.records = []
