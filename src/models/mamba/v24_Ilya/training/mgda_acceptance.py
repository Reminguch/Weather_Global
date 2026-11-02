"""Host-side measured-loss acceptance for a fixed candidate RMS direction.

The optimizer direction and its tentative next RMS state must be computed once
before calling :func:`backtrack_losses`. ``evaluate_candidate`` must replay a
complete closed-loop batch from the same initial parameters, memory, inputs and
random keys for every candidate learning rate. The callback must not commit
parameters, recurrent memory, RNG or optimizer state. This module deliberately
owns none of those objects; the caller commits them only after acceptance.

Directional derivatives are useful diagnostics, but do not establish the
finite-step acceptance condition checked here. Finite repeatability noise is
advisory by default; it must not prevent an otherwise accepted measured decrease.
Such a decrease is not evidence of improvement above the measured noise floor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Callable, Literal, Sequence

import numpy as np


LossPair = tuple[float, float]
AcceptanceMode = Literal["mgda", "mse_control", "fixed_sum"]


def _pair(value, *, name: str, finite: bool = True) -> LossPair:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly two scalar losses")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return float(array[0]), float(array[1])


def _nonnegative_pair(value, *, name: str) -> LossPair:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        array = np.repeat(array, 2)
    result = _pair(array, name=name)
    if min(result) < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class LossTolerances:
    """Measured repeatability and minimum meaningful decreases, in raw units."""

    absolute_floor: LossPair
    relative_floor: LossPair
    repeatability_span: LossPair
    reference_losses: LossPair
    repeats: int
    noise_multiplier: float = 4.0
    max_relative_noise: float = 1e-3
    repeatability_policy: Literal["warn", "error"] = "warn"

    def __post_init__(self):
        for name in ("absolute_floor", "relative_floor", "repeatability_span"):
            object.__setattr__(self, name, _nonnegative_pair(getattr(self, name), name=name))
        object.__setattr__(self, "reference_losses", _pair(self.reference_losses, name="reference_losses"))
        if not isinstance(self.repeats, (int, np.integer)) or self.repeats < 3:
            raise ValueError("Repeatability qualification requires at least three identical repeats")
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier < 1:
            raise ValueError("noise_multiplier must be finite and at least one")
        if not math.isfinite(self.max_relative_noise) or not 0 <= self.max_relative_noise < 1:
            raise ValueError("max_relative_noise must be finite and in [0, 1)")
        if self.repeatability_policy not in ("warn", "error"):
            raise ValueError("repeatability_policy must be 'warn' or 'error'")
        if any(self.noise_exceeded):
            message = ("Identical-forward repeatability noise exceeds its configured bound: "
                       f"span={self.repeatability_span}, bound={self.noise_bounds}")
            if self.repeatability_policy == "error":
                raise ValueError(message)
            warnings.warn(message + "; advisory only, continuing", RuntimeWarning, stacklevel=2)

    @property
    def noise_bounds(self) -> LossPair:
        return tuple(float(x) for x in np.maximum(
            self.absolute_floor, np.abs(self.reference_losses) * self.max_relative_noise))

    @property
    def noise_exceeded(self) -> tuple[bool, bool]:
        return tuple(span > bound for span, bound in zip(
            self.repeatability_span, self.noise_bounds, strict=True))

    def thresholds(self, before_losses: Sequence[float]) -> LossPair:
        before = np.asarray(_pair(before_losses, name="before_losses"))
        values = np.maximum(self.absolute_floor, np.asarray(self.relative_floor) * np.abs(before))
        if self.repeatability_policy == "error":
            values = np.maximum(values, self.noise_multiplier * np.asarray(self.repeatability_span))
        return float(values[0]), float(values[1])

    def noise_thresholds(self, before_losses: Sequence[float]) -> LossPair:
        """Retain the original noise-aware margin as an interpretation diagnostic."""
        return tuple(float(x) for x in np.maximum(
            self.thresholds(before_losses), self.noise_multiplier * np.asarray(self.repeatability_span)))

    def to_dict(self) -> dict:
        return {"absolute_floor": list(self.absolute_floor),
                "relative_floor": list(self.relative_floor),
                "repeatability_span": list(self.repeatability_span),
                "reference_losses": list(self.reference_losses),
                "repeats": int(self.repeats),
                "noise_multiplier": float(self.noise_multiplier),
                "max_relative_noise": float(self.max_relative_noise),
                "repeatability_policy": self.repeatability_policy}


def calibrate_loss_tolerances(
    repeated_losses: Sequence[Sequence[float]], *, absolute_floor=1e-8,
    relative_floor=1e-7, noise_multiplier: float = 4.0,
    max_relative_noise: float = 1e-3,
    repeatability_policy: Literal["warn", "error"] = "warn",
) -> LossTolerances:
    """Qualify >=3 forward evaluations of one unchanged batch/state/RNG.

    A caller qualifying several batches should retain their individual records;
    rows here must be repeats of the same batch, not different training samples.
    """
    values = np.asarray(repeated_losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 3:
        raise ValueError("Repeatability qualification requires shape (at least 3, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Repeatability losses must be finite")
    return LossTolerances(
        absolute_floor=_nonnegative_pair(absolute_floor, name="absolute_floor"),
        relative_floor=_nonnegative_pair(relative_floor, name="relative_floor"),
        repeatability_span=_pair(np.ptp(values, axis=0), name="repeatability_span"),
        reference_losses=_pair(np.mean(values, axis=0), name="reference_losses"),
        repeats=len(values), noise_multiplier=noise_multiplier,
        max_relative_noise=max_relative_noise,
        repeatability_policy=repeatability_policy,
    )


@dataclass(frozen=True)
class BacktrackingTrial:
    index: int
    learning_rate: float
    losses: LossPair
    decreases: LossPair
    accepted: bool
    reason: str
    measured_decrease: bool = False

    def to_dict(self) -> dict:
        # Failed nonfinite forwards are recorded as JSON null, never NaN JSON.
        finite_list = lambda values: [float(x) if math.isfinite(x) else None for x in values]
        return {"index": self.index, "learning_rate": self.learning_rate,
                "losses": finite_list(self.losses), "decreases": finite_list(self.decreases),
                "accepted": self.accepted, "reason": self.reason,
                "measured_decrease": self.measured_decrease}


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    learning_rate: float | None
    losses: LossPair | None
    before_losses: LossPair
    thresholds: LossPair
    trials: tuple[BacktrackingTrial, ...]
    reason: str
    mode: AcceptanceMode
    measured_decrease: bool = False
    decrease_policy: str = "error"

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "learning_rate": self.learning_rate,
                "losses": list(self.losses) if self.losses is not None else None,
                "before_losses": list(self.before_losses), "thresholds": list(self.thresholds),
                "trials": [trial.to_dict() for trial in self.trials],
                "reason": self.reason, "mode": self.mode,
                "measured_decrease": self.measured_decrease,
                "decrease_policy": self.decrease_policy}


def backtrack_losses(
    before_losses: Sequence[float],
    evaluate_candidate: Callable[[float], Sequence[float]], *,
    mode: AcceptanceMode, tolerances: LossTolerances,
    initial_learning_rate: float = 1e-6, max_halvings: int = 8,
    shrink: float = 0.5,
    decrease_policy: Literal["warn", "error"] = "warn",
) -> AcceptanceResult:
    """Commit a finite scheduled step, recording loss changes as diagnostics.

    Under the default warning policy, evaluate the scheduled step once and
    retain it even when measured losses increase or tie. Do not select among
    repeated noisy forecasts. Explicit ``error`` policy retains the historical
    loss-decrease search with at most ``max_halvings + 1`` candidates.

    All candidates are displacements from the same original parameters along a
    single fixed optimizer direction. Rejected candidates must not become the
    starting point for subsequent trials. The caller supplies that pure replay
    through ``evaluate_candidate(learning_rate)`` and caches any accepted state.

    The measured-decrease diagnostic requires both losses for MGDA, the first
    for MSE control, and their sum for fixed-sum. Only explicit error policy
    makes that diagnostic blocking. Nonfinite losses never commit.
    Callback exceptions propagate so that model/IO failures cannot masquerade
    as ordinary optimizer stalls.
    """
    before = _pair(before_losses, name="before_losses")
    if mode not in ("mgda", "mse_control", "fixed_sum"):
        raise ValueError(f"Unsupported acceptance mode: {mode}")
    if decrease_policy not in ("warn", "error"):
        raise ValueError("decrease_policy must be 'warn' or 'error'")
    if not math.isfinite(initial_learning_rate) or initial_learning_rate <= 0:
        raise ValueError("initial_learning_rate must be finite and positive")
    if not isinstance(max_halvings, (int, np.integer)) or not 0 <= max_halvings <= 8:
        raise ValueError("max_halvings must be an integer in [0, 8]")
    if shrink != 0.5:
        raise ValueError("The qualified backtracking policy halves each candidate step")
    thresholds = tolerances.thresholds(before)
    trials = []
    for index in range(1 if decrease_policy == "warn" else max_halvings + 1):
        learning_rate = float(initial_learning_rate * shrink ** index)
        candidate = _pair(evaluate_candidate(learning_rate), name="candidate_losses", finite=False)
        decreases = before[0] - candidate[0], before[1] - candidate[1]
        finite = all(math.isfinite(value) for value in (*candidate, *decreases))
        if not finite:
            measured, accepted, reason = False, False, "nonfinite_loss"
        else:
            if mode == "mgda":
                measured = all(delta > floor for delta, floor in zip(decreases, thresholds, strict=True))
            elif mode == "mse_control":
                measured = decreases[0] > thresholds[0]
            else:
                measured = math.fsum(decreases) > math.fsum(thresholds)
            accepted = measured or decrease_policy == "warn"
            reason = ("accepted" if measured else "advisory_no_decrease" if accepted
                      else "insufficient_measured_decrease")
            if accepted and not measured:
                warnings.warn(f"{mode}: no measured decrease; committing finite scheduled update "
                              f"with losses {before} -> {candidate}", RuntimeWarning, stacklevel=2)
        trials.append(BacktrackingTrial(index, learning_rate, candidate, decreases, accepted, reason, measured))
        if accepted:
            return AcceptanceResult(True, learning_rate, candidate, before, thresholds,
                                    tuple(trials), reason, mode, measured, decrease_policy)
    return AcceptanceResult(False, None, None, before, thresholds, tuple(trials),
                            "nonfinite_loss" if decrease_policy == "warn" else "backtracking_exhausted",
                            mode, False, decrease_policy)
