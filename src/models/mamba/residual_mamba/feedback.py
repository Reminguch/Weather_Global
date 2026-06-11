from __future__ import annotations

import xarray as xr


RESIDUAL_AR_FEEDBACK_BASELINE = "baseline"
RESIDUAL_AR_FEEDBACK_BASELINE_PLUS_RESIDUAL = "baseline_plus_residual"
RESIDUAL_AR_FEEDBACK = RESIDUAL_AR_FEEDBACK_BASELINE
RESIDUAL_AR_FEEDBACK_CHOICES = (
    RESIDUAL_AR_FEEDBACK_BASELINE,
    RESIDUAL_AR_FEEDBACK_BASELINE_PLUS_RESIDUAL,
)


def validate_residual_ar_feedback(mode: str) -> str:
    if mode not in RESIDUAL_AR_FEEDBACK_CHOICES:
        raise ValueError(
            f"Unknown residual AR feedback mode {mode!r}; "
            f"expected one of {', '.join(RESIDUAL_AR_FEEDBACK_CHOICES)}."
        )
    return mode


def residual_physical_feedback(
    *,
    baseline_pred: xr.Dataset,
    full_pred: xr.Dataset,
    mode: str,
) -> xr.Dataset:
    validate_residual_ar_feedback(mode)
    if mode == RESIDUAL_AR_FEEDBACK_BASELINE:
        return baseline_pred
    return full_pred
