"""Separate finite numerical diagnostics from experiment integrity checks."""
from __future__ import annotations

import numpy as np


def is_advisory(protocol):
    policy = protocol.get("numerical_policy", "strict")
    if policy not in ("strict", "advisory"):
        raise ValueError(f"Unknown numerical_policy: {policy!r}")
    return policy == "advisory"


def numerical_comparison(candidate, reference, *, rtol, atol, label, advisory=False):
    """Always reject invalid arrays; optionally warn about finite repeat drift."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (candidate.shape != reference.shape or not candidate.size
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError(f"{label}: invalid shape or nonfinite numerical comparison")
    difference = np.abs(candidate - reference)
    tolerance = atol + rtol * np.abs(reference)
    nonzero = np.abs(reference) > 0
    passed = bool(np.all(difference <= tolerance))
    scale = float(np.abs(reference).mean())
    record = {
        "label": label, "status": "passed" if passed else "warning", "passed": passed,
        "rtol": float(rtol), "atol": float(atol),
        "outside_tolerance_count": int(np.count_nonzero(difference > tolerance)),
        "value_count": int(reference.size),
        "max_absolute_difference": float(difference.max()),
        "max_relative_difference": float((difference[nonzero] / np.abs(reference[nonzero])).max()) if nonzero.any() else 0.,
        "pooled_absolute_difference": float(abs(candidate.mean() - reference.mean())),
        "pooled_relative_difference": float(abs(candidate.mean() - reference.mean()) / scale) if scale else None,
    }
    if not passed and not advisory:
        raise ValueError(f"{label}: numerical repeatability failed: {record}")
    return record
