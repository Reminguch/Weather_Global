#!/usr/bin/env python3
"""Verify the matched v24 zero-residual validation reproduces the FP32 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_LOSS = 5.307552242279053
RAW_BF16_LOSS = 8.113585
EXPECTED_BY_HORIZON = {
    "1": 0.4421875,
    "4": 1.49375,
    "8": 2.478125,
    "12": 3.609375,
    "16": 5.35625,
    "20": 7.5,
}


def verify_fp32_validation(path: Path, *, tolerance: float = 1e-3) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline = payload.get("baseline", payload)
    if baseline.get("role") != "zero_residual_baseline":
        raise ValueError(f"{path} does not contain a zero-residual baseline result")

    loss = float(baseline["loss"])
    if abs(loss - RAW_BF16_LOSS) <= tolerance:
        raise ValueError(
            f"Validation reproduced the raw-BF16 baseline {RAW_BF16_LOSS:.6f}"
        )
    if abs(loss - EXPECTED_LOSS) > tolerance:
        raise ValueError(
            f"FP32 baseline loss mismatch: observed={loss:.10f}, "
            f"expected={EXPECTED_LOSS:.10f}, tolerance={tolerance}"
        )
    observed_by_horizon = baseline.get("loss_by_horizon")
    if not isinstance(observed_by_horizon, dict):
        raise ValueError("Validation result has no loss_by_horizon mapping")
    for horizon, expected in EXPECTED_BY_HORIZON.items():
        observed = float(observed_by_horizon[horizon])
        if abs(observed - expected) > tolerance:
            raise ValueError(
                f"FP32 horizon {horizon} loss mismatch: observed={observed:.10f}, "
                f"expected={expected:.10f}, tolerance={tolerance}"
            )
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    baseline = verify_fp32_validation(args.result, tolerance=args.tolerance)
    print(
        "v24_fp32_validation_passed "
        f"loss={float(baseline['loss']):.10f} tolerance={args.tolerance}"
    )


if __name__ == "__main__":
    main()
