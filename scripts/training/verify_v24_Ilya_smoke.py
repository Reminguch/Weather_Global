#!/usr/bin/env python3
"""Validate v24_Ilya serial or data-parallel smoke artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v24_Ilya.training.smoke import (  # noqa: E402
    validate_smoke_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=5)
    parser.add_argument("--max-median-step-seconds", type=float, required=True)
    parser.add_argument("--gpu-peak-mib", type=int)
    parser.add_argument("--max-gpu-peak-mib", type=int)
    parser.add_argument("--require-loss-by-horizon", action="store_true")
    args = parser.parse_args()
    summary = validate_smoke_artifacts(
        args.metrics,
        args.checkpoint,
        expected_steps=args.expected_steps,
        max_median_step_seconds=args.max_median_step_seconds,
        gpu_peak_mib=args.gpu_peak_mib,
        max_gpu_peak_mib=args.max_gpu_peak_mib,
        require_loss_by_horizon=args.require_loss_by_horizon,
    )
    gpu_text = (
        "unknown" if summary.gpu_peak_mib is None else str(summary.gpu_peak_mib)
    )
    print(
        "v24_smoke_gate_passed "
        f"steps={summary.expected_steps} "
        f"median_steady_step_seconds={summary.median_steady_step_seconds:.2f} "
        f"gpu_peak_mib={gpu_text}"
    )


if __name__ == "__main__":
    main()
