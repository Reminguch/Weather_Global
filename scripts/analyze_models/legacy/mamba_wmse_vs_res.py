#!/usr/bin/env python3
"""Legacy compatibility wrapper for the old mamba-vs-res entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_models.unified_resolution_eval import discover_model_entries, main as unified_main

OUTPUT_DATA_DIR = "plots/analyze_models/data/mamba_res"
OUTPUT_IMAGE_DIR = "plots/analyze_models/images/mamba_res"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for mamba-vs-res evaluation.")
    parser.add_argument("--n-eval-days", type=int, default=365)
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--lead-days", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--resolutions", type=int, nargs="+", default=None)
    parser.add_argument("--output-data-dir", type=Path, default=ROOT / OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=ROOT / OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-csv-name", type=str, default="mamba_wmse_vs_res.csv")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=24)
    parser.add_argument("--trunk-steps", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolutions = args.resolutions
    if resolutions is None:
        residual_entries = discover_model_entries(["residual_mamba"], None)
        resolutions = sorted({entry.res for entry in residual_entries})
    argv = [
        "unified_resolution_eval.py",
        "--families",
        "graphcast",
        "residual_mamba",
        "--metrics",
        "weighted_allvars",
        "per_variable",
        "--eval-modes",
        "cold",
        "warm",
        "--n-eval-days",
        str(args.n_eval_days),
        "--window-batch-size",
        str(args.window_batch_size),
        "--warmup-steps",
        str(args.warmup_steps),
        "--trunk-steps",
        str(args.trunk_steps),
        "--output-data-dir",
        str(args.output_data_dir),
        "--output-csv-name",
        args.output_csv_name,
        "--lead-days",
        *[str(day) for day in args.lead_days],
        "--resolutions",
        *[str(res) for res in resolutions],
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        unified_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
