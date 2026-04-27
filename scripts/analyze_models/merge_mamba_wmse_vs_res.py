#!/usr/bin/env python3
"""Merge per-resolution mamba WMSE CSVs and regenerate aggregate plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_models.mamba_wmse_vs_res import OUTPUT_DATA_DIR, OUTPUT_IMAGE_DIR, _plot_wmse_vs_res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-resolution mamba WMSE CSV shards.")
    p.add_argument("--input-data-dir", type=Path, required=True)
    p.add_argument("--resolutions", type=int, nargs="+", default=None)
    p.add_argument("--output-data-dir", type=Path, default=ROOT / OUTPUT_DATA_DIR)
    p.add_argument("--output-image-dir", type=Path, default=ROOT / OUTPUT_IMAGE_DIR)
    p.add_argument("--output-csv-name", type=str, default="mamba_wmse_vs_res.csv")
    p.add_argument("--lead-days", type=int, nargs="+", default=[1, 2, 4])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    expected_res = sorted(args.resolutions) if args.resolutions is not None else None

    csv_paths = sorted(args.input_data_dir.glob("mamba_wmse_vs_res_res*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No shard CSVs found under {args.input_data_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["model_type", "di", "res", "lead_days"]).reset_index(drop=True)

    found_res = sorted(df["res"].dropna().astype(int).unique().tolist())
    if expected_res is not None and found_res != expected_res:
        raise ValueError(f"Expected shard resolutions {expected_res}, found {found_res}")

    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_data_dir / args.output_csv_name
    df.to_csv(csv_path, index=False)
    print(f"Saved merged CSV: {csv_path}")

    for lead_day in args.lead_days:
        out_png = args.output_image_dir / f"mamba_grid15_wmse_vs_res_lead{lead_day}d.png"
        _plot_wmse_vs_res(
            df,
            lead_day=lead_day,
            metric_col="grid15_weighted_mse_allvars_normalized_cold",
            metric_label="15-grid Weighted MSE (all vars, normalized) [cold]",
            out_path=out_png,
        )
        warm_png = args.output_image_dir / f"mamba_grid15_wmse_vs_res_lead{lead_day}d_warm.png"
        _plot_wmse_vs_res(
            df,
            lead_day=lead_day,
            metric_col="grid15_weighted_mse_allvars_normalized_warm",
            baseline_metric_col="grid15_weighted_mse_allvars_normalized_cold",
            metric_label="15-grid Weighted MSE (all vars, normalized) [warm]",
            out_path=warm_png,
        )


if __name__ == "__main__":
    main()
