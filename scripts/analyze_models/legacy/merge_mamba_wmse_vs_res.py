#!/usr/bin/env python3
"""Legacy compatibility wrapper for the old mamba-vs-res merge entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_models.merge_resolution_eval import plot_vs_res, render_default_plots
from scripts.analyze_models.legacy.mamba_wmse_vs_res import OUTPUT_DATA_DIR, OUTPUT_IMAGE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-resolution mamba WMSE CSV shards.")
    parser.add_argument("--input-data-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=None)
    parser.add_argument("--output-data-dir", type=Path, default=ROOT / OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=ROOT / OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-csv-name", type=str, default="mamba_wmse_vs_res.csv")
    parser.add_argument("--lead-days", type=int, nargs="+", default=[1, 2, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.input_data_dir.glob("mamba_wmse_vs_res_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No shard CSVs found under {args.input_data_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    if args.resolutions is not None:
        found_res = sorted(df["res"].dropna().astype(int).unique().tolist())
        if found_res != sorted(args.resolutions):
            raise ValueError(f"Expected shard resolutions {sorted(args.resolutions)}, found {found_res}")

    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_data_dir / args.output_csv_name
    df.to_csv(csv_path, index=False)
    print(f"Saved merged CSV: {csv_path}")

    render_default_plots(df, lead_days=args.lead_days, image_dir=args.output_image_dir, plot_prefix="mamba")
    for lead_day in args.lead_days:
        cold_png = args.output_image_dir / f"mamba_grid15_wmse_vs_res_lead{lead_day}d.png"
        plot_vs_res(
            df,
            lead_day=lead_day,
            metric_kind="weighted_allvars",
            eval_mode="cold",
            variable=None,
            out_path=cold_png,
            title=f"15-grid Weighted MSE (normalized) vs res | lead={lead_day}d",
            ylabel="15-grid Weighted MSE (normalized) [cold]",
        )
        warm_png = args.output_image_dir / f"mamba_grid15_wmse_vs_res_lead{lead_day}d_warm.png"
        plot_vs_res(
            df,
            lead_day=lead_day,
            metric_kind="weighted_allvars",
            eval_mode="warm",
            variable=None,
            out_path=warm_png,
            title=f"15-grid Weighted MSE (normalized) vs res | lead={lead_day}d",
            ylabel="15-grid Weighted MSE (normalized) [warm]",
        )


if __name__ == "__main__":
    main()
