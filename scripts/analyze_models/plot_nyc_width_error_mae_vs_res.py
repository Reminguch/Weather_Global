#!/usr/bin/env python3
"""Plot NYC loss-vs-resolution curves from per-(res, mp) CSVs.

Each figure corresponds to one (mp, lead_days) pair:
  x-axis: resolution group (res)
  y-axis: selected loss metric
  curves: different model widths

Expected input CSVs are produced by:
  scripts/analyze_models/nyc_width_error_by_res_mp.py

Examples:
  python scripts/analyze_models/plot_nyc_width_error_mae_vs_res.py
  python scripts/analyze_models/plot_nyc_width_error_mae_vs_res.py --mp 1
  python scripts/analyze_models/plot_nyc_width_error_mae_vs_res.py --mp 1 2 --lead-days 1 2 4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = ROOT / "plots/analyze_models"
DEFAULT_INPUT_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_IMAGE_DIR = DEFAULT_BASE_DIR / "images"
DEFAULT_MERGED_DATA_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_RES_LIST = [1, 2, 4, 6, 9, 12, 15]
METRIC_SPECS = [
    {
        "name": "mae",
        "col": "mae_c",
        "ylabel": "Point MAE (degC)",
        "title": "Point 2m Temperature MAE vs res @ (30N, 90W)",
        "filename_pattern": "nyc_mae_vs_res_mp{mp}_lead{lead_days}d.png",
    },
    {
        "name": "pointall_wmse",
        "col": "point_weighted_mse_allvars_normalized",
        "fallback_col": "point_weighted_mse_allvars",
        "ylabel": "Point Weighted MSE (all vars, normalized)",
        "title": "Point All-Feature Weighted MSE (normalized) vs res @ (30N, 90W)",
        "filename_pattern": "nyc_pointall_wmse_vs_res_mp{mp}_lead{lead_days}d.png",
    },
    {
        "name": "grid15_wmse",
        "col": "grid15_weighted_mse_allvars_normalized",
        "fallback_col": "grid15_weighted_mse_allvars",
        "ylabel": "15-grid Weighted MSE (all vars, normalized)",
        "title": "15-grid Averaged Weighted MSE (normalized) vs res @ (30N, 90W)",
        "filename_pattern": "nyc_grid15_wmse_vs_res_mp{mp}_lead{lead_days}d.png",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MAE vs res with width curves for each (mp, lead_days).")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing nyc_width_error_res*_mp*.csv")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory where plots are written")
    p.add_argument("--merged-data-dir", type=Path, default=DEFAULT_MERGED_DATA_DIR, help="Directory where merged CSV is written")
    p.add_argument("--mp", type=int, nargs="+", default=None, help="Optional mp filter, e.g. --mp 1 2")
    p.add_argument("--lead-days", type=int, nargs="+", default=None, help="Optional lead-day filter, e.g. --lead-days 1 2 4")
    p.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=DEFAULT_RES_LIST,
        help="Resolution filter; default is 1 2 4 6 9 12 15",
    )
    p.add_argument("--width", type=int, nargs="+", default=None, help="Optional width filter")
    p.add_argument("--per-width-only", action="store_true", help="Use only per-width CSVs (res*_mp*_w*.csv), ignoring old combined files")
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=[str(spec["name"]) for spec in METRIC_SPECS],
        default=[str(spec["name"]) for spec in METRIC_SPECS],
        help="Metric plots to generate. Default: all metrics.",
    )
    p.add_argument(
        "--combine-mp-label",
        type=str,
        default=None,
        help="If set, combine all selected mp values into one plot per lead/metric and use this label in filenames.",
    )
    return p.parse_args()


def _extract_res_mp(path: Path) -> tuple[int, int] | None:
    # Matches both combined (res*_mp*.csv) and per-width (res*_mp*_w*.csv) files.
    m = re.match(r"nyc_width_error_res(\d+)_mp(\d+)(?:_w\d+)?\.csv$", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _discover_csvs(input_dir: Path, per_width_only: bool = False) -> list[Path]:
    all_paths = sorted(input_dir.glob("nyc_width_error_res*_mp*.csv"))
    if per_width_only:
        all_paths = [p for p in all_paths if re.search(r"_w\d+\.csv$", p.name)]
    return [p for p in all_paths if _extract_res_mp(p) is not None]


def _load_csv(path: Path) -> pd.DataFrame:
    parsed = _extract_res_mp(path)
    if parsed is None:
        raise ValueError(f"Unexpected CSV filename format: {path.name}")
    file_res, file_mp = parsed

    df = pd.read_csv(path)
    required = {"width", "lead_days", "mae_c", "point_weighted_mse_allvars", "grid15_weighted_mse_allvars"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    if "res" not in df.columns:
        df["res"] = file_res
    if "mp" not in df.columns:
        df["mp"] = file_mp
    return df


def _dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["mp", "lead_days", "res", "width"]
    if "ckpt_step" in df.columns:
        work = df.sort_values(keys + ["ckpt_step"])
    else:
        work = df.copy()
    out = work.drop_duplicates(subset=keys, keep="last")
    return out.sort_values(["mp", "lead_days", "res", "width"]).reset_index(drop=True)


def _apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.mp:
        out = out[out["mp"].astype(int).isin(set(args.mp))]
    if args.lead_days:
        out = out[out["lead_days"].astype(int).isin(set(args.lead_days))]
    if args.res:
        out = out[out["res"].astype(int).isin(set(args.res))]
    if args.width:
        out = out[out["width"].astype(int).isin(set(args.width))]
    return out


def _plot_one(
    df: pd.DataFrame,
    mp: int | None,
    lead_days: int,
    metric_col: str,
    metric_fallback_col: str | None,
    ylabel: str,
    title_prefix: str,
    out_png: Path,
    mp_label: str | None = None,
) -> bool:
    ddf = df[df["lead_days"].astype(int) == lead_days]
    if mp is not None:
        ddf = ddf[ddf["mp"].astype(int) == mp]
    if ddf.empty:
        return False
    col = metric_col if metric_col in ddf.columns else metric_fallback_col
    if col is None or col not in ddf.columns:
        return False

    res_values = sorted(ddf["res"].astype(int).unique().tolist())
    width_values = sorted(ddf["width"].astype(int).unique().tolist())
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(width_values)))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for color, width in zip(colors, width_values):
        wdf = ddf[ddf["width"].astype(int) == width].sort_values("res")
        ax.plot(wdf["res"], wdf[col], marker="o", color=color, label=f"w={width}")

    ax.set_xlabel("Resolution group (res)")
    ax.set_ylabel(ylabel)
    title_mp = f"mp={mp}" if mp_label is None else mp_label
    ax.set_title(f"{title_prefix} | {title_mp}, lead={lead_days}d")
    ax.set_xticks(res_values)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model width")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.merged_data_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _discover_csvs(args.input_dir, per_width_only=args.per_width_only)
    if not csv_paths:
        raise RuntimeError(f"No CSVs found in {args.input_dir} matching nyc_width_error_res*_mp*.csv")

    dfs = [_load_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df = _dedupe_latest(df)
    df = _apply_filters(df, args)

    if df.empty:
        raise RuntimeError("No rows left after filters.")

    merged_csv = args.merged_data_dir / "nyc_width_error_merged_for_mae_vs_res.csv"
    df.to_csv(merged_csv, index=False)
    print(f"Loaded {len(csv_paths)} source CSVs")
    print(f"Merged CSV: {merged_csv}")

    lead_values = sorted(df["lead_days"].astype(int).unique().tolist())
    metrics = [metric for metric in METRIC_SPECS if metric["name"] in args.metrics]

    if args.combine_mp_label:
        for lead_days in lead_values:
            for metric in metrics:
                out_png = args.image_dir / metric["filename_pattern"].format(
                    mp=args.combine_mp_label,
                    lead_days=lead_days,
                )
                if _plot_one(
                    df,
                    mp=None,
                    lead_days=lead_days,
                    metric_col=metric["col"],
                    metric_fallback_col=metric.get("fallback_col"),
                    ylabel=metric["ylabel"],
                    title_prefix=metric["title"],
                    out_png=out_png,
                    mp_label=args.combine_mp_label,
                ):
                    print(f"Saved plot: {out_png}")
                else:
                    print(f"[warn] no data for {args.combine_mp_label}, lead={lead_days}d, metric={metric['col']}")
        return

    mp_values = sorted(df["mp"].astype(int).unique().tolist())
    for mp in mp_values:
        for lead_days in lead_values:
            for metric in metrics:
                out_png = args.image_dir / metric["filename_pattern"].format(mp=mp, lead_days=lead_days)
                if _plot_one(
                    df,
                    mp,
                    lead_days,
                    metric_col=metric["col"],
                    metric_fallback_col=metric.get("fallback_col"),
                    ylabel=metric["ylabel"],
                    title_prefix=metric["title"],
                    out_png=out_png,
                ):
                    print(f"Saved plot: {out_png}")
                else:
                    print(f"[warn] no data for mp={mp}, lead={lead_days}d, metric={metric['col']}")


if __name__ == "__main__":
    main()
