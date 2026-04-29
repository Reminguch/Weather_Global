#!/usr/bin/env python3
"""Merge unified resolution-eval shards and render vs-res plots."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval"
DEFAULT_OUTPUT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge unified resolution-eval shards and plot vs-res curves.")
    parser.add_argument("--input-data-dir", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-csv-name", type=str, default="resolution_eval.csv")
    parser.add_argument("--lead-days", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--expected-shards", nargs="*", default=None)
    parser.add_argument("--shard-glob", type=str, default="resolution_eval_*.csv")
    parser.add_argument("--plot-prefix", type=str, default="resolution_eval")
    parser.add_argument(
        "--include-per-variable",
        action="store_true",
        help="Also render one plot per variable. By default only weighted_allvars plots are written.",
    )
    return parser.parse_args()


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "value"


def _curve_label(sub: pd.DataFrame) -> str:
    family = str(sub["family"].iloc[0])
    variant = str(sub["variant"].iloc[0])
    return variant if variant.startswith(family) else f"{family}:{variant}"


def plot_vs_res(
    df: pd.DataFrame,
    *,
    lead_day: int,
    metric_kind: str,
    eval_mode: str,
    variable: str | None,
    out_path: Path,
    title: str,
    ylabel: str,
) -> bool:
    sub = df[(df["lead_days"].astype(int) == int(lead_day)) & (df["metric_kind"] == metric_kind) & (df["eval_mode"] == eval_mode)]
    if metric_kind == "per_variable":
        sub = sub[sub["variable"] == (variable or "")]
    else:
        sub = sub[sub["variable"].fillna("") == ""]
    if sub.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    for _, curve_df in sub.groupby(["family", "variant"], sort=True):
        curve_df = curve_df.sort_values("res")
        if curve_df["value"].notna().sum() == 0:
            continue
        ax.plot(curve_df["res"], curve_df["value"], marker="o", label=_curve_label(curve_df))
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Resolution group (res)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved image: {out_path}")
    return True


def render_default_plots(
    df: pd.DataFrame,
    *,
    lead_days: list[int],
    image_dir: Path,
    plot_prefix: str,
    include_per_variable: bool = False,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for lead_day in lead_days:
        for eval_mode in sorted(df["eval_mode"].dropna().astype(str).unique().tolist()):
            weighted_png = image_dir / f"{plot_prefix}_weighted_allvars_vs_res_lead{lead_day}d_{eval_mode}.png"
            plot_vs_res(
                df,
                lead_day=lead_day,
                metric_kind="weighted_allvars",
                eval_mode=eval_mode,
                variable=None,
                out_path=weighted_png,
                title=f"Normalized weighted all-variable MSE vs res | lead={lead_day}d | {eval_mode}",
                ylabel="Normalized weighted MSE",
            )
            if not include_per_variable:
                continue
            for variable in sorted(
                df[(df["lead_days"].astype(int) == int(lead_day)) & (df["metric_kind"] == "per_variable")]["variable"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ):
                variable_png = image_dir / (
                    f"{plot_prefix}_per_variable_{_sanitize_filename_part(variable)}_vs_res_lead{lead_day}d_{eval_mode}.png"
                )
                plot_vs_res(
                    df,
                    lead_day=lead_day,
                    metric_kind="per_variable",
                    eval_mode=eval_mode,
                    variable=variable,
                    out_path=variable_png,
                    title=f"{variable} normalized MSE vs res | lead={lead_day}d | {eval_mode}",
                    ylabel=f"{variable} normalized MSE",
                )


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.input_data_dir.glob(args.shard_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No shard CSVs matching {args.shard_glob} found under {args.input_data_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["family", "variant", "res", "lead_days", "eval_mode", "metric_kind", "variable"]).reset_index(drop=True)

    found_shards = sorted({f"{row.family}:{int(row.res)}" for row in df[["family", "res"]].drop_duplicates().itertuples(index=False)})
    if args.expected_shards is not None and sorted(args.expected_shards) != found_shards:
        raise ValueError(f"Expected shards {sorted(args.expected_shards)}, found {found_shards}")

    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_data_dir / args.output_csv_name
    df.to_csv(csv_path, index=False)
    print(f"Saved merged CSV: {csv_path}")

    render_default_plots(
        df,
        lead_days=args.lead_days,
        image_dir=args.output_image_dir,
        plot_prefix=args.plot_prefix,
        include_per_variable=args.include_per_variable,
    )


if __name__ == "__main__":
    main()
