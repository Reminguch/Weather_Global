#!/usr/bin/env python3
"""Merge standard GraphCast resolution-eval shards and plot vs-res curves.

This is for unified-resolution-eval CSVs with rows keyed by
family/variant/res/lead_days/eval_mode/metric_kind/variable/value.
For GraphCast sweeps, variants encode width and message-passing count:
  res2_m4_w1024_mp1_h6_bs4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "plots/analyze_models/data/resolution_eval/shards"
DEFAULT_OUTPUT_CSV = ROOT / "plots/analyze_models/data/resolution_eval/standard_gc_resolution_eval.csv"
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/standard_gc/resolution_eval"
DEFAULT_SHARD_GLOB = "resolution_eval_graphcast_res*.csv"

VARIANT_RE = re.compile(r"res(?P<res>\d+)_m(?P<mesh>\d+)_w(?P<width>\d+)_mp(?P<mp>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot standard GraphCast resolution-eval shards.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--shard-glob", type=str, default=DEFAULT_SHARD_GLOB)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--lead-days", type=int, nargs="+", default=None)
    parser.add_argument(
        "--include-per-variable",
        action="store_true",
        help="Also render one plot per variable. By default only weighted_allvars plots are written.",
    )
    return parser.parse_args()


def _parse_variant(variant: str) -> dict[str, int]:
    match = VARIANT_RE.search(variant)
    if not match:
        raise ValueError(f"Could not parse GraphCast variant: {variant}")
    return {key: int(value) for key, value in match.groupdict().items()}


def load_shards(input_dir: Path, shard_glob: str) -> pd.DataFrame:
    paths = sorted(input_dir.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"No shards matching {shard_glob} under {input_dir}")

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    parsed = df["variant"].astype(str).map(_parse_variant).apply(pd.Series)
    for col in ["width", "mp", "mesh"]:
        df[col] = parsed[col].astype(int)

    df = df.sort_values(
        ["family", "res", "mp", "width", "lead_days", "eval_mode", "metric_kind", "variable"]
    ).reset_index(drop=True)
    return df


def _metric_subset(df: pd.DataFrame, metric_kind: str, variable: str) -> pd.DataFrame:
    if metric_kind == "weighted_allvars":
        return df[(df["metric_kind"] == metric_kind) & (df["variable"].fillna("") == "")]
    return df[(df["metric_kind"] == metric_kind) & (df["variable"].fillna("") == variable)]


def _metric_name(metric_kind: str, variable: str) -> str:
    return "weighted_allvars" if metric_kind == "weighted_allvars" else variable


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def plot_metric(df: pd.DataFrame, *, metric_kind: str, variable: str, lead_days: int, eval_mode: str, mp: int, out: Path) -> bool:
    sub = _metric_subset(df, metric_kind, variable)
    sub = sub[
        (sub["lead_days"].astype(int) == int(lead_days))
        & (sub["eval_mode"].astype(str) == eval_mode)
        & (sub["mp"].astype(int) == int(mp))
    ]
    if sub.empty:
        return False

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    widths = sorted(sub["width"].astype(int).unique().tolist())
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(widths)))
    plotted = False
    for color, width in zip(colors, widths):
        curve = sub[sub["width"].astype(int) == width].sort_values("res")
        if curve["value"].notna().sum() == 0:
            continue
        ax.plot(curve["res"], curve["value"], marker="o", color=color, label=f"w={width}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    metric_label = _metric_name(metric_kind, variable)
    ax.set_xlabel("Resolution group (res)")
    ax.set_ylabel("Normalized MSE")
    ax.set_title(f"Standard GraphCast {metric_label} vs res | mp={mp}, lead={lead_days}d, {eval_mode}")
    ax.set_xticks(sorted(sub["res"].astype(int).unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Width", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def render_plots(
    df: pd.DataFrame,
    image_dir: Path,
    lead_days: list[int],
    *,
    include_per_variable: bool = False,
) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    metric_specs = [("weighted_allvars", "")]
    if include_per_variable:
        variables = sorted(
            df[df["metric_kind"] == "per_variable"]["variable"].dropna().astype(str).unique().tolist()
        )
        metric_specs.extend(("per_variable", variable) for variable in variables)

    for lead_day in lead_days:
        for eval_mode in sorted(df["eval_mode"].dropna().astype(str).unique().tolist()):
            for mp in sorted(df["mp"].astype(int).unique().tolist()):
                for metric_kind, variable in metric_specs:
                    metric = _safe_name(_metric_name(metric_kind, variable))
                    out = image_dir / f"standard_gc_{metric}_vs_res_mp{mp}_lead{lead_day}d_{eval_mode}.png"
                    if plot_metric(
                        df,
                        metric_kind=metric_kind,
                        variable=variable,
                        lead_days=lead_day,
                        eval_mode=eval_mode,
                        mp=mp,
                        out=out,
                    ):
                        saved.append(out)
                        print(f"Saved image: {out}")
    return saved


def main() -> None:
    args = parse_args()
    df = load_shards(args.input_dir, args.shard_glob)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved merged CSV: {args.output_csv}")

    lead_days = args.lead_days or sorted(df["lead_days"].astype(int).unique().tolist())
    saved = render_plots(df, args.image_dir, lead_days, include_per_variable=args.include_per_variable)
    print(f"Saved {len(saved)} images")


if __name__ == "__main__":
    main()
