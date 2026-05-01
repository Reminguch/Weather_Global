#!/usr/bin/env python3
"""Plot width-overlaid resolution-eval sweeps from merged CSVs or shard dirs."""

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
VARIANT_RE = re.compile(r"res(?P<res>\d+)_m(?P<mesh>\d+)_w(?P<width>\d+)_mp(?P<mp>\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot width-overlaid resolution-eval sweeps by message-passing count. "
            "Reads either one merged CSV via --input-csv or many shard CSVs via --input-dir/--shard-glob."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=None, help="Merged unified resolution-eval CSV to plot.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Shard directory used when --input-csv is not provided.",
    )
    parser.add_argument("--shard-glob", type=str, default=DEFAULT_SHARD_GLOB)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Annotated CSV with width/mp/mesh columns.")
    parser.add_argument("--no-output-csv", action="store_true", help="Skip writing the annotated CSV.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--plot-prefix", type=str, default="standard_gc")
    parser.add_argument("--title-prefix", type=str, default="Standard GraphCast")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--lead-days", type=int, nargs="+", default=None)
    parser.add_argument("--eval-modes", nargs="+", default=None)
    parser.add_argument("--mp", type=int, nargs="+", default=None)
    parser.add_argument("--width", type=int, nargs="+", default=None)
    parser.add_argument("--include-per-variable", action="store_true")
    return parser.parse_args()


def _parse_variant(variant: str) -> dict[str, int]:
    match = VARIANT_RE.search(variant)
    if not match:
        raise ValueError(f"Could not parse resolution/mesh/width/mp tokens from variant: {variant}")
    return {key: int(value) for key, value in match.groupdict().items()}


def load_input(input_csv: Path | None, input_dir: Path, shard_glob: str) -> pd.DataFrame:
    if input_csv is not None:
        return pd.read_csv(input_csv)

    paths = sorted(input_dir.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"No shards matching {shard_glob} under {input_dir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def annotate_width_sweep(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["variant"].astype(str).map(_parse_variant).apply(pd.Series)
    for col in ["mesh", "width", "mp"]:
        df[col] = parsed[col].astype(int)
    df["parsed_res"] = parsed["res"].astype(int)
    bad_res = df[df["res"].astype(int) != df["parsed_res"].astype(int)]
    if not bad_res.empty:
        variants = sorted(bad_res["variant"].astype(str).unique().tolist())
        raise ValueError(f"Variant res token disagrees with CSV res for: {variants}")
    return df.drop(columns=["parsed_res"]).sort_values(
        ["res", "mp", "width", "lead_days", "eval_mode", "metric_kind", "variable", "variant"]
    )


def _metric_subset(df: pd.DataFrame, metric_kind: str, variable: str) -> pd.DataFrame:
    if metric_kind == "weighted_allvars":
        return df[(df["metric_kind"] == metric_kind) & (df["variable"].fillna("") == "")]
    return df[(df["metric_kind"] == metric_kind) & (df["variable"].fillna("") == variable)]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "value"


def _metric_name(metric_kind: str, variable: str) -> str:
    return "weighted_allvars" if metric_kind == "weighted_allvars" else variable


def plot_metric(
    df: pd.DataFrame,
    *,
    metric_kind: str,
    variable: str,
    lead_day: int,
    eval_mode: str,
    mp: int,
    title_prefix: str,
    out_path: Path,
) -> bool:
    sub = _metric_subset(df, metric_kind, variable)
    sub = sub[
        (sub["lead_days"].astype(int) == int(lead_day))
        & (sub["eval_mode"].astype(str) == eval_mode)
        & (sub["mp"].astype(int) == int(mp))
    ]
    if sub.empty:
        return False

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    widths = sorted(sub["width"].astype(int).unique().tolist())
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(widths)))
    plotted = False
    for color, width in zip(colors, widths):
        curve = (
            sub[sub["width"].astype(int) == width]
            .groupby("res", as_index=False)["value"]
            .mean()
            .sort_values("res")
        )
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
    ax.set_title(f"{title_prefix} {metric_label} vs res | mp={mp}, lead={lead_day}d, {eval_mode}")
    ax.set_xticks(sorted(sub["res"].astype(int).unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Width", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")
    return True


def render_plots(
    df: pd.DataFrame,
    *,
    image_dir: Path,
    plot_prefix: str,
    title_prefix: str,
    lead_days: list[int],
    include_per_variable: bool,
) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = [("weighted_allvars", "")]
    if include_per_variable:
        variables = sorted(df[df["metric_kind"] == "per_variable"]["variable"].dropna().astype(str).unique().tolist())
        metric_specs.extend(("per_variable", variable) for variable in variables)

    saved: list[Path] = []
    eval_modes = sorted(df["eval_mode"].dropna().astype(str).unique().tolist())
    mp_values = sorted(df["mp"].astype(int).unique().tolist())
    for lead_day in lead_days:
        for eval_mode in eval_modes:
            for mp in mp_values:
                for metric_kind, variable in metric_specs:
                    metric = _safe_name(_metric_name(metric_kind, variable))
                    out_path = image_dir / f"{plot_prefix}_{metric}_vs_res_mp{mp}_lead{lead_day}d_{eval_mode}.png"
                    if plot_metric(
                        df,
                        metric_kind=metric_kind,
                        variable=variable,
                        lead_day=lead_day,
                        eval_mode=eval_mode,
                        mp=mp,
                        title_prefix=title_prefix,
                        out_path=out_path,
                    ):
                        saved.append(out_path)
    return saved


def main() -> None:
    args = parse_args()
    df = annotate_width_sweep(load_input(args.input_csv, args.input_dir, args.shard_glob))
    if args.families is not None:
        df = df[df["family"].astype(str).isin(set(args.families))]
    if args.lead_days is not None:
        df = df[df["lead_days"].astype(int).isin(set(args.lead_days))]
    if args.eval_modes is not None:
        df = df[df["eval_mode"].astype(str).isin(set(args.eval_modes))]
    if args.mp is not None:
        df = df[df["mp"].astype(int).isin(set(args.mp))]
    if args.width is not None:
        df = df[df["width"].astype(int).isin(set(args.width))]
    if df.empty:
        raise ValueError("No rows left after filters.")

    if not args.no_output_csv and args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved annotated CSV: {args.output_csv}")

    lead_days = args.lead_days or sorted(df["lead_days"].astype(int).unique().tolist())
    saved = render_plots(
        df,
        image_dir=args.image_dir,
        plot_prefix=args.plot_prefix,
        title_prefix=args.title_prefix,
        lead_days=lead_days,
        include_per_variable=args.include_per_variable,
    )
    print(f"Saved {len(saved)} images")


if __name__ == "__main__":
    main()
