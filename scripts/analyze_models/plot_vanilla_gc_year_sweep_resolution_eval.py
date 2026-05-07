#!/usr/bin/env python3
"""Plot vanilla GraphCast resolution sweeps with message-passing curves split by training years."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = ROOT / "plots/analyze_models/data/resolution_eval/vanilla_gc_3y_7y_nores1/resolution_eval.csv"
DEFAULT_OUTPUT_CSV = ROOT / "plots/analyze_models/data/resolution_eval/vanilla_gc_3y_7y_nores1/annotated_resolution_eval.csv"
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval/vanilla_gc_3y_7y_nores1/mp_curves_by_year"
DEFAULT_RESOLUTIONS = [2, 3, 4, 6, 9, 15, 18]
DEFAULT_MP = [2, 4, 6]

VARIANT_RE = re.compile(
    r"vanilla_gc_(?P<years>\d+)y_.*?res(?P<res>\d+)_m(?P<mesh>\d+)_w(?P<width>\d+)_mp(?P<mp>\d+)(?:_|$)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--no-output-csv", action="store_true")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--plot-prefix", type=str, default="vanilla_gc")
    parser.add_argument("--title-prefix", type=str, default="Vanilla GraphCast")
    parser.add_argument("--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--mp", type=int, nargs="+", default=DEFAULT_MP)
    parser.add_argument("--lead-days", type=int, nargs="+", default=None)
    parser.add_argument("--eval-modes", nargs="+", default=None)
    parser.add_argument("--include-per-variable", action="store_true")
    return parser.parse_args()


def _parse_variant(variant: str) -> dict[str, int]:
    match = VARIANT_RE.search(variant)
    if not match:
        raise ValueError(f"Could not parse year/res/mesh/width/mp tokens from variant: {variant}")
    return {key: int(value) for key, value in match.groupdict().items()}


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["variant"].astype(str).map(_parse_variant).apply(pd.Series)
    for col in ["years", "mesh", "width", "mp"]:
        df[col] = parsed[col].astype(int)
    df["parsed_res"] = parsed["res"].astype(int)
    bad_res = df[df["res"].astype(int) != df["parsed_res"]]
    if not bad_res.empty:
        variants = sorted(bad_res["variant"].astype(str).unique().tolist())
        raise ValueError(f"Variant res token disagrees with CSV res for: {variants}")
    return df.drop(columns=["parsed_res"]).sort_values(
        ["years", "res", "mp", "mesh", "lead_days", "eval_mode", "metric_kind", "variable", "variant"]
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
    years: int,
    metric_kind: str,
    variable: str,
    lead_day: int,
    eval_mode: str,
    title_prefix: str,
    out_path: Path,
) -> bool:
    sub = _metric_subset(df, metric_kind, variable)
    sub = sub[
        (sub["years"].astype(int) == int(years))
        & (sub["lead_days"].astype(int) == int(lead_day))
        & (sub["eval_mode"].astype(str) == eval_mode)
    ]
    if sub.empty:
        return False

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = {2: "#2166ac", 4: "#1b9e77", 6: "#d95f02"}
    plotted = False
    for mp in sorted(sub["mp"].astype(int).unique().tolist()):
        curve = sub[sub["mp"].astype(int) == mp].groupby("res", as_index=False)["value"].mean().sort_values("res")
        if curve["value"].notna().sum() == 0:
            continue
        ax.plot(curve["res"], curve["value"], marker="o", linewidth=2.0, color=colors.get(mp), label=f"mp={mp}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    metric_label = _metric_name(metric_kind, variable)
    ax.set_xlabel("Resolution group (res)")
    ax.set_ylabel("Normalized MSE")
    ax.set_title(f"{title_prefix} {years}y {metric_label} vs res | lead={lead_day}d, {eval_mode}")
    ax.set_xticks(sorted(sub["res"].astype(int).unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Message passing", fontsize=8)
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
    include_per_variable: bool,
) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = [("weighted_allvars", "")]
    if include_per_variable:
        variables = sorted(df[df["metric_kind"] == "per_variable"]["variable"].dropna().astype(str).unique().tolist())
        metric_specs.extend(("per_variable", variable) for variable in variables)

    saved: list[Path] = []
    years_values = sorted(df["years"].astype(int).unique().tolist())
    lead_days = sorted(df["lead_days"].astype(int).unique().tolist())
    eval_modes = sorted(df["eval_mode"].dropna().astype(str).unique().tolist())
    for years in years_values:
        for lead_day in lead_days:
            for eval_mode in eval_modes:
                for metric_kind, variable in metric_specs:
                    metric = _safe_name(_metric_name(metric_kind, variable))
                    out_path = image_dir / f"{plot_prefix}_{years}y_{metric}_mp_vs_res_lead{lead_day}d_{eval_mode}.png"
                    if plot_metric(
                        df,
                        years=years,
                        metric_kind=metric_kind,
                        variable=variable,
                        lead_day=lead_day,
                        eval_mode=eval_mode,
                        title_prefix=title_prefix,
                        out_path=out_path,
                    ):
                        saved.append(out_path)
    return saved


def main() -> None:
    args = parse_args()
    df = annotate(pd.read_csv(args.input_csv))
    if args.resolutions is not None:
        df = df[df["res"].astype(int).isin(set(args.resolutions))]
    if args.mp is not None:
        df = df[df["mp"].astype(int).isin(set(args.mp))]
    if args.lead_days is not None:
        df = df[df["lead_days"].astype(int).isin(set(args.lead_days))]
    if args.eval_modes is not None:
        df = df[df["eval_mode"].astype(str).isin(set(args.eval_modes))]
    if df.empty:
        raise ValueError("No rows left after filters.")

    if not args.no_output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved annotated CSV: {args.output_csv}")

    saved = render_plots(
        df,
        image_dir=args.image_dir,
        plot_prefix=args.plot_prefix,
        title_prefix=args.title_prefix,
        include_per_variable=args.include_per_variable,
    )
    print(f"Saved {len(saved)} images")


if __name__ == "__main__":
    main()
