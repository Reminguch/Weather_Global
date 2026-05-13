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
DEFAULT_RESOLUTIONS = [1, 2, 3, 4, 6, 9, 18]
DEFAULT_PER_VARIABLE_PLOTS = ["2m_temperature", "2m_temperature_nyc"]
RES_MESH_TOKEN_RE = re.compile(r"_res\d+_m\d+(?=_)")
DI_DS_TOKEN_RE = re.compile(r"_di(?P<di>\d+)_ds(?P<ds>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge unified resolution-eval shards and plot vs-res curves.")
    parser.add_argument("--input-data-dir", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-csv-name", type=str, default="resolution_eval.csv")
    parser.add_argument("--lead-days", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--lead-steps",
        type=int,
        nargs="+",
        default=None,
        help="Render explicit autoregressive lead steps. Overrides --lead-days when set.",
    )
    parser.add_argument("--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--expected-shards", nargs="*", default=None)
    parser.add_argument("--shard-glob", type=str, default="resolution_eval_*.csv")
    parser.add_argument("--plot-prefix", type=str, default="resolution_eval")
    parser.add_argument(
        "--include-per-variable",
        action="store_true",
        help=(
            "Render one plot per variable. By default only weighted_allvars plus "
            "selected default per-variable plots are written."
        ),
    )
    parser.add_argument(
        "--default-per-variable-plots",
        nargs="*",
        default=DEFAULT_PER_VARIABLE_PLOTS,
        help=(
            "Per-variable metrics to render by default when present. "
            "Use an empty value to render only weighted_allvars unless --include-per-variable is set."
        ),
    )
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--baseline-label", type=str, default="DeepMind GraphCast small res1 cold")
    parser.add_argument("--baseline-eval-mode", type=str, default="cold")
    parser.add_argument("--baseline-res", type=int, default=1)
    parser.add_argument("--baseline-variant", type=str, default=None)
    return parser.parse_args()


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "value"


def _ensure_lead_steps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lead_days = pd.to_numeric(df["lead_days"], errors="coerce")
    derived_steps = (lead_days * 24.0 / 6.0).round()
    if "lead_steps" not in df.columns:
        df["lead_steps"] = derived_steps
    else:
        lead_steps = pd.to_numeric(df["lead_steps"], errors="coerce")
        df["lead_steps"] = lead_steps.fillna(derived_steps)
    df["lead_steps"] = df["lead_steps"].astype(int)
    df["lead_days"] = lead_days
    return df


def _lead_label(lead_step: int) -> str:
    hours = int(lead_step) * 6
    if hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


def _lead_steps_from_days(lead_days: list[int]) -> list[int]:
    return [int((24 * int(day)) // 6) for day in lead_days]


def _curve_label(sub: pd.DataFrame) -> str:
    family = str(sub["family"].iloc[0])
    model_key = str(sub["model_key"].iloc[0])
    return model_key if model_key.startswith(family) else f"{family}:{model_key}"


def _model_key(variant: str) -> str:
    return RES_MESH_TOKEN_RE.sub("", variant, count=1)


def _curve_style(curve_df: pd.DataFrame) -> dict[str, str]:
    family = str(curve_df["family"].iloc[0])
    variant = str(curve_df["variant"].iloc[0])
    marker = "o"
    match = DI_DS_TOKEN_RE.search(variant)
    if match:
        di = int(match.group("di"))
        marker = "^" if di >= 256 else "s"

    if family == "graphcast":
        return {"color": "#2f2f2f", "marker": "o", "linestyle": "-"}
    if "_frozen50k_release20k" in variant:
        return {"color": "#d62728", "marker": marker, "linestyle": "--"}
    if "_frozen50k" in variant:
        return {"color": "#1f77b4", "marker": marker, "linestyle": "-"}
    return {"marker": marker, "linestyle": "-"}


def _matching_baseline_rows(
    baseline_df: pd.DataFrame | None,
    *,
    lead_step: int,
    metric_kind: str,
    eval_mode: str,
    variable: str | None,
    baseline_res: int,
    baseline_variant: str | None,
) -> pd.DataFrame:
    if baseline_df is None or baseline_df.empty:
        return pd.DataFrame()
    sub = baseline_df[
        (baseline_df["lead_steps"].astype(int) == int(lead_step))
        & (baseline_df["metric_kind"] == metric_kind)
        & (baseline_df["eval_mode"].astype(str) == eval_mode)
        & (baseline_df["res"].astype(int) == int(baseline_res))
    ]
    if baseline_variant:
        sub = sub[sub["variant"].astype(str) == baseline_variant]
    if metric_kind == "per_variable":
        sub = sub[sub["variable"] == (variable or "")]
    else:
        sub = sub[sub["variable"].fillna("") == ""]
    return sub


def plot_vs_res(
    df: pd.DataFrame,
    *,
    lead_step: int,
    metric_kind: str,
    eval_mode: str,
    variable: str | None,
    out_path: Path,
    title: str,
    ylabel: str,
    baseline_df: pd.DataFrame | None = None,
    baseline_label: str = "Baseline",
    baseline_eval_mode: str = "cold",
    baseline_res: int = 1,
    baseline_variant: str | None = None,
) -> bool:
    sub = df[
        (df["lead_steps"].astype(int) == int(lead_step))
        & (df["metric_kind"] == metric_kind)
        & (df["eval_mode"] == eval_mode)
    ]
    if metric_kind == "per_variable":
        sub = sub[sub["variable"] == (variable or "")]
    else:
        sub = sub[sub["variable"].fillna("") == ""]
    if sub.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    sub = sub.copy()
    sub["model_key"] = sub["variant"].astype(str).map(_model_key)
    for _, curve_df in sub.groupby(["family", "model_key"], sort=True):
        curve_df = curve_df.sort_values("res")
        if curve_df["value"].notna().sum() == 0:
            continue
        ax.plot(curve_df["res"], curve_df["value"], label=_curve_label(curve_df), **_curve_style(curve_df))
        plotted = True

    baseline_rows = _matching_baseline_rows(
        baseline_df,
        lead_step=lead_step,
        metric_kind=metric_kind,
        eval_mode=baseline_eval_mode,
        variable=variable,
        baseline_res=baseline_res,
        baseline_variant=baseline_variant,
    )
    if not baseline_rows.empty:
        for _, row in baseline_rows.dropna(subset=["value"]).groupby("variant", sort=True).first().iterrows():
            ax.axhline(
                float(row["value"]),
                color="#1f77b4",
                linestyle=":",
                linewidth=2.2,
                label=baseline_label,
            )
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
    lead_steps: list[int],
    image_dir: Path,
    plot_prefix: str,
    include_per_variable: bool = False,
    default_per_variable_plots: list[str] | None = None,
    baseline_df: pd.DataFrame | None = None,
    baseline_label: str = "Baseline",
    baseline_eval_mode: str = "cold",
    baseline_res: int = 1,
    baseline_variant: str | None = None,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    default_per_variable_plots = default_per_variable_plots or []
    for lead_step in lead_steps:
        lead_label = _lead_label(lead_step)
        for eval_mode in sorted(df["eval_mode"].dropna().astype(str).unique().tolist()):
            weighted_png = image_dir / f"{plot_prefix}_weighted_allvars_vs_res_lead{lead_label}_{eval_mode}.png"
            plot_vs_res(
                df,
                lead_step=lead_step,
                metric_kind="weighted_allvars",
                eval_mode=eval_mode,
                variable=None,
                out_path=weighted_png,
                title=f"Normalized weighted all-variable MSE vs res | lead={lead_label} | {eval_mode}",
                ylabel="Normalized weighted MSE",
                baseline_df=baseline_df,
                baseline_label=baseline_label,
                baseline_eval_mode=baseline_eval_mode,
                baseline_res=baseline_res,
                baseline_variant=baseline_variant,
            )
            available_variables = sorted(
                df[
                    (df["lead_steps"].astype(int) == int(lead_step))
                    & (df["metric_kind"] == "per_variable")
                    & (df["eval_mode"] == eval_mode)
                ]["variable"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if include_per_variable:
                variables = available_variables
            else:
                selected_variables = set(default_per_variable_plots)
                variables = [variable for variable in available_variables if variable in selected_variables]
            for variable in variables:
                variable_png = image_dir / (
                    f"{plot_prefix}_per_variable_{_sanitize_filename_part(variable)}_vs_res_lead{lead_label}_{eval_mode}.png"
                )
                plot_vs_res(
                    df,
                    lead_step=lead_step,
                    metric_kind="per_variable",
                    eval_mode=eval_mode,
                    variable=variable,
                    out_path=variable_png,
                    title=f"{variable} normalized MSE vs res | lead={lead_label} | {eval_mode}",
                    ylabel=f"{variable} normalized MSE",
                    baseline_df=baseline_df,
                    baseline_label=baseline_label,
                    baseline_eval_mode=baseline_eval_mode,
                    baseline_res=baseline_res,
                    baseline_variant=baseline_variant,
                )


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.input_data_dir.glob(args.shard_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No shard CSVs matching {args.shard_glob} found under {args.input_data_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    df = _ensure_lead_steps(pd.concat(frames, ignore_index=True))
    baseline_df = None
    if args.baseline_csv is not None:
        baseline_csv = args.baseline_csv if args.baseline_csv.is_absolute() else ROOT / args.baseline_csv
        if not baseline_csv.exists():
            raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")
        baseline_df = _ensure_lead_steps(pd.read_csv(baseline_csv))
    if args.resolutions is not None:
        df = df[df["res"].astype(int).isin(set(args.resolutions))]
        if df.empty:
            raise ValueError("No rows left after applying --resolutions filter.")
    df = df.sort_values(
        ["family", "variant", "res", "lead_steps", "lead_days", "eval_mode", "metric_kind", "variable"]
    ).reset_index(drop=True)

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
        lead_steps=sorted({int(step) for step in args.lead_steps}) if args.lead_steps is not None else _lead_steps_from_days(args.lead_days),
        image_dir=args.output_image_dir,
        plot_prefix=args.plot_prefix,
        include_per_variable=args.include_per_variable,
        default_per_variable_plots=args.default_per_variable_plots,
        baseline_df=baseline_df,
        baseline_label=args.baseline_label,
        baseline_eval_mode=args.baseline_eval_mode,
        baseline_res=args.baseline_res,
        baseline_variant=args.baseline_variant,
    )


if __name__ == "__main__":
    main()
