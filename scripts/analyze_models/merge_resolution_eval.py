#!/usr/bin/env python3
"""Merge unified resolution-eval shards and render default lead-curve plots."""

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
DEFAULT_RESOLUTIONS = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 18.0]
DEFAULT_RMSE_VARIABLE = "2m_temperature"
DEFAULT_PER_VARIABLE_PLOTS = ["2m_temperature", "2m_temperature_nyc"]
RES_MESH_TOKEN_RE = re.compile(r"_res\d+_m\d+(?=_)")
DI_DS_TOKEN_RE = re.compile(r"_di(?P<di>\d+)_ds(?P<ds>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge unified resolution-eval shards and plot lead curves.")
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
    parser.add_argument("--resolutions", type=float, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--expected-shards", nargs="*", default=None)
    parser.add_argument("--shard-glob", type=str, default="resolution_eval_*.csv")
    parser.add_argument("--plot-prefix", type=str, default="resolution_eval")
    parser.add_argument(
        "--include-per-variable",
        action="store_true",
        help=(
            "Render one plot per variable. By default 2m_temperature RMSE-K is "
            "preferred, plus normalized weighted/default per-variable plots when present."
        ),
    )
    parser.add_argument(
        "--default-rmse-variable",
        default=DEFAULT_RMSE_VARIABLE,
        help="Physical RMSE variable to render by default when rmse_k rows are present.",
    )
    parser.add_argument(
        "--default-per-variable-plots",
        nargs="*",
        default=DEFAULT_PER_VARIABLE_PLOTS,
        help=(
            "Normalized per-variable metrics to render by default when present. "
            "Use an empty value to render only weighted_allvars unless --include-per-variable is set."
        ),
    )
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--baseline-label", type=str, default="DeepMind GraphCast small res1 cold")
    parser.add_argument("--baseline-eval-mode", type=str, default="cold")
    parser.add_argument("--baseline-res", type=float, default=1)
    parser.add_argument("--baseline-variant", type=str, default=None)
    parser.add_argument(
        "--plot-res-axis",
        action="store_true",
        help="Also render legacy resolution-axis plots. Defaults now use lead time on the x axis.",
    )
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


def _format_resolution_value(resolution: float) -> str:
    value = float(resolution)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def _isin_resolutions(series: pd.Series, resolutions: list[float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=series.index)
    for resolution in resolutions:
        mask = mask | (values.sub(float(resolution)).abs() < 1e-6)
    return mask


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


def _curve_label_with_res(curve_df: pd.DataFrame) -> str:
    label = _curve_label(curve_df)
    res_values = sorted(pd.to_numeric(curve_df["res"], errors="coerce").dropna().unique().tolist())
    if len(res_values) == 1:
        label = f"{label} res{_format_resolution_value(float(res_values[0]))}"
    return label


def _matching_baseline_rows(
    baseline_df: pd.DataFrame | None,
    *,
    lead_step: int,
    metric_kind: str,
    eval_mode: str,
    variable: str | None,
    baseline_res: float,
    baseline_variant: str | None,
) -> pd.DataFrame:
    if baseline_df is None or baseline_df.empty:
        return pd.DataFrame()
    sub = baseline_df[
        (baseline_df["lead_steps"].astype(int) == int(lead_step))
        & (baseline_df["metric_kind"] == metric_kind)
        & (baseline_df["eval_mode"].astype(str) == eval_mode)
        & (pd.to_numeric(baseline_df["res"], errors="coerce").sub(float(baseline_res)).abs() < 1e-6)
    ]
    if baseline_variant:
        sub = sub[sub["variant"].astype(str) == baseline_variant]
    if metric_kind in {"per_variable", "rmse_k"}:
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
    baseline_res: float = 1,
    baseline_variant: str | None = None,
) -> bool:
    sub = df[
        (df["lead_steps"].astype(int) == int(lead_step))
        & (df["metric_kind"] == metric_kind)
        & (df["eval_mode"] == eval_mode)
    ]
    if metric_kind in {"per_variable", "rmse_k"}:
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


def plot_vs_lead(
    df: pd.DataFrame,
    *,
    lead_steps: list[int],
    metric_kind: str,
    eval_mode: str,
    variable: str | None,
    out_path: Path,
    title: str,
    ylabel: str,
    baseline_df: pd.DataFrame | None = None,
    baseline_label: str = "Baseline",
    baseline_eval_mode: str = "cold",
    baseline_res: float = 1,
    baseline_variant: str | None = None,
) -> bool:
    keep_steps = {int(step) for step in lead_steps}
    sub = df[
        (df["lead_steps"].astype(int).isin(keep_steps))
        & (df["metric_kind"] == metric_kind)
        & (df["eval_mode"] == eval_mode)
    ]
    if metric_kind in {"per_variable", "rmse_k"}:
        sub = sub[sub["variable"] == (variable or "")]
    else:
        sub = sub[sub["variable"].fillna("") == ""]
    if sub.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    sub = sub.copy()
    sub["model_key"] = sub["variant"].astype(str).map(_model_key)
    for _, curve_df in sub.groupby(["family", "model_key", "res"], sort=True):
        curve_df = curve_df.sort_values("lead_steps")
        if curve_df["value"].notna().sum() == 0:
            continue
        ax.plot(
            curve_df["lead_steps"].astype(int),
            curve_df["value"].astype(float),
            label=_curve_label_with_res(curve_df),
            **_curve_style(curve_df),
        )
        plotted = True

    if baseline_df is not None and not baseline_df.empty:
        baseline_rows = baseline_df[
            (baseline_df["lead_steps"].astype(int).isin(keep_steps))
            & (baseline_df["metric_kind"] == metric_kind)
            & (baseline_df["eval_mode"].astype(str) == baseline_eval_mode)
            & (pd.to_numeric(baseline_df["res"], errors="coerce").sub(float(baseline_res)).abs() < 1e-6)
        ].copy()
        if baseline_variant:
            baseline_rows = baseline_rows[baseline_rows["variant"].astype(str) == baseline_variant]
        if metric_kind in {"per_variable", "rmse_k"}:
            baseline_rows = baseline_rows[baseline_rows["variable"] == (variable or "")]
        else:
            baseline_rows = baseline_rows[baseline_rows["variable"].fillna("") == ""]
        baseline_rows["model_key"] = baseline_rows["variant"].astype(str).map(_model_key)
        for _, curve_df in baseline_rows.groupby(["variant", "res"], sort=True):
            curve_df = curve_df.sort_values("lead_steps").dropna(subset=["value"])
            if curve_df.empty:
                continue
            label = baseline_label
            if baseline_variant is None:
                label = f"{baseline_label}: {_model_key(str(curve_df['variant'].iloc[0]))}"
            ax.plot(
                curve_df["lead_steps"].astype(int),
                curve_df["value"].astype(float),
                color="#1f77b4",
                linestyle=":",
                linewidth=2.2,
                marker="D",
                markersize=4,
                label=label,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xticks(lead_steps)
    ax.set_xticklabels([_lead_label(step) for step in lead_steps])
    ax.set_xlabel("Lead time")
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
    lead_steps: list[int] | None = None,
    lead_days: list[int] | None = None,
    image_dir: Path,
    plot_prefix: str,
    include_per_variable: bool = False,
    default_rmse_variable: str = DEFAULT_RMSE_VARIABLE,
    default_per_variable_plots: list[str] | None = None,
    baseline_df: pd.DataFrame | None = None,
    baseline_label: str = "Baseline",
    baseline_eval_mode: str = "cold",
    baseline_res: float = 1,
    baseline_variant: str | None = None,
    plot_res_axis: bool = False,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    default_per_variable_plots = default_per_variable_plots or []
    if lead_steps is None:
        lead_steps = _lead_steps_from_days(lead_days or [1, 2, 4])
    lead_steps = sorted({int(step) for step in lead_steps})
    for eval_mode in sorted(df["eval_mode"].dropna().astype(str).unique().tolist()):
        rmse_variables = sorted(
            df[
                (df["lead_steps"].astype(int).isin(lead_steps))
                & (df["metric_kind"] == "rmse_k")
                & (df["eval_mode"] == eval_mode)
            ]["variable"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for variable in [v for v in rmse_variables if v == default_rmse_variable]:
            rmse_png = image_dir / (
                f"{plot_prefix}_rmse_k_{_sanitize_filename_part(variable)}_vs_lead_{eval_mode}.png"
            )
            plot_vs_lead(
                df,
                lead_steps=lead_steps,
                metric_kind="rmse_k",
                eval_mode=eval_mode,
                variable=variable,
                out_path=rmse_png,
                title=f"{variable} physical RMSE vs lead time | {eval_mode}",
                ylabel=f"{variable} RMSE (K)" if variable in {"2m_temperature", "temperature"} else f"{variable} RMSE",
                baseline_df=baseline_df,
                baseline_label=baseline_label,
                baseline_eval_mode=baseline_eval_mode,
                baseline_res=baseline_res,
                baseline_variant=baseline_variant,
            )
        weighted_png = image_dir / f"{plot_prefix}_weighted_allvars_vs_lead_{eval_mode}.png"
        plot_vs_lead(
            df,
            lead_steps=lead_steps,
            metric_kind="weighted_allvars",
            eval_mode=eval_mode,
            variable=None,
            out_path=weighted_png,
            title=f"Normalized weighted all-variable MSE vs lead time | {eval_mode}",
            ylabel="Normalized weighted MSE",
            baseline_df=baseline_df,
            baseline_label=baseline_label,
            baseline_eval_mode=baseline_eval_mode,
            baseline_res=baseline_res,
            baseline_variant=baseline_variant,
        )
        available_variables = sorted(
            df[
                (df["lead_steps"].astype(int).isin(lead_steps))
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
                f"{plot_prefix}_per_variable_{_sanitize_filename_part(variable)}_vs_lead_{eval_mode}.png"
            )
            plot_vs_lead(
                df,
                lead_steps=lead_steps,
                metric_kind="per_variable",
                eval_mode=eval_mode,
                variable=variable,
                out_path=variable_png,
                title=f"{variable} normalized MSE vs lead time | {eval_mode}",
                ylabel=f"{variable} normalized MSE",
                baseline_df=baseline_df,
                baseline_label=baseline_label,
                baseline_eval_mode=baseline_eval_mode,
                baseline_res=baseline_res,
                baseline_variant=baseline_variant,
            )
    if not plot_res_axis:
        return
    for lead_step in lead_steps:
        lead_label = _lead_label(lead_step)
        for eval_mode in sorted(df["eval_mode"].dropna().astype(str).unique().tolist()):
            rmse_variables = sorted(
                df[
                    (df["lead_steps"].astype(int) == int(lead_step))
                    & (df["metric_kind"] == "rmse_k")
                    & (df["eval_mode"] == eval_mode)
                ]["variable"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            for variable in [v for v in rmse_variables if v == default_rmse_variable]:
                rmse_png = image_dir / (
                    f"{plot_prefix}_rmse_k_{_sanitize_filename_part(variable)}_vs_res_lead{lead_label}_{eval_mode}.png"
                )
                plot_vs_res(
                    df,
                    lead_step=lead_step,
                    metric_kind="rmse_k",
                    eval_mode=eval_mode,
                    variable=variable,
                    out_path=rmse_png,
                    title=f"{variable} physical RMSE vs res | lead={lead_label} | {eval_mode}",
                    ylabel=f"{variable} RMSE (K)" if variable in {"2m_temperature", "temperature"} else f"{variable} RMSE",
                    baseline_df=baseline_df,
                    baseline_label=baseline_label,
                    baseline_eval_mode=baseline_eval_mode,
                    baseline_res=baseline_res,
                    baseline_variant=baseline_variant,
                )
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
        df = df[_isin_resolutions(df["res"], args.resolutions)]
        if df.empty:
            raise ValueError("No rows left after applying --resolutions filter.")
    df = df.sort_values(
        ["family", "variant", "res", "lead_steps", "lead_days", "eval_mode", "metric_kind", "variable"]
    ).reset_index(drop=True)

    found_shards = sorted(
        {
            f"{row.family}:{_format_resolution_value(float(row.res))}"
            for row in df[["family", "res"]].drop_duplicates().itertuples(index=False)
        }
    )
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
        default_rmse_variable=args.default_rmse_variable,
        default_per_variable_plots=args.default_per_variable_plots,
        baseline_df=baseline_df,
        baseline_label=args.baseline_label,
        baseline_eval_mode=args.baseline_eval_mode,
        baseline_res=args.baseline_res,
        baseline_variant=args.baseline_variant,
        plot_res_axis=args.plot_res_axis,
    )


if __name__ == "__main__":
    main()
