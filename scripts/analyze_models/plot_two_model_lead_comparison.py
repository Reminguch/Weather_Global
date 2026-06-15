#!/usr/bin/env python3
"""Plot two-model resolution-eval lead curves and error ratios."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEAD_STEPS = [4, 8, 16, 24, 32, 40]
METRIC_SPECS = [
    {
        "key": "2m_temperature_rmse_k",
        "metric_kind": "rmse_k",
        "variable": "2m_temperature",
        "title": "2m temperature RMSE",
        "ylabel": "RMSE (K)",
    },
    {
        "key": "weighted_allvars",
        "metric_kind": "weighted_allvars",
        "variable": "",
        "title": "Weighted all variables",
        "ylabel": "Normalized weighted MSE",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="two_model_lead_comparison")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--ratio-label", default=None)
    parser.add_argument("--lead-steps", type=int, nargs="+", default=DEFAULT_LEAD_STEPS)
    parser.add_argument("--eval-mode", default="warm")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _lead_label(lead_step: int) -> str:
    hours = int(lead_step) * 6
    if hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


def _ensure_lead_steps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lead_days = pd.to_numeric(df["lead_days"], errors="coerce")
    if "lead_steps" not in df.columns:
        df["lead_steps"] = (lead_days * 24.0 / 6.0).round()
    else:
        lead_steps = pd.to_numeric(df["lead_steps"], errors="coerce")
        df["lead_steps"] = lead_steps.fillna((lead_days * 24.0 / 6.0).round())
    df["lead_steps"] = df["lead_steps"].astype(int)
    df["lead_days"] = lead_days
    df["variable"] = df["variable"].fillna("").astype(str)
    return df


def _load_metric_rows(
    path: Path,
    *,
    label: str,
    lead_steps: list[int],
    eval_mode: str,
    metric_kind: str,
    variable: str,
) -> pd.DataFrame:
    df = _ensure_lead_steps(pd.read_csv(_resolve(path)))
    rows = df[
        df["lead_steps"].astype(int).isin(lead_steps)
        & df["eval_mode"].astype(str).eq(eval_mode)
        & df["metric_kind"].astype(str).eq(metric_kind)
        & df["variable"].eq(variable)
    ].copy()
    if rows.empty:
        raise ValueError(f"No {metric_kind}/{variable or '<blank>'} rows found in {path}")
    rows = rows.sort_values(["lead_steps", "variant"]).groupby("lead_steps", as_index=False).first()
    found = set(rows["lead_steps"].astype(int))
    missing = [step for step in lead_steps if step not in found]
    if missing:
        raise ValueError(
            f"{path} is missing lead steps {missing} for metric={metric_kind} variable={variable or '<blank>'}"
        )
    rows["curve"] = label
    return rows


def _collect_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead_steps = [int(step) for step in args.lead_steps]
    plotted_frames = []
    ratio_frames = []
    ratio_label = args.ratio_label or f"{args.candidate_label} / {args.baseline_label}"

    for spec in METRIC_SPECS:
        candidate = _load_metric_rows(
            args.candidate_csv,
            label=args.candidate_label,
            lead_steps=lead_steps,
            eval_mode=args.eval_mode,
            metric_kind=spec["metric_kind"],
            variable=spec["variable"],
        )
        baseline = _load_metric_rows(
            args.baseline_csv,
            label=args.baseline_label,
            lead_steps=lead_steps,
            eval_mode=args.eval_mode,
            metric_kind=spec["metric_kind"],
            variable=spec["variable"],
        )
        candidate["metric_key"] = spec["key"]
        baseline["metric_key"] = spec["key"]
        plotted_frames.extend([candidate, baseline])

        merged = candidate[["lead_steps", "lead_days", "value"]].merge(
            baseline[["lead_steps", "value"]],
            on="lead_steps",
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        if merged["value_baseline"].isna().any() or (merged["value_baseline"].astype(float) == 0).any():
            raise ValueError(f"Invalid baseline denominator for metric={spec['key']}")
        merged["metric_key"] = spec["key"]
        merged["curve"] = ratio_label
        merged["ratio_percent"] = merged["value_candidate"].astype(float) / merged["value_baseline"].astype(float) * 100.0
        ratio_frames.append(merged)

    plotted_rows = pd.concat(plotted_frames, ignore_index=True)
    ratio_rows = pd.concat(ratio_frames, ignore_index=True)
    return plotted_rows, ratio_rows


def _plot_absolute(rows: pd.DataFrame, *, lead_steps: list[int], out_path: Path, candidate_label: str, baseline_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), dpi=180)
    styles = {
        candidate_label: {"color": "#2ca02c", "marker": "o", "linewidth": 2.1},
        baseline_label: {"color": "#222222", "marker": "D", "linestyle": "--", "linewidth": 2.2},
    }
    for ax, spec in zip(axes, METRIC_SPECS, strict=True):
        sub = rows[rows["metric_key"].eq(spec["key"])]
        for curve, curve_df in sub.groupby("curve", sort=False):
            curve_df = curve_df.sort_values("lead_steps")
            ax.plot(
                curve_df["lead_steps"].astype(int),
                curve_df["value"].astype(float),
                label=str(curve),
                **styles.get(str(curve), {"marker": "o", "linewidth": 2.0}),
            )
        ax.set_title(spec["title"])
        ax.set_xlabel("Lead time")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xticks(lead_steps)
        ax.set_xticklabels([_lead_label(step) for step in lead_steps])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def _plot_ratio(rows: pd.DataFrame, *, lead_steps: list[int], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), dpi=180)
    for ax, spec in zip(axes, METRIC_SPECS, strict=True):
        sub = rows[rows["metric_key"].eq(spec["key"])].sort_values("lead_steps")
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["ratio_percent"].astype(float),
            color="#1f77b4",
            marker="o",
            linewidth=2.1,
            label=str(sub["curve"].iloc[0]),
        )
        ax.axhline(100.0, color="#222222", linestyle="--", linewidth=1.3, alpha=0.8)
        ax.set_title(spec["title"])
        ax.set_xlabel("Lead time")
        ax.set_ylabel("Error ratio (%)")
        ax.set_xticks(lead_steps)
        ax.set_xticklabels([_lead_label(step) for step in lead_steps])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    lead_steps = [int(step) for step in args.lead_steps]
    output_data_dir = _resolve(args.output_data_dir)
    output_image_dir = _resolve(args.output_image_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    plotted_rows, ratio_rows = _collect_rows(args)
    plotted_csv = output_data_dir / f"{args.output_prefix}_plotted_rows.csv"
    ratio_csv = output_data_dir / f"{args.output_prefix}_ratio_rows.csv"
    plotted_rows.to_csv(plotted_csv, index=False)
    ratio_rows.to_csv(ratio_csv, index=False)
    print(f"Saved CSV: {plotted_csv}")
    print(f"Saved CSV: {ratio_csv}")

    _plot_absolute(
        plotted_rows,
        lead_steps=lead_steps,
        out_path=output_image_dir / f"{args.output_prefix}_absolute_errors.png",
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
    )
    _plot_ratio(
        ratio_rows,
        lead_steps=lead_steps,
        out_path=output_image_dir / f"{args.output_prefix}_error_ratio_percent.png",
    )


if __name__ == "__main__":
    main()
