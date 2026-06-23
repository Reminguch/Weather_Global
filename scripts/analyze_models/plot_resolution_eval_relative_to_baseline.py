#!/usr/bin/env python3
"""Plot resolution-eval candidate improvement relative to a baseline CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEAD_STEPS = [4, 8, 16, 24, 32, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="relative_to_baseline")
    parser.add_argument("--candidate-eval-mode", default="warm")
    parser.add_argument("--baseline-eval-mode", default="cold")
    parser.add_argument("--candidate-family", default=None)
    parser.add_argument("--metric-kind", default="rmse_k")
    parser.add_argument("--variable", default="2m_temperature")
    parser.add_argument("--lead-steps", type=int, nargs="+", default=DEFAULT_LEAD_STEPS)
    parser.add_argument("--baseline-label", default="GC_small")
    parser.add_argument("--title", default=None)
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
    derived_steps = (lead_days * 24.0 / 6.0).round()
    if "lead_steps" not in df.columns:
        df["lead_steps"] = derived_steps
    else:
        df["lead_steps"] = pd.to_numeric(df["lead_steps"], errors="coerce").fillna(derived_steps)
    df["lead_steps"] = df["lead_steps"].astype(int)
    df["lead_days"] = lead_days
    df["variable"] = df["variable"].fillna("").astype(str)
    return df


def _filter_metric_rows(
    df: pd.DataFrame,
    *,
    eval_mode: str,
    metric_kind: str,
    variable: str,
    lead_steps: list[int],
) -> pd.DataFrame:
    rows = df[
        df["lead_steps"].astype(int).isin(lead_steps)
        & df["eval_mode"].astype(str).eq(eval_mode)
        & df["metric_kind"].astype(str).eq(metric_kind)
        & df["variable"].eq(variable)
    ].copy()
    if rows.empty:
        raise ValueError(
            f"No rows for eval_mode={eval_mode!r}, metric_kind={metric_kind!r}, "
            f"variable={variable or '<blank>'!r}."
        )
    return rows


def _curve_label(row: pd.Series) -> str:
    plot_label = str(row.get("plot_label", "") or "")
    if plot_label:
        return plot_label
    family_label = {
        "residual_mamba": "Residual Mamba",
        "gc_mamba": "GC-Mamba",
        "graphcast": "GraphCast",
    }.get(str(row.get("family", "")), str(row.get("family", "")) or "Candidate")
    variant = str(row.get("variant", ""))
    match = re.search(r"_k(\d+)(?=_|$)", variant)
    if match:
        return f"{family_label} k{match.group(1)}"
    return f"{family_label}: {variant}" if variant else family_label


def _style(curve: str) -> dict[str, object]:
    match = re.search(r"k(\d+)", curve)
    k_value = int(match.group(1)) if match else None
    colors = {
        1: "#2f6f9f",
        4: "#b14b2d",
        8: "#2f7d4f",
        12: "#7b4fa3",
        16: "#8a6d2f",
        24: "#4f6fa8",
    }
    markers = {
        1: "o",
        4: "s",
        8: "^",
        12: "D",
        16: "P",
        24: "X",
    }
    return {
        "color": colors.get(k_value),
        "marker": markers.get(k_value, "o"),
        "linewidth": 2.0,
        "markersize": 5.5,
    }


def _collect_relative_rows(args: argparse.Namespace) -> pd.DataFrame:
    lead_steps = [int(step) for step in args.lead_steps]
    candidate_df = _ensure_lead_steps(pd.read_csv(_resolve(args.candidate_csv)))
    baseline_df = _ensure_lead_steps(pd.read_csv(_resolve(args.baseline_csv)))

    candidate = _filter_metric_rows(
        candidate_df,
        eval_mode=args.candidate_eval_mode,
        metric_kind=args.metric_kind,
        variable=args.variable,
        lead_steps=lead_steps,
    )
    if args.candidate_family:
        candidate = candidate[candidate["family"].astype(str).eq(args.candidate_family)].copy()
        if candidate.empty:
            raise ValueError(f"No candidate rows with family={args.candidate_family!r}.")
    baseline = _filter_metric_rows(
        baseline_df,
        eval_mode=args.baseline_eval_mode,
        metric_kind=args.metric_kind,
        variable=args.variable,
        lead_steps=lead_steps,
    )
    baseline = baseline.sort_values(["lead_steps", "variant"]).groupby("lead_steps", as_index=False).first()
    missing = sorted(set(lead_steps) - set(baseline["lead_steps"].astype(int)))
    if missing:
        raise ValueError(f"Baseline is missing lead steps {missing}.")
    if baseline["value"].isna().any() or baseline["value"].astype(float).eq(0).any():
        raise ValueError("Baseline contains NaN or zero values; cannot compute relative errors.")

    candidate["curve"] = candidate.apply(_curve_label, axis=1)
    merged = candidate.merge(
        baseline[["lead_steps", "value", "variant", "ckpt_path"]],
        on="lead_steps",
        suffixes=("_candidate", "_baseline"),
        validate="many_to_one",
    )
    merged["candidate_value"] = merged["value_candidate"].astype(float)
    merged["baseline_value"] = merged["value_baseline"].astype(float)
    merged["ratio_percent"] = merged["candidate_value"] / merged["baseline_value"] * 100.0
    merged["relative_error_percent"] = (merged["candidate_value"] - merged["baseline_value"]) / merged[
        "baseline_value"
    ] * 100.0
    merged["relative_improvement_percent"] = (merged["baseline_value"] - merged["candidate_value"]) / merged[
        "baseline_value"
    ] * 100.0
    return merged.sort_values(["curve", "lead_steps"]).reset_index(drop=True)


def _plot_relative(rows: pd.DataFrame, *, args: argparse.Namespace, out_path: Path) -> None:
    lead_steps = [int(step) for step in args.lead_steps]
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=180)
    for curve, sub in rows.groupby("curve", sort=True):
        sub = sub.sort_values("lead_steps")
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["relative_improvement_percent"].astype(float),
            label=str(curve),
            **_style(str(curve)),
        )
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.3, alpha=0.8, label=args.baseline_label)
    ax.set_xticks(lead_steps)
    ax.set_xticklabels([_lead_label(step) for step in lead_steps])
    ax.set_xlabel("Lead time")
    ax.set_ylabel(f"Relative improvement vs {args.baseline_label} (%)")
    title = args.title or f"{args.metric_kind} {args.variable or '<blank>'}: relative improvement vs {args.baseline_label}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    output_data_dir = _resolve(args.output_data_dir)
    output_image_dir = _resolve(args.output_image_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_relative_rows(args)
    csv_path = output_data_dir / f"{args.output_prefix}_relative_rows.csv"
    rows.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    _plot_relative(rows, args=args, out_path=output_image_dir / f"{args.output_prefix}_relative_improvement_percent.png")


if __name__ == "__main__":
    main()
