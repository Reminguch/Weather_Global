#!/usr/bin/env python3
"""Plot res2 lead curves for vanilla GC, GC-Mamba, and residual Mamba."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "plots/analyze_models/data/resolution_eval/res2_small_mamba_lead_curves/shards"
DEFAULT_OUTPUT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval/res2_small_mamba_lead_curves"
DEFAULT_OUTPUT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval/res2_small_mamba_lead_curves"
LEAD_STEPS = [1, 4, 8, 12, 16]
LEAD_LABELS = {1: "6h", 4: "1d", 8: "2d", 12: "3d", 16: "4d"}
VARIANTS = {
    "Vanilla GC": "vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k",
    "GC-Mamba": (
        "vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_"
        "gc_mamba_tc2_di256_ds128_frozen50k_release20k"
    ),
    "Residual Mamba": (
        "vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_"
        "residual_mamba_tc2_di256_ds128_frozen50k_release20k"
    ),
}
STYLES = {
    "Vanilla GC": {"color": "#2f2f2f", "marker": "o", "linestyle": "-"},
    "GC-Mamba": {"color": "#d62728", "marker": "^", "linestyle": "--"},
    "Residual Mamba": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--shard-glob", default="resolution_eval_*_res2_lead_curve.csv")
    return parser.parse_args()


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


def _load_rows(input_dir: Path, shard_glob: str) -> pd.DataFrame:
    paths = sorted(input_dir.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"No shard CSVs matching {shard_glob} under {input_dir}")
    df = _ensure_lead_steps(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    df = df[
        (df["res"].astype(int) == 2)
        & (df["eval_mode"].astype(str) == "warm")
        & (df["lead_steps"].astype(int).isin(LEAD_STEPS))
        & (df["variant"].astype(str).isin(VARIANTS.values()))
    ].copy()
    if df.empty:
        raise ValueError("No matching res2 warm rows found for the requested variants.")
    label_by_variant = {variant: label for label, variant in VARIANTS.items()}
    df["curve_label"] = df["variant"].map(label_by_variant)
    return df.sort_values(["curve_label", "lead_steps", "metric_kind", "variable"]).reset_index(drop=True)


def _metric_rows(df: pd.DataFrame, *, metric_kind: str, variable: str | None) -> pd.DataFrame:
    rows = df[df["metric_kind"].astype(str) == metric_kind].copy()
    if variable is None:
        rows = rows[rows["variable"].fillna("").astype(str) == ""]
    else:
        rows = rows[rows["variable"].astype(str) == variable]
    missing = []
    for label in VARIANTS:
        got = set(rows[rows["curve_label"] == label]["lead_steps"].astype(int))
        want = set(LEAD_STEPS)
        if got != want:
            missing.append(f"{label}: missing {sorted(want - got)}")
    if missing:
        raise ValueError("; ".join(missing))
    return rows


def _plot(rows: pd.DataFrame, *, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label in VARIANTS:
        sub = rows[rows["curve_label"] == label].sort_values("lead_steps")
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["value"].astype(float),
            label=label,
            linewidth=2.0,
            markersize=6,
            **STYLES[label],
        )
    ax.set_xticks(LEAD_STEPS)
    ax.set_xticklabels([LEAD_LABELS[step] for step in LEAD_STEPS])
    ax.set_xlabel("Lead time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    df = _load_rows(args.input_dir, args.shard_glob)
    merged_csv = args.output_data_dir / "resolution_eval.csv"
    df.to_csv(merged_csv, index=False)
    print(f"Saved merged CSV: {merged_csv}")

    weighted = _metric_rows(df, metric_kind="weighted_allvars", variable=None)
    weighted.to_csv(args.output_data_dir / "plotted_rows_weighted_allvars.csv", index=False)
    _plot(
        weighted,
        title="Res2 warm lead curve | weighted all variables",
        ylabel="Normalized weighted MSE",
        out_path=args.output_image_dir / "res2_small_mamba_vs_residual_vs_vanilla_weighted_allvars_by_lead.png",
    )

    temp2m = _metric_rows(df, metric_kind="per_variable", variable="2m_temperature")
    temp2m.to_csv(args.output_data_dir / "plotted_rows_2m_temperature.csv", index=False)
    _plot(
        temp2m,
        title="Res2 warm lead curve | 2m temperature",
        ylabel="2m temperature normalized MSE",
        out_path=args.output_image_dir / "res2_small_mamba_vs_residual_vs_vanilla_2m_temperature_by_lead.png",
    )


if __name__ == "__main__":
    main()
