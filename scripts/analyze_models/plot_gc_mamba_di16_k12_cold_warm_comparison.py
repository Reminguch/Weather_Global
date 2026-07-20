#!/usr/bin/env python3
"""Compare cold, old-warm, and recent-warm evaluations of frozen GC-Mamba.

All three curves evaluate the same frozen di16/target-step-12 checkpoint.
The recent-warm source is the completed warm-recheck shard, which has not yet
been merged into a top-level ``resolution_eval.csv``.  The sources share the
same res2 endpoint lead grid (1--9 days).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OLD_VARIANT = (
    "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_"
    "gc_mamba_tc2_di16_ds16_20k_target_step12_bptt16"
)
SOURCES = (
    (
        "Frozen checkpoint, cold",
        ROOT / "plots/analyze_models/data/resolution_eval/"
        "res2_di16_ds16_ts12_frozen_cold_leads1_9d/resolution_eval.csv",
        OLD_VARIANT,
        "cold",
        "#0072B2",
        "o-",
    ),
    (
        "Frozen checkpoint, warm (old)",
        ROOT / "plots/analyze_models/data/resolution_eval/"
        "res2_ds16_gc_mamba_target_steps_bptt16_warm_leads1_9d/resolution_eval.csv",
        OLD_VARIANT,
        "warm",
        "#E69F00",
        "s-",
    ),
    (
        "Frozen checkpoint, warm (recent recheck)",
        ROOT / "plots/analyze_models/data/resolution_eval/"
        "res2_di16_ds16_ts12_frozen_warm_recheck_leads1_9d/shards/"
        "resolution_eval_gc_mamba_res2_warm_recheck.csv",
        OLD_VARIANT,
        "warm",
        "#009E73",
        "^-",
    ),
)
DEFAULT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval/gc_mamba_di16_k12_cold_warm_comparison"
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval/gc_mamba_di16_k12_cold_warm_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    return parser.parse_args()


def load_curve(
    csv_path: Path, variant: str, eval_mode: str, metric_kind: str, variable: str = ""
) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    rows = data[
        (data["variant"] == variant)
        & (data["eval_mode"] == eval_mode)
        & (data["metric_kind"] == metric_kind)
        & (data["variable"].fillna("") == variable)
    ].copy()
    if rows.empty:
        raise ValueError(f"No rows for {variant!r} in {csv_path}")
    if rows["lead_days"].duplicated().any():
        raise ValueError(f"Duplicate lead-day rows in {csv_path}")
    return rows.sort_values("lead_days")


def plot(curves: list[tuple[str, pd.DataFrame, str, str]], output: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    for label, curve, color, style in curves:
        ax.plot(curve["lead_days"], curve["value"], style, color=color, linewidth=2.2, markersize=6, label=label)
    ax.set(
        xlabel="Lead time (days)",
        ylabel=ylabel,
        title=title,
        xlim=(0.8, 9.2),
        xticks=range(1, 10),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {output}")


def main() -> None:
    args = parse_args()
    metric_specs = (
        ("weighted_allvars", "", "weighted_allvars", "Normalized weighted MSE (lower is better)"),
        ("rmse_k", "2m_temperature", "2m_temperature_rmse_k", "2 m temperature RMSE (K; lower is better)"),
    )
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for metric_kind, variable, stem, ylabel in metric_specs:
        curves = []
        records = []
        for label, csv_path, variant, mode, color, style in SOURCES:
            curve = load_curve(csv_path, variant, mode, metric_kind, variable)
            curve["comparison_label"] = label
            curve["source_csv"] = str(csv_path)
            records.append(curve)
            curves.append((label, curve, color, style))
        combined = pd.concat(records, ignore_index=True)
        combined.to_csv(args.data_dir / f"{stem}_curves.csv", index=False)
        plot(
            curves,
            args.image_dir / f"gc_mamba_di16_k12_cold_warm_{stem}.png",
            f"GC-Mamba di16, target steps 12: cold vs warm evaluations — {stem.replace('_', ' ')}",
            ylabel,
        )


if __name__ == "__main__":
    main()
