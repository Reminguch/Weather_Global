#!/usr/bin/env python3
"""Compare frozen and released GC-Mamba warm-evaluation curves at res2.

The input CSV contains paired checkpoints with the same Mamba d_inner and
target-step count: the earlier frozen-Mamba run and the later
``_release_all20k`` continuation.  This script renders one three-panel plot
per metric (target steps 4, 8, and 12), with colour encoding d_inner and line
style encoding frozen versus released.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / (
    "plots/analyze_models/data/resolution_eval/"
    "res2_ds16_gc_mamba_release_all20k_warm_leads1_9d/resolution_eval.csv"
)
DEFAULT_DATA_DIR = ROOT / (
    "plots/analyze_models/data/resolution_eval/"
    "res2_ds16_gc_mamba_release_all20k_warm_leads1_9d/release_vs_frozen"
)
DEFAULT_IMAGE_DIR = ROOT / (
    "plots/analyze_models/images/resolution_eval/"
    "res2_ds16_gc_mamba_release_all20k_warm_leads1_9d/release_vs_frozen"
)

PAIR_RE = re.compile(
    r"_gc_mamba_tc2_di(?P<d_inner>16|64|256)_ds16_20k_target_step"
    r"(?P<target_steps>4|8|12)_bptt16(?P<released>_release_all20k)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    return parser.parse_args()


def paired_rows(data: pd.DataFrame, *, metric_kind: str, variable: str = "") -> pd.DataFrame:
    rows = data[
        (data["family"] == "gc_mamba")
        & (data["eval_mode"] == "warm")
        & (data["metric_kind"] == metric_kind)
        & (data["variable"].fillna("") == variable)
    ].copy()
    parsed = rows["variant"].astype(str).str.extract(PAIR_RE)
    rows = rows[parsed["d_inner"].notna()].copy()
    rows["d_inner"] = parsed.loc[rows.index, "d_inner"].astype(int)
    rows["target_steps"] = parsed.loc[rows.index, "target_steps"].astype(int)
    rows["training"] = parsed.loc[rows.index, "released"].notna().map(
        {False: "Frozen Mamba", True: "Released all 20k"}
    )
    return rows.sort_values(["target_steps", "d_inner", "training", "lead_days"])


def validate_pairs(rows: pd.DataFrame) -> None:
    expected = {(target, d_inner, training) for target in (4, 8, 12) for d_inner in (16, 64, 256)
                for training in ("Frozen Mamba", "Released all 20k")}
    observed = set(rows[["target_steps", "d_inner", "training"]].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Expected 18 matched curves; missing={missing}, extra={extra}")


def plot_metric(rows: pd.DataFrame, *, metric_label: str, ylabel: str, output: Path) -> None:
    colors = {16: "#0072B2", 64: "#E69F00", 256: "#009E73"}
    styles = {"Frozen Mamba": "-", "Released all 20k": "--"}
    markers = {"Frozen Mamba": "o", "Released all 20k": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, target_steps in zip(axes, (4, 8, 12)):
        panel = rows[rows["target_steps"] == target_steps]
        for d_inner in (16, 64, 256):
            for training in ("Frozen Mamba", "Released all 20k"):
                curve = panel[(panel["d_inner"] == d_inner) & (panel["training"] == training)]
                ax.plot(
                    curve["lead_days"], curve["value"],
                    color=colors[d_inner], linestyle=styles[training], marker=markers[training],
                    linewidth=2, markersize=4.5,
                    label=f"di={d_inner}, {training}",
                )
        ax.set_title(f"Target steps = {target_steps}")
        ax.set_xlabel("Lead time (days)")
        ax.set_xticks(range(1, 10))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"GC-Mamba res2 warm evaluation: frozen vs released continuation — {metric_label}", y=1.15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {output}")


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input_csv)
    allvars = paired_rows(data, metric_kind="weighted_allvars")
    temperature = paired_rows(data, metric_kind="rmse_k", variable="2m_temperature")
    validate_pairs(allvars)
    validate_pairs(temperature)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    allvars.to_csv(args.data_dir / "paired_weighted_allvars.csv", index=False)
    temperature.to_csv(args.data_dir / "paired_2m_temperature_rmse_k.csv", index=False)
    plot_metric(
        allvars,
        metric_label="weighted all-variable error",
        ylabel="Normalized weighted MSE (lower is better)",
        output=args.image_dir / "gc_mamba_release_vs_frozen_warm_weighted_allvars.png",
    )
    plot_metric(
        temperature,
        metric_label="2 m temperature",
        ylabel="2 m temperature RMSE (K; lower is better)",
        output=args.image_dir / "gc_mamba_release_vs_frozen_warm_2m_temperature_rmse_k.png",
    )


if __name__ == "__main__":
    main()
