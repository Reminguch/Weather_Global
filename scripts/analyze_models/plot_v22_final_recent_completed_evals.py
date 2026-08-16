#!/usr/bin/env python3
"""Plot the recent v22 cold-rollout SWA and 20k evaluation batch.

The SWA plots make only matched comparisons: all JSONs use the same 32 cold
anchors, 40 six-hour leads, and vanilla GC500k baseline.  The separate 20k
figure is descriptive because its three checkpoints differ in more than one
training/architecture setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/checkpoints/v22_final")
SWA_ROOT = ROOT / "res2_gc500k_k12_20k_di_crosscheck"
OUTPUT_DIR = Path(
    "plots/analyze_models/images/resolution_eval/"
    "v22_final_recent_completed_evals_20260814"
)

WIDTHS = {
    "di16": "di16_closed_sg_stateful_20k",
    "di32": "di32_closed_sg_stateful_20k",
    "di64 BCG4": "di64_bcg4_closed_sg_stateful_20k",
}
WINDOWS = {
    "SWA 4k–12k": "swa_step04000-12000.json",
    "SWA 6k–14k": "swa_step06000-14000.json",
    "SWA 8k–16k": "swa_step08000-16000.json",
}
COLOURS = {"di16": "#31688E", "di32": "#35B779", "di64 BCG4": "#E76F51"}
WINDOW_COLOURS = {name: colour for name, colour in zip(WINDOWS, ("#440154", "#21918C", "#FDE725"), strict=True)}

OTHER_20K = {
    "di16 carry-state, 20k": ROOT / "res2_gc500k_k12_full_mamba_state_ablation_20k/di16_closed_sg_carry_20k/eval/cold_full_zero/step20000.json",
    "di16 reset-every-anchor, 20k": ROOT / "res2_gc500k_k12_full_mamba_state_ablation_20k/di16_closed_sg_reset_every_anchor_20k/eval/cold_full_zero/step20000.json",
    "di64 BCG4 MP4, 20k": SWA_ROOT / "di64_bcg4_mp4_closed_sg_stateful_20k/eval/cold_full_zero/step20000.json",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"chosen_idx", "target_steps", "per_variable_per_step"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return value


def rel_allvar_rmse_improvement(value: dict) -> np.ndarray:
    curves = []
    for metrics in value["per_variable_per_step"].values():
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        full = np.asarray(metrics["rmse_full"], dtype=float)
        curves.append(100.0 * (1.0 - full / baseline))
    return np.mean(np.stack(curves), axis=0)


def two_m_rmse(value: dict, field: str) -> np.ndarray:
    return np.asarray(value["per_variable_per_step"]["2m_temperature"][field], dtype=float)


def validate_matched(values: list[dict], description: str) -> tuple[np.ndarray, np.ndarray]:
    anchors = {tuple(value["chosen_idx"]) for value in values}
    horizons = {int(value["target_steps"]) for value in values}
    if len(anchors) != 1 or len(horizons) != 1:
        raise ValueError(f"{description}: evaluations do not share anchors/horizon")
    baselines = [two_m_rmse(value, "rmse_baseline") for value in values]
    # Independently run JAX baseline evaluations differ slightly numerically.
    # Normalized panels retain each evaluation baseline; absolute RMSE panels use the mean.
    max_baseline_delta = max(float(np.max(np.abs(baselines[0] - candidate))) for candidate in baselines[1:])
    print(f"{description}: maximum baseline variation = {max_baseline_delta:.6g} K")
    lead_days = np.arange(1, horizons.pop() + 1) * 0.25
    return lead_days, np.mean(np.stack(baselines), axis=0)


def decorate(axis: plt.Axes, lead_days: np.ndarray) -> None:
    axis.set_xlim(0.25, lead_days[-1])
    axis.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1))
    axis.grid(alpha=0.28)


def plot_swa_by_window(swa: dict[str, dict[str, dict]], lead_days: np.ndarray, baseline_2m: np.ndarray, metric: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharex=True, sharey=metric == "allvar")
    for axis, (window, _) in zip(axes, WINDOWS.items(), strict=True):
        if metric == "allvar":
            axis.axhline(0.0, color="0.15", lw=1.2, label="Vanilla GC500k")
            for width in WIDTHS:
                axis.plot(lead_days, rel_allvar_rmse_improvement(swa[width][window]), lw=2.2, color=COLOURS[width], label=width)
            axis.set_ylabel("equal-variable RMSE reduction (%)")
        else:
            axis.plot(lead_days, baseline_2m, color="0.15", lw=1.5, label="Vanilla GC500k")
            for width in WIDTHS:
                axis.plot(lead_days, two_m_rmse(swa[width][window], "rmse_full"), lw=2.2, color=COLOURS[width], label=width)
            axis.set_ylabel("2 m-temperature RMSE (K)")
        axis.set_title(window)
        axis.set_xlabel("forecast lead time (days)")
        decorate(axis, lead_days)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.96), frameon=False)
    title = "Matched SWA width comparison: all variables" if metric == "allvar" else "Matched SWA width comparison: 2 m temperature"
    figure.suptitle(title, y=1.03)
    figure.text(0.5, -0.035, "Closed-SG stateful residual-Mamba; 32 matched cold anchors; 40 six-hour leads; zero initial temporal state.", ha="center", fontsize=8.5, color="0.25")
    figure.tight_layout()
    suffix = "allvars_relative_rmse_improvement" if metric == "allvar" else "2m_temperature_rmse"
    figure.savefig(OUTPUT_DIR / f"swa_width_comparison_by_window_{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_swa_by_width(swa: dict[str, dict[str, dict]], lead_days: np.ndarray, baseline_2m: np.ndarray, metric: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharex=True, sharey=metric == "allvar")
    for axis, width in zip(axes, WIDTHS, strict=True):
        if metric == "allvar":
            axis.axhline(0.0, color="0.15", lw=1.2, label="Vanilla GC500k")
            for window in WINDOWS:
                axis.plot(lead_days, rel_allvar_rmse_improvement(swa[width][window]), lw=2.2, color=WINDOW_COLOURS[window], label=window)
            axis.set_ylabel("equal-variable RMSE reduction (%)")
        else:
            axis.plot(lead_days, baseline_2m, color="0.15", lw=1.5, label="Vanilla GC500k")
            for window in WINDOWS:
                axis.plot(lead_days, two_m_rmse(swa[width][window], "rmse_full"), lw=2.2, color=WINDOW_COLOURS[window], label=window)
            axis.set_ylabel("2 m-temperature RMSE (K)")
        axis.set_title(width)
        axis.set_xlabel("forecast lead time (days)")
        decorate(axis, lead_days)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.96), frameon=False)
    title = "Matched SWA-window comparison: all variables" if metric == "allvar" else "Matched SWA-window comparison: 2 m temperature"
    figure.suptitle(title, y=1.03)
    figure.text(0.5, -0.035, "Closed-SG stateful residual-Mamba; 32 matched cold anchors; 40 six-hour leads; zero initial temporal state.", ha="center", fontsize=8.5, color="0.25")
    figure.tight_layout()
    suffix = "allvars_relative_rmse_improvement" if metric == "allvar" else "2m_temperature_rmse"
    figure.savefig(OUTPUT_DIR / f"swa_window_comparison_by_width_{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_other_20k(values: dict[str, dict], lead_days: np.ndarray, baseline_2m: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    colours = ("#31688E", "#35B779", "#E76F51")
    axes[0].axhline(0.0, color="0.15", lw=1.2, label="Vanilla GC500k")
    axes[1].plot(lead_days, baseline_2m, color="0.15", lw=1.5, label="Vanilla GC500k")
    for colour, (label, value) in zip(colours, values.items(), strict=True):
        axes[0].plot(lead_days, rel_allvar_rmse_improvement(value), lw=2.2, color=colour, label=label)
        axes[1].plot(lead_days, two_m_rmse(value, "rmse_full"), lw=2.2, color=colour, label=label)
    axes[0].set(ylabel="equal-variable RMSE reduction (%)", title="All variables")
    axes[1].set(ylabel="2 m-temperature RMSE (K)", title="2 m temperature")
    for axis in axes:
        axis.set_xlabel("forecast lead time (days)")
        decorate(axis, lead_days)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    figure.suptitle("Completed 20k checkpoints: descriptive curves (not a matched ablation)", y=1.04)
    figure.text(0.5, -0.035, "All use the same 32 cold anchors and GC500k baseline, but differ in state policy and/or architecture; do not attribute differences to one factor.", ha="center", fontsize=8.3, color="0.25")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "heterogeneous_20k_completed_evaluations.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    swa = {
        width: {
            window: load(SWA_ROOT / run / "eval/cold_full_zero" / filename)
            for window, filename in WINDOWS.items()
        }
        for width, run in WIDTHS.items()
    }
    flat_swa = [value for by_window in swa.values() for value in by_window.values()]
    lead_days, baseline_2m = validate_matched(flat_swa, "SWA batch")
    plot_swa_by_window(swa, lead_days, baseline_2m, "allvar")
    plot_swa_by_window(swa, lead_days, baseline_2m, "2m")
    plot_swa_by_width(swa, lead_days, baseline_2m, "allvar")
    plot_swa_by_width(swa, lead_days, baseline_2m, "2m")

    other = {label: load(path) for label, path in OTHER_20K.items()}
    other_leads, other_baseline = validate_matched(list(other.values()), "20k descriptive batch")
    if not np.array_equal(lead_days, other_leads):
        raise ValueError("SWA and 20k evaluation lead times differ")
    plot_other_20k(other, other_leads, other_baseline)
    print(f"Saved five plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
