#!/usr/bin/env python3
"""Compare matched v22_final state-policy SWA evaluations against GC500k."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/checkpoints/v22_final")
CARRY_RUN = ROOT / "res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k"
RESET_RUN = ROOT / (
    "res2_gc500k_k12_full_mamba_state_ablation_20k/"
    "di16_closed_sg_reset_every_anchor_20k"
)
OUTPUT_DIR = Path(
    "plots/analyze_models/images/resolution_eval/"
    "v22_final_res2_gc500k_k12_state_policy_swa"
)
DATA_DIR = Path(
    "plots/analyze_models/data/resolution_eval/"
    "v22_final_res2_gc500k_k12_state_policy_swa"
)
IMAGE_NAME = "state_policy_swa_4k_12k_allvars_and_2m.png"
CSV_NAME = "state_policy_swa_4k_12k.csv"
ANNOTATION_NAME = "state_policy_swa_4k_12k_annotations.md"

SPECS = (
    (
        "Carry-trained SWA; carry eval",
        "carry",
        "carry",
        CARRY_RUN / "eval/cold_full_zero/swa_step04000-12000.json",
        "#31688E",
        "-",
    ),
    (
        "Carry-trained SWA; reset eval",
        "carry",
        "reset",
        CARRY_RUN
        / "eval/cold_full_reset_every_step_zero/swa_step04000-12000.json",
        "#E76F51",
        "--",
    ),
    (
        "Reset-trained SWA; reset eval",
        "reset",
        "reset",
        RESET_RUN
        / "eval/cold_full_reset_every_step_zero/swa_step04000-12000.json",
        "#35B779",
        "-.",
    ),
)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"chosen_idx", "target_steps", "per_variable_per_step"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return value


def allvar_improvement(value: dict) -> np.ndarray:
    curves = []
    for metrics in value["per_variable_per_step"].values():
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        full = np.asarray(metrics["rmse_full"], dtype=float)
        curves.append(100.0 * (1.0 - full / baseline))
    return np.mean(np.stack(curves), axis=0)


def two_m(value: dict, field: str) -> np.ndarray:
    return np.asarray(value["per_variable_per_step"]["2m_temperature"][field])


def summarize(improvement: np.ndarray, rmse_2m: np.ndarray) -> dict[str, float]:
    return {
        "whole_improvement": float(np.mean(improvement)),
        "day4_improvement": float(improvement[15]),
        "day10_improvement": float(improvement[39]),
        "day4_2m_rmse": float(rmse_2m[15]),
        "day10_2m_rmse": float(rmse_2m[39]),
    }


def main() -> None:
    loaded = [(label, train, evaluation, load(path), color, style, path)
              for label, train, evaluation, path, color, style in SPECS]
    anchors = {tuple(value["chosen_idx"]) for _, _, _, value, _, _, _ in loaded}
    horizons = {int(value["target_steps"]) for _, _, _, value, _, _, _ in loaded}
    if len(anchors) != 1 or len(horizons) != 1:
        raise ValueError("SWA evaluations do not share anchors and horizon")
    for label, _, evaluation, value, _, _, _ in loaded:
        expected_reset = evaluation == "reset"
        if bool(value.get("rs_reset_every_step", False)) != expected_reset:
            raise ValueError(f"{label} has unexpected reset-state metadata")

    horizon = horizons.pop()
    lead_days = np.arange(1, horizon + 1) * 0.25
    baseline_2m = np.mean(np.stack([two_m(value, "rmse_baseline")
                                    for _, _, _, value, _, _, _ in loaded]), axis=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.6),
        gridspec_kw={"width_ratios": (1.25, 1.25, 0.9)},
    )
    axes[0].axhline(0.0, color="0.2", lw=1.3, label="Vanilla GC500k")
    axes[1].plot(lead_days, baseline_2m, color="0.2", lw=1.5,
                 label="Vanilla GC500k")

    rows = []
    summaries = {}
    sources = {}
    for label, train, evaluation, value, color, style, path in loaded:
        improvement = allvar_improvement(value)
        rmse_2m = two_m(value, "rmse_full")
        summaries[(train, evaluation)] = summarize(improvement, rmse_2m)
        sources[(train, evaluation)] = path
        axes[0].plot(lead_days, improvement, color=color, ls=style, lw=2.4,
                     label=label)
        axes[1].plot(lead_days, rmse_2m, color=color, ls=style, lw=2.4,
                     label=label)
        for axis, values in ((axes[0], improvement), (axes[1], rmse_2m)):
            axis.scatter(
                [4.0, 10.0],
                [values[15], values[39]],
                color=color,
                s=26,
                zorder=4,
            )
        for index, lead_day in enumerate(lead_days):
            rows.append({
                "label": label,
                "training_policy": train,
                "evaluation_policy": evaluation,
                "lead_days": float(lead_day),
                "equal_variable_rmse_improvement_pct": float(improvement[index]),
                "two_m_temperature_rmse_k": float(rmse_2m[index]),
                "eval_json": str(path),
            })

    axes[0].set_title("All variables")
    axes[0].set_ylabel("equal-variable RMSE reduction (%)")
    axes[1].set_title("2 m temperature")
    axes[1].set_ylabel("RMSE (K)")
    for axis in axes[:2]:
        axis.set_xlabel("forecast lead time (days)")
        axis.set_xlim(0.25, 10.0)
        axis.set_xticks(np.arange(1, 11))
        axis.axvline(4.0, color="0.55", lw=0.9, ls=":", zorder=0)
        axis.grid(alpha=0.28)

    summary_order = (
        ("Carry / carry", summaries[("carry", "carry")]),
        ("Reset / reset", summaries[("reset", "reset")]),
        ("Carry / reset", summaries[("carry", "reset")]),
    )
    table_values = [
        [
            f"{summary['whole_improvement']:.2f}",
            f"{summary['day4_improvement']:.2f}",
            f"{summary['day10_improvement']:.2f}",
        ]
        for _, summary in summary_order
    ]
    axes[2].axis("off")
    axes[2].set_title("All-variable summary (%)", pad=13)
    table = axes[2].table(
        cellText=table_values,
        rowLabels=[label for label, _ in summary_order],
        colLabels=["40-lead mean", "Day 4", "Day 10"],
        cellLoc="center",
        rowLoc="left",
        loc="center",
        colWidths=[0.40, 0.27, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.65)
    axes[2].text(
        0.5,
        0.23,
        "Positive = lower RMSE than GC500k",
        transform=axes[2].transAxes,
        ha="center",
        color="0.3",
        fontsize=8.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 0.98), frameon=False)
    figure.suptitle("v22 res2 state-policy SWA(4k–12k) comparison", y=1.04)
    figure.text(
        0.5,
        -0.045,
        "Vanilla GC500k baseline; closed-full rollout; 32 matched cold anchors; "
        "40 six-hour leads; zero initial state. Carry and reset SWAs are distinct "
        "v22 training executions; see sidecar annotations.",
        ha="center",
        fontsize=8.5,
        color="0.25",
    )
    figure.tight_layout()
    image_path = OUTPUT_DIR / IMAGE_NAME
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    csv_path = DATA_DIR / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {image_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
