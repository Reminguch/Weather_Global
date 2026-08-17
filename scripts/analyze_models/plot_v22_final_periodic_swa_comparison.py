#!/usr/bin/env python3
"""Compare periodic-SWA res2 stages with matched carry/reset references."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/checkpoints/v22_final")
PERIODIC_ROOT = ROOT / "res2_gc500k_k12_swa_stabilized_20260814"
DATA_DIR = Path(
    "plots/analyze_models/data/resolution_eval/"
    "v22_final_res2_periodic_swa_state_policy"
)
IMAGE_DIR = Path(
    "plots/analyze_models/images/resolution_eval/"
    "v22_final_res2_periodic_swa_state_policy"
)

VARIANTS = {"di16": "#31688E", "di64 / BCG4": "#E76F51"}
STAGES = {
    "Phase 1 SWA": ("phase1_8k", "swa_step02000-08000"),
    "Phase 2 SWA": ("phase2_8k", "swa_step02000-08000"),
    "Final +4k": ("phase3_4k", "step4000"),
}
POLICIES = {
    "carry": "cold_full_zero",
    "forced reset": "cold_full_reset_every_step_zero",
}

REFERENCE_SPECS = (
    (
        "Existing carry-trained SWA / carry",
        "carry",
        ROOT / "res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/"
        "eval/cold_full_zero/swa_step04000-12000.json",
    ),
    (
        "Existing carry-trained SWA / forced reset",
        "forced reset",
        ROOT / "res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/"
        "eval/cold_full_reset_every_step_zero/swa_step04000-12000.json",
    ),
    (
        "Existing reset-trained SWA / reset",
        "forced reset",
        ROOT / "res2_gc500k_k12_full_mamba_state_ablation_20k/"
        "di16_closed_sg_reset_every_anchor_20k/eval/"
        "cold_full_reset_every_step_zero/swa_step04000-12000.json",
    ),
)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    missing = {"chosen_idx", "target_steps", "per_variable_per_step"} - value.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return value


def allvars_rmse_improvement(value: dict) -> np.ndarray:
    """Canonical equal-variable allvars RMSE improvement percentage."""
    per_variable = []
    for metrics in value["per_variable_per_step"].values():
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        model = np.asarray(metrics["rmse_full"], dtype=float)
        per_variable.append(100.0 * (1.0 - model / baseline))
    result = np.mean(np.stack(per_variable), axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite allvars RMSE improvement")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_rollouts(curves: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16.0, 5.8), sharex=True, sharey=True)
    for axis, policy in zip(axes, POLICIES, strict=True):
        axis.axhline(0.0, color="0.2", lw=1.0, label="Vanilla GC500k")
        selected = [row for row in curves if row["evaluation_policy"] == policy]
        keys = list(dict.fromkeys((row["label"], row["kind"]) for row in selected))
        for label, kind in keys:
            rows = [row for row in selected if row["label"] == label]
            rows.sort(key=lambda row: float(row["lead_days"]))
            if kind == "reference":
                color = "#222222" if "carry-trained" in label else "#35B779"
                style, width = "--", 2.6
            else:
                color = VARIANTS["di16"] if label.startswith("di16") else VARIANTS["di64 / BCG4"]
                style = {"Phase 1 SWA": "-", "Phase 2 SWA": "--", "Final +4k": ":"}[
                    str(rows[0]["stage"])
                ]
                width = 2.0
            axis.plot(
                [float(row["lead_days"]) for row in rows],
                [float(row["allvars_rmse_improvement_pct"]) for row in rows],
                color=color,
                ls=style,
                lw=width,
                label=label,
            )
        axis.set_title(f"{policy.title()} evaluation")
        axis.set_xlabel("forecast lead time (days)")
        axis.set_ylabel("equal-variable allvars RMSE improvement (%)")
        axis.set_xlim(0.25, 10.0)
        axis.set_xticks(np.arange(1, 11))
        axis.grid(alpha=0.28)
        axis.legend(fontsize=7.3, frameon=False, loc="upper left")
    figure.suptitle("v22 res2 periodic-SWA restarts versus matched state-policy references")
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "allvars_rmse_rollout_comparison.png", dpi=180)
    plt.close(figure)


def plot_stage_summary(summary: list[dict]) -> None:
    periodic = [row for row in summary if row["kind"] == "periodic"]
    references = {row["label"]: row for row in summary if row["kind"] == "reference"}
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    metrics = (
        ("mean_allvars_rmse_improvement_pct", "40-lead mean improvement (%)"),
        ("day10_allvars_rmse_improvement_pct", "Day-10 improvement (%)"),
    )
    x = np.arange(len(STAGES))
    for column, variant in enumerate(VARIANTS):
        for row_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column]
            for policy, color, marker in (
                ("carry", "#31688E", "o"),
                ("forced reset", "#E76F51", "s"),
            ):
                values = []
                for stage in STAGES:
                    match = next(
                        row for row in periodic
                        if row["variant"] == variant
                        and row["stage"] == stage
                        and row["evaluation_policy"] == policy
                    )
                    values.append(float(match[metric]))
                axis.plot(x, values, color=color, marker=marker, lw=2.2, label=policy)
            if variant == "di16":
                axis.axhline(
                    float(references["Existing carry-trained SWA / carry"][metric]),
                    color="#222222", ls="--", lw=1.5, label="existing carry/carry",
                )
                axis.axhline(
                    float(references["Existing reset-trained SWA / reset"][metric]),
                    color="#35B779", ls="--", lw=1.5, label="existing reset/reset",
                )
            axis.axhline(0.0, color="0.3", lw=0.8)
            axis.set_title(variant)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.28)
    for axis in axes[-1]:
        axis.set_xticks(x, list(STAGES), rotation=18, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.946),
        frameon=False,
    )
    figure.suptitle(
        "Periodic-SWA stage summary: canonical allvars RMSE improvement",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(IMAGE_DIR / "allvars_rmse_stage_summary.png", dpi=180)
    plt.close(figure)


def plot_rankings(summary: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), sharex=True)
    for axis, policy in zip(axes, POLICIES, strict=True):
        rows = sorted(
            (row for row in summary if row["evaluation_policy"] == policy),
            key=lambda row: float(row["mean_allvars_rmse_improvement_pct"]),
        )
        labels = [str(row["label"]) for row in rows]
        values = [float(row["mean_allvars_rmse_improvement_pct"]) for row in rows]
        colors = ["#555555" if row["kind"] == "reference" else "#31688E" for row in rows]
        axis.barh(range(len(rows)), values, color=colors)
        axis.set_yticks(range(len(rows)), labels, fontsize=7.8)
        axis.axvline(0.0, color="0.2", lw=1.0)
        axis.set_title(f"{policy.title()} evaluation")
        axis.set_xlabel("40-lead mean allvars RMSE improvement (%)")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Matched res2 evaluation ranking")
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "allvars_rmse_rankings.png", dpi=180)
    plt.close(figure)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    evaluations: list[tuple[str, str, str, str, Path, dict]] = []

    for variant in VARIANTS:
        variant_token = "di16" if variant == "di16" else "di64_bcg4"
        for stage, (phase, artifact) in STAGES.items():
            run_dir = PERIODIC_ROOT / f"{variant_token}_swa_restart_{phase}"
            for policy, eval_dir in POLICIES.items():
                path = run_dir / "eval" / eval_dir / f"{artifact}.json"
                label = f"{variant}, {stage}"
                evaluations.append((label, "periodic", variant, stage, path, load(path)))

    for label, policy, path in REFERENCE_SPECS:
        evaluations.append((label, "reference", "reference", "reference", path, load(path)))

    anchors = {tuple(value["chosen_idx"]) for *_, value in evaluations}
    horizons = {int(value["target_steps"]) for *_, value in evaluations}
    if len(anchors) != 1 or horizons != {40}:
        raise ValueError("Evaluation set is not matched on anchors and horizon")

    reference_policy = {label: policy for label, policy, _ in REFERENCE_SPECS}
    summary: list[dict] = []
    curves: list[dict] = []
    for label, kind, variant, stage, path, value in evaluations:
        policy = reference_policy[label] if kind == "reference" else (
            "forced reset" if bool(value.get("rs_reset_every_step", False)) else "carry"
        )
        expected_reset = policy == "forced reset"
        if bool(value.get("rs_reset_every_step", False)) != expected_reset:
            raise ValueError(f"Unexpected reset-state metadata: {path}")
        improvement = allvars_rmse_improvement(value)
        summary.append({
            "label": label,
            "kind": kind,
            "variant": variant,
            "stage": stage,
            "evaluation_policy": policy,
            "mean_allvars_rmse_improvement_pct": float(np.mean(improvement)),
            "day4_allvars_rmse_improvement_pct": float(improvement[15]),
            "day10_allvars_rmse_improvement_pct": float(improvement[39]),
            "source_json": str(path),
        })
        for index, metric in enumerate(improvement):
            curves.append({
                "label": label,
                "kind": kind,
                "variant": variant,
                "stage": stage,
                "evaluation_policy": policy,
                "lead_days": (index + 1) * 0.25,
                "allvars_rmse_improvement_pct": float(metric),
            })

    summary.sort(
        key=lambda row: (
            str(row["evaluation_policy"]),
            -float(row["mean_allvars_rmse_improvement_pct"]),
        )
    )
    write_csv(DATA_DIR / "evaluation_summary.csv", summary)
    write_csv(DATA_DIR / "evaluation_lead_curves.csv", curves)
    plot_rollouts(curves)
    plot_stage_summary(summary)
    plot_rankings(summary)

    lines = [
        "# v22 res2 periodic-SWA evaluation",
        "",
        "All improvement percentages below are the canonical equal-variable allvars RMSE reduction relative to vanilla GC500k. All evaluations use the same 32 cold anchors and 40 six-hour leads.",
        "",
        "## Conclusions",
        "",
        "- The existing carry-trained SWA remains best under carry evaluation: 16.13% mean, 1.23 percentage points above the best periodic model.",
        "- The existing reset-trained SWA remains best under forced-reset evaluation: 14.81% mean, 4.29 points above the best periodic model.",
        "- The best periodic carry model is di64/BCG4 Phase 1 SWA at 14.90%; di16 Phase 1 is effectively tied at 14.86%.",
        "- More periodic continuation is harmful under carry: both variants decline from Phase 1 to Phase 2 and decline again after the final 4k raw-checkpoint phase.",
        "- Under forced reset, di64/BCG4 Phase 2 is the best periodic artifact at 10.52%, but it still trails purpose-trained reset/reset by 4.29 points.",
        "- Recommendation: retain the existing carry/carry and reset/reset SWAs as policy-specific winners; do not continue the periodic-SWA restart recipe as currently configured.",
        "",
    ]
    for policy in POLICIES:
        lines += [f"## {policy.title()} evaluation", ""]
        rows = [row for row in summary if row["evaluation_policy"] == policy]
        lines += [
            "| Rank | Model | 40-lead mean | Day 4 | Day 10 |",
            "|---:|---|---:|---:|---:|",
        ]
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row['label']} | "
                f"{float(row['mean_allvars_rmse_improvement_pct']):.2f}% | "
                f"{float(row['day4_allvars_rmse_improvement_pct']):.2f}% | "
                f"{float(row['day10_allvars_rmse_improvement_pct']):.2f}% |"
            )
        lines.append("")
    (DATA_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Validated and compared {len(evaluations)} matched evaluations")
    print(f"Saved report/CSVs in {DATA_DIR}")
    print(f"Saved three plots in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
