#!/usr/bin/env python3
"""Classify, report, and plot the matched v22 evaluations completed 2026-08-15."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path("artifacts/checkpoints/v22_final")
RES1_ROOT = ROOT / "res1_dm_7yr_k20_di_bcg_20k_20260813"
DATA_DIR = Path(
    "plots/analyze_models/data/resolution_eval/"
    "v22_final_res1_state_policy_20260815"
)
IMAGE_DIR = Path(
    "plots/analyze_models/images/resolution_eval/"
    "v22_final_res1_state_policy_20260815"
)

VARIANTS = {
    "di16 / BCG1": "di16_bcg1_closed_sg_stateful_20k",
    "di16 / BCG4": "di16_bcg4_closed_sg_stateful_20k",
    "di32 / BCG1": "di32_bcg1_closed_sg_stateful_20k",
    "di32 / BCG4": "di32_bcg4_closed_sg_stateful_20k",
}
CHECKPOINTS = {
    "SWA 2k–8k": "swa_step02000-08000.json",
    "SWA 4k–10k": "swa_step04000-10000.json",
    "SWA 6k–12k": "swa_step06000-12000.json",
    "SWA 8k–14k": "swa_step08000-14000.json",
    "SWA 10k–16k": "swa_step10000-16000.json",
    "Step 16k": "step16000.json",
}
CHECKPOINT_X = {
    "SWA 2k–8k": 5,
    "SWA 4k–10k": 7,
    "SWA 6k–12k": 9,
    "SWA 8k–14k": 11,
    "SWA 10k–16k": 13,
    "Step 16k": 16,
}
POLICIES = {
    "carry": "cold_full_zero",
    "forced reset": "cold_full_reset_every_step_zero",
}
COLOURS = {
    "di16 / BCG1": "#31688E",
    "di16 / BCG4": "#35B779",
    "di32 / BCG1": "#E76F51",
    "di32 / BCG4": "#7E57C2",
}

# Matches the metric used by LH's earlier v22cl comparison plots. It is not
# numerically interchangeable with the equal-variable mean RMSE reduction.
PAPER_VARIABLE_WEIGHTS = {
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
    "geopotential": 1.0,
    "temperature": 1.0,
    "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0,
    "vertical_velocity": 1.0,
    "specific_humidity": 1.0,
}

RES2_SPECS = (
    (
        "Carry-trained / carry eval",
        ROOT / "res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/"
        "eval/cold_full_zero/swa_step04000-12000.json",
    ),
    (
        "Carry-trained / forced reset",
        ROOT / "res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/"
        "eval/cold_full_reset_every_step_zero/swa_step04000-12000.json",
    ),
    (
        "Reset-trained / reset eval",
        ROOT / "res2_gc500k_k12_full_mamba_state_ablation_20k/"
        "di16_closed_sg_reset_every_anchor_20k/eval/"
        "cold_full_reset_every_step_zero/swa_step04000-12000.json",
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
        model = np.asarray(metrics["rmse_full"], dtype=float)
        curves.append(100.0 * (1.0 - model / baseline))
    result = np.mean(np.stack(curves), axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite all-variable improvement")
    return result


def two_m(value: dict, field: str) -> np.ndarray:
    result = np.asarray(
        value["per_variable_per_step"]["2m_temperature"][field], dtype=float
    )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite 2 m metric: {field}")
    return result


def load_paper_sigmas() -> dict[str, float]:
    path = Path("data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
    with xr.open_dataset(path) as dataset:
        return {
            variable: float(
                dataset[variable].values
                if dataset[variable].values.ndim == 0
                else dataset[variable].values.mean()
            )
            for variable in PAPER_VARIABLE_WEIGHTS
            if variable in dataset
        }


def load_diff_scales() -> xr.Dataset:
    return xr.load_dataset("data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")


def paper_weighted_mse_improvement(
    value: dict, sigmas: dict[str, float]
) -> np.ndarray:
    baseline_total = np.zeros(int(value["target_steps"]), dtype=float)
    model_total = np.zeros_like(baseline_total)
    for variable, metrics in value["per_variable_per_step"].items():
        if variable not in sigmas:
            continue
        weight = PAPER_VARIABLE_WEIGHTS[variable]
        sigma = sigmas[variable]
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        model = np.asarray(metrics["rmse_full"], dtype=float)
        baseline_total += weight * baseline**2 / sigma**2
        model_total += weight * model**2 / sigma**2
    result = 100.0 * (1.0 - model_total / baseline_total)
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite paper-weighted MSE improvement")
    return result


def graphcast_weighted_normalized_mse_improvement(
    value: dict, diff_scales: xr.Dataset
) -> np.ndarray:
    """Reconstruct GraphCast's normalized training-loss weighting from metrics.

    The saved channel RMSEs already contain the latitude-weighted spatial mean.
    We apply per-channel residual normalization, pressure-proportional level
    weights, mean over levels within an atmospheric variable, then sum the
    GraphCast per-variable weights.
    """
    target_steps = int(value["target_steps"])
    grouped: dict[str, list[tuple[int | None, dict]]] = defaultdict(list)
    for channel, metrics in value["per_channel_per_step"].items():
        level_match = re.fullmatch(r"(.+)_level(\d+)", channel)
        if level_match:
            variable, level_text = level_match.groups()
            grouped[variable].append((int(level_text), metrics))
        else:
            grouped[channel].append((None, metrics))

    totals = {
        "baseline": np.zeros(target_steps, dtype=float),
        "full": np.zeros(target_steps, dtype=float),
    }
    for variable, channels in grouped.items():
        variable_weight = PAPER_VARIABLE_WEIGHTS.get(variable, 1.0)
        if channels[0][0] is None:
            if len(channels) != 1:
                raise ValueError(f"Unexpected surface channels for {variable}: {len(channels)}")
            scale = float(diff_scales[variable].values)
            metrics = channels[0][1]
            for branch, field in (("baseline", "rmse_baseline"), ("full", "rmse_full")):
                rmse = np.asarray(metrics[field], dtype=float)
                totals[branch] += variable_weight * rmse**2 / scale**2
            continue

        channels.sort(key=lambda item: int(item[0]))
        levels = np.asarray([int(level) for level, _ in channels], dtype=float)
        normalized_level_weights = levels / levels.mean()
        for branch, field in (("baseline", "rmse_baseline"), ("full", "rmse_full")):
            normalized_channel_mses = []
            for level, metrics in channels:
                scale = float(diff_scales[variable].sel(level=int(level)).values)
                rmse = np.asarray(metrics[field], dtype=float)
                normalized_channel_mses.append(rmse**2 / scale**2)
            per_variable = np.mean(
                np.stack(normalized_channel_mses)
                * normalized_level_weights[:, None],
                axis=0,
            )
            totals[branch] += variable_weight * per_variable

    result = 100.0 * (1.0 - totals["full"] / totals["baseline"])
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite GraphCast-weighted normalized MSE improvement")
    return result


def outcome_class(mean_improvement: float) -> str:
    if mean_improvement >= 15.0:
        return "strong"
    if mean_improvement >= 10.0:
        return "useful"
    if mean_improvement >= 0.0:
        return "marginal"
    return "regression"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict], key: str, metric: str
) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row[metric]))
    return sorted(
        ((label, float(np.mean(values))) for label, values in grouped.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def plot_checkpoint_sweep(rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), sharex=True)
    metrics = (
        (
            "mean_graphcast_weighted_mse_improvement_pct",
            "Mean weighted MSE reduction (%)",
        ),
        (
            "day10_graphcast_weighted_mse_improvement_pct",
            "Day-10 weighted MSE reduction (%)",
        ),
    )
    for row_index, (metric, ylabel) in enumerate(metrics):
        for col_index, policy in enumerate(POLICIES):
            axis = axes[row_index, col_index]
            axis.axhline(0.0, color="0.2", lw=1.0)
            for variant in VARIANTS:
                selected = [
                    row for row in rows
                    if row["evaluation_policy"] == policy
                    and row["variant"] == variant
                ]
                selected.sort(key=lambda row: CHECKPOINT_X[str(row["checkpoint_or_swa"])])
                axis.plot(
                    [CHECKPOINT_X[str(row["checkpoint_or_swa"])] for row in selected],
                    [float(row[metric]) for row in selected],
                    marker="o",
                    lw=2.0,
                    color=COLOURS[variant],
                    label=variant,
                )
            axis.set_title(f"{policy.title()} evaluation")
            if col_index == 0:
                axis.set_ylabel(ylabel)
            axis.grid(alpha=0.28)
    for axis in axes[-1]:
        axis.set_xlabel("checkpoint step / SWA midpoint (thousands)")
        axis.set_xticks(list(CHECKPOINT_X.values()))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        frameon=False,
    )
    figure.suptitle(
        "v22 res1: GraphCast-loss-weighted architecture and SWA comparison",
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Positive values reduce normalized weighted MSE versus vanilla GC500k; 83 channels; 32 matched cold anchors.",
        ha="center",
        fontsize=8.5,
        color="0.25",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.885))
    figure.savefig(IMAGE_DIR / "checkpoint_sweep_allvars.png", dpi=180)
    plt.close(figure)


def plot_policy_heatmap(rows: list[dict]) -> None:
    lookup = {
        (str(row["variant"]), str(row["checkpoint_or_swa"]), str(row["evaluation_policy"])):
        float(row["mean_graphcast_weighted_mse_improvement_pct"])
        for row in rows
    }
    matrix = np.asarray([
        [lookup[(variant, checkpoint, "carry")] - lookup[(variant, checkpoint, "forced reset")]
         for checkpoint in CHECKPOINTS]
        for variant in VARIANTS
    ])
    figure, axis = plt.subplots(figsize=(11.8, 4.8))
    limit = max(1.0, float(np.max(np.abs(matrix))))
    image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(col_index, row_index, f"{matrix[row_index, col_index]:+.2f}",
                      ha="center", va="center", fontsize=9)
    axis.set_xticks(range(len(CHECKPOINTS)), list(CHECKPOINTS), rotation=25, ha="right")
    axis.set_yticks(range(len(VARIANTS)), list(VARIANTS))
    axis.set_title(
        "Carry-state benefit: 40-lead mean GraphCast-loss-weighted MSE"
    )
    colourbar = figure.colorbar(image, ax=axis, pad=0.02)
    colourbar.set_label("carry minus forced-reset benefit (percentage points)")
    figure.tight_layout()
    figure.savefig(
        IMAGE_DIR / "carry_benefit_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_graphcast_weighted_checkpoint_sweep(rows: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), sharex=True)
    for axis, policy in zip(axes, POLICIES, strict=True):
        axis.axhline(0.0, color="0.2", lw=1.0)
        for variant in VARIANTS:
            selected = [
                row for row in rows
                if row["evaluation_policy"] == policy
                and row["variant"] == variant
            ]
            selected.sort(key=lambda row: CHECKPOINT_X[str(row["checkpoint_or_swa"])])
            axis.plot(
                [CHECKPOINT_X[str(row["checkpoint_or_swa"])] for row in selected],
                [float(row["day10_graphcast_weighted_mse_improvement_pct"])
                 for row in selected],
                marker="o",
                lw=2.0,
                color=COLOURS[variant],
                label=variant,
            )
        axis.set_title(f"{policy.title()} evaluation")
        axis.set_xlabel("checkpoint step / SWA midpoint (thousands)")
        axis.set_ylabel("day-10 weighted normalized MSE reduction (%)")
        axis.set_xticks(list(CHECKPOINT_X.values()))
        axis.grid(alpha=0.28)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, 0.96), frameon=False)
    figure.suptitle("v22 res1: GraphCast-loss-weighted day-10 forecast improvement", y=1.02)
    figure.text(
        0.5,
        -0.025,
        "83 channels; per-level difference normalization; pressure-level, variable, and latitude weighting.",
        ha="center",
        fontsize=8.5,
        color="0.25",
    )
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "checkpoint_sweep_graphcast_weighted_mse.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)


def plot_best_curves(curve_rows: list[dict], summary_rows: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharex=True, sharey=True)
    for axis, policy in zip(axes, POLICIES, strict=True):
        ranked = sorted(
            (row for row in summary_rows if row["evaluation_policy"] == policy),
            key=lambda row: float(
                row["mean_graphcast_weighted_mse_improvement_pct"]
            ),
            reverse=True,
        )[:4]
        axis.axhline(0.0, color="0.2", lw=1.1, label="Vanilla GC500k")
        for rank, summary in enumerate(ranked, start=1):
            selected = [
                row for row in curve_rows
                if row["variant"] == summary["variant"]
                and row["checkpoint_or_swa"] == summary["checkpoint_or_swa"]
                and row["evaluation_policy"] == policy
            ]
            selected.sort(key=lambda row: float(row["lead_days"]))
            label = f"#{rank} {summary['variant']}, {summary['checkpoint_or_swa']}"
            axis.plot(
                [float(row["lead_days"]) for row in selected],
                [float(row["graphcast_weighted_mse_improvement_pct"])
                 for row in selected],
                lw=2.1,
                label=label,
            )
        axis.set_title(f"{policy.title()} evaluation")
        axis.set_xlabel("forecast lead time (days)")
        axis.set_ylabel("GraphCast-loss-weighted normalized MSE reduction (%)")
        axis.set_xlim(0.25, 10.0)
        axis.set_xticks(np.arange(1, 11))
        axis.grid(alpha=0.28)
        axis.legend(fontsize=8.1, frameon=False)
    figure.suptitle(
        "Top res1 evaluations by 40-lead mean GraphCast-loss-weighted MSE"
    )
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "top_evaluation_rollout_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    paper_sigmas = load_paper_sigmas()
    diff_scales = load_diff_scales()
    loaded: dict[tuple[str, str, str], tuple[dict, Path]] = {}
    for variant, run in VARIANTS.items():
        for policy, eval_dir in POLICIES.items():
            for checkpoint, filename in CHECKPOINTS.items():
                path = RES1_ROOT / run / "eval" / eval_dir / filename
                loaded[(variant, checkpoint, policy)] = (load(path), path)
    if len(loaded) != 48:
        raise AssertionError(f"Expected 48 evaluations, found {len(loaded)}")

    anchors = {tuple(value["chosen_idx"]) for value, _ in loaded.values()}
    horizons = {int(value["target_steps"]) for value, _ in loaded.values()}
    if len(anchors) != 1 or horizons != {40}:
        raise ValueError("The res1 evaluation batch is not matched on anchors/horizon")

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    for (variant, checkpoint, policy), (value, path) in loaded.items():
        expected_reset = policy == "forced reset"
        if bool(value.get("rs_reset_every_step", False)) != expected_reset:
            raise ValueError(f"Unexpected reset metadata in {path}")
        improvement = allvar_improvement(value)
        weighted_mse = paper_weighted_mse_improvement(value, paper_sigmas)
        graphcast_mse = graphcast_weighted_normalized_mse_improvement(
            value, diff_scales
        )
        full_2m = two_m(value, "rmse_full")
        baseline_2m = two_m(value, "rmse_baseline")
        mean_improvement = float(np.mean(improvement))
        mean_graphcast_improvement = float(np.mean(graphcast_mse))
        summary_rows.append({
            "variant": variant,
            "checkpoint_or_swa": checkpoint,
            "evaluation_policy": policy,
            "outcome_class": outcome_class(mean_graphcast_improvement),
            "mean_improvement_pct": mean_improvement,
            "day4_improvement_pct": float(improvement[15]),
            "day10_improvement_pct": float(improvement[39]),
            "mean_paper_weighted_mse_improvement_pct": float(np.mean(weighted_mse)),
            "day4_paper_weighted_mse_improvement_pct": float(weighted_mse[15]),
            "day10_paper_weighted_mse_improvement_pct": float(weighted_mse[39]),
            "mean_graphcast_weighted_mse_improvement_pct": mean_graphcast_improvement,
            "day4_graphcast_weighted_mse_improvement_pct": float(graphcast_mse[15]),
            "day10_graphcast_weighted_mse_improvement_pct": float(graphcast_mse[39]),
            "day4_2m_rmse_k": float(full_2m[15]),
            "day10_2m_rmse_k": float(full_2m[39]),
            "day4_baseline_2m_rmse_k": float(baseline_2m[15]),
            "day10_baseline_2m_rmse_k": float(baseline_2m[39]),
            "source_json": str(path),
        })
        for index in range(40):
            curve_rows.append({
                "variant": variant,
                "checkpoint_or_swa": checkpoint,
                "evaluation_policy": policy,
                "lead_days": (index + 1) * 0.25,
                "allvar_improvement_pct": float(improvement[index]),
                "paper_weighted_mse_improvement_pct": float(weighted_mse[index]),
                "graphcast_weighted_mse_improvement_pct": float(graphcast_mse[index]),
                "two_m_rmse_k": float(full_2m[index]),
            })

    for policy in POLICIES:
        ranked = sorted(
            (row for row in summary_rows if row["evaluation_policy"] == policy),
            key=lambda row: float(
                row["mean_graphcast_weighted_mse_improvement_pct"]
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["policy_rank"] = rank
    summary_rows.sort(key=lambda row: (str(row["evaluation_policy"]), int(row["policy_rank"])))
    write_csv(DATA_DIR / "evaluation_summary.csv", summary_rows)
    write_csv(DATA_DIR / "evaluation_lead_curves.csv", curve_rows)

    plot_checkpoint_sweep(summary_rows)
    plot_graphcast_weighted_checkpoint_sweep(summary_rows)
    plot_policy_heatmap(summary_rows)
    plot_best_curves(curve_rows, summary_rows)

    res2 = []
    for label, path in RES2_SPECS:
        value = load(path)
        improvement = graphcast_weighted_normalized_mse_improvement(
            value, diff_scales
        )
        full_2m = two_m(value, "rmse_full")
        res2.append((label, float(np.mean(improvement)), float(improvement[15]),
                     float(improvement[39]), float(full_2m[15]), float(full_2m[39])))

    best_by_policy = {
        policy: min(
            (row for row in summary_rows if row["evaluation_policy"] == policy),
            key=lambda row: int(row["policy_rank"]),
        )
        for policy in POLICIES
    }
    paired = []
    lookup = {
        (str(row["variant"]), str(row["checkpoint_or_swa"]), str(row["evaluation_policy"])):
        float(row["mean_graphcast_weighted_mse_improvement_pct"])
        for row in summary_rows
    }
    for variant in VARIANTS:
        for checkpoint in CHECKPOINTS:
            paired.append(lookup[(variant, checkpoint, "carry")] -
                          lookup[(variant, checkpoint, "forced reset")])

    class_counts: dict[str, dict[str, int]] = {
        policy: {
            label: sum(
                row["evaluation_policy"] == policy
                and row["outcome_class"] == label
                for row in summary_rows
            )
            for label in ("strong", "useful", "marginal", "regression")
        }
        for policy in POLICIES
    }

    lines = [
        "# v22 evaluations completed 2026-08-15",
        "",
        "## Classification",
        "",
        "All 48 res1 results are matched on 32 cold anchors, 40 six-hour leads, and the vanilla GC500k baseline. Every ranking and unqualified result in this report uses GraphCast-loss-weighted normalized all-variable MSE reduction. Outcome labels use its 40-lead mean: **strong ≥15%**, **useful 10–15%**, **marginal 0–10%**, and **regression <0%**.",
        "",
        "## Res1 headline rankings",
        "",
        "| Evaluation policy | Best variant | Checkpoint/SWA | Mean GraphCast MSE | Day-10 GraphCast MSE | LH legacy MSE | Day-10 equal-var RMSE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for policy, row in best_by_policy.items():
        lines.append(
            f"| {policy} | {row['variant']} | {row['checkpoint_or_swa']} | "
            f"{float(row['mean_graphcast_weighted_mse_improvement_pct']):.2f}% | "
            f"{float(row['day10_graphcast_weighted_mse_improvement_pct']):.2f}% | "
            f"{float(row['day10_paper_weighted_mse_improvement_pct']):.2f}% | "
            f"{float(row['day10_improvement_pct']):.2f}% |"
        )
    lines += [
        "",
        "The GraphCast MSE column reconstructs the actual normalized GraphCast loss weighting from all 83 saved channels: latitude-weighted MSE, each pressure level's own difference scale, pressure-proportional level weights, and GraphCast variable weights. The LH legacy column uses variable-level RMSE and a mean scale across pressure levels, so it is retained only for historical plot reconciliation.",
        "",
        "Because the JSON stores latitude-aggregated channel RMSE rather than raw grid-cell errors, the reconstruction inherits the evaluator's cosine-latitude aggregation. An exactly official pole-cell weighting would require the raw rollout fields or a new evaluation; the expected difference is small.",
        "",
        f"Across the 24 matched configuration pairs, carrying state changes the mean improvement by **{np.mean(paired):+.2f} percentage points on average** (range {np.min(paired):+.2f} to {np.max(paired):+.2f} pp).",
        "",
        f"Weighted-MSE outcome counts: carry evaluation has **{class_counts['carry']['marginal']} marginal improvements and {class_counts['carry']['regression']} regressions**; forced-reset evaluation has **{class_counts['forced reset']['regression']} regressions**. Rankings below use the same weighted metric.",
        "",
        "### Architecture averages across all six checkpoint choices",
        "",
    ]
    for policy in POLICIES:
        lines += [f"**{policy.title()} evaluation**", ""]
        policy_rows = [row for row in summary_rows if row["evaluation_policy"] == policy]
        for rank, (label, value) in enumerate(
            aggregate(
                policy_rows,
                "variant",
                "mean_graphcast_weighted_mse_improvement_pct",
            ),
            start=1,
        ):
            lines.append(f"{rank}. {label}: {value:.2f}%")
        lines.append("")
    lines += [
        "## Res2 matched state-policy SWA(4k–12k), GraphCast-loss-weighted MSE",
        "",
        "| Training / evaluation | Mean | Day 4 | Day 10 | 2 m day 4 | 2 m day 10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, mean, day4, day10, rmse4, rmse10 in res2:
        lines.append(
            f"| {label} | {mean:.2f}% | {day4:.2f}% | {day10:.2f}% | "
            f"{rmse4:.3f} K | {rmse10:.3f} K |"
        )
    lines += [
        "",
        "The res2 carry-trained/carry-evaluated model is strongest. Forced reset on that same model is a direct state ablation; comparing it with the separately trained reset model also changes the training execution and is therefore not a perfectly isolated causal comparison.",
        "",
        "## Interpretation limits",
        "",
        "- Rankings are valid within each matched res1 evaluation policy. Carry and forced-reset inference use the same checkpoints, making their paired difference interpretable as a recurrent-state ablation.",
        "- These JSONs contain aggregate RMSE curves rather than per-anchor samples, so this report does not attach confidence intervals or claim statistical significance.",
        "- The res1 and res2 studies differ in resolution/training setup and should not be ranked against each other numerically.",
    ]
    (DATA_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validated {len(loaded)} matched res1 evaluations")
    print(f"Saved report and CSVs in {DATA_DIR}")
    print(f"Saved four plots in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
