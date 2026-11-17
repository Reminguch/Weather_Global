#!/usr/bin/env python3
"""Plot saved MGDA/MSE continuation checkpoints on their shared spectral cohort.

Run from the repository with:
    source scripts/graphcast_env.sh
    python scripts/analyze_models/plot_v24_mgda_steps_summary_20261113.py

No forecasting or checkpoint loading is performed. The separate 32-start exact
evaluation is deliberately not used: both plotted metrics use the same eight
2023 forecast initializations in the saved spectral evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STEM = "v24_training_steps_and_k_20261113"
CAMPAIGN = ROOT / "artifacts/checkpoints/v24_Ilya/res1_mgda_gc_power_2k_recovery_20261007"
DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / STEM
IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / STEM
STEPS = (0, 500, 1000, 1500, 2000)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def source_record(path: Path) -> dict:
    return {
        "path": relative_path(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def collect(campaign: Path) -> tuple[list[dict], dict]:
    report_path = campaign / "evaluation/report/mgda_report.json"
    report = read_json(report_path)
    if report["status"] != "complete":
        raise ValueError("The saved MGDA report must be complete.")
    records = {record["id"]: record for record in report["records"]}
    parent_path = campaign / "evaluation/results/mature_mse/spectral/evaluation.json"
    parent = read_json(parent_path)
    dates = parent["global_initialization_times"]
    parent_mse = parent["original_graphcast_loss"]["full_rollout"]
    parent_power = records["mature_mse"]["original_gc_context"]["mean_relative_power_error_pct"]
    sources = {report_path, parent_path, campaign / "evaluation/manifest.json"}
    if len(dates) != 8 or any(not date.startswith("2023-") for date in dates):
        raise ValueError("Expected the documented eight-start 2023 spectral cohort.")

    rows = []
    for family, label in (("mgda", "MGDA"), ("mse", "MSE control")):
        config_path = campaign / "runs" / family / "run_config.json"
        config = read_json(config_path)
        sources.add(config_path)
        expected_sequence = {
            "ar_tail_k": 20, "bptt_steps": 24, "segment_steps": 96,
            "feedback_mode": "closed_loop_sg",
        }
        if any(config["sequence"][key] != value for key, value in expected_sequence.items()) or config["optimizer"]["seed"] != 22:
            raise ValueError("This summary requires the matched K20/BPTT24/seed22 continuation recipe.")
        for step in STEPS:
            model_id = "mature_mse" if step == 0 else f"{family}_step{step:05d}"
            record = records[model_id]
            evaluation_path = campaign / "evaluation/results" / model_id / "spectral/evaluation.json"
            evaluation = read_json(evaluation_path)
            sources.add(evaluation_path)
            if not (
                evaluation["evaluation_status"] == "complete"
                and evaluation["evaluated_samples"] == evaluation["n_samples"] == 8
                and evaluation["target_steps"] == 40
                and evaluation["global_initialization_times"] == dates
                and evaluation["eval_mode"] == "cold_full"
                and evaluation["residual_state_init"] == "zero"
            ):
                raise ValueError(f"Incomplete or unmatched evaluation: {evaluation_path}")
            mse = evaluation["original_graphcast_loss"]["full_rollout"]
            power = record["original_gc_context"]["mean_relative_power_error_pct"]
            comparison = record["comparisons"]["initialization"]
            mse_ratio = comparison["exact_gc_loss_ratio"]
            power_ratio = comparison["mean_power_error_ratio"]
            # This establishes that the report ratios belong to the spectral
            # cohort, rather than the separate 32-start exact evaluations.
            np.testing.assert_allclose(mse_ratio, mse / parent_mse, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(power_ratio, power / parent_power, rtol=1e-10, atol=1e-12)
            rows.append({
                "family": family,
                "label": label,
                "model_id": model_id,
                "additional_training_updates": step,
                "training_k": 20,
                "training_seed": config["optimizer"]["seed"],
                "evaluation_starts": 8,
                "evaluation_year": 2023,
                "evaluation_leads": 40,
                "exact_gc_rollout_mse": mse,
                "exact_mse_ratio_vs_initialization": mse_ratio,
                "exact_mse_change_vs_initialization_pct": 100.0 * (mse_ratio - 1.0),
                "zonal_power_mismatch_pct": power,
                "zonal_power_mismatch_ratio_vs_initialization": power_ratio,
                "zonal_power_mismatch_change_vs_initialization_pct": 100.0 * (power_ratio - 1.0),
                "repeated_public_gc_rollout_mse": evaluation["original_graphcast_loss"]["baseline_rollout"],
                "source_evaluation": relative_path(evaluation_path),
                "checkpoint_sha256": evaluation["checkpoint_sha256"],
            })
    if not all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value, (float, int))):
        raise ValueError("Nonfinite metric in saved results.")

    # Duplicate step-zero rows are intentional for drawing each trajectory;
    # the repeated-baseline diagnostic counts that evaluation only once.
    unique_rows = {row["model_id"]: row for row in rows}
    baseline_values = np.array([row["repeated_public_gc_rollout_mse"] for row in unique_rows.values()])
    parent_baseline = parent["original_graphcast_loss"]["baseline_rollout"]
    provenance = {
        "scope": "Raw MGDA and matched MSE continuation checkpoints; fixed K=20; one seed; eight shared 2023 starts.",
        "source_campaign": relative_path(campaign),
        "source_report": relative_path(report_path),
        "sources": [source_record(path) for path in sorted(sources)],
        "script": source_record(Path(__file__)),
        "initialization_times": dates,
        "additional_training_updates": list(STEPS),
        "metric_definitions": {
            "exact_gc_rollout_mse": "Original GraphCast normalized latitude-, pressure-level- and variable-weighted grid MSE, averaged across all forty six-hour leads and the eight dates.",
            "zonal_power_mismatch": "Saved report mean absolute relative zonal-power error: pool powers over starts first, then positive wavenumbers; nine primitive fields at forecast days 1, 3, 7, 10. Unsmoothed and unfloored.",
            "plotted_change_pct": "100 * (metric_at_checkpoint / metric_at_mature_MSE_initialization - 1); negative values indicate improvement.",
        },
        "cohort_verification": "Each exact MSE ratio was checked against full_rollout from the eight-start spectral/evaluation.json. Separate 32-start exact evaluations are not used.",
        "training": {"ar_tail_k": 20, "bptt_steps": 24, "segment_steps": 96, "weather_feedback": "closed_loop_sg", "seed": 22},
        "repeated_baseline_diagnostic": {
            "evaluation_count": len(unique_rows),
            "initialization_evaluation_public_gc_mse": parent_baseline,
            "minimum_public_gc_mse": float(baseline_values.min()),
            "maximum_public_gc_mse": float(baseline_values.max()),
            "max_minus_min_public_gc_mse": float(np.ptp(baseline_values)),
            "maximum_absolute_difference_from_initialization_public_gc_mse": float(np.max(np.abs(baseline_values - parent_baseline))),
            "maximum_absolute_difference_from_initialization_public_gc_pct": float(100.0 * np.max(np.abs(baseline_values - parent_baseline)) / parent_baseline),
            "interpretation": "Finite variation across repeated public-GC forecasts is a warning, not an acceptance gate or estimate of the corrected models' full noise. Plot uses direct model losses, so their changing paired baselines are not denominators.",
        },
        "limitations": [
            "A mature-parent continuation does not describe fresh-training convergence or principal M1.",
            "One training seed and eight development/exploratory dates do not establish general or annual trends.",
            "Power amplitude mismatch does not measure spatial alignment; stored cross/error powers support separate analyses.",
            "The nine-field zonal metric is distinct from the newer 83-channel spherical T179 metric; scores must not be combined.",
            "SWA omitted because its averaged checkpoints do not represent a single raw training-update coordinate.",
            "Curves are point estimates; no new uncertainty or significance analysis is performed.",
            "Training K is fixed, so these curves do not identify training-rollout-length effects.",
        ],
    }
    return rows, provenance


def plot(rows: list[dict], image_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#253448", "text.color": "#253448",
        "axes.edgecolor": "#aab4c0", "xtick.color": "#4d5b6b", "ytick.color": "#4d5b6b",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.65))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.30, top=0.73, wspace=0.29)
    fig.suptitle("MGDA continuation: accuracy and power vs training updates", x=0.075, y=0.96,
                 ha="left", fontsize=16, weight="bold")
    fig.text(0.075, 0.902, "Shared mature MSE initialization  ·  Training K = 20  ·  One seed  ·  8 matched 2023 starts", fontsize=10.5)
    metrics = [
        ("exact_mse_change_vs_initialization_pct", "Exact GC rollout MSE", (-0.15, 3.8)),
        ("zonal_power_mismatch_change_vs_initialization_pct", "Zonal power mismatch", (-17.0, 1.1)),
    ]
    colors = {"mgda": "#2166ac", "mse": "#c87b20"}
    for ax, (metric, title, limits) in zip(axes, metrics):
        ax.axhline(0, color="#8d99a8", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for family in ("mgda", "mse"):
            group = [row for row in rows if row["family"] == family]
            x = [row["additional_training_updates"] for row in group]
            y = [row[metric] for row in group]
            ax.plot(x, y, marker="o" if family == "mgda" else "s", markersize=5.5,
                    linewidth=2.2, color=colors[family], label=group[0]["label"], zorder=3)
            ax.annotate(f"{y[-1]:+.2f}%", (x[-1], y[-1]), xytext=(-4, 11 if metric.startswith("exact") else -17),
                        textcoords="offset points", ha="right", color=colors[family], fontsize=10, weight="bold")
        ax.set_title(title, loc="left", fontsize=12.5, weight="bold", pad=12)
        ax.set_xlabel("Additional training updates", labelpad=9)
        ax.set_ylabel("Change from initialization (%)", labelpad=9)
        ax.set_xticks(STEPS, ["0", "500", "1,000", "1,500", "2,000"])
        ax.set_xlim(-60, 2090)
        ax.set_ylim(*limits)
        ax.grid(axis="y", color="#e8edf2", linewidth=0.9)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.069, 0.865), ncol=2, frameon=False,
               handlelength=2.5, columnspacing=2.2)
    fig.text(0.075, 0.168, "Lower is better in both panels. MSE averages all 40 six-hour forecast leads (10 days).", fontsize=10)
    fig.text(0.075, 0.120, "Power: pooled zonal spectra; 9 fields at days 1, 3, 7, 10. This differs from the 83-channel spherical metric.", fontsize=9.3, color="#536273")
    fig.text(0.075, 0.076, "Raw checkpoints; point estimates on a small exploratory cohort. SWA and the separate 32-start evaluation are omitted.", fontsize=9.3, color="#536273")
    for extension in ("png", "pdf"):
        fig.savefig(image_dir / f"mgda_steps.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    args = parser.parse_args()
    rows, provenance = collect(args.campaign)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.data_dir / "mgda_steps.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.image_dir)
    provenance["outputs"] = [source_record(csv_path)] + [
        source_record(args.image_dir / f"mgda_steps.{extension}") for extension in ("png", "pdf")
    ]
    provenance_path = args.data_dir / "mgda_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "data": str(csv_path), "provenance": str(provenance_path),
                      "figure": str(args.image_dir / "mgda_steps.png")}, indent=2))


if __name__ == "__main__":
    main()
