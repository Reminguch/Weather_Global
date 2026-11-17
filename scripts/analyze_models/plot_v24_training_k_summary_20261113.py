#!/usr/bin/env python3
"""Summarize existing matched v24 AR-tail evaluations without model execution.

Run from the repository with the GraphCast environment activated. The inputs
are the September K1/4/8/12/16 sweep and the historical matched-recipe K20 M1.
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
CAMPAIGN = "v24_training_steps_and_k_20261113"
STAGES = (
    ("shared_start", "500 updates", "#3975AF", "o"),
    ("step2000", "2,500 updates · raw", "#D07A2C", "s"),
    ("swa_step00500-02000", "SWA · same 2,500-update budget", "#208B77", "D"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_evidence() -> tuple[list[dict], dict]:
    rows = []
    sources = []
    common_origins = None
    baseline_by_lead = []
    config_paths = set()
    for k in (1, 4, 8, 12, 16, 20):
        if k == 20:
            campaign = ROOT / "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908"
            run = campaign / "runs/di16_fast_mamba"
            warmup = campaign / "seeds/di16_fast_mamba_warmup"
        else:
            campaign = ROOT / "artifacts/checkpoints/v24_Ilya/res1_ar_sweep_best_20260924"
            run = campaign / f"runs/k{k:02d}"
            warmup = campaign / f"warmup/k{k:02d}"
        config_paths.update((run / "run_config.json", warmup / "run_config.json"))
        for checkpoint, label, _, _ in STAGES:
            path = run / "eval/cold_full_zero_exact_gc/matched32" / f"{checkpoint}.json"
            evaluation = json.loads(path.read_text())
            if evaluation["n_samples"] != 32 or evaluation["target_steps"] != 40:
                raise ValueError(f"Unexpected evaluation dimensions: {path}")
            origins = evaluation["chosen_idx"]
            if len(origins) != 32 or len(set(origins)) != 32:
                raise ValueError(f"Unexpected origin set: {path}")
            if common_origins is None:
                common_origins = origins
            if origins != common_origins:
                raise ValueError(f"Evaluation origins differ: {path}")
            if evaluation.get("completed_chosen_idx", origins) != origins:
                raise ValueError(f"Incomplete evaluation: {path}")
            loss = evaluation["original_graphcast_loss"]
            baseline = np.asarray(loss["baseline_per_step"], dtype=float)
            full = np.asarray(loss["full_per_step"], dtype=float)
            reduction = np.asarray(loss["improvement_pct_per_step"], dtype=float)
            if any(array.shape != (40,) for array in (baseline, full, reduction)):
                raise ValueError(f"Unexpected lead array: {path}")
            if not all(np.isfinite(array).all() for array in (baseline, full, reduction)):
                raise ValueError(f"Nonfinite evaluation: {path}")
            if not np.allclose(reduction, 100 * (1 - full / baseline), atol=1e-10, rtol=1e-10):
                raise ValueError(f"Saved per-lead loss ratios disagree: {path}")
            baseline_rollout = float(loss["baseline_rollout"])
            full_rollout = float(loss["full_rollout"])
            rollout_reduction = float(loss["improvement_pct_rollout"])
            if not np.isclose(rollout_reduction, 100 * (1 - full_rollout / baseline_rollout), atol=1e-10):
                raise ValueError(f"Saved rollout loss ratio disagrees: {path}")
            if not np.allclose([baseline_rollout, full_rollout], [baseline.mean(), full.mean()], atol=1e-10):
                raise ValueError(f"Saved rollout aggregation disagrees: {path}")
            baseline_by_lead.append(baseline)
            sha = digest(path)
            rows.append({
                "k_label": k,
                "ar_tail_k": 0 if k == 1 else k,
                "historical_reference": k == 20,
                "checkpoint": checkpoint,
                "stage_label": label,
                "total_training_updates": 500 if checkpoint == "shared_start" else 2500,
                "parameter_average": checkpoint.startswith("swa"),
                "n_samples": 32,
                "target_steps": 40,
                "baseline_rollout": baseline_rollout,
                "full_rollout": full_rollout,
                "rollout_reduction_pct": rollout_reduction,
                "baseline_day10": float(baseline[-1]),
                "full_day10": float(full[-1]),
                "day10_reduction_pct": float(reduction[-1]),
                "source": relative(path),
                "source_sha256": sha,
            })
            sources.append({"path": relative(path), "sha256": sha})

    configurations = []
    for path in sorted(config_paths):
        config = json.loads(path.read_text())
        configurations.append({
            "path": relative(path), "sha256": digest(path),
            "sequence": config["sequence"], "optimizer": config["optimizer"],
        })
    rollout_baselines = np.asarray([row["baseline_rollout"] for row in rows])
    day10_baselines = np.asarray([row["baseline_day10"] for row in rows])

    def spread(values: np.ndarray) -> dict:
        return {
            "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": float(values.mean()), "range": float(np.ptp(values)),
            "relative_range_pct_of_mean": float(100 * np.ptp(values) / values.mean()),
        }

    baseline_spread = {"rollout": spread(rollout_baselines), "day10": spread(day10_baselines)}
    replay_warning = (
        "K8 SWA rollout reduction is smaller than the relative range of saved baseline "
        "replays. This is a limited baseline-only numerical reference, not a training "
        "seed interval, model repeatability bound or acceptance gate."
    )
    provenance = {
        "schema_version": 1,
        "created_date": "2026-11-13",
        "generator": relative(Path(__file__).resolve()),
        "generator_sha256": digest(Path(__file__).resolve()),
        "sources": sources,
        "configuration_sources": configurations,
        "evaluation": {
            "year": 2022, "cohort": "32 development initializations",
            "chosen_idx": common_origins, "forecast_lead_hours": list(range(6, 241, 6)),
            "mode": "cold full-feedback; zero initial recurrent state",
            "metric": "Exact original GraphCast normalized latitude- and pressure-weighted grid MSE",
            "reduction": "100 * (1 - aggregate full loss / aggregate paired-baseline loss)",
            "baseline_policy": "Each evaluation's own saved paired baseline; no percentages averaged across leads",
        },
        "baseline_replay_spread": baseline_spread,
        "baseline_per_lead_relative_range_pct": (
            100 * np.ptp(baseline_by_lead, axis=0) / np.mean(baseline_by_lead, axis=0)
        ).tolist(),
        "warnings": [replay_warning],
        "interpretation": [
            "K1 is the historical label for ar_tail_k=0, i.e. teacher forcing throughout.",
            "BPTT24 and segment96 are fixed; increasing AR tail changes the teacher-forced prefix.",
            "Each arm independently trains its own first 500 updates at its K; shared_start is not shared weights.",
            "K20 is the historical M1 campaign with the matching two-stage recipe, not part of the later shorter-K submission.",
            "All arms use seed22; no seed robustness or untouched-test inference is established.",
            "Main-stage step2000 means 2500 total updates, including the first-stage500.",
            "SWA averages main-stage500/1000/1500/2000 (total1000/1500/2000/2500); it is not a later update.",
            "Training losses mix teacher-forced and feedback-conditioned steps and are not forecast validation loss.",
            "No matched spectral-versus-K results are present in these source evaluations.",
        ],
        "validation": {
            "evaluations": len(rows), "matched_origin_order": True,
            "all_saved_loss_ratios_checked": True, "all_saved_rollout_aggregations_checked": True,
            "all_values_finite": True,
        },
    }
    return rows, provenance


def plot(rows: list[dict], provenance: dict, image_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#253447", "text.color": "#253447",
        "xtick.color": "#526175", "ytick.color": "#526175",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8.2))
    fig.subplots_adjust(left=.075, right=.975, top=.76, bottom=.33, wspace=.20)
    fig.suptitle("Forecast MSE across AR training tails", fontsize=22, fontweight="bold", y=.96)
    fig.text(.5, .905, "GC–Mamba · 32 matched 2022 development starts · 40 leads through day 10 · one training seed",
             ha="center", fontsize=11.5, color="#526175")
    panels = (("rollout_reduction_pct", "a   Whole-rollout loss", (-14, 20)),
              ("day10_reduction_pct", "b   Day-10 loss", (-14, 29)))
    handles = []
    for ax, (metric, title, limits) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=14)
        ax.axhline(0, color="#58677A", linewidth=1.0, zorder=1)
        ax.axvspan(18.3, 21.4, color="#F1F3F6", zorder=0)
        ax.text(19.85, -8.0, "Historical\nK20 reference", ha="center", va="top",
                fontsize=9, color="#627083")
        ax.grid(axis="y", alpha=.18, linewidth=.8)
        for checkpoint, label, color, marker in STAGES:
            series = [row for row in rows if row["checkpoint"] == checkpoint]
            x = np.asarray([row["ar_tail_k"] for row in series])
            y = np.asarray([row[metric] for row in series])
            line, = ax.plot(x[:-1], y[:-1], color=color, marker=marker, markersize=6.5,
                            linewidth=2.0, label=label, zorder=3)
            ax.plot(x[-2:], y[-2:], color=color, linestyle=(0, (2.5, 3)), linewidth=1.7, zorder=2)
            ax.plot(x[-1], y[-1], color=color, marker=marker, markersize=7.5,
                    markerfacecolor="white", markeredgewidth=1.8, linestyle="none", zorder=4)
            if ax is axes[0]:
                handles.append(line)
            if checkpoint.startswith("swa"):
                for xpos, ypos in zip(x, y):
                    offset = (0, 10) if xpos != 20 else (-8, 8)
                    ax.annotate(f"{ypos:+.2f}%", (xpos, ypos), xytext=offset,
                                textcoords="offset points", ha="center", va="bottom",
                                color=color, fontsize=9, fontweight="bold")
        ax.set_xlim(-1.2, 21.4)
        ax.set_ylim(*limits)
        ax.set_xticks([0, 4, 8, 12, 16, 20], ["TF\n(tail 0)", "4", "8", "12", "16", "20"])
        ax.set_xlabel("Training AR tail (six-hour steps)", labelpad=9)
        ax.set_ylabel("Exact GC weighted-MSE reduction (%)")
        ax.text(.02, .97, "Higher = lower forecast MSE", transform=ax.transAxes,
                fontsize=9.5, va="top", color="#526175")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .865), ncol=3,
               frameon=False, handlelength=2.4, columnspacing=2.3, fontsize=10.5)
    spread = provenance["baseline_replay_spread"]["rollout"]["relative_range_pct_of_mean"]
    fig.text(.075, .195, "Reading the comparison", fontsize=11, fontweight="bold")
    footnotes = (
        "Fixed BPTT24: a longer feedback tail shortens the teacher-forced prefix. TF is the historical “K1” arm.\n"
        "All arms have independent 500-update first stages. K20 uses the matching historical M1 recipe (open markers / dotted lines).\n"
        "SWA averages total updates 1,000 / 1,500 / 2,000 / 2,500; it is not a later training step. Green labels show SWA results.\n"
        f"K8 SWA (+0.087%) is below the observed baseline-only replay range ({spread:.3f}% of mean loss); this is an advisory diagnostic."
    )
    fig.text(.075, .168, footnotes, fontsize=9.3, va="top", linespacing=1.65, color="#526175")
    image_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(image_dir / f"k_dependence.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "plots/analyze_models/data/resolution_eval" / CAMPAIGN)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "plots/analyze_models/images/resolution_eval" / CAMPAIGN)
    args = parser.parse_args()
    rows, provenance = read_evidence()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / "k_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.data_dir / "k_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot(rows, provenance, args.image_dir)
    print(json.dumps({
        "evaluations": len(rows), "data_dir": str(args.data_dir), "image_dir": str(args.image_dir),
        "baseline_replay_spread": provenance["baseline_replay_spread"],
    }, indent=2))


if __name__ == "__main__":
    main()
