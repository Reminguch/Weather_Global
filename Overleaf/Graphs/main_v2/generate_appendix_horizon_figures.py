#!/usr/bin/env python3
"""Rebuild the historical v22 appendix figures from archived evaluation JSONs.

From the repository root:
    source scripts/graphcast_env.sh
    python Overleaf/Graphs/main_v2/generate_appendix_horizon_figures.py

The horizon sweep uses the historical, approximate paper-formula weighted MSE
reconstruction. Aggregated RMSE summaries cannot recover the exact original
GraphCast loss. No forecasts, training runs, or exact-loss evaluations are run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/main_v2_appendix_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from paper_style import ANNOTATION_SIZE, apply_style

EVAL_DIR = ROOT / "results/v22/eval_jsons"
STATS = ROOT / "data/graphcast/graphcast/stats/diffs_stddev_by_level.nc"
ARCHIVED_PLOT = ROOT / "results/2026-05-23-v22/plots/v22_vs_baseline.png"
FORMULA_SOURCES = (
    ROOT / "scripts/training/full_mamba_v23/plot_v22cl_res2_cold_full.py",
    ROOT / "scripts/analyze_models/analyze_v22_final_state_evals_20260815.py",
)
KS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
VARIABLE_WEIGHTS = {
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


def source_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def metric_array(metrics: dict, field: str) -> np.ndarray:
    values = np.asarray(metrics[field], dtype=float)
    if values.shape != (40,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected 40 finite values in {field}")
    return values


def historical_loss(document: dict, sigmas: dict[str, float]) -> dict:
    """Reproduce the variable-level formula used by archived K-sweep plots."""
    variables = document["per_variable_per_step"]
    if set(variables) != set(VARIABLE_WEIGHTS):
        raise ValueError("Historical metric requires all 11 variable groups")
    totals = {"baseline": np.zeros(40), "full": np.zeros(40)}
    for variable, weight in VARIABLE_WEIGHTS.items():
        for branch in totals:
            rmse = metric_array(variables[variable], f"rmse_{branch}")
            totals[branch] += weight * rmse**2 / sigmas[variable]**2
    if np.any(totals["baseline"] <= 0):
        raise ValueError("Nonpositive baseline denominator")
    totals["reduction_pct"] = 100 * (1 - totals["full"] / totals["baseline"])
    return {name: values.tolist() for name, values in totals.items()}


def format_leads(ax):
    ax.set_xlim(0, 10.15)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Forecast lead time (days)")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, stem: str, description: str, preview_dir: Path | None):
    fig.savefig(HERE / f"{stem}.pdf", metadata={
        "Title": stem.replace("_", " "),
        "Subject": description,
        "Creator": Path(__file__).name + "; archived evaluations only",
        "CreationDate": None,
        "ModDate": None,
    })
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{stem}.png", dpi=180)
    plt.close(fig)


def horizon_figure(records, preview_dir=None):
    """Draw the archived curves without changing their historical metric."""
    training_tail_k, step_hours = 20, 6
    training_tail_hours = training_tail_k * step_hours
    training_tail_days = training_tail_hours / 24
    days = np.arange(1, 41) / 4
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.18, top=0.90)
    cmap, norm = plt.colormaps["viridis"], plt.Normalize(1, 22)
    for k in KS:
        ax.plot(days, records[str(k)]["reduction_pct"],
                color=cmap(norm(k)), linewidth=1.35, marker="o", markersize=2.3,
                label=rf"$K={k}$")
    ax.axhline(0, color="0.45", linewidth=0.7)
    ax.set_ylim(-12, 28)
    ax.set_yticks(np.arange(-10, 26, 5))
    ax.set_ylabel("GraphCast loss reduction (%)")
    ax.set_title("Training tail label K", loc="left")
    ax.axvline(training_tail_days, color="0.25", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(training_tail_days, .97, r"$K=20$: 120 h (5 days)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=ANNOTATION_SIZE, color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.5})
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, .83),
              framealpha=0.95, borderpad=0.4,
              handlelength=2, columnspacing=1.4, labelspacing=0.2)
    format_leads(ax)
    ax.set_xlim(0, 10)
    save(fig, "training_ar_horizon",
         "Archived v22 K=1..22, 32 samples, 40 six-hour leads. Approximate "
         "paper-formula weighted MSE reduction reconstructed from variable-level "
         "latitude-weighted RMSE with mean-level difference scales. "
         "This is not the exact original GraphCast loss. Vertical dashed line: "
         "K=20 training tail, 20 six-hour steps = 120 h = 5 days.", preview_dir)
    (HERE / "training_ar_horizon.reference.json").write_text(json.dumps({
        "reference_type": "vertical training-tail duration marker",
        "ar_tail_K": training_tail_k,
        "step_hours": step_hours,
        "lead_hours": training_tail_hours,
        "lead_days": training_tail_days,
        "display_label": "K=20: 120 h (5 days)",
        "shared_y_limits": [-12, 28],
        "interpretation": "Marks the forecast-feedback training tail for K=20, not the full training window or memory history. Other K curves have their own tail durations.",
    }, indent=2) + "\n")


def main():
    import xarray as xr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    apply_style()
    with xr.open_dataset(STATS) as dataset:
        sigmas = {
            variable: float(dataset[variable].values.mean())
            for variable in VARIABLE_WEIGHTS
        }
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in sigmas.values()):
        raise ValueError("Invalid difference standard deviation")

    documents, records = {}, {}
    for k in KS:
        path = EVAL_DIR / f"v22_K{k}_K40.json"
        document = json.loads(path.read_text())
        if document["target_steps"] != 40 or document["n_samples"] != 32:
            raise ValueError(f"Unexpected evaluation size for K={k}")
        documents[k] = document
        records[str(k)] = {
            **source_record(path),
            "checkpoint_recorded_in_json": document["ckpt"],
            "n_samples": document["n_samples"],
            **historical_loss(document, sigmas),
        }

    # The archive lacks forecast-start indices, and the baseline summaries
    # differ slightly across K. Preserve this limitation in the provenance.
    reference = documents[22]["per_channel_per_step"]
    baseline_difference = {}
    for k, document in documents.items():
        differences = []
        for channel, metrics in document["per_channel_per_step"].items():
            baseline = metric_array(metrics, "rmse_baseline")
            ref = metric_array(reference[channel], "rmse_baseline")
            differences.append(float(np.max(np.abs(baseline / ref - 1))))
        baseline_difference[str(k)] = max(differences)

    horizon_figure(records, args.preview_dir)

    days = np.arange(1, 41) / 4

    k22 = documents[22]
    channels = k22["per_channel_per_step"]
    if len(channels) != 83:
        raise ValueError(f"Expected 83 channels, found {len(channels)}")
    positive = []
    for metrics in channels.values():
        baseline = metric_array(metrics, "rmse_baseline")
        model = metric_array(metrics, "rmse_full")
        reduction = metric_array(metrics, "improvement_pct_rmse")
        np.testing.assert_allclose(100 * (1 - model / baseline), reduction,
                                   atol=1e-10, rtol=1e-10)
        positive.append(model < baseline)
    counts = np.stack(positive).sum(axis=0)
    variable_day10 = {}
    for variable, metrics in k22["per_variable_per_step"].items():
        baseline = metric_array(metrics, "rmse_baseline")
        model = metric_array(metrics, "rmse_full")
        reduction = metric_array(metrics, "improvement_pct_rmse")
        np.testing.assert_allclose(100 * (1 - model / baseline), reduction,
                                   atol=1e-10, rtol=1e-10)
        variable_day10[variable] = float(reduction[-1])

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.19, top=0.88)
    ax.axhline(83, color="0.5", linewidth=0.8, linestyle="--", label="All 83 channels")
    ax.plot(days, counts, color="#0072B2", marker="o", markersize=2.8, linewidth=1.6)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Channels with lower RMSE (of 83)")
    ax.set_title(r"Channel coverage ($K=22$)", loc="left")
    ax.annotate("81 / 83", xy=(4, counts[15]), xytext=(3.5, 64), fontsize=ANNOTATION_SIZE,
                arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.7})
    ax.annotate("82 / 83", xy=(10, counts[-1]), xytext=(9.0, 66), fontsize=ANNOTATION_SIZE,
                arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.7})
    ax.legend(loc="lower right", frameon=False)
    format_leads(ax)
    save(fig, "channels_improved_k22",
         "Archived v22 K=22, 32 samples, 40 six-hour leads. Number of the 83 "
         "variable-level channels with strictly lower latitude-weighted RMSE "
         "than their paired GraphCast baseline; includes precipitation. "
         "81/83 at day four and 82/83 at day ten.", args.preview_dir)

    provenance = {
        "generator": source_record(Path(__file__).resolve()),
        "archived_plot_for_visual_comparison": source_record(ARCHIVED_PLOT),
        "historical_formula_sources": [source_record(path) for path in FORMULA_SOURCES],
        "difference_standard_deviation_source": source_record(STATS),
        "metric": {
            "name": "approximate paper-formula weighted MSE reduction",
            "formula": "100 * (1 - sum_v(w_v * RMSE_full_v**2 / sigma_v**2) / sum_v(w_v * RMSE_baseline_v**2 / sigma_v**2))",
            "sigma_definition": "Arithmetic mean of saved difference standard deviations across levels; scalar for surface variables.",
            "variable_weights": VARIABLE_WEIGHTS,
            "difference_scales": sigmas,
            "exact_original_graphcast_loss": False,
            "limitations": "Historical variable-level RMSE aggregation and mean-level scales differ from the exact GraphCast loss; exact latitude treatment and original reduction order cannot be recovered from these archived summaries.",
        },
        "lead_hours": (np.arange(1, 41) * 6).tolist(),
        "baseline_max_relative_difference_from_K22_by_K": baseline_difference,
        "comparison_note": "Each curve uses its own saved baseline; baseline summaries differ slightly across K and forecast-start indices are absent, so matching start indices cannot be verified.",
        "forecast_start_indices_available": False,
        "evaluations": records,
        "k22_channel_diagnostic": {
            "definition": "Count of variable-level channels with strictly lower saved latitude-weighted RMSE than paired baseline, including precipitation.",
            "total_channels": len(channels),
            "channels_improved_per_lead": counts.tolist(),
            "day10_variable_group_rmse_reduction_pct": variable_day10,
            "day10_groups_improved": sum(value > 0 for value in variable_day10.values()),
        },
        "outputs": ["training_ar_horizon.pdf", "channels_improved_k22.pdf"],
    }
    (HERE / "appendix_horizon_sources.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print("Wrote training_ar_horizon.pdf, channels_improved_k22.pdf, appendix_horizon_sources.json")
    print("Day-10 historical reduction:",
          {k: round(records[str(k)]["reduction_pct"][-1], 6) for k in (16, 18, 20, 22)})
    print("K22 channels improved at 96h / 240h:", int(counts[15]), int(counts[-1]))
    print("K22 day-10 variable-group RMSE reductions:",
          round(min(variable_day10.values()), 6), "to", round(max(variable_day10.values()), 6))


if __name__ == "__main__":
    main()
