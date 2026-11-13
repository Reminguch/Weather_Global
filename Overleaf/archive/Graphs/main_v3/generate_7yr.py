#!/usr/bin/env python3
"""Rebuild main_v3 seven-year variable figure from saved RMSE evaluations.

Run from the repository root:
    source scripts/graphcast_env.sh
    python Overleaf/Graphs/main_v3/generate_7yr.py

This script does not evaluate any checkpoint or reconstruct GraphCast loss.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/main_v3_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from paper_style import ANNOTATION_SIZE, LABEL_SIZE, apply_style
SOURCES = {
    "development_2022": ROOT / "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json",
    "pilot_2023": ROOT / "plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/models/M1/evaluation.json",
}
VARIABLES = (
    ("geopotential", "Geopotential"),
    ("temperature", "Temperature"),
    ("specific_humidity", "Specific humidity"),
    ("u_component_of_wind", "Zonal wind"),
    ("v_component_of_wind", "Meridional wind"),
    ("vertical_velocity", "Pressure vertical velocity"),
    ("10m_u_component_of_wind", "10 m zonal wind"),
    ("10m_v_component_of_wind", "10 m meridional wind"),
    ("2m_temperature", "2 m temperature"),
    ("mean_sea_level_pressure", "Mean sea-level pressure"),
    ("total_precipitation_6hr", "6 h accumulated precipitation"),
)
DAYS = np.arange(1, 41) / 4
apply_style()


def reduction(metrics):
    baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
    model = np.asarray(metrics["rmse_full"], dtype=float)
    assert baseline.shape == model.shape == (40,)
    assert np.isfinite(baseline).all() and (baseline > 0).all()
    assert np.isfinite(model).all() and (model >= 0).all()
    values = 100 * (1 - model / baseline)
    if "improvement_pct_rmse" in metrics:
        np.testing.assert_allclose(values, metrics["improvement_pct_rmse"], atol=1e-10)
    return values


def validate(doc, expected_samples):
    assert doc["evaluated_samples"] == expected_samples and doc["target_steps"] == 40
    assert len(doc["chosen_idx"]) == expected_samples
    assert doc["warmup_steps"] == 0 and doc["eval_mode"] == "cold_full"
    assert doc["residual_state_init"] == "zero"
    assert doc["eval_feedback"] == "full" and not doc["rs_reset_every_step"]
    assert len(doc["per_channel_per_step"]) == 83
    assert len(doc["per_variable_per_step"]) == 11
    for metrics in doc["per_channel_per_step"].values():
        reduction(metrics)
    for key, metrics in doc["per_variable_per_step"].items():
        reduction(metrics)
        channels = [c for k, c in doc["per_channel_per_step"].items()
                    if k == key or k.startswith(key + "_level")]
        assert len(channels) in (1, 13)
        for field in ("rmse_baseline", "rmse_full"):
            pooled = np.sqrt(np.mean([np.square(c[field]) for c in channels], axis=0))
            # Aggregated fields have float32 reduction noise; agreement confirms
            # equal-level MSE pooling rather than a mean of channel percentages.
            np.testing.assert_allclose(pooled, metrics[field], rtol=2e-7, atol=0)


def save(fig, stem, description):
    fig.savefig(HERE / f"{stem}.pdf", metadata={
        "Title": stem.replace("_", " "), "Subject": description,
        "Creator": "main_v3/generate_7yr.py; saved evaluation artifacts only",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(HERE / f"{stem}.png", dpi=190)
    plt.close(fig)


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lead_axes(ax):
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Forecast lead time (days)")
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)


def overview_panels(matrix, counts):
    """Compact, separately captioned panels for the opening rollout overview."""
    fig = plt.figure(figsize=(6.5, 3.2))
    ax = fig.add_axes([.335, .17, .515, .72])
    heatmap = matrix[:, [19, 23, 27, 31, 35, 39]]
    im = ax.imshow(heatmap, cmap="YlGnBu", vmin=0, vmax=20, aspect="auto")
    ax.set_yticks(range(11), [label for _, label in VARIABLES])
    ax.set_xticks(range(6), ["5", "6", "7", "8", "9", "10"])
    ax.set_xlabel("Forecast lead time (days)", labelpad=3)
    ax.set_title("RMSE reduction relative to GraphCast", loc="right", pad=9)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), value in np.ndenumerate(heatmap):
        ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=ANNOTATION_SIZE,
                color="white" if value >= 12 else "#202020")
    cb = fig.colorbar(im, cax=fig.add_axes([.875, .17, .016, .72]),
                      ticks=[0, 5, 10, 15, 20])
    cb.set_label("RMSE reduction (%)", labelpad=4, fontsize=LABEL_SIZE)
    cb.outline.set_visible(False)
    save(fig, "overview_7yr_heatmap", "Figure 1 variable-group panel; same values as breadth_7yr. "
         "V24 versus paired GraphCast-small, 32 development starts in 2022.")

    fig = plt.figure(figsize=(6.5, 1.3))
    ax = fig.add_axes([.12, .40, .785, .54])
    ax.plot(DAYS, counts, color="#0072B2", linewidth=1.65)
    ax.axhline(83, color=".5", linewidth=.8, linestyle="--")
    ax.scatter([10], [82], s=16, color="#0072B2", zorder=3)
    ax.annotate("82/83 at day 10", xy=(10, 82), xytext=(6.2, 24),
                fontsize=ANNOTATION_SIZE, arrowprops={"arrowstyle": "-", "lw": .8, "color": ".4"})
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 40, 83])
    ax.set_ylabel("Improved\nchannels")
    lead_axes(ax)
    ax.set_xlabel("Forecast lead time (days)", labelpad=2)
    save(fig, "overview_7yr_coverage", "Figure 1 channel-coverage panel; same counts as breadth_7yr. "
         "Channels with strictly lower RMSE than paired GraphCast-small; no significance test.")


def breadth(doc):
    matrix = np.array([reduction(doc["per_variable_per_step"][k]) for k, _ in VARIABLES])
    channel_values = np.array([reduction(m) for m in doc["per_channel_per_step"].values()])
    counts = (channel_values > 0).sum(axis=0)
    assert (matrix[:, 19:] > 0).all(), "All groups must improve at 120--240 h"
    assert counts[19] == 78 and counts[27] == 81 and counts[39] == 82
    assert np.array_equal(np.flatnonzero(counts == 83), [35, 36, 37])
    exceptions = {key: float(reduction(metrics)[-1])
                  for key, metrics in doc["per_channel_per_step"].items()
                  if reduction(metrics)[-1] <= 0}
    assert list(exceptions) == ["geopotential_level50"]
    overview_panels(matrix, counts)
    fig = plt.figure(figsize=(6.5, 5.7))
    ax = fig.add_axes([.335, .395, .515, .525])
    heatmap = matrix[:, [19, 23, 27, 31, 35, 39]]
    im = ax.imshow(heatmap, cmap="YlGnBu", vmin=0, vmax=20, aspect="auto")
    ax.set_yticks(range(11), [label for _, label in VARIABLES])
    ax.set_xticks(range(6), ["5", "6", "7", "8", "9", "10"])
    ax.set_xlabel("Forecast lead time (days)", labelpad=4)
    ax.set_title("(a) RMSE reduction by variable group", loc="right", pad=9)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), value in np.ndenumerate(heatmap):
        ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=ANNOTATION_SIZE,
                color="white" if value >= 12 else "#202020")
    cax = fig.add_axes([.875, .395, .016, .525])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 5, 10, 15, 20])
    cb.set_label("RMSE reduction (%)", labelpad=4, fontsize=LABEL_SIZE)
    cb.outline.set_visible(False)
    bottom = fig.add_axes([.12, .105, .785, .16])
    bottom.plot(DAYS, counts, color="#0072B2", linewidth=1.65)
    bottom.axhline(83, color=".5", linewidth=.8, linestyle="--")
    bottom.scatter([10], [82], s=16, color="#0072B2", zorder=3)
    bottom.annotate("82/83 at day 10", xy=(10, 82), xytext=(6.2, 42),
                    fontsize=ANNOTATION_SIZE, arrowprops={"arrowstyle": "-", "lw": .8, "color": ".4"})
    bottom.set_ylim(0, 90)
    bottom.set_yticks([0, 40, 83])
    bottom.set_ylabel("Improved\nchannels")
    bottom.set_title("(b) Coverage across all 83 channels", loc="left", pad=7)
    lead_axes(bottom)
    save(fig, "breadth_7yr", "Seven-year V24 correction (2015--2021), di16, SWA500--2000; 32 development starts. "
         "Paired RMSE reduction: equal-pressure-level MSE pooling for groups; "
         "each channel is evaluated separately for the positive-reduction count.")
    write_csv("breadth_7yr_all_leads.csv", [
        {"variable": key, "lead_hours": (i + 1) * 6, "lead_days": day,
         "rmse_baseline": doc["per_variable_per_step"][key]["rmse_baseline"][i],
         "rmse_model": doc["per_variable_per_step"][key]["rmse_full"][i],
         "rmse_reduction_pct": matrix[v, i]}
        for v, (key, _) in enumerate(VARIABLES) for i, day in enumerate(DAYS)
    ])
    write_csv("breadth_7yr_channels.csv", [
        {"channel": key, "lead_hours": (i + 1) * 6, "lead_days": day,
         "rmse_baseline": metrics["rmse_baseline"][i],
         "rmse_model": metrics["rmse_full"][i], "rmse_reduction_pct": value}
        for key, metrics in doc["per_channel_per_step"].items()
        for i, (day, value) in enumerate(zip(DAYS, reduction(metrics)))
    ])
    write_csv("breadth_7yr_channel_counts.csv", [
        {"lead_hours": (i + 1) * 6, "lead_days": day,
         "improved_channels": int(count), "total_channels": 83}
        for i, (day, count) in enumerate(zip(DAYS, counts))
    ])
    return {"all_groups_improve_from_120h": True,
            "all_channels_improve_at_hours": ((np.flatnonzero(counts == 83) + 1) * 6).tolist(),
            "channel_counts": counts.tolist(), "day10_nonimproving_channels": exceptions,
            "day5_group_reduction_range_pct": [float(matrix[:, 19].min()), float(matrix[:, 19].max())],
            "day10_group_reduction_range_pct": [float(matrix[:, -1].min()), float(matrix[:, -1].max())]}


def main():
    docs = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    validate(docs["development_2022"], 32)
    validate(docs["pilot_2023"], 8)
    assert docs["development_2022"]["ckpt"] == docs["pilot_2023"]["ckpt"]
    assert Path(docs["development_2022"]["ckpt_in"]).resolve() == Path(docs["pilot_2023"]["ckpt_in"]).resolve()
    findings = breadth(docs["development_2022"])
    day10 = [{"variable": key, "label": label,
              **{name: float(reduction(doc["per_variable_per_step"][key])[-1])
                 for name, doc in docs.items()}}
             for key, label in VARIABLES]
    write_csv("seven_year_day10.csv", day10)
    training_script = ROOT / "scripts/experiments/run_v24_res1_batch4_lr.py"
    training_config = ROOT / "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/run_config.json"
    provenance = {
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formula": "100*(1-rmse_full/rmse_baseline)",
        "metric": "cosine-latitude-weighted RMSE reduction; not GraphCast loss",
        "group_aggregation": "mean of physical-unit squared errors over starts and equally over pressure levels, then square root",
        "correction_training_years": [2015, 2021],
        "training_sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (training_script, training_config)},
        "sources": {name: {"path": str(path.relative_to(ROOT)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "ckpt": docs[name]["ckpt"], "chosen_idx": docs[name]["chosen_idx"],
                           "evaluated_samples": docs[name]["evaluated_samples"]}
                    for name, path in SOURCES.items()},
        "findings": findings, "day10_variable_rmse_reduction_pct": day10,
        "uncertainty": "Descriptive aggregate scores; no confidence intervals or significance claims. Development starts were used for model selection; the pilot contains eight dates.",
    }
    (HERE / "provenance_7yr.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
