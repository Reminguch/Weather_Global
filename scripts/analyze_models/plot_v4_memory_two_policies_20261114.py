#!/usr/bin/env python3
"""Render the matched carry/carry–reset/reset comparison from saved v4 curves.

This is a presentation change only: no forecasts or bootstrap draws are run.
All retained curves and intervals come directly from the November 13 exports.
Historical scripts, figures and data remain unchanged. No Overleaf files are
written by this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "plots/analyze_models/data/resolution_eval/v4_seed_uncertainty_20261113"
RUN = "v4_memory_two_policies_20261114"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / RUN
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / RUN
NAME = "fig6_memory_policies_v4_two_policies"
SEEDS = tuple(range(22, 27))
RAW_MODELS = tuple(f"s{s}_carry_raw2000" for s in SEEDS)
RESET_MODEL = "s22_reset_reset"
LEADS = np.arange(6, 241, 6)
BLUE, GREEN, GRAY = "#0072B2", "#009E73", "#737373"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(SOURCE / name, float_precision="round_trip")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    original_all = read("figure6_original_swa_curves.csv")
    original = original_all.loc[original_all.model_id.isin(["M1", "R_matched"])].copy()
    raw_all = read("figure6_raw_curves.csv")
    raw = raw_all.loc[raw_all.model.isin([*RAW_MODELS, RESET_MODEL])].copy()
    aggregate = read("figure6_raw_carry_mean_sd_curves.csv")
    contrasts_all = read("original_memory_paired_contrasts.csv")
    contrasts = contrasts_all.loc[contrasts_all.contrast_id == "M1_vs_R_matched"].copy()
    endpoints_all = read("raw_memory_endpoints.csv")
    endpoints = endpoints_all.loc[endpoints_all.model.isin([*RAW_MODELS, RESET_MODEL])].copy()

    assert set(original.model_id) == {"M1", "R_matched"}
    assert set(raw.model) == {*RAW_MODELS, RESET_MODEL}
    assert np.array_equal(aggregate.lead_hours, LEADS)
    for _, frame in original.loc[original.endpoint == "lead"].groupby("model_id"):
        assert np.array_equal(frame.sort_values("lead_hours").lead_hours, LEADS)
        assert (frame.initialization_count == 122).all()
        assert np.isfinite(frame[["improvement_pct", "pointwise_ci95_lower", "pointwise_ci95_upper"]]).all().all()
    for _, frame in raw.groupby("model"):
        assert np.array_equal(frame.sort_values("lead_hours").lead_hours, LEADS)
        assert np.isfinite(frame.loss_reduction_pct).all()
    assert (aggregate.training_seed_count == 5).all()
    matrix = raw.loc[raw.model.isin(RAW_MODELS)].pivot(index="lead_hours", columns="model", values="loss_reduction_pct")
    np.testing.assert_allclose(matrix.mean(axis=1), aggregate.loss_reduction_pct_mean, atol=1e-12, rtol=0)
    np.testing.assert_allclose(matrix.std(axis=1, ddof=1), aggregate.loss_reduction_pct_sample_sd, atol=1e-12, rtol=0)
    for frame, key, metric, baseline in [
        (original.loc[original.endpoint == "lead"], "model_id", "model_loss", "baseline_loss"),
        (raw, "model", "model_exact_loss", "public_graphcast_exact_loss"),
    ]:
        score = "improvement_pct" if key == "model_id" else "loss_reduction_pct"
        np.testing.assert_allclose(100 * (1 - frame[metric] / frame[baseline]), frame[score], atol=1e-10, rtol=0)
        bases = frame.pivot(index="lead_hours", columns=key, values=baseline).to_numpy()
        np.testing.assert_array_equal(bases, np.repeat(bases[:, :1], bases.shape[1], axis=1))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 10,
                         "legend.fontsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "lines.linewidth": 1.6})
    fig, axes = plt.subplots(1, 2, figsize=(6, 4.05), sharey=True)
    fig.subplots_adjust(left=.115, right=.985, bottom=.29, top=.84, wspace=.21)
    for model, label, color, linestyle in [("M1", "Carry/carry", BLUE, "-"), ("R_matched", "Reset/reset", GREEN, "-.")]:
        frame = original.loc[(original.model_id == model) & (original.endpoint == "lead")].sort_values("lead_hours")
        x = frame.lead_hours.to_numpy() / 24
        axes[0].fill_between(x, frame.pointwise_ci95_lower, frame.pointwise_ci95_upper, color=color, alpha=.12, lw=0)
        axes[0].plot(x, frame.improvement_pct, color=color, ls=linestyle, label=label)
    for model in RAW_MODELS:
        frame = raw.loc[raw.model == model].sort_values("lead_hours")
        axes[1].plot(frame.lead_hours / 24, frame.loss_reduction_pct, color=BLUE, lw=.7, alpha=.22)
    x = aggregate.lead_hours.to_numpy() / 24
    mean = aggregate.loss_reduction_pct_mean.to_numpy()
    sd = aggregate.loss_reduction_pct_sample_sd.to_numpy()
    axes[1].fill_between(x, mean - sd, mean + sd, color=BLUE, alpha=.16, lw=0)
    axes[1].plot(x, mean, color=BLUE, lw=2, label="Carry/carry (5 seeds)")
    reset = raw.loc[raw.model == RESET_MODEL].sort_values("lead_hours")
    axes[1].plot(reset.lead_hours / 24, reset.loss_reduction_pct, color=GREEN, ls="-.", label="Reset/reset (1 seed)")
    axes[0].set_title("(a) Original SWA", loc="left", pad=17)
    axes[1].set_title("(b) Independent replication\n(raw checkpoints)", loc="left", pad=8)
    axes[0].set_ylabel("GraphCast loss reduction (%)")
    lead_original = original.loc[original.endpoint == "lead"]
    lower = min(float(lead_original.pointwise_ci95_lower.min()), float(raw.loss_reduction_pct.min()), float((mean - sd).min()))
    upper = max(float(lead_original.pointwise_ci95_upper.max()), float(raw.loss_reduction_pct.max()), float((mean + sd).max()))
    limits = [min(-2, float(np.floor(lower - .5))), max(27, float(np.ceil(upper + .5)))]
    for ax in axes:
        ax.set(xlim=(0, 10), xticks=[0, 2, 4, 6, 8, 10], xlabel="Forecast lead (days)", ylim=limits)
        ax.grid(alpha=.18, linewidth=.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color=GRAY, lw=.7)
        ax.legend(loc="upper left", frameon=False, fontsize=10, borderaxespad=.1, labelspacing=.25, handlelength=1.8)
    fig.text(.33, .14, "Bands: pointwise 95% date CI", ha="center", fontsize=10)
    fig.text(.765, .14, "Band: ±1 SD over carry seeds", ha="center", fontsize=10)
    fig.text(.54, .073, "Each panel uses one common GraphCast baseline;", ha="center", fontsize=10)
    fig.text(.54, .035, "the replication uses repaired verification fields.", ha="center", fontsize=10)
    figure_paths = []
    for suffix in ("pdf", "png"):
        path = IMAGES / f"{NAME}.{suffix}"
        fig.savefig(path, dpi=240, facecolor="white")
        figure_paths.append(path)
    plt.close(fig)

    tables = {"figure6_original_swa_curves.csv": original,
              "figure6_raw_curves.csv": raw,
              "original_memory_paired_contrasts.csv": contrasts,
              "raw_memory_endpoints.csv": endpoints}
    for name, frame in tables.items():
        path = DATA / name
        frame.to_csv(path, index=False, lineterminator="\n")
        reloaded = pd.read_csv(path, float_precision="round_trip")
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), reloaded, check_exact=True)
    # This saved aggregate table is copied byte-for-byte rather than recomputed.
    aggregate_name = "figure6_raw_carry_mean_sd_curves.csv"
    shutil.copy2(SOURCE / aggregate_name, DATA / aggregate_name)
    source_paths = [SOURCE / name for name in [*tables, aggregate_name, "summary.json"]]
    source_paths += [Path(__file__), ROOT / "scripts/analyze_models/plot_v4_seed_replication.py"]
    manifest = {
        "date": "2026-11-14", "purpose": "Figure6 restricted to matched carry/carry and reset/reset policies",
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "figures": {str(p.relative_to(ROOT)): digest(p) for p in figure_paths},
        "tables": {p.name: digest(p) for p in sorted(DATA.glob("*.csv"))},
        "figure_size_inches": [6, 4.05], "y_limits_pct": limits,
        "original_policy_ids": ["M1", "R_matched"], "replication_model_ids": [*RAW_MODELS, RESET_MODEL],
        "forecast_dates_per_model": 122, "lead_count": 40,
        "checks": {"retained_curve_exports_exact": True, "saved_seed_aggregate_copied_byte_for_byte": True,
                   "original_confidence_intervals_reused": True, "common_baseline_within_each_panel": True,
                   "retained_five_seed_statistics_verified": True},
        "scope": ["Presentation-only subset of the existing validated Figure6 exports.",
                  "Original SWA panel retains paired pointwise 95% date-block-bootstrap intervals.",
                  "Replication panel retains five raw carry seeds, mean and sample SD, versus one raw reset seed.",
                  "The two panels retain distinct checkpoints, environments and verification provenance.",
                  "No training, forecasts, resampling, or numerical result recomputation; no Overleaf files modified."]}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": [str(p.relative_to(ROOT)) for p in figure_paths], "data": str(DATA.relative_to(ROOT)), "y_limits_pct": limits}, indent=2))


if __name__ == "__main__":
    main()
