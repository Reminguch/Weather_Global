#!/usr/bin/env python3
"""Rebuild the main_v2 figures from saved evaluations; never run evaluation.

From the repository root:
    source scripts/graphcast_env.sh
    python Overleaf/Graphs/main_v2/generate_figures.py
Optional PNG previews go to the directory passed with --preview-dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/main_v2_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT = ROOT / "docs/reports/v24_optimizer_search_2026-09-10.sources.json"
OPT_CSV = ROOT / "docs/reports/v24_optimizer_search_2026-09-10.csv"
LR_REPORT = ROOT / "docs/reports/v24_mamba_lr_di_10day_2026-09-14.sources.json"
LR_CSV = ROOT / "docs/reports/v24_mamba_lr_di_10day_2026-09-14.csv"
MEM_CSV = ROOT / "plots/analyze_models/data/resolution_eval/res1_reset_training_gc_loss_approx/curves.csv"
OPT_ROOT = ROOT / "artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904"
REF = ROOT / "artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json"
OPT = OPT_ROOT / "mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k/eval/cold_full_zero_exact_gc/matched_v22_reference32/swa_step02000-08000.json"
BEST = ROOT / "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json"
POLICIES = (
    ("Mamba: carry train / carry eval", "Carry / carry", "#0072B2", "-"),
    ("Mamba: carry train / reset eval", "Carry / reset", "#D55E00", "--"),
    ("Mamba: reset train / reset eval", "Reset / reset", "#009E73", "-."),
)
WINDOWS = ("swa_step02000-08000", "swa_step04000-10000", "swa_step06000-12000")
WINDOW_LABELS = ("SWA 2k–8k", "SWA 4k–10k", "SWA 6k–12k")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def summary(document):
    loss = document["original_graphcast_loss"]
    return {
        "evaluated_samples": document["evaluated_samples"],
        "chosen_idx": document["chosen_idx"],
        "eval_mode": document["eval_mode"],
        "residual_state_init": document["residual_state_init"],
        "baseline_rollout": loss["baseline_rollout"],
        "full_rollout": loss["full_rollout"],
        "rollout_loss_reduction_pct": loss["improvement_pct_rollout"],
        "day10_loss_reduction_pct": loss["improvement_pct_per_step"][-1],
    }


def verify_exact(document, indices):
    assert document["evaluated_samples"] == 32
    assert document["target_steps"] == 40
    assert document["resolution"] == 1
    assert document["eval_mode"] == "cold_full"
    assert document["residual_state_init"] == "zero"
    assert document["chosen_idx"] == indices
    loss = document["original_graphcast_loss"]
    baseline = np.asarray(loss["baseline_per_step"])
    full = np.asarray(loss["full_per_step"])
    assert baseline.shape == full.shape == (40,)
    np.testing.assert_allclose(100 * (1 - full / baseline),
                               loss["improvement_pct_per_step"], atol=1e-10)
    np.testing.assert_allclose(baseline.mean(), loss["baseline_rollout"], atol=1e-10)
    np.testing.assert_allclose(full.mean(), loss["full_rollout"], atol=1e-10)
    np.testing.assert_allclose(100 * (1 - full.mean() / baseline.mean()),
                               loss["improvement_pct_rollout"], atol=1e-10)


def temperature_metrics(document):
    metrics = document["per_variable_per_step"]["2m_temperature"]
    baseline = np.asarray(metrics["rmse_baseline"])
    full = np.asarray(metrics["rmse_full"])
    reduction = np.asarray(metrics["improvement_pct_rmse"])
    assert baseline.shape == full.shape == reduction.shape == (40,)
    assert np.isfinite(baseline).all() and (baseline > 0).all()
    assert np.isfinite(full).all() and (full >= 0).all()
    np.testing.assert_allclose(100 * (1 - full / baseline), reduction, atol=1e-10)
    return metrics


def save(fig, stem, description, preview_dir):
    fig.savefig(HERE / f"{stem}.pdf", metadata={
        "Title": stem.replace("_", " "), "Subject": description,
        "Creator": "generate_figures.py; saved evaluation artifacts only",
        "CreationDate": None, "ModDate": None,
    })
    if preview_dir:
        fig.savefig(preview_dir / f"{stem}.png", dpi=170)
    plt.close(fig)


def format_leads(ax):
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Forecast lead (days)")
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)


def exact_figure(best, preview_dir):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.15))
    fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.935,
                        wspace=.35, hspace=.52)
    days = np.arange(1, 41) / 4
    loss = best["original_graphcast_loss"]
    temperature = temperature_metrics(best)
    for ax, baseline in ((axes[0, 0], loss["baseline_per_step"]),
                         (axes[1, 0], temperature["rmse_baseline"])):
        ax.plot(days, baseline, color=".45", linestyle="--", linewidth=1.5,
                label="GraphCast baseline")
    curves = (loss["full_per_step"], loss["improvement_pct_per_step"],
              temperature["rmse_full"], temperature["improvement_pct_rmse"])
    for ax, values in zip(axes.flat, curves):
        ax.plot(days, values, color="#0072B2", linewidth=1.6, label="3× Mamba LR + SWA")
    axes[0, 0].set_ylabel("Exact GraphCast loss")
    axes[0, 1].set_ylabel("Exact loss reduction (%)")
    axes[1, 0].set_ylabel("2 m temperature RMSE (K)")
    axes[1, 1].set_ylabel("RMSE reduction (%)")
    for ax in axes[:, 0]:
        ax.set_ylim(bottom=0)
    axes[0, 1].set_ylim(-2, 26)
    for ax in axes[:, 1]:
        ax.axhline(0, color=".5", linewidth=.8)
    titles = ("(a) Forecast error", "(b) Loss reduction",
              "(c) 2 m temperature error", "(d) RMSE reduction")
    for ax, title in zip(axes.flat, titles):
        format_leads(ax)
        ax.set_title(title, loc="left")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(.5, .04),
               handlelength=2, columnspacing=1.2)
    fig.text(.5, .014, r"Mamba: $d_{\mathrm{inner}}=16$, one $B,C$ group; SWA steps 500–2000",
             ha="center", va="bottom", fontsize=7.5)
    save(fig, "exact_gc_loss", "32 matched starts, cold_full evaluation, zero initial state; "
         "40 six-hour steps. Threefold Mamba learning rate with SWA, inner dimension 16, "
         "one B,C group, SWA steps 500–2000; "
         "reduction uses its saved paired baseline. "
         "Top: exact GraphCast loss. Bottom: 2 m temperature RMSE in kelvin and RMSE reduction. "
         "Gray dashed curves show the selected model's paired GraphCast baseline in the absolute-error panels.", preview_dir)


def memory_figure(rows, full, preview_dir):
    if full:
        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.35), sharex=True, sharey=True)
        fig.subplots_adjust(left=.1, right=.985, bottom=.22, top=.92, wspace=.12, hspace=.28)
        panels = [(axes[r, c], bcg, window, f"({chr(97 + 3*r+c)}) BCG = {bcg}, {WINDOW_LABELS[c]}")
                  for r, bcg in enumerate((1, 4)) for c, window in enumerate(WINDOWS)]
    else:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        fig.subplots_adjust(left=.105, right=.98, bottom=.29, top=.88)
        panels = [(ax, 1, WINDOWS[0], r"Temporal memory ablation ($d_{\mathrm{inner}}=16$)")]
    for ax, bcg, window, title in panels:
        for policy, label, color, linestyle in POLICIES:
            subset = sorted((r for r in rows if int(r["bcg"]) == bcg and
                             r["swa"] == window and r["policy"] == policy),
                            key=lambda r: float(r["lead_days"]))
            assert len(subset) == 40
            days = [float(r["lead_days"]) for r in subset]
            np.testing.assert_allclose(days, np.arange(1, 41) / 4)
            ax.plot(days, [float(r["approximate_gc_loss_reduction_pct"]) for r in subset],
                    color=color, linestyle=linestyle, linewidth=1.6, label=label)
        format_leads(ax)
        ax.axhline(0, color=".5", linewidth=.8)
        ax.set_ylim(-42 if full else -22, 26)
        ax.set_title(title, loc="left", fontsize=8.5 if full else 10)
    if full:
        for ax in axes[0]:
            ax.set_xlabel("")
        fig.supylabel("Approx. GraphCast loss reduction (%)", x=.015, y=.57, fontsize=9)
    else:
        panels[0][0].set_ylabel("Approx. GraphCast loss\nreduction (%)")
    fig.legend(*panels[0][0].get_legend_handles_labels(), loc="lower center",
               ncol=3, frameon=False, title="Training state / evaluation state",
               title_fontsize=8, bbox_to_anchor=(.5, .01), columnspacing=2)
    stem = "memory_reset_all" if full else "memory_reset"
    save(fig, stem, "32 matched starts; cold_full, zero initial state. "
         "Approximate loss reconstructed from saved channel RMSE, not exact GraphCast loss. "
         "Carry/carry, carry/reset and reset/reset policies. " +
         ("Three correlated SWA windows, not independent training repeats." if full else
          "Inner dimension 16, one B,C group, SWA 2k–8k."), preview_dir)


def optimizer_figure(rows, preview_dir):
    groups = [("di16", "legacy"), ("di16", "mamba1"),
              ("di64", "legacy"), ("di64", "mamba1")]
    settings = [(lr, b1, b2) for lr in (1e-4, 5e-5)
                for b1, b2 in ((.8, .98), (.9, .98), (.9, .999))]
    values = np.full((4, 6), np.nan)
    shorter = set()
    for row in rows:
        i = groups.index((row["shape"], row["init"]))
        j = settings.index(tuple(float(row[k]) for k in ("lr", "beta1", "beta2")))
        if row["swa_day10_gc_improvement_pct"]:
            values[i, j] = float(row["swa_day10_gc_improvement_pct"])
            if row["swa_tag"] != "swa_step02000-08000":
                assert row["swa_tag"] == "swa_step04000-08000"
                shorter.add((i, j))
    assert np.isnan(values).sum() == 2
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.subplots_adjust(left=.24, right=.99, bottom=.34, top=.81)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(values, vmin=12, vmax=24, cmap=cmap, aspect="auto")
    for i in range(4):
        for j in range(6):
            value = values[i, j]
            label = "—" if np.isnan(value) else f"{value:.1f}" + ("*" if (i, j) in shorter else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if np.isfinite(value) and value >= 20 else "#252525")
    ax.set_xticks(np.arange(6), ["(.8, .98)", "(.9, .98)", "(.9, .999)"] * 2)
    ax.set_xlabel(r"AdamW moment parameters $(\beta_1,\beta_2)$", labelpad=7)
    ax.set_yticks(np.arange(4), ["Legacy init.\ndi16, BCG1", "Mamba1 init.\ndi16, BCG1",
                               "Legacy init.\ndi64, BCG4", "Mamba1 init.\ndi64, BCG4"])
    ax.tick_params(axis="both", length=0)
    ax.text(.25, 1.06, r"Learning rate $10^{-4}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9)
    ax.text(.75, 1.06, r"Learning rate $5\!\times\!10^{-5}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    ax.axvline(2.5, color="white", linewidth=2.5)
    ax.axhline(1.5, color="white", linewidth=2.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.27, .12, .65, .04])
    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[12, 16, 20, 24])
    colorbar.set_label("Day-10 exact GraphCast loss reduction (%)", fontsize=9)
    save(fig, "optimizer_sensitivity", "32 matched starts, cold_full zero-state evaluation; "
         "SWA day-10 exact GraphCast loss reductions in the frozen 2026-09-10 report. "
         "SWA 2k–8k, except starred cells use SWA 4k–8k. "
         "Gray dash cells lack evaluations in the source report.", preview_dir)
    return {"rows": groups, "columns_lr_beta1_beta2": settings,
            "day10_loss_reduction_pct": [[None if np.isnan(v) else float(v) for v in row] for row in values],
            "swa_4k_8k_cells_zero_indexed": sorted(shorter),
            "missing_cells_meaning": "No evaluation in the frozen source report; not a zero score."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    reference, optimizer, best = read_json(REF), read_json(OPT), read_json(BEST)
    indices = reference["chosen_idx"]
    verify_exact(reference, indices)
    verify_exact(optimizer, indices)
    verify_exact(best, indices)
    source_hashes = {}
    for report in (REPORT, LR_REPORT):
        for item in read_json(report)["sources"]:
            assert source_hashes.get(item["path"], item["sha256"]) == item["sha256"]
            source_hashes[item["path"]] = item["sha256"]
    sources = {REF, OPT, BEST, REPORT, OPT_CSV, LR_REPORT, LR_CSV, MEM_CSV}
    candidates = [summary(reference)]
    rows = read_csv(OPT_CSV)
    for row in rows:
        if not row["swa_day10_gc_improvement_pct"]:
            continue
        path = OPT_ROOT / row["run"] / "eval/cold_full_zero_exact_gc/matched_v22_reference32" / (row["swa_tag"] + ".json")
        document = read_json(path)
        verify_exact(document, indices)
        loss = document["original_graphcast_loss"]
        np.testing.assert_allclose(float(row["swa_day10_gc_improvement_pct"]),
                                   loss["improvement_pct_per_step"][-1], atol=1e-10)
        np.testing.assert_allclose(float(row["swa"]), loss["improvement_pct_rollout"], atol=1e-10)
        sources.add(path)
        candidates.append(summary(document))
    lr_rows = [row for row in read_csv(LR_CSV)
               if row["status"] == "complete" and row["artifact"].startswith("swa_")]
    for row in lr_rows:
        path = ROOT / row["source"]
        document = read_json(path)
        verify_exact(document, indices)
        candidate = summary(document)
        np.testing.assert_allclose(float(row["rollout_reduction_pct"]),
                                   candidate["rollout_loss_reduction_pct"], atol=1e-10)
        np.testing.assert_allclose(float(row["day10_reduction_pct"]),
                                   candidate["day10_loss_reduction_pct"], atol=1e-10)
        sources.add(path)
        candidates.append(candidate)
    for metric in ("rollout_loss_reduction_pct", "day10_loss_reduction_pct"):
        np.testing.assert_allclose(summary(best)[metric], max(item[metric] for item in candidates), atol=1e-10)
    memory = read_csv(MEM_CSV)
    assert len(memory) == 720
    reconstruction = ROOT / "scripts/analyze_models/analyze_v22_final_state_evals_20260815.py"
    stats = ROOT / "data/graphcast/graphcast/stats/diffs_stddev_by_level.nc"
    sources.update((reconstruction, stats))
    sys.path.insert(0, str(reconstruction.parent))
    from analyze_v22_final_state_evals_20260815 import graphcast_weighted_normalized_mse_improvement
    diff_scales = xr.load_dataset(stats)
    for path in {ROOT / r["source"] for r in memory}:
        document = read_json(path)
        assert document["chosen_idx"] == indices
        assert document["evaluated_samples"] == 32
        assert document["target_steps"] == 40
        assert document["eval_mode"] == "cold_full"
        assert document["residual_state_init"] == "zero"
        saved_curve = sorted((r for r in memory if r["source"] == rel(path)),
                             key=lambda r: float(r["lead_days"]))
        np.testing.assert_allclose(
            [float(r["approximate_gc_loss_reduction_pct"]) for r in saved_curve],
            graphcast_weighted_normalized_mse_improvement(document, diff_scales), atol=1e-10)
        sources.add(path)
    for path in sources:
        if rel(path) in source_hashes:
            assert sha(path) == source_hashes[rel(path)], f"Report source changed: {path}"
    exact_figure(best, args.preview_dir)
    memory_figure(memory, False, args.preview_dir)
    memory_figure(memory, True, args.preview_dir)
    optimizer_metadata = optimizer_figure(rows, args.preview_dir)
    provenance = {
        "generator": rel(Path(__file__).resolve()),
        "generator_sha256": sha(Path(__file__).resolve()),
        "sources": [{"path": rel(path), "sha256": sha(path)} for path in sorted(sources)],
        "exact_gc_loss": {
            "best_model": {"path": rel(BEST), **summary(best)},
            "selection_scope": "Reference and completed SWA configurations in the frozen optimizer and learning-rate reports.",
            "selection_criterion": "Highest aggregate ten-day exact-loss reduction; also highest day-ten reduction.",
            "verified_candidate_count": len(candidates),
            "matched_indices_verified": True,
            "ratio": "100 * (1 - full_loss / paired_baseline_loss)",
            "rollout_ratio": "Ratio after aggregating losses across all 40 leads, not mean lead-wise reduction.",
            "absolute_baseline_display": "Gray dashed curves show the selected model's saved paired GraphCast baseline; reductions use that same baseline.",
            "temperature_2m": {
                "source_field": "per_variable_per_step.2m_temperature",
                "metric": "RMSE (K)",
                "aggregation": "Cosine-latitude-weighted spatial MSE averaged over starts before square root.",
                "ratio": "100 * (1 - rmse_full / rmse_baseline)",
                "saved_ratios_verified": True,
                "day10": {key: temperature_metrics(best)[key][-1]
                          for key in ("rmse_baseline", "rmse_full", "improvement_pct_rmse")},
            },
        },
        "memory_reset": {
            "metric": "Approximate GraphCast loss reduction reconstructed from saved channel RMSE.",
            "all_18_sources_use_same_32_indices": True,
            "main_inner_dimension": 16,
            "main_bc_groups": [1],
            "appendix_bc_groups": [1, 4],
            "main_window": WINDOWS[0], "appendix_windows": WINDOWS,
            "day10_main_window": [r for r in memory if int(r["bcg"]) == 1 and r["swa"] == WINDOWS[0] and float(r["lead_days"]) == 10],
            "independence": "SWA windows overlap and are not independent training repetitions.",
        },
        "optimizer_sensitivity": optimizer_metadata,
        "validation": "All completed optimizer and learning-rate SWA scores checked against exact evaluation JSON and best-model ranking verified; selected model's 40-lead 2 m temperature RMSE reductions checked against saved paired RMSEs; all 720 memory values checked against the existing reconstruction function and source JSON; report source hashes verified. No new evaluations.",
    }
    (HERE / "figure_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"files": [str(HERE / (stem + ".pdf")) for stem in
                               ("exact_gc_loss", "memory_reset", "memory_reset_all", "optimizer_sensitivity")],
                      "best_model": summary(best),
                      "verified_swa_cells": 22, "verified_lr_swa_models": len(lr_rows)}, indent=2))


if __name__ == "__main__":
    main()
