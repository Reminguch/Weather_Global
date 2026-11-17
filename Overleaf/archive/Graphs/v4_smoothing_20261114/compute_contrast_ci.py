#!/usr/bin/env python3
"""Reuse the frozen smoothing reporter on saved, paired 92-date losses only.

Run from the repository after sourcing scripts/graphcast_env.sh. No forecasts,
filter fitting, training, or changes to the existing campaign are performed.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "artifacts/evaluations/v24_optimal_smoothing_v4_20261112"
POWER = ROOT / "plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113"


def pin(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    report_path = CAMPAIGN / "interim_20261113/report.json"
    manifest_path = CAMPAIGN / "manifest.json"
    reporter_path = CAMPAIGN / "source/scripts/analyze_models/report_v24_optimal_smoothing.py"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    losses_pin = report["outputs"]["paired_losses.npz"]
    if pin(losses_pin["path"]) != losses_pin:
        raise ValueError("Pinned paired-loss archive changed")
    if pin(report["filter"]["path"]) != report["filter"]:
        raise ValueError("Pinned frozen-filter metadata changed")
    with np.load(losses_pin["path"], allow_pickle=False) as saved:
        losses = saved["exact_gc_loss"]
        canonical = saved["canonical_v4_raw"]
        ids = saved["initialization_ids"].tolist()
    if (losses.shape != (92, 40, 2, 3) or canonical.shape != (92, 40, 2)
            or ids != report["initialization_ids"] or ids != sorted(set(ids))
            or not np.isfinite(losses).all() or not np.isfinite(canonical).all()):
        raise ValueError("Saved loss shapes, dates, or finite values differ")
    specification = importlib.util.spec_from_file_location("frozen_smoothing_reporter", reporter_path)
    reporter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reporter)
    bootstrap = {key: report["bootstrap"][key] for key in ("replicates", "block_length", "seed")}
    original = reporter.bootstrap_endpoints(losses, canonical, **bootstrap)
    original_ci = np.asarray(original["filtered_pair_reduction_ci95"])
    saved_ci = np.asarray(report["direct_comparisons"]["optimal"]["ci95"])
    np.testing.assert_allclose(original_ci, saved_ci, rtol=0, atol=1e-10)

    # The unchanged reporter compares M1 slot 2 with GC slot 2. Substitute
    # raw M1 in that comparator slot to reuse exactly its paired resampling.
    comparator = losses.copy()
    comparator[:, :, 1, 2] = losses[:, :, 1, 0]
    raw_ci = reporter.bootstrap_endpoints(comparator, canonical, **bootstrap)["filtered_pair_reduction_ci95"]
    endpoint_means = np.stack((losses.mean(axis=(0, 1)), losses[:, -1].mean(axis=0)))
    raw_points = 100 * (1 - endpoint_means[:, 1, 0] / endpoint_means[:, 0, 2])
    expected = report["direct_comparisons"]["raw_m1_vs_optimal_gc"]
    np.testing.assert_allclose(raw_points, [expected["rollout_reduction_pct"], expected["day10_reduction_pct"]], rtol=0, atol=1e-10)

    power_summary_path = POWER / "summary.json"
    power_summary = json.loads(power_summary_path.read_text())
    if power_summary["initialization_ids"] != ids:
        raise ValueError("Power and loss cohorts differ")
    with np.load(POWER / "power_pooled.npz", allow_pickle=False) as pooled:
        pf, pt = pooled["prediction_power"], pooled["truth_power"]
    if pf.shape != (40, 2, 3, 83, 180) or pt.shape != (40, 83, 180) or np.any(pt[..., 1:] <= 0):
        raise ValueError("Unexpected pooled power layout or undefined nonzero-degree ratio")
    relative_power_error = 100 * np.abs(pf[..., 1:] / pt[:, None, None, :, 1:] - 1)
    rollout_power_error = relative_power_error.mean(axis=(0, 3, 4))
    power_rows = list(csv.DictReader((POWER / "summary.csv").open()))
    models, variants = ("GC", "M1"), ("raw", "projected", "optimal")
    for row in power_rows:
        value = rollout_power_error[models.index(row["model"]), variants.index(row["variant"])]
        np.testing.assert_allclose(value, float(row["rollout_power_error_pct"]), rtol=0, atol=1e-10)

    output = {
        "status": "saved_completed_92_date_subset_only",
        "campaign_hash": report["campaign_hash"],
        "evaluation_count": len(ids), "planned_count": report["planned_count"],
        "initialization_ids": ids,
        "scope": "Fixed original GC and M1 checkpoints, frozen filters fitted on 32 2022 starts; paired 92 completed 2023 starts. Not an annual result; no new forecasts or fits.",
        "checkpoint": manifest["checkpoint"], "baseline": manifest["baseline"],
        "filter": report["filter"],
        "loss_layout": {"axes": ["initialization", "lead", "model", "variant"], "shape": list(losses.shape), "models": models, "variants": variants},
        "contrast_definition": "100 * (1 - mean(raw M1 native-grid GraphCast loss) / mean(filtered GC native-grid GraphCast loss)); pool losses before ratio",
        "bootstrap": {**bootstrap, "method": "Existing frozen bootstrap_endpoints: paired non-wrapping blocks of five consecutive initialization dates, sampled identically for both forecasts; percentile 2.5 and 97.5 bounds", "interpretation": original["interpretation"], "additional_exclusions": "Does not quantify filter-estimation uncertainty; pointwise intervals, no simultaneous coverage claim"},
        "raw_m1_vs_optimal_gc": {
            name: {"reduction_pct": float(raw_points[i]), "ci95_pct": np.asarray(raw_ci)[:, i].tolist(), "raw_m1_loss": float(endpoint_means[i, 1, 0]), "optimal_gc_loss": float(endpoint_means[i, 0, 2])}
            for i, name in enumerate(("rollout", "day10"))},
        "existing_filtered_pair_validation": {"max_ci_absolute_difference_pct": float(np.max(np.abs(original_ci - saved_ci))), "ci95_rows_lower_upper_columns_rollout_day10": original_ci.tolist()},
        "power_metric": {"definition": "100 * mean over 40 leads, 83 physical channels and degrees 1..179 of abs(date-pooled prediction power / date-pooled truth power - 1); equal weights on leads, channels and degrees, excluding degree zero", "rollout_pct": {f"{model}_{variant}": float(rollout_power_error[bi, vi]) for bi, model in enumerate(models) for vi, variant in enumerate(variants)}, "interpretation": "Diagnostic of relative spectral power mismatch, not an energy fraction, phase metric or native-GraphCast-weighted loss; no spectral confidence intervals"},
        "sources": {"evaluation_report": pin(report_path), "manifest": pin(manifest_path), "paired_losses": losses_pin, "frozen_reporter": pin(reporter_path), "power_summary": pin(power_summary_path), "power_csv": pin(POWER / "summary.csv"), "power_pooled": pin(POWER / "power_pooled.npz"), "script": pin(__file__)},
    }
    destination = Path(__file__).with_name("contrast_ci.json")
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["raw_m1_vs_optimal_gc"], indent=2))
    print("Written:", destination)


if __name__ == "__main__":
    main()
