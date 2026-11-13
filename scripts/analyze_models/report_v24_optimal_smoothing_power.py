#!/usr/bin/env python3
"""Compare exact GC loss and power fidelity on one pinned completed date set.

Reads saved statistics only. A partial evaluation report remains explicitly
partial; this tool never writes the campaign's annual completion artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments.run_v24_mgda import check, pinned, seal, unseal
from scripts.analyze_models.report_v24_optimal_smoothing import json_finite, ratio, write_csv
from src.analysis.smoothing_observer import atomic_npz
from src.analysis.spherical_smoothing import json_digest


def finite_mean(values, axis=None):
    finite = np.isfinite(values)
    return ratio(np.where(finite, values, 0).sum(axis=axis), finite.sum(axis=axis))


def spectral_scores(prediction, truth, cross, error):
    """Ratios of date-pooled powers; never average per-date power ratios."""
    reference = truth[:, None, None]
    power_ratio = ratio(prediction, reference)
    correlation = ratio(cross, np.sqrt(prediction * reference))
    relative_error = ratio(error, reference)
    absolute_error = 100 * np.abs(power_ratio[..., 1:] - 1)
    return {"power_ratio": power_ratio, "spectral_correlation": correlation,
            "relative_error_power": relative_error,
            "macro_power_error_pct": finite_mean(absolute_error, axis=(-2, -1)),
            "mean_power_ratio_by_degree": finite_mean(power_ratio[..., 1:], axis=-2),
            "mean_power_error_by_degree_pct": finite_mean(absolute_error, axis=-2),
            "macro_spectral_correlation": finite_mean(correlation[..., 1:], axis=(-2, -1)),
            "undefined_power_bins": (~np.isfinite(absolute_error)).sum(axis=(-2, -1))}


def load_pooled(campaign, evaluation_report):
    manifest, report = unseal(campaign / "manifest.json"), unseal(evaluation_report)
    if report["campaign_hash"] != manifest["record_hash"]:
        raise ValueError("Requested report belongs to a different campaign")
    check(report["filter"])
    frozen = unseal(report["filter"]["path"])
    filter_digest = json_digest(frozen)
    keys = ("prediction_power", "truth_power", "cross_power", "error_power",
            "exact_gc_loss", "spectral_weighted_gc_loss")
    if "pooled_statistics" in report:
        check(report["pooled_statistics"])
        with np.load(report["pooled_statistics"]["path"], allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["campaign_hash"] != manifest["record_hash"] or metadata["filter_digest"] != filter_digest:
                raise ValueError("Pooled statistics provenance differs")
            pooled = {key: archive[key] for key in keys}
        ids = metadata["initialization_ids"]
        count = report["count"]
        baseline_rollout = report["canonical_v4_gc_rollout_loss"]
        baseline_day10 = report["canonical_v4_gc_loss_by_lead"][-1]
        status = "complete_annual_evaluation"
    else:
        count = report["evaluation_count"]
        ids = report["initialization_ids"]
        if count != len(ids) or len(set(ids)) != count:
            raise ValueError("Incomplete or duplicated interim date list")
        arrays, seen = None, []
        for index, pin in enumerate(report["statistics"]):
            check(pin)
            path = Path(pin["path"])
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                if (metadata["campaign_hash"] != manifest["record_hash"]
                        or metadata["filter_digest"] != filter_digest
                        or metadata["source_digest"] != manifest["source_digest"]
                        or metadata["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]
                        or metadata["layout"] != frozen["layout"]
                        or metadata["initialization_id"] != path.stem
                        or metadata["mode"] != "evaluate"):
                    raise ValueError("Saved spectral statistics have different provenance")
                if arrays is None:
                    arrays = {key: np.zeros_like(archive[key], dtype=np.float64) for key in keys}
                for key in keys:
                    value = archive[key]
                    if value.shape != arrays[key].shape or not np.isfinite(value).all():
                        raise ValueError("Nonfinite or changed spectral statistic layout")
                    arrays[key] += value
            seen.append(path.stem)
            if (index + 1) % 16 == 0 or index + 1 == count:
                print(f"Pooled spectra for {index + 1}/{count} matched starts", flush=True)
        if sorted(seen) != sorted(ids):
            raise ValueError("Power spectrum cohort differs from MSE report")
        pooled = {key: values / count for key, values in arrays.items()}
        baseline_rollout = report["canonical_v4_gc_rollout"]
        baseline_day10 = report["canonical_v4_gc_day10"]
        status = report["status"]
    if count != len(ids) or pooled["prediction_power"].shape != (40, 2, 3, 83, 180):
        raise ValueError("Unexpected date count or spectrum dimensions")
    if not all(np.isfinite(value).all() for value in pooled.values()):
        raise ValueError("Nonfinite pooled spectrum")
    return manifest, report, pooled, ids, baseline_rollout, baseline_day10, status


def build(campaign, evaluation_report, data_dir, image_dir):
    manifest, original, pooled, ids, baseline, baseline_day10, status = load_pooled(campaign, evaluation_report)
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    scores = spectral_scores(pooled["prediction_power"], pooled["truth_power"],
                             pooled["cross_power"], pooled["error_power"])
    rows, degree_rows, lead_rows = [], [], []
    for bi, model in enumerate(("GC", "M1")):
        for vi, variant in enumerate(("raw", "projected", "optimal")):
            loss = pooled["exact_gc_loss"][:, bi, vi]
            macro = scores["macro_power_error_pct"][:, bi, vi]
            rows.append({"model": model, "variant": variant,
                "exact_gc_rollout_loss": loss.mean(),
                "rollout_loss_reduction_pct": 100 * (1 - loss.mean() / baseline),
                "day10_loss_reduction_pct": 100 * (1 - loss[-1] / baseline_day10),
                "rollout_power_error_pct": finite_mean(macro), "day10_power_error_pct": macro[-1],
                "rollout_mean_spectral_correlation": finite_mean(scores["macro_spectral_correlation"][:, bi, vi]),
                "day10_mean_spectral_correlation": scores["macro_spectral_correlation"][-1, bi, vi],
                "rollout_gc_weighted_spectral_mse": pooled["spectral_weighted_gc_loss"][:, bi, vi].mean(),
                "undefined_power_bins": scores["undefined_power_bins"][:, bi, vi].sum()})
            for li in range(40):
                lead_rows.append({"model": model, "variant": variant, "lead_hours": 6 * (li + 1),
                    "exact_gc_loss": loss[li], "power_error_pct": macro[li],
                    "mean_spectral_correlation": scores["macro_spectral_correlation"][li, bi, vi]})
                for degree in range(1, 180):
                    degree_rows.append({"model": model, "variant": variant,
                        "lead_hours": 6 * (li + 1), "degree": degree,
                        "mean_channel_forecast_truth_power_ratio": scores["mean_power_ratio_by_degree"][li, bi, vi, degree - 1],
                        "mean_channel_absolute_relative_power_error_pct": scores["mean_power_error_by_degree_pct"][li, bi, vi, degree - 1]})
    # Reproduce the preceding MSE report on exactly its fixed subset.
    for current, preceding in zip(rows, original["summary"]):
        key = "rollout_exact_gc_loss" if "rollout_exact_gc_loss" in preceding else "exact_gc_rollout_loss"
        if current["model"] != preceding["model"] or current["variant"] != preceding["variant"]:
            raise ValueError("MSE row order differs")
        if not np.isclose(current["exact_gc_rollout_loss"], preceding[key], rtol=1e-12, atol=1e-12):
            raise ValueError("Power report uses a different MSE cohort or reduction")
    for name, content in (("summary", rows), ("power_by_degree", degree_rows), ("leadwise", lead_rows)):
        write_csv(data_dir / f"{name}.csv", content)
    metadata = {"campaign_hash": manifest["record_hash"], "evaluation_report": pinned(evaluation_report),
        "initialization_ids": ids, "count": len(ids), "status": status,
        "branches": ["GC", "M1"], "variants": ["raw", "projected", "optimal"]}
    atomic_npz(data_dir / "power_pooled.npz", {**pooled, **scores}, metadata)
    maximum_correlation = float(np.nanmax(np.abs(scores["spectral_correlation"][..., 1:])))
    diagnostics = {"zero_truth_positive_degree_bins": int(np.sum(pooled["truth_power"][..., 1:] == 0)),
        "maximum_absolute_spectral_correlation": maximum_correlation,
        "correlation_reference": 1.0003,
        "raw_projected_power_max_difference": float(np.max(np.abs(
            pooled["prediction_power"][:, :, 0] - pooled["prediction_power"][:, :, 1]))),
        "warnings": (["Finite correlation exceeds unit magnitude plus reference; retained"] if maximum_correlation > 1.0003 else [])}
    result = json_finite({**metadata, "first_initialization": ids[0], "last_initialization": ids[-1],
        "planned_count": 122, "fit_year": 2022, "rows": rows,
        "metric": "100 * mean across 40 leads,83 channels,positive degrees1..179 of abs(date-pooled forecast power/date-pooled truth power - 1)",
        "mean_curve": "Equal-channel mean of dimensionless per-channel power ratios; not total power across mixed physical units",
        "ratio_policy": "Pool over the same dates before dividing; exact zero denominators undefined and counted, no floor",
        "spectral_convention": "Physical fields, no climatology subtraction; exclude degree0 from amplitude summary. Coefficient-domain powers; projected and raw share represented coefficients.",
        "uncertainty": "Descriptive point estimates; no spectral confidence intervals or new model selection",
        "diagnostics": diagnostics,
        "data": {name: pinned(data_dir / name) for name in ("summary.csv", "leadwise.csv", "power_by_degree.csv", "power_pooled.npz")},
        "builder": pinned(__file__)})
    seal(data_dir / "summary.json", result)
    from scripts.analyze_models.plot_v24_optimal_smoothing_power import plot
    plot(data_dir, image_dir)
    print(json.dumps({"count": len(ids), "rows": result["rows"], "diagnostics": diagnostics}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    data = ROOT / "plots/analyze_models/data/resolution_eval" / args.name
    images = ROOT / "plots/analyze_models/images/resolution_eval" / args.name
    build(args.campaign.resolve(), args.input_report.resolve(), data, images)


if __name__ == "__main__":
    main()
