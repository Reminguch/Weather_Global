#!/usr/bin/env python3
"""Add paired date-block intervals to the fixed-checkpoint ENS comparison.

This report only resamples saved per-initialization MSEs. It performs no model
inference, introduces no training seeds, and preserves the original ENS masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.merge_v24_2023_eval import cadence_block_counts

MODELS = ("m1", "gc1", "gc025", "ens")
STYLES = {
    "m1": ("#26609f", "-", "GraphCast–Mamba M1 (1°)"),
    "gc1": ("#8b919b", "--", "Public GraphCast (1°)"),
    "gc025": ("#1fa450", "-", "Public GraphCast (0.25°)"),
}
SOURCE_NAME = "public_checkpoints_2023_repaired_ens_20261018"
OUTPUT_NAME = "v4_common_grid_uncertainty_20261113"
PLOT_SPECIFICATION = {
    "figure_width_inches": 6.2,
    "figure5_height_inches": 2.45,
    "figure9_height_inches": 6.1,
    "tick_and_legend_font_points": 8,
    "axis_label_font_points": 8.5,
    "panel_title_font_points": 9.5,
    "legend_columns": 2,
    "scope_note": "Pointwise fixed-checkpoint date-sampling intervals; interpretation is in the manuscript captions.",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def identity(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_paired_scores(source: Path):
    summary_path = source / "summary.json"
    csv_path = source / "matched_per_initialization_mse.csv"
    validation_path = source / "independent_validation.json"
    summary = read_json(summary_path)
    validation = read_json(validation_path)
    if summary.get("status") != "complete" or validation.get("status") != "passed":
        raise ValueError("The source comparison must be complete and independently validated")
    for key, path in (("summary", summary_path), ("matched_csv", csv_path)):
        if validation[key]["sha256"] != identity(path)["sha256"]:
            raise ValueError(f"Source integrity mismatch: {path}")
    times, leads, diagnostics = (summary[key] for key in
                                 ("initialization_times", "lead_hours", "diagnostics"))
    if len(times) != 122 or leads != list(range(6, 241, 6)) or len(diagnostics) != 12:
        raise ValueError("Unexpected source coverage")
    lookups = [{value: i for i, value in enumerate(axis)}
               for axis in (MODELS, times, leads, diagnostics)]
    values = np.full((4, len(times), len(leads), len(diagnostics)), np.nan)
    seen = set()
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["model"], row["initialization_time"], int(row["lead_hours"]),
                   row["diagnostic"])
            if key in seen:
                raise ValueError(f"Duplicate source score: {key}")
            seen.add(key)
            index = tuple(lookup[value] for lookup, value in zip(lookups, key))
            score = float(row["mse_si"])
            if not np.isfinite(score) or score < 0:
                raise ValueError(f"Invalid source MSE: {key}")
            values[index] = score
    valid = np.isfinite(values[3])
    for model_values in values:
        np.testing.assert_array_equal(np.isfinite(model_values), valid,
                                      err_msg="Models do not use identical paired date masks")
    counts = valid.sum(axis=0)
    np.testing.assert_array_equal(counts, summary["matched_initializations_per_lead_diagnostic"])
    if len(seen) != 4 * int(valid.sum()) or int(valid.sum()) != summary["matched_score_cells_per_model"]:
        raise ValueError("Source matched-cell totals disagree")
    if np.any(counts == 0) or np.any(values[3][valid] <= 0):
        raise ValueError("ENS denominator is unusable")
    pooled = np.sqrt(np.where(valid[None], values, 0).sum(axis=1) / counts[None])
    ratios = 100 * (pooled / pooled[3])
    for i, model in enumerate(MODELS):
        np.testing.assert_allclose(pooled[i], summary["rmse"][model], rtol=1e-12, atol=0)
        np.testing.assert_allclose(ratios[i], summary["rmse_pct_of_ens"][model],
                                   rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(ratios[3], np.full(counts.shape, 100.0))
    return summary, values, valid, counts, pooled, ratios


def bootstrap(times, values, valid, *, draws: int, block_starts: int, seed: int):
    weights = cadence_block_counts(times, stride_days=3, block_starts=block_starts,
                                   replicates=draws, seed=seed, segmentation="annual")
    np.testing.assert_array_equal(weights.sum(axis=1), np.full(draws, len(times)))
    shape = valid.shape[1:]
    denominator = weights @ valid.reshape(len(times), -1).astype(np.float64)
    if np.any(denominator <= 0):
        raise ValueError("A bootstrap replicate has no paired observations for a point")
    boot_rmse = np.empty((4, draws, *shape), dtype=np.float64)
    for i in range(len(MODELS)):
        data = np.where(valid, values[i], 0).reshape(len(times), -1)
        boot_rmse[i] = np.sqrt((weights @ data) / denominator).reshape((draws, *shape))
    if not np.isfinite(boot_rmse).all() or np.any(boot_rmse[3] <= 0):
        raise ValueError("Nonfinite bootstrap calculation or invalid ENS denominator")
    # Independent expanded-index check exercises pairing and mask-after-draw order.
    checked_draws = sorted({0, min(17, draws - 1), draws - 1})
    for draw in checked_draws:
        indices = np.repeat(np.arange(len(times)), weights[draw].astype(int))
        for diagnostic in range(shape[1]):
            eligible = indices[valid[indices, -1, diagnostic]]
            for model in range(len(MODELS)):
                expanded_rmse = np.sqrt(values[model, eligible, -1, diagnostic].mean())
                np.testing.assert_allclose(boot_rmse[model, draw, -1, diagnostic], expanded_rmse,
                                           rtol=1e-12, atol=0)
    boot_ratio = 100 * (boot_rmse / boot_rmse[3:4])
    np.testing.assert_array_equal(boot_ratio[3], np.full((draws, *shape), 100.0))
    rmse_ci = np.percentile(boot_rmse, (2.5, 97.5), axis=1)
    ratio_ci = np.percentile(boot_ratio, (2.5, 97.5), axis=1)
    reduction_draws = 100 * (1 - boot_rmse[0] / boot_rmse[2])
    reduction_ci = np.percentile(reduction_draws, (2.5, 97.5), axis=0)
    diagnostics = {
        "every_draw_has_122_sampled_positions": True,
        "minimum_available_count_after_resampling": int(denominator.min()),
        "maximum_available_count_after_resampling": int(denominator.max()),
        "expanded_index_check_draws": checked_draws,
        "expanded_index_check_all_day10_diagnostics_and_models": True,
        "all_bootstrap_statistics_finite": True,
        "ens_relative_reference_exactly_100": True,
    }
    return weights, rmse_ci, ratio_ci, reduction_ci, diagnostics


def write_curves(path, models, leads, diagnostics, counts, rmse, ratios, rmse_ci, ratio_ci):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "lead_hours", "lead_days", "diagnostic", "matched_initializations",
                         "rmse_si", "rmse_ci95_lower_si", "rmse_ci95_upper_si",
                         "rmse_pct_of_ens", "rmse_pct_of_ens_ci95_lower", "rmse_pct_of_ens_ci95_upper"])
        for i, model in enumerate(models):
            for j, lead in enumerate(leads):
                for k, diagnostic in enumerate(diagnostics):
                    writer.writerow([model, lead, lead / 24, diagnostic, counts[j, k], rmse[i, j, k],
                                     rmse_ci[0, i, j, k], rmse_ci[1, i, j, k], ratios[i, j, k],
                                     ratio_ci[0, i, j, k], ratio_ci[1, i, j, k]])


def plot_figures(images, diagnostics, leads, ratios, ratio_ci, counts):
    outputs = []
    for name, selected, shape in (("fig5_common_grid_comparison_v4_ci", ["Q700", "T2m", "U850"], (1, 3)),
                                   ("fig9_common_grid_all_variables_v4_ci", diagnostics, (3, 4))):
        short = shape[0] == 1
        fig, axes = plt.subplots(*shape, figsize=(6.2, 2.45 if short else 6.1), squeeze=False)
        x = np.asarray(leads) / 24
        for ax, diagnostic in zip(axes.flat, selected, strict=True):
            k = diagnostics.index(diagnostic)
            for model, (color, line, label) in STYLES.items():
                i = MODELS.index(model)
                ax.fill_between(x, ratio_ci[0, i, :, k], ratio_ci[1, i, :, k],
                                color=color, alpha=.13, linewidth=0, zorder=1)
                ax.plot(x, ratios[i, :, k], color=color, linestyle=line, linewidth=1.25,
                        label=label, zorder=3)
            ax.axhline(100, color=".5", linewidth=.8, zorder=2, label="Matched ENS (100%)")
            ax.set_title(diagnostic, fontsize=9.5, pad=4)
            ax.set(xlim=(.25, 10), xticks=[2, 4, 6, 8, 10])
            ax.tick_params(labelsize=8, length=3, pad=2)
            ax.grid(alpha=.18)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .998),
                   ncol=2, frameon=False, fontsize=8, columnspacing=1.5,
                   handlelength=2.2, labelspacing=.35, borderaxespad=.2)
        fig.supxlabel("Forecast lead (days)", fontsize=8.5, y=.015)
        fig.supylabel("RMSE (% of matched ENS)", fontsize=8.5, x=.008)
        fig.subplots_adjust(left=.105, right=.989, bottom=.195 if short else .073,
                            top=.75 if short else .885, wspace=.36 if short else .43,
                            hspace=.43)
        for extension in ("pdf", "png"):
            path = images / f"{name}.{extension}"
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(path, dpi=220, metadata=metadata)
            outputs.append(path)
        plt.close(fig)
    return outputs


def replot_saved_report(tables, images, *, copy_to_overleaf):
    """Redraw saved intervals without recomputing bootstrap statistics."""
    report_path = tables / "summary.json"
    report = read_json(report_path)
    if report.get("status") != "complete":
        raise ValueError("Saved uncertainty report is not complete")
    # Check every saved numerical artifact before reusing its summary arrays.
    numerical_outputs = [item for item in report["outputs"]
                         if Path(item["path"]).suffix in (".csv", ".npz")]
    for item in numerical_outputs:
        if identity(ROOT / item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"Saved numerical artifact changed: {item['path']}")
    diagnostics, leads = report["diagnostics"], report["lead_hours"]
    ratios = np.asarray([report["rmse_pct_of_ens"][m] for m in MODELS])
    ratio_ci = np.asarray([[report[key][m] for m in MODELS] for key in
                           ("rmse_pct_of_ens_ci95_lower", "rmse_pct_of_ens_ci95_upper")])
    counts = np.asarray(report["matched_initializations_per_lead_diagnostic"])
    figure_paths = plot_figures(images, diagnostics, leads, ratios, ratio_ci, counts)
    copied = []
    if copy_to_overleaf:
        for path in figure_paths:
            if path.suffix == ".pdf":
                target = ROOT / "Overleaf/arxiv_pics" / path.name
                shutil.copy2(path, target)
                copied.append(target)
    source_snapshot = tables / "source" / Path(__file__).name
    if "computation_script" not in report:
        original = source_snapshot.with_name(source_snapshot.stem + "_computation.py")
        shutil.copy2(source_snapshot, original)
        report["computation_script"] = identity(original)
    shutil.copy2(__file__, source_snapshot)
    current_script = identity(Path(__file__))
    report["sources"] = [current_script if item["path"] == current_script["path"] else item
                         for item in report["sources"]]
    report["outputs"] = [*numerical_outputs, *[identity(p) for p in [*figure_paths, *copied]]]
    report["plotting"] = {**PLOT_SPECIFICATION, "replotted_from_saved_summary": True,
                          "numerical_artifact_hashes_unchanged": True,
                          "bootstrap_recomputed": False}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(report_path), "figures": [str(p) for p in figure_paths],
                      "plotting": report["plotting"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "plots/analyze_models/data/resolution_eval" / SOURCE_NAME)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--block-starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--copy-to-overleaf", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw figures from saved summary; do not rerun the bootstrap")
    args = parser.parse_args()
    tables = ROOT / "plots/analyze_models/data/resolution_eval" / args.output_name
    images = ROOT / "plots/analyze_models/images/resolution_eval" / args.output_name
    if args.plot_only:
        replot_saved_report(tables, images, copy_to_overleaf=args.copy_to_overleaf)
        return
    source = args.source_dir.resolve()
    summary, values, valid, counts, rmse, ratios = load_paired_scores(source)
    times, leads, diagnostics = (summary[key] for key in
                                 ("initialization_times", "lead_hours", "diagnostics"))
    weights, rmse_ci, ratio_ci, reduction_ci, validation = bootstrap(
        times, values, valid, draws=args.replicates, block_starts=args.block_starts, seed=args.seed)
    reduction = 100 * (1 - rmse[0] / rmse[2])
    tables = ROOT / "plots/analyze_models/data/resolution_eval" / args.output_name
    images = ROOT / "plots/analyze_models/images/resolution_eval" / args.output_name
    tables.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    curve_path = tables / "curves.csv"
    write_curves(curve_path, MODELS, leads, diagnostics, counts, rmse, ratios, rmse_ci, ratio_ci)
    contrast_path = tables / "m1_vs_gc025_paired_rmse_reduction.csv"
    with contrast_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["diagnostic", "lead_hours", "matched_initializations", "rmse_reduction_pct",
                         "ci95_lower_pct", "ci95_upper_pct"])
        for k, diagnostic in enumerate(diagnostics):
            for j, lead in enumerate(leads):
                writer.writerow([diagnostic, lead, counts[j, k], reduction[j, k],
                                 reduction_ci[0, j, k], reduction_ci[1, j, k]])
    bootstrap_path = tables / "bootstrap_weights_and_availability.npz"
    np.savez_compressed(bootstrap_path, counts=weights.astype(np.uint16), availability=valid,
                        initialization_times=np.asarray(times), lead_hours=np.asarray(leads),
                        diagnostics=np.asarray(diagnostics))
    figure_paths = plot_figures(images, diagnostics, leads, ratios, ratio_ci, counts)
    copied = []
    if args.copy_to_overleaf:
        for path in figure_paths:
            if path.suffix == ".pdf":
                target = ROOT / "Overleaf/arxiv_pics" / path.name
                shutil.copy2(path, target)
                copied.append(target)
    source_dir = tables / "source"
    source_dir.mkdir(exist_ok=True)
    shutil.copy2(__file__, source_dir / Path(__file__).name)
    helper_path = ROOT / "scripts/analyze_models/merge_v24_2023_eval.py"
    shutil.copy2(helper_path, source_dir / helper_path.name)
    day10 = {diagnostic: {
        "matched_initializations": int(counts[-1, k]),
        "m1_rmse_si": float(rmse[0, -1, k]),
        "gc025_rmse_si": float(rmse[2, -1, k]),
        "m1_vs_gc025_rmse_reduction_pct": float(reduction[-1, k]),
        "ci95_lower_pct": float(reduction_ci[0, -1, k]),
        "ci95_upper_pct": float(reduction_ci[1, -1, k]),
    } for k, diagnostic in enumerate(diagnostics)}
    result = {
        "schema_version": 1, "status": "complete", "report_date": "2026-11-13",
        "scope": "Existing fixed-checkpoint 2023 common-grid comparison; no forecasts or new seeds",
        "repair_protocol_hash": summary["repair_protocol_hash"],
        "reference_protocol_hash": summary["reference_protocol_hash"],
        "candidate_initializations": len(times), "initialization_times": times,
        "lead_hours": leads, "diagnostics": diagnostics, "models": list(MODELS),
        "figure5_diagnostics": ["Q700", "T2m", "U850"],
        "matched_initializations_per_lead_diagnostic": counts.tolist(),
        "matched_initializations_min_max": [int(counts.min()), int(counts.max())],
        "matched_score_cells_per_model": int(valid.sum()),
        "bootstrap": {
            "method": "paired non-wrapping moving-block percentile bootstrap on original regular date grid",
            "replicates": args.replicates, "seed": args.seed, "block_starts": args.block_starts,
            "initialization_stride_days": 3, "block_elapsed_days": (args.block_starts - 1) * 3,
            "draw_length": len(times), "segmentation": "annual",
            "mask_order": "Draw shared date positions first, then apply each lead/diagnostic ENS availability mask",
            "joint_resampling": "Same date multiplicities for all models, leads and diagnostics",
            "percentiles": [2.5, 97.5], "intervals": "pointwise 95%; fixed-checkpoint date-sampling uncertainty",
        },
        "aggregation": "Pool paired per-start spatial MSE, take square roots, then compute model/ENS RMSE ratios. The M1-versus-GC025 contrast is 100*(1-RMSE_M1/RMSE_GC025).",
        "rmse": {m: rmse[i].tolist() for i, m in enumerate(MODELS)},
        "rmse_pct_of_ens": {m: ratios[i].tolist() for i, m in enumerate(MODELS)},
        "rmse_pct_of_ens_ci95_lower": {m: ratio_ci[0, i].tolist() for i, m in enumerate(MODELS)},
        "rmse_pct_of_ens_ci95_upper": {m: ratio_ci[1, i].tolist() for i, m in enumerate(MODELS)},
        "day10_m1_vs_gc025": day10,
        "validation": {**validation, "source_hashes_match_independent_validation": True,
                       "all_models_have_identical_masks": True, "point_estimates_match_saved_summary": True,
                       "source_csv_rows": int(valid.sum()) * 4},
        "limitations": [
            "Intervals describe date-sampling uncertainty conditional on the selected fixed checkpoints, observed year, and availability masks; they do not measure training-seed or between-year variability.",
            "Intervals are pointwise, not simultaneous confidence bands across leads or diagnostics; overlapping marginal intervals are not a paired significance test.",
            "Ten-day forecast windows overlap; five-start blocks retain some dependence but do not guarantee independence or exact nominal coverage. Non-wrapping blocks underweight boundary dates.",
            "The year had earlier numerical-pilot exposure. This analysis does not create a previously untouched test year.",
            "ENS availability changes with lead and diagnostic. No missing forecasts are interpolated, and separate full-122 native-grid intervals are not transferred.",
            "Comparison on the common 1.5-degree grid does not isolate resolution, native-grid fine-scale skill, or an inference speedup.",
        ],
        "sources": [identity(source / name) for name in
                    ("summary.json", "matched_per_initialization_mse.csv", "independent_validation.json")]
                   + [identity(Path(__file__)), identity(helper_path),
                      identity(ROOT / "scripts/experiments/numerical_policy.py")],
        "outputs": [identity(p) for p in [curve_path, contrast_path, bootstrap_path, *figure_paths, *copied]],
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "plotting": PLOT_SPECIFICATION,
    }
    (tables / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(tables / "summary.json"), "day10": day10,
                      "figures": [str(p) for p in figure_paths], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
