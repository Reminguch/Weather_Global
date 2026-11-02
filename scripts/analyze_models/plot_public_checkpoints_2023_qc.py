#!/usr/bin/env python3
"""Apply approved identical truth-quality exclusions to three fixed 2023 models.

This CPU-only sensitivity analysis preserves the original scalar errors and
marks eligibility separately. Figure outputs remain manuscript previews.
No forecasts, source truth, production summaries, or manuscript files change.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.common_grid_metrics import DIAGNOSTICS

DEFAULT_WORKFLOW = ROOT / "artifacts/evaluations/graphcast_common_grid_2023_20261015"
DEFAULT_NAME = "public_checkpoints_2023_qc_20261018"
INVALID_TIMES = tuple(f"2023-11-10T{hour}:00:00Z" for hour in ("06", "12", "18"))
INVALID_DIAGNOSTICS = ("Z500", "Z850", "T500", "Q700", "Q850")
STYLES = {
    "m1": ("#26609f", "-", "GraphCast–Mamba M1 (1°)"),
    "gc1": ("#8b919b", "--", "Public DeepMind GraphCast (1°)"),
    "gc025": ("#1fa450", "-", "Public DeepMind GraphCast (0.25°)"),
}
DISPLAY_UNITS = {
    "m2 s-2": r"m$^2$ s$^{-2}$", "K": "K", "Pa": "Pa",
    "kg kg-1": r"kg kg$^{-1}$", "m s-1": r"m s$^{-1}$",
}


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def eligibility(protocol, audit):
    """Exclude by fixed source-field identity, never by model error."""
    diagnostics = protocol["verification"]["diagnostics"]
    if audit["status"] != "complete" or audit["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Truth-quality audit does not match this protocol")
    observed = {(diagnostic, stamp)
                for diagnostic, groups in audit["repeat_groups"].items()
                for group in groups for stamp in group["valid_times"]}
    expected = {(diagnostic, stamp) for diagnostic in INVALID_DIAGNOSTICS
                for stamp in INVALID_TIMES}
    if observed != expected:
        raise ValueError("Truth repeats differ from the fixed preview exclusion")
    mask = np.ones((122, 40, 12), dtype=bool)
    excluded = []
    invalid = set(map(utc, INVALID_TIMES))
    for i, start in enumerate(protocol["initialization_times"]):
        origin = utc(start)
        if {origin - timedelta(hours=6), origin} & invalid:
            raise ValueError("A selected forecast input overlaps excluded verification times")
        for j, lead in enumerate(protocol["lead_hours"]):
            valid = origin + timedelta(hours=lead)
            if valid in invalid:
                for diagnostic in INVALID_DIAGNOSTICS:
                    mask[i, j, diagnostics.index(diagnostic)] = False
                excluded.append({"initialization_time": start, "lead_hours": lead,
                                 "valid_time": valid.isoformat().replace("+00:00", "Z")})
    counts = mask.sum(axis=0)
    if len(excluded) != 9 or np.count_nonzero(~mask) != 45:
        raise ValueError("Unexpected excluded cell count")
    if set(np.unique(counts)) != {121, 122} or np.count_nonzero(counts == 121) != 45:
        raise ValueError("Unexpected eligible start counts")
    return mask, counts, excluded


def make_plot(images, filename, variables, diagnostics, days, rmse, full):
    fig, axes = plt.subplots(3 if full else 1, 4 if full else 3,
                             figsize=(11.4, 8.1 if full else 3.5), squeeze=False)
    for ax, diagnostic in zip(axes.flat, variables, strict=True):
        k = diagnostics.index(diagnostic)
        for model, (color, style, label) in STYLES.items():
            ax.plot(days, rmse[model][:, k], color=color, ls=style, lw=1.7, label=label)
        unit = DISPLAY_UNITS[DIAGNOSTICS[diagnostic].units]
        ax.set(title=diagnostic, xlabel="Forecast lead (days)", ylabel=f"RMSE ({unit})",
               xlim=(.25, 10), ylim=(0, None))
        ax.set_xticks((2, 4, 6, 8, 10))
        ax.grid(alpha=.20)
        ax.tick_params(labelsize=8)
        if diagnostic.startswith("Q"):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("2023 public-checkpoint comparison · common 1.5° grid",
                 y=.94 if full else .895, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91 if full else .855))
    outputs = []
    for suffix in ("pdf", "png"):
        path = images / f"{filename}.{suffix}"
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}
        fig.savefig(path, dpi=220, bbox_inches="tight", **kwargs)
        outputs.append(identity(path))
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--replot", action="store_true",
                        help="Replace plots from saved summary; preserve summary and original-score CSV.")
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    tables = ROOT / "plots/analyze_models/data/resolution_eval" / args.name
    images = ROOT / "plots/analyze_models/images/resolution_eval" / args.name
    if args.replot:
        summary_path = tables / "summary.json"
        saved = json.loads(summary_path.read_text())
        days = np.asarray(saved["lead_hours"]) / 24
        rmse = {model: np.asarray(cube) for model, cube in saved["rmse"].items()}
        outputs = []
        for name, variables, full in (("fig5_public_checkpoints_2023_qc_preview", ["Z500", "T2m", "U850"], False),
                                      ("fig9_public_checkpoints_2023_qc_preview", saved["diagnostics"], True)):
            outputs += make_plot(images, name, variables, saved["diagnostics"], days, rmse, full)
        metadata = {"status": "plots_regenerated_from_saved_qc_analysis", "summary_at_render": identity(summary_path),
                    "plotter": identity(__file__), "outputs": outputs,
                    "display": "Physical-unit RMSE; detailed eligibility remains in analysis metadata."}
        (images / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"images": str(images), "plotter": metadata["plotter"], "outputs": outputs}, indent=2))
        return
    intended = [tables / "summary.json", tables / "per_initialization_mse_with_eligibility.csv"]
    intended += [images / f"fig{number}_public_checkpoints_2023_qc_preview.{suffix}"
                 for number in (5, 9) for suffix in ("pdf", "png")]
    if any(path.exists() for path in intended):
        raise FileExistsError("An analysis output already exists; use a new --name")
    gather_path = workflow / "source/scripts/experiments/run_common_grid_2023.py"
    spec = importlib.util.spec_from_file_location("frozen_common_grid_gather", gather_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol, values, sources = module.gather(workflow)
    diagnostics = protocol["verification"]["diagnostics"]
    audit_path = workflow / "audits/truth_repeats_20261018.json"
    audit = json.loads(audit_path.read_text())
    mask, counts, excluded = eligibility(protocol, audit)
    rmse = {model: np.sqrt(np.where(mask, cube, 0).sum(axis=0) / counts)
            for model, cube in values.items()}
    original = {model: np.sqrt(cube.mean(axis=0)) for model, cube in values.items()}
    old_summary_path = workflow / "report/summary.json"
    old_summary = json.loads(old_summary_path.read_text())
    for model in values:
        np.testing.assert_allclose(original[model], old_summary["rmse"][model], rtol=2e-14, atol=0)
        np.testing.assert_array_equal(rmse[model][counts == 122], original[model][counts == 122])
    checkpoints = {
        "m1": {**identity(protocol["m1"]["checkpoint"]), "model": "selected M1", "resolution": 1.0},
        "gc1": {**identity(protocol["gc1"]["path"]), "model": "GraphCast_small", "resolution": 1.0,
                "pressure_levels": 13, "mesh_size": 5, "training_years": [1979, 2015]},
        "gc025": {**identity(protocol["gc025_checkpoint"]), "model": "GraphCast", "resolution": .25,
                  "pressure_levels": 37, "mesh_size": 6, "training_years": [1979, 2017]},
    }
    expected_hashes = {"m1": protocol["m1"]["checkpoint_sha256"],
                       "gc1": protocol["gc1"]["checkpoint_sha256"],
                       "gc025": protocol["gc025_checkpoint_sha256"]}
    if any(checkpoints[m]["sha256"] != expected_hashes[m] for m in checkpoints):
        raise ValueError("Checkpoint identity changed")
    tables.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "per_initialization_mse_with_eligibility.csv"
    with csv_path.open("x") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "initialization_time", "lead_hours", "valid_time",
                         "diagnostic", "mse_si", "eligible", "exclusion_reason"))
        for model, cube in values.items():
            for i, start in enumerate(protocol["initialization_times"]):
                for j, lead in enumerate(protocol["lead_hours"]):
                    valid = (utc(start) + timedelta(hours=lead)).isoformat().replace("+00:00", "Z")
                    for k, diagnostic in enumerate(diagnostics):
                        writer.writerow((model, start, lead, valid, diagnostic, cube[i, j, k],
                                         bool(mask[i, j, k]), "" if mask[i, j, k]
                                         else "confirmed_repeated_source_truth_field"))
    days = np.asarray(protocol["lead_hours"]) / 24
    outputs = []
    for name, variables, full in (("fig5_public_checkpoints_2023_qc_preview", ["Z500", "T2m", "U850"], False),
                                  ("fig9_public_checkpoints_2023_qc_preview", diagnostics, True)):
        outputs += make_plot(images, name, variables, diagnostics, days, rmse, full)
    summary = {
        "schema_version": 1, "status": "user_approved_qc_analysis_manuscript_preview",
        "protocol_hash": protocol["protocol_hash"], "source_digest": protocol["source_digest"],
        "initializations": 122, "initialization_times": protocol["initialization_times"],
        "lead_hours": protocol["lead_hours"], "diagnostics": diagnostics,
        "units": {d: DIAGNOSTICS[d].units for d in diagnostics},
        "selection": "Original 122 initializations retained; objective source-truth exclusions, identical for all models.",
        "invalid_truth_times": INVALID_TIMES, "invalid_diagnostics": INVALID_DIAGNOSTICS,
        "excluded_start_lead_pairs": excluded, "excluded_diagnostic_cells_per_model": 45,
        "eligible_initializations_per_lead_diagnostic": counts.tolist(),
        "aggregation": "Square root of arithmetic mean of original physical-unit spatial MSE over eligible paired starts.",
        "rmse": {m: a.tolist() for m, a in rmse.items()},
        "unfiltered_rmse": {m: a.tolist() for m, a in original.items()},
        "rmse_change_from_unfiltered_pct": {m: (100 * (rmse[m] / original[m] - 1)).tolist() for m in values},
        "m1_physical_rmse_reduction_pct": {m: (100 * (1 - rmse["m1"] / rmse[m])).tolist() for m in ("gc1", "gc025")},
        "checkpoints": checkpoints, "numerics": protocol["numerics"], "verification": protocol["verification"],
        "uncertainty": "Point estimates only; existing unfiltered intervals are not reused.",
        "limitations": ["User approved identical source-truth exclusions; manuscript figures await visual review.",
                        "Exclusions were defined after identifying a truth-data defect; original errors are preserved.",
                        "Fixed-model comparison does not isolate resolution, mesh, pressure levels, or training coverage.",
                        "January 1 independent upstream rollout is a limited inference check, not full-year equivalence.",
                        "This preview does not repair native-grid losses or other experiments using the affected source."],
        "checks": {"paired_protocol_checkpoint_truth_hashes": "passed by frozen gather()",
                   "original_summary_reproduction": "passed at rtol 2e-14",
                   "unaffected_pooled_rmse": "bitwise unchanged"},
        "source_files": sources + [identity(workflow / "protocol.json"), identity(gather_path),
                                    identity(audit_path), identity(old_summary_path), identity(__file__),
                                    identity(ROOT / "scripts/analyze_models/common_grid_metrics.py")],
        "outputs": [identity(csv_path), *outputs],
    }
    summary_path = tables / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(summary_path), "images": str(images),
                      "excluded_cells_per_model": 45, "counts": [int(counts.min()), int(counts.max())]}, indent=2))


if __name__ == "__main__":
    main()
