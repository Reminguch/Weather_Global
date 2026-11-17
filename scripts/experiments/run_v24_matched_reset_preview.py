#!/usr/bin/env python3
"""Four fixed-date reset/reset preview, separate from the ongoing 92-date run.

One bounded GPU job evaluates all four dates and then builds matching figures.
The observer, forecasting code and reset configuration are reused unchanged
from the full campaign; existing GC/M1 statistics are pooled on the same dates.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments import run_v24_matched_reset_spectra as full
from scripts.experiments.frozen_runtime import FORMAT, _digest, frozen_command, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/run_v24_matched_reset_preview.py"
WRAPPER = "scripts/experiments/v24_matched_reset_preview.slurm"
OVERLAYS = (DRIVER, WRAPPER, full.PLOTTER)
PARENT = "artifacts/evaluations/v24_matched_reset_spectra_20261113"
OPTIMAL = "artifacts/evaluations/v24_optimal_smoothing_v4_20261112"
NAME = "v24_literature_spectral_preview4_20261113"
IDS = ["20230101T000000Z", "20230401T000000Z", "20230703T000000Z", "20231001T000000Z"]
POWER_KEYS = ("prediction_power", "truth_power", "cross_power", "error_power",
              "exact_gc_loss", "spectral_weighted_gc_loss")


def freeze(workspace, parent, destination):
    previous = verify_source(parent)
    if destination.exists():
        raise FileExistsError(destination)
    files = {}
    for relative in sorted(set(previous["files"]) | set(OVERLAYS)):
        source = (workspace if relative in OVERLAYS else parent) / relative
        files[relative] = full.copy_pin(source, destination / relative)["sha256"]
    verify_source(parent)
    write(destination / "snapshot.json", {
        "format": FORMAT, "workspace": str(workspace), "files": files,
        "source_digest": _digest(files), "parent_snapshot": pinned(parent / "snapshot.json"),
        "overlays": list(OVERLAYS),
        "forecast_computation": "Byte-identical observer, evaluation driver and numerical source from full92 campaign"})
    return verify_source(destination)


def save_pooled(path, arrays, metadata):
    from src.analysis.smoothing_observer import atomic_npz
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"])) != metadata:
                raise ValueError("Existing preview pooled metadata differs")
            for key, value in arrays.items():
                if not np.array_equal(archive[key], value):
                    raise ValueError("Existing preview pooled values differ")
    else:
        atomic_npz(path, arrays, metadata)


def pool_reference(root, optimal, filters, records):
    from src.analysis.spherical_smoothing import json_digest
    aggregate = None
    if [Path(record["path"]).stem for record in records] != IDS:
        raise ValueError("Reference must contain the same four dates in fixed order")
    shapes = {key: (40, 2, 3, 83, 180) for key in ("prediction_power", "cross_power", "error_power")}
    shapes.update(truth_power=(40, 83, 180), exact_gc_loss=(40, 2, 3), spectral_weighted_gc_loss=(40, 2, 3))
    for record in records:
        check(record)
        with np.load(record["path"], allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if (metadata["campaign_hash"] != optimal["record_hash"]
                    or metadata["checkpoint_sha256"] != optimal["checkpoint"]["sha256"]
                    or metadata["source_digest"] != optimal["source_digest"]
                    or metadata["filter_digest"] != json_digest(filters)
                    or metadata["layout"] != filters["layout"]
                    or metadata["branches"] != ["GC", "M1"]
                    or metadata["lead_hours"] != full.LEADS
                    or metadata["initialization_id"] != Path(record["path"]).stem):
                raise ValueError("GC/M1 preview reference identity differs")
            if aggregate is None:
                aggregate = {key: np.zeros(shape, dtype=float) for key, shape in shapes.items()}
            for key, shape in shapes.items():
                value = archive[key]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Invalid reference statistic shape or finite values")
                aggregate[key] += value
    pooled = {key: value / 4 for key, value in aggregate.items()}
    metadata = {"campaign_hash": optimal["record_hash"], "count": 4, "initialization_ids": IDS,
                "status": "preliminary_four_date_preview", "branches": ["GC", "M1"],
                "variants": ["raw", "projected", "optimal"], "layout": filters["layout"],
                "filter_digest": json_digest(filters), "sources": records}
    path = root / "gc_m1_reference/power_pooled.npz"
    save_pooled(path, pooled, metadata)
    rows = [{"model": model, "variant": variant,
             "exact_gc_rollout_loss": float(pooled["exact_gc_loss"][:, bi, vi].mean())}
            for bi, model in enumerate(metadata["branches"])
            for vi, variant in enumerate(metadata["variants"])]
    return seal(root / "gc_m1_reference/summary.json", {
        **metadata, "planned_count": 4, "fit_year": 2022, "rows": rows,
        "scope": "Matching four-date descriptive preview; not an annual evaluation",
        "data": {"power_pooled.npz": pinned(path)}})


def prepare(root, workspace, parent_root):
    root, workspace, parent_root = [Path(p).resolve() for p in (root, workspace, parent_root)]
    if root.exists():
        raise FileExistsError("Use a new preview campaign directory")
    parent = full.verify(parent_root, frozen=False)
    cohort = read(parent["cohorts"]["evaluate"]["path"])
    entries = dict(zip(full.original.expected_ids(cohort), cohort["initializations"]))
    selected = {**cohort, "initializations": [entries[key] for key in IDS]}
    optimal_root = workspace / OPTIMAL
    optimal = unseal(optimal_root / "manifest.json")
    interim = unseal(optimal_root / "interim_20261113/report.json")
    filter_pin = interim["filter"]
    check(filter_pin)
    filters = unseal(filter_pin["path"])
    if interim["campaign_hash"] != optimal["record_hash"] or filters["campaign_hash"] != optimal["record_hash"]:
        raise ValueError("Mismatched reference filter campaign")
    by_id = {Path(record["path"]).stem: record for record in interim["statistics"]}
    reference_records = [by_id[key] for key in IDS]
    for record in reference_records:
        check(record)
    root.mkdir(parents=True)
    for name in ("logs", "submissions", "checks", "manifests", "references", "evaluate", "report", "gc_m1_reference"):
        (root / name).mkdir()
    parent_pin = full.copy_pin(parent_root / "manifest.json", root / "references/full92_manifest.json")
    write(root / "manifests/evaluate.json", selected)
    cohort_pin = pinned(root / "manifests/evaluate.json")
    frozen = freeze(workspace, Path(parent["source_root"]), root / "source")
    reference = pool_reference(root, optimal, filters, reference_records)
    manifest = copy.deepcopy(parent)
    manifest.pop("record_hash")
    manifest.update(format="v24_matched_reset_preview4_campaign_v1",
        workspace=str(workspace), source_root=str(root / "source"), source_digest=frozen["source_digest"],
        initialization_ids=IDS, cohorts={"evaluate": cohort_pin},
        shards={"evaluate": [{"manifest": cohort_pin, "count": 4}]},
        parent_full92=parent_pin, reference_filter=filter_pin, reference_statistics=reference_records,
        reference_summary=pinned(root / "gc_m1_reference/summary.json"),
        inputs=[*parent["inputs"], parent_pin, cohort_pin, filter_pin, *reference_records,
                pinned(root / "gc_m1_reference/summary.json"), reference["data"]["power_pooled.npz"]],
        preview={"preselected_dates": IDS, "date_selection": "First available date per quarter in existing92-date cohort; fixed before reset preview",
                 "scope": "Four-date descriptive preview only; full92 campaign continues unchanged", "plot_name": NAME},
        resources={"memory_gib": 16, "cpus": 4, "minutes": 59, "constraint": "gpu80&pcie", "qos": "gpu-test",
                   "evidence": "Existing8-date raw-style jobs17430308_0..3:22:00–23:06 and9.59–9.82GiB,4CPU. "
                               "Preview4dates plus pooling/plotting in one GPUjob; user-requested59-minute cap; reuse validated observer."})
    return seal(root / "manifest.json", manifest)


def verify(root, *, frozen=True):
    manifest = full.original.verify(root, frozen=frozen)
    if (manifest["format"] != "v24_matched_reset_preview4_campaign_v1"
            or manifest["initialization_ids"] != IDS or manifest["checkpoint"]["sha256"] != full.RESET_SHA256
            or manifest["reset_state_every_step"] is not True or manifest["model_id"] != "R_matched"
            or len(manifest["shards"]["evaluate"]) != 1 or manifest["shards"]["evaluate"][0]["count"] != 4
            or full.original.expected_ids(read(manifest["cohorts"]["evaluate"]["path"])) != IDS):
        raise ValueError("Preview checkpoint, reset policy or fixed four-date cohort differs")
    return manifest


def report(root, manifest):
    from src.analysis.raw_spectral_observer import STATISTICS_SCHEMA
    paths = full.original.statistics_paths(root, manifest, "evaluate")
    if [path.stem for path in paths] != IDS:
        raise ValueError("All four fixed preview dates must complete before plotting")
    keys = (*POWER_KEYS, "normalized_grid_mse")
    pooled, layout, individual = None, None, []
    for path in paths:
        arrays, metadata = full.original.load_statistics(path, manifest, "evaluate")
        if (metadata["schema"] != STATISTICS_SCHEMA or metadata["checkpoint_sha256"] != full.RESET_SHA256
                or metadata["reset_state_every_step"] is not True or metadata["source_digest"] != manifest["source_digest"]
                or metadata["branches"] != ["GC", "R_matched"] or metadata["lead_hours"] != full.LEADS):
            raise ValueError("Preview reset statistics identity differs")
        if pooled is None:
            layout = metadata["layout"]
            pooled = {key: np.zeros_like(arrays[key], dtype=float) for key in keys}
        if metadata["layout"] != layout or arrays["prediction_power"].shape != (40, 2, 83, 180):
            raise ValueError("Preview reset spectral layout differs")
        for key in keys:
            if arrays[key].shape != pooled[key].shape:
                raise ValueError("Preview reset array shape changed")
            pooled[key] += arrays[key]
        individual.append(arrays["exact_gc_loss"])
    pooled = {key: value / 4 for key, value in pooled.items()}
    diagnostics = {"canonical_reset_metric_drift": full.drift_diagnostic(
        np.asarray(individual)[:, :, 1], full.canonical_reset_losses(manifest), rtol=1e-3,
        label="preview reset forecast/archived reset checkpoint evaluation"),
        "worker_grid_mse_roundoff": unseal(root / "checks/evaluate_00.json")["grid_mse_roundoff"]}
    metadata = {"format": "v24_matched_reset_preview4_pooled_v1", "status": "complete", "count": 4,
        "scope": "Preliminary four-date preview; not annual or training-seed evidence",
        "campaign_hash": manifest["record_hash"], "initialization_ids": IDS, "model_id": "R_matched",
        "reset_state_every_step": True, "checkpoint": manifest["checkpoint"], "checkpoint_sha256": full.RESET_SHA256,
        "source_digest": manifest["source_digest"], "branches": ["GC", "R_matched"], "variants": ["raw"],
        "lead_hours": full.LEADS, "layout": layout, "lmax": 179, "filter_fitting": False,
        "sources": [pinned(root / "manifest.json"), *[pinned(path) for path in paths]],
        "numerical_diagnostics": diagnostics}
    path = root / "report/power_pooled.npz"
    save_pooled(path, pooled, metadata)
    summary = seal(root / "report/summary.json", {**metadata,
        "format": "v24_matched_reset_preview4_report_v1", "data": {"power_pooled.npz": pinned(path)},
        "exact_gc_loss_by_lead": pooled["exact_gc_loss"].tolist()})
    command, options = frozen_command(Path(manifest["source_root"]), Path(manifest["workspace"]), full.PLOTTER,
        ["--preview", "--name", NAME, "--workspace", manifest["workspace"],
         "--source", str(root / "gc_m1_reference"), "--reset-source", str(root / "report"),
         "--filters", manifest["reference_filter"]["path"]])
    subprocess.run(command, check=True, **options)
    write(root / "checks/plot_refresh.json", {"status": "complete", "campaign_hash": manifest["record_hash"],
        "report": pinned(root / "report/summary.json"),
        "plot_summary": pinned(Path(manifest["workspace"]) / f"plots/analyze_models/data/resolution_eval/{NAME}/summary.json")})
    return summary


def sbatch(root, manifest):
    budget = manifest["resources"]
    return ["sbatch", "--parsable", "--account=dabanin", "--job-name=reset-preview4",
        f"--chdir={manifest['workspace']}", f"--mem={budget['memory_gib']}G", f"--time={budget['minutes']}",
        f"--cpus-per-task={budget['cpus']}", "--gres=gpu:1", f"--constraint={budget['constraint']}",
        f"--qos={budget['qos']}", f"--output={root}/logs/preview-%j.out", f"--error={root}/logs/preview-%j.err",
        str(Path(manifest["source_root"]) / WRAPPER), "run", str(root), manifest["source_root"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "run", "report", "submit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--parent", type=Path, default=ROOT / PARENT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == "prepare":
        result = prepare(root, args.workspace, args.parent)
    else:
        manifest = verify(root, frozen=args.stage != "verify")
        if args.stage == "verify":
            result = {"status": "verified", "campaign_hash": manifest["record_hash"], "command": sbatch(root, manifest)}
        elif args.stage == "submit":
            result = {"preview": submit_once(root, "preview", sbatch(root, manifest))}
        else:
            if args.stage == "run" and not (root / "checks/evaluate_00.json").exists():
                full.evaluate(root, manifest, 0)
            result = report(root, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
