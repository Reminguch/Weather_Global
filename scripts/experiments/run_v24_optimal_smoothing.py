#!/usr/bin/env python3
"""Fit GraphCast-style spherical filters on 2022 and score v4's 2023 cohort.

The forecast evaluator is preserved from the completed GC/M1 smoothing run.
Only new observation, fitting, and reporting code is overlaid into a new source
snapshot. No training or existing campaign is modified.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments import run_v24_smoothing as original
from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/run_v24_optimal_smoothing.py"
WRAPPER = "scripts/experiments/v24_optimal_smoothing.slurm"
OVERLAYS = (DRIVER, WRAPPER, "scripts/experiments/run_v24_smoothing.py",
            "src/analysis/optimal_spherical_filter.py", "src/analysis/optimal_filter_observer.py",
            "scripts/analyze_models/report_v24_optimal_smoothing.py")
PARENT = "artifacts/evaluations/v24_smoothing_m1_20261107"
V4_METRICS = "artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009/shards"
M1_SHA256 = "b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0"


def freeze_extension(workspace, parent, destination):
    workspace, parent, destination = map(Path, (workspace, parent, destination))
    if destination.exists():
        raise FileExistsError(destination)
    previous = verify_source(parent)
    files = {}
    for relative in sorted(set(previous["files"]) | set(OVERLAYS)):
        source = (workspace if relative in OVERLAYS else parent) / relative
        before = file_sha256(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha256(target) != before or file_sha256(source) != before:
            raise ValueError(f"Source changed while freezing: {source}")
        files[relative] = before
    verify_source(parent)
    write(destination / "snapshot.json", {
        "format": FORMAT, "workspace": str(workspace.resolve()), "files": files,
        "source_digest": _digest(files), "parent_snapshot": pinned(parent / "snapshot.json"),
        "overlays": list(OVERLAYS),
        "forecast_computation": "All source outside the listed observer/report/driver overlays is byte-identical"})
    return verify_source(destination)


def validate_cohorts(development, annual):
    dev_ids, annual_ids = map(original.expected_ids, (development, annual))
    if len(dev_ids) != 32 or any(not item.startswith("2022") for item in dev_ids):
        raise ValueError("Fitting requires the fixed 32 initializations in 2022")
    first = datetime(2023, 1, 1, tzinfo=timezone.utc)
    expected = [(first + timedelta(days=3 * i)).strftime("%Y%m%dT%H%M%SZ") for i in range(122)]
    if annual_ids != expected:
        raise ValueError("Evaluation must use v4's exact ordered 122-start 2023 cohort")
    if set(dev_ids) & set(annual_ids):
        raise ValueError("Development and evaluation overlap")


def prepare(root, workspace, parent_root, *, calibration_shards=4, evaluation_shards=8):
    root, workspace, parent_root = map(lambda p: Path(p).resolve(), (root, workspace, parent_root))
    if root.exists():
        raise FileExistsError("Use a new optimal-filter campaign directory")
    parent = original.verify(parent_root, frozen=False)
    completed = unseal(parent_root / "report.json")
    if completed["status"] != "complete" or completed["campaign_hash"] != parent["record_hash"]:
        raise ValueError("Parent GC/M1 evaluation is incomplete or mismatched")
    if parent["model_id"] != "M1" or parent["checkpoint"]["sha256"] != M1_SHA256:
        raise ValueError("This comparison requires the principal v4 M1 SWA checkpoint")
    development, annual = [read(parent["cohorts"][stage]["path"]) for stage in ("calibrate", "evaluate")]
    validate_cohorts(development, annual)
    if not 1 <= calibration_shards <= 32 or not 1 <= evaluation_shards <= 122:
        raise ValueError("Invalid shard count")
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "checks", "manifests", "calibrate", "evaluate", "references"):
        (root / sub).mkdir()
    frozen = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    # Preserve the exact manuscript and original metric artifacts used as references.
    manuscript = root / "references/v4.tex"
    shutil.copy2(workspace / "Overleaf/v4.tex", manuscript)
    metric_inputs = []
    for i in range(8):
        source = workspace / V4_METRICS / f"M1_{i:02d}.json"
        target = root / "references" / source.name
        shutil.copy2(source, target)
        if file_sha256(source) != file_sha256(target):
            raise ValueError("Canonical v4 reference changed while copying")
        metric_inputs.append(pinned(target))
    inputs = copy.deepcopy(parent["inputs"])
    inputs.extend([pinned(parent_root / "manifest.json"), pinned(parent_root / "report.json"),
                   pinned(manuscript), *metric_inputs])
    cohorts, shards = {}, {}
    for stage, payload, count in (("calibrate", development, calibration_shards),
                                  ("evaluate", annual, evaluation_shards)):
        path = root / f"manifests/{stage}.json"
        write(path, payload)
        cohorts[stage] = pinned(path)
        inputs.append(cohorts[stage])
        shards[stage] = []
        for index, chunk in enumerate(np.array_split(np.asarray(payload["initializations"], dtype=object), count)):
            path = root / f"manifests/{stage}_{index:02d}.json"
            write(path, {**payload, "initializations": list(chunk)})
            record = pinned(path)
            inputs.append(record)
            shards[stage].append({"manifest": record, "count": len(chunk)})
    for item in inputs:
        check(item)
    manifest = copy.deepcopy(parent)
    manifest.pop("record_hash")
    manifest.update(format="v24_optimal_smoothing_campaign_v1", workspace=str(workspace),
        source_root=str(root / "source"), source_digest=frozen["source_digest"],
        parent_campaign=pinned(parent_root / "manifest.json"), inputs=inputs,
        cohorts=cohorts, shards=shards, training_campaign=None,
        manuscript_reference=pinned(manuscript), v4_reference_metrics=metric_inputs,
        policy={"method": "GraphCast Supplement7.4.2 degree-wise real least squares",
            "paper": "https://arxiv.org/html/2212.12794v2", "lmax": 179,
            "fit_year": 2022, "evaluation_year": 2023, "fit_initializations": 32,
            "evaluation_initializations": 122, "lead_hours": list(range(6, 241, 6)),
            "independent_axes": ["predictor", "lead", "variable", "pressure_level", "degree"],
            "uv_tied": False, "regularization": None, "gain_bounds": None,
            "fit_objective": "physical spectral MSE, independent per channel and lead",
            "zero_forecast_power": "identity gain; unidentifiable modes counted, no denominator floor",
            "operator": "S[b A f], including fitted degree zero; no residual or mean correction",
            "variants": ["raw", "projected", "optimal"],
            "feedback": "unfiltered forecast", "clipping": "none",
            "primary_metric": "exact original GraphCast native-grid loss",
            "primary_denominator": "canonical v4 raw GraphCast on paired initialization/lead",
            "spectral_metric": "physical per-channel coefficient MSE; weighted aggregate explicitly secondary",
            "finite_drift": "warning", "known_exposure": "2023 has informed previous analysis; no held-out claim",
            "scope": "public GraphCast_small and principal M1 SWA only"},
        resource_measurement={"memory_gib": 20, "calibration_minutes": 62,
            "evaluation_minutes": 90, "cpus": 4,
            "basis": "Completed17022624:9m57s/9.17GiB; prior development10-11starts30-33min; "
                     "17028677_1-7:17-18starts43-47min/9.9-10.3GiB. "
                     "New calibration8starts and evaluation15-16starts; retain buffers/headroom, no extra qualification job."})
    return seal(root / "manifest.json", manifest)


def verify(root, *, frozen=True):
    manifest = original.verify(root, frozen=frozen)
    if manifest["format"] != "v24_optimal_smoothing_campaign_v1":
        raise ValueError("Wrong campaign format")
    validate_cohorts(*[read(manifest["cohorts"][stage]["path"]) for stage in ("calibrate", "evaluate")])
    return manifest


def fit(root, manifest):
    from src.analysis.optimal_spherical_filter import fit_degree_gains
    paths = original.statistics_paths(root, manifest, "calibrate")
    aggregate, layout = None, None
    expected_branches = ["GC", manifest["model_id"]]
    for path in paths:
        arrays, metadata = original.load_statistics(path, manifest, "calibrate")
        if metadata["branches"] != expected_branches or metadata["lead_hours"] != list(range(6, 241, 6)):
            raise ValueError("Calibration predictor or lead identity differs")
        if aggregate is None:
            layout = metadata["layout"]
            aggregate = {key: np.zeros_like(arrays[key], dtype=np.float64)
                         for key in ("prediction_power", "cross_power")}
        if metadata["layout"] != layout:
            raise ValueError("Calibration channel identity changed")
        expected_shape = (2, 40, len(layout["channels"]), manifest["policy"]["lmax"] + 1)
        for key in aggregate:
            if arrays[key].shape != expected_shape:
                raise ValueError("Calibration must retain every lead/channel/degree")
            aggregate[key] += arrays[key]
    gains, diagnostics = fit_degree_gains(aggregate["prediction_power"], aggregate["cross_power"])
    return seal(root / "filters.json", {
        "format": "v24_optimal_spherical_filters_v1", "campaign_hash": manifest["record_hash"],
        "layout": layout, "lmax": manifest["policy"]["lmax"], "lead_hours": list(range(6, 241, 6)),
        "branches": {branch: {"gains": gains[i].tolist()} for i, branch in enumerate(expected_branches)},
        "development_count": len(paths), "fit_year": 2022, "fit_diagnostics": diagnostics,
        "calibration_inputs": [pinned(path) for path in paths],
        "objective": "sum over development dates and longitudinal modes of |b*f_forecast-f_truth|^2",
        "operator": manifest["policy"]["operator"], "policy": manifest["policy"]})


def sbatch(root, manifest, stage, *, memory=8, minutes=20, array=None, dependency=None):
    command = original.sbatch(root, manifest, stage, memory=memory, minutes=minutes,
                              array=array, dependency=dependency)
    command = [f"--job-name=optblur-M1-{stage}" if item.startswith("--job-name=") else item
               for item in command]
    command[-5] = str(Path(manifest["source_root"]) / WRAPPER)
    return command


def submit(root, manifest):
    budget = manifest["resource_measurement"]
    calibration = submit_once(root, "calibrate", sbatch(root, manifest, "calibrate",
        memory=budget["memory_gib"], minutes=budget["calibration_minutes"],
        array=f"0-{len(manifest['shards']['calibrate'])-1}"))
    fitted = submit_once(root, "fit", sbatch(root, manifest, "fit", memory=8, minutes=20,
                                            dependency=original.job_id(calibration)))
    evaluation = submit_once(root, "evaluate", sbatch(root, manifest, "evaluate",
        memory=budget["memory_gib"], minutes=budget["evaluation_minutes"],
        array=f"0-{len(manifest['shards']['evaluate'])-1}", dependency=original.job_id(fitted)))
    report = submit_once(root, "report", sbatch(root, manifest, "report", memory=12, minutes=30,
                                               dependency=original.job_id(evaluation)))
    return {"calibrate": calibration, "fit": fitted, "evaluate": evaluation, "report": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "submit", "calibrate", "fit", "evaluate", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--parent", type=Path, default=ROOT / PARENT)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == "prepare":
        result = prepare(root, args.workspace, args.parent)
    else:
        manifest = verify(root)
        if args.stage in ("calibrate", "evaluate"):
            from src.analysis.optimal_filter_observer import OptimalFilterObserver
            result = original.run_worker(root, manifest, args.stage, args.task,
                                         observer_class=OptimalFilterObserver)
        elif args.stage == "report":
            from scripts.analyze_models.report_v24_optimal_smoothing import report
            result = report(root, manifest)
        else:
            result = {"fit": fit, "submit": submit}[args.stage](root, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
