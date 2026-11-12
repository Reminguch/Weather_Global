#!/usr/bin/env python3
"""Portable preparation for the existing fitted-filter and strength-sweep engines.

Forecasting, fitting, spectral moments and reporting use the established runtime.
This adapter supplies local input paths and fixed cohorts; it never submits jobs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments import run_v24_smoothing as original
from scripts.experiments import run_v24_smoothing_sweep as strength
from scripts.experiments.frozen_runtime import freeze_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, unseal, write

DRIVER = "scripts/experiments/run_v24_portable_smoothing.py"
WRAPPER = "scripts/experiments/v24_portable_smoothing.sh"


def require(value, message):
    if not value:
        raise ValueError(message)


def data_metadata(path):
    path = Path(path).resolve()
    if path.is_file():
        return pinned(path)
    for name in (".zmetadata", "metadata.json", "zarr.json", ".zgroup"):
        if (path / name).is_file():
            return pinned(path / name)
    raise FileNotFoundError(f"No supported data metadata at {path}")


def validate_cohort(package_root, supplied, reference_name, count, year):
    reference_path = Path(package_root) / "portable" / reference_name
    payload, reference = read(supplied), read(reference_path)
    actual_ids, reference_ids = original.expected_ids(payload), original.expected_ids(reference)
    require(actual_ids == reference_ids and len(actual_ids) == count,
            f"Cohort differs from the fixed {year} reference")
    require(all(value.startswith(str(year)) for value in actual_ids), "Cohort year differs")
    return payload


def make_shards(root, stage, payload, count, *, retain_first=True):
    entries = payload["initializations"]
    if retain_first:
        chunks = [[entries[0]], *[list(part) for part in
            np.array_split(np.asarray(entries[1:], dtype=object), count - 1)]]
    else:
        chunks = [list(part) for part in np.array_split(np.asarray(entries, dtype=object), count)]
    records = []
    for index, chunk in enumerate(chunks):
        path = root / f"manifests/{stage}_{index:02d}.json"
        write(path, {**payload, "initializations": chunk})
        records.append({"manifest": pinned(path), "count": len(chunk)})
    return records


def prepare(package_root, experiment, *, checkpoint, run_config, model_id,
            development_data, annual_data, development_manifest, annual_manifest,
            baseline_checkpoint, stats_dir):
    package_root, root = Path(package_root).resolve(), Path(experiment).resolve()
    require(package_root == ROOT, "Run prepare from this package's driver")
    if root.exists():
        raise FileExistsError("Use a new smoothing campaign directory")
    require(model_id not in ("", "GC", ".", "..") and "/" not in model_id and "\\" not in model_id,
            "Use a distinct path-safe model identifier")
    development = validate_cohort(package_root, development_manifest, "development32.json", 32, 2022)
    annual = validate_cohort(package_root, annual_manifest, "annual122.json", 122, 2023)
    configuration = read(run_config)
    require(configuration.get("model_family", "gc_mamba") in ("gc_mamba", "full_gc"),
            "Unsupported model family")
    require("architecture" in configuration, "Run configuration lacks architecture")
    inputs = [pinned(checkpoint), pinned(run_config), pinned(baseline_checkpoint),
              data_metadata(development_data), data_metadata(annual_data)]
    inputs.extend(pinned(Path(stats_dir) / name) for name in
                  ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"))
    reference = read(package_root / "portable/input_reference.json")
    baseline_stats = {("baseline_checkpoint", ""): Path(baseline_checkpoint),
        **{("stats_dir", name): Path(stats_dir) / name for name in
           ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")}}
    expected = {(entry["input"], entry.get("relative_path", "")): entry["sha256"]
                for entry in reference["files"]}
    for identity, path in baseline_stats.items():
        require(identity in expected and pinned(path)["sha256"] == expected[identity],
                f"Baseline/statistics differ from the package reference: {identity}")
    for entry in inputs:
        check(entry)
    for sub in ("logs", "checks", "manifests", "calibrate", "evaluate"):
        (root / sub).mkdir(parents=True)
    cohorts = {}
    shards = {}
    for stage, payload, count in (("calibrate", development, 4), ("evaluate", annual, 8)):
        cohort_path = root / f"manifests/{stage}.json"
        write(cohort_path, payload)
        cohorts[stage] = pinned(cohort_path)
        inputs.append(cohorts[stage])
        shards[stage] = make_shards(root, stage, payload, count)
        inputs.extend(shard["manifest"] for shard in shards[stage])
    frozen = freeze_source(package_root, root / "source")
    return seal(root / "manifest.json", {
        "format": "v24_smoothing_campaign_v1", "portable_format": "v24_portable_smoothing_v1",
        "workspace": str(package_root), "experiment": str(root),
        "source_root": str(root / "source"), "source_digest": frozen["source_digest"],
        "model_id": model_id, "model_family": configuration.get("model_family", "gc_mamba"),
        "training_campaign": None, "checkpoint": pinned(checkpoint), "run_config": pinned(run_config),
        "architecture": configuration["architecture"], "inputs": inputs,
        "baseline": pinned(baseline_checkpoint), "stats_dir": str(Path(stats_dir).resolve()),
        "data": {"calibrate": str(Path(development_data).resolve()),
                 "evaluate": str(Path(annual_data).resolve())},
        "cohorts": cohorts, "shards": shards, "target_steps": 40,
        "policy": {"eta": .01, "lmax": 179, "knots": [1, 10, 40, 179],
            "gaussian_widths_degrees": [0, .25, .5, 1, 2, 4, 8],
            "lead_group_ends_hours": [24, 72, 168, 240], "uv_tied": True,
            "clipping": "none", "feedback": "raw forecast", "fit_year": 2022,
            "evaluation_year": 2023, "finite_drift": "warning",
            "known_exposure": "2023 has informed previous analysis; no held-out claim",
            "uncertainty_unit": "initialization date; overlapping forecasts are dependent"},
        "resource_measurement": {"first_initialization_retained": True,
            "memory_gib": 20, "minutes": 150, "cpus": 4,
            "basis": "Completed17028677_1-7:17/18 dates43:03–46:29,peak10.3GiB; "
                     "local runtime must be recorded; four/eight fixed shards"}})


def prepare_sweep(package_root, experiment, parent_root):
    package_root, root, parent_root = map(lambda p: Path(p).resolve(),
                                         (package_root, experiment, parent_root))
    require(package_root == ROOT, "Run prepare-sweep from this package's driver")
    if root.exists():
        raise FileExistsError("Use a new smoothing-strength directory")
    parent = original.verify(parent_root, frozen=False)
    require(parent.get("portable_format") == "v24_portable_smoothing_v1",
            "Strength sweep requires a portable fitted-filter parent")
    filters, completion = unseal(parent_root / "filters.json"), unseal(parent_root / "report.json")
    require(filters["campaign_hash"] == parent["record_hash"]
            and completion["campaign_hash"] == parent["record_hash"]
            and completion["status"] == "complete", "Parent filters/report are incomplete or mismatched")
    cohort = validate_cohort(package_root, parent["cohorts"]["evaluate"]["path"],
                            "annual122.json", 122, 2023)
    for sub in ("logs", "checks", "manifests", "evaluate"):
        (root / sub).mkdir(parents=True)
    frozen = freeze_source(package_root, root / "source")
    require(frozen["source_digest"] == parent["source_digest"],
            "Use the same immutable package source for parent and strength sweep")
    inputs = copy.deepcopy(parent["inputs"])
    inputs.extend(pinned(parent_root / name) for name in ("manifest.json", "filters.json", "report.json"))
    shards = make_shards(root, "evaluate", cohort, 8, retain_first=False)
    inputs.extend(shard["manifest"] for shard in shards)
    manifest = copy.deepcopy(parent)
    manifest.pop("record_hash")
    manifest.update(format="v24_smoothing_sweep_campaign_v1", portable_format="v24_portable_smoothing_sweep_v1",
        experiment=str(root), source_root=str(root / "source"), source_digest=frozen["source_digest"],
        parent_campaign=pinned(parent_root / "manifest.json"), parent_filters=pinned(parent_root / "filters.json"),
        inputs=inputs, shards={"evaluate": shards}, training_campaign=None,
        sweep_policy={"fixed_gaussian_widths_degrees": strength.WIDTHS,
            "profile_blend_alphas": [i / 10 for i in range(11)],
            "profiles": ["fitted_gaussian", "fitted_band"],
            "blend": "forecast + alpha * (filtered_forecast - forecast)",
            "selection": "all prespecified points; exploratory 2023 tradeoff curves, no winner selection",
            "refitting": f"none; frozen independent GC/{parent['model_id']} development2022 profiles",
            "statistics": "native-grid error/delta moments and actual synthesized-delta spectral moments",
            "spectral_metric": "pool date powers first; mean abs(Pf/Pt-1),83channels,degrees1:179,40leads",
            "undefined_bins": "NaN without denominator floor; report counts",
            "numerical_policy": "finite differences and drift are warnings; nonfinite/corrupt inputs are errors"},
        resource_measurement={"memory_gib": 20, "minutes": 180, "cpus": 4,
            "basis": "Completed17028677_1-7:17/18dates43:03–46:29,peak10.3GiB; "
                     "11 corrections versus2; balanced15/16date shards, ~3x runtime plus margin"})
    for record in inputs:
        check(record)
    manifest = seal(root / "manifest.json", manifest)
    copied = {k: v for k, v in filters.items() if k not in ("record_hash", "campaign_hash")}
    seal(root / "filters.json", {**copied, "campaign_hash": manifest["record_hash"],
        "origin_filters": pinned(parent_root / "filters.json"), "origin_campaign_hash": parent["record_hash"]})
    return manifest


def verify(root, *, sweep=False, frozen=True):
    manifest = (strength.verify if sweep else original.verify)(root, frozen=frozen)
    expected = "v24_portable_smoothing_sweep_v1" if sweep else "v24_portable_smoothing_v1"
    require(manifest.get("portable_format") == expected, "Unexpected portable smoothing campaign")
    require(manifest.get("training_campaign") is None, "Unexpected automatic downstream campaign")
    return manifest


def report_sweep(root, manifest):
    from scripts.analyze_models.report_v24_smoothing_sweep import report
    result = report(root, manifest)
    path = Path(result["data_dir"]) / "report.md"
    text = path.read_text().replace("# GC/M1 smoothing strength tradeoffs",
                                   f"# GC/{manifest['model_id']} smoothing strength tradeoffs")
    text = text.replace("fixed GC and M1 checkpoints.", f"fixed GC and {manifest['model_id']} checkpoints.")
    path.write_text(text)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    stages = result.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare")
    prep.add_argument("--package-root", type=Path, default=ROOT)
    prep.add_argument("--experiment", type=Path, required=True)
    for name in ("checkpoint", "run-config", "development-data", "annual-data", "development-manifest",
                 "annual-manifest", "baseline-checkpoint", "stats-dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--model-id", required=True)
    sweep = stages.add_parser("prepare-sweep")
    sweep.add_argument("--package-root", type=Path, default=ROOT)
    sweep.add_argument("--experiment", type=Path, required=True)
    sweep.add_argument("--parent-root", type=Path, required=True)
    for name in ("calibrate", "fit", "evaluate", "report", "sweep-evaluate", "sweep-report"):
        stage = stages.add_parser(name)
        stage.add_argument("--experiment", type=Path, required=True)
        if name in ("calibrate", "evaluate", "sweep-evaluate"):
            stage.add_argument("--task", type=int, required=True)
    return result


def main():
    args = parser().parse_args()
    root = args.experiment.resolve()
    if args.stage == "prepare":
        values = vars(args).copy()
        values.pop("stage")
        result = prepare(**values)
    elif args.stage == "prepare-sweep":
        result = prepare_sweep(args.package_root, root, args.parent_root)
    else:
        is_sweep = args.stage.startswith("sweep-")
        manifest = verify(root, sweep=is_sweep)
        if args.stage in ("calibrate", "evaluate", "sweep-evaluate"):
            stage = "evaluate" if is_sweep else args.stage
            require(0 <= args.task < len(manifest["shards"][stage]), "Invalid shard index")
            observer_class = None
            if is_sweep:
                from src.analysis.smoothing_sweep_observer import SmoothingSweepObserver
                observer_class = SmoothingSweepObserver
            result = original.run_worker(root, manifest, stage, args.task, observer_class=observer_class)
        elif args.stage == "fit":
            result = original.fit(root, manifest)
        elif args.stage == "report":
            result = original.report(root, manifest)
        else:
            result = report_sweep(root, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
