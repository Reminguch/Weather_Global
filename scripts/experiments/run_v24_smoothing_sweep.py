#!/usr/bin/env python3
"""Evaluate prespecified output-smoother strengths using frozen GC/M1 forecasts.

Reuses development-fitted profiles; no training, fitting, or model selection.
Only ``submit`` submits jobs. Existing campaigns and snapshots are immutable.
"""
from __future__ import annotations

import argparse
import copy
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

DRIVER = "scripts/experiments/run_v24_smoothing_sweep.py"
WRAPPER = "scripts/experiments/v24_smoothing_sweep.slurm"
OVERLAYS = (DRIVER, WRAPPER, "scripts/experiments/run_v24_smoothing.py",
            "src/analysis/smoothing_sweep_observer.py",
            "scripts/analyze_models/report_v24_smoothing_sweep.py")
WIDTHS = [0., .25, .5, .75, 1., 1.5, 2., 3., 4., 8.]


def freeze_extension(workspace, parent, destination):
    """Copy verified parent code and only explicit extensions into a new snapshot."""
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
        "format": FORMAT, "workspace": str(workspace.resolve()),
        "files": files, "source_digest": _digest(files),
        "parent_snapshot": pinned(parent / "snapshot.json"), "overlays": list(OVERLAYS)})
    return verify_source(destination)


def prepare(root, workspace, parent_root, *, shards=8):
    root, workspace, parent_root = map(lambda p: Path(p).resolve(), (root, workspace, parent_root))
    if root.exists():
        raise FileExistsError("Use a new smoothing sweep directory")
    parent = original.verify(parent_root, frozen=False)
    filters = unseal(parent_root / "filters.json")
    completion = unseal(parent_root / "report.json")
    if (filters["campaign_hash"] != parent["record_hash"] or
            completion["campaign_hash"] != parent["record_hash"] or completion["status"] != "complete"):
        raise ValueError("Parent smoothing campaign is incomplete or mismatched")
    cohort = read(parent["cohorts"]["evaluate"]["path"])
    if parent["model_id"] != "M1" or len(original.expected_ids(cohort)) != 122:
        raise ValueError("This sweep requires principal M1 and the established annual122 cohort")
    if not 1 <= shards <= 122:
        raise ValueError("Invalid shard count")
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "checks", "manifests", "evaluate"):
        (root / sub).mkdir()
    frozen = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    inputs = copy.deepcopy(parent["inputs"])
    inputs += [pinned(parent_root / name) for name in ("manifest.json", "filters.json", "report.json")]
    chunks = np.array_split(np.asarray(cohort["initializations"], dtype=object), shards)
    evaluation_shards = []
    for index, chunk in enumerate(chunks):
        path = root / f"manifests/evaluate_{index:02d}.json"
        write(path, {**cohort, "initializations": list(chunk)})
        record = pinned(path)
        inputs.append(record)
        evaluation_shards.append({"manifest": record, "count": len(chunk)})
    for item in inputs:
        check(item)
    manifest = copy.deepcopy(parent)
    manifest.pop("record_hash")
    manifest.update(format="v24_smoothing_sweep_campaign_v1", workspace=str(workspace),
        source_root=str(root / "source"), source_digest=frozen["source_digest"],
        parent_campaign=pinned(parent_root / "manifest.json"),
        parent_filters=pinned(parent_root / "filters.json"),
        inputs=inputs, shards={"evaluate": evaluation_shards}, training_campaign=None,
        sweep_policy={"fixed_gaussian_widths_degrees": WIDTHS,
            "profile_blend_alphas": [i / 10 for i in range(11)],
            "profiles": ["fitted_gaussian", "fitted_band"],
            "blend": "forecast + alpha * (filtered_forecast - forecast)",
            "selection": "all prespecified points; exploratory 2023 tradeoff curves, no winner selection",
            "refitting": "none; frozen independent GC/M1 development2022 profiles",
            "statistics": "native-grid error/delta moments and actual synthesized-delta spectral moments",
            "spectral_metric": "pool date powers first; mean abs(Pf/Pt-1), 83 channels, degrees1:179,40 leads",
            "undefined_bins": "NaN without denominator floor; report counts",
            "numerical_policy": "finite differences and drift are warnings; nonfinite/corrupt inputs are errors"},
        resource_measurement={"memory_gib": 20, "minutes": 180, "cpus": 4,
            "basis": "Completed17028677_1-7:17/18dates43:03-46:29,peak10.3GiB; "
                "new11 corrections versus2, no zonal diagnostics; balanced15/16date shards; "
                "~3x runtime plus margin,20GiB includes larger saved moments; reuse prior measurements"})
    manifest = seal(root / "manifest.json", manifest)
    copied = {k: v for k, v in filters.items() if k not in ("record_hash", "campaign_hash")}
    seal(root / "filters.json", {**copied, "campaign_hash": manifest["record_hash"],
        "origin_filters": pinned(parent_root / "filters.json"),
        "origin_campaign_hash": parent["record_hash"]})
    return manifest


def verify(root, *, frozen=True):
    manifest = original.verify(root, frozen=frozen)
    if manifest["format"] != "v24_smoothing_sweep_campaign_v1":
        raise ValueError("Not a smoothing-strength sweep")
    filters = unseal(root / "filters.json")
    parent = unseal(manifest["parent_filters"]["path"])
    expected = {k: v for k, v in parent.items() if k not in ("record_hash", "campaign_hash")}
    actual = {k: v for k, v in filters.items()
              if k not in ("record_hash", "campaign_hash", "origin_filters", "origin_campaign_hash")}
    if actual != expected or filters["campaign_hash"] != manifest["record_hash"]:
        raise ValueError("Frozen development filters changed")
    if filters["origin_filters"] != manifest["parent_filters"] or filters["origin_campaign_hash"] != parent["campaign_hash"]:
        raise ValueError("Filter origin identity changed")
    return manifest


def sbatch(root, manifest, stage, *, dependency=None):
    budget = manifest["resource_measurement"]
    command = original.sbatch(root, manifest, stage,
        memory=budget["memory_gib"] if stage == "evaluate" else 8,
        minutes=budget["minutes"] if stage == "evaluate" else 45,
        array=f"0-{len(manifest['shards']['evaluate'])-1}" if stage == "evaluate" else None,
        dependency=dependency)
    command = [arg.replace("--job-name=smth-", "--job-name=smthsweep-") for arg in command]
    command[-5] = str(Path(manifest["source_root"]) / WRAPPER)
    return command


def submission_plan(root, manifest):
    return {"evaluate": sbatch(root, manifest, "evaluate"),
            "report": sbatch(root, manifest, "report", dependency="EVALUATION_ARRAY_JOB_ID")}


def submit(root, manifest):
    evaluation = submit_once(root, "evaluate", sbatch(root, manifest, "evaluate"))
    report = submit_once(root, "report", sbatch(root, manifest, "report", dependency=original.job_id(evaluation)))
    return {"evaluate": evaluation, "report": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "plan", "submit", "evaluate", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == "prepare":
        if args.parent_root is None:
            parser.error("--parent-root is required")
        manifest = prepare(root, args.workspace, args.parent_root, shards=args.shards)
        result = submission_plan(root, manifest)
        seal(root / "submission_plan.json", result)
    else:
        manifest = verify(root, frozen=args.stage in ("evaluate", "report"))
        if args.stage == "evaluate":
            from src.analysis.smoothing_sweep_observer import SmoothingSweepObserver
            result = original.run_worker(root, manifest, "evaluate", args.task, observer_class=SmoothingSweepObserver)
        elif args.stage == "report":
            from scripts.analyze_models.report_v24_smoothing_sweep import report
            result = report(root)
        elif args.stage == "submit":
            result = submit(root, manifest)
        elif args.stage == "plan":
            result = submission_plan(root, manifest)
        else:
            result = {"status": "verified", "campaign_hash": manifest["record_hash"]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
