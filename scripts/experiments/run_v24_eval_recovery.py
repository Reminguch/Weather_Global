"""Prepare immutable runtimes and recover the September 2026 evaluations only.

The original training snapshots, manifests and checkpoints are never rewritten.
Spatial model files come from its original checksum-verified snapshot; supporting
modules come from the complete, verified AMSE snapshot and are recorded explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "amse": "artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919",
    "spatial": "artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916_8k",
}
SELF = "scripts/experiments/run_v24_eval_recovery.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def copy_verified(source, destination, expected):
    if sha(source) != expected:
        raise ValueError(f"Source checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha(destination) != expected:
        raise ValueError(f"Copied checksum mismatch: {destination}")


def prepare(workspace, family, recovery):
    if recovery.exists():
        raise FileExistsError(f"Recovery already exists: {recovery}")
    experiment = workspace / EXPERIMENTS[family]
    provider = workspace / EXPERIMENTS["amse"] / "source"
    parent = read(provider.parent / "manifest.json")
    if sha(provider / "snapshot.json") != parent["source_digest"]:
        raise ValueError("Supporting source manifest changed")
    runtime = recovery / "source"
    runtime.mkdir(parents=True)
    files = read(provider / "snapshot.json")["files"]
    for relative, expected in files.items():
        copy_verified(provider / relative, runtime / relative, expected)
    overrides = {}
    if family == "spatial":
        original = experiment / "source"
        for line in (original / "SHA256SUMS").read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            copy_verified(original / "snapshot" / relative, runtime / relative, expected)
            overrides[relative] = expected
    else:
        relative = "scripts/analyze_models/evaluate_v24_amse_pair.py"
        copy_verified(workspace / relative, runtime / relative, sha(workspace / relative))
    for relative in (SELF, "scripts/experiments/v24_eval_recovery.slurm"):
        copy_verified(workspace / relative, runtime / relative, sha(workspace / relative))
    inputs = {}

    def record(path):
        path = Path(path)
        if not path.is_absolute():
            path = workspace / path
        inputs[str(path)] = sha(path)

    record(experiment / "manifest.json")
    manifest = read(experiment / "manifest.json")
    for entry in manifest["entries"]:
        record(entry["config"])
        cfg = read(entry["config"])
        record(cfg["data"]["baseline_checkpoint"])
        for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"):
            record(Path(cfg["data"]["stats_dir"]) / name)
        directory = Path(entry.get("run_dir", experiment / "runs" / entry["name"]))
        record(directory / "run_config.json")
        if family == "amse":
            completed = read(experiment / "checks" / f"{entry['name']}_completed.json")
            if not completed["passed"] or sha(completed["checkpoint"]) != completed["sha256"]:
                raise ValueError("Completed training checkpoint changed")
            record(completed["checkpoint"])
            record(directory / "swa/swa_step00500-02000.pkl")
        else:
            if sha(entry["init_from"]) != entry["init_sha256"]:
                raise ValueError("Shared starting checkpoint changed")
            record(entry["init_from"])
            record(directory / "checkpoints/checkpoint_step00008000.pkl")
            record(directory / "swa/swa_step02000-08000.pkl")
    if manifest.get("evaluation_reference"):
        record(manifest["evaluation_reference"])
    files = {str(p.relative_to(runtime)): sha(p) for p in sorted(runtime.rglob("*"))
             if p.is_file() and p.suffix in (".py", ".sh", ".slurm")}
    write(recovery / "manifest.json", {
        "family": family, "workspace": str(workspace), "experiment": str(experiment),
        "runtime": str(runtime), "created_utc": datetime.now(timezone.utc).isoformat(),
        "supporting_source": str(provider), "supporting_source_digest": parent["source_digest"],
        "original_spatial_overrides": overrides, "files": files, "inputs": inputs,
    })
    (recovery / "logs").mkdir()
    print(recovery / "manifest.json", flush=True)


def verify(recovery):
    manifest = read(recovery / "manifest.json")
    runtime = Path(manifest["runtime"])
    for relative, expected in manifest["files"].items():
        if sha(runtime / relative) != expected:
            raise ValueError(f"Recovery source changed: {relative}")
    for path, expected in manifest["inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"Recovery input changed: {path}")
    return manifest


def isolated_command(runtime, workspace, args):
    """Resolve scripts and data independently; never import workspace model code."""
    args = list(map(str, args))
    args[0] = str(runtime / args[0])
    for option in ("--ckpt-in", "--stats-dir", "--ckpt"):
        if option in args:
            index = args.index(option) + 1
            if not Path(args[index]).is_absolute():
                args[index] = str(workspace / args[index])
    return [sys.executable, "-u", *args]


def check_dataset(path, year):
    import numpy as np
    import xarray as xr

    with xr.open_zarr(path, consolidated=True) as dataset:
        subset = dataset.sel(time=slice(str(year), str(year)))
        if subset.sizes.get("time", 0) < 40:
            raise ValueError(f"Missing evaluation-year coverage: {path}, {year}")
        sample = subset["2m_temperature"].isel(time=0, lat=90, lon=0).load()
        if not np.isfinite(sample.values).all():
            raise ValueError(f"Non-finite sample in evaluation dataset: {path}")
        return {"path": str(path), "year": year, "sizes": dict(subset.sizes),
                "first_time": str(subset.time.values[0]), "last_time": str(subset.time.values[-1]),
                "sample_t2m": float(sample.values)}


def run_spatial(manifest, recovery, phase, task):
    runtime, workspace = Path(manifest["runtime"]), Path(manifest["workspace"])
    experiment = Path(manifest["experiment"])
    relative = Path(EXPERIMENTS["spatial"]) / "source/driver_8k.py"
    spec = importlib.util.spec_from_file_location("spatial_recovery_driver", runtime / relative)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    # The historical driver prepends the live workspace on import. Remove it.
    sys.path[:] = [str(runtime), *[p for p in sys.path if p not in (str(workspace), str(runtime), "")]]
    driver.EXPERIMENT = experiment
    original = read(experiment / "manifest.json")
    driver.REFERENCE = Path(original["evaluation_reference"])
    if phase == "report":
        (experiment / "results").mkdir(exist_ok=True)
        driver.summarize(original)
        return
    entry = original["entries"][task]
    smoke_output = recovery / "smoke" / f"spatial_{task}.json"

    class SmokeComplete(Exception):
        pass

    def call(*args):
        args = list(map(str, args))
        if phase == "smoke":
            for option, value in (("--ckpt", experiment / "runs" / entry["name"] /
                                   "checkpoints/checkpoint_step00008000.pkl"),
                                  ("--target-steps", 2), ("--n-samples", 1), ("--out-json", smoke_output)):
                args[args.index(option) + 1] = str(value)
        subprocess.run(isolated_command(runtime, workspace, args), cwd=runtime, check=True)
        if phase == "smoke":
            value = read(smoke_output)
            if value.get("evaluated_samples") != 1 or len(value["original_graphcast_loss"]["full_per_step"]) != 2:
                raise ValueError("Spatial smoke evaluation incomplete")
            raise SmokeComplete

    driver.run = call
    smoke_output.parent.mkdir(exist_ok=True)
    try:
        driver.evaluate(entry)
    except SmokeComplete:
        pass


def run(recovery, phase, task):
    manifest = verify(recovery)
    runtime = Path(manifest["runtime"])
    if ROOT != runtime:
        raise ValueError("Run from the frozen recovery source")
    os.environ["PYTHONPATH"] = f"{runtime}:{runtime / 'third_party/graphcast'}"
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    if manifest["family"] == "amse":
        from scripts.analyze_models import evaluate_v24_amse_pair as workflow
        experiment = Path(manifest["experiment"])
        metadata = workflow.validate_evaluation_data(experiment)
        if phase == "validate":
            datasets = {}
            for arm in range(2):
                for tag in workflow.TAGS:
                    for stage in ("exact", "spectral"):
                        spec = workflow.make_specification(experiment, arm, tag, stage)
                        workflow.verify_specification(spec)
                        if stage not in datasets:
                            datasets[stage] = check_dataset(spec["data_path"], spec["evaluation_year"])
            write(recovery / "data_validation.json", {"metadata": metadata, "datasets": datasets})
        elif phase == "report":
            workflow.report(experiment)
        elif phase == "smoke":
            # Use a separate experiment view so original preflight records stay intact.
            view = recovery / "smoke" / str(task)
            write(view / "manifest.json", read(experiment / "manifest.json"))
            checkpoint = Path(read(experiment / "manifest.json")["entries"][task]["run_dir"]) / "checkpoints/checkpoint_step00002000.pkl"
            workflow.smoke(view, task, checkpoint)
        else:
            workflow.evaluate(experiment, task)
    elif phase != "validate":
        run_spatial(manifest, recovery, phase, task)
    else:
        # Import the actual frozen model, not the changing workspace module.
        from src.models.mamba.v24_Ilya import config, evaluation
        for module in (config, evaluation):
            if not Path(module.__file__).is_relative_to(runtime):
                raise ValueError(f"Unfrozen module import: {module.__file__}")
        write(recovery / "import_validation.json", {m.__name__: m.__file__ for m in (config, evaluation)})
        write(recovery / "data_validation.json", check_dataset(
            "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/"
            "wb2_res1_levels13_2015_2022.zarr", 2022))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate", "smoke", "evaluate", "report"))
    parser.add_argument("--family", choices=EXPERIMENTS)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.family is None:
            parser.error("prepare requires --family")
        prepare(args.workspace.resolve(), args.family, args.recovery.resolve())
    else:
        run(args.recovery.resolve(), args.phase, args.task)


if __name__ == "__main__":
    main()
