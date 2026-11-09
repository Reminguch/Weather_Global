#!/usr/bin/env python3
"""Freeze and run matched 2023 extreme-event evaluation, without training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_v24_spectral_pilot import (
    check_file, frozen_file, protocol_digest, read_json, validate_data, write_json,
)
from src.data_operations.loaders.zarr_metadata import frozen_zarr_metadata, verify_frozen_zarr_metadata

REFERENCE = ROOT / "artifacts/evaluations/v24_mgda_power_2023_20261106"
CLIMATE = ROOT / "data/graphcast/graphcast/dataset/wb2_res1_levels13_1982_2022.zarr"
NEW_FILES = (
    "src/analysis/__init__.py",
    "src/analysis/extremes_climatology.py",
    "src/analysis/extremes_observer.py",
    "scripts/analyze_models/report_v24_extremes_2023.py",
    "scripts/experiments/run_v24_extremes_2023.py",
    "scripts/experiments/v24_extremes_2023.slurm",
)


def prepare(root, reference=REFERENCE, climate=CLIMATE):
    import xarray as xr

    root, reference, climate = (Path(p).resolve() for p in (root, reference, climate))
    if root.exists():
        raise FileExistsError(f"Refusing to replace campaign: {root}")
    for relative in NEW_FILES:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    parent = read_json(reference / "campaign.json")
    if [m["id"] for m in parent["models"]] != ["M1", "mgda_swa"]:
        raise ValueError("Expected frozen principal M1 and relative-power MGDA SWA")
    dates = parent["dates"]
    if len(dates) != 122 or dates != sorted(set(dates)):
        raise ValueError("Expected all 122 unique 2023 initializations")
    prototype = read_json(reference / "shards/00/manifest.json")["protocols"]["spectral"]
    validation = validate_data(Path(prototype["data_path"]), dates)
    climate_metadata = frozen_zarr_metadata(climate)
    with xr.open_zarr(climate, consolidated=True, chunks=None) as historical:
        for name in ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation_6hr"):
            if name not in historical:
                raise ValueError(f"Historical field missing: {name}")
        times = np.asarray(historical.time.values, dtype="datetime64[ns]")
        required = np.arange(np.datetime64("1990-12-31T06", "h"), np.datetime64("2021-01-01T00", "h"), np.timedelta64(6, "h"))
        locations = np.searchsorted(times, required)
        if np.any(locations >= len(times)) or not np.array_equal(times[locations], required):
            raise ValueError("Historical source lacks complete six-hour 1991–2020 coverage and TP24 history")
        history_lat, history_lon = historical.lat.values.copy(), historical.lon.values.copy()
    with xr.open_zarr(prototype["data_path"], consolidated=True, chunks=None) as test:
        if not np.array_equal(history_lat, test.lat.values) or not np.array_equal(history_lon, test.lon.values):
            raise ValueError("Historical and forecast grids differ")
        land = test["land_sea_mask"]
        for dim in tuple(land.dims):
            if dim not in ("lat", "lon"):
                if land.sizes[dim] != 1:
                    raise ValueError("Land mask must be static")
                land = land.isel({dim: 0}, drop=True)
        land = np.asarray(land.transpose("lat", "lon"), dtype=np.float32)
        if not np.isfinite(land).all() or np.any((land < 0) | (land > 1)):
            raise ValueError("Invalid fixed land-sea mask")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "climatology").mkdir()
    np.save(root / "land_mask.npy", land, allow_pickle=False)
    write_json(root / "data_validation.json", validation)
    # Retain the exact inference implementation used by the paired spectral campaign.
    old_hashes = read_json(reference / "source_hashes.json")
    for relative, digest in old_hashes.items():
        check_file({"path": str(reference / "source" / relative), "sha256": digest})
    shutil.copytree(reference / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in NEW_FILES:
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    hashes = {str(p.relative_to(root / "source")): frozen_file(p)["sha256"]
              for p in sorted((root / "source").rglob("*")) if p.is_file()}
    write_json(root / "source_hashes.json", hashes)
    (root / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    (root / "working_tree_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    inputs = [frozen_file(reference / "campaign.json"), frozen_file(reference / "source_hashes.json"),
              climate_metadata, prototype["data_metadata"], *prototype["statistics"],
              {"path": prototype["baseline"]["path"], "sha256": prototype["baseline"]["checkpoint_sha256"]},
              frozen_file(root / "land_mask.npy")]
    for model in parent["models"]:
        inputs.extend(model[key] for key in ("checkpoint_record", "run_config_record", "planned_config_record") if key in model)
    for record in inputs:
        check_file(record)
    for shard in range(parent["shards"]):
        old = reference / f"shards/{shard:02d}"
        new = root / f"shards/{shard:02d}"
        new.mkdir(parents=True)
        shutil.copy2(old / "initializations.json", new / "initializations.json")
        manifest = copy.deepcopy(read_json(old / "manifest.json"))
        manifest["output_dir"] = str(new)
        for task in manifest["task_specs"]:
            task["reuse"] = {}
        manifest["observer_factory"] = "src.analysis.extremes_observer:factory"
        manifest["selection_policy"] = {"selection": "none; fixed existing checkpoints"}
        manifest["evaluation_role"] = "descriptive_2023_extremes"
        manifest["scope"] = "All 122 matched 2023 starts; occurrence and intensity of hot/cold T2m, WS10 and TP24."
        manifest["limitations"] = "Previously explored year; 1-degree ERA5 grid; no station/cyclone verification or training-seed uncertainty."
        manifest["preparation_sources"].append(frozen_file(old / "manifest.json"))
        proto = manifest["protocols"]["spectral"]
        proto["initialization_manifest"] = frozen_file(new / "initializations.json")
        proto["initialization_manifest_sha256"] = proto["initialization_manifest"]["sha256"]
        manifest.pop("manifest_hash", None)
        write_json(new / "manifest.template.json", manifest)
        inputs.append(frozen_file(new / "manifest.template.json"))
    campaign = {
        "schema_version": 1, "protocol_id": root.name, "models": parent["models"], "dates": dates,
        "shards": parent["shards"], "reference_campaign": str(reference), "inputs": inputs,
        "climatology_source": str(climate), "climatology_source_metadata": climate_metadata,
        "climatology": {"years": [1991, 2020], "calendar": "365 days, exclude February 29 reference targets",
                        "window": "circular +/-30 calendar days at the same valid UTC hour",
                        "quantiles": [.02, .05, .95, .98], "estimator": "linear",
                        "observables": ["T2m", "WS10", "TP24"], "hours": [0, 6, 12, 18],
                        "precipitation": "sum four consecutive six-hour accumulations ending at valid time; mm"},
        "events": {"primary": "q98 hot/wind/rain, q2 cold", "sensitivity": "q95 hot/wind/rain, q5 cold",
                   "domains": ["global", "land mask >=0.5"], "rain_eligibility": "historical local TP24 q95 >1mm",
                   "rain_leads": "24h through 240h; four forecast accumulations, no truth-filled early leads",
                   "weights": "spherical cell areas with finite polar caps, normalized over globe",
                   "scoring": "all cases include false alarms; truth-tail MAE/bias/RMSE reported separately"},
        "uncertainty": {"method": "paired circular moving blocks of 5 consecutive starts (15 days)",
                        "replicates": 2000, "seed": 20261106, "scope": "pointwise date-sampling intervals; not training-seed variability"},
        "numerical_policy": "Finite drift and weak metrics are warnings; no numerical-variance or scientific-success gate.",
        "resources": {"climatology": {"tasks": 12, "cpus": 2, "mem_gib": 8, "hours": 2},
                      "gpu": {"tasks": 2 * parent["shards"], "cpus": 8, "mem_gib": 16, "minutes": 62, "gpus": 1},
                      "evidence": "16982943_0/1/4: 22m48–24m22, peak host 7.44–7.73GiB. Climatology: 48 real source chunks in0.202s; 1830x1024 quantile tile0.05144s, extrapolated20min/worker plus staging/I/O. Worker mmap2.7GiB+output0.36GiB and bounded tile buffers."},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "jax", "jaxlib", "xarray", "zarr", "pandas")},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "output_data_dir": str(ROOT / "plots/analyze_models/data/resolution_eval" / root.name),
        "output_image_dir": str(ROOT / "plots/analyze_models/images/resolution_eval" / root.name),
    }
    campaign["campaign_hash"] = protocol_digest(campaign)
    write_json(root / "campaign.json", campaign)
    print(f"Prepared {root}: 12 historical threshold tasks, 16 GPU shards, automatic report", flush=True)


def verify(root):
    meta = read_json(root / "campaign.json")
    expected = meta.pop("campaign_hash")
    if protocol_digest(meta) != expected:
        raise ValueError("Campaign configuration changed")
    meta["campaign_hash"] = expected
    for relative, digest in read_json(root / "source_hashes.json").items():
        check_file({"path": str(root / "source" / relative), "sha256": digest})
    for record in meta["inputs"]:
        check_file(record)
    verify_frozen_zarr_metadata(meta["climatology_source"], meta["climatology_source_metadata"])
    for name, version in meta["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Package version changed: {name}")
    return meta


def bind(root):
    meta = verify(root)
    from src.analysis.extremes_climatology import finalize
    finalize(root / "climatology")
    climate = frozen_file(root / "climatology/metadata.json")
    bound = []
    for shard in range(meta["shards"]):
        path = root / f"shards/{shard:02d}/manifest.json"
        if path.exists():
            raise FileExistsError(path)
        manifest = read_json(path.with_name("manifest.template.json"))
        manifest["extremes"] = {"climatology_dir": str(root / "climatology"),
                                "climatology_metadata": climate, "land_mask": frozen_file(root / "land_mask.npy"),
                                "protocol_id": meta["protocol_id"]}
        manifest["preparation_sources"].extend([climate, frozen_file(root / "campaign.json")])
        manifest["manifest_hash"] = protocol_digest(manifest)
        write_json(path, manifest)
        bound.append(frozen_file(path))
    write_json(root / "bound.json", {"status": "complete", "climatology_metadata": climate, "manifests": bound})


def submit(root):
    meta = verify(root)
    if (root / "submissions.json").exists():
        raise FileExistsError("Submission record exists; inspect existing jobs before resubmitting")
    workspace = Path(os.environ.get("WEATHER_WORKSPACE", ROOT)).resolve()
    wrapper = root / "source/scripts/experiments/v24_extremes_2023.slurm"
    records = {}
    stages = (("climatology", 8, 2, 120, "0-11", False),
              ("bind", 4, 2, 15, None, False),
              ("worker", 16, 8, 62, f"0-{len(meta['models']) * meta['shards'] - 1}", True),
              ("report", 8, 2, 30, None, False))
    dependency = None
    for stage, memory, cpus, minutes, array, gpu in stages:
        command = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={workspace}",
                   f"--job-name=v24-extreme-{stage}", f"--mem={memory}G", f"--cpus-per-task={cpus}",
                   f"--time={minutes}", f"--output={root}/logs/{stage}_%A_%a.out",
                   f"--error={root}/logs/{stage}_%A_%a.err"]
        if array:
            command.append(f"--array={array}")
        if gpu:
            command.extend(["--gres=gpu:1", "--constraint=gpu80", "--qos=gpu-short"])
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        command.extend([str(wrapper), stage, str(root)])
        print(shlex.join(command), flush=True)
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unrecognized sbatch result: {job}")
        records[stage] = {"job_id": job, "command": command, "dependency": dependency}
        write_json(root / "submissions.json", records)
        dependency = job
        print(f"Submitted {stage}: {job}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "submit", "climatology", "bind", "worker", "report", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--climate-source", type=Path, default=CLIMATE)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root, args.reference, args.climate_source)
        return
    if args.stage == "submit":
        submit(root)
        return
    meta = verify(root)
    if args.stage != "verify" and ROOT != root / "source":
        raise RuntimeError("Execution must use the frozen campaign source")
    if args.stage == "climatology":
        subprocess.run([sys.executable, "-m", "src.analysis.extremes_climatology", "--source", meta["climatology_source"],
                        "--output-dir", str(root / "climatology"), "--task-index", str(args.task)], check=True)
    elif args.stage == "bind":
        bind(root)
    elif args.stage in ("worker", "report"):
        bound = read_json(root / "bound.json")
        check_file(bound["climatology_metadata"])
        for record in bound["manifests"]:
            check_file(record)
        if args.stage == "worker":
            from scripts.analyze_models.evaluate_v24_amse_regime import evaluate
            model, shard = divmod(args.task, meta["shards"])
            if not 0 <= model < len(meta["models"]):
                raise ValueError("Invalid model/shard index")
            evaluate(root / f"shards/{shard:02d}/manifest.json", model, "spectral")
        else:
            from scripts.analyze_models.report_v24_extremes_2023 import report
            report(root)


if __name__ == "__main__":
    main()
