"""Immutable experiment definitions and explicit Slurm dependency submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from .config import SCHEMA, STAGES, initial_matrix, load_config
from .io import digest, read_json, sha256, versions, write_json
from .numerics import POLICY

ROOT = Path(__file__).resolve().parents[3]


def source_snapshot(workspace, destination):
    workspace, destination = Path(workspace), Path(destination)
    manifest_path = destination / "snapshot.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for relative, expected in manifest["files"].items():
            if sha256(destination / relative) != expected:
                raise ValueError(f"Executed source snapshot changed: {relative}")
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Incomplete source snapshot exists")
    files = {}
    for directory in ("src", "scripts", "third_party/graphcast", "third_party/neuralgcm",
                      "tests/neuralgcm_residual", "configs/experiments/neuralgcm_residual"):
        for source in sorted((workspace / directory).rglob("*")):
            if (not source.is_file() or "__pycache__" in source.parts or "build" in source.parts or
                    any(part.endswith((".egg-info", ".dist-info")) for part in source.parts) or
                    (source.suffix not in (".py", ".sh", ".json", ".nc", ".pkl", ".txt", ".toml") and source.name != "LICENSE")):
                continue
            relative = source.relative_to(workspace)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[str(relative)] = sha256(target)
    manifest = {"files": files, "source_id": digest(files)}
    write_json(manifest_path, manifest, immutable=True)
    return manifest


def prepare_experiment(root, *, data_root=None, prepared_root=None, python=None, era5_source=None,
                       resolutions=("res2p8", "res1p4"), checkpoint_root=None, workspace=ROOT):
    from .data import ARCO_SOURCE
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    data_root = Path(data_root or workspace / "data/neuralgcm").resolve()
    prepared_root = Path(prepared_root or data_root / "prepared/era5_2015_2023").resolve()
    checkpoint_root = Path(checkpoint_root or data_root / "checkpoints").resolve()
    resolutions = tuple(resolutions)
    if not resolutions or len(set(resolutions)) != len(resolutions) or set(resolutions) - {"res2p8", "res1p4"}:
        raise ValueError("Select distinct supported resolutions")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Experiment definitions already exist; select a new experiment root")
    snapshot = source_snapshot(workspace, root / "source")
    configurations = {}
    for config in initial_matrix():
        if config.resolution_id not in resolutions:
            continue
        path = root / "configs" / (config.run_id + ".json")
        write_json(path, config.to_dict(), immutable=True)
        configurations[config.run_id] = {"path": str(path), "sha256": sha256(path), "resolution": config.resolution_id}
    resources = {}
    for rid in resolutions:
        resources[rid] = {"checkpoint_index": str(checkpoint_root / f"{rid}.json"),
                          "prepared": str(prepared_root / rid),
                          "cache_root": str(data_root / "cache_k1" / root.name / rid),
                          "statistics": str(root / "manifests" / f"{rid}_statistics.json"),
                          "validation_origins": str(root / "manifests" / f"{rid}_val_origins.json"),
                          "test_origins": str(root / "manifests" / f"{rid}_test_origins.json")}
    python = str(Path(python or sys.executable).absolute())
    manifest = {"schema": SCHEMA, "experiment_id": root.name, "source_id": snapshot["source_id"],
                "source_root": str(root / "source"), "workspace": str(workspace), "data_root": str(data_root),
                "python": python, "libraries": versions(), "numerical_policy": POLICY,
                "configs": configurations, "resources": resources, "active_resolutions": list(resolutions),
                "era5_source": era5_source or ARCO_SOURCE, "seed": 22,
                "production_status": "requires_real_model_gates_and_measured_resources"}
    write_json(manifest_path, manifest, immutable=True)
    for name in ("logs", "checks", "manifests", "baselines", "reports", "submissions"):
        (root / name).mkdir(exist_ok=True)
    return manifest


def load_experiment(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unknown experiment schema")
    if manifest["numerical_policy"] != POLICY:
        raise ValueError("Numerical execution policy differs")
    active = manifest.get("active_resolutions", ["res2p8", "res1p4"])
    expected_ids = {c.run_id for c in initial_matrix() if c.resolution_id in active}
    if (not active or len(active) != len(set(active)) or set(active) - {"res2p8", "res1p4"}
            or set(manifest["resources"]) != set(active) or set(manifest["configs"]) != expected_ids):
        raise ValueError("Active resolution matrix is inconsistent")
    snapshot = source_snapshot(manifest["workspace"], manifest["source_root"])
    if snapshot["source_id"] != manifest["source_id"]:
        raise ValueError("Source identity differs")
    for entry in manifest["configs"].values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Locked run configuration changed")
    return manifest


def resolve_resources(manifest, resolution):
    resources = dict(manifest["resources"][resolution])
    index = read_json(resources["checkpoint_index"])
    resources.update(checkpoint=index["path"], checkpoint_sha256=index["sha256"])
    if sha256(resources["checkpoint"]) != resources["checkpoint_sha256"]:
        raise ValueError("Pinned checkpoint changed")
    return resources


def targets(manifest, stage, run_id=None, resolution=None):
    if stage in ("pretrain", "finetune", "evaluate"):
        if resolution:
            raise ValueError("Training/evaluation selects run-id, not resolution")
        ids = [run_id] if run_id else list(manifest["configs"])
        for rid in ids:
            if rid not in manifest["configs"]:
                raise ValueError(f"Unknown run ID: {rid}")
        return [{"run_id": rid} for rid in ids]
    if run_id:
        raise ValueError("Shared stages select resolution, not run-id")
    if stage == "report":
        return [{}]
    if resolution and resolution not in manifest["resources"]:
        raise ValueError("Resolution is outside this experiment")
    return [{"resolution": rid} for rid in ([resolution] if resolution else manifest["resources"])]


def submit(root, stage, *, run_id=None, resolution=None, shard_id=None, after=(), dry_run=False,
           cpus=8, memory_gb=None, hours=None, account=None, partition=None, constraint=None, qos=None,
           phase="inspect", resume=None, start=None, end=None):
    root = Path(root).resolve()
    manifest = load_experiment(root)
    if stage == "preflight":
        if qos not in (None, "gpu-test"):
            raise ValueError("User policy requires gpu-test for all GPU preflight checks")
        if hours is not None and hours > 1:
            raise ValueError("GPU preflight checks must fit one hour; split longer checks")
        qos, hours = "gpu-test", hours or 1
    if memory_gb is None or hours is None:
        if stage in ("preflight", "data"):
            memory_gb, hours = memory_gb or 64, hours or 2
        else:
            raise ValueError("Supply explicit --memory-gb and --hours based on measured profiles")
    if any(not str(job).isdigit() for job in after):
        raise ValueError("afterok dependencies must be Slurm job IDs")
    plans = []
    expanded = []
    for target in targets(manifest, stage, run_id, resolution):
        if stage == "cache" and shard_id is None:
            cache_manifest = read_json(Path(manifest["resources"][target["resolution"]]["cache_root"]) / "manifest.json")
            expanded.extend([{**target, "shard_id": s["id"]} for s in cache_manifest["shards"]])
        else:
            expanded.append(target)
    for target in expanded:
        resolved = (target.get("resolution") or manifest["configs"][target["run_id"]]["resolution"]) if target else None
        if stage in ("pretrain", "finetune", "evaluate") and not after:
            required = "verify-cache" if stage == "pretrain" else "pretrain" if stage == "finetune" else "finetune"
            receipt_id = resolved if stage == "pretrain" else target["run_id"]
            if not (root / "checks" / f"{required}_{receipt_id}.json").exists():
                if not dry_run:
                    raise ValueError(f"Missing successful {required} receipt; specify upstream --after job IDs")
        command = [manifest["python"], str(Path(manifest["source_root"]) / "scripts/experiments/run_neuralgcm_residual.py"),
                   "execute", "--experiment-root", str(root), "--stage", stage, "--phase", phase]
        for key, value in target.items():
            command += ["--" + key.replace("_", "-"), value]
        for key, value in (("shard-id", shard_id), ("resume", resume), ("start", start), ("end", end)):
            if value:
                command += ["--" + key, str(value)]
        name = "ngcm-" + stage + "-" + (target.get("run_id") or target.get("resolution") or "all")
        if target.get("shard_id"):
            name += "-" + target["shard_id"]
        batch = ["sbatch", "--parsable", "--job-name", name, "--cpus-per-task", str(cpus),
                 "--mem", f"{memory_gb}G", "--time", f"{hours}:00:00",
                 "--output", str(root / "logs" / (name + "-%j.out")),
                 "--error", str(root / "logs" / (name + "-%j.err"))]
        if stage not in ("data", "report"):
            batch += ["--gres", "gpu:1"]
        for key, value in (("account", account), ("partition", partition), ("constraint", constraint), ("qos", qos)):
            if value:
                batch += ["--" + key, str(value)]
        if after:
            batch += ["--dependency", "afterok:" + ":".join(map(str, after))]
        shell = "#!/usr/bin/env bash\nset -euo pipefail\nulimit -c 0\nexport XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        shell += "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1\nunset PYTHONPATH PYTHONHOME\n"
        if stage not in ("data", "report"):
            shell += "export JAX_PLATFORMS=cuda\n"
        shell += f"export OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus}\n"
        shell += "cd " + shlex.quote(manifest["source_root"]) + "\nexec " + shlex.join(command) + "\n"
        plan = {"stage": stage, **target, "command": command, "sbatch": batch, "script": shell,
                "afterok": list(after), "source_id": manifest["source_id"]}
        if not dry_run:
            result = subprocess.run(batch, input=shell, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError(f"Unexpected Slurm job ID: {result.stdout}")
            plan["job_id"] = job_id
            write_json(root / "submissions" / f"{job_id}.json", plan, immutable=True)
        plans.append(plan)
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--experiment-root", required=True, type=Path)
    prepare.add_argument("--data-root", type=Path)
    prepare.add_argument("--prepared-root", type=Path)
    prepare.add_argument("--checkpoint-root", type=Path)
    prepare.add_argument("--python")
    prepare.add_argument("--era5-source")
    prepare.add_argument("--resolutions", nargs="+", choices=("res2p8", "res1p4"), default=("res2p8", "res1p4"))
    for name in ("submit", "execute"):
        p = commands.add_parser(name)
        p.add_argument("--experiment-root", required=True, type=Path)
        p.add_argument("--stage", choices=STAGES, required=True)
        p.add_argument("--resolution", choices=("res2p8", "res1p4"))
        p.add_argument("--run-id")
        p.add_argument("--shard-id")
        p.add_argument("--phase", choices=("inspect", "numerical", "profile"), default="inspect")
        p.add_argument("--resume", type=Path)
        p.add_argument("--start")
        p.add_argument("--end")
        if name == "submit":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--after", action="append", default=[])
            p.add_argument("--cpus", type=int, default=8)
            p.add_argument("--memory-gb", type=int)
            p.add_argument("--hours", type=int)
            p.add_argument("--account")
            p.add_argument("--partition")
            p.add_argument("--constraint")
            p.add_argument("--qos")
    args = parser.parse_args(argv)
    kwargs = vars(args).copy()
    command, root = kwargs.pop("command"), kwargs.pop("experiment_root")
    if command == "prepare":
        result = prepare_experiment(root, **kwargs)
    elif command == "submit":
        result = submit(root, **kwargs)
    else:
        manifest = load_experiment(root)
        executed_root = Path(__file__).resolve().parents[3]
        snapshot = Path(manifest["source_root"])
        if executed_root != snapshot:
            # Jobs and local workers run exactly the same immutable source.
            os.execv(manifest["python"], [manifest["python"], str(snapshot / "scripts/experiments/run_neuralgcm_residual.py"),
                                        *(argv if argv is not None else sys.argv[1:])])
        from .worker import execute
        result = execute(root, manifest, **kwargs)
    import json
    print(json.dumps(result, indent=2, default=str))
