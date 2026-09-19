#!/usr/bin/env python3
"""Prepare, gate and run matched 4k online/cached-stepwise experiments."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v24_Ilya.checkpoint import atomic_json_dump, file_sha256
from src.models.mamba.v24_Ilya.training.cached_checkpoint import (
    build_provenance, code_fingerprints, digest_json, latest_checkpoint_for_run,
    project_shared_initialization, reconcile_run_metrics, require_production_gates,
)
from src.models.mamba.v24_Ilya.training.config import load_training_config


def write_once_json(path: Path, value: dict):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing artifact differs: {path}")
    else:
        atomic_json_dump(value, path)


def freeze_source(workspace: Path, destination: Path) -> dict:
    """Capture Python and launcher sources before jobs can become pending."""
    required = {"src/models/mamba/v24_Ilya/training/" + name for name in
                ("cached_parity.py", "cached_report.py", "cached_validation.py", "cached_stepwise.py")}
    manifest_path = destination / "snapshot.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not required.issubset(manifest["files"]):
            raise ValueError("Source snapshot lacks a required parity, validation or final-report module")
        for relative, expected in manifest["files"].items():
            if file_sha256(destination / relative) != expected:
                raise ValueError(f"Snapshot source was modified: {relative}")
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Incomplete source snapshot: {destination}")
    missing = [relative for relative in sorted(required) if not (workspace / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Implement required pipeline modules before freezing source: {missing}")
    paths = []
    for directory in ("src", "third_party/graphcast", "scripts"):
        for path in (workspace / directory).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".sh", ".slurm"):
                paths.append(path)
    files = {}
    for source in sorted(paths):
        relative = source.relative_to(workspace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[str(relative)] = file_sha256(target)
    manifest = {"format": "v24_pair_source_v1", "files": files, "source_digest": digest_json(files)}
    atomic_json_dump(manifest, manifest_path)
    return manifest


def prepare_experiment(experiment_root: Path, *, cache_root: Path, workspace: Path = ROOT,
                       training_config: Path | None = None, cached_only: bool = False,
                       smoke_time_minutes: int = 120, smoke_memory_gib: int = 128,
                       job_name: str | None = None) -> dict:
    experiment_root, cache_root, workspace = (Path(p).resolve() for p in (experiment_root, cache_root, workspace))
    cache_manifest = json.loads((cache_root / "manifest.json").read_text())
    ready = json.loads((cache_root / "READY.json").read_text())
    if ready.get("partial") or ready.get("manifest_sha256") != cache_manifest.get("manifest_sha256"):
        raise ValueError("Experiment requires a complete verified GC cache")
    if smoke_time_minutes <= 0 or smoke_memory_gib <= 0:
        raise ValueError("Smoke time and memory must be positive")
    template_path = training_config or workspace / "configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json"
    template = json.loads(template_path.read_text())
    for key in ("prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir"):
        path = Path(template["data"][key])
        template["data"][key] = str(path.resolve() if path.is_absolute() else (workspace / path).resolve())
    if training_config is None:
        template["optimizer"].update(max_steps=4000, checkpoint_every=500, learning_rate=1e-4,
                                  learning_rate_schedule="constant", end_learning_rate=None, warmup_steps=0, seed=22)
        template["architecture"].update(resolution=1.0, mesh_size=5, width=512, residual_width=None,
                                     residual_initialization="baseline_overlay", baseline_msg_steps=16,
                                     residual_msg_steps=2, temporal_d_inner=64, temporal_bc_groups=2,
                                     temporal_stateful=True, temporal_dropout=0.0)
        template["validation"].update(enabled=True, every_steps=500, num_segments=None, final_num_segments=None)
        template["sequence"].update(segment_steps=96, bptt_steps=24, ar_tail_k=20,
                                 feedback_mode="baseline", temporal_state_policy="carry")
        template["objective"] = {"loss_mode": "all_steps"}
        template["distributed"].update(mode="single", num_devices=1, per_device_batch_size=1)
    configs = {}
    for name in ("online", "cached", "smoke_online", "smoke_cached"):
        config = copy.deepcopy(template)
        smoke = name.startswith("smoke_")
        backend = name.removeprefix("smoke_")
        config["output"] = {"output_root": str(experiment_root / ("smoke" if smoke else "runs")), "run_name": backend}
        if smoke:
            config["optimizer"].update(max_steps=20, checkpoint_every=20)
            config["validation"]["enabled"] = False
        path = experiment_root / "configs" / f"{name}.json"
        write_once_json(path, config)
        load_training_config(path)
        configs[name] = str(path)
    execution = {"backend": "cached_stepwise", "cache_root": str(cache_root), "prefetch_batches": 0,
                 "expected_manifest_sha256": cache_manifest["manifest_sha256"]}
    snapshot = freeze_source(workspace, experiment_root / "source")
    manifest = {"format": "v24_cached_pair_v1", "workspace": str(workspace), "experiment_root": str(experiment_root),
                "cache_root": str(cache_root), "source_root": str(experiment_root / "source"),
                "source_digest": snapshot["source_digest"],
                "shared_init": str(experiment_root / "shared/checkpoint_step00000000.pkl"), "configs": configs,
                "reports": {name: str(experiment_root / "reports" / f"{name}.json")
                            for name in ("preflight", "full_parity", "paired20", "benchmark", "resource_check")},
                "execution": execution}
    if training_config is not None or cached_only:
        manifest["production_backends"] = ["cached"] if cached_only else ["online", "cached"]
        manifest["production_schedule"] = {
            "max_steps": template["optimizer"]["max_steps"],
            "checkpoint_every": template["optimizer"]["checkpoint_every"],
            "validation_every": template["validation"]["every_steps"] if template["validation"]["enabled"] else None,
        }
    if smoke_time_minutes != 120 or smoke_memory_gib != 128:
        manifest["smoke_budget"] = {"memory_gib": smoke_memory_gib, "time_minutes": smoke_time_minutes,
                                    "cpus": 8, "gpu": "gpu40&nomig"}
    if job_name:
        manifest["job_name"] = job_name
    write_once_json(experiment_root / "manifest.json", manifest)
    (experiment_root / "reports").mkdir(exist_ok=True)
    (experiment_root / "logs").mkdir(exist_ok=True)
    return manifest


def load_experiment(root: Path) -> dict:
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if manifest.get("format") != "v24_cached_pair_v1":
        raise ValueError("Unsupported experiment manifest")
    snapshot = freeze_source(Path(manifest["workspace"]), Path(manifest["source_root"]))
    if snapshot["source_digest"] != manifest["source_digest"]:
        raise ValueError("Experiment source snapshot identity differs")
    return manifest


def experiment_provenance(manifest: dict) -> dict:
    return build_provenance(load_training_config(Path(manifest["configs"]["cached"])),
                            Path(manifest["shared_init"]), Path(manifest["cache_root"]),
                            Path(manifest["source_root"]), execution=manifest["execution"])


def check_gates(manifest: dict) -> dict:
    return require_production_gates(Path(manifest["reports"]["full_parity"]),
                                    Path(manifest["reports"]["paired20"]), experiment_provenance(manifest))


def require_report(manifest: dict, name: str) -> dict:
    report = json.loads(Path(manifest["reports"][name]).read_text())
    if report.get("passed") is not True or report.get("provenance") != experiment_provenance(manifest):
        raise ValueError(f"Required {name} report is missing, failed or stale")
    return report


def require_resource_check(manifest: dict, budget: dict) -> dict:
    report = require_report(manifest, "resource_check")
    if int(report.get("completed_steps", 0)) < 70:
        raise ValueError("Resource check must complete seventy updates and full validation")
    allocation = report.get("allocation", {})
    if allocation.get("memory_gib") != budget["memory_gib"] or allocation.get("cpus") != 4:
        raise ValueError("Resource check did not use the proposed cached production allocation")
    if report.get("full_validation_completed") is not True:
        raise ValueError("Resource check did not finish full cached validation")
    return report


def measured_budget(benchmark: dict, backend: str, schedule: dict | None = None) -> dict:
    """Size production from separately measured full-process backend peaks."""
    record = benchmark["backends"]["cached_4cpu" if backend == "cached" else backend]
    peak = float(record["peak_host_gib"])
    update = float(record["p90_end_to_end_seconds"])
    startup = float(record["startup_seconds"])
    if not all(math.isfinite(v) and v > 0 for v in (peak, update)) or not math.isfinite(startup) or startup < 0:
        raise ValueError("Benchmark has invalid memory/timing observations")
    minimum_memory = 128 if backend == "online" else 32
    memory = max(minimum_memory, math.ceil(1.4 * peak / 16) * 16)
    schedule = schedule or {"max_steps": 4000, "checkpoint_every": 500, "validation_every": 500}
    updates = int(schedule["max_steps"])
    checkpoints = math.ceil(updates / int(schedule["checkpoint_every"]))
    validations = math.ceil(updates / int(schedule["validation_every"])) if schedule["validation_every"] else 0
    validation = float(benchmark["validation_seconds"])
    checkpoint = float(record["checkpoint_seconds"])
    if not all(math.isfinite(v) and v >= 0 for v in (validation, checkpoint)):
        raise ValueError("Invalid validation/checkpoint timing")
    seconds = 1.35 * (startup + updates * update + validations * validation + checkpoints * checkpoint)
    return {"memory_gib": memory, "time_minutes": max(62, math.ceil(seconds / 1800) * 30),
            "cpus": 8 if backend == "online" else 4, "gpu": "gpu40&nomig", "measured_peak_host_gib": peak}


def submit_command(args: list[str]) -> str:
    command = ["sbatch", "--parsable", *args]
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError(f"Unexpected Slurm submission response: {result.stdout!r}")
    return job


def job_qos(time_minutes: int) -> str:
    for limit, qos in ((1440, "gpu-short"), (4320, "gpu-medium"), (8640, "gpu-long")):
        if time_minutes <= limit:
            return qos
    raise ValueError(f"Measured job budget {time_minutes} minutes exceeds the 144-hour Slurm limit")


def cached_production_chunks(manifest: dict, benchmark: dict) -> list[tuple[int, dict]]:
    """Split a long constant-LR trajectory at native checkpoint boundaries."""
    schedule = manifest["production_schedule"]
    total, interval = int(schedule["max_steps"]), int(schedule["checkpoint_every"])
    if total <= 0 or interval <= 0:
        raise ValueError("Production steps and checkpoint interval must be positive")
    completed, chunks = 0, []
    while completed < total:
        count = total - completed
        budget = measured_budget(benchmark, "cached", {**schedule, "max_steps": count})
        while budget["time_minutes"] > 8640:
            count = ((count - 1) // interval) * interval
            if count <= 0:
                raise ValueError("One checkpoint interval exceeds the Slurm walltime limit")
            budget = measured_budget(benchmark, "cached", {**schedule, "max_steps": count})
        completed += count
        chunks.append((completed, budget))
    return chunks


def submit_stage(manifest: dict, stage: str, *, dry_run: bool = False, dependency: str | None = None) -> dict:
    root = Path(manifest["experiment_root"])
    script = str(Path(manifest["source_root"]) / "scripts/experiments/v24_cached_pair.slurm")
    requests = {}
    launches = {}
    gates = None
    if stage == "smoke":
        requests["smoke"] = manifest.get("smoke_budget", {"memory_gib": 128, "time_minutes": 120, "cpus": 8, "gpu": "gpu40&nomig"})
    elif stage == "gates":
        report = require_report(manifest, "preflight")
        budget = report["gates_budget"]
        if budget.get("cpus") != 8 or budget.get("gpu") != "gpu40&nomig":
            raise ValueError("Full gates require eight CPUs and a complete A100 40GB")
        if any(type(budget.get(key)) is not int or budget[key] <= 0 for key in ("memory_gib", "time_minutes")):
            raise ValueError("Invalid measured gates budget")
        requests["gates"] = dict(budget)
    else:
        gates = check_gates(manifest)
        benchmark = require_report(manifest, "benchmark")
        cached_budget = measured_budget(benchmark, "cached", manifest.get("production_schedule"))
        if stage == "resource-check":
            sample = benchmark["backends"]["cached_4cpu"]
            seconds = 1.35 * (sample["startup_seconds"] + 70 * sample["p90_end_to_end_seconds"] + benchmark["validation_seconds"])
            requests["resource-check"] = {**cached_budget, "time_minutes": max(120, math.ceil(seconds / 1800) * 30)}
        else:
            require_resource_check(manifest, cached_budget)
            if manifest.get("production_backends") == ["cached"]:
                chunks = cached_production_chunks(manifest, benchmark)
                for index, (stop, budget) in enumerate(chunks):
                    label = "cached" if len(chunks) == 1 else f"cached_step{stop:08d}"
                    requests[label] = budget
                    launches[label] = {"backend": "cached", "stop_step": stop, "resume_latest": index > 0}
            else:
                for backend in manifest.get("production_backends", ("online", "cached")):
                    requests[backend] = measured_budget(benchmark, backend, manifest.get("production_schedule"))
            if set(requests) == {"online", "cached"}:
                preflight_online = require_report(manifest, "preflight")["backends"]["online"]
                final_seconds = 1.35 * 6 * (preflight_online["startup_seconds"] + 4 * preflight_online["p90_end_to_end_seconds"])
                requests["finalize"] = {"memory_gib": 128, "time_minutes": max(120, math.ceil(final_seconds / 1800) * 30),
                                        "cpus": 8, "gpu": "gpu40&nomig"}
    if dependency is not None and not dependency.isdigit():
        raise ValueError("Stage dependency must be a numeric Slurm job ID")
    identity = {"source_digest": manifest["source_digest"], "requests": requests, "gates": gates, "dependency": dependency}
    if launches:
        identity["launches"] = launches
    for budget in requests.values():
        job_qos(budget["time_minutes"])
    record_path = root / f"submission_{stage}.json"
    if dry_run:
        return identity
    with (root / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(record_path.read_text()) if record_path.exists() else {**identity, "jobs": {}}
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("Submission record belongs to different source/gates/resources")
        for backend, budget in requests.items():
            if backend in record["jobs"]:
                continue
            # Pin a stable Slurm comment. A crash after sbatch but before this
            # record is written must be reconciled with accounting manually;
            # leave intent on disk and reject automatic duplicate submission.
            intent = root / f".submission_{stage}_{backend}.intent.json"
            if intent.exists():
                raise RuntimeError(f"Unresolved submission intent; inspect Slurm before retry: {intent}")
            launch = launches.get(backend, {"backend": backend})
            args = [f"--mem={budget['memory_gib']}G", f"--time={budget['time_minutes']}",
                    f"--qos={job_qos(budget['time_minutes'])}",
                    f"--cpus-per-task={budget['cpus']}", f"--constraint={budget['gpu']}",
                    f"--chdir={manifest.get('workspace', ROOT)}",
                    f"--comment=v24cached:{digest_json(identity)[:16]}:{backend}",
                    f"--output={root}/logs/{backend}_%j.out", f"--error={root}/logs/{backend}_%j.err",
                    script, launch["backend"], str(root), str(manifest.get("workspace", ROOT))]
            if "stop_step" in launch:
                args += [str(launch["stop_step"]), "1" if launch["resume_latest"] else "0"]
            if manifest.get("job_name"):
                args.insert(0, f"--job-name={manifest['job_name']}-{backend}")
            if backend == "finalize":
                args.insert(0, f"--dependency=afterok:{record['jobs']['online']}:{record['jobs']['cached']}")
            elif launch.get("resume_latest"):
                previous = list(requests)[list(requests).index(backend) - 1]
                args.insert(0, f"--dependency=afterok:{record['jobs'][previous]}")
            elif dependency is not None:
                args.insert(0, f"--dependency=afterok:{dependency}")
            atomic_json_dump({"args": args, "identity": identity}, intent)
            record["jobs"][backend] = submit_command(args)
            atomic_json_dump(record, record_path)
            intent.unlink()
    return record


def enable_automation(manifest: dict) -> dict:
    value = {"enabled": True, "source_digest": manifest["source_digest"],
             "manifest_sha256": digest_json(manifest),
             "config_sha256": {name: file_sha256(Path(path)) for name, path in manifest["configs"].items()}}
    write_once_json(Path(manifest["experiment_root"]) / "automation.json", value)
    return value


def advance_pipeline(manifest: dict, completed_stage: str) -> dict | None:
    path = Path(manifest["experiment_root"]) / "automation.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved.get("enabled") is not True:
        return None
    expected = {"enabled": True, "source_digest": manifest["source_digest"],
                "manifest_sha256": digest_json(manifest),
                "config_sha256": {name: file_sha256(Path(config)) for name, config in manifest["configs"].items()}}
    if saved != expected:
        raise ValueError("Automatic pipeline identity changed; refusing to advance")
    next_stage = {"smoke": "gates", "gates": "resource-check", "resource-check": "production"}.get(completed_stage)
    if next_stage is None:
        return None
    current_job = os.environ.get("SLURM_JOB_ID", "")
    if not current_job.isdigit():
        raise ValueError("Automatic advancement requires the current Slurm job ID")
    result = submit_stage(manifest, next_stage, dependency=current_job)
    atomic_json_dump({"completed_stage": completed_stage, "completed_job_id": current_job,
                      "submitted_stage": next_stage, "jobs": result["jobs"],
                      "source_digest": manifest["source_digest"]},
                     Path(manifest["experiment_root"]) / "pipeline_status.json")
    return result


def execute_stage(manifest: dict, stage: str, *, resume: Path | None = None,
                  stop_step: int | None = None, resume_latest: bool = False):
    root = Path(manifest["experiment_root"])
    if Path(__file__).resolve().parents[2] != Path(manifest["source_root"]).resolve():
        raise ValueError("Execute jobs from the experiment's immutable source snapshot")
    if resume is not None and stage not in ("online", "cached"):
        raise ValueError("--resume applies only to online or cached production training")
    if stage == "smoke":
        subprocess.run([sys.executable, "scripts/training/train_v24_cached_open_loop.py", "prepare",
                        "--config", manifest["configs"]["online"], "--output-dir", str(root / "shared")], check=True)
        subprocess.run([sys.executable, "-m", "src.models.mamba.v24_Ilya.training.cached_parity",
                        "--experiment-root", str(root), "--preflight"], check=True)
        return advance_pipeline(manifest, "smoke")
    if stage == "gates":
        require_report(manifest, "preflight")
        subprocess.run([sys.executable, "-m", "src.models.mamba.v24_Ilya.training.cached_parity",
                        "--experiment-root", str(root)], check=True)
        return advance_pipeline(manifest, "gates")
    check_gates(manifest)
    if stage == "resource-check":
        subprocess.run([sys.executable, "-m", "src.models.mamba.v24_Ilya.training.cached_parity",
                        "--experiment-root", str(root), "--resource-check"], check=True)
        return advance_pipeline(manifest, "resource-check")
    if stage == "finalize":
        subprocess.run([sys.executable, "-m", "src.models.mamba.v24_Ilya.training.cached_report",
                        "--experiment-root", str(root)], check=True)
        return
    config_path = Path(manifest["configs"][stage])
    config = load_training_config(config_path)
    if stop_step is not None and (stage != "cached" or not 1 <= stop_step <= config.max_steps):
        raise ValueError("Explicit stopping step requires cached training and must be within its trajectory")
    if resume_latest:
        if stage != "cached" or resume is not None:
            raise ValueError("--resume-latest requires cached training without --resume")
        resume = latest_checkpoint_for_run(config)
        if resume is None:
            raise FileNotFoundError("A continuation requires the previous native checkpoint")
    require_resource_check(manifest, measured_budget(require_report(manifest, "benchmark"), "cached", manifest.get("production_schedule")))
    if stage == "cached":
        from src.models.mamba.v24_Ilya.training.cached_runner import run_cached_training
        result = run_cached_training(config_path, Path(manifest["shared_init"]), cache_root=Path(manifest["cache_root"]),
                                   resume=resume, max_steps=stop_step,
                                   expected_manifest_sha256=manifest["execution"]["expected_manifest_sha256"],
                                   prefetch_batches=manifest["execution"]["prefetch_batches"],
                                   parity_report=Path(manifest["reports"]["full_parity"]),
                                   paired_report=Path(manifest["reports"]["paired20"]))
        atomic_json_dump({"checkpoint": str(result), "completed_step": stop_step or config.max_steps,
                          "target_steps": config.max_steps, "complete": (stop_step or config.max_steps) == config.max_steps},
                         root / "reports/training_status.json")
        return result
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import require_matched_gpu_environment
    require_matched_gpu_environment()
    from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config
    from src.models.mamba.v24_Ilya.training.cached_data import open_cached_training_data
    from src.models.mamba.v24_Ilya.training.cached_validation import make_online_validation_callback
    from src.models.mamba.v24_Ilya.training.cached_stepwise import make_cached_stepwise_train_step
    from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainInvocation
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer
    from src.models.mamba.v24_Ilya.training.runner import run_training
    import dataclasses
    resolved = load_cached_training_config(config_path, cache_root=Path(manifest["cache_root"]),
                                           expected_manifest_sha256=manifest["execution"]["expected_manifest_sha256"],
                                           prefetch_batches=0)
    data = open_cached_training_data(resolved)
    _, _, task, stats, _, baseline_model = data.context
    model = dataclasses.replace(baseline_model, latent_size=config.architecture.residual_width or config.architecture.width,
                                gnn_msg_steps=config.architecture.residual_msg_steps)
    optimizer, _ = build_optimizer(config)
    evaluator = make_cached_stepwise_train_step(config=config, model_config=model, task_config=task, stats=stats,
                                               optimizer=optimizer)
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    if resume is None:
        existing = latest_checkpoint_for_run(config)
        if existing is not None and load_v24_Ilya_training_checkpoint(existing).completed_step > 0:
            raise ValueError(f"Existing training run requires explicit --resume {existing}")
        resume = project_shared_initialization(Path(manifest["shared_init"]), config,
                                                provenance=experiment_provenance(manifest), execution=manifest["execution"])
    if Path(resume).resolve().parent.parent != config.run_dir.resolve():
        raise ValueError("Online resume checkpoint must belong to this run directory")
    completed = load_v24_Ilya_training_checkpoint(resume)
    reconcile_run_metrics(config.run_dir, completed.completed_step)
    if completed.completed_step >= config.max_steps:
        from src.models.mamba.v24_Ilya.training.cached_validation import run_cached_validation
        from src.models.mamba.v24_Ilya.training.validation import append_validation_record, load_validation_records, update_best_validation
        import jax
        import jax.numpy as jnp
        validation_path = config.run_dir / "validation_metrics.jsonl"
        records = load_validation_records(validation_path)
        if config.validation.enabled and not any(int(row["step"]) == completed.completed_step for row in records):
            record = run_cached_validation(train_step=evaluator, cached_data=data, config=config,
                params=completed.residual_params, zero_state=jax.tree_util.tree_map(jnp.zeros_like, completed.residual_state),
                step=completed.completed_step)
            append_validation_record(validation_path, record)
            update_best_validation(records=load_validation_records(validation_path),
                checkpoint_for_step=lambda n: config.run_dir / "checkpoints" / f"checkpoint_step{n:08d}.pkl",
                output_path=config.run_dir / "best_validation.json", subset_fingerprint=record["subset_fingerprint"])
        return resume
    return run_training(V24IlyaTrainInvocation(config=config, config_path=config_path, resume=resume),
                        validation_callback=make_online_validation_callback(train_step=evaluator, cached_data=data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "submit-smoke", "submit-gates", "submit-resource-check", "submit-production", "execute"))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--backend", choices=("smoke", "gates", "resource-check", "online", "cached", "finalize"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--auto-advance", action="store_true")
    args = parser.parse_args()
    if args.auto_advance and args.stage != "submit-smoke":
        parser.error("--auto-advance is enabled explicitly when submitting the preflight smoke")
    if (args.resume is not None or args.stop_step is not None or args.resume_latest) and args.stage != "execute":
        parser.error("--resume applies to execute --backend online|cached")
    if args.stage == "prepare":
        if args.cache_root is None:
            parser.error("prepare requires --cache-root")
        result = prepare_experiment(args.experiment_root, cache_root=args.cache_root)
    else:
        manifest = load_experiment(args.experiment_root)
        if args.stage == "execute":
            if args.backend is None:
                parser.error("execute requires --backend")
            result = execute_stage(manifest, args.backend, resume=args.resume,
                                   stop_step=args.stop_step, resume_latest=args.resume_latest)
        else:
            if args.auto_advance and not args.dry_run:
                enable_automation(manifest)
            result = submit_stage(manifest, args.stage.removeprefix("submit-"), dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
