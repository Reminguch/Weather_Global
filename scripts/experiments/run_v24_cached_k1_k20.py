"""Frozen, measured-release fresh-residual k1 -> live k20 curriculum.

Prepare/submit are local entry points. Every other phase executes the immutable
source snapshot. Failed parity, memory, or checkpoint gates stop dependencies.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.frozen_runtime import file_sha256, freeze_source, verify_source
from src.data_operations.loaders.zarr_metadata import frozen_zarr_metadata, verify_frozen_zarr_metadata

DRIVER = "scripts/experiments/run_v24_cached_k1_k20.py"
LAUNCHER = "scripts/experiments/v24_cached_k1_k20.slurm"
EPOCHS = 25
CHECK_EPOCHS = (5, 10, 15, 20, 25)
TRAIN_SEGMENTS = 106
CHUNKS = 4
RECOVERY_FILES = (DRIVER, LAUNCHER, "src/models/mamba/v24_Ilya/training/cached_k1_runner.py")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def steps_per_epoch(batch):
    return math.ceil(TRAIN_SEGMENTS / batch) * CHUNKS


def run_dir(config):
    config = read(config) if isinstance(config, (str, Path)) else config
    return Path(config["output"]["output_root"]) / config["output"]["run_name"]


def checkpoint(config, step):
    return run_dir(config) / "checkpoints" / f"checkpoint_step{step:08d}.pkl"


def make_configs(producer, experiment, *, widths=(128, 512)):
    if not widths or len(set(widths)) != len(widths) or any(w not in (128, 512) for w in widths):
        raise ValueError("Select unique supported residual widths: 128 and/or 512")
    configs, entries = {}, []
    for width in widths:
        batch = {128: 8, 512: 4}[width]
        name = f"w{width}"
        config = copy.deepcopy(producer)
        config["architecture"].update(residual_width=width, residual_initialization="fresh")
        config["distributed"]["per_device_batch_size"] = batch
        per_epoch = steps_per_epoch(batch)
        config["optimizer"].update(max_steps=EPOCHS * per_epoch, checkpoint_every=per_epoch,
            learning_rate=1e-4, learning_rate_schedule="cosine", end_learning_rate=1e-5,
            warmup_steps=per_epoch)
        config["validation"].update(enabled=True, every_steps=per_epoch,
                                     num_segments=None, final_num_segments=None)
        config["output"] = dict(output_root=str(experiment / "k1"), run_name=name)
        configs[f"{name}_k1"] = config
        smoke = copy.deepcopy(config)
        smoke["output"]["output_root"] = str(experiment / "k1_preflight")
        configs[f"{name}_k1_preflight"] = smoke
        live = copy.deepcopy(config)
        live["sequence"].update(ar_tail_k=20, feedback_mode="closed_loop_sg")
        live["optimizer"].update(max_steps=2000 // batch, checkpoint_every=50,
            learning_rate=1e-5, learning_rate_schedule="constant", end_learning_rate=None,
            warmup_steps=0)
        live["validation"].update(every_steps=50, num_segments=8, final_num_segments=8)
        live["output"]["output_root"] = str(experiment / "k20")
        configs[f"{name}_k20"] = live
        smoke_live = copy.deepcopy(live)
        smoke_live["optimizer"].update(max_steps=8, checkpoint_every=8)
        smoke_live["validation"]["enabled"] = False
        smoke_live["output"]["output_root"] = str(experiment / "k20_preflight")
        configs[f"{name}_k20_preflight"] = smoke_live
        entries.append(dict(name=name, width=width, batch=batch, steps_per_epoch=per_epoch,
            k1_steps=EPOCHS * per_epoch, k20_steps=2000 // batch,
            **{key: str(experiment / "configs" / f"{name}_{key}.json")
               for key in ("k1", "k1_preflight", "k20", "k20_preflight")}))
    return configs, entries


def prepare(experiment, *, widths=(128, 512)):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    if experiment.exists():
        raise FileExistsError(f"Refusing to replace {experiment}")
    producer = read(ROOT / "artifacts/cache_builds/v24_res1_gcsmall_bptt24_k0_20260919/config.json")
    producer["output"] = dict(output_root=str(experiment / "unused"), run_name="producer")
    experiment.mkdir(parents=True)
    configs, entries = make_configs(producer, experiment, widths=widths)
    configs["producer"] = producer
    for name, value in configs.items():
        path = experiment / "configs" / f"{name}.json"
        write(path, value)
        load_training_config(path)
    from scripts.experiments.run_v24_res1_width128 import REFERENCE, REF_EVAL
    reference = read(ROOT / REFERENCE / REF_EVAL)
    if reference["evaluation_status"] != "complete" or reference["evaluated_samples"] != 32:
        raise ValueError("Incomplete matched32 reference")
    write(experiment / "reference_evaluation.json", reference)
    eval_data = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr")
    eval_metadata = frozen_zarr_metadata(eval_data)
    cache = ROOT / "data/graphcast/graphcast/dataset/precomputed_baselines" / experiment.name
    if cache.exists() or cache.with_name(cache.name + "_pilot").exists():
        raise FileExistsError("Use a new experiment/cache name")
    for directory in ("logs", "checks", "eval"):
        (experiment / directory).mkdir()
    manifest = dict(workspace=str(ROOT), experiment=str(experiment), entries=entries,
        producer_config=str(experiment / "configs/producer.json"), cache_root=str(cache),
        pilot_root=str(cache.with_name(cache.name + "_pilot")), shards=8,
        epochs=EPOCHS, check_epochs=list(CHECK_EPOCHS), eval_data=str(eval_data), eval_metadata=eval_metadata,
        selection_samples=8, final_samples=32, eval_leads=40,
        k20_feedback="closed_loop_sg", stage_transfer="parameters_only_zero_state_new_optimizer_rng_cursor",
        reference_evaluation_sha256=file_sha256(experiment / "reference_evaluation.json"),
        resource_evidence={"cache_pilot_job": "14143122", "elapsed_seconds": 2803,
            "host_peak_gib": 9581660 / 1024**2, "historical_pilot_request": "48G / 180 minutes",
            "new_pilot_request": "16G / 90 minutes",
            "k1_preflight": "96G / 120 minutes; first B8/B4 measurements",
            "k20_preflight": "128G / 120 minutes; live weather tapes",
            "eval_history": "res1 matched32: approximately 40 minutes and 13GiB; measured again in gates",
            "production_host_memory": "Include the prepared-array mapped working set; short preflights do not touch all training years. Completed live jobs13900119/14117544 peaked at243GiB."})
    write(experiment / "manifest.json", manifest)
    write(experiment / "config_hashes.json", {str(p.relative_to(experiment)): file_sha256(p)
        for p in sorted((experiment / "configs").glob("*.json"))})
    (experiment / "workspace.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    (experiment / "workspace_status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    frozen = freeze_source(ROOT, experiment / "source")
    write(experiment / "identity.json", {"source_digest": frozen["source_digest"],
        "manifest_sha256": file_sha256(experiment / "manifest.json"),
        "config_hashes_sha256": file_sha256(experiment / "config_hashes.json")})
    print(json.dumps(manifest, indent=2))


def verify(experiment, *, frozen=True):
    identity = read(experiment / "identity.json")
    source = verify_source(experiment / "source")
    if source["source_digest"] != identity["source_digest"]:
        raise ValueError("Source identity changed")
    for name in ("manifest", "config_hashes"):
        if file_sha256(experiment / f"{name}.json") != identity[f"{name}_sha256"]:
            raise ValueError(f"Changed {name}")
    for relative, expected in read(experiment / "config_hashes.json").items():
        if file_sha256(experiment / relative) != expected:
            raise ValueError(f"Changed configuration: {relative}")
    if frozen and ROOT != experiment / "source":
        raise RuntimeError("Run stages from the prepared source snapshot")
    manifest = read(experiment / "manifest.json")
    if manifest.get('numerical_policy', 'strict') not in ('strict', 'advisory'):
        raise ValueError('Unknown numerical policy')
    if file_sha256(experiment / "reference_evaluation.json") != manifest["reference_evaluation_sha256"]:
        raise ValueError("Reference evaluation changed")
    verify_frozen_zarr_metadata(manifest["eval_data"], manifest["eval_metadata"])
    if "cache_reuse" in manifest:
        reused = manifest["cache_reuse"]
        for name, expected in reused["files"].items():
            if file_sha256(Path(manifest["cache_root"]) / name) != expected:
                raise ValueError(f"Reused cache identity changed: {name}")
        if file_sha256(experiment / "checks/cache_release.json") != reused["cache_release_sha256"]:
            raise ValueError("Reused cache release evidence changed")
    return manifest


def prepare_recovery(experiment, previous, *, deterministic=False, numerical_policy='advisory'):
    """Clone the producer-compatible snapshot and patch only the K1 consumer."""
    if experiment.exists():
        raise FileExistsError(f"Refusing to replace {experiment}")
    if numerical_policy not in ('strict', 'advisory'):
        raise ValueError('Unknown numerical policy')
    old = verify(previous, frozen=False)
    cache = Path(old["cache_root"])
    saved, ready = read(cache / "manifest.json"), read(cache / "READY.json")
    release = read(previous / "checks/cache_release.json")
    expected_ready = dict(manifest_sha256=saved["manifest_sha256"], chunks=len(saved["chunks"]),
        predictions=sum(len(chunk["raw_anchor_indices"]) for chunk in saved["chunks"]),
        shards=saved["num_shards"], partial=False)
    if (saved["partial"] or any(ready.get(k) != v for k, v in expected_ready.items())
            or release.get("passed") is not True or release["manifest_sha256"] != saved["manifest_sha256"]):
        raise ValueError("Recovery requires a complete verified cache")
    experiment.mkdir(parents=True)
    for name in ("logs", "checks", "eval"):
        (experiment / name).mkdir()
    producer = read(old["producer_config"])
    configs, entries = make_configs(producer, experiment, widths=tuple(e["width"] for e in old["entries"]))
    configs["producer"] = producer
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    for name, config in configs.items():
        path = experiment / "configs" / f"{name}.json"
        write(path, config)
        load_training_config(path)
    shutil.copy2(previous / "reference_evaluation.json", experiment / "reference_evaluation.json")
    shutil.copy2(previous / "checks/cache_release.json", experiment / "checks/cache_release.json")
    # Freeze a derivation of the historical producer, never unrelated workspace
    # model edits. The temporary staging directory is not the runtime workspace.
    from scripts.experiments.frozen_runtime import SOURCE_DIRS
    with tempfile.TemporaryDirectory(prefix="k1-recovery-") as directory:
        staging = Path(directory)
        for relative in SOURCE_DIRS:
            shutil.copytree(previous / "source" / relative, staging / relative)
        for relative in RECOVERY_FILES:
            shutil.copy2(ROOT / relative, staging / relative)
        snapshot = freeze_source(staging, experiment / "source")
        snapshot["workspace"] = old["workspace"]
        write(experiment / "source/snapshot.json", snapshot)
    manifest = copy.deepcopy(old)
    manifest.update(experiment=str(experiment), entries=entries,
        producer_config=str(experiment / "configs/producer.json"), k1_deterministic=bool(deterministic),
        numerical_policy=numerical_policy,
        runtime={'xla_flags': '--xla_gpu_enable_triton_gemm=false' +
                 (' --xla_gpu_deterministic_ops=true' if deterministic else ''),
                 'xla_python_client_preallocate': 'false', 'xla_python_client_allocator': 'platform'},
        cache_reuse=dict(previous_experiment=str(previous),
            previous_identity_sha256=file_sha256(previous / "identity.json"),
            previous_source_digest=read(previous / "identity.json")["source_digest"],
            patched_files=list(RECOVERY_FILES), files={name: file_sha256(cache / name)
                for name in ("manifest.json", "READY.json", *(
                    f"shard_{i:03d}/COMPLETE.json" for i in range(saved["num_shards"])))},
            cache_release_sha256=file_sha256(experiment / "checks/cache_release.json")))
    write(experiment / "manifest.json", manifest)
    write(experiment / "config_hashes.json", {str(p.relative_to(experiment)): file_sha256(p)
        for p in sorted((experiment / "configs").glob("*.json"))})
    write(experiment / "identity.json", dict(source_digest=snapshot["source_digest"],
        manifest_sha256=file_sha256(experiment / "manifest.json"),
        config_hashes_sha256=file_sha256(experiment / "config_hashes.json")))
    verify(experiment, frozen=False)
    print(json.dumps(manifest, indent=2))


def verify_reused_cache(experiment):
    """Rebuild producer identity from the recovery snapshot before submission."""
    from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config
    from src.models.mamba.v24_Ilya.training.cached_data import open_cached_training_data
    m = verify(experiment)
    if "cache_reuse" not in m:
        raise ValueError("This experiment does not reuse a cache")
    config = load_cached_training_config(Path(m["producer_config"]), cache_root=Path(m["cache_root"]),
        expected_manifest_sha256=read(experiment / "checks/cache_release.json")["manifest_sha256"])
    cached = open_cached_training_data(config)
    from src.models.mamba.v24_Ilya.baseline_cache import verify_shard
    markers = [verify_shard(Path(m["cache_root"]), cached.manifest, i, checksums=False)
               for i in range(cached.manifest["num_shards"])]
    write(experiment / "checks/cache_reuse.json", dict(passed=True, cache_identity=cached.identity,
        verified_shards=len(markers), checksum_policy="original verified checksums; all shard schemas/coverage rechecked",
        source_digest=read(experiment / "identity.json")["source_digest"],
        manifest_sha256=file_sha256(experiment / "manifest.json")))


def command(experiment, script, arguments):
    from scripts.experiments.frozen_runtime import frozen_command
    args, options = frozen_command(experiment / "source", read(experiment / "manifest.json")["workspace"], script, arguments)
    print("EXEC " + shlex.join(args), flush=True)
    started = time.monotonic()
    subprocess.run(args, check=True, **options)
    return time.monotonic() - started


def submit_job(experiment, key, phase, *, task=0, block=0, memory=8, minutes=20,
               gpu=False, dependency=None, array=None):
    """Serialize submissions; persist each ID before another external action."""
    manifest = read(experiment / "manifest.json")
    with (experiment / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = experiment / "submissions.json"
        record = read(path) if path.exists() else {}
        if key in record:
            if not record[key].get('job_id'):
                raise RuntimeError(f'Unresolved submission intent; inspect Slurm before retry: {key}')
            return record[key]["job_id"]
        args = ["sbatch", "--parsable", "--account=dabanin", f"--job-name=k1k20-{key}",
            f"--chdir={manifest['workspace']}", f"--mem={memory}G", f"--time={minutes}",
            f"--cpus-per-task={8 if gpu else 2}",
            f"--output={experiment}/logs/{key}_%A_%a.out",
            f"--error={experiment}/logs/{key}_%A_%a.err"]
        if gpu:
            qos = "gpu-short" if minutes <= 1440 else "gpu-medium"
            args += [f"--qos={qos}", "--gres=gpu:1", "--constraint=gpu40&nomig"]
        else:
            args += ["--qos=short"]
        if dependency:
            args += ["--dependency=afterok:" + ":".join(map(str, dependency))]
        if array is not None:
            if '%' in str(array):
                raise ValueError('Recovery arrays must be unthrottled')
            args += [f"--array={array}"]
        args += [str(experiment / "source" / LAUNCHER), str(experiment), phase,
                 str(manifest["workspace"]), str(task), str(block)]
        print(shlex.join(args), flush=True)
        record[key] = dict(command=args, memory_gib=memory, time_minutes=minutes)
        write(path, record)
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Unexpected sbatch output: {result.stdout}")
        record[key] = dict(job_id=job, command=args, memory_gib=memory, time_minutes=minutes)
        write(path, record)
        return job


def submit(experiment):
    m = verify(experiment, frozen=False)
    if "cache_reuse" in m:
        evidence = read(experiment / "checks/cache_reuse.json")
        if (evidence.get("passed") is not True
                or evidence["source_digest"] != read(experiment / "identity.json")["source_digest"]
                or evidence["manifest_sha256"] != file_sha256(experiment / "manifest.json")):
            raise ValueError("Recovery cache verification is missing or stale")
        preflight = submit_job(experiment, "k1-preflight", "k1-preflight", gpu=True,
            array=f"0-{len(m['entries']) - 1}", memory=96, minutes=120)
        return submit_job(experiment, "k1-release", "k1-release", dependency=[preflight], memory=8, minutes=20)
    pilot = submit_job(experiment, "cache-pilot", "pilot", gpu=True, memory=16, minutes=90)
    submit_job(experiment, "cache-release", "cache-release", dependency=[pilot], memory=16, minutes=30)


def cache_command(experiment, phase, task):
    m = read(experiment / "manifest.json")
    pilot = phase == "pilot"
    args = ["--config", m["producer_config"], "--output-root", m["pilot_root"] if pilot else m["cache_root"],
            "--num-shards", "1" if pilot else str(m["shards"])]
    if pilot:
        args += ["--max-chunks", "4", "--pilot-parity"]
    elif phase == "cache-verify":
        args += ["--verify"]
    else:
        args += ["--shard-index", str(task), "--resume"]
    command(experiment, "scripts/preprocessing/precompute_v24_baseline.py", args)


def cache_release(experiment):
    from src.models.mamba.v24_Ilya.baseline_cache import load_context, build_manifest, verify_cache, ensure_manifest
    from scripts.preprocessing.submit_v24_baseline_production import production_budget
    m = read(experiment / "manifest.json")
    context = load_context(Path(m["producer_config"]))
    full = build_manifest(context, m["shards"])
    pilot = build_manifest(context, 1, 4)
    report = read(Path(m["pilot_root"]) / "pilot_report.json")
    if read(Path(m["pilot_root"]) / "manifest.json") != pilot:
        raise ValueError("Pilot differs from frozen producer")
    if (report.get("passed") is not True or report["chunks"] != 4
            or report["manifest_sha256"] != pilot["manifest_sha256"]
            or report["compatibility_sha256"] != full["compatibility_sha256"]
            or not report["max_normalized_errors"]
            or max(report["max_normalized_errors"].values()) > 1e-5):
        raise ValueError("Cache parity gate failed")
    verify_cache(Path(m["pilot_root"]), allow_partial=True)
    if full["partial"] or len(full["chunks"]) != (106 + 15) * 4:
        raise ValueError("Unexpected train/validation coverage")
    largest = max(sum(c["shard"] == i for c in full["chunks"]) for i in range(m["shards"]))
    memory, minutes = production_budget(report, largest)
    # A four-chunk pilot touches only a small part of the memory-mapped source.
    # A production shard eventually maps its fraction of all prepared years.
    source_gib = sum(v["size"] for v in full["compatibility"]["prepared_files"].values()) / 1024**3
    memory = max(memory, math.ceil((source_gib / m["shards"] + report["peak_host_gib"]) * 1.3 / 8) * 8)
    ensure_manifest(Path(m["cache_root"]), full)
    if shutil.disk_usage(m["cache_root"]).free < 1.1 * full["prediction_bytes"]:
        raise RuntimeError("Insufficient cache disk space")
    write(experiment / "checks/cache_release.json", dict(passed=True, manifest_sha256=full["manifest_sha256"],
        pilot_report=report, memory_gib=memory, minutes=minutes, largest_shard_chunks=largest,
        prepared_source_gib=source_gib))
    shards = submit_job(experiment, "cache-shards", "cache-shard", gpu=True, array="0-7", memory=memory, minutes=minutes)
    ready = submit_job(experiment, "cache-verify", "cache-verify", dependency=[shards], memory=16, minutes=60)
    preflight = submit_job(experiment, "k1-preflight", "k1-preflight", gpu=True, array=f"0-{len(m['entries']) - 1}",
                           dependency=[ready], memory=96, minutes=120)
    submit_job(experiment, "k1-release", "k1-release", dependency=[preflight], memory=8, minutes=20)


def k1_worker(experiment, entry, *, preflight, stop, resume=None, validate=False):
    from src.models.mamba.v24_Ilya.training.cached_k1_runner import run_cached_k1_training
    m = read(experiment / "manifest.json")
    deterministic = "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "").split()
    if deterministic != m.get("k1_deterministic", False):
        raise ValueError("K1 runtime differs from the pinned deterministic setting")
    return run_cached_k1_training(Path(entry["k1_preflight" if preflight else "k1"]),
        Path(m["producer_config"]), cache_root=Path(m["cache_root"]),
        expected_manifest_sha256=read(experiment / "checks/cache_release.json")["manifest_sha256"],
        epochs=EPOCHS, max_steps=stop, resume=resume, validate_at_stop=validate,
        validation_segments=2 if preflight else None, verify_roundtrip=True,
        numerical_policy=m.get('numerical_policy', 'strict'))


def eval_model(experiment, entry, path, label, *, samples=8, leads=40):
    from scripts.experiments.run_v24_res1_width128 import eval_command
    output = experiment / "eval" / entry["name"] / f"{label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    # The maintained command builder accepts the common architecture/data config.
    args = eval_command(experiment, {**entry, "config": entry["k1"]}, path, output)
    for key, value in (("--n-samples", samples), ("--target-steps", leads)):
        args[args.index(key) + 1] = str(value)
    provenance_path = output.with_suffix(".provenance.json")
    baseline_path = Path(read(entry["k1"])["data"]["baseline_checkpoint"])
    provenance = dict(checkpoint=str(Path(path).resolve()), checkpoint_sha256=file_sha256(path),
        baseline_checkpoint=str(baseline_path.resolve()), baseline_sha256=file_sha256(baseline_path),
        source_digest=read(experiment / "identity.json")["source_digest"],
        samples=samples, leads=leads, mode="cold_full", initial_state="zero", seed=0)
    started = time.monotonic()
    if output.exists() and (not provenance_path.exists() or read(provenance_path) != provenance):
        raise ValueError("Existing evaluation has missing or different checkpoint provenance")
    if not output.exists():
        write(provenance_path, provenance)
        command(experiment, "scripts/analyze_models/eval_v24_Ilya.py", args[3:])
    result = read(output)
    if (result["evaluation_status"] != "complete" or result["evaluated_samples"] != samples
            or result["target_steps"] != leads):
        raise ValueError("Incomplete native live evaluation")
    if (Path(result["ckpt"]).resolve() != Path(path).resolve()
            or Path(result["ckpt_in"]).resolve() != baseline_path.resolve()
            or result["eval_mode"] != "cold_full"
            or result["residual_state_init"] != "zero"
            or result["eval_feedback"] != "full"):
        raise ValueError("Evaluation checkpoint or rollout protocol differs")
    loss = float(result["original_graphcast_loss"]["full_rollout"])
    if not math.isfinite(loss) or loss <= 0:
        raise ValueError("Nonfinite live rollout loss")
    if samples == 32 and result["chosen_idx"] != read(experiment / "reference_evaluation.json")["chosen_idx"]:
        raise ValueError("Final evaluation origins differ from matched32 reference")
    return dict(path=str(output), evaluation_sha256=file_sha256(output),
                checkpoint=str(path), checkpoint_sha256=provenance["checkpoint_sha256"],
                loss=loss, chosen_idx=result["chosen_idx"], elapsed_seconds=time.monotonic() - started)


def gpu_peak(experiment, phase, task):
    samples = []
    for path in (experiment / "logs").glob(f"{phase}_*_{task}_gpu.csv"):
        for row in csv.reader(path.open()):
            if len(row) >= 5:
                try:
                    used, total = float(row[3]), float(row[4])
                except ValueError:
                    continue
                if total > 43000 or "A100" not in row[2]:
                    raise ValueError("Preflight did not use the requested A100 40GB")
                samples.append((used, total))
    if not samples:
        raise ValueError("No sampled GPU memory evidence")
    used, total = max(samples)
    if used > total * 0.93:
        raise ValueError(f"Insufficient GPU headroom: {used}/{total} MiB")
    return dict(peak_mib=used, total_mib=total, interval_seconds=5,
                note="Sampled peak; brief allocations may be missed")


def training_measurement(config):
    values = rows(run_dir(config) / "train_metrics.jsonl")
    if len(values) < 8 or any(not math.isfinite(r[key]) for r in values for key in ("loss", "gradient_norm")):
        raise ValueError("Insufficient finite preflight updates")
    if not any(r["gradient_norm"] > 0 for r in values):
        raise ValueError("No parameter learning in preflight")
    steady = sorted(float(r["end_to_end_seconds"]) for r in values[4:])
    return dict(updates=len(values), steady_seconds=steady[math.ceil(.9 * len(steady)) - 1],
        max_first_update_seconds=max(float(r["end_to_end_seconds"]) for r in values[:4]))


def k1_preflight(experiment, entry, task):
    started = time.monotonic()
    # Crossing a saved checkpoint in the middle of a segment exercises carried
    # state, RNG, optimizer, and sampler restoration on the actual GPU shape.
    for stop in (2, 20):
        command(experiment, DRIVER, ["k1-worker", "--experiment", experiment, "--task", task,
            "--preflight", "--stop", stop, *(["--resume", checkpoint(entry["k1_preflight"], 2)] if stop == 20 else []),
            *(["--validate"] if stop == 20 else [])])
    roundtrip = read(run_dir(entry["k1_preflight"]) / "checkpoint_roundtrip.json")
    if roundtrip.get("passed") is not True:
        raise ValueError("Native checkpoint roundtrip gate failed")
    if read(experiment / 'manifest.json').get('numerical_policy') == 'advisory' and roundtrip.get('integrity_passed') is not True:
        raise ValueError('Native checkpoint integrity check is missing')
    native = eval_model(experiment, entry, checkpoint(entry["k1_preflight"], 20), "preflight", samples=1, leads=2)
    measurement = training_measurement(entry["k1_preflight"])
    from src.models.mamba.v24_Ilya.training.cached_k1_runner import parameter_update_report
    learned = parameter_update_report(checkpoint(entry["k1_preflight"], 0), checkpoint(entry["k1_preflight"], 20))
    if learned.get("passed") is not True:
        raise ValueError("Cached preflight did not update spatial, temporal, and output parameters")
    execution = read(run_dir(entry["k1_preflight"]) / "cached_k1_execution.json")
    validation = rows(run_dir(entry["k1_preflight"]) / "validation_metrics.jsonl")[-1]
    write(experiment / "checks" / f"{entry['name']}_k1_preflight.json", dict(passed=True, integrity_passed=True,
        status=roundtrip.get('status', 'complete'), numerical_policy=read(experiment / 'manifest.json').get('numerical_policy', 'strict'),
        **measurement, native_evaluation=native, roundtrip=roundtrip, parameter_updates=learned,
        peak_host_gib=max(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
                          execution.get("peak_host_gib", 0)),
        elapsed_seconds=time.monotonic() - started, validation=validation,
        gpu=gpu_peak(experiment, "k1-preflight", task)))


def measured_budget(report, updates, *, extra_seconds=0):
    if report.get("passed") is not True:
        raise ValueError("Cannot release a failed preflight")
    peak = float(report["peak_host_gib"])
    seconds = float(report["steady_seconds"])
    if not math.isfinite(peak) or peak <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid measured resource evidence")
    memory = max(24, math.ceil(peak * 1.5 / 8) * 8)
    # Measured total preflight bounds startup, checkpoint and compile overhead.
    total = 1.4 * (float(report["elapsed_seconds"]) + updates * seconds + extra_seconds)
    minutes = max(62, math.ceil(total / 1800) * 30)
    if minutes > 4320:
        raise ValueError("Measured stage exceeds 72h; split before release")
    return memory, minutes


def production_memory(experiment, measured):
    # Both trainers eventually read all prepared years. Keep full-data headroom
    # even when the short preflight's RSS is much smaller (historical peak243GiB).
    source_gib = read(experiment / "checks/cache_release.json")["prepared_source_gib"]
    return max(measured, math.ceil(source_gib * 1.3 / 8) * 8)


def k1_release(experiment):
    m = read(experiment / "manifest.json")
    fine_checks = []
    for task, entry in enumerate(m["entries"]):
        report = read(experiment / "checks" / f"{entry['name']}_k1_preflight.json")
        if m.get('numerical_policy') == 'advisory' and report.get('integrity_passed') is not True:
            raise ValueError('Advisory preflight lacks integrity evidence')
        # Validation is measured on two full segments. Scale to all15 at each
        # epoch, with a conservative 90min allowance for the live8 evaluation.
        val_seconds = float(report["validation"]["duration_seconds"])
        memory, minutes = measured_budget(report, 5 * entry["steps_per_epoch"],
                                         extra_seconds=5 * 7.5 * val_seconds + 5400)
        memory = production_memory(experiment, memory)
        previous = None
        for block, epoch in enumerate(CHECK_EPOCHS):
            previous = submit_job(experiment, f"{entry['name']}-epoch{epoch}", "k1-block", task=task,
                block=block, gpu=True, memory=memory, minutes=minutes,
                dependency=[previous] if previous else None)
        fine_checks.append(submit_job(experiment, f"{entry['name']}-k20-preflight", "k20-preflight",
            task=task, gpu=True, memory=128, minutes=120, dependency=[previous]))
    submit_job(experiment, "k20-release", "k20-release", dependency=fine_checks, memory=8, minutes=20)


def k1_block(experiment, entry, task, block):
    epoch = CHECK_EPOCHS[block]
    stop = epoch * entry["steps_per_epoch"]
    args = ["k1-worker", "--experiment", experiment, "--task", task, "--stop", stop]
    if block:
        args += ["--resume", checkpoint(entry["k1"], CHECK_EPOCHS[block - 1] * entry["steps_per_epoch"])]
    command(experiment, DRIVER, args)
    result = eval_model(experiment, entry, checkpoint(entry["k1"], stop), f"epoch{epoch}_dev8")
    write(experiment / "checks" / f"{entry['name']}_epoch{epoch}.json", result)


def select_parent(experiment, entry):
    results = [read(experiment / "checks" / f"{entry['name']}_epoch{epoch}.json") for epoch in CHECK_EPOCHS]
    if any(r["chosen_idx"] != results[0]["chosen_idx"] for r in results):
        raise ValueError("Selection evaluation origins differ across epochs")
    for result in results:
        if file_sha256(result["path"]) != result["evaluation_sha256"]:
            raise ValueError("Selection evaluation changed")
    best = min(results, key=lambda r: r["loss"])
    if file_sha256(best["checkpoint"]) != best["checkpoint_sha256"]:
        raise ValueError("Selected parent checkpoint changed")
    write(experiment / "checks" / f"{entry['name']}_parent.json", dict(**best,
        criterion="minimum exact physical GraphCast loss on fixed dev8 x40 cold leads",
        candidates=results, state_transfer="zero", optimizer_transfer="reset"))
    return Path(best["checkpoint"])


def live_train(experiment, entry, *, preflight):
    parent = select_parent(experiment, entry)
    config = entry["k20_preflight" if preflight else "k20"]
    command(experiment, "scripts/training/train_v24_Ilya.py",
            ["--config", config, "--init-from", parent, "--zero-state-on-init"])
    return checkpoint(config, 8 if preflight else entry["k20_steps"])


def k20_preflight(experiment, entry, task):
    started = time.monotonic()
    path = live_train(experiment, entry, preflight=True)
    native = eval_model(experiment, entry, path, "k20_preflight", samples=1, leads=2)
    from src.models.mamba.v24_Ilya.training.cached_k1_runner import parameter_update_report
    learned = parameter_update_report(select_parent(experiment, entry), path)
    if learned.get("passed") is not True:
        raise ValueError("Live k20 preflight did not update all parameter groups")
    write(experiment / "checks" / f"{entry['name']}_k20_preflight.json", dict(passed=True, integrity_passed=True,
        **training_measurement(entry["k20_preflight"]), native_evaluation=native, parameter_updates=learned,
        elapsed_seconds=time.monotonic() - started,
        peak_host_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
        gpu=gpu_peak(experiment, "k20-preflight", task)))


def k20_release(experiment):
    m = read(experiment / "manifest.json")
    evaluations = []
    for task, entry in enumerate(m["entries"]):
        report = read(experiment / "checks" / f"{entry['name']}_k20_preflight.json")
        if m.get('numerical_policy') == 'advisory' and report.get('integrity_passed') is not True:
            raise ValueError('Advisory preflight lacks integrity evidence')
        # Eight validation segments every50updates, each96 live leads. The
        # forward-only work is bounded conservatively by measured BPTT updates.
        validations = math.ceil(entry["k20_steps"] / 50)
        extra = validations * 8 * CHUNKS * report["steady_seconds"] / entry["batch"]
        memory, minutes = measured_budget(report, entry["k20_steps"], extra_seconds=extra)
        memory = production_memory(experiment, memory)
        trained = submit_job(experiment, f"{entry['name']}-k20", "k20-train", task=task,
            gpu=True, memory=memory, minutes=minutes)
        evaluations.append(submit_job(experiment, f"{entry['name']}-final-eval", "final-eval",
            task=task, gpu=True, memory=24, minutes=180, dependency=[trained]))
    submit_job(experiment, "report", "report", dependency=evaluations, memory=4, minutes=10)


def final_eval(experiment, entry):
    parent = select_parent(experiment, entry)
    before = eval_model(experiment, entry, parent, "selected_k1_matched32", samples=32)
    after = eval_model(experiment, entry, checkpoint(entry["k20"], entry["k20_steps"]), "k20_final_matched32", samples=32)
    write(experiment / "checks" / f"{entry['name']}_final.json", dict(k1=before, k20=after))


def report(experiment):
    m = read(experiment / "manifest.json")
    lines = ["# Fresh cached k1 to live k20 results", "",
        "Selection uses fixed dev8 cold rollouts. Final comparisons use the existing matched32 2022 development cohort.", ""]
    results = []
    numerical_records = []
    for phase in ('k1_preflight', 'k1'):
        for path in sorted((experiment / phase).glob('*/checkpoints/*.roundtrip.json')):
            record = read(path)
            if record.get('integrity_passed') is not True:
                raise ValueError('Checkpoint diagnostic records failed integrity')
            numerical_records.append({'path': str(path), 'sha256': file_sha256(path),
                                      'status': record['status'], 'numerical_status': record['numerical_status']})
    if any(row['numerical_status'] == 'warning' for row in numerical_records):
        lines += ['Numerical warning: repeated/reloaded forward calculations differ; exact checkpoint serialization passed.', '']
    for entry in m["entries"]:
        pair = read(experiment / "checks" / f"{entry['name']}_final.json")
        values = {}
        for stage, selected in pair.items():
            evaluation = read(selected["path"])
            values[stage] = evaluation["original_graphcast_loss"]
        results.append(dict(**entry, results=values))
        lines.append(f"- Width{entry['width']}, batch{entry['batch']}: {entry['k1_steps']} cached updates, {entry['k20_steps']} live updates. "
            f"Exact rollout improvement: k1 {values['k1']['improvement_pct_rollout']:.3f}%; k20 {values['k20']['improvement_pct_rollout']:.3f}%.")
    write(experiment / "results.json", results)
    write(experiment / 'complete.json', {'integrity_passed': True,
        'numerical_policy': m.get('numerical_policy', 'strict'), 'numerical_diagnostics': numerical_records,
        'status': 'complete_with_warnings' if any(row['numerical_status'] == 'warning' for row in numerical_records) else 'complete'})
    (experiment / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "recover", "verify-reused-cache", "submit", "verify", "pilot", "cache-release",
        "cache-shard", "cache-verify", "k1-worker", "k1-preflight", "k1-release", "k1-block",
        "k20-preflight", "k20-release", "k20-train", "final-eval", "report"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--from-experiment", type=Path)
    parser.add_argument("--k1-deterministic", action="store_true")
    parser.add_argument("--widths", type=int, nargs="+", choices=(128, 512), default=(128, 512),
                        help="Residual widths to prepare; submit uses the saved manifest")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.phase == "prepare":
        return prepare(experiment, widths=args.widths)
    if args.phase == "recover":
        if args.from_experiment is None:
            parser.error("recover requires --from-experiment")
        return prepare_recovery(experiment, args.from_experiment.resolve(), deterministic=args.k1_deterministic)
    if args.phase == "submit":
        return submit(experiment)
    m = verify(experiment)
    if args.phase == "verify-reused-cache":
        return verify_reused_cache(experiment)
    if args.phase in ("pilot", "cache-shard", "cache-verify"):
        return cache_command(experiment, args.phase, args.task)
    global_stages = {"cache-release": cache_release, "k1-release": k1_release,
                     "k20-release": k20_release, "report": report}
    if args.phase in global_stages:
        return global_stages[args.phase](experiment)
    if args.phase == "verify":
        return
    entry = m["entries"][args.task]
    if args.phase == "k1-worker":
        k1_worker(experiment, entry, preflight=args.preflight, stop=args.stop, resume=args.resume, validate=args.validate)
    elif args.phase == "k1-preflight":
        k1_preflight(experiment, entry, args.task)
    elif args.phase == "k1-block":
        k1_block(experiment, entry, args.task, args.block)
    elif args.phase == "k20-preflight":
        k20_preflight(experiment, entry, args.task)
    elif args.phase == "k20-train":
        live_train(experiment, entry, preflight=False)
    elif args.phase == "final-eval":
        final_eval(experiment, entry)


if __name__ == "__main__":
    main()
