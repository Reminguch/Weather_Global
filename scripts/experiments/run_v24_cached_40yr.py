#!/usr/bin/env python3
"""Frozen, restartable download → prepared arrays → cache → cached training campaign.

``prepare`` and ``submit --dry-run`` never submit jobs. Only ``submit``/``resume``
authorize automatic advancement. GPU work executes through the frozen snapshot.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import freeze_source, verify_source

FORMAT = "v24_cached_40yr_v1"
YEARS = list(range(1982, 2023))
GPU_TYPES = {"full": "gpu40&nomig", "mig": "intel&gpu40"}
DEFAULT_ROOT = ROOT / "artifacts/checkpoints/v24_Ilya/res1_cached_40yr_20261012"
DEFAULT_DOWNLOAD = ROOT / "artifacts/data_staging/res1_1982_2022_20261012/download_status.json"
TEMPLATE = "configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json"
RESERVE = 256 * 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def locked(root, name=".submission.lock"):
    with (Path(root) / name).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def training_config(template, paths, root, width, *, batch=4):
    config = copy.deepcopy(template)
    config["data"].update(prepared_root=paths["prepared_root"], anchor_manifest_root=paths["anchor_root"],
                          baseline_checkpoint=paths["baseline_checkpoint"], stats_dir=paths["stats_dir"],
                          allow_incomplete_prepared_store=False, time_start=None, time_end=None)
    config["architecture"].update(resolution=1., mesh_size=5, width=512, residual_width=width,
        residual_initialization="fresh", baseline_msg_steps=16, residual_msg_steps=2,
        temporal_d_inner=16, temporal_bc_groups=1, temporal_d_state=16, temporal_d_conv=4,
        temporal_dt_rank="32", temporal_layers=2, temporal_init_scheme="mamba1",
        temporal_stateful=True, temporal_zero_init_out=True, temporal_dropout=0.)
    config["sequence"].update(segment_steps=96, bptt_steps=24, ar_tail_k=20,
                              feedback_mode="baseline", temporal_state_policy="carry")
    config["objective"] = {"loss_mode": "all_steps"}
    config["distributed"].update(mode="single", num_devices=1, per_device_batch_size=batch,
                                  drop_incomplete_replica_group=True)
    config["optimizer"].update(max_steps=5000, checkpoint_every=250, warmup_steps=50,
        learning_rate=1e-4, learning_rate_schedule="constant", end_learning_rate=None,
        adam_beta1=.9, adam_beta2=.98, weight_decay=1e-4, grad_clip=1., seed=22, precision="bf16")
    config["validation"].update(enabled=True, every_steps=500, num_segments=None, final_num_segments=None)
    config["output"] = {"output_root": str(Path(root) / "training" / f"w{width}" / "runs"), "run_name": "cached"}
    return config


def prepare(root, *, workspace=ROOT, download_status=DEFAULT_DOWNLOAD, stop_after="report",
            widths=(128, 512), init_from=None, download_wait_hours=72):
    if not 1 <= download_wait_hours <= 336:
        raise ValueError('Download wait must be between one hour and fourteen days')
    widths = list(widths)
    if not widths or len(set(widths)) != len(widths) or any(w not in (128, 512) for w in widths):
        raise ValueError('Choose distinct supported widths')
    init_from = str(Path(init_from).resolve()) if init_from is not None else None
    if stop_after not in ("cache_verify", "report"):
        raise ValueError("stop_after must be cache_verify or report")
    root, workspace, download_status = (Path(p).resolve() for p in (root, workspace, download_status))
    if (root / "campaign.json").exists():
        campaign = load_campaign(root)
        if campaign["workspace"] != str(workspace) or campaign["paths"]["download_status"] != str(download_status):
            raise ValueError("Existing campaign belongs to different inputs")
        if campaign.get("stop_after", "report") != stop_after:
            raise ValueError("Existing campaign has a different execution endpoint")
        if campaign.get('widths', [128, 512]) != widths or campaign.get('init_from') != init_from:
            raise ValueError('Existing campaign has a different model or initialization')
        if campaign.get('download_wait_hours', 72) != download_wait_hours:
            raise ValueError('Existing campaign has a different download wait budget')
        return campaign
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty campaign root; incomplete preparation is not overwritten")
    status = read(download_status)
    if (status.get("start_year"), status.get("end_year")) != (1982, 2022):
        raise ValueError("Download must cover exactly 1982–2022")
    template = read(workspace / TEMPLATE)
    parent = None
    if init_from:
        from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
        import numpy as np
        parent = load_v24_Ilya_training_checkpoint(Path(init_from))
        template = copy.deepcopy(parent.resolved_training_config)
        arch = template['architecture']
        if widths != [arch['residual_width'] or arch['width']]:
            raise ValueError('Warm start requires exactly its native width')
        if not all(np.isfinite(v).all() for module in parent.residual_params.values() for v in module.values()):
            raise ValueError('Nonfinite parent checkpoint')
    dataset = workspace / "data/graphcast/graphcast/dataset"
    resolve = lambda p: str((workspace / p).resolve())
    paths = dict(source_zarr=str(Path(status["output"]).resolve()), download_status=str(download_status),
        download_report=str(Path(status["output"]).with_name(Path(status["output"]).name + ".stage_report.json")),
        prepared_root=str(dataset / "prepared_stream_1982_2022/res1"),
        anchor_root=str(dataset / "anchor_manifests/v24_Ilya_res1_1982_2022"),
        cache_root=str(dataset / "precomputed_baselines/v24_res1_gcsmall_1982_2022_bptt24_k20"),
        baseline_checkpoint=resolve(template["data"]["baseline_checkpoint"]),
        stats_dir=resolve(template["data"]["stats_dir"]), reports=str(root / "reports"),
        cache_config=str(root / "configs/cache.json"))
    # A new campaign must never claim another producer's data or cache.
    for key in ("prepared_root", "anchor_root", "cache_root"):
        path = Path(paths[key])
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Output already exists: {path}")
    assets = {paths["baseline_checkpoint"]: sha(paths["baseline_checkpoint"])}
    if parent is not None:
        if parent.baseline_checkpoint_fingerprint != assets[paths['baseline_checkpoint']]:
            raise ValueError('Warm-start frozen baseline differs')
        assets[init_from] = sha(init_from)
    assets.update({str(p): sha(p) for p in sorted(Path(paths["stats_dir"]).glob("*.nc"))})
    if len(assets) < 2:
        raise ValueError("Missing normalization statistics")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("reports", "logs", "configs", "intents"):
        (root / name).mkdir()
    configs = {}
    for width in widths:
        path = root / "configs" / f"w{width}.json"
        config = training_config(template, paths, root, width)
        if parent is not None:
            config['architecture'] = copy.deepcopy(template['architecture'])
        write(path, config)
        configs[str(width)] = str(path)
    write(paths["cache_config"], training_config(template, paths, root, 128, batch=1))
    snapshot = freeze_source(workspace, root / "source")
    campaign = dict(format=FORMAT, root=str(root), workspace=str(workspace), source=str(root / "source"),
        source_digest=snapshot["source_digest"], paths=paths, configs=configs, assets=assets,
        config_sha256={p: sha(p) for p in [*configs.values(), paths["cache_config"]]},
        download_source_sha256=status["source_sha256"], years=YEARS, training_years=[1982, 2021],
        validation_year=2022, numerical_policy="advisory", batch_size=4, max_steps=5000,
        runtime={"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "WEATHER_PREPARED_BOUNDED_READS": "1"},
        storage_reserve_bytes=RESERVE, stop_after=stop_after, created_at=now(),
        widths=widths, init_from=init_from, download_wait_hours=download_wait_hours,
        initialization='parent weights; fresh optimizer/RNG/cursor and zero memory' if parent else 'fresh zero-head residual')
    campaign["workflow_id"] = digest(campaign)
    write(root / "campaign.json", campaign)
    write(root / "state.json", dict(workflow_id=campaign["workflow_id"], automation_enabled=False, jobs={}))
    return campaign


def load_campaign(root):
    campaign = read(Path(root) / "campaign.json")
    body = {k: v for k, v in campaign.items() if k != "workflow_id"}
    if campaign.get("format") != FORMAT or digest(body) != campaign.get("workflow_id"):
        raise ValueError("Campaign identity changed")
    if Path(campaign["root"]).resolve() != Path(root).resolve():
        raise ValueError("Campaign root changed")
    if verify_source(campaign["source"])["source_digest"] != campaign["source_digest"]:
        raise ValueError("Frozen source changed")
    for path, expected in {**campaign["config_sha256"], **campaign["assets"]}.items():
        if sha(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    return campaign


def report_path(campaign, key):
    return Path(campaign["root"]) / "reports" / f"{key}.json"


def save_report(campaign, key, result, *, job_key=None, expected_job_id=None):
    payload = dict(workflow_id=campaign["workflow_id"], key=key, created_at=now(), result=result)
    payload["sha256"] = digest(payload)
    with locked(campaign["root"]):
        if expected_job_id is not None:
            current = read(Path(campaign["root"])/"state.json")["jobs"].get(job_key or key)
            if current is None or current["job_id"] != expected_job_id:
                raise RuntimeError("Retired job attempt cannot publish a report")
        write(report_path(campaign, key), payload)
    return result


def require_report(campaign, key):
    payload = read(report_path(campaign, key))
    if payload.get("workflow_id") != campaign["workflow_id"] or payload.get("sha256") != digest({k:v for k,v in payload.items() if k != "sha256"}):
        raise ValueError(f"Stale or changed report: {key}")
    return payload["result"]


def require_download(campaign):
    paths = campaign["paths"]
    status = read(paths["download_status"])
    if status.get("status") == "failed":
        raise RuntimeError(f"Download failed: {status.get('error')}")
    if status.get("status") != "complete":
        return False
    if Path(status["output"]).resolve() != Path(paths["source_zarr"]).resolve() or status.get("source_sha256") != campaign["download_source_sha256"]:
        raise ValueError("Completed download identity differs")
    report = read(paths["download_report"])
    covered = sorted(set(report.get("existing_years", [])) | set(report.get("appended_years", [])))
    if report.get("dry_run") is not False or report.get("requested_years") != YEARS or covered != YEARS:
        raise ValueError("Download report is dry-run or has incomplete coverage")
    if Path(report["output"]).resolve() != Path(paths["source_zarr"]).resolve():
        raise ValueError("Download report names another store")
    import numpy as np
    import pandas as pd
    import xarray as xr
    with xr.open_zarr(paths["source_zarr"], consolidated=True, chunks=None) as ds:
        expected = pd.date_range("1982-01-01", "2022-12-31T18:00", freq="6h").values
        if not np.array_equal(ds.time.values.astype("datetime64[ns]"), expected):
            raise ValueError("Source does not contain all six-hour timestamps")
        if ds.sizes.get("lat") != 181 or ds.sizes.get("lon") != 360 or ds.sizes.get("level") != 13:
            raise ValueError("Source must be 1 degree with 13 pressure levels")
    return True


def wait_download(campaign, *, timeout_seconds=None, poll_seconds=60):
    if timeout_seconds is None:
        timeout_seconds = campaign.get('download_wait_hours', 72) * 3600
    started = time.monotonic()
    while not require_download(campaign):
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("Download gate exceeded its wait budget; downstream jobs were not released")
        time.sleep(poll_seconds)
    return dict(passed=True, download_report_sha256=sha(campaign["paths"]["download_report"]))


def storage_check(campaign, outstanding_bytes):
    free = shutil.disk_usage(campaign["workspace"]).free
    required = int(outstanding_bytes) + int(campaign["storage_reserve_bytes"])
    if free < required:
        raise RuntimeError(f"Insufficient storage: {free/1024**4:.3f} TiB free; {required/1024**4:.3f} TiB needed including 256 GiB reserve")
    return dict(free_bytes=free, outstanding_bytes=int(outstanding_bytes), reserve_bytes=campaign["storage_reserve_bytes"])


def allocated_bytes(path):
    return sum(p.stat().st_blocks * 512 for p in Path(path).rglob("*") if p.is_file()) if Path(path).exists() else 0


def gpu_selection(reports):
    if any(r.get("failure_kind") == "integrity" for r in reports.values()):
        raise ValueError("GPU qualification found an integrity failure")
    qualified = {k: r for k,r in reports.items() if r.get("passed") is True}
    if not qualified:
        raise RuntimeError("Neither 40 GB GPU type qualified; scientific settings were not changed")
    constraint = "gpu40" if len(qualified) == 2 else GPU_TYPES[next(iter(qualified))]
    return constraint, qualified


def training_budget(reports):
    constraint, qualified = gpu_selection(reports)
    for r in qualified.values():
        for key in ("integrity_passed", "bounded_memory_passed", "full_validation_completed"):
            if r.get(key) is not True:
                raise ValueError(f"Training qualification lacks {key}")
        if r.get("completed_steps", 0) < 70 or not 0 < r.get("peak_gpu_gib", float("inf")) < 32:
            raise ValueError("Training qualification lacks 70 updates or GPU headroom")
    maxima = {k: max(float(r[k]) for r in qualified.values()) for k in (
        "peak_host_gib", "p90_end_to_end_seconds", "startup_seconds", "validation_seconds", "checkpoint_seconds")}
    if any(not math.isfinite(v) or v < 0 for v in maxima.values()) or min(maxima["peak_host_gib"], maxima["p90_end_to_end_seconds"]) <= 0:
        raise ValueError("Invalid measured training resource values")
    return {**maxima, "constraint": constraint, "gpu_types": sorted(qualified), "cpus": 4,
            "memory_gib": max(32, math.ceil(1.4 * maxima["peak_host_gib"] / 16) * 16)}


def training_chunks(budget, *, total=5000, interval=250, validation_every=500):
    result, start = [], 0
    while start < total:
        chosen = None
        for stop in range(start + interval, total + 1, interval):
            validations = stop // validation_every - start // validation_every
            seconds = 1.35 * (budget["startup_seconds"] + (stop-start)*budget["p90_end_to_end_seconds"]
                + validations*budget["validation_seconds"] + ((stop-start)//interval)*budget["checkpoint_seconds"])
            minutes = max(62, math.ceil(seconds/1800)*30)
            if minutes > 1440:
                break
            chosen = dict(start_step=start, stop_step=stop, time_minutes=minutes)
        if chosen is None:
            raise RuntimeError("One checkpoint interval cannot fit in 24 hours")
        result.append(chosen)
        start = chosen["stop_step"]
    return result


def spec(stage, *, width=None, gpu_kind=None, array=None, memory=8, cpus=2, minutes=20, constraint=None, stop=None):
    return dict(stage=stage, width=width, gpu_kind=gpu_kind, array=array, memory_gib=memory,
                cpus=cpus, time_minutes=minutes, constraint=constraint, stop_step=stop)


def sbatch_args(campaign, key, job, dependencies=(), condition="afterok", attempt=1):
    if condition not in ("afterok", "afterany") or any(not str(j).isdigit() for j in dependencies):
        raise ValueError("Invalid Slurm dependency")
    root = Path(campaign["root"])
    token = digest([campaign["workflow_id"], key, attempt])[:24]
    args = ["sbatch", "--parsable", f"--job-name=c40-{key}", "--account=dabanin",
        f"--cpus-per-task={job['cpus']}", f"--mem={job['memory_gib']}G", f"--time={job['time_minutes']}",
        f"--chdir={campaign['workspace']}", f"--comment=c40:{token}",
        f"--output={root}/logs/{key}_%A_%a.out", f"--error={root}/logs/{key}_%A_%a.err"]
    if dependencies:
        args.append(f"--dependency={condition}:" + ":".join(map(str, dependencies)))
    if job["array"] is not None:
        args.append(f"--array=0-{job['array']-1}")
    if job["constraint"]:
        args += ["--partition=gpu", "--gres=gpu:1", f"--constraint={job['constraint']}"]
    else:
        args.append("--partition=cpu")
        if job['time_minutes'] > 4320:
            args.append('--qos=vlong')
    args += [str(Path(campaign["source"])/"scripts/experiments/v24_cached_40yr.slurm"), str(root), key, campaign["workspace"]]
    return args


def submit_job(campaign, key, job, *, dependencies=(), condition="afterok", retry=False):
    if campaign.get("stop_after") == "cache_verify" and (
        job["stage"] in ("training_pilot", "training_release", "train", "evaluate", "report")
        or job.get("width") is not None
    ):
        raise ValueError("Precompute-only campaign cannot submit training or evaluation")
    root = Path(campaign["root"])
    with locked(root):
        state = read(root / "state.json")
        if not state.get("automation_enabled"):
            raise RuntimeError("Campaign submission has not been enabled")
        old = state["jobs"].get(key)
        identity = dict(spec=job, dependencies=list(map(str,dependencies)), condition=condition)
        if old and not retry:
            if any(old[k] != v for k,v in identity.items()):
                raise ValueError(f"Existing job {key} has another identity")
            return old["job_id"]
        retired = [v for v in state.get("retired_jobs", []) if v["key"] == key]
        attempt = old.get("attempt", 1)+1 if old else max([v.get("attempt",1) for v in retired]+[0])+1
        args = sbatch_args(campaign, key, job, dependencies, condition, attempt)
        intent = root / "intents" / f"{key}_{attempt}.json"
        if intent.exists():
            raise RuntimeError(f"Unresolved submission intent: {intent}; reconcile job ID before retrying")
        write(intent, dict(command=args, identity=identity, created_at=now()))
        print(shlex.join(args), flush=True)
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            # A definite sbatch rejection did not create a job. Preserve evidence
            # while allowing a subsequent explicit retry. Ambiguous transport or
            # parsing failures keep the intent and require reconciliation.
            write(intent.with_suffix(".rejected.json"), dict(command=args, error=exc.stderr, created_at=now()))
            intent.unlink()
            raise
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Invalid sbatch response: {result.stdout!r}")
        entry = dict(**identity, job_id=job_id, attempt=attempt, status="submitted", submitted_at=now())
        if old:
            entry["history"] = [*old.get("history", []), {k:v for k,v in old.items() if k != "history"}]
        state["jobs"][key] = entry
        write(root / "state.json", state)
        intent.unlink()
        return job_id


def mark_job(campaign, key, *, expected_job_id=None, **fields):
    with locked(campaign["root"]):
        path = Path(campaign["root"]) / "state.json"
        state = read(path)
        if expected_job_id is not None and state["jobs"].get(key, {}).get("job_id") != expected_job_id:
            raise RuntimeError("Retired job attempt cannot update campaign state")
        state["jobs"][key].update(fields)
        write(path, state)


def launch_pilots(campaign, *, family, dependency, width=None):
    collectors = []
    prefix = family if width is None else f"{family}_w{width}"
    for kind, constraint in GPU_TYPES.items():
        key = f"{prefix}_{kind}"
        pilot = spec(f"{family}_pilot", width=width, gpu_kind=kind, memory=64 if family=="cache" else 128,
                     cpus=4, minutes=(240 if kind=="full" else 360) if family=="cache" else 360, constraint=constraint)
        jid = submit_job(campaign, key, pilot, dependencies=[dependency])
        # afterany is ONLY for collecting qualification failures. Scientific work
        # depends afterok on the collector and then inspects its integrity result.
        collector = submit_job(campaign, key+"_collect", spec("collect", width=width, gpu_kind=kind),
                               dependencies=[jid], condition="afterany")
        collectors.append(collector)
    return submit_job(campaign, prefix+"_release", spec(f"{family}_release", width=width, memory=16, minutes=30), dependencies=collectors)


def advance(campaign, key, job, result):
    state = read(Path(campaign["root"]) / "state.json")
    if not state.get("automation_enabled"):
        return
    jid = state["jobs"][key]["job_id"]
    stage = job["stage"]
    if stage == "download_gate":
        submit_job(campaign, "prepare_pilot", spec("prepare_pilot", memory=32, cpus=4, minutes=240), dependencies=[jid])
    elif stage == "prepare_pilot":
        memory = max(16, math.ceil(1.4*result["peak_host_gib"]/16)*16)
        minutes = max(62, math.ceil(1.35*result["slowest_year_seconds"]/1800)*30)
        array = submit_job(campaign, "prepare_years", spec("prepare_year", array=41, memory=memory, cpus=4, minutes=minutes), dependencies=[jid])
        submit_job(campaign, "prepare_verify", spec("prepare_verify", memory=16, cpus=4, minutes=240), dependencies=[array])
    elif stage == "prepare_verify":
        launch_pilots(campaign, family="cache", dependency=jid)
    elif stage == "cache_release":
        b = result["budget"]
        array = submit_job(campaign, "cache_generate", spec("cache_generate", array=result["num_shards"], memory=b["memory_gib"],
            cpus=4, minutes=b["time_minutes"], constraint=b["constraint"]), dependencies=[jid])
        submit_job(campaign, "cache_verify", spec("cache_verify", memory=16, cpus=2, minutes=240), dependencies=[array])
    elif stage == "cache_verify":
        if campaign.get("stop_after") == "cache_verify":
            save_report(campaign, "precompute_complete", dict(passed=True, cache=result))
            return
        for width in campaign.get('widths', [128, 512]):
            launch_pilots(campaign, family="training", dependency=jid, width=width)
    elif stage == "training_release":
        width, b = job["width"], result["budget"]
        previous = jid
        for chunk in result["chunks"]:
            previous = submit_job(campaign, f"train_w{width}_{chunk['stop_step']:05d}", spec("train", width=width,
                memory=b["memory_gib"], cpus=4, minutes=chunk["time_minutes"], constraint=b["constraint"], stop=chunk["stop_step"]), dependencies=[previous])
        submit_job(campaign, f"evaluate_w{width}", spec("evaluate", width=width, memory=b["memory_gib"], cpus=4,
            minutes=240, constraint=b["constraint"]), dependencies=[previous])
        maybe_submit_report(campaign)


def maybe_submit_report(campaign):
    jobs = read(Path(campaign["root"])/"state.json")["jobs"]
    widths = campaign.get('widths', [128, 512])
    if all(f"evaluate_w{w}" in jobs for w in widths):
        submit_job(campaign, "report", spec("report", memory=4, minutes=20), dependencies=[jobs[f"evaluate_w{w}"]["job_id"] for w in widths])


def collect_pilot(campaign, key):
    pilot_key = key.removesuffix("_collect")
    path = report_path(campaign, pilot_key)
    if path.exists():
        result = require_report(campaign, pilot_key)
        if "passed" in result:
            return result
    state = read(Path(campaign["root"])/"state.json")
    pilot = state["jobs"][pilot_key]
    completed = subprocess.run(["sacct", "-X", "-j", pilot["job_id"], "-n", "-P", "--format=State"],
                               check=True, capture_output=True, text=True).stdout
    states = {line.split("|")[0].split()[0] for line in completed.splitlines() if line.strip()}
    resource = bool(states & {"OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"})
    return dict(passed=False, failure_kind="resource" if resource else "integrity",
                error=pilot.get("error", "Qualification ended without a valid report"), slurm_states=sorted(states))


def require_cache(campaign):
    manifest = read(Path(campaign["paths"]["cache_root"])/"manifest.json")
    ready = read(Path(campaign["paths"]["cache_root"])/"READY.json")
    saved = require_report(campaign, "cache_verify")
    if ready.get("partial") or ready.get("manifest_sha256") != manifest.get("manifest_sha256") or saved["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Training requires the complete verified campaign cache")
    return manifest


def _validate_campaign_anchors(campaign, target):
    """Verify resumed anchors against the actual prepared clock and pinned task."""
    import numpy as np
    import pandas as pd
    from src.models.graphcast.training.core.batching import input_steps_from_duration
    from src.models.graphcast.training.core.model import load_graphcast_checkpoint
    paths = campaign["paths"]
    prepared = Path(paths["prepared_root"])
    checkpoint_path = Path(paths["baseline_checkpoint"]).resolve()
    if (prepared/".incomplete").exists() or (target/".incomplete").exists():
        raise ValueError("Prepared data and anchor manifest must both be complete")
    ready = read(prepared/"READY.json")
    prepared_metadata = read(prepared/"metadata.json")
    prepared_sha = sha(prepared/"metadata.json")
    checkpoint_sha = sha(checkpoint_path)
    if (ready.get("metadata_sha256") != prepared_sha or ready.get("years") != YEARS
            or ready.get("status") != "complete"
            or ready.get("identity", {}).get("checkpoint_sha256") != checkpoint_sha
            or prepared_metadata.get("checkpoint_sha256") != checkpoint_sha
            or campaign["assets"].get(str(checkpoint_path)) != checkpoint_sha
            or Path(prepared_metadata["source_data_path"]).resolve() != Path(paths["source_zarr"]).resolve()
            or ready.get("identity", {}).get("source_metadata_sha256") != sha(Path(paths["source_zarr"])/".zmetadata")):
        raise ValueError("Prepared/checkpoint identity differs from the anchor campaign")
    time_path = prepared/"coords/time.npy"
    if ready.get("root_file_sha256", {}).get("coords/time.npy") != sha(time_path):
        raise ValueError("Prepared time coordinate fingerprint changed")
    times = np.load(time_path, allow_pickle=False)
    expected_times = pd.date_range("1982-01-01", "2022-12-31 18:00", freq="6h").values
    if not np.issubdtype(times.dtype, np.datetime64) or not np.array_equal(times, expected_times):
        raise ValueError("Prepared anchor clock must cover the exact 1982–2022 six-hour grid")
    checkpoint = load_graphcast_checkpoint(checkpoint_path)
    task = checkpoint.task_config
    input_steps = input_steps_from_duration(task.input_duration, pd.Timedelta(hours=6))
    anchors = np.arange(input_steps, len(times)-2, dtype=np.int64)
    years = pd.DatetimeIndex(times[anchors]).year.to_numpy()
    expected = {"anchor_indices.npy": anchors, "anchor_times.npy": times[anchors],
                "split_train.npy": np.flatnonzero(years <= 2021), "split_val.npy": np.flatnonzero(years == 2022)}
    for name, values in expected.items():
        actual = np.load(target/"anchors"/name, allow_pickle=False)
        kind = np.datetime64 if name == "anchor_times.npy" else np.integer
        if actual.ndim != 1 or not np.issubdtype(actual.dtype, kind) or not np.array_equal(actual, values):
            raise ValueError(f"Anchor indices/timestamps/split differ from prepared data: {name}")
    metadata = read(target/"metadata.json")
    required = {"train_end_year": 2021, "validation_year": 2022, "input_steps": input_steps,
                "target_steps": 1, "input_duration": str(task.input_duration),
                "pressure_levels": list(task.pressure_levels), "target_variables": list(task.target_variables),
                "n_anchors_total": len(anchors), "n_anchors_train": len(expected["split_train.npy"]),
                "n_anchors_val": len(expected["split_val.npy"]),
                "source_time_start": str(times[0]), "source_time_end": str(times[-1]),
                "campaign_prepared_metadata_sha256": prepared_sha,
                "campaign_baseline_checkpoint_sha256": checkpoint_sha,
                "campaign_workflow_id": campaign["workflow_id"]}
    if (Path(metadata["source_prepared_root"]).resolve() != prepared.resolve()
            or Path(metadata["baseline_checkpoint"]).resolve() != checkpoint_path
            or any(metadata.get(key) != value for key, value in required.items())):
        raise ValueError("Anchor metadata differs from the pinned prepared data/checkpoint")


def prepare_anchors(campaign):
    """Publish anchors atomically; interrupted private builds remain reviewable."""
    from src.models.mamba.v24_Ilya.anchor_manifest import build_anchor_manifest
    paths=campaign["paths"]
    target=Path(paths["anchor_root"])
    if target.exists():
        _validate_campaign_anchors(campaign,target)
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.with_name(target.name+".building-"+campaign["workflow_id"][:12])
    if staging.exists():
        archived=staging.with_name(staging.name+f".interrupted-{time.time_ns()}")
        staging.rename(archived)
    build_anchor_manifest(prepared_root=Path(paths["prepared_root"]),baseline_checkpoint=Path(paths["baseline_checkpoint"]),output_root=staging)
    if (staging/".incomplete").exists():
        raise ValueError("Anchor builder did not finish")
    metadata=read(staging/"metadata.json")
    metadata.update(campaign_prepared_metadata_sha256=sha(Path(paths["prepared_root"])/"metadata.json"),
                    campaign_baseline_checkpoint_sha256=sha(paths["baseline_checkpoint"]),
                    campaign_workflow_id=campaign["workflow_id"])
    write(staging/"metadata.json",metadata)
    _validate_campaign_anchors(campaign,staging)
    staging.rename(target)


def verify_target_split(manifest):
    train=val=0
    for chunk in manifest["chunks"]:
        years={int(timestamp[:4]) for timestamp in chunk["target_timestamps"]}
        if chunk["split"]=="train":
            if not years or not years.issubset(set(range(1982,2022))):
                raise ValueError("Training target timestamps leak outside 1982–2021")
            train+=1
        elif chunk["split"]=="val":
            if years!={2022}:
                raise ValueError("Validation targets must be in 2022")
            val+=1
        else:
            raise ValueError("Unexpected cache split")
    if not train or not val:
        raise ValueError("Both training and validation require complete chunks")
    return {"train_chunks":train,"validation_chunks":val,"excluded_tail_anchors":manifest.get("excluded_tail_anchors",{})}


def execute_stage(campaign, key, job, *, index=None):
    stage, paths = job["stage"], campaign["paths"]
    if stage == "download_gate":
        return wait_download(campaign)
    if stage in ("prepare_pilot", "prepare_year", "prepare_verify"):
        from src.data_operations.preprocessing.prepare_cached_40yr import prepare_year, finalize_prepared
        if not require_download(campaign):
            raise RuntimeError("Preprocessing requires completed download")
        if stage == "prepare_verify":
            result = finalize_prepared(Path(paths["source_zarr"]), Path(paths["prepared_root"]), YEARS, Path(paths["baseline_checkpoint"]))
            prepare_anchors(campaign)
            from src.models.mamba.v24_Ilya.baseline_cache import build_manifest,load_context
            split=verify_target_split(build_manifest(load_context(Path(paths["cache_config"])),40))
            return dict(passed=True, preparation=result, anchors=paths["anchor_root"], split=split)
        if stage == "prepare_year":
            require_report(campaign, "prepare_pilot")
        # Conservative remaining prepared/cache payloads; refined after generation.
        storage_check(campaign, max(0, int(1.23*1024**4)-allocated_bytes(paths["prepared_root"])) + int(1.20*1024**4))
        years = [1982, 1984] if stage == "prepare_pilot" else [YEARS[index]]
        import resource
        results, elapsed = [], []
        for year in years:
            started = time.monotonic()
            results.append(prepare_year(Path(paths["source_zarr"]), Path(paths["prepared_root"]), year, Path(paths["baseline_checkpoint"])))
            elapsed.append(time.monotonic()-started)
        return dict(passed=True, years=years, results=results, slowest_year_seconds=max(elapsed), peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    if stage == "collect":
        return collect_pilot(campaign, key)
    if stage == "cache_pilot":
        require_report(campaign, "prepare_verify")
        from scripts.experiments.cached_40yr_qualification import execute_cache_pilot
        return execute_cache_pilot(campaign, job["gpu_kind"])
    if stage == "cache_release":
        from src.models.mamba.v24_Ilya.baseline_cache import build_manifest, load_context, ensure_manifest
        reports = {k:require_report(campaign, f"cache_{k}_collect") for k in GPU_TYPES}
        constraint, passing = gpu_selection(reports)
        for report in passing.values():
            if report.get("gpu_memory_passed") is not True or not 0 < report.get("peak_gpu_gib", float("inf")) < 32:
                raise ValueError("Cache qualification lacks measured GPU headroom")
        context = load_context(Path(paths["cache_config"]))
        initial = build_manifest(context, 40)
        steady = max(float(r["steady_chunk_seconds"]) for r in passing.values())
        startup = max(float(r["initialization_seconds"])+float(r["compilation_seconds_estimate"]) for r in passing.values())
        if not math.isfinite(steady) or steady <= 0 or not math.isfinite(startup) or startup < 0:
            raise ValueError("Invalid measured cache throughput")
        capacity = math.floor((24*3600/1.35-startup)/steady)
        if capacity < 1:
            raise RuntimeError("One cache chunk cannot fit in a 24-hour allocation")
        n = max(40, math.ceil(len(initial["chunks"])/capacity))
        manifest = build_manifest(context,n)
        storage_check(campaign, max(0,manifest["prediction_bytes"]-allocated_bytes(paths["cache_root"])))
        ensure_manifest(Path(paths["cache_root"]),manifest)
        peak = max(float(r["peak_host_gib"]) for r in passing.values())
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("Invalid measured cache host memory")
        minutes = max(62,math.ceil(1.35*(startup+math.ceil(len(manifest["chunks"])/n)*steady)/1800)*30)
        return dict(passed=True,num_shards=n,manifest_sha256=manifest["manifest_sha256"],budget=dict(constraint=constraint,
            memory_gib=max(32,math.ceil(1.4*peak/16)*16),time_minutes=minutes))
    if stage == "cache_generate":
        from src.models.mamba.v24_Ilya.baseline_cache import build_manifest, load_context, generate_shard
        release = require_report(campaign,"cache_release")
        context = load_context(Path(paths["cache_config"]))
        manifest = build_manifest(context,release["num_shards"])
        if manifest["manifest_sha256"] != release["manifest_sha256"]:
            raise ValueError("Cache producer inputs changed after release")
        storage_check(campaign,max(0,manifest["prediction_bytes"]-allocated_bytes(paths["cache_root"])))
        return generate_shard(context,manifest,Path(paths["cache_root"]),index,resume=True)
    if stage == "cache_verify":
        from src.models.mamba.v24_Ilya.baseline_cache import verify_cache
        require_report(campaign,"cache_release")
        return verify_cache(Path(paths["cache_root"]))
    if stage == "training_pilot":
        require_cache(campaign)
        from scripts.experiments.cached_40yr_qualification import execute_training_pilot
        return execute_training_pilot(campaign,job["width"],job["gpu_kind"])
    if stage == "training_release":
        require_cache(campaign)
        width = job["width"]
        reports = {k: require_report(campaign,f"training_w{width}_{k}_collect") for k in GPU_TYPES}
        budget = training_budget(reports)
        chosen = next(reports[k] for k in budget["gpu_types"])
        proof = {k: chosen[k] for k in ("shared_init", "integrity_report")}
        proof["integrity_sha256"] = sha(proof["integrity_report"])
        return dict(passed=True,budget=budget,chunks=training_chunks(budget),proof=proof)
    if stage == "train":
        cache_manifest = require_cache(campaign)
        release = require_report(campaign,f"training_w{job['width']}_release")
        proof = release["proof"]
        if sha(proof["integrity_report"]) != proof["integrity_sha256"]:
            raise ValueError("Training integrity report changed")
        storage_check(campaign,0)
        from src.models.mamba.v24_Ilya.training.config import load_training_config
        from src.models.mamba.v24_Ilya.training.cached_runner import run_cached_training, latest_checkpoint_for_run
        from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
        config_path = Path(campaign["configs"][str(job["width"])])
        config = load_training_config(config_path)
        resume = latest_checkpoint_for_run(config)
        chunk = next(c for c in release["chunks"] if c["stop_step"] == job["stop_step"])
        recovered_step = load_v24_Ilya_training_checkpoint(resume).completed_step if resume is not None else 0
        if not chunk["start_step"] <= recovered_step <= chunk["stop_step"]:
            raise ValueError("Recovered checkpoint is outside this allocation; refusing to restart or rewind training")
        path = run_cached_training(config_path,Path(proof["shared_init"]),cache_root=Path(paths["cache_root"]),
            resume=resume,max_steps=job["stop_step"],numerical_policy="advisory",
            expected_manifest_sha256=cache_manifest["manifest_sha256"],
            parity_report=Path(proof["integrity_report"]),paired_report=Path(proof["integrity_report"]),
            integrity_report=Path(proof["integrity_report"]))
        completed = load_v24_Ilya_training_checkpoint(path).completed_step
        if completed != job["stop_step"]:
            raise RuntimeError("Training did not reach its required checkpoint")
        return dict(passed=True,checkpoint=str(path),completed_steps=completed,consumed_sequence_chunks=4*completed)
    if stage == "evaluate":
        require_cache(campaign)
        completed = require_report(campaign, f"train_w{job['width']}_05000")
        if completed.get("completed_steps") != 5000:
            raise ValueError("Evaluation requires the completed 5,000-update training branch")
        from scripts.experiments.cached_40yr_qualification import execute_evaluation
        return execute_evaluation(campaign,job["width"])
    if stage == "report":
        evaluations = {str(w):require_report(campaign,f"evaluate_w{w}") for w in campaign.get('widths', [128,512])}
        result = dict(passed=True,workflow_id=campaign["workflow_id"],evaluations=evaluations,
            source_digest=campaign["source_digest"],cache_manifest_sha256=require_cache(campaign)["manifest_sha256"],
            prepared_data=require_report(campaign,"prepare_verify"),
            consumed_sequence_chunks_per_width=20000,
            jobs=read(Path(campaign["root"])/"state.json")["jobs"],
            qualification={str(w):require_report(campaign,f"training_w{w}_release") for w in campaign.get('widths', [128,512])})
        write(Path(campaign["root"])/"results.json",result)
        lines = ["# Cached 40-year training comparison", "", "Training: 1982–2021; validation/evaluation: 2022.",
            f"Initialization: {campaign.get('initialization', 'fresh')}; open-loop K20; batch 4; 5,000 updates / 20,000 sequence chunks per width.", "",
            f"Workflow: `{campaign['workflow_id']}`", f"Cache: `{result['cache_manifest_sha256']}`", "",
            "| Width | Selected update | Exact validation loss | Open-loop improvement | Closed-loop improvement |",
            "| --- | --- | --- | --- | --- |"]
        for w, evaluation in evaluations.items():
            selected=evaluation["selection"];modes=evaluation["modes"]
            lines += [f"| {w} | {selected['step']} | {selected['selection_loss']:.8f} | {modes['cold_bp']['improvement_pct']:.3f}% | {modes['cold_full']['improvement_pct']:.3f}% |"]
        lines += ["", "Positive improvement means lower aggregate GraphCast loss than the frozen baseline on the 32-start, 10-day protocol.",
                  "", "## Measured resources", "", "| Width | Qualified GPUs | Update seconds (p90) | Host peak GiB | Requested RAM GiB |",
                  "| --- | --- | --- | --- | --- |"]
        for w, qualification in result["qualification"].items():
            b=qualification["budget"]
            lines += [f"| {w} | {', '.join(b['gpu_types'])} | {b['p90_end_to_end_seconds']:.2f} | {b['peak_host_gib']:.2f} | {b['memory_gib']} |"]
        lines += ["", "`results.json` contains all job IDs, dependencies, data identities, qualification measurements, and evaluation artifacts.", ""]
        (Path(campaign["root"])/"REPORT.md").write_text("\n".join(lines))
        return result
    raise ValueError(f"Unknown stage: {stage}")


def execute(campaign, key, *, index=None):
    root=Path(campaign["root"])
    entry=read(root/"state.json")["jobs"][key]
    job=entry["spec"]
    if os.environ.get("SLURM_JOB_ID") and str(os.environ.get("SLURM_ARRAY_JOB_ID",os.environ["SLURM_JOB_ID"])) != entry["job_id"]:
        raise ValueError("Stale job attempt may not execute this campaign")
    report_key=key if job["array"] is None else f"{key}_{index:03d}"
    if job["array"] is None:
        mark_job(campaign,key,expected_job_id=entry["job_id"],status="running",started_at=now())
    try:
        result=execute_stage(campaign,key,job,index=index)
        save_report(campaign,report_key,result,job_key=key,expected_job_id=entry["job_id"])
        if job["array"] is None:
            mark_job(campaign,key,expected_job_id=entry["job_id"],status="complete",ended_at=now())
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        save_report(campaign,report_key,dict(passed=False,error=error,failure_kind="integrity"),job_key=key,expected_job_id=entry["job_id"])
        mark_job(campaign,key,expected_job_id=entry["job_id"],status="failed",ended_at=now(),error=error)
        raise
    if job["array"] is None:
        try:
            advance(campaign,key,job,result)
        except BaseException as exc:
            mark_job(campaign,key,expected_job_id=entry["job_id"],status="release_failed",error=f"{type(exc).__name__}: {exc}")
            # Scientific work succeeded. Keep its Slurm dependency successful;
            # status/resume exposes and repairs an interrupted submission tail.
            print(f"Automatic release failed for {key}: {exc}; use resume --stage {key}",file=sys.stderr,flush=True)
    return result


def dry_run(campaign):
    first=spec("download_gate",memory=1,cpus=1,minutes=campaign.get('download_wait_hours',72)*60+5)
    result = dict(submitted=False,workflow_id=campaign["workflow_id"],first_command=sbatch_args(campaign,"download_gate",first),
        stages=["download_gate", "prepare_pilot", "prepare_years[41]", "prepare_verify",
                "cache_full+cache_mig", "cache_collectors", "cache_release", "cache_generate[>=40]", "cache_verify",
                '+'.join(f'training_w{w}_{kind}' for w in campaign.get('widths', [128,512]) for kind in GPU_TYPES),
                "training_collectors", "per_width_release", "per_width_training_<=24h", "per_width_evaluation", "report"],
        training={"batch_size":4,"updates_per_width":5000,"sequence_chunks_per_width":20000},
        note="Later submission commands are resolved automatically from successful measured pilots. No jobs submitted.")
    result["stop_after"] = campaign.get("stop_after", "report")
    if result["stop_after"] == "cache_verify":
        result["stages"] = result["stages"][:result["stages"].index("cache_verify")+1]
        result.pop("training")
    return result


def submit(campaign, *, dry=False):
    if dry:
        return dry_run(campaign)
    with locked(campaign["root"]):
        path=Path(campaign["root"])/"state.json"
        state=read(path);state["automation_enabled"]=True;write(path,state)
    return {"download_gate":submit_job(campaign,"download_gate",spec("download_gate",memory=1,cpus=1,minutes=campaign.get('download_wait_hours',72)*60+5))}


def status(campaign):
    state=read(Path(campaign["root"])/"state.json")
    result=dict(workflow_id=campaign["workflow_id"],automation_enabled=state["automation_enabled"],
                stop_after=campaign.get("stop_after", "report"),
                download=read(campaign["paths"]["download_status"]),jobs=state["jobs"])
    if state["jobs"]:
        ids=",".join(j["job_id"] for j in state["jobs"].values())
        result["accounting"]=subprocess.run(["sacct","-X","-j",ids,"-n","-P","--format=JobID,State,Elapsed,Submit,Start,End"],
            check=True,capture_output=True,text=True).stdout
    return result


def resume(campaign, key):
    """Explicit retry. Descendants are released afresh; no scientific settings change."""
    with locked(campaign["root"], ".resume.lock"):
        return _resume(campaign, key)


def _resume(campaign, key):
    root=Path(campaign["root"])
    state=read(root/"state.json")
    old=state["jobs"].get(key)
    if old is None:
        raise ValueError(f"No recorded stage: {key}")
    active=subprocess.run(["squeue","-h","-j",old["job_id"],"-o","%T"],check=True,capture_output=True,text=True).stdout.strip()
    if active:
        raise RuntimeError("Stage still has a queued/running job; resume will not duplicate it")
    # Successful execution may have been interrupted while submitting successors.
    if old.get("status") in ("complete", "release_failed") and report_path(campaign,key).exists():
        result = require_report(campaign,key)
        if result.get("passed", True):
            accounting = subprocess.run(["sacct", "-X", "-j", old["job_id"], "-n", "-P", "--format=State"],
                                        check=True, capture_output=True, text=True).stdout
            states = {row.split("|")[0].split()[0] for row in accounting.splitlines() if row.strip()}
            if states == {"COMPLETED"}:
                advance(campaign,key,old["spec"],result)
                mark_job(campaign,key,status="complete")
                return {"advanced":key}
            if not states:
                raise RuntimeError("Slurm has not published final accounting; retry resume after it is available")
    # Retire only descendants of this failed attempt. Independent width/GPU
    # branches keep running. Rebuilt successors use the new successful job IDs.
    dependent, invalid_ids = set(), {old["job_id"]}
    while True:
        found={k for k,v in state["jobs"].items() if k not in dependent and any(j in invalid_ids for j in v["dependencies"])}
        if not found:
            break
        dependent.update(found);invalid_ids.update(state["jobs"][k]["job_id"] for k in found)
    for child in sorted(dependent):
        entry=state["jobs"][child]
        queued=subprocess.run(["squeue","-h","-j",entry["job_id"],"-o","%T|%r"],check=True,capture_output=True,text=True).stdout.strip()
        rows=[row.split("|",1) for row in queued.splitlines()]
        if any(s != "PENDING" or reason not in ("Dependency", "DependencyNeverSatisfied") for s,reason in rows):
            raise RuntimeError(f"Dependent stage {child} is active; it cannot be replaced")
        if entry.get("status")=="complete" and entry["spec"]["stage"] not in ("collect", "cache_release", "training_release"):
            raise RuntimeError(f"Dependent stage {child} already produced scientific output; refusing automatic replacement")
    for child in sorted(dependent):
        entry=state["jobs"][child]
        subprocess.run(["scancel",entry["job_id"]],check=False,capture_output=True,text=True)
    for child in sorted(dependent):
        entry=state["jobs"][child]
        active=subprocess.run(["squeue","-h","-j",entry["job_id"],"-o","%T"],check=True,capture_output=True,text=True).stdout.strip()
        if active:
            raise RuntimeError(f"Dependent stage {child} has not stopped after cancellation; retry resume after it exits")
    with locked(root):
        current=read(root/"state.json")
        if any(current["jobs"].get(k, {}).get("job_id") != state["jobs"][k]["job_id"] for k in {*dependent, key}):
            raise RuntimeError("Campaign changed during recovery; no jobs were replaced")
        if any(k not in dependent and any(j in invalid_ids for j in entry["dependencies"])
               for k,entry in current["jobs"].items()):
            raise RuntimeError("New dependent jobs appeared during recovery; retry resume before replacement")
        state=current
        for child in sorted(dependent):
            entry=state["jobs"].pop(child)
            state.setdefault("retired_jobs",[]).append(dict(key=child,retired_at=now(),**entry))
            path=report_path(campaign,child)
            if path.exists():
                path.rename(path.with_name(path.stem+f".attempt{entry['attempt']}.retired.json"))
        write(root/"state.json",state)
        path=report_path(campaign,key)
        if path.exists():
            path.rename(path.with_name(path.stem+f".attempt{old['attempt']}.retired.json"))
    new_id=submit_job(campaign,key,old["spec"],dependencies=old["dependencies"],condition=old["condition"],retry=True)
    stage=old["spec"]["stage"]
    if stage in ("cache_pilot", "training_pilot"):
        family=stage.removesuffix("_pilot")
        launch_pilots(campaign,family=family,dependency=old["dependencies"][0],width=old["spec"]["width"])
    elif stage=="collect":
        family="training" if old["spec"]["width"] is not None else "cache"
        pilot=state["jobs"][key.removesuffix("_collect")]
        launch_pilots(campaign,family=family,dependency=pilot["dependencies"][0],width=old["spec"]["width"])
    elif stage=="prepare_year":
        submit_job(campaign,"prepare_verify",spec("prepare_verify",memory=16,cpus=4,minutes=240),dependencies=[new_id])
    elif stage=="cache_generate":
        submit_job(campaign,"cache_verify",spec("cache_verify",memory=16,cpus=2,minutes=240),dependencies=[new_id])
    elif stage=="train":
        width=old["spec"]["width"]
        release_key=f"training_w{width}_release"
        # Rewire all continuations using the new ID; preceding completed chunks
        # retain their original recorded dependency and are not resubmitted.
        b=require_report(campaign,release_key)["budget"]
        previous=new_id
        for chunk in require_report(campaign,release_key)["chunks"]:
            if chunk["stop_step"] <= old["spec"]["stop_step"]:
                continue
            previous=submit_job(campaign,f"train_w{width}_{chunk['stop_step']:05d}",spec("train",width=width,
                memory=b["memory_gib"],cpus=4,minutes=chunk["time_minutes"],constraint=b["constraint"],stop=chunk["stop_step"]),dependencies=[previous])
        submit_job(campaign,f"evaluate_w{width}",spec("evaluate",width=width,memory=b["memory_gib"],cpus=4,minutes=240,constraint=b["constraint"]),dependencies=[previous])
        maybe_submit_report(campaign)
    elif stage=="evaluate":
        maybe_submit_report(campaign)
    return {key:new_id}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","submit","status","resume","execute"))
    parser.add_argument("--campaign-root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--download-status",type=Path,default=DEFAULT_DOWNLOAD)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument('--widths', type=int, nargs='+', choices=(128,512), default=[128,512])
    parser.add_argument('--init-from', type=Path)
    parser.add_argument('--download-wait-hours', type=int, default=72)
    parser.add_argument("--stage")
    parser.add_argument("--index",type=int)
    parser.add_argument("--stop-after",choices=("cache_verify", "report"),default="report",
                        help="Immutable endpoint set by prepare; cache_verify enables precomputation only")
    args=parser.parse_args()
    if args.command=="prepare":
        result=prepare(args.campaign_root,download_status=args.download_status,stop_after=args.stop_after,
                       widths=args.widths,init_from=args.init_from,download_wait_hours=args.download_wait_hours)
    else:
        campaign=load_campaign(args.campaign_root)
        if args.command=="submit":result=submit(campaign,dry=args.dry_run)
        elif args.command=="status":result=status(campaign)
        elif not args.stage:parser.error("execute/resume requires --stage")
        elif args.command=="resume":result=resume(campaign,args.stage)
        else:result=execute(campaign,args.stage,index=args.index)
    print(json.dumps(result,indent=2,default=str),flush=True)


if __name__=="__main__":
    main()
