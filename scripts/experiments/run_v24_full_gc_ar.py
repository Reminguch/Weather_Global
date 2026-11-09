#!/usr/bin/env python3
"""Full-GC AR-horizon comparison; preparation does not submit jobs."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments.frozen_runtime import freeze_source, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal
from scripts.experiments import run_v24_scalar_lambda as scalar

PARENT = "artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107"
K20_ID = "gcft_s22_l000_lr3em7_fresh_20261107"
DEFAULT = "artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109"
HORIZONS = (8, 12, 16, 20)
EARLY = (500, 1000)
LATE = (1500, 2000, 2500, "swa")
SWA_STEPS = (1000, 1500, 2000, 2500)
WRAPPER = "scripts/experiments/v24_full_gc_ar.slurm"
DRIVER = "scripts/experiments/run_v24_full_gc_ar.py"


def make_config(template, experiment, horizon):
    if horizon not in HORIZONS[:-1]:
        raise ValueError("Only the three new horizons can be prepared")
    config = copy.deepcopy(template)
    config["sequence"]["ar_tail_k"] = horizon
    config["output"] = {"output_root": str(experiment / "runs"),
                        "run_name": f"gcft_s22_k{horizon:02d}_lr3em7"}
    return config


def task_index(stage, index):
    points = EARLY if stage == "evaluate-early" else LATE if stage == "evaluate-late" else None
    if points is None or not 0 <= index < len(HORIZONS) * len(points):
        raise ValueError("Invalid evaluation task")
    return HORIZONS[index // len(points)], points[index % len(points)]


def checkpoint_path(root, manifest, horizon, point):
    if point == "swa":
        return root / "swa" / f"k{horizon:02d}_steps01000-02500.pkl"
    if horizon not in HORIZONS or point not in EARLY + LATE[:-1]:
        raise ValueError("Invalid checkpoint")
    run = Path(manifest["k20_run"]) if horizon == 20 else root / "runs" / manifest["tasks"][str(horizon)]["id"]
    return run / "checkpoints" / f"checkpoint_step{point:08d}.pkl"


def config_path(manifest, horizon):
    return Path(manifest["k20_config"]["path"]) if horizon == 20 else Path(manifest["tasks"][str(horizon)]["config"]["path"])


def prepare(root, workspace):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    parent = workspace / PARENT
    original = scalar.verify(parent)
    parent_source = verify_source(parent / "source")
    # Existing source files must be identical; only this new driver/wrapper may be added.
    for relative, expected in parent_source["files"].items():
        from scripts.experiments.frozen_runtime import file_sha256
        if not (workspace / relative).is_file() or file_sha256(workspace / relative) != expected:
            raise ValueError(f"Training source differs from K20 frozen source: {relative}")
    base = parent / "configs" / f"{K20_ID}.json"
    template = read(base)
    if (template["model_family"] != "full_gc" or template["sequence"]["ar_tail_k"] != 20
        or template["sequence"]["bptt_steps"] != 24 or template["optimizer"]["max_steps"] != 2500
        or template["optimizer"]["learning_rate"] != 3e-7
        or template["objective"]["spectral_excess_weight"] != 0):
        raise ValueError("Existing K20 configuration differs from the comparison contract")
    if root.exists():
        raise FileExistsError("Use a new campaign directory")
    root.mkdir(parents=True)
    for name in ("configs", "runs", "checks", "submissions", "logs", "swa", "evaluation", "report"):
        (root / name).mkdir()
    tasks = {}
    for horizon in HORIZONS[:-1]:
        config = make_config(template, root, horizon)
        path = root / "configs" / f"k{horizon:02d}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        load_training_config(path)
        tasks[str(horizon)] = {"id": config["output"]["run_name"], "model_family": "full_gc",
                               "lambda": 0., "ar_tail_k": horizon, "config": pinned(path)}
    source = freeze_source(workspace, root / "source")
    manifest = seal(root / "manifest.json", {
        "format": "v24_full_gc_ar_v1", "workspace": str(workspace), "experiment": str(root),
        "source_root": str(root / "source"), "source_digest": source["source_digest"],
        "parent_source_digest": original["source_digest"], "parent_manifest": pinned(parent / "manifest.json"),
        "parent_inputs": original["inputs"], "development_manifest": original["development_manifest"],
        "development_data": original["development_data"], "k20_config": pinned(base),
        "k20_run": str(parent / "runs" / K20_ID), "k20_continuation_job": "17049163_0",
        "tasks": tasks, "horizons": list(HORIZONS), "bptt_steps": 24,
        "teacher_prefix_steps": {str(k): 24-k for k in HORIZONS}, "segment_steps": 96,
        "train_order": pinned(template["data"]["train_order_manifest"]),
        "target_steps": 40, "development_starts": 32, "training_steps": 2500,
        "early_steps": list(EARLY), "late_steps": list(LATE[:-1]), "swa_steps": list(SWA_STEPS),
        "policy": {"loss": "exact original GC grid MSE", "seed": 22, "peak_lr": 3e-7,
                   "finite_drift": "warning", "improvement_required": False,
                   "forecast_power": "physical spherical T179, degree zero retained",
                   "selection": "all fixed checkpoints and SWA reported; development set only",
                   "swa": "equal weight on native updates 1000/1500/2000/2500; inference only"},
        "budget_evidence": {"job": "17049162_0", "elapsed": "20:07:26",
                            "requested_memory_gib": 384, "batch_maxrss_kib": 235440960,
                            "measured_p90_seconds": 57.799, "train_stage_minutes": 1800,
                            "spherical_evaluation_reference": "18 starts in45.8min, 9.8GiB RSS; 32-start extrapolation about80min",
                            "evaluation_minutes": 150, "evaluation_memory_gib": 20}})
    seal(root / "submission_plan.json", submission_plan(root, manifest))
    return manifest


def verify(root, frozen=True):
    root = Path(root).resolve()
    manifest = unseal(root / "manifest.json")
    if manifest["format"] != "v24_full_gc_ar_v1" or manifest["experiment"] != str(root):
        raise ValueError("Campaign identity differs")
    if frozen and ROOT.resolve() != Path(manifest["source_root"]).resolve():
        raise ValueError("Worker must run the frozen source")
    if verify_source(manifest["source_root"])["source_digest"] != manifest["source_digest"]:
        raise ValueError("Source snapshot differs")
    for record in [manifest["parent_manifest"], manifest["development_manifest"],
                   manifest["train_order"], manifest["k20_config"], *manifest["parent_inputs"],
                   *(t["config"] for t in manifest["tasks"].values())]:
        check(record)
    return manifest


def build_swa(root, manifest, horizon):
    import jax
    from src.models.mamba.v24_Ilya.checkpoint import build_v24_Ilya_swa, load_v24_Ilya_checkpoint
    output = checkpoint_path(root, manifest, horizon, "swa")
    inputs = [checkpoint_path(root, manifest, horizon, step) for step in SWA_STEPS]
    records = [pinned(path) for path in inputs]
    payload = build_v24_Ilya_swa(inputs, list(SWA_STEPS), output, residual_state_policy="zero")
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(payload["residual_params"])):
        raise ValueError("Nonfinite SWA parameters")
    loaded = load_v24_Ilya_checkpoint(output)
    if loaded.model_family != "full_gc":
        raise ValueError("SWA does not identify as full GC")
    for record in records:
        check(record)
    return seal(root / "checks" / f"swa_k{horizon:02d}.json", {
        "status": "complete", "horizon": horizon, "sources": records,
        "source_steps": list(SWA_STEPS), "checkpoint": pinned(output),
        "manifest_hash": manifest["record_hash"]})


class RawSphericalObserver:
    """Stream raw spherical powers for both forecasts on each development start."""
    def __init__(self, output_dir, diffs_stddev, identity):
        from src.analysis.smoothing_observer import SmoothingObserver
        self.base = SmoothingObserver(output_dir, diffs_stddev, identity, mode="calibrate", compute_zonal=False)
        self.base._calibrate = self._record

    def begin_sample(self, metadata):
        self.base.begin_sample(metadata)

    def _record(self, predictions, target, group):
        base = self.base
        channels, degrees = len(base.layout.channels), base.lmax + 1
        record = {"truth_power": np.empty((channels, degrees)),
                  "prediction_power": np.empty((2, channels, degrees)),
                  "error_power": np.empty((2, channels, degrees)),
                  "cross_power": np.empty((2, channels, degrees))}
        for branch, forecast in enumerate(predictions):
            for start in range(0, channels, base.chunk_channels):
                selection = slice(start, min(start + base.chunk_channels, channels))
                stats = base.operator.spectral_chunk(forecast[selection], target[selection])
                for key in ("prediction_power", "error_power", "cross_power"):
                    record[key][branch, selection] = stats[key]
                if branch == 0:
                    record["truth_power"][selection] = stats["truth_power"]
        if not all(np.isfinite(value).all() for value in record.values()):
            raise ValueError("Nonfinite spherical powers")
        base.records.append(record)

    def update_step(self, index, truth, baseline, full):
        self.base.update_step(index, truth, baseline, full)

    def finish_sample(self):
        from src.analysis.smoothing_observer import atomic_npz
        base = self.base
        if base.sample is None or base.steps != base.target_steps or len(base.records) != base.target_steps:
            raise ValueError("Incomplete spherical forecast")
        arrays = {key: np.stack([r[key] for r in base.records]) for key in base.records[0]}
        metadata = {**base.sample, "status": "complete", "layout": base.layout.metadata(),
                    "branches": ["public_gc", "full_gc"], "lmax": base.lmax,
                    "spectrum": "physical spherical fitted coefficients; degree zero retained"}
        atomic_npz(base.path, arrays, metadata)
        base.sample = base.accumulated = None
        base.records = []


def evaluate(root, manifest, horizon, point):
    from src.models.graphcast.training.core.model import load_stats
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    scalar.gpu_required()
    checkpoint = checkpoint_path(root, manifest, horizon, point)
    reference = pinned(checkpoint)
    config = read(config_path(manifest, horizon))
    name = f"step{point:05d}" if isinstance(point, int) else "swa"
    dest = root / "evaluation" / f"k{horizon:02d}" / name
    if dest.exists():
        raise FileExistsError(f"Evaluation output exists: {dest}")
    dest.mkdir(parents=True)
    observer = RawSphericalObserver(dest / "spectra", load_stats(Path(config["data"]["stats_dir"]))["diffs_stddev_by_level"],
        {"campaign_hash": manifest["record_hash"], "horizon": horizon, "checkpoint_sha256": reference["sha256"]})
    cfg = V24IlyaEvalConfig(ckpt=checkpoint, out_json=dest / "evaluation.json", model_family="full_gc",
        ckpt_in=Path(config["data"]["baseline_checkpoint"]), stats_dir=Path(config["data"]["stats_dir"]),
        data_path=manifest["development_data"], initialization_manifest=Path(manifest["development_manifest"]["path"]),
        model_id=f"k{horizon:02d}_{name}", val_year=2022, train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=32, warmup_steps=0, eval_mode="cold_full", residual_state_init="zero",
        omit_rms_bias=True, **config["architecture"])
    result = evaluate_v24_Ilya(cfg, prediction_observer=observer)
    check(reference)
    files = sorted((dest / "spectra").glob("*.npz"))
    if len(files) != 32 or result["evaluated_samples"] != 32:
        raise ValueError("Incomplete forecast/spectrum coverage")
    total = None
    layout = None
    expected = {x["initialization_time"].replace("-", "").replace(":", "") for x in read(manifest["development_manifest"]["path"])["initializations"]}
    actual = {p.stem for p in files}
    if actual != expected:
        raise ValueError("Spectrum dates differ from the development manifest")
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["checkpoint_sha256"] != reference["sha256"] or meta["campaign_hash"] != manifest["record_hash"]:
                raise ValueError("Spectrum provenance differs")
            if layout is not None and meta["layout"] != layout:
                raise ValueError("Spectrum channel layout differs")
            layout = meta["layout"]
            values = {key: np.asarray(data[key], dtype=np.float64) for key in ("truth_power", "prediction_power", "error_power", "cross_power")}
            if any(not np.isfinite(value).all() for value in values.values()):
                raise ValueError("Nonfinite spectrum")
            if total is None:
                total = {key: np.zeros_like(value) for key, value in values.items()}
            for key in total:
                total[key] += values[key] / len(files)
    pooled = dest / "pooled_spectra.npz"
    np.savez_compressed(pooled, **total, metadata=np.asarray(json.dumps({"layout": layout, "count": len(files)})))
    metrics = result["original_graphcast_loss"]
    if not all(math.isfinite(float(metrics[key])) for key in ("baseline_rollout", "full_rollout")):
        raise ValueError("Nonfinite exact GC loss")
    return seal(dest / "complete.json", {"status": "complete", "horizon": horizon, "point": point,
        "checkpoint": reference, "evaluation": pinned(dest / "evaluation.json"),
        "pooled_spectra": pinned(pooled), "spectra_count": len(files),
        "baseline_rollout": metrics["baseline_rollout"], "full_rollout": metrics["full_rollout"],
        "baseline_day10": metrics["baseline_per_step"][-1], "full_day10": metrics["full_per_step"][-1],
        "own_baseline_reduction_pct": 100 * (1 - metrics["full_rollout"] / metrics["baseline_rollout"]),
        "own_baseline_day10_reduction_pct": metrics["improvement_pct_per_step"][-1],
        "manifest_hash": manifest["record_hash"]})


def report(root, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records, arrays, layouts = {}, {}, {}
    for horizon in HORIZONS:
        for point in EARLY + LATE:
            name = f"step{point:05d}" if isinstance(point, int) else "swa"
            dest = root / "evaluation" / f"k{horizon:02d}" / name
            record = unseal(dest / "complete.json")
            if record["manifest_hash"] != manifest["record_hash"] or record["horizon"] != horizon or record["point"] != point:
                raise ValueError("Evaluation identity differs")
            check(record["evaluation"]); check(record["checkpoint"]); check(record["pooled_spectra"])
            with np.load(dest / "pooled_spectra.npz", allow_pickle=False) as data:
                arrays[(horizon, point)] = {key: np.asarray(data[key]) for key in ("truth_power", "prediction_power", "error_power", "cross_power")}
                layouts[(horizon, point)] = json.loads(str(data["metadata"]))["layout"]
            records[(horizon, point)] = record
    layout = layouts[(20, 500)]
    if any(value != layout for value in layouts.values()):
        raise ValueError("Spectrum channel layouts differ")
    differences = []
    reference = arrays[(20, 500)]
    for key, value in arrays.items():
        for field in ("truth_power", "prediction_power"):
            a = value[field] if field == "truth_power" else value[field][:, 0]
            b = reference[field] if field == "truth_power" else reference[field][:, 0]
            delta = float(np.max(np.abs(a - b)))
            scale = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
            differences.append({"point": str(key), "field": field, "max_abs_difference": delta,
                                "reference_max_abs": scale, "reference_relative": 1e-6,
                                "exceeds_reference": delta > 1e-6 * scale})
    channels = [tuple(x) for x in layout["channels"]]
    selected = [("2m_temperature", None), ("geopotential", 500), ("u_component_of_wind", 850)]
    indices = [channels.index(item) for item in selected]
    outputs = []
    for point in EARLY + LATE:
        name = f"step{point:05d}" if isinstance(point, int) else "swa"
        fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
        for row, (channel, label) in enumerate(zip(indices, selected)):
            for col, lead in enumerate((19, 39)):
                ax = axes[row, col]
                ref = arrays[(20, point)]
                degrees = np.arange(1, 180)
                ax.loglog(degrees, ref["truth_power"][lead, channel, 1:], color="black", label="ERA5")
                ax.loglog(degrees, ref["prediction_power"][lead, 0, channel, 1:], color="gray", linestyle="--", label="public GC")
                for horizon in HORIZONS:
                    ax.loglog(degrees, arrays[(horizon, point)]["prediction_power"][lead, 1, channel, 1:], label=f"K{horizon}")
                ax.set_title(f"{label[0]} {label[1] or ''} · day {(lead+1)/4:g}")
                ax.set_xlabel("Spherical degree"); ax.set_ylabel("Physical forecast power")
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        path = root / "report" / f"power_{name}.png"
        fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path))
    rows = [records[(k, p)] for k in HORIZONS for p in EARLY + LATE]
    canonical_loss = records[(20, 500)]["baseline_rollout"]
    canonical_day10 = records[(20, 500)]["baseline_day10"]
    baseline_spread = max(abs(row["baseline_rollout"] - canonical_loss) for row in rows)
    comparisons = [{"horizon": row["horizon"], "point": row["point"],
                    "rollout_reduction_pct": 100 * (1 - row["full_rollout"] / canonical_loss),
                    "day10_reduction_pct": 100 * (1 - row["full_day10"] / canonical_day10)}
                   for row in rows]
    loss_differences = [{"horizon": row["horizon"], "point": row["point"],
                         "baseline_minus_model_loss": row["baseline_rollout"] - row["full_rollout"],
                         "magnitude_exceeds_baseline_spread": abs(row["baseline_rollout"] - row["full_rollout"]) > baseline_spread}
                        for row in rows]
    return seal(root / "report" / "complete.json", {"status": "complete", "manifest_hash": manifest["record_hash"],
        "cohort": "32 development starts in 2022; 40 six-hour leads", "records": rows,
        "common_gc_baseline": {"source": "K20 update500 evaluation", "rollout_loss": canonical_loss,
                               "day10_loss": canonical_day10, "comparisons": comparisons},
        "figures": outputs, "pooled_spectra": "Each evaluation directory contains all 83 channels and 40 leads",
        "finite_reference_differences": differences, "reference_relative": 1e-6,
        "warnings": [item for item in differences if item["exceeds_reference"]],
        "baseline_repeatability": {"canonical_rollout_loss": canonical_loss,
            "max_abs_difference": baseline_spread, "loss_differences": loss_differences,
            "scope": "Observed repeat spread of pooled baseline losses only; not full model repeatability or a decision gate."},
        "interpretation": "Fixed-checkpoint development comparisons; no training-seed or 2023 generalization claim."})


def sbatch(root, manifest, stage, *, array=None, dependency=None, dependency_kind="afterok"):
    gpu = stage.startswith("train") or stage.startswith("evaluate")
    minutes = 1800 if stage.startswith("train") else 150 if stage.startswith("evaluate") else 40
    memory = 384 if stage.startswith("train") else 20 if stage.startswith("evaluate") else 16
    args = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={manifest['workspace']}",
            "--kill-on-invalid-dep=yes", f"--job-name=gc-ar-{stage}", f"--mem={memory}G",
            f"--time={minutes}", f"--cpus-per-task={8 if stage.startswith('train') else 4}",
            f"--output={root}/logs/{stage}_%A_%a.out", f"--error={root}/logs/{stage}_%A_%a.err"]
    if gpu:
        args += ["--gres=gpu:1", "--constraint=gpu80", "--qos=gpu-medium" if minutes > 480 else "--qos=gpu-short"]
    if array is not None:
        if "%" in array:
            raise ValueError("Array throttles are not authorized")
        args.append(f"--array={array}")
    if dependency:
        args.append(f"--dependency={dependency_kind}:{dependency}")
    return args + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(root)]


def submission_plan(root, manifest):
    return {"submitted": False, "nodes": [
        {"id": "train0", "command": sbatch(root, manifest, "train0", array="0-2")},
        {"id": "train1", "command": sbatch(root, manifest, "train1", array="0-2", dependency="<train0>", dependency_kind="aftercorr")},
        {"id": "evaluate-early", "command": sbatch(root, manifest, "evaluate-early", array="0-7", dependency="<train0>")},
        {"id": "swa", "command": sbatch(root, manifest, "swa", array="0-3", dependency="<train1>:<existing-k20-if-active>")},
        {"id": "evaluate-late", "command": sbatch(root, manifest, "evaluate-late", array="0-15", dependency="<swa>")},
        {"id": "report", "command": sbatch(root, manifest, "report", dependency="<evaluate-early>:<evaluate-late>")}],
        "k20_dependency": "Use existing K20 continuation if pending/running; if completed verify its final checkpoint first."}


def k20_job_state(job):
    # sacct can omit pending array elements; check the live queue first.
    queued = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"],
                            check=True, capture_output=True, text=True).stdout.strip().splitlines()
    if queued:
        return queued[0].strip().split()[0]
    history = subprocess.run(["sacct", "-n", "-X", "-j", job, "--format=State", "-P"],
                             check=True, capture_output=True, text=True).stdout.strip().splitlines()
    return history[0].split("|")[0].split()[0] if history else "UNKNOWN"


def submit(root, manifest):
    # Resolve the pre-existing K20 stage before creating any new jobs.
    state = k20_job_state(manifest["k20_continuation_job"])
    if state == "COMPLETED":
        pinned(checkpoint_path(root, manifest, 20, 2500))
        parent_dependency = None
    elif state in ("PENDING", "RUNNING", "CONFIGURING"):
        parent_dependency = manifest["k20_continuation_job"]
    else:
        raise RuntimeError(f"Existing K20 continuation is {state}; diagnose before submission")
    pinned(checkpoint_path(root, manifest, 20, 500)); pinned(checkpoint_path(root, manifest, 20, 1000))
    def job(name, command):
        value = submit_once(root, name, command)
        return str(value.get("job_id", value.get("jobid"))) if isinstance(value, dict) else str(value)
    first = job("train0", sbatch(root, manifest, "train0", array="0-2"))
    second = job("train1", sbatch(root, manifest, "train1", array="0-2", dependency=first, dependency_kind="aftercorr"))
    early = job("evaluate-early", sbatch(root, manifest, "evaluate-early", array="0-7", dependency=first))
    dependency = ":".join(x for x in (second, parent_dependency) if x)
    swa = job("swa", sbatch(root, manifest, "swa", array="0-3", dependency=dependency))
    late = job("evaluate-late", sbatch(root, manifest, "evaluate-late", array="0-15", dependency=swa))
    summary = job("report", sbatch(root, manifest, "report", dependency=f"{early}:{late}"))
    return {"train0": first, "train1": second, "evaluate-early": early, "swa": swa,
            "evaluate-late": late, "report": summary, "existing_k20_state": state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "submit", "train0", "train1", "swa", "evaluate-early", "evaluate-late", "report"))
    parser.add_argument("--experiment", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args(); root = args.experiment.resolve()
    if args.stage == "prepare":
        result = prepare(root, args.workspace)
    else:
        manifest = verify(root, frozen=args.stage != "submit")
        if args.stage == "submit": result = submit(root, manifest)
        elif args.stage in ("train0", "train1"):
            if args.task not in range(3): raise ValueError("Invalid train task")
            task = manifest["tasks"][str(HORIZONS[args.task])]
            result = scalar.run_training_stage(root, manifest, task, stop=1250 if args.stage == "train0" else 2500,
                                               previous=None if args.stage == "train0" else 1250)
        elif args.stage == "swa":
            if args.task not in range(4): raise ValueError("Invalid SWA task")
            result = build_swa(root, manifest, HORIZONS[args.task])
        elif args.stage.startswith("evaluate"):
            horizon, point = task_index(args.stage, args.task)
            result = evaluate(root, manifest, horizon, point)
        else: result = report(root, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
