"""Prepare and execute the matched fresh-spatial width ablation."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/graphcast"))
REFERENCE = Path("artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904/"
                 "mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k")
REF_EVAL = Path("eval/cold_full_zero_exact_gc/matched_v22_reference32/swa_step02000-08000.json")
SWA_STEPS = list(range(2000, 8001, 1000))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(args):
    print("EXEC " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def prepare(experiment):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    if experiment.exists():
        raise FileExistsError(f"Refusing to replace {experiment}")
    saved = read(ROOT / REFERENCE / "run_config.json")
    reference_eval = ROOT / REFERENCE / REF_EVAL
    reference = read(reference_eval)
    assert reference["evaluation_status"] == "complete" and reference["evaluated_samples"] == 32
    experiment.mkdir(parents=True)
    write(experiment / "reference_evaluation.json", reference)
    write(experiment / "reference_run_config.json", saved)
    entries = []
    for width in (512, 128):
        config = {k: v for k, v in saved.items() if k not in ("derived", "config_source")}
        config = json.loads(json.dumps(config))
        config["architecture"].update(residual_width=width, residual_initialization="fresh",
                                      temporal_dt_rank="32")
        for key in ("prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir"):
            config["data"][key] = str(ROOT / config["data"][key])
        name = f"w{width}_fresh_mamba1_di16_bcg1_rank32_seed22"
        config["output"] = dict(output_root=str(experiment / "runs"), run_name=name)
        path = experiment / "configs" / f"{name}.json"
        write(path, config)
        load_training_config(path)
        smoke = json.loads(json.dumps(config))
        smoke["optimizer"].update(max_steps=3, checkpoint_every=1, warmup_steps=0)
        smoke["output"]["output_root"] = str(experiment / "smoke")
        smoke_path = experiment / "configs" / f"{name}_smoke.json"
        write(smoke_path, smoke)
        load_training_config(smoke_path)
        entries.append(dict(name=name, width=width, config=str(path), smoke_config=str(smoke_path)))
    eval_data = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr")
    if not eval_data.exists():
        raise FileNotFoundError(eval_data)
    manifest = dict(workspace=str(ROOT), experiment=str(experiment), entries=entries,
                    reference_run=str(ROOT / REFERENCE), reference_eval=str(reference_eval),
                    eval_data=str(eval_data), source_revision=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    primary_swa_steps=SWA_STEPS,
                    reference_improvement_pct=reference["original_graphcast_loss"]["improvement_pct_rollout"])
    write(experiment / "manifest.json", manifest)
    (experiment / "logs").mkdir()
    (experiment / "checks").mkdir()
    (experiment / "workspace.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    (experiment / "workspace_status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    snapshot = experiment / "source"
    for directory in ("src", "third_party", "scripts"):
        shutil.copytree(ROOT / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    write(experiment / "source_hashes.json", {
        str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()
    })
    write(experiment / "config_hashes.json", {
        str(p.relative_to(experiment)): sha(p) for p in sorted((experiment / "configs").glob("*.json"))
    })
    print(json.dumps(manifest, indent=2))


def verify(experiment):
    for relative, expected in read(experiment / "source_hashes.json").items():
        if sha(experiment / "source" / relative) != expected:
            raise RuntimeError(f"Snapshot changed: {relative}")
    for relative, expected in read(experiment / "config_hashes.json").items():
        if sha(experiment / relative) != expected:
            raise RuntimeError(f"Config changed: {relative}")
    if ROOT != experiment / "source":
        raise RuntimeError("Execute GPU stages from the saved source snapshot")


def eval_command(experiment, entry, checkpoint, output, *, smoke=False):
    config = read(entry["config"])
    data = config["data"]
    manifest = read(experiment / "manifest.json")
    return [sys.executable, "-u", ROOT / "scripts/analyze_models/eval_v24_Ilya.py",
            "--ckpt", checkpoint, "--ckpt-in", data["baseline_checkpoint"],
            "--data-path", manifest["eval_data"], "--stats-dir", data["stats_dir"],
            "--resolution", "1", "--mesh-size", "5", "--width", "512",
            "--residual-width", entry["width"], "--residual-initialization", "fresh",
            "--baseline-msg-steps", "16", "--residual-msg-steps", "2",
            "--val-year", "2022", "--train-start-year", "2015", "--train-end-year", "2021",
            "--input-duration", "12h", "--temporal-location", "mesh_processor_interleaved",
            "--temporal-d-inner", "16", "--temporal-d-state", "16", "--temporal-d-conv", "4",
            "--temporal-dt-rank", "32", "--temporal-init-scheme", "mamba1",
            "--temporal-layers", "2", "--temporal-bc-groups", "1", "--temporal-stateful",
            "--eval-mode", "cold_full", "--residual-state-init", "zero", "--residual-alpha", "1",
            "--target-steps", "2" if smoke else "40", "--warmup-steps", "24",
            "--n-samples", "1" if smoke else "32", "--seed", "0", "--out-json", output]


def smoke_train(experiment, entry):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.training import runner
    from src.models.mamba.v24_Ilya.training.config import parse_cli
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.endpoint_step import is_mamba_parameter
    if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
        raise RuntimeError("Preflight requires one allocated GPU")
    initial_hashes = {}
    counts = collections.Counter()
    baseline = {}
    original_overlay = runner.overlay_matching_params
    original_baseline = runner.overlay_frozen_baseline_params

    def digest(value):
        return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()

    def checked_overlay(initial, source, **kwargs):
        assert not source, "Fresh residual unexpectedly received pretrained parameters"
        assert all(np.all(np.asarray(v) == 0) for v in initial["temporal_residual_head"].values())
        spatial_projections = [v["w"] for m, v in initial.items()
                               if "encoder_nodes_grid_nodes_mlp" in m and m.endswith("linear_0")]
        assert spatial_projections and all(v.shape[-1] == entry["width"] for v in spatial_projections)
        dt_projections = [v["w"] for m, v in initial.items() if m.endswith("dt_proj")]
        assert dt_projections and all(tuple(v.shape) == (32, 16) for v in dt_projections)
        for module, values in initial.items():
            for name, value in values.items():
                initial_hashes[(module, name)] = digest(value)
                group = "mamba" if is_mamba_parameter(module) else module.split("/")[0]
                counts[group] += value.size
        loaded, stats = original_overlay(initial, source, **kwargs)
        assert stats.copied == 0
        return loaded, stats

    def checked_baseline(initial, source):
        loaded, stats = original_baseline(initial, source)
        assert sum(v.size for m in loaded.values() for v in m.values()) == 35979347
        assert stats.copied == 262 and not stats.ignored_source
        for module, values in loaded.items():
            for name, value in values.items():
                assert digest(value) == digest(source[module][name])
        baseline.update(params=loaded, initial_hashes={
            (m, n): digest(v) for m, vs in loaded.items() for n, v in vs.items()})
        return loaded, stats

    runner.overlay_matching_params = checked_overlay
    runner.overlay_frozen_baseline_params = checked_baseline
    invocation = parse_cli(["--config", entry["smoke_config"]])
    directory = invocation.config.run_dir
    if directory.exists():
        raise FileExistsError(directory)
    runner.run_training(invocation)
    checkpoint = load_v24_Ilya_training_checkpoint(directory / "checkpoints/checkpoint_step00000003.pkl")
    changed = collections.Counter()
    for module, values in checkpoint.residual_params.items():
        for name, value in values.items():
            assert np.all(np.isfinite(value))
            if digest(value) != initial_hashes[(module, name)]:
                changed["mamba" if is_mamba_parameter(module) else "spatial"] += 1
    assert changed["spatial"] > 0 and changed["mamba"] > 0
    for module, values in baseline["params"].items():
        for name, value in values.items():
            assert digest(value) == baseline["initial_hashes"][(module, name)]
    metadata = read(directory / "run_config.json")
    reference = read(experiment / "reference_run_config.json")
    for key in ("baseline_checkpoint_fingerprint", "anchor_manifest_fingerprint"):
        assert metadata["derived"][key] == reference["derived"][key], key
    records = [json.loads(line) for line in (directory / "train_metrics.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert all(math.isfinite(r[k]) for r in records for k in ("loss", "gradient_norm"))
    write(experiment / "checks" / f"{entry['name']}_training.json", dict(
        passed=True, baseline_unchanged=True, baseline_parameters=35979347,
        residual_copied=0, parameter_counts=dict(counts), changed_tensors=dict(changed),
        step_seconds=[r["step_seconds"] for r in records],
        device_memory=[r.get("device_memory") for r in records]))


def train(experiment, entry):
    name = entry["name"]
    directory = experiment / "runs" / name
    if directory.exists():
        raise FileExistsError(directory)
    # Separate processes release compilation and device allocations before training.
    command([sys.executable, "-u", __file__, "smoke", "--experiment", experiment,
             "--task", read(experiment / "manifest.json")["entries"].index(entry)])
    checkpoint = experiment / "smoke" / name / "checkpoints/checkpoint_step00000003.pkl"
    smoke_eval = experiment / "checks" / f"{name}_roundtrip.json"
    command(eval_command(experiment, entry, checkpoint, smoke_eval, smoke=True))
    result = read(smoke_eval)
    assert result["evaluation_status"] == "complete" and result["evaluated_samples"] == 1
    assert math.isfinite(result["original_graphcast_loss"]["full_rollout"])
    write(experiment / "checks" / f"{name}_preflight.json", dict(passed=True, checkpoint_roundtrip=True))
    started = time.monotonic()
    command([sys.executable, "-u", ROOT / "scripts/training/train_v24_Ilya.py", "--config", entry["config"]])
    seconds = time.monotonic() - started
    swa = directory / "swa/swa_step02000-08000.pkl"
    command([sys.executable, "-u", ROOT / "scripts/training/build_v24_Ilya_swa.py", "--inputs",
             *[directory / f"checkpoints/checkpoint_step{s:08d}.pkl" for s in SWA_STEPS],
             "--source-steps", *SWA_STEPS, "--output", swa])
    records = [json.loads(line) for line in (directory / "train_metrics.jsonl").read_text().splitlines()]
    assert len(records) == 10000 and records[-1]["step"] == 10000
    assert all(math.isfinite(r[k]) for r in records for k in ("loss", "gradient_norm"))
    write(experiment / "checks" / f"{name}_completed.json", dict(
        passed=True, training_seconds=seconds,
        median_step_seconds_101_1000=statistics.median(r["step_seconds"] for r in records[100:1000])))


def evaluate(experiment, entry):
    import numpy as np
    directory = experiment / "runs" / entry["name"]
    reference = read(experiment / "reference_evaluation.json")
    for tag, checkpoint in (("swa_step02000-08000", directory / "swa/swa_step02000-08000.pkl"),
                            ("step10000", directory / "checkpoints/checkpoint_step00010000.pkl")):
        output = directory / "eval/cold_full_zero_exact_gc/matched32" / f"{tag}.json"
        if output.exists():
            raise FileExistsError(output)
        command(eval_command(experiment, entry, checkpoint, output))
        result = read(output)
        for key in ("chosen_idx", "evaluated_samples", "evaluation_status", "target_steps",
                    "anchor_history_steps", "anchor_index_semantics", "baseline_branch",
                    "residual_state_init_resolved", "eval_mode", "sample_total_steps"):
            assert result[key] == reference[key], key
        np.testing.assert_allclose(result["original_graphcast_loss"]["baseline_per_step"],
                                   reference["original_graphcast_loss"]["baseline_per_step"], rtol=1e-5, atol=1e-7)
    write(experiment / "checks" / f"{entry['name']}_evaluation.json", dict(passed=True))


def summarize(experiment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = read(experiment / "manifest.json")
    reference = read(experiment / "reference_evaluation.json")
    active_entries = [manifest["entries"][i] for i in
                      manifest.get("active_tasks", range(len(manifest["entries"])))]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, evaluation, params, timing in [
        ("w512_pretrained_reference", reference, 10348879, None),
        *[(e["name"], read(experiment / "runs" / e["name"] / "eval/cold_full_zero_exact_gc/matched32/swa_step02000-08000.json"),
           sum(read(experiment / "checks" / f"{e['name']}_training.json")["parameter_counts"].values()),
           read(experiment / "checks" / f"{e['name']}_completed.json")["median_step_seconds_101_1000"])
          for e in active_entries]]:
        metric = evaluation["original_graphcast_loss"]
        improvement = metric["improvement_pct_rollout"]
        rows.append(dict(arm=name, residual_parameters=params, improvement_pct=improvement,
                         delta_vs_reference_pp=improvement-manifest["reference_improvement_pct"],
                         retained_reference_improvement_pct=100*improvement/manifest["reference_improvement_pct"],
                         median_step_seconds_101_1000=timing))
        ax.plot([(i+1)/4 for i in range(40)], metric["improvement_pct_per_step"], label=name)
    ax.set(xlabel="Forecast lead (days)", ylabel="Exact GraphCast loss reduction (%)")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output = experiment / "results"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "width_ablation_loss.png", dpi=180)
    comparison = dict(rows=rows, comparison="Active candidates versus saved pretrained reference")
    if len(rows) == 3:
        comparison["width_effect_pp"] = rows[2]["improvement_pct"]-rows[1]["improvement_pct"]
    else:
        comparison["interpretation"] = (
            "Direct comparison with the existing best Mamba1 run; spatial width "
            "and pretrained-versus-fresh spatial initialization both differ."
        )
    write(output / "summary.json", comparison)
    gpu_peaks = {}
    for path in (experiment / "logs").glob("train_*_gpu.csv"):
        with path.open() as handle:
            values = [float(row[3].strip()) for row in csv.reader(handle) if len(row) >= 4]
        gpu_peaks[path.name] = max(values) if values else None
    write(output / "sampled_gpu_peaks_mib.json", gpu_peaks)
    jobs = read(experiment / "jobs.json")
    accounting = subprocess.run(["sacct", "-j", str(jobs["training_array"]),
        "--format=JobID,State,Elapsed,ReqMem,MaxRSS", "-P"], capture_output=True, text=True)
    (output / "training_accounting.txt").write_text(accounting.stdout + accounting.stderr)
    with (output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify", "smoke", "train", "eval", "summarize"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage == "prepare":
        prepare(experiment)
        return
    verify(experiment)
    if args.stage == "verify":
        return
    if args.stage == "summarize":
        summarize(experiment)
        return
    entry = read(experiment / "manifest.json")["entries"][args.task]
    {"smoke": smoke_train, "train": train, "eval": evaluate}[args.stage](experiment, entry)


if __name__ == "__main__":
    main()
