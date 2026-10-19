"""Read-only numerical probes using an existing experiment's frozen model code.

Run each numerical mode in a fresh process on the same allocated GPU. Outputs
are diagnostic artifacts only; no checkpoint, experiment, or gate is modified.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(experiment):
    source = experiment / "source"
    sys.path[:0] = [str(source), str(source / "third_party/graphcast")]
    from scripts.experiments.frozen_runtime import verify_source
    snapshot = verify_source(source)
    manifest = read(experiment / "manifest.json")
    config_path = Path(manifest["entries"][0]["k1_preflight"])
    config = read(config_path)
    run = Path(config["output"]["output_root"]) / config["output"]["run_name"]
    checkpoint = run / "checkpoints/checkpoint_step00000002.pkl"
    sidecar = read(checkpoint.with_suffix(".cached_k1.json"))
    if sha(checkpoint) != sidecar["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    return manifest, config_path, config, checkpoint, sidecar, snapshot


def one_step(args, manifest, config_path, checkpoint_path, sidecar):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config
    from src.models.mamba.v24_Ilya.training.cached_data import open_cached_training_data
    from src.models.mamba.v24_Ilya.training.cached_k1_data import CachedK1Cursor, CachedK1Data
    from src.models.mamba.v24_Ilya.training.cached_k1_runner import load_k1_checkpoint
    from src.models.mamba.v24_Ilya.training.cached_k1_step import make_cached_k1_step
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import tree_comparison
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer

    config = load_training_config(config_path)
    producer = load_cached_training_config(Path(manifest["producer_config"]),
        cache_root=Path(manifest["cache_root"]),
        expected_manifest_sha256=sidecar["provenance"]["cache_manifest_sha256"])
    with jax.default_device(jax.devices("cpu")[0]):
        cached = open_cached_training_data(producer)
        data = CachedK1Data(cached, batch_size=config.distributed.global_batch_size,
                            seed=config.seed, shuffle=True)
        checkpoint, backend = load_k1_checkpoint(checkpoint_path, sidecar["provenance"], data=data)
        batch = data.build_batch(CachedK1Cursor.from_mapping(backend["cursor"]))
    _, _, task, stats, _, baseline_config = cached.context
    model = dataclasses.replace(baseline_config,
        latent_size=config.architecture.residual_width or config.architecture.width,
        gnn_msg_steps=config.architecture.residual_msg_steps)
    optimizer, _ = build_optimizer(config)
    trainer = make_cached_k1_step(config=config, model_config=model,
        task_config=task, stats=stats, optimizer=optimizer)
    params, state, rng = jax.device_put((checkpoint.residual_params,
        checkpoint.residual_state, checkpoint.rng_key), jax.devices()[0])
    key = jax.random.fold_in(rng, 0x4B31)
    arguments = (key, batch.inputs[0], batch.targets[0], batch.forcings[0], batch.valid_mask)
    expected = jax.device_get(trainer.evaluate_step(params, state, *arguments))
    records = []
    for index in range(5):
        actual = jax.device_get(trainer.evaluate_step(params, state, *arguments))
        record = {"repeat": index + 1, "components": {
            name: tree_comparison(a, b) for name, a, b in
            zip(("loss", "state", "prediction"), actual, expected, strict=True)}}
        if not all(np.isfinite(item.get("relative_l2", np.nan)) for item in record["components"].values()):
            raise ValueError("Invalid numerical comparison")
        records.append(record)
        write(args.output / args.mode / "one_step.json", {
            "checkpoint_step": checkpoint.completed_step, "batch_size": data.batch_size,
            "cursor": backend["cursor"], "repeats": records})
        print("ONE_STEP", json.dumps(record), flush=True)


class FieldObserver:
    """Compare selected physical-field leads; avoid combining unlike units."""
    leads = {0, 3, 7, 15, 23, 39}

    def __init__(self, directory, repeat):
        self.directory, self.repeat = Path(directory), repeat
        self.records = []
        self.directory.mkdir(parents=True, exist_ok=True)

    def begin_sample(self, metadata):
        self.initialization = metadata["initialization_time"]
        self.token = hashlib.sha256(self.initialization.encode()).hexdigest()[:16]

    def update_step(self, index, truth, baseline, full):
        if index not in self.leads:
            return
        import numpy as np
        from src.models.mamba.v24_Ilya.training.cached_diagnostics import tree_comparison
        values = {f"{branch}::{name}": np.asarray(dataset[name].values)
            for branch, dataset in (("truth", truth), ("baseline", baseline), ("full", full))
            for name in sorted(dataset.data_vars)}
        path = self.directory / f"{self.token}_lead{index+1:02d}.npz"
        if self.repeat == 0:
            np.savez(path, **values)
            return
        with np.load(path, allow_pickle=False) as reference:
            if set(reference.files) != set(values):
                raise ValueError("Field names changed between repetitions")
            comparisons = {}
            for name, value in values.items():
                if name.startswith("truth::"):
                    if not np.array_equal(value, reference[name]):
                        raise ValueError("Truth changed between repetitions")
                else:
                    comparisons[name] = tree_comparison(value, reference[name])
        self.records.append({"initialization_time": self.initialization,
            "lead_hours": 6 * (index + 1), "fields": comparisons})
        write(self.directory.parent / f"field_comparison_{self.repeat}.json", self.records)

    def finish_sample(self):
        pass


def rollout(args, manifest, raw_config, checkpoint):
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    dates = args.output / "initializations.json"
    if not dates.exists():
        write(dates, {"schema_version": 1, "initializations": [
            {"initialization_time": f"2022-{month:02d}-15T00:00:00Z"} for month in (1, 7)]})
    for repeat in range(3):
        out = args.output / args.mode / f"rollout_{repeat}.json"
        config = V24IlyaEvalConfig(ckpt=checkpoint, out_json=out,
            eval_mode="cold_full", warmup_steps=0, residual_state_init="zero",
            ckpt_in=Path(raw_config["data"]["baseline_checkpoint"]),
            stats_dir=Path(raw_config["data"]["stats_dir"]), data_path=manifest["eval_data"],
            val_year=2022, train_start_year=2015, train_end_year=2021,
            target_steps=40, n_samples=2, seed=22, omit_rms_bias=True,
            initialization_manifest=dates, model_id="k1_step2_diagnostic",
            **raw_config["architecture"])
        observer = FieldObserver(args.output / args.mode / "reference_fields", repeat)
        result = evaluate_v24_Ilya(config, prediction_observer=observer)
        write(out, result)
        print("ROLLOUT_COMPLETE", args.mode, repeat, flush=True)


def compare_metrics(reference, candidate):
    import numpy as np
    left, right = reference["original_graphcast_loss"], candidate["original_graphcast_loss"]
    result = {}
    for branch in ("baseline", "full"):
        a = np.asarray(left[f"{branch}_per_step"], dtype=float)
        b = np.asarray(right[f"{branch}_per_step"], dtype=float)
        if a.shape != (40,) or b.shape != (40,) or not np.isfinite([a, b]).all() or np.any(a <= 0):
            raise ValueError("Invalid exact GraphCast losses")
        result[branch] = {"relative_change_pct_per_lead": (100 * (b / a - 1)).tolist(),
            "rollout_relative_change_pct": float(100 * (right[f"{branch}_rollout"] / left[f"{branch}_rollout"] - 1))}
    old = 100 * (1 - left["full_rollout"] / left["baseline_rollout"])
    new = 100 * (1 - right["full_rollout"] / right["baseline_rollout"])
    result["improvement_change_percentage_points"] = new - old
    result["reference_improvement_pct"] = old
    result["candidate_improvement_pct"] = new
    return result


def summarize(output):
    report = {"scope": "Step-2 checkpoint; two 2022 dates; three repeated 40-lead cold full-feedback forecasts per mode. Not a trained-model qualification.",
              "modes": {}, "cross_mode": None}
    for mode in ("ordinary", "deterministic"):
        directory = output / mode
        item = {"one_step": read(directory / "one_step.json") if (directory / "one_step.json").exists() else None,
                "rollout_comparisons": []}
        paths = [directory / f"rollout_{i}.json" for i in range(3)]
        item["completed_rollouts"] = sum(p.exists() for p in paths)
        if paths[0].exists():
            for path in paths[1:]:
                if path.exists():
                    item["rollout_comparisons"].append(compare_metrics(read(paths[0]), read(path)))
        report["modes"][mode] = item
    ordinary, deterministic = (output / mode / "rollout_0.json" for mode in ("ordinary", "deterministic"))
    if ordinary.exists() and deterministic.exists():
        report["cross_mode"] = compare_metrics(read(ordinary), read(deterministic))
    write(output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "one-step", "rollout", "summarize"))
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("ordinary", "deterministic"), default="ordinary")
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    args.output = args.output.resolve()
    if args.phase == "summarize":
        summarize(args.output)
        return
    manifest, config_path, raw, checkpoint, sidecar, snapshot = configure(args.experiment)
    if args.phase == "validate":
        from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
        V24IlyaEvalConfig(ckpt=checkpoint, out_json=args.output / "validation.json",
                         eval_mode="cold_full", **raw["architecture"])
        print("Validated frozen source and checkpoint", snapshot["source_digest"], sha(checkpoint))
        return
    import jax
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Diagnostic requires one allocated GPU")
    deterministic = "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "").split()
    if deterministic != (args.mode == "deterministic"):
        raise ValueError("Runtime flags differ from the requested mode")
    started = time.monotonic()
    metadata = {"source_digest": snapshot["source_digest"], "checkpoint_sha256": sha(checkpoint),
                "diagnostic_sha256": sha(__file__), "jax": jax.__version__,
                "device": jax.devices()[0].device_kind, "xla_flags": os.environ.get("XLA_FLAGS"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "phase": args.phase}
    write(args.output / args.mode / f"{args.phase}_runtime.json", metadata)
    if args.phase == "one-step":
        one_step(args, manifest, config_path, checkpoint, sidecar)
    else:
        rollout(args, manifest, raw, checkpoint)
    metadata.update(elapsed_seconds=time.monotonic() - started,
                    peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    write(args.output / args.mode / f"{args.phase}_runtime.json", metadata)


if __name__ == "__main__":
    main()
