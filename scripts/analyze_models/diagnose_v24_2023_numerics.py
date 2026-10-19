#!/usr/bin/env python3
"""Run frozen 2022-only numerical probes inside an existing GPU allocation.

Invoke this file from the campaign's frozen source with its PYTHONPATH and
XLA_FLAGS. This utility does not submit jobs or start evaluation subprocesses.
Independent invocations provide fresh-process and matching-GPU evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_v24_2023_eval as campaign
from scripts.experiments.v24_2023_qualification import (
    NumericalObserver, compare_losses, dataset_fingerprint, file_record,
)


def write_new_json(path, value):
    """Publish atomically without ever replacing another probe's evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".probe-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--model-id", default="M1")
    parser.add_argument("--label", required=True, help="Unique identifier for this process invocation")
    parser.add_argument("--mode", choices=("interleaved", "standalone", "bridge"), default="interleaved")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.label):
        parser.error("--label must start with an alphanumeric and contain at most 80 filename-safe characters")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.mode != "interleaved" and args.model_id != "M1":
        parser.error("standalone and bridge probes support M1 only")
    return args


def gpu_identity(protocol):
    import jax
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("Numerical probes require exactly one visible GPU")
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = [tuple(part.strip() for part in line.split(",")) for line in text.splitlines() if line.strip()]
    if any(len(row) != 5 for row in rows):
        raise ValueError("Cannot identify GPU from nvidia-smi")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if len(rows) != 1 and visible:
        selectors = visible.split(",")
        if len(selectors) != 1:
            raise ValueError("Cannot unambiguously map the visible GPU UUID")
        rows = [row for row in rows if row[0] == selectors[0] or row[1].startswith(selectors[0])]
    if len(rows) != 1:
        raise ValueError("Cannot unambiguously identify the visible GPU UUID")
    _, gpu_uuid, name, driver, memory = rows[0]
    memory = float(memory)
    if not 79000 <= memory <= 85000:
        raise ValueError("Numerical probes require an 80 GB GPU")
    kind = devices[0].device_kind
    expected = protocol["resources"]["gpu"].get("device_kind")
    if expected is not None and kind != expected:
        raise ValueError("GPU kind differs from the frozen numerical runtime")
    return {"gpu_uuid": gpu_uuid, "gpu_kind": kind, "gpu_name": name,
            "driver_version": driver, "gpu_memory_mib": memory,
            "devices": [{"platform": "gpu", "kind": kind}]}


def validate_probe_protocol(protocol, model_id):
    if protocol.get("schema_version") != 2 or protocol.get("target_steps") != 40:
        raise ValueError("Numerical probes require a frozen schema-2, 40-lead campaign")
    models = [m for m in protocol["models"] if m["model_id"] == model_id]
    if len(models) != 1:
        raise ValueError(f"Unknown or duplicate model ID {model_id!r}")
    expected = [f"2022-{month:02d}-15T00:00:00Z" for month in (1, 4, 7, 10)]
    if protocol["pilot"]["initialization_times"] != expected:
        raise ValueError("Numerical probes require the four frozen 2022 pilot dates")
    manifest = campaign.read_json(protocol["pilot"]["manifest"]["path"])
    actual = [r["initialization_time"] if isinstance(r, dict) else r for r in manifest["initializations"]]
    if actual != expected:
        raise ValueError("Pilot manifest differs from frozen 2022 dates")
    forecast = protocol["forecast"]
    if (forecast.get("warmup_steps") != 0 or forecast.get("eval_mode") != "cold_full"
            or forecast.get("residual_state_init") != "zero"):
        raise ValueError("Numerical probes require zero-state cold full-feedback forecasts")
    return models[0]


def make_config(protocol, model, output, manifest):
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    count = len(campaign.read_json(manifest)["initializations"])
    return V24IlyaEvalConfig(
        ckpt=Path(model["checkpoint"]), out_json=Path(output),
        ckpt_in=Path(protocol["baseline"]["path"]), stats_dir=Path(protocol["stats_dir"]),
        data_path=protocol["pilot"]["data_path"], val_year=2022,
        train_start_year=2015, train_end_year=2021, target_steps=40, n_samples=count,
        omit_rms_bias=True, initialization_manifest=Path(manifest), model_id=model["model_id"],
        anchor_shard_count=1, anchor_shard_index=0,
        reset_state_every_step=model.get("reset_state_every_step", False),
        **protocol["forecast"], **model["architecture"],
    )


def bridge_expected_times(protocol, archived):
    """Use the qualification gate's synchronous historical time-coordinate map."""
    from scripts.experiments.v24_2023_qualification import development_initialization_times
    return development_initialization_times(protocol, archived)


def bridge_manifest(protocol, model, config, destination):
    """Recover historical indices in the historical validation-year slice."""

    source = Path(model["run_config"]["path"]).parent / "eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json"
    source_record = file_record(source)
    archived = campaign.read_json(source)
    for field, expected in (("target_steps", 40), ("evaluated_samples", 32), ("evaluation_status", "complete"),
                            ("eval_mode", "cold_full"), ("warmup_steps", 0), ("residual_state_init", "zero"),
                            ("rs_reset_every_step", False), ("seed", config.seed), ("eval_feedback", "full")):
        if archived.get(field) != expected:
            raise ValueError(f"Archived development evaluation has incompatible {field}")
    for field, expected_path, expected_hash in (
        ("ckpt", model["checkpoint"], model["checkpoint_sha256"]),
        ("ckpt_in", protocol["baseline"]["path"], protocol["baseline"]["checkpoint_sha256"]),
    ):
        path = Path(archived[field])
        if not path.is_absolute():
            path = Path(protocol["project_root"]) / path
        if path.resolve() != Path(expected_path).resolve() or campaign.sha256(path) != expected_hash:
            raise ValueError("Archived development checkpoint identity differs")
    starts = bridge_expected_times(protocol, archived)
    write_new_json(destination, {"schema_version": 1, "initializations": [{"initialization_time": t} for t in starts]})
    return source_record, archived, starts


def compare_development(archived, rerun):
    """Describe aggregate drift, without pretending old per-start data exist."""
    result = {"criterion": "descriptive_only_no_historical_repeatability_tolerance", "branches": {}}
    for branch in ("baseline", "full"):
        old = np.asarray(archived["original_graphcast_loss"][f"{branch}_per_step"], dtype=float)
        new = np.asarray(rerun["original_graphcast_loss"][f"{branch}_per_step"], dtype=float)
        if old.shape != (40,) or new.shape != (40,) or not np.isfinite([old, new]).all() or np.any(old < 0) or np.any(new < 0):
            raise ValueError("Invalid development bridge losses")
        result["branches"][branch] = {"archived_per_step": old.tolist(), "rerun_per_step": new.tolist(),
            "difference_per_step": (new-old).tolist(), "archived_rollout": float(old.mean()),
            "rerun_rollout": float(new.mean()), "rollout_difference": float(new.mean()-old.mean())}
    effects = {}
    for label in ("archived", "rerun"):
        baseline = result["branches"]["baseline"][f"{label}_rollout"]
        if baseline <= 0:
            raise ValueError("Development bridge baseline loss must be positive")
        effects[label] = 100 * (1-result["branches"]["full"][f"{label}_rollout"]/baseline)
    result.update(improvement_pct=effects, improvement_difference_pp=effects["rerun"]-effects["archived"])
    return result


def record_forcing(observer, record, step, forcing):
    observer.data.setdefault("forcings", []).append({"initialization_time": record["initialization_time"],
        "step_index": step, **dataset_fingerprint(forcing)})


@contextmanager
def capture_interleaved_forcings(observer, family):
    """Observe the actual forcing tensors, restoring module hooks afterward."""
    from src.models.mamba.v24_Ilya import evaluation
    from src.models.mamba.v24_Ilya.rollout import physical_fields_fp32
    if family == "v22_final":
        from scripts.analyze_models import eval_v22_2023_manifest as module
        name = "run_v22_final_rollout"
        original = getattr(module, name)

        def wrapped(**kwargs):
            record = observer.data["initializations"][-1]
            for step in range(kwargs["target_steps"]):
                record_forcing(observer, record, step, kwargs["all_forcings"].isel(time=slice(step, step+1)))
            return original(**kwargs)
    else:
        module, name = evaluation, "iter_eval_steps"
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            for step, (truth, forcing) in enumerate(original(*args, **kwargs)):
                record_forcing(observer, observer.data["initializations"][-1], step, physical_fields_fp32(forcing))
                yield truth, forcing
    setattr(module, name, wrapped)
    try:
        yield
    finally:
        setattr(module, name, original)


def standalone_baseline(config, observer):
    """Pure frozen V24 baseline: no residual initialization or forward calls."""
    import jax
    from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats
    from src.models.mamba.v24_Ilya.checkpoint import overlay_frozen_baseline_params
    from src.models.mamba.v24_Ilya.data import build_eval_batch, iter_eval_steps, open_eval_dataset, resolve_initialization_manifest
    from src.models.mamba.v24_Ilya.evaluation import (
        _dataset_to_host, _release_host_memory, build_initialization_loss_record,
        collect_initialization_variable_mse, initialization_rng_key,
    )
    from src.models.mamba.v24_Ilya.metrics import V24IlyaMetricAccumulator
    from src.models.mamba.v24_Ilya.model import build_predictors
    from src.models.mamba.v24_Ilya.rollout import physical_fields_fp32, shift_inputs_with_field, state_boundary_fp32

    checkpoint = load_graphcast_checkpoint(config.ckpt_in)
    task = dataclasses.replace(checkpoint.task_config, input_duration=config.input_duration)
    stats = load_stats(config.stats_dir)
    data = open_eval_dataset(config, task)
    records = resolve_initialization_manifest(data, config.initialization_manifest, target_steps=40)
    predictors = build_predictors(checkpoint.model_config, task, stats, config)
    sample = tuple(physical_fields_fp32(x) for x in build_eval_batch(data, indices=[records[0]["dataset_index"]], target_steps=1, task_config=task))
    _, key, _ = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    params, state_init = predictors.baseline.init(key, *sample)
    params, _ = overlay_frozen_baseline_params(params, checkpoint.params)
    state_init = state_boundary_fp32(state_init)
    observer.setup(config=config, baseline_model_config=checkpoint.model_config, task_config=task,
                   baseline_params=params, residual_params={}, baseline_state=state_init, residual_state={}, stats=stats)
    del sample

    @jax.jit
    def baseline_step(params, state, key, inputs, targets, forcings):
        return predictors.baseline.apply(params, state, key, inputs, targets, forcings)

    metrics = V24IlyaMetricAccumulator(40, data.dataset.lat.values, diffs_stddev_by_level=stats["diffs_stddev_by_level"], store_spatial_bias=False)
    loss_records, runtimes = [], []
    for record in records:
        started = time.monotonic()
        inputs, _, _ = build_eval_batch(data, indices=[record["dataset_index"]], target_steps=1, task_config=task)
        rng = initialization_rng_key(config.seed, record["initialization_time"], stream="rollout")
        observer.initialization(record, inputs, rng)
        inputs, state = physical_fields_fp32(inputs), state_init
        losses, variable_mse = {"baseline": [], "full": []}, {}
        steps = iter_eval_steps(data, final_input_idx=record["dataset_index"], total_steps=40,
                               block_steps=config.stream_block_steps, task_config=task)
        count = 0
        for step, (truth, forcing) in enumerate(steps):
            truth, forcing = physical_fields_fp32(truth), physical_fields_fp32(forcing)
            record_forcing(observer, record, step, forcing)
            rng, key, _ = jax.random.split(rng, 3)
            prediction, state = baseline_step(params, state, key, inputs, truth, forcing)
            prediction, state = physical_fields_fp32(prediction), state_boundary_fp32(state)
            truth_host, prediction_host = _dataset_to_host(truth), _dataset_to_host(prediction)
            exact = metrics.update_step(step, truth_host, prediction_host, prediction_host)
            for branch in losses:
                if np.asarray(exact[branch]).size != 1:
                    raise ValueError("Standalone probe requires one initialization per batch")
                losses[branch].append(float(np.asarray(exact[branch]).item()))
            collect_initialization_variable_mse(variable_mse, step_index=step, target_steps=40,
                                                contributions=metrics.last_step_variable_mse)
            observer.prediction(record, step, truth_host, prediction_host, prediction_host)
            if step < 39:
                inputs = shift_inputs_with_field(inputs, prediction, forcing, time_step=data.time_step, input_steps=data.input_steps)
            del truth_host, prediction_host
            _release_host_memory()
            count += 1
        if count != 40:
            raise ValueError("Incomplete standalone probe trajectory")
        loss_records.append(build_initialization_loss_record(record, losses, variable_mse=variable_mse))
        runtimes.append(time.monotonic()-started)
    data.dataset.close()
    return {"evaluation_status": "complete", "metrics_complete": True, "evaluated_samples": len(records), "target_steps": 40,
            "rs_reset_every_step": config.reset_state_every_step,
            "chosen_idx": [r["dataset_index"] for r in records], "sample_elapsed_seconds": runtimes,
            "original_graphcast_loss_per_initialization": loss_records,
            "variable_mse_metadata": metrics.variable_mse_metadata, **metrics.finalize()}


def run(args):
    workflow = args.workflow.resolve()
    protocol = campaign.verify_protocol(workflow)
    model = validate_probe_protocol(protocol, args.model_id)
    if args.mode != "interleaved" and args.model_id != "M1":
        raise ValueError("Standalone and bridge probes support M1 only")
    campaign.validate_runtime(protocol)
    expected_script = Path(protocol["source_root"]) / "scripts/analyze_models/diagnose_v24_2023_numerics.py"
    if Path(__file__).resolve() != expected_script.resolve():
        raise RuntimeError("Run the probe utility from the campaign's frozen source directory")
    for record in (model["run_config"], {"path": model["checkpoint"], "sha256": model["checkpoint_sha256"]},
                   {"path": protocol["baseline"]["path"], "sha256": protocol["baseline"]["checkpoint_sha256"]}):
        campaign.check_file(record)
    gpu = gpu_identity(protocol)
    directory = workflow / "numerics"
    prefix = f"{args.label}_{args.model_id}_{args.mode}"
    outputs = [directory / f"{prefix}_{repeat:02d}.probe.json" for repeat in range(args.repeats)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Numerical probe outputs already exist; use a new label")
    reservation = directory / f".{prefix}"
    reservation.mkdir(parents=True, exist_ok=False)
    manifest = Path(protocol["pilot"]["manifest"]["path"])
    config = make_config(protocol, model, reservation / "pending.json", manifest)
    archived, source = None, None
    expected_times = protocol["pilot"]["initialization_times"]
    if args.mode == "bridge":
        manifest = reservation / "development_initializations.json"
        source, archived, expected_times = bridge_manifest(protocol, model, config, manifest)
    process_run_id, previous = str(uuid.uuid4()), None
    for repeat, output in enumerate(outputs):
        config = make_config(protocol, model, reservation / f"repeat_{repeat:02d}.unpublished.json", manifest)
        observer = NumericalObserver()
        started = time.monotonic()
        if args.mode == "standalone":
            result = standalone_baseline(config, observer)
        else:
            family = model.get("evaluator_family", "v24_Ilya")
            with capture_interleaved_forcings(observer, family):
                if family == "v22_final":
                    from scripts.analyze_models.eval_v22_2023_manifest import evaluate_legacy_manifest
                    result = evaluate_legacy_manifest(config, model["architecture"], numerical_observer=observer)
                else:
                    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
                    result = evaluate_v24_Ilya(config, numerical_observer=observer)
        if result.get("evaluation_status") != "complete" or [r["initialization_time"] for r in result["original_graphcast_loss_per_initialization"]] != expected_times:
            raise ValueError("Numerical probe returned incomplete or unexpected dates")
        compare_losses(result, result)  # Validate every exported exact-loss lead.
        result.update(protocol_hash=protocol["protocol_hash"], model_id=model["model_id"],
            checkpoint_sha256=model["checkpoint_sha256"], baseline_checkpoint_sha256=protocol["baseline"]["checkpoint_sha256"],
            diagnostic_mode=args.mode, label=args.label, repeat_index=repeat, pid=os.getpid(), process_run_id=process_run_id,
            hostname=socket.gethostname(), numerical_observation=observer.data, elapsed_seconds=time.monotonic()-started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            evaluator_family=model.get("evaluator_family", "v24_Ilya"), initialization_manifest=file_record(manifest), **gpu)
        if previous is not None:
            result["within_process_comparison"] = compare_losses(previous, result, branches=("baseline",) if args.mode == "standalone" else ("baseline", "full"))
        if archived is not None:
            result.update(bridge_source=source, bridge_source_chosen_indices=archived.get("global_chosen_idx", archived["chosen_idx"]),
                          bridge_initialization_manifest=file_record(manifest),
                          bridge_dataset_time_mapping={"data_path": protocol["pilot"]["data_path"], "source_selected_year": 2022,
                                                       "selection": "decoded_source_time_coordinate[year==2022][chosen_idx]"},
                          development_bridge=compare_development(archived, result))
        write_new_json(output, result)
        config.out_json.unlink(missing_ok=True)
        previous = result
        print(f"Published numerical probe {output}", flush=True)
    return outputs


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
