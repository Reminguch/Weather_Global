"""Training stages for the matched fresh-residual data-size experiment.

This helper never submits jobs. Each GPU stage runs in its own process, and
production optimization uses the maintained cached/closed-loop trainers.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SELECTION_STEPS = tuple(range(2000, 20001, 2000))
ZERO_CURSOR = {"epoch": 0, "segment_index": 0, "segment_offset": 0}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from src.models.mamba.v24_Ilya.checkpoint import atomic_json_dump
    atomic_json_dump(value, Path(path))


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Conflicting training artifact: {path}")
    else:
        write(path, value)


def _clean(template):
    value = copy.deepcopy(template)
    for key in ("derived", "config_source", "shared_init_sha256"):
        value.pop(key, None)
    return value


def _require_protocol(value, *, closed=False):
    architecture, sequence = value["architecture"], value["sequence"]
    required = {"width": 512, "residual_initialization": "fresh", "residual_msg_steps": 2,
                "temporal_d_inner": 64, "temporal_bc_groups": 1, "temporal_dt_rank": "32",
                "temporal_d_state": 16, "temporal_d_conv": 4, "temporal_layers": 2,
                "temporal_stateful": True, "temporal_zero_init_out": True}
    if any(architecture.get(k) != v for k, v in required.items()):
        raise ValueError("Template differs from the historical fresh di64 control")
    if architecture.get("residual_width") not in (128, 512):
        raise ValueError("Unsupported residual width")
    if sequence != {"ar_tail_k": 20, "bptt_steps": 24, "segment_steps": 96,
                    "feedback_mode": "closed_loop_sg" if closed else "baseline",
                    "temporal_state_policy": "carry"}:
        raise ValueError("Template does not use the matched K20/BPTT24/segment96 protocol")
    optimizer = value["optimizer"]
    if (optimizer["learning_rate"] != (1e-5 if closed else 1e-4)
            or optimizer["warmup_steps"] != 200
            or optimizer["learning_rate_schedule"] != "constant"
            or optimizer["seed"] != 22 or optimizer["precision"] != "bf16"
            or value["distributed"]["per_device_batch_size"] != 1
            or value["distributed"]["mode"] != "single"
            or value["objective"] != {"loss_mode": "all_steps"}):
        raise ValueError("Optimizer/batch/objective differs from the historical control")


def open_config(template, output_root, arm, width, paths=None):
    """Preserve the actual control protocol, including its 200-step warmup."""
    value = _clean(template)
    _require_protocol(value)
    if width not in (128, 512):
        raise ValueError("Width must be 128 or 512")
    value["architecture"]["residual_width"] = width
    if paths:
        for key in ("prepared_root", "anchor_manifest_root"):
            value["data"][key] = str(paths[key])
    value["optimizer"].update(max_steps=20000, checkpoint_every=1000)
    value["validation"].update(enabled=True, every_steps=2000,
                                num_segments=None, final_num_segments=None)
    value["output"] = {"output_root": str(output_root), "run_name": arm}
    return value


def closed_config(template, output_root, arm, width):
    value = _clean(template)
    _require_protocol(value, closed=True)
    if width not in (128, 512):
        raise ValueError("Width must be 128 or 512")
    value["architecture"]["residual_width"] = width
    value["optimizer"].update(max_steps=2000, checkpoint_every=500)
    value["validation"].update(enabled=True, every_steps=500, num_segments=8, final_num_segments=8)
    value["output"] = {"output_root": str(output_root), "run_name": arm}
    return value


def initialization_config_identity(value):
    """Only dataset locations may differ when sharing fresh numerical state."""
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import scientific_config
    value = scientific_config(value)
    for key in ("prepared_root", "anchor_manifest_root"):
        value["data"].pop(key, None)
    return value


def check_fresh_initialization(source, config):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import numerical_tree_sha256
    from src.models.mamba.v24_Ilya.training.cached_parity import _initialization_checks
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer
    initial = _initialization_checks(source, config)
    optimizer, _ = build_optimizer(config)
    checks = {**initial["checks"], "step_zero": source.completed_step == 0,
              "fresh_cursor": source.training_cursor == ZERO_CURSOR,
              "fresh_optimizer": numerical_tree_sha256(source.optimizer_state)
                == numerical_tree_sha256(optimizer.init(source.residual_params)),
              "finite_parameters": all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(source.residual_params)),
              "same_scientific_protocol": initialization_config_identity(source.resolved_training_config)
                == initialization_config_identity(config.to_dict())}
    return {"passed": all(checks.values()), "checks": checks}


def prepare_init(config_path, output_dir, source_init=None):
    """Create a fresh initialization or rebind an audited step-zero state.

    Rebinding changes only dataset/config metadata. Parameters, optimizer,
    recurrent state, RNG and cursor remain byte-identical across data arms.
    """
    from src.models.mamba.v24_Ilya.checkpoint import (
        atomic_pickle_dump, file_sha256, load_v24_Ilya_training_checkpoint,
        manifest_sha256, training_checkpoint_payload)
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import numerical_checkpoint_identity
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    config_path, output_dir = Path(config_path), Path(output_dir)
    config = load_training_config(config_path)
    _require_protocol(config.to_dict())
    output = output_dir / "checkpoint_step00000000.pkl"
    marker = output_dir / "fresh_initialization.json"
    identity = {"config_sha256": file_sha256(config_path),
                "source_init": str(Path(source_init).resolve()) if source_init else None,
                "source_init_sha256": file_sha256(Path(source_init)) if source_init else None,
                "baseline_sha256": file_sha256(config.baseline_checkpoint),
                "anchor_manifest_sha256": manifest_sha256(config.anchor_manifest_root)}
    if marker.exists():
        saved = read(marker)
        if saved["identity"] != identity or file_sha256(output) != saved["checkpoint_sha256"]:
            raise ValueError("Fresh initialization identity changed")
        return output
    if source_init is None:
        from src.models.mamba.v24_Ilya.training.cached_runner import prepare_shared_initialization
        prepare_shared_initialization(config_path, output_dir, numerical_policy="advisory")
    else:
        source = load_v24_Ilya_training_checkpoint(Path(source_init))
        checks = check_fresh_initialization(source, config)
        if not checks["passed"] or source.baseline_checkpoint_fingerprint != identity["baseline_sha256"]:
            raise ValueError(f"Only matching fresh step-zero weights may be transferred: {checks}")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Incomplete/conflicting fresh initialization: {output_dir}")
        metadata = dict(source.parameter_overlay_metadata)
        metadata.update(matched_fresh_source=str(Path(source_init).resolve()),
                        matched_fresh_source_sha256=identity["source_init_sha256"],
                        matched_fresh_source_step=0)
        atomic_pickle_dump(training_checkpoint_payload(
            completed_step=0, residual_params=source.residual_params, residual_state=source.residual_state,
            optimizer_state=source.optimizer_state, rng_key=source.rng_key, training_cursor=source.training_cursor,
            resolved_training_config=config.to_dict(), baseline_checkpoint_path=str(config.baseline_checkpoint),
            baseline_checkpoint_fingerprint=identity["baseline_sha256"],
            anchor_manifest_fingerprint=identity["anchor_manifest_sha256"],
            parameter_overlay_metadata=metadata), output)
        if numerical_checkpoint_identity(load_v24_Ilya_training_checkpoint(output)) != numerical_checkpoint_identity(source):
            raise ValueError("Dataset rebinding changed numerical initialization")
    target = load_v24_Ilya_training_checkpoint(output)
    checks = check_fresh_initialization(target, config)
    if not checks["passed"]:
        raise ValueError(f"Fresh initialization audit failed: {checks}")
    write(marker, {"identity": identity, "checkpoint": str(output), "checkpoint_sha256": file_sha256(output),
                   "numerical_identity": numerical_checkpoint_identity(target), **checks})
    return output


def _bounded_environment():
    if os.environ.get("WEATHER_PREPARED_BOUNDED_READS") != "1":
        raise ValueError("All data-size workers require WEATHER_PREPARED_BOUNDED_READS=1")


def cache_execution(cache_root):
    """Pin the original cache when using the maintained advisory consumer."""
    cache_root = Path(cache_root)
    manifest = read(cache_root / "manifest.json")
    ready = read(cache_root / "READY.json")
    fingerprint = manifest["manifest_sha256"]
    if ready.get("partial") or ready.get("manifest_sha256") != fingerprint:
        raise ValueError("Training requires the complete matching cache READY marker")
    return {"cache_root": cache_root, "prefetch_batches": 0,
            "expected_manifest_sha256": fingerprint}


def _qualify_worker(config_path, init_path, cache_root, directory, *, replay=False):
    """Seventy measured updates and an independently restarted step-2 replay."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scripts.experiments.cached_40yr_qualification import _sample_years
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.model import build_model_configs
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import (
        build_provenance, validate_cached_checkpoint, write_cached_checkpoint, numerical_tree_sha256)
    from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config
    from src.models.mamba.v24_Ilya.training.cached_data import open_cached_training_data
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import (
        memory_snapshot, require_matched_gpu_environment, tree_comparison, comparison_integrity)
    from src.models.mamba.v24_Ilya.training.cached_parity import _forbid_gc, _raw_integrity
    from src.models.mamba.v24_Ilya.training.cached_runner import execution_metadata
    from src.models.mamba.v24_Ilya.training.cached_stepwise import make_cached_stepwise_train_step
    from src.models.mamba.v24_Ilya.training.cached_validation import run_cached_validation
    from src.models.mamba.v24_Ilya.training.data import TrainingCursor
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer
    _bounded_environment()
    started = time.monotonic()
    runtime = require_matched_gpu_environment("advisory")
    directory = Path(directory)
    resolved = load_cached_training_config(Path(config_path), **cache_execution(cache_root))
    config = resolved.common
    cached = open_cached_training_data(resolved, numerical_policy="advisory")
    source = load_v24_Ilya_training_checkpoint(Path(init_path))
    initial = check_fresh_initialization(source, config)
    if not initial["passed"]:
        raise ValueError("Fresh initialization failed")
    execution = execution_metadata(resolved.execution)
    provenance = build_provenance(config, Path(init_path), Path(cache_root), execution=execution)
    models = build_model_configs(cached.context[1].model_config, config.architecture)
    optimizer, _ = build_optimizer(config)
    trainer = make_cached_stepwise_train_step(config=config, model_config=models.residual,
        task_config=cached.context[2], stats=cached.context[3], optimizer=optimizer)
    context = ({}, resolved, cached, source, models, optimizer, trainer)
    bounded = None if replay else _sample_years(cached)
    raw = None if replay else _raw_integrity(context)
    if not replay and (not raw["integrity_passed"] or not bounded["passed"]):
        raise ValueError("Raw cache integrity or bounded weather reads failed")
    checkpoints = directory / "checkpoints"
    if replay:
        source = validate_cached_checkpoint(checkpoints / "checkpoint_step00000002.pkl", provenance, execution)
    params, state, opt, rng = (source.residual_params, source.residual_state,
                              source.optimizer_state, source.rng_key)
    cursor = TrainingCursor.from_mapping(source.training_cursor)
    config_cursor = cursor
    cached.training_data.validate_cursor(cursor, config)
    zero = jax.tree_util.tree_map(jnp.zeros_like, state)
    stop = 3 if replay else 70
    iterator = cached.iter_chunks(cursor, prefetch_batches=0, max_chunks=stop-source.completed_step)
    records, checkpoint_seconds = [], []
    setup = time.monotonic() - started
    with _forbid_gc():
        try:
            for step in range(source.completed_step + 1, stop + 1):
                tick = time.monotonic()
                batch = next(iterator)
                loaded = time.monotonic()
                if cursor.segment_offset == 0 and step > 1:
                    state = jax.tree_util.tree_map(jnp.zeros_like, state)
                rng, *keys = jax.random.split(rng, config.bptt_steps + 1)
                params, state, opt, loss, gradient, _ = trainer(
                    params, state, opt, jnp.stack(keys), batch.inputs, batch.targets, batch.forcings)
                loss, gradient = jax.device_get((loss, gradient))
                if not math.isfinite(float(loss)) or not math.isfinite(float(gradient)):
                    raise FloatingPointError("Nonfinite qualification update")
                cursor = batch.next_cursor
                records.append({"step": step, "loss": float(loss), "gradient_norm": float(gradient),
                                "seconds": time.monotonic()-tick, "compute_seconds": time.monotonic()-loaded})
                if not replay and step in (2, 3, 70):
                    saved_at = time.monotonic()
                    path = checkpoints / f"checkpoint_step{step:08d}.pkl"
                    write_cached_checkpoint(path, config=config, completed_step=step,
                        residual_params=params, residual_state=state, optimizer_state=opt, rng_key=rng,
                        cursor=cursor, baseline_fingerprint=source.baseline_checkpoint_fingerprint,
                        anchor_fingerprint=source.anchor_manifest_fingerprint,
                        overlay_metadata=source.parameter_overlay_metadata, provenance=provenance, execution=execution)
                    saved = validate_cached_checkpoint(path, provenance, execution)
                    for name, value in (("residual_params", params), ("residual_state", state),
                                        ("optimizer_state", opt), ("rng_key", rng), ("training_cursor", cursor.to_dict())):
                        if numerical_tree_sha256(value) != numerical_tree_sha256(getattr(saved, name)):
                            raise ValueError(f"Checkpoint serialization changed {name}")
                    checkpoint_seconds.append(time.monotonic()-saved_at)
                if not replay:
                    write(directory / "updates.json", records)
                del batch
        finally:
            iterator.close()
        if replay:
            expected = validate_cached_checkpoint(checkpoints / "checkpoint_step00000003.pkl", provenance, execution)
            comparisons = {name: tree_comparison(value, getattr(expected, name)) for name, value in (
                ("residual_params", params), ("residual_state", state), ("optimizer_state", opt))}
            exact_cursor = cursor.to_dict() == expected.training_cursor
            exact_rng = numerical_tree_sha256(rng) == numerical_tree_sha256(expected.rng_key)
            integrity = exact_cursor and exact_rng and all(comparison_integrity(v) for v in comparisons.values())
            result = {"integrity_passed": integrity, "passed": integrity and all(v["passed"] for v in comparisons.values()),
                      "cursor_equal": exact_cursor, "rng_equal": exact_rng, "checks": comparisons,
                      "resume_cursor": config_cursor.to_dict(), "peak_host_gib": memory_snapshot()["host_peak_gib"]}
            write(directory / "replay.json", result)
            if not integrity:
                raise ValueError("Independent checkpoint replay integrity failed")
            return result
        before = numerical_tree_sha256((params, state, opt, rng, cursor.to_dict()))
        validation = run_cached_validation(train_step=trainer, cached_data=cached, config=config,
                                            params=params, zero_state=zero, step=70)
        unchanged = before == numerical_tree_sha256((params, state, opt, rng, cursor.to_dict()))
        if (not unchanged or validation["selected_segments"] != 15
                or validation["available_segments"] != 15
                or not math.isfinite(validation["original_graphcast_loss"]["full_rollout"])):
            raise ValueError("Full common validation is incomplete, nonfinite or mutates training")
        write(directory / "validation.json", validation)
    result = {"completed_steps": 70, "runtime": runtime, "initialization": initial,
              "raw_integrity": raw, "bounded_memory": bounded, "serialization_exact": True,
              "validation_unchanged": unchanged, "full_validation_completed": True,
              "validation_seconds": validation["duration_seconds"], "wall_seconds": time.monotonic()-started,
              "startup_seconds": setup + records[0]["seconds"],
              "checkpoint_seconds": max(checkpoint_seconds), "peak_host_gib": memory_snapshot()["host_peak_gib"],
              "median_end_to_end_seconds": float(np.median([r["seconds"] for r in records[20:]])),
              "p90_end_to_end_seconds": float(np.percentile([r["seconds"] for r in records[20:]], 90)),
              "provenance": provenance}
    write(directory / "benchmark.json", result)
    return result


def qualify(config_path, init_path, cache_root, report_path):
    """Isolate GPU processes, then publish the immutable production gate."""
    from src.models.mamba.v24_Ilya.checkpoint import file_sha256
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import build_provenance, require_integrity_report
    from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config
    from src.models.mamba.v24_Ilya.training.cached_runner import execution_metadata
    _bounded_environment()
    report_path = Path(report_path)
    directory = report_path.parent / (report_path.stem + "_probe")
    directory.mkdir(parents=True, exist_ok=True)
    smoke = _clean(read(config_path))
    smoke["output"] = {"output_root": str(directory), "run_name": "run"}
    smoke_path = directory / "config.json"
    write_once(smoke_path, smoke)
    arguments = ["--config", str(smoke_path), "--init", str(init_path), "--cache-root", str(cache_root),
                 "--report", str(directory)]
    for stage, marker in (("qualify-worker", "benchmark.json"), ("replay-worker", "replay.json")):
        if not (directory / marker).exists():
            with (directory / f"{stage}.log").open("a") as stream:
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), stage, *arguments],
                               check=True, stdout=stream, stderr=subprocess.STDOUT)
    benchmark, replay = read(directory / "benchmark.json"), read(directory / "replay.json")
    resolved = load_cached_training_config(Path(config_path), **cache_execution(cache_root))
    provenance = build_provenance(resolved.common, Path(init_path), Path(cache_root),
                                  execution=execution_metadata(resolved.execution))
    if benchmark["provenance"] != provenance:
        raise ValueError("Qualification and production provenance differ")
    checks = {**benchmark["initialization"]["checks"],
              "raw_cache_integrity": benchmark["raw_integrity"]["integrity_passed"],
              "bounded_memory": benchmark["bounded_memory"]["passed"],
              "serialization_exact": benchmark["serialization_exact"],
              "full_validation_completed": benchmark["full_validation_completed"],
              "validation_unchanged": benchmark["validation_unchanged"],
              "independent_resume_integrity": replay["integrity_passed"],
              "seventy_finite_updates": benchmark["completed_steps"] == 70}
    result = {"kind": "cached_integrity", "numerical_policy": "advisory", "provenance": provenance,
              "passed": all(checks.values()), "integrity_passed": all(checks.values()), "checks": checks,
              "benchmark": benchmark, "replay": replay,
              "numerical_warnings": [] if replay["passed"] else ["Independent GPU replay tolerance drift"],
              "probe_config_sha256": file_sha256(smoke_path)}
    write_once(report_path, result)
    require_integrity_report(report_path, provenance)
    return result


def reusable_evidence_checks(evidence, previous_config, config, previous_init_sha256, provenance):
    """Reuse scientific execution evidence only with identical arithmetic/settings."""
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import digest_json, scientific_config
    prior = evidence["provenance"]
    # The frozen training implementation is the original control's implementation.
    # Prepared storage and campaign orchestration may change independently.
    prefixes = ("src/models/mamba/v24_Ilya/training/", "src/models/mamba/modules/",
                "src/models/mamba/training/", "third_party/graphcast/")
    arithmetic = {k: v for k, v in prior["code_sha256"].items()
                  if k.startswith(prefixes) or k == "src/models/mamba/v24_Ilya/model.py"}
    checks = {
        "prior_execution_integrity": evidence.get("kind") == "cached_integrity"
            and evidence.get("numerical_policy") == "advisory"
            and evidence.get("integrity_passed") is True
            and bool(evidence.get("checks")) and all(v is True for v in evidence["checks"].values()),
        "prior_config_identity": prior["scientific_config_sha256"] == digest_json(scientific_config(previous_config)),
        "same_scientific_protocol": initialization_config_identity(previous_config)
            == initialization_config_identity(config),
        "prior_initialization_identity": prior["shared_init_sha256"] == previous_init_sha256,
        "same_training_arithmetic": bool(arithmetic)
            and all(provenance["code_sha256"].get(k) == v for k, v in arithmetic.items()),
        "same_environment": prior["environment"] == provenance["environment"],
    }
    if not all(checks.values()):
        raise ValueError(f"Scientific qualification evidence cannot be reused: {checks}")
    return checks


def startup(config_path, init_path, cache_root, report_path, *, evidence_path,
            previous_config_path, previous_init_path, benchmark_path=None):
    """Bind existing execution evidence to new inputs without optimizer probes.

    The completed cache verifier checks all prediction files; alignment checks
    every overlapping weather value. This stage verifies identities and bounded
    samples at the beginning, middle and end of the new cache on CPU.
    """
    import numpy as np
    from src.models.mamba.v24_Ilya import baseline_cache
    from src.models.mamba.v24_Ilya.checkpoint import file_sha256, load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import build_provenance, require_integrity_report
    from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config
    from src.models.mamba.v24_Ilya.training.cached_runner import execution_metadata
    _bounded_environment()
    config_path, init_path, cache_root = map(Path, (config_path, init_path, cache_root))
    resolved = load_cached_training_config(config_path, **cache_execution(cache_root))
    provenance = build_provenance(resolved.common, init_path, cache_root,
                                  execution=execution_metadata(resolved.execution))
    evidence = read(evidence_path)
    previous = _clean(read(previous_config_path))
    checks = reusable_evidence_checks(evidence, previous, resolved.common.to_dict(),
                                      file_sha256(Path(previous_init_path)), provenance)
    initialization = check_fresh_initialization(load_v24_Ilya_training_checkpoint(init_path), resolved.common)
    checks.update(initialization["checks"])
    alignment = read(cache_root / "alignment.json")
    manifest = read(cache_root / "manifest.json")
    ready = read(cache_root / "READY.json")
    if (alignment.get("passed") is not True or not alignment.get("overlap", {}).get("bitwise_equal")
            or alignment.get("manifest_sha256") != manifest["manifest_sha256"]
            or ready.get("chunks") != len(manifest["chunks"])
            or ready.get("shards") != manifest["num_shards"]):
        raise ValueError("New cache lacks complete matching alignment/verification evidence")
    context = baseline_cache.load_context(config_path)
    if baseline_cache.build_manifest(context, manifest["num_shards"]) != manifest:
        raise ValueError("New cache inputs differ from its completed manifest")
    selected = sorted({0, len(manifest["chunks"]) // 2, len(manifest["chunks"]) - 1})
    samples = []
    for index in selected:
        item = manifest["chunks"][index]
        baseline_cache.verify_shard(cache_root, manifest, item["shard"], checksums=False)
        for name, schema in manifest["variables"].items():
            path = cache_root / f'shard_{item["shard"]:03d}/chunk_{item["id"]:06d}/{name}.npy'
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (list(array.shape) != [manifest["compatibility"]["bptt_steps"], *schema["shape"]]
                    or str(array.dtype) != schema["dtype"] or not np.isfinite(array[[0, -1]]).all()):
                raise ValueError(f"Invalid bounded cache sample: {path}")
        # Load only a single prepared frame per selected window.
        data = context[4]
        raw = item["raw_anchor_indices"][0]
        for name, value in data.store.data_vars.items():
            values = (value.data.take(np.asarray([raw]), axis=value.dims.index("time"))
                      if "time" in value.dims else np.asarray(value.data))
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite bounded prepared sample: {name}")
        samples.append(dict(chunk_id=item["id"], first_target=item["target_timestamps"][0]))
    checks.update(new_cache_identity=True, bounded_first_middle_last_finite=True,
                  full_prepared_overlap_verified=True)
    if not all(checks.values()):
        raise ValueError(f"New cache startup integrity failed: {checks}")
    benchmark = evidence.get("benchmark")
    if benchmark_path is not None:
        historical = read(benchmark_path)
        benchmark = historical.get("backends", {}).get("cached_4cpu", historical)
    if not benchmark:
        raise ValueError("Missing reusable training resource evidence")
    paths = [evidence_path, previous_config_path, previous_init_path,
             cache_root / "READY.json", cache_root / "alignment.json"]
    if benchmark_path is not None:
        paths.append(benchmark_path)
    result = dict(kind="cached_integrity", numerical_policy="advisory", passed=True,
                  integrity_passed=True, checks=checks, provenance=provenance, benchmark=benchmark,
                  qualification_reused=True, optimizer_probe_updates=0, samples=samples,
                  numerical_warnings=evidence.get("numerical_warnings", []),
                  reused_evidence=[dict(path=str(Path(p).resolve()), sha256=file_sha256(Path(p))) for p in paths])
    write_once(report_path, result)
    require_integrity_report(Path(report_path), provenance)
    return result


def train_open(config_path, init_path, cache_root, report_path, *, stop_step, integrity_report):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import file_sha256, load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import latest_checkpoint_for_run
    from src.models.mamba.v24_Ilya.training.cached_runner import run_cached_training
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    _bounded_environment()
    config = load_training_config(Path(config_path))
    if stop_step not in range(4000, 20001, 4000):
        raise ValueError("Production open-loop stages end at 4k intervals")
    resumed = latest_checkpoint_for_run(config)
    previous_step = load_v24_Ilya_training_checkpoint(resumed).completed_step if resumed else 0
    if not stop_step - 4000 <= previous_step <= stop_step:
        raise ValueError("Open-loop stage requires its completed preceding 4k boundary or a local exact resume")
    started = time.monotonic()
    output = run_cached_training(Path(config_path), Path(init_path), **cache_execution(cache_root),
        resume=resumed, max_steps=stop_step,
        # The frozen control runner requires both legacy arguments before its
        # advisory branch selects and validates only the integrity report.
        parity_report=Path(integrity_report), paired_report=Path(integrity_report),
        numerical_policy="advisory", integrity_report=Path(integrity_report))
    saved = load_v24_Ilya_training_checkpoint(output)
    if saved.completed_step != stop_step or saved.resolved_training_config != config.to_dict():
        raise ValueError("Open-loop stage checkpoint has the wrong completed step/configuration")
    if not all(np.isfinite(v).all() for tree in (saved.residual_params, saved.residual_state, saved.optimizer_state)
               for v in jax.tree_util.tree_leaves(tree)):
        raise ValueError("Nonfinite open-loop checkpoint")
    rows = [json.loads(line) for line in (config.run_dir / "train_metrics.jsonl").read_text().splitlines() if line]
    if [row["step"] for row in rows] != list(range(1, stop_step + 1)):
        raise ValueError("Missing/duplicated open-loop steps")
    result = {"passed": True, "completed_step": stop_step, "checkpoint": str(output),
              "checkpoint_sha256": file_sha256(output), "resumed_from": str(resumed) if resumed else None,
              "wall_seconds": time.monotonic()-started,
              "median_step_seconds": float(np.median([r["step_seconds"] for r in rows[20:]])),
              "cursor": saved.training_cursor,
              "year_exposure": year_exposure(rows, read(Path(cache_root) / "manifest.json"))}
    write(report_path, result)
    return result


def year_exposure(rows, manifest):
    """Count actual consumed forecast timestamps, including partial epochs."""
    chunks = {row["id"]: row for row in manifest["chunks"] if row["split"] == "train"}
    counts, updates = {}, {}
    for row in rows:
        for chunk_id in row["chunk_ids"] if "chunk_ids" in row else [row["chunk_id"]]:
            chunk = chunks[chunk_id]
            first_year = chunk["target_timestamps"][0][:4]
            updates[first_year] = updates.get(first_year, 0) + 1
            for timestamp in chunk["target_timestamps"]:
                year = timestamp[:4]
                counts[year] = counts.get(year, 0) + 1
    return {"weather_steps_by_year": dict(sorted(counts.items())),
            "chunk_uses_by_first_target_year": dict(sorted(updates.items())),
            "total_updates": len(rows), "total_weather_steps": sum(counts.values())}


def select_records(records):
    """A complete fixed grid prevents failed/partial runs biasing selection."""
    rows = [row for row in records if row.get("role") == "fixed_checkpoint"]
    if sorted(int(row["step"]) for row in rows) != list(SELECTION_STEPS):
        raise ValueError("Selection requires exactly one validation at every 2k update through 20k")
    for row in rows:
        if (row.get("selected_segments") != 15 or row.get("available_segments") != 15
                or row.get("validation_protocol") != "cached_stepwise_all_segments_v1"
                or not math.isfinite(float(row["original_graphcast_loss"]["full_rollout"]))):
            raise ValueError("Selection requires full common-2022 exact validation")
    if len({row["subset_fingerprint"] for row in rows}) != 1:
        raise ValueError("Validation subset changed within the run")
    return min(rows, key=lambda row: (float(row["original_graphcast_loss"]["full_rollout"]), int(row["step"])))


def select_parent(config_path, report_path):
    from src.models.mamba.v24_Ilya.checkpoint import file_sha256, load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    config = load_training_config(Path(config_path))
    records = [json.loads(line) for line in (config.run_dir / "validation_metrics.jsonl").read_text().splitlines() if line]
    selected = select_records(records)
    path = config.run_dir / "checkpoints" / f"checkpoint_step{int(selected['step']):08d}.pkl"
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    if checkpoint.completed_step != selected["step"] or checkpoint.resolved_training_config != config.to_dict():
        raise ValueError("Selected checkpoint identity differs from configuration")
    result = {"passed": True, "step": selected["step"], "checkpoint": str(path),
              "checkpoint_sha256": file_sha256(path), "selection_loss": selected["original_graphcast_loss"]["full_rollout"],
              "selection_metric": "exact_original_graphcast_loss", "selection_grid": list(SELECTION_STEPS),
              "validation": selected, "endpoint_validation": next(r for r in records if r["step"] == 20000)}
    write_once(report_path, result)
    return result


def train_closed(config_path, parent, report_path):
    """Exact 2k continuation, fresh optimizer and zero memory, shared 7-year data."""
    from src.models.mamba.v24_Ilya.checkpoint import file_sha256, load_v24_Ilya_training_checkpoint, manifest_sha256
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from scripts.experiments.run_v24_cached_w128_closed_loop import check_training
    _bounded_environment()
    config = load_training_config(Path(config_path))
    _require_protocol(config.to_dict(), closed=True)
    if config.output_root.name != "runs":
        raise ValueError("Closed-loop output_root must end in /runs")
    parent = Path(parent)
    source = load_v24_Ilya_training_checkpoint(parent)
    if source.resolved_training_config["architecture"] != config.to_dict()["architecture"]:
        raise ValueError("Closed-loop transfer changes residual architecture")
    latest = config.run_dir / "latest_checkpoint.json"
    args = [sys.executable, "-u", str(ROOT / "scripts/training/train_v24_Ilya.py"), "--config", str(config_path)]
    if latest.exists():
        resume = Path(read(latest)["checkpoint"])
        saved = load_v24_Ilya_training_checkpoint(resume)
        if (saved.parameter_overlay_metadata.get("init_from_sha256") != file_sha256(parent)
                or saved.resolved_training_config != config.to_dict()):
            raise ValueError("Closed-loop resume belongs to another parent/configuration")
        args += ["--resume", str(resume)]
    else:
        args += ["--init-from", str(parent), "--zero-state-on-init"]
    started = time.monotonic()
    subprocess.run(args, check=True)
    # The maintained checker expects <experiment>/runs/<arm>; any actual output
    # layout is handled by passing the output root's parent and run name.
    entry = {"arm": config.run_name, "config": str(config_path),
             "parent": {"path": str(parent), "sha256": file_sha256(parent)},
             "baseline_fingerprint": source.baseline_checkpoint_fingerprint,
             "anchor_fingerprint": manifest_sha256(config.anchor_manifest_root)}
    result = check_training(config.output_root.parent, entry)
    final = config.run_dir / "checkpoints/checkpoint_step00002000.pkl"
    result.update(checkpoint=str(final), checkpoint_sha256=file_sha256(final),
                  parent=str(parent), wall_seconds=time.monotonic()-started)
    write(report_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "qualify", "startup", "qualify-worker", "replay-worker",
                                          "train-open", "select", "train-closed"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-init", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--integrity-report", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--previous-config", type=Path)
    parser.add_argument("--previous-init", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--stop-step", type=int)
    args = parser.parse_args()
    if args.stage == "init":
        result = {"checkpoint": str(prepare_init(args.config, args.output_dir, args.source_init))}
    elif args.stage == "qualify":
        result = qualify(args.config, args.init, args.cache_root, args.report)
    elif args.stage == "startup":
        result = startup(args.config, args.init, args.cache_root, args.report,
                         evidence_path=args.evidence, previous_config_path=args.previous_config,
                         previous_init_path=args.previous_init, benchmark_path=args.benchmark)
    elif args.stage in ("qualify-worker", "replay-worker"):
        result = _qualify_worker(args.config, args.init, args.cache_root, args.report,
                                 replay=args.stage == "replay-worker")
    elif args.stage == "train-open":
        result = train_open(args.config, args.init, args.cache_root, args.report,
                            stop_step=args.stop_step, integrity_report=args.integrity_report)
    elif args.stage == "select":
        result = select_parent(args.config, args.report)
    else:
        result = train_closed(args.config, args.parent, args.report)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
