"""Real stage workers. Completion receipts are written only after successful work."""
from __future__ import annotations

from pathlib import Path
import json
import time
import jax
import numpy as np
from .backbone import FrozenBackbone, download_checkpoint
from .cache import (CacheReader, initialize_cache, live_record, producer_identity, produce_shard,
                    valid_k1_origins, verify_cache)
from .config import initial_matrix, load_config, FIELDS
from .data import PreparedStore, prepare_era5, origin_manifest, STEP, require_complete_experiment_data
from .native_state import NativeAdapter
from .normalization import fit_statistics
from .io import digest, read_json, sha256, write_json, write_pickle, versions
from .launcher import resolve_resources


def require_receipt(root, stage, identifier, source_id):
    path = Path(root) / "checks" / f"{stage}_{identifier}.json"
    value = read_json(path)
    if not value.get("passed") or value.get("source_id") != source_id:
        raise ValueError(f"Missing or stale successful stage receipt: {path}")
    return value


def inspect_backbone(root, manifest, resolution):
    import neuralgcm
    r = 2.8 if resolution == "res2p8" else 1.4
    index = Path(manifest["resources"][resolution]["checkpoint_index"])
    if not index.exists():
        path = download_checkpoint(r, index.parent)
        write_json(index, {"path": str(path.resolve()), "sha256": sha256(path), "resolution": r}, immutable=True)
    resources = resolve_resources(manifest, resolution)
    backbone = FrozenBackbone.load(resources["checkpoint"], resources["checkpoint_sha256"])
    # Official demo is for structural inspection only, never statistics or skill claims.
    inputs, forcing = backbone.model.data_from_xarray(neuralgcm.demo.load_data(backbone.model.data_coords).isel(time=0))
    state = backbone.encode(inputs, forcing)
    adapter = NativeAdapter(backbone.model, state)
    adapter.assert_zero_identity(state)
    out = Path(root) / "manifests" / resolution
    write_json(out / "backbone.json", backbone.manifest(state), immutable=True)
    write_json(out / "native_schema.json", adapter.schema, immutable=True)
    from .model import geometry_manifest
    from .config import Architecture
    write_json(out / "geometry.json", geometry_manifest(Architecture(), adapter.grid.latitudes,
               adapter.grid.longitudes, adapter.output_size), immutable=True)
    write_pickle(out / "inspection_state.pkl", jax.device_get(state))
    return {"checkpoint": resources["checkpoint_sha256"], "native_schema": adapter.identity,
            "channels": adapter.output_size, "inspection_only": True}


def prepare_data(root, manifest, resolution, *, start=None, end=None, shard_id=None):
    resources = resolve_resources(manifest, resolution)
    backbone = FrozenBackbone.load(resources["checkpoint"], resources["checkpoint_sha256"])
    output = Path(resources["prepared"])
    if shard_id == "merge":
        from .data import merge_prepared
        shards = sorted(p.parent for p in (output / "shards").glob("*/READY.json"))
        merge_prepared(shards, output)
    else:
        if shard_id:
            if not start or not end:
                raise ValueError("Data time shards require explicit start/end")
            output = output / "shards" / shard_id
        prepare_era5(manifest["era5_source"], backbone.model, output, start or "2015-01-01", end or "2024-01-01")
    store = PreparedStore(output)
    return {"dataset_id": store.identity, "timestamps": len(store.times), "root": str(output),
            "shard": shard_id}


def native_setup(resources):
    backbone = FrozenBackbone.load(resources["checkpoint"], resources["checkpoint_sha256"])
    store = PreparedStore(resources["prepared"])
    require_complete_experiment_data(store)
    origins = valid_k1_origins(store, "train")
    if not origins:
        raise ValueError("No eligible training k=1 origins")
    inputs, forcing = store.inputs_and_forcing(backbone.model, origins[0])
    adapter = NativeAdapter(backbone.model, backbone.encode(inputs, forcing))
    return backbone, store, adapter


def build_cache(manifest, resolution, shard_id):
    resources = resolve_resources(manifest, resolution)
    b, store, adapter = native_setup(resources)
    producer = producer_identity(b, adapter, store)
    cache_manifest = initialize_cache(resources["cache_root"], producer,
                                      {s: valid_k1_origins(store, s) for s in ("train", "val")})
    if shard_id == "plan":
        return {"producer_id": cache_manifest["producer_id"], "planned_shards": len(cache_manifest["shards"])}
    if shard_id and shard_id.startswith("lane-"):
        # Reuse one initialized backbone across disjoint groups of small shards.
        _, lane, count = shard_id.split("-")
        lane, count = int(lane), int(count)
        if not 0 <= lane < count:
            raise ValueError("Invalid cache lane")
        ids = [s["id"] for s in cache_manifest["shards"]][lane::count]
    else:
        ids = [shard_id] if shard_id else [s["id"] for s in cache_manifest["shards"]]
    shards = []
    for sid in ids:
        path = Path(resources["cache_root"])
        # A completed pilot is reusable, but an interrupted/corrupt shard is not.
        if (path / f"{sid}.json").exists():
            previous = read_json(path / f"{sid}.json")
            planned = next(s for s in cache_manifest["shards"] if s["id"] == sid)
            if (previous["producer_id"] != cache_manifest["producer_id"]
                    or previous["records"] != len(planned["origins"])
                    or previous["sha256"] != sha256(path / f"{sid}.zip")):
                raise ValueError(f"Completed cache shard changed: {sid}")
            shards.append(previous)
        else:
            shards.append(produce_shard(resources["cache_root"], sid, b, adapter, store))
        print(json.dumps({"cache_shard_complete": sid, "records": shards[-1]["records"]}), flush=True)
    return {"producer_id": cache_manifest["producer_id"], "completed_shards": shards}


def verify_and_fit(root, manifest, resolution):
    resources = resolve_resources(manifest, resolution)
    b, store, adapter = native_setup(resources)
    ready = verify_cache(resources["cache_root"], fresh_record=lambda t: live_record(b, store, t))
    cache_manifest = read_json(Path(resources["cache_root"]) / "manifest.json")
    if {s["split"] for s in cache_manifest["shards"]} != {"train", "val"}:
        raise ValueError("Production requires complete train and validation cache splits")
    reader = CacheReader(resources["cache_root"], store, "train")
    def statistics_records():
        for i in range(len(reader)):
            record = reader[i]
            inputs, forcing = store.inputs_and_forcing(b.model, record["valid"])
            record["next_truth_state"] = jax.device_get(b.encode(inputs, forcing))
            record["truth"] = store.frame(record["origin"])
            yield record
    if not Path(resources["statistics"]).exists():
        fit_statistics(statistics_records(), adapter, b.model.data_coords.horizontal.latitudes,
                       dataset_id=store.identity, output=resources["statistics"])
    for split, count, steps, key in (("val", 32, 20, "validation_origins"), ("test", 128, 40, "test_origins")):
        write_json(resources[key], origin_manifest(store.times, split, count, steps, store.identity), immutable=True)
    # The sealed test origin file is never used by training/selection.
    return {"producer_id": ready["producer_id"], "statistics_sha256": sha256(resources["statistics"]),
            "ready_sha256": sha256(Path(resources["cache_root"]) / "READY.json")}


def preflight(root, manifest, resolution, phase):
    if phase == "inspect":
        return inspect_backbone(root, manifest, resolution)
    from .runtime import build_runtime, runtime_identities
    from .checks import numerical_preflight, profile_runtime
    require_receipt(root, "verify-cache", resolution, manifest["source_id"])
    run_id = ("r2p8_w128_di16" if phase == "numerical" else "r2p8_w256_di32") if resolution == "res2p8" else "r1p4_w256_di32"
    config = load_config(manifest["configs"][run_id]["path"])
    resources = resolve_resources(manifest, resolution)
    runtime = build_runtime(config, resources, stage="pretrain")
    reader = CacheReader(resources["cache_root"], runtime.store, "train")
    identities = runtime_identities(runtime, config, manifest["source_id"], reader.manifest["producer_id"])
    origins = read_json(resources["validation_origins"])["origins"][:4]
    path = Path(root) / "checks" / f"{phase}_{resolution}.json"
    if phase == "numerical":
        return numerical_preflight(runtime, reader, origins, path, identities)
    return profile_runtime(runtime, reader, origins[0], path, identities)


def require_production_gates(root, manifest, resolution, runtime):
    # Require every resolution explicitly selected in the immutable experiment.
    for rid in manifest["resources"]:
        numerical = read_json(Path(root) / "checks" / f"numerical_{rid}.json")
        profile = read_json(Path(root) / "checks" / f"profile_{rid}.json")
        for report in (numerical, profile):
            if report["identities"]["source"] != manifest["source_id"]:
                raise ValueError("Stale preflight source identity")
        if not numerical["passed"] or not profile["measured"] or not profile["gpu_measured"]:
            raise ValueError("Production requires passed real numerical gates and GPU profiles at all active resolutions")
        if rid == resolution:
            for key, value in (("backbone", runtime.backbone.checkpoint_sha256),
                               ("native_schema", runtime.adapter.identity),
                               ("dataset", runtime.store.identity), ("normalization", runtime.normalization.identity)):
                if numerical["identities"][key] != value or profile["identities"][key] != value:
                    raise ValueError(f"Preflight dependency changed: {key}")


def validation_callback(root, manifest, runtime, resources, output, stage):
    from .evaluate import evaluate_origins, select_checkpoint
    from .model import zero_memory
    val_origins = read_json(resources["validation_origins"])
    reader = CacheReader(resources["cache_root"], runtime.store, "val")
    def validate(params, checkpoint, index, ar=True):
        step_root = Path(output) / "validation" / f"{index:06d}"
        checkpoint_id = sha256(checkpoint)
        if (step_root / "COMPLETE.json").exists():
            if read_json(step_root / "COMPLETE.json")["checkpoint_sha256"] != checkpoint_id:
                raise ValueError("Validation checkpoint identity changed")
            return
        if step_root.exists():
            # Interrupted validation must not pollute resumed scores or train memory.
            step_root.rename(step_root.with_name(step_root.name + f".interrupted_{time.time_ns()}"))
        if stage == "pretrain":
            h, previous, losses, since_reset = zero_memory(runtime.memory), None, [], 0
            for i in range(len(reader)):
                record = reader[i]
                if since_reset == 96 or (previous is not None and np.datetime64(record["origin"]) - previous != STEP):
                    h = zero_memory(runtime.memory)
                    since_reset = 0
                item = runtime.trainer.record(record["origin_state"], record["baseline_state"], record["forcing"],
                    record["target"], runtime.known(record["origin_state"], record["forcing"]))
                value, h, _ = runtime.trainer.forward(params, h, jax.random.PRNGKey(i), *item)
                losses.append(float(value))
                previous = np.datetime64(record["origin"])
                since_reset += 1
            score = float(np.mean(losses))
            write_json(step_root / "one_step.json", {"loss": score, "records": len(losses)}, immutable=True)
            best_path = Path(output) / "best_one_step.json"
            if not best_path.exists() or score < read_json(best_path)["score"]:
                write_json(best_path, {"score": score, "checkpoint": str(checkpoint), "sha256": sha256(checkpoint)})
        if ar:
            for warm in (False, True):
                report = evaluate_origins(runtime, params, val_origins, step_root / ("warm" if warm else "cold"), warm=warm)
                if not warm:
                    select_checkpoint(Path(output) / "selected.json", checkpoint, report, split="val", epoch_or_update=index)
        write_json(step_root / "COMPLETE.json", {"checkpoint_sha256": checkpoint_id, "ar": ar}, immutable=True)
    return validate


def train(root, manifest, run_id, stage, resume):
    from .runtime import build_runtime, runtime_identities
    from .pretrain import run_pretrain
    from .finetune import run_finetune
    config = load_config(manifest["configs"][run_id]["path"])
    resources = resolve_resources(manifest, config.resolution_id)
    require_receipt(root, "verify-cache", config.resolution_id, manifest["source_id"])
    runtime = build_runtime(config, resources, stage=stage)
    require_production_gates(root, manifest, config.resolution_id, runtime)
    reader = CacheReader(resources["cache_root"], runtime.store, "train")
    identities = runtime_identities(runtime, config, manifest["source_id"], reader.manifest["producer_id"])
    output = Path(root) / "runs" / run_id / "seed22" / stage
    callback = validation_callback(root, manifest, runtime, resources, output, stage)
    if stage == "pretrain":
        run_pretrain(runtime, reader, config, output, identities, resume=resume, validate=callback)
    else:
        require_receipt(root, "pretrain", run_id, manifest["source_id"])
        parent_selection = read_json(output.parent / "pretrain/selection_locked.json")
        parent = parent_selection["checkpoint"]
        if sha256(parent) != parent_selection["sha256"]:
            raise ValueError("Selected pretrained parent changed")
        run_finetune(runtime, config, output, identities, parent, resume=resume, validate=callback)
    selected = read_json(output / "selected.json")
    write_json(output / "selection_locked.json", selected, immutable=True)
    write_json(output / "resolved_config.json", config.to_dict(), immutable=True)
    return {"selected": selected, "config": config.identity, "output": str(output)}


def evaluate_run(root, manifest, run_id):
    from dinosaur import horizontal_interpolation
    from .runtime import build_runtime
    from .checkpoint import load_checkpoint
    from .evaluate import evaluate_origins
    from .loss import WeatherLoss
    from .normalization import Normalization
    config = load_config(manifest["configs"][run_id]["path"])
    require_receipt(root, "finetune", run_id, manifest["source_id"])
    resources = resolve_resources(manifest, config.resolution_id)
    runtime = build_runtime(config, resources, stage="finetune")
    require_production_gates(root, manifest, config.resolution_id, runtime)
    test_origins = read_json(resources["test_origins"])
    if test_origins["split"] != "test" or test_origins["steps"] != 40 or len(test_origins["origins"]) != 128:
        raise ValueError("Held-out evaluation requires the sealed 128-origin ten-day manifest")
    coarse = resolve_resources(manifest, "res2p8")
    coarse_b = FrozenBackbone.load(coarse["checkpoint"], coarse["checkpoint_sha256"])
    scales = Normalization.load(coarse["statistics"]).manifest["loss_scales"]
    common = (horizontal_interpolation.ConservativeRegridder(runtime.backbone.model.data_coords.horizontal,
               coarse_b.model.data_coords.horizontal),
              WeatherLoss(coarse_b.model.data_coords.horizontal.latitudes, coarse_b.model.data_coords.vertical.centers, scales))
    runs = {}
    for stage in ("baseline", "pretrain", "finetune"):
        params = None
        if stage != "baseline":
            selection = read_json(Path(root) / "runs" / run_id / "seed22" / stage / "selection_locked.json")
            path = Path(selection["checkpoint"])
            if sha256(path) != selection["sha256"]:
                raise ValueError("Locked selection checkpoint changed")
            metadata = read_json(path.with_suffix(".json"))
            params = load_checkpoint(path, stage=stage, identities=metadata["identities"])["params"]
        for warm in ((False,) if stage == "baseline" else (False, True)):
            label = stage + ("_warm" if warm else "_cold")
            output = Path(root) / "evaluation" / run_id / label
            runs[label] = evaluate_origins(runtime, params, test_origins, output,
                                           warm=warm, diagnostics=True, common=common)
    if any(r["failures"] for r in runs.values()):
        raise FloatingPointError("Held-out failures recorded; comparison is incomplete and cannot be marked successful")
    return runs


def report_experiment(root, manifest):
    from .report import build_report
    for run_id in manifest["configs"]:
        require_receipt(root, "evaluate", run_id, manifest["source_id"])
    return build_report(root, manifest)


def execute(root, manifest, stage, *, resolution=None, run_id=None, shard_id=None, phase="inspect",
            resume=None, start=None, end=None):
    from .numerics import verify_runtime
    verify_runtime()
    root = Path(root).resolve()
    current = versions()
    if current != manifest["libraries"]:
        raise ValueError(f"Worker dependency versions differ from prepared experiment: {current}")
    if stage in ("pretrain", "finetune", "evaluate"):
        if not run_id or resolution:
            raise ValueError("Worker requires one run-id")
    elif stage != "report" and (not resolution or run_id):
        raise ValueError("Shared worker requires one resolution")
    started = time.perf_counter()
    if stage == "preflight":
        result = preflight(root, manifest, resolution, phase)
    elif stage == "data":
        require_receipt(root, "inspect", resolution, manifest["source_id"])
        result = prepare_data(root, manifest, resolution, start=start, end=end, shard_id=shard_id)
    elif stage == "cache":
        result = build_cache(manifest, resolution, shard_id)
    elif stage == "verify-cache":
        result = verify_and_fit(root, manifest, resolution)
    elif stage in ("pretrain", "finetune"):
        result = train(root, manifest, run_id, stage, resume)
    elif stage == "evaluate":
        result = evaluate_run(root, manifest, run_id)
    elif stage == "report":
        result = report_experiment(root, manifest)
    else:
        raise ValueError(stage)
    name = phase if stage == "preflight" else stage
    identifier = run_id or resolution or "all"
    if shard_id:
        identifier += "_" + shard_id
    receipt = {"stage": stage, "phase": phase if stage == "preflight" else None,
               "source_id": manifest["source_id"], "passed": True,
               "seconds": time.perf_counter() - started, "result": result}
    # Numerical/profile reports have their own content-bound filenames.
    filename = f"receipt_{name}_{identifier}.json" if name in ("numerical", "profile") else f"{name}_{identifier}.json"
    write_json(root / "checks" / filename, receipt, immutable=True)
    return receipt
