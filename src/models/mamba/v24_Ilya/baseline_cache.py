"""Immutable, sharded FP32 baseline trajectories for v24 open-loop training.

No residual model is constructed. Chunk identities follow the training manifest,
not calendar-year partitions, so the truth-prefix/AR transitions remain exact.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any
import uuid

import numpy as np
import xarray as xr

FORMAT = "v24_baseline_cache_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_manifest(root: Path, manifest: dict) -> None:
    """Publish once, safely even when several array workers start together."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    tmp = root / (".manifest." + uuid.uuid4().hex)
    atomic_json(tmp, manifest)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Incompatible existing cache manifest: {path}")
    finally:
        tmp.unlink(missing_ok=True)


def enumerate_chunks(data, config) -> tuple[list[dict], dict]:
    chunks, excluded = [], {}
    times = np.asarray(data.store.time.values).astype("datetime64[ns]")
    dt = np.timedelta64(int(data.time_step.value), "ns")
    for split, segments, indices in (
        ("train", data.segments, data.train_split),
        ("val", data.validation_segments, data.val_split),
    ):
        used = sum(len(s) for s in segments)
        excluded[split] = {
            "count": int(len(indices) - used),
            "raw_anchor_indices": data.anchor_indices[indices[used:]].tolist(),
            "reason": "incomplete segment dropped by existing trainer",
        }
        for segment_id, segment in enumerate(segments):
            for offset in range(0, config.segment_steps, config.bptt_steps):
                raw = data.anchor_indices[segment[offset:offset + config.bptt_steps]]
                if len(raw) != config.bptt_steps or not np.all(np.diff(raw) == 1):
                    raise ValueError("Cache requires complete, contiguous training chunks")
                if not np.all(np.diff(times[raw]) == dt):
                    raise ValueError("Nonuniform time spacing inside a chunk")
                target_times = times[raw + 1]
                chunks.append({
                    "id": len(chunks), "split": split, "segment_id": segment_id,
                    "segment_offset": offset, "chunk_index": offset // config.bptt_steps,
                    "raw_anchor_indices": raw.tolist(),
                    "target_timestamps": [str(t) for t in target_times],
                })
    # Fail closed if an anchor manifest allows targets to overlap splits.
    train_targets = {t for c in chunks if c["split"] == "train" for t in c["target_timestamps"]}
    val_targets = {t for c in chunks if c["split"] == "val" for t in c["target_timestamps"]}
    if train_targets & val_targets:
        raise ValueError("Training and validation targets overlap")
    return chunks, excluded


def select_chunks(chunks: list[dict], max_chunks: int | None, num_shards: int) -> list[dict]:
    if num_shards <= 0 or (max_chunks is not None and max_chunks <= 0):
        raise ValueError("num_shards and max_chunks must be positive")
    selected = chunks
    if max_chunks is not None and max_chunks < len(chunks):
        # Four-chunk pilot: first/last train and first/last validation.
        groups = [[c for c in chunks if c["split"] == split] for split in ("train", "val")]
        chosen = []
        for i, group in enumerate(groups):
            count = min(len(group), max_chunks // 2 + (max_chunks % 2 if i == 0 else 0))
            if count:
                chosen += [group[j] for j in np.linspace(0, len(group) - 1, count, dtype=int)]
        selected = sorted(chosen, key=lambda c: c["id"])
    if len(selected) < num_shards:
        raise ValueError("More shards than selected chunks")
    assigned = []
    for shard, positions in enumerate(np.array_split(np.arange(len(selected)), num_shards)):
        assigned += [{**selected[int(i)], "shard": shard} for i in positions]
    return assigned


def load_context(config_path: Path):
    from .training.config import load_training_config
    from .training.data import open_training_data
    from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats, validate_stats_coverage
    from .model import build_model_configs

    config = load_training_config(config_path)
    if config.feedback_mode != "baseline":
        raise ValueError("Baseline caching requires feedback_mode='baseline'")
    # Cache all leads regardless of a later consumer's loss selection.
    config = dataclasses.replace(config, loss_mode="all_steps", supervised_horizons=(), supervised_weights=())
    checkpoint = load_graphcast_checkpoint(config.baseline_checkpoint)
    task = dataclasses.replace(checkpoint.task_config, input_duration=config.input_duration)
    stats = load_stats(config.stats_dir)
    validate_stats_coverage(task, stats)
    data = open_training_data(config, task)
    if data.input_steps != 2:
        raise ValueError("v24 cache requires two input frames")
    model = build_model_configs(checkpoint.model_config, config.architecture).baseline
    return config, checkpoint, task, stats, data, model


def build_manifest(context, num_shards: int, max_chunks: int | None = None) -> dict:
    config, checkpoint, task, stats, data, model = context
    from .checkpoint import manifest_sha256
    chunks, excluded = enumerate_chunks(data, config)
    selected = select_chunks(chunks, max_chunks, num_shards)
    variables = {}
    for name in task.target_variables:
        source = data.store.data_vars[name]
        dims = [d for d in source.dims if d != "time"]
        variables[name] = {"dims": dims, "shape": [int(source.data.shape[source.dims.index(d)]) for d in dims], "dtype": "float32"}
    coordinates = {
        d: np.asarray(data.store.coords[d]).tolist()
        for d in sorted({d for v in variables.values() for d in v["dims"]})
    }
    prepared_files = {}
    for path in sorted(config.prepared_root.rglob("*.npy")):
        stat = path.stat()
        prepared_files[str(path.relative_to(config.prepared_root))] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    repo = Path(__file__).resolve().parents[4]
    code_paths = [
        Path(__file__), repo / "scripts/preprocessing/precompute_v24_baseline.py",
        repo / "src/models/mamba/v24_Ilya/model.py",
        repo / "src/models/mamba/v24_Ilya/training/endpoint_step.py",
        repo / "src/models/mamba/v24_Ilya/training/frame_data.py",
        repo / "src/models/mamba/v24_Ilya/training/data.py",
    ]
    # Include the GraphCast implementation and normalizers, not only the wrapper.
    import inspect
    from graphcast import graphcast, normalization
    code_paths += [Path(inspect.getfile(graphcast)), Path(inspect.getfile(normalization)),
                   repo / "src/models/graphcast/training/core/model.py"]
    compatibility = {
        "baseline_sha256": sha256(config.baseline_checkpoint),
        "baseline_model": dataclasses.asdict(model),
        "task": dataclasses.asdict(task),
        "stats_sha256": {p.name: sha256(p) for p in sorted(config.stats_dir.glob("*.nc"))},
        "source_manifest_sha256": manifest_sha256(config.anchor_manifest_root),
        "prepared_metadata_sha256": sha256(config.prepared_root / "metadata.json"),
        "prepared_files": prepared_files,
        "coordinate_sha256": {p.name: sha256(p) for p in sorted((config.prepared_root / "coords").glob("*.npy"))},
        "source_selection": data.store.selection_metadata,
        "precision": config.precision, "physical_dtype": "float32",
        "gpu_deterministic_ops": True,
        "segment_steps": config.segment_steps, "bptt_steps": config.bptt_steps,
        "truth_prefix_steps": config.truth_prefix_steps, "ar_tail_k": config.ar_tail_k,
        "feedback_mode": "baseline", "input_steps": data.input_steps,
        "time_step_ns": int(data.time_step.value),
        "generation_code_sha256": {str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p): sha256(p) for p in code_paths},
    }
    # Canonical JSON representation also converts checkpoint tuples to lists.
    compatibility = json.loads(json.dumps(compatibility))
    manifest = {
        "format": FORMAT, "compatibility": compatibility,
        "compatibility_sha256": digest(compatibility),
        "source_paths": {"prepared_root": str(config.prepared_root.resolve()), "baseline_checkpoint": str(config.baseline_checkpoint.resolve()), "anchor_manifest_root": str(config.anchor_manifest_root.resolve())},
        "variables": variables, "coordinates": coordinates,
        "num_shards": num_shards, "chunks": selected, "excluded_tail_anchors": excluded,
        "full_chunk_count": len(chunks), "partial": len(selected) != len(chunks),
    }
    manifest["prediction_bytes"] = sum(int(np.prod(v["shape"])) * 4 for v in variables.values()) * config.bptt_steps * len(selected)
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def host_dataset(ds):
    from graphcast import xarray_jax
    return xr.Dataset({name: xr.DataArray(np.asarray(xarray_jax.unwrap_data(v), dtype=np.float32), dims=v.dims, coords=v.coords) for name, v in ds.data_vars.items()}, coords=ds.coords)


def make_predictor(context):
    import haiku as hk
    import jax
    from .model import make_baseline_predictor
    from .checkpoint import overlay_frozen_baseline_params
    config, checkpoint, task, stats, data, model = context

    def fn(inputs, targets, forcings):
        return make_baseline_predictor(model, task, stats, use_bf16=config.precision == "bf16")(
            inputs, targets_template=targets, forcings=forcings)
    transform = hk.transform_with_state(fn)
    anchor = int(data.anchor_indices[data.train_split[0]])
    sample = data.store.build_batch_from_indices(indices=[anchor], input_steps=2, target_steps=1, task_cfg=task, dt=data.time_step)
    sample = tuple(host_dataset(ds) for ds in sample)
    initialized, state = transform.init(jax.random.PRNGKey(0), *sample)
    params, _ = overlay_frozen_baseline_params(initialized, checkpoint.params)
    if jax.tree_util.tree_leaves(state):
        raise ValueError("Frozen baseline unexpectedly has recurrent state")

    @jax.jit
    def predict(inputs, targets, forcing):
        # Same stop-gradient and one-step JIT boundary as make_train_step.
        prediction, _ = transform.apply(params, state, jax.random.PRNGKey(0), jax.tree_util.tree_map(jax.lax.stop_gradient, inputs), targets, forcing)
        return jax.tree_util.tree_map(jax.lax.stop_gradient, prediction)
    return predict


def load_chunk(context, item):
    config, _, task, _, data, _ = context
    segments = data.segments if item["split"] == "train" else data.validation_segments
    chunk = data.load_segment_chunk(segments[item["segment_id"]], item["segment_offset"], config, task)
    if chunk.raw_anchor_indices.tolist() != item["raw_anchor_indices"]:
        raise ValueError("Source chunk no longer matches cache manifest")
    return chunk


def iter_trajectory(chunk, predict, truth_prefix_steps, time_step):
    """Host-orchestrated baseline-only counterpart of the training forward pass."""
    from .training.endpoint_step import input_window_from_frames, next_dynamic_frame
    frames = list(chunk.input_frames)
    n = len(chunk.forcings)
    for i in range(n):
        inputs = input_window_from_frames(frames[i], frames[i + 1], chunk.static_inputs, step_index=i, truth_prefix_steps=truth_prefix_steps, time_step=time_step)
        prediction = host_dataset(predict(inputs, chunk.truths[i], chunk.forcings[i]))
        yield i, inputs, prediction
        if truth_prefix_steps - 1 <= i < n - 1:
            frames.append(host_dataset(next_dynamic_frame(inputs, prediction, chunk.forcings[i], time_step=time_step)))


def pack_prediction(prediction, name, spec):
    value = prediction[name]
    if value.sizes.get("batch") != 1 or value.sizes.get("time") != 1:
        raise ValueError("Expected single-sample single-step predictions")
    values = np.asarray(value.isel(batch=0, time=0).transpose(*spec["dims"]).data, dtype=np.float32)
    if list(values.shape) != spec["shape"] or not np.isfinite(values).all():
        raise ValueError(f"Invalid prediction shape/values for {name}")
    return values


def prediction_dataset(arrays, step, manifest, template):
    return xr.Dataset({
        name: xr.DataArray(np.asarray(arrays[name][step])[None, None],
                           dims=("batch", "time", *spec["dims"]),
                           coords={"batch": template.batch.values, "time": template.time.values,
                                   **{d: manifest["coordinates"][d] for d in spec["dims"]}})
        .transpose(*template[name].dims)
        for name, spec in manifest["variables"].items()
    })


def validate_manifest(manifest):
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported baseline cache format")
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(payload) != manifest.get("manifest_sha256") or digest(manifest["compatibility"]) != manifest["compatibility_sha256"]:
        raise ValueError("Manifest fingerprint mismatch")
    ids = [c["id"] for c in manifest["chunks"]]
    targets = [(c["split"], t) for c in manifest["chunks"] for t in c["target_timestamps"]]
    if len(ids) != len(set(ids)) or len(targets) != len(set(targets)):
        raise ValueError("Duplicate chunks or target timestamps")
    n = manifest["compatibility"]["bptt_steps"]
    for c in manifest["chunks"]:
        if len(c["raw_anchor_indices"]) != n or len(c["target_timestamps"]) != n:
            raise ValueError("Incomplete manifest chunk")
        if not 0 <= c["shard"] < manifest["num_shards"]:
            raise ValueError("Invalid shard assignment")


def verify_shard(root: Path, manifest: dict, shard: int, *, directory: Path | None = None, checksums=True):
    directory = directory or root / f"shard_{shard:03d}"
    marker = json.loads((directory / "COMPLETE.json").read_text())
    expected = [c for c in manifest["chunks"] if c["shard"] == shard]
    if marker["manifest_sha256"] != manifest["manifest_sha256"] or marker["chunk_ids"] != [c["id"] for c in expected]:
        raise ValueError("Shard completion metadata mismatch")
    expected_files = {f"chunk_{c['id']:06d}/{name}.npy" for c in expected for name in manifest["variables"]}
    if set(marker["files"]) != expected_files:
        raise ValueError("Shard file coverage mismatch")
    steps = manifest["compatibility"]["bptt_steps"]
    for rel, saved in marker["files"].items():
        path = directory / rel
        if checksums and sha256(path) != saved["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        name = path.stem
        if arr.dtype != np.float32 or list(arr.shape) != [steps, *manifest["variables"][name]["shape"]]:
            raise ValueError(f"Array schema mismatch: {path}")
        if checksums:
            for frame in arr:
                if not np.isfinite(frame).all():
                    raise ValueError(f"Nonfinite cache values: {path}")
    return marker


def verify_cache(root: Path, *, allow_partial=False):
    (root / "READY.json").unlink(missing_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest["partial"] and not allow_partial:
        raise ValueError("A pilot/subset cache cannot be marked as the full dataset")
    if not manifest["partial"] and len(manifest["chunks"]) != manifest["full_chunk_count"]:
        raise ValueError("Missing full-dataset chunks")
    markers = [verify_shard(root, manifest, s) for s in range(manifest["num_shards"])]
    ready = {"manifest_sha256": manifest["manifest_sha256"], "chunks": len(manifest["chunks"]), "predictions": sum(len(c["raw_anchor_indices"]) for c in manifest["chunks"]), "shards": len(markers), "partial": manifest["partial"]}
    atomic_json(root / "READY.json", ready)
    return ready


class BaselineCacheReader:
    """Read complete shards; production consumers require the global READY marker.

    Returned generators must be consumed before loading the next source chunk:
    the source loader deliberately reuses its owned truth buffers.
    """
    def __init__(self, root: Path, expected_compatibility: str, *, require_ready=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        validate_manifest(self.manifest)
        if self.manifest["compatibility_sha256"] != expected_compatibility:
            raise ValueError("Incompatible baseline cache")
        if require_ready:
            ready = json.loads((self.root / "READY.json").read_text())
            if ready["manifest_sha256"] != self.manifest["manifest_sha256"] or ready["partial"]:
                raise ValueError("Cache is not a complete production dataset")
        self.items = {c["id"]: c for c in self.manifest["chunks"]}
        self.verified_shards = set()

    def arrays(self, chunk_id):
        item = self.items[chunk_id]
        shard = item["shard"]
        if shard not in self.verified_shards:
            verify_shard(self.root, self.manifest, shard, checksums=False)
            self.verified_shards.add(shard)
        directory = self.root / f"shard_{shard:03d}" / f"chunk_{chunk_id:06d}"
        return {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in self.manifest["variables"]}

    def iter_chunk(self, chunk_id, chunk, time_step):
        from .training.endpoint_step import input_window_from_frames, next_dynamic_frame
        if chunk.raw_anchor_indices.tolist() != self.items[chunk_id]["raw_anchor_indices"]:
            raise ValueError("Reader source chunk indices do not match")
        arrays = self.arrays(chunk_id)
        frames = list(chunk.input_frames)
        prefix = self.manifest["compatibility"]["truth_prefix_steps"]
        for i, truth in enumerate(chunk.truths):
            inputs = input_window_from_frames(frames[i], frames[i + 1], chunk.static_inputs, step_index=i, truth_prefix_steps=prefix, time_step=time_step)
            baseline = prediction_dataset(arrays, i, self.manifest, truth)
            residual = truth.astype(np.float32) - baseline
            yield inputs, baseline, residual, chunk.forcings[i]
            if prefix - 1 <= i < len(chunk.truths) - 1:
                frames.append(host_dataset(next_dynamic_frame(inputs, baseline, chunk.forcings[i], time_step=time_step)))


def normalized_error(actual, expected, scales):
    maximum = 0.0
    if set(actual.data_vars) != set(expected.data_vars):
        raise AssertionError("Parity variable mismatch")
    for name in actual.data_vars:
        a, b = actual[name], expected[name]
        if a.dims != b.dims or a.shape != b.shape:
            raise AssertionError(f"Parity schema mismatch for {name}")
        for d in a.dims:
            np.testing.assert_array_equal(a.coords[d], b.coords[d])
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if name in scales:
            scale = np.abs(scales[name])
            if bool((scale <= 0).any()):
                raise ValueError(f"Invalid normalization scale: {name}")
            delta = delta / scale
        error = float(delta.max())
        if not np.isfinite(error):
            raise AssertionError(f"Nonfinite parity result: {name}")
        maximum = max(maximum, error)
    return maximum


def verify_pilot_parity(context, manifest, root):
    """Independent online windows use the prepared batch builder and xr.concat.

    The prediction call is the maintained training baseline transform. No Mamba
    model is initialized. This checks forcing/time alignment independently of
    the cache frame-reconstruction helper, over every pilot lead.
    """
    import jax
    from .training.endpoint_step import build_training_transforms
    config, checkpoint, task, stats, data, model = context
    reader = BaselineCacheReader(root, manifest["compatibility_sha256"], require_ready=False)
    max_errors = {"inputs": 0.0, "predictions": 0.0, "residuals": 0.0}
    # Reuse the exact training transform in a separately initialized predictor;
    # constructing transforms defines residual closures but never invokes them.
    from .model import V24IlyaModelConfigs
    from .checkpoint import overlay_frozen_baseline_params
    transforms = build_training_transforms(V24IlyaModelConfigs(baseline=model, residual=model), task, stats, config)
    baseline = transforms.baseline_predict
    sample_anchor = int(data.anchor_indices[data.train_split[0]])
    sample = tuple(host_dataset(ds) for ds in data.store.build_batch_from_indices(indices=[sample_anchor], input_steps=2, target_steps=1, task_cfg=task, dt=data.time_step))
    init, state = baseline.init(jax.random.PRNGKey(7), *sample)
    params, _ = overlay_frozen_baseline_params(init, checkpoint.params)

    @jax.jit
    def online(inputs, truth, forcing):
        prediction, _ = baseline.apply(params, state, jax.random.PRNGKey(19), jax.tree_util.tree_map(jax.lax.stop_gradient, inputs), truth, forcing)
        return jax.tree_util.tree_map(jax.lax.stop_gradient, prediction)

    for item in manifest["chunks"]:
        chunk = load_chunk(context, item)
        current = None
        for i, (cached_inputs, saved, residual, forcing) in enumerate(reader.iter_chunk(item["id"], chunk, data.time_step)):
            if i < config.truth_prefix_steps:
                current, _, _ = data.store.build_batch_from_indices(indices=[item["raw_anchor_indices"][i]], input_steps=2, target_steps=1, task_cfg=task, dt=data.time_step)
                current = host_dataset(current)
            predicted = host_dataset(online(current, chunk.truths[i], forcing))
            comparisons = {
                "inputs": (cached_inputs, current, stats["stddev_by_level"]),
                "predictions": (saved, predicted, stats["diffs_stddev_by_level"]),
                "residuals": (residual, chunk.truths[i].astype(np.float32) - predicted, stats["diffs_stddev_by_level"]),
            }
            for key, (actual, expected, scales) in comparisons.items():
                max_errors[key] = max(max_errors[key], normalized_error(actual, expected, scales))
            if config.truth_prefix_steps - 1 <= i < config.bptt_steps - 1:
                new_time = current.time.values[-1:] + np.timedelta64(int(data.time_step.value), "ns")
                following = xr.merge([predicted.assign_coords(time=new_time), forcing.assign_coords(time=new_time)])
                names = [name for name, v in current.data_vars.items() if "time" in v.dims]
                dynamic = xr.concat([current[names], following[names]], dim="time", data_vars="all", coords="minimal", compat="override").isel(time=slice(-2, None))
                static = current.drop_vars(names)
                current = xr.merge([dynamic, static.drop_dims("time", errors="ignore")])
        print(f"parity chunk={item['id']} max_normalized_errors={max_errors}", flush=True)
    if max(max_errors.values()) > 1e-5:
        raise AssertionError(f"Pilot normalized error exceeds 1e-5: {max_errors}")
    return {"passed": True, "tolerance": 1e-5, "max_normalized_errors": max_errors, "chunks": len(manifest["chunks"])}


def finish_pilot(context, manifest, root, marker, started):
    if not marker.get("exact_roundtrip_verified", False):
        raise ValueError("Pilot lacks verified original-prediction disk round trips")
    result = verify_pilot_parity(context, manifest, root)
    timings = marker["timings"]
    steady = max(t["seconds"] for t in timings[1:]) if len(timings) > 1 else timings[0]["seconds"]
    report = {**result, "compatibility_sha256": manifest["compatibility_sha256"],
              "manifest_sha256": manifest["manifest_sha256"],
              "initialization_seconds": marker["initialization_seconds"],
              "compilation_seconds_estimate": max(0.0, timings[0]["seconds"] - steady),
              "steady_chunk_seconds": steady, "timings": timings,
              "peak_host_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "elapsed_seconds": time.monotonic() - started}
    verify_cache(root, allow_partial=True)
    atomic_json(root / "pilot_report.json", report)
    print(json.dumps(report), flush=True)


def generate_shard(context, manifest, root: Path, shard: int, *, resume=False, parity=False):
    if not 0 <= shard < manifest["num_shards"]:
        raise ValueError("shard-index outside num-shards")
    if parity and (manifest["num_shards"] != 1 or not manifest["partial"]):
        raise ValueError("Pilot parity requires a single-shard partial cache")
    validate_manifest(manifest)
    ensure_manifest(root, manifest)
    final = root / f"shard_{shard:03d}"
    temp = root / f".shard_{shard:03d}.incomplete"
    if final.exists():
        if not resume:
            raise FileExistsError(f"Shard already exists: {final}")
        marker = verify_shard(root, manifest, shard)
        report_path = root / "pilot_report.json"
        if not parity or report_path.exists():
            if parity:
                report = json.loads(report_path.read_text())
                if report.get("passed") is not True or report["manifest_sha256"] != manifest["manifest_sha256"]:
                    raise ValueError("Invalid saved pilot parity report")
            print(f"Verified completed shard {shard}; skipping", flush=True)
            return
        started = time.monotonic()
        finish_pilot(context, manifest, root, marker, started)
        return
    # An advisory lock releases on process death and prevents duplicate writers.
    import fcntl
    with (root / f".shard_{shard:03d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.exists():
            raise FileExistsError(final)
        if temp.exists():
            if not resume:
                raise FileExistsError(f"Incomplete shard exists; use --resume: {temp}")
            shutil.rmtree(temp)
        temp.mkdir()
        config, _, _, _, data, _ = context
        started = time.monotonic()
        predict = make_predictor(context)
        initialization_seconds = time.monotonic() - started
        items = [c for c in manifest["chunks"] if c["shard"] == shard]
        timings, files = [], {}
        for item in items:
            chunk_start = time.monotonic()
            chunk = load_chunk(context, item)
            directory = temp / f"chunk_{item['id']:06d}"
            directory.mkdir()
            arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=np.float32, shape=(config.bptt_steps, *spec["shape"])) for name, spec in manifest["variables"].items()}
            originals = {name: np.empty(array.shape, dtype=np.float32) for name, array in arrays.items()} if parity else None
            write_seconds = 0.0
            for i, inputs, prediction in iter_trajectory(chunk, predict, config.truth_prefix_steps, data.time_step):
                write_start = time.monotonic()
                for name, spec in manifest["variables"].items():
                    values = pack_prediction(prediction, name, spec)
                    arrays[name][i] = values
                    if originals is not None:
                        originals[name][i] = values
                write_seconds += time.monotonic() - write_start
            write_start = time.monotonic()
            for array in arrays.values():
                array.flush()
            arrays.clear()
            for name in manifest["variables"]:
                path = directory / f"{name}.npy"
                with path.open("rb") as f:
                    os.fsync(f.fileno())
                if originals is not None:
                    saved = np.load(path, mmap_mode="r", allow_pickle=False)
                    np.testing.assert_array_equal(saved, originals[name], err_msg=f"Disk round trip: chunk={item['id']} variable={name}")
                files[str(path.relative_to(temp))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            del originals
            write_seconds += time.monotonic() - write_start
            record = {"chunk_id": item["id"], "seconds": time.monotonic() - chunk_start, "write_seconds": write_seconds}
            timings.append(record)
            print(json.dumps(record), flush=True)
        marker = {"manifest_sha256": manifest["manifest_sha256"], "chunk_ids": [c["id"] for c in items], "files": files, "initialization_seconds": initialization_seconds, "timings": timings, "exact_roundtrip_verified": parity}
        atomic_json(temp / "COMPLETE.json", marker)
        verify_shard(root, manifest, shard, directory=temp)
        os.rename(temp, final)
        if parity:
            finish_pilot(context, manifest, root, marker, started)
