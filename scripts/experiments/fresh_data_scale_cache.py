"""Extend the seven-year K=20 cache while preserving every overlapping chunk.

The normal cache manifest and consumer remain unchanged. Historical anchors are
truncated to complete segments before appending the original training split.
After complete bitwise overlap verification, existing prediction files are hard
linked into ordinary production shards; only earlier chunks run GraphCast.
When a complete historical cache is supplied, those chunks are linked too:
the aligned cache then needs neither prediction generation nor duplicate data.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from src.models.mamba.v24_Ilya import baseline_cache as cache


PROOF_FORMAT = "fresh_data_scale_aligned_cache_v1"
SEMANTIC_KEYS = (
    "baseline_sha256", "baseline_model", "task", "stats_sha256", "precision",
    "physical_dtype", "gpu_deterministic_ops", "segment_steps", "bptt_steps",
    "truth_prefix_steps", "ar_tail_k", "feedback_mode", "input_steps", "time_step_ns",
)
PREPARED_IDENTITY_KEYS = (
    "prepared_metadata_sha256", "prepared_files", "coordinate_sha256", "source_selection",
)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def read_source_cache(root: Path) -> dict:
    root = Path(root)
    manifest = read(root / "manifest.json")
    cache.validate_manifest(manifest)
    ready = read(root / "READY.json")
    required = dict(manifest_sha256=manifest["manifest_sha256"],
                    chunks=len(manifest["chunks"]),
                    predictions=sum(len(c["raw_anchor_indices"]) for c in manifest["chunks"]),
                    shards=manifest["num_shards"], partial=False)
    if manifest["partial"] or any(ready.get(k) != v for k, v in required.items()):
        raise ValueError("Source cache must be a complete verified production cache")
    return manifest


def aligned_anchor_arrays(full_times, old_times, old_arrays, *, input_steps=2, segment_steps=96):
    """Map original anchors by time, adding only complete earlier segments."""
    full_times = np.asarray(full_times).astype("datetime64[ns]")
    old_times = np.asarray(old_times).astype("datetime64[ns]")
    if full_times.ndim != 1 or old_times.ndim != 1 or segment_steps <= 0:
        raise ValueError("Invalid clock/segment shape")
    for values in (full_times, old_times):
        if len(values) < 4 or not np.all(np.diff(values) == np.timedelta64(6, "h")):
            raise ValueError("Expected complete six-hour clocks")
    mapped_time = np.searchsorted(full_times, old_times)
    if np.any(mapped_time >= len(full_times)) or not np.array_equal(full_times[mapped_time], old_times):
        raise ValueError("Full prepared clock does not contain the original clock exactly")
    old_raw = np.asarray(old_arrays["anchor_indices"], dtype=np.int64)
    if np.any(old_raw < input_steps) or np.any(old_raw >= len(old_times) - 2):
        raise ValueError("Original anchors violate the source clock safety margin")
    if not np.array_equal(old_arrays["anchor_times"], old_times[old_raw]):
        raise ValueError("Original anchor times differ from its prepared clock")
    full_raw = np.arange(input_steps, len(full_times) - 2, dtype=np.int64)
    mapped_raw = mapped_time[old_raw]
    mapped_positions = mapped_raw - input_steps
    old_train = np.asarray(old_arrays["split_train"], dtype=np.int64)
    old_val = np.asarray(old_arrays["split_val"], dtype=np.int64)
    if not len(old_train) or not len(old_val):
        raise ValueError("Original train and validation splits must both be nonempty")
    for split in (old_train, old_val):
        if np.any(split < 0) or np.any(split >= len(old_raw)) or np.any(np.diff(split) <= 0):
            raise ValueError("Invalid original split indices")
    earlier = np.flatnonzero(full_times[full_raw] < old_times[0])
    retained = len(earlier) // segment_steps * segment_steps
    train = np.concatenate([earlier[:retained], mapped_positions[old_train]])
    val = mapped_positions[old_val]
    if np.intersect1d(train, val).size or np.any(np.diff(train) <= 0):
        raise ValueError("Aligned train/validation splits overlap or are unordered")
    for split in (train, val):
        for start in range(0, len(split) - segment_steps + 1, segment_steps):
            if not np.all(np.diff(full_raw[split[start:start + segment_steps]]) == 1):
                raise ValueError("Aligned segment crosses a discontinuity")
    arrays = dict(anchor_indices=full_raw, anchor_times=full_times[full_raw], split_train=train, split_val=val)
    info = dict(historical_segments=retained // segment_steps, historical_anchors=retained,
                omitted_boundary_anchor_indices=full_raw[earlier[retained:]].tolist(),
                overlap_raw_index_offset=int(mapped_time[0]),
                original_train_anchors=len(old_train), original_validation_anchors=len(old_val))
    return arrays, info


def _array_values(data, indices, axis):
    # PreparedFileArray/PreparedShardedArray take bounded owned selections.
    if isinstance(data, np.ndarray):
        return np.take(data, indices, axis=axis)
    return data.take(indices, axis=axis)


def _same_bits(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def verify_prepared_overlap(full_store, old_store, *, block_steps=8) -> dict:
    """Check every shared input, target, forcing and static value, not a sample."""
    if block_steps < 1:
        raise ValueError("block_steps must be positive")
    full_times = np.asarray(full_store.time.values).astype("datetime64[ns]")
    old_times = np.asarray(old_store.time.values).astype("datetime64[ns]")
    mapped = np.searchsorted(full_times, old_times)
    if np.any(mapped >= len(full_times)) or not np.array_equal(full_times[mapped], old_times):
        raise ValueError("Prepared overlap timestamps differ")
    if set(full_store.coords) != set(old_store.coords):
        raise ValueError("Prepared coordinate names differ")
    for name in old_store.coords:
        if name == "time":
            continue
        a, b = np.asarray(full_store.coords[name]), np.asarray(old_store.coords[name])
        if not _same_bits(a, b):
            raise ValueError(f"Prepared coordinate differs: {name}")
    if set(full_store.data_vars) != set(old_store.data_vars):
        raise ValueError("Prepared variable coverage differs")
    checked = {}
    for name, original in old_store.data_vars.items():
        expanded = full_store.data_vars[name]
        if original.dims != expanded.dims or original.data.dtype != expanded.data.dtype:
            raise ValueError(f"Prepared variable schema differs: {name}")
        digest = hashlib.sha256()
        elements = 0
        if "time" in original.dims:
            axis = original.dims.index("time")
            for start in range(0, len(old_times), block_steps):
                positions = np.arange(start, min(start + block_steps, len(old_times)))
                a = _array_values(original.data, positions, axis)
                b = _array_values(expanded.data, mapped[positions], axis)
                if not _same_bits(a, b) or not np.isfinite(a).all():
                    raise ValueError(f"Prepared overlap values differ: {name}, original frames {start}:{start + len(positions)}")
                digest.update(np.ascontiguousarray(a).tobytes())
                elements += a.size
        else:
            a, b = np.asarray(original.data), np.asarray(expanded.data)
            if not _same_bits(a, b) or not np.isfinite(a).all():
                raise ValueError(f"Prepared static field differs: {name}")
            digest.update(np.ascontiguousarray(a).tobytes())
            elements = a.size
        checked[name] = dict(elements=int(elements), overlap_sha256=digest.hexdigest())
        print(json.dumps(dict(overlap_variable=name, **checked[name])), flush=True)
    return dict(passed=True, bitwise_equal=True, time_steps=len(old_times),
                time_start=str(old_times[0]), time_end=str(old_times[-1]), variables=checked)


def match_source_chunks(manifest, source) -> dict[int, dict]:
    """Match entire windows and memory positions, ignoring dataset index offsets."""
    for key in SEMANTIC_KEYS:
        if manifest["compatibility"].get(key) != source["compatibility"].get(key):
            raise ValueError(f"Cache numerical semantics differ: {key}")
    if manifest["variables"] != source["variables"] or manifest["coordinates"] != source["coordinates"]:
        raise ValueError("Cache variable/coordinate schemas differ")
    key = lambda c: (c["split"], tuple(c["target_timestamps"]))
    originals = {key(c): c for c in source["chunks"]}
    if len(originals) != len(source["chunks"]):
        raise ValueError("Original cache has duplicate windows")
    result = {}
    for item in manifest["chunks"]:
        old = originals.get(key(item))
        if old is None:
            continue
        if item["segment_offset"] != old["segment_offset"] or item["chunk_index"] != old["chunk_index"]:
            raise ValueError("Shared cache windows moved within a recurrent segment")
        result[item["id"]] = old
    if len(result) != len(source["chunks"]):
        raise ValueError("Aligned cache does not preserve every original train/validation chunk")
    for item in manifest["chunks"]:
        if item["id"] not in result:
            if item["split"] != "train" or any(int(t[:4]) >= 2015 or int(t[:4]) < 1982 for t in item["target_timestamps"]):
                raise ValueError("New cache chunks must contain only 1982–2014 training targets")
    return result


def match_historical_chunks(manifest, historical, recent_chunks) -> dict[int, dict]:
    """Import earlier windows only from the identical full prepared source."""
    for key in (*SEMANTIC_KEYS, *PREPARED_IDENTITY_KEYS):
        if (key not in manifest["compatibility"] or key not in historical["compatibility"]
                or manifest["compatibility"][key] != historical["compatibility"][key]):
            raise ValueError(f"Historical cache source/numerical identity differs: {key}")
    if manifest["variables"] != historical["variables"] or manifest["coordinates"] != historical["coordinates"]:
        raise ValueError("Historical cache variable/coordinate schemas differ")
    if (Path(manifest["source_paths"]["prepared_root"]).resolve()
            != Path(historical["source_paths"]["prepared_root"]).resolve()):
        raise ValueError("Historical cache must use the identical prepared root")
    key = lambda c: (c["split"], tuple(c["target_timestamps"]))
    originals = {key(c): c for c in historical["chunks"]}
    if len(originals) != len(historical["chunks"]):
        raise ValueError("Historical cache has duplicate windows")
    result = {}
    for item in manifest["chunks"]:
        if item["id"] in recent_chunks:
            continue
        old = originals.get(key(item))
        if old is None:
            raise ValueError(f"Historical source lacks aligned chunk {item['id']}")
        for field in ("raw_anchor_indices", "segment_id", "segment_offset", "chunk_index"):
            if item[field] != old[field]:
                raise ValueError(f"Historical cache window/segment identity differs: {field}")
        result[item["id"]] = old
    return result


def _require_prepared_ready(root, source):
    root = Path(root)
    ready = read(root / "READY.json")
    if ((root / ".incomplete").exists() or ready.get("status") != "complete"
            or ready.get("years") != list(range(1982, 2023))
            or ready.get("metadata_sha256") != cache.sha256(root / "metadata.json")
            or ready.get("identity", {}).get("checkpoint_sha256") != source["compatibility"]["baseline_sha256"]):
        raise ValueError("Aligned cache requires the complete matching 1982–2022 prepared store")
    for relative, expected in ready.get("root_file_sha256", {}).items():
        if cache.sha256(root / relative) != expected:
            raise ValueError(f"Prepared READY root fingerprint changed: {relative}")
    return ready


def _prepare_anchors(common, source):
    from src.models.mamba.v24_Ilya.anchor_manifest import build_anchor_manifest
    from src.models.mamba.v24_Ilya.checkpoint import manifest_sha256
    old_root = Path(source["source_paths"]["anchor_manifest_root"])
    if manifest_sha256(old_root) != source["compatibility"]["source_manifest_sha256"]:
        raise ValueError("Original anchor manifest fingerprint changed")
    old_prepared = Path(source["source_paths"]["prepared_root"])
    old_times = np.load(old_prepared / "coords/time.npy", allow_pickle=False)
    if cache.sha256(old_prepared / "metadata.json") != source["compatibility"]["prepared_metadata_sha256"]:
        raise ValueError("Original prepared metadata fingerprint changed")
    full_times = np.load(common.prepared_root / "coords/time.npy", allow_pickle=False)
    for times, start, end in ((full_times, "1982-01-01", "2023-01-01"),
                              (old_times, "2015-01-01", "2023-01-01")):
        expected = np.arange(np.datetime64(start, "h"), np.datetime64(end, "h"), np.timedelta64(6, "h"))
        if not np.array_equal(times, expected):
            raise ValueError("Data-scale campaign requires the exact 1982–2022 and 2015–2022 clocks")
    old_arrays = {name: np.load(old_root / "anchors" / f"{name}.npy", allow_pickle=False)
                  for name in ("anchor_indices", "anchor_times", "split_train", "split_val")}
    arrays, alignment = aligned_anchor_arrays(full_times, old_times, old_arrays,
        input_steps=source["compatibility"]["input_steps"], segment_steps=common.segment_steps)
    target = common.anchor_manifest_root
    identity = dict(format=PROOF_FORMAT, source_cache_manifest_sha256=source["manifest_sha256"],
                    prepared_metadata_sha256=cache.sha256(common.prepared_root / "metadata.json"),
                    **alignment)
    if target.exists():
        metadata = read(target / "metadata.json")
        if (target / ".incomplete").exists() or metadata.get("data_scale_alignment") != identity:
            raise ValueError("Existing aligned anchor provenance differs")
        for name, expected in arrays.items():
            if not np.array_equal(np.load(target / "anchors" / f"{name}.npy", allow_pickle=False), expected):
                raise ValueError(f"Existing aligned anchor array differs: {name}")
        return identity
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".building-" + uuid.uuid4().hex)
    metadata = build_anchor_manifest(prepared_root=common.prepared_root,
        baseline_checkpoint=common.baseline_checkpoint, output_root=staging)
    (staging / ".incomplete").write_text("aligning source cache segments\n")
    for name, values in arrays.items():
        np.save(staging / "anchors" / f"{name}.npy", values, allow_pickle=False)
    metadata.update(n_anchors_train=len(arrays["split_train"]), n_anchors_val=len(arrays["split_val"]),
                    data_scale_alignment=identity)
    cache.atomic_json(staging / "metadata.json", metadata)
    (staging / ".incomplete").unlink()
    staging.rename(target)
    return identity


def prepare_aligned_cache(config_path, source_cache_root, cache_root, *, num_shards=40,
                          historical_cache_root=None):
    """CPU gate: exact inputs, aligned anchors, standard manifest, import proof."""
    from src.models.graphcast.training.core.prepared_array import PreparedArrayStore
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    config_path, source_cache_root, cache_root = map(Path, (config_path, source_cache_root, cache_root))
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / ".alignment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = read_source_cache(source_cache_root)
        common = load_training_config(config_path)
        _require_prepared_ready(common.prepared_root, source)
        alignment = _prepare_anchors(common, source)
        context = cache.load_context(config_path)
        manifest = cache.build_manifest(context, num_shards)
        imported = match_source_chunks(manifest, source)
        historical = None
        historical_imports = {}
        if historical_cache_root is not None:
            historical_cache_root = Path(historical_cache_root)
            historical = read_source_cache(historical_cache_root)
            historical_imports = match_historical_chunks(manifest, historical, imported)
        identity = dict(format=PROOF_FORMAT, manifest_sha256=manifest["manifest_sha256"],
                        source_cache_root=str(source_cache_root.resolve()),
                        source_manifest_sha256=source["manifest_sha256"], alignment=alignment,
                        historical_cache_root=str(historical_cache_root.resolve()) if historical is not None else None,
                        historical_manifest_sha256=historical["manifest_sha256"] if historical is not None else None)
        proof_path = cache_root / "alignment.json"
        if proof_path.exists():
            proof = read(proof_path)
            if any(proof.get(k) != v for k, v in identity.items()) or not proof.get("overlap", {}).get("bitwise_equal"):
                raise ValueError("Existing cache overlap proof differs from current inputs")
        else:
            # Check actual original producer files before relying on their values.
            old_root = Path(source["source_paths"]["prepared_root"])
            for relative, expected in source["compatibility"]["prepared_files"].items():
                stat = (old_root / relative).stat()
                if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
                    raise ValueError(f"Original prepared source changed: {relative}")
            for name, expected in source["compatibility"]["coordinate_sha256"].items():
                if cache.sha256(old_root / "coords" / name) != expected:
                    raise ValueError(f"Original prepared coordinate changed: {name}")
            full_store = PreparedArrayStore(common.prepared_root, bounded_reads=True)
            old_store = PreparedArrayStore(old_root, bounded_reads=True)
            overlap = verify_prepared_overlap(full_store, old_store)
            proof = dict(**identity, passed=True, overlap=overlap, num_shards=num_shards,
                         imported_chunks=len(imported), historical_chunks=len(manifest["chunks"]) - len(imported),
                         imported_chunk_ids={str(k): v["id"] for k, v in imported.items()},
                         imported_historical_chunks=len(historical_imports),
                         generated_chunks=len(manifest["chunks"]) - len(imported) - len(historical_imports),
                         historical_chunk_ids={str(k): v["id"] for k, v in historical_imports.items()})
            cache.atomic_json(proof_path, proof)
        cache.ensure_manifest(cache_root, manifest)
        return proof


def _historical_manifest(manifest, items):
    subset = copy.deepcopy(manifest)
    subset["chunks"] = items
    subset["partial"] = True
    if "prediction_bytes" in subset:
        subset["prediction_bytes"] = manifest["prediction_bytes"] // len(manifest["chunks"]) * len(items)
    subset.pop("manifest_sha256")
    subset["manifest_sha256"] = cache.digest(subset)
    return subset


def generate_aligned_shard(config_path, source_cache_root, cache_root, shard, *, historical_cache_root=None):
    """Import both sources on CPU, or generate history when no cache is supplied."""
    config_path, source_cache_root, cache_root = map(Path, (config_path, source_cache_root, cache_root))
    source = read_source_cache(source_cache_root)
    manifest = read(cache_root / "manifest.json")
    context = cache.load_context(config_path)
    if cache.build_manifest(context, manifest["num_shards"]) != manifest:
        raise ValueError("Aligned cache producer inputs changed after preparation")
    proof = read(cache_root / "alignment.json")
    if (proof.get("manifest_sha256") != manifest["manifest_sha256"]
            or proof.get("source_manifest_sha256") != source["manifest_sha256"]
            or proof.get("source_cache_root") != str(source_cache_root.resolve())
            or not proof.get("overlap", {}).get("bitwise_equal")):
        raise ValueError("Missing exact prepared overlap proof")
    if not 0 <= shard < manifest["num_shards"]:
        raise ValueError("Invalid aligned shard index")
    imported = match_source_chunks(manifest, source)
    historical_source = None
    historical_imports = {}
    if historical_cache_root is not None:
        historical_cache_root = Path(historical_cache_root)
        historical_source = read_source_cache(historical_cache_root)
        historical_imports = match_historical_chunks(manifest, historical_source, imported)
    expected_historical = dict(
        historical_cache_root=str(historical_cache_root.resolve()) if historical_source is not None else None,
        historical_manifest_sha256=historical_source["manifest_sha256"] if historical_source is not None else None)
    if any(proof.get(k) != v for k, v in expected_historical.items()):
        raise ValueError("Historical cache differs from the pinned alignment proof")
    items = [c for c in manifest["chunks"] if c["shard"] == shard]
    historical = [c for c in items if c["id"] not in imported]
    final = cache_root / f"shard_{shard:03d}"
    with (cache_root / f".aligned_shard_{shard:03d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.exists():
            return dict(passed=True, shard=shard, resumed=True,
                        files=len(cache.verify_shard(cache_root, manifest, shard)["files"]))
        generated_root = cache_root / ".historical" / f"shard_{shard:03d}"
        generated_marker = None
        if historical and historical_source is None:
            subset = _historical_manifest(manifest, historical)
            cache.generate_shard(context, subset, generated_root, shard, resume=True)
            generated_marker = cache.verify_shard(generated_root, subset, shard)
        source_markers = {}
        for source_shard in sorted({imported[c["id"]]["shard"] for c in items if c["id"] in imported}):
            source_markers[source_shard] = cache.verify_shard(source_cache_root, source, source_shard, checksums=False)
        historical_markers = {}
        for source_shard in sorted({historical_imports[c["id"]]["shard"] for c in historical if c["id"] in historical_imports}):
            historical_markers[source_shard] = cache.verify_shard(
                historical_cache_root, historical_source, source_shard, checksums=False)
        staging = cache_root / f".aligned_shard_{shard:03d}.{uuid.uuid4().hex}.incomplete"
        staging.mkdir()
        files = {}
        for item in items:
            chunk_dir = staging / f"chunk_{item['id']:06d}"
            chunk_dir.mkdir()
            old = imported.get(item["id"])
            if old is not None:
                source_dir = source_cache_root / f"shard_{old['shard']:03d}" / f"chunk_{old['id']:06d}"
                marker = source_markers[old["shard"]]
                source_id = old["id"]
            elif item["id"] in historical_imports:
                old = historical_imports[item["id"]]
                source_dir = historical_cache_root / f"shard_{old['shard']:03d}" / f"chunk_{old['id']:06d}"
                marker = historical_markers[old["shard"]]
                source_id = old["id"]
            else:
                source_dir = generated_root / f"shard_{shard:03d}" / f"chunk_{item['id']:06d}"
                marker = generated_marker
                source_id = item["id"]
            for name in manifest["variables"]:
                os.link(source_dir / f"{name}.npy", chunk_dir / f"{name}.npy")
                files[f"chunk_{item['id']:06d}/{name}.npy"] = marker["files"][f"chunk_{source_id:06d}/{name}.npy"]
        marker = dict(manifest_sha256=manifest["manifest_sha256"], chunk_ids=[c["id"] for c in items],
                      files=files, imported_chunks=len(items) - len(historical), historical_chunks=len(historical),
                      imported_historical_chunks=len(historical) if historical_source is not None else 0,
                      generated_chunks=0 if historical_source is not None else len(historical),
                      source_manifest_sha256=source["manifest_sha256"],
                      historical_manifest_sha256=expected_historical["historical_manifest_sha256"],
                      overlap_proof_sha256=cache.sha256(cache_root / "alignment.json"))
        cache.atomic_json(staging / "COMPLETE.json", marker)
        cache.verify_shard(cache_root, manifest, shard, directory=staging)
        staging.rename(final)
        return dict(passed=True, shard=shard, imported_chunks=marker["imported_chunks"], historical_chunks=len(historical),
                    imported_historical_chunks=marker["imported_historical_chunks"], generated_chunks=marker["generated_chunks"])


def verify_aligned_cache(cache_root):
    cache_root = Path(cache_root)
    proof = read(cache_root / "alignment.json")
    manifest = read(cache_root / "manifest.json")
    if proof.get("manifest_sha256") != manifest["manifest_sha256"] or not proof.get("overlap", {}).get("bitwise_equal"):
        raise ValueError("Aligned cache overlap proof mismatch")
    ready = cache.verify_cache(cache_root)
    return dict(passed=True, **ready, imported_chunks=proof["imported_chunks"], historical_chunks=proof["historical_chunks"],
                imported_historical_chunks=proof.get("imported_historical_chunks", 0),
                generated_chunks=proof.get("generated_chunks", proof["historical_chunks"]),
                overlap_proof_sha256=cache.sha256(cache_root / "alignment.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "generate", "verify"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--historical-cache", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=40)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.command != "verify" and (args.config is None or args.source_cache is None):
        parser.error("prepare/generate require --config and --source-cache")
    if args.command == "prepare":
        result = prepare_aligned_cache(args.config, args.source_cache, args.cache_root, num_shards=args.num_shards,
                                       historical_cache_root=args.historical_cache)
    elif args.command == "generate":
        if args.shard is None:
            parser.error("generate requires --shard")
        result = generate_aligned_shard(args.config, args.source_cache, args.cache_root, args.shard,
                                        historical_cache_root=args.historical_cache)
    else:
        result = verify_aligned_cache(args.cache_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
