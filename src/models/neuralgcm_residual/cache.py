"""Independent truth-encoded k=1 records, bounded lossless shards and verification."""
from __future__ import annotations

from pathlib import Path
import pickle
import time
import zipfile
import zlib
import jax
import numpy as np
from .data import STEP, split_for, stamp
from .io import digest, sha256, versions, write_json, read_json
from .numerics import POLICY

PRODUCER_CONTRACT = {
    "forecast_step_hours": 6, "prediction_horizon_steps": 1,
    "origin_state_policy": "encode_truth_at_every_timestamp", "physical_feedback": "none",
    "forcing_policy": "lag24h_then_persist_for_one_forecast",
}


def producer_identity(backbone, adapter, store):
    files = ("backbone.py", "native_state.py", "data.py", "cache.py")
    root = Path(__file__).parent
    return {**PRODUCER_CONTRACT, "version": "neuralgcm_k1_v1",
            "checkpoint_sha256": backbone.checkpoint_sha256, "schema": adapter.schema,
            "native_schema_id": adapter.identity, "dataset_id": store.identity,
            "code": {p: sha256(root / p) for p in files}, "libraries": versions(),
            "storage": "lossless_deflate_numpy_pytree", "dtype": "float32", "numerical_policy": POLICY}


def valid_k1_origins(store, split):
    available = set(store.times)
    return [stamp(t) for t in store.times if split_for(t) == split
            and all(x in available and split_for(x) == split
                    for x in (t - np.timedelta64(24, "h"), t + STEP))]


def live_record(backbone, store, origin):
    t = np.datetime64(origin, "h")
    inputs, forcing = store.inputs_and_forcing(backbone.model, t)
    state = backbone.encode(inputs, forcing)
    predicted = backbone.advance_6h(state, forcing)
    backbone.assert_time(state, predicted)
    return {"origin": stamp(t), "valid": stamp(t + STEP), "split": split_for(t),
            "origin_state": jax.device_get(state), "baseline_state": jax.device_get(predicted),
            "forcing": jax.device_get(forcing), "target_time": stamp(t + STEP)}


def initialize_cache(root, producer, origins_by_split, shard_records=24):
    root = Path(root)
    if shard_records < 1:
        raise ValueError("shard_records must be positive")
    shards = []
    for split, origins in origins_by_split.items():
        if split not in ("train", "val") or not origins:
            raise ValueError("Cache requires nonempty train/validation origin lists")
        for i in range(0, len(origins), shard_records):
            shards.append({"id": f"{split}-{i // shard_records:05d}", "split": split,
                           "origins": origins[i:i + shard_records]})
    manifest = {"producer": producer, "producer_id": digest(producer), "shards": shards,
                "record_count": sum(len(s["origins"]) for s in shards)}
    write_json(root / "manifest.json", manifest, immutable=True)
    return manifest


def produce_shard(root, shard_id, backbone, adapter, store):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if digest(producer_identity(backbone, adapter, store)) != manifest["producer_id"]:
        raise ValueError("Producer identity changed")
    shard = next(s for s in manifest["shards"] if s["id"] == shard_id)
    final, meta = root / f"{shard_id}.zip", root / f"{shard_id}.json"
    if final.exists() or meta.exists():
        raise FileExistsError(f"Completed or interrupted shard exists: {shard_id}")
    temp = final.with_suffix(".partial")
    started = time.perf_counter()
    raw_bytes = 0
    storage_profile = None
    with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for index, origin in enumerate(shard["origins"]):
            # Always encode truth here. No state is propagated across records.
            record = live_record(backbone, store, origin)
            payload = pickle.dumps(record, protocol=5)
            if storage_profile is None and adapter is not None:
                started_profile = time.perf_counter()
                nodal = np.stack([np.asarray(adapter.features(record[k])) for k in ("origin_state", "baseline_state")])
                compressed = zlib.compress(nodal.tobytes(), level=1)
                if not np.array_equal(np.frombuffer(zlib.decompress(compressed), nodal.dtype).reshape(nodal.shape), nodal):
                    raise AssertionError("Nodal compression round trip differs")
                storage_profile = {"native_record_pickle_bytes": len(payload),
                    "native_record_compressed_bytes": len(zlib.compress(payload, level=1)),
                    "dynamic_nodal_pair_bytes": nodal.nbytes,
                    "dynamic_nodal_pair_compressed_bytes": len(compressed),
                    "comparison_seconds": time.perf_counter() - started_profile,
                    "note": "Nodal pair excludes auxiliary carry and forcing; native record includes them. Targets are shared references."}
            roundtrip = pickle.loads(payload)
            for a, b in zip(jax.tree_util.tree_leaves(record["baseline_state"]),
                            jax.tree_util.tree_leaves(roundtrip["baseline_state"]), strict=True):
                np.testing.assert_array_equal(a, b)
            raw_bytes += len(payload)
            z.writestr(f"{index:04d}.pkl", payload)
    temp.replace(final)
    value = {"shard_id": shard_id, "producer_id": manifest["producer_id"],
             "sha256": sha256(final), "records": len(shard["origins"]),
             "raw_bytes": raw_bytes, "stored_bytes": final.stat().st_size,
             "storage_profile": storage_profile,
             "seconds": time.perf_counter() - started}
    write_json(meta, value, immutable=True)
    return value


def verify_cache(root, *, fresh_record=None, sample_indices=(0, -1)):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest["producer_id"] != digest(manifest["producer"]):
        raise ValueError("Producer manifest identity differs")
    shard_hashes, count = {}, 0
    for shard in manifest["shards"]:
        sid = shard["id"]
        meta = read_json(root / f"{sid}.json")
        path = root / f"{sid}.zip"
        if (meta["producer_id"] != manifest["producer_id"] or meta["records"] != len(shard["origins"])
                or meta["sha256"] != sha256(path)):
            raise ValueError(f"Incomplete or corrupt cache shard: {sid}")
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None or len(z.namelist()) != len(shard["origins"]):
                raise ValueError(f"Invalid shard contents: {sid}")
            for i, t in enumerate(shard["origins"]):
                record = pickle.loads(z.read(f"{i:04d}.pkl"))
                if (record["origin"] != t or record["split"] != shard["split"]
                        or np.datetime64(record["valid"]) != np.datetime64(t) + STEP):
                    raise ValueError("Record time/split contract mismatch")
                for field in ("origin_state", "baseline_state", "forcing"):
                    if any(not np.isfinite(a).all() for a in jax.tree_util.tree_leaves(record[field])):
                        raise ValueError("Cache contains nonfinite native state")
                if fresh_record is not None and i in {j % len(shard["origins"]) for j in sample_indices}:
                    live = fresh_record(t)
                    for name in ("origin_state", "baseline_state", "forcing"):
                        for a, b in zip(jax.tree_util.tree_leaves(record[name]),
                                        jax.tree_util.tree_leaves(live[name]), strict=True):
                            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        shard_hashes[sid] = meta["sha256"]
        count += len(shard["origins"])
    if count != manifest["record_count"]:
        raise ValueError("Partial cache cannot release consumers")
    if fresh_record is None:
        raise ValueError("Live-versus-cached checks are required before READY")
    ready = {"manifest_sha256": sha256(root / "manifest.json"), "producer_id": manifest["producer_id"],
             "shard_hashes": shard_hashes, "record_count": count, "live_verified": True}
    write_json(root / "READY.json", ready, immutable=True)
    return ready


class CacheReader:
    def __init__(self, root, store, split):
        self.root, self.store = Path(root), store
        self.manifest = read_json(self.root / "manifest.json")
        self.ready = read_json(self.root / "READY.json")
        if (self.ready["manifest_sha256"] != sha256(self.root / "manifest.json")
                or not self.ready["live_verified"] or self.manifest["producer"]["dataset_id"] != store.identity):
            raise ValueError("Cache is incomplete or targets have changed")
        self.index = [(s["id"], i, t) for s in self.manifest["shards"] if s["split"] == split
                      for i, t in enumerate(s["origins"])]
        self._verified = set()

    @property
    def times(self):
        return [r[2] for r in self.index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        sid, i, _ = self.index[index]
        path = self.root / f"{sid}.zip"
        if sid not in self._verified:
            if sha256(path) != self.ready["shard_hashes"][sid]:
                raise ValueError(f"Cache shard changed: {sid}")
            self._verified.add(sid)
        with zipfile.ZipFile(path) as z:
            record = pickle.loads(z.read(f"{i:04d}.pkl"))
        record["target"] = {k: v for k, v in self.store.frame(record["target_time"]).items()
                            if k not in ("sea_surface_temperature", "sea_ice_cover")}
        def readonly(x):
            if isinstance(x, np.ndarray):
                x.setflags(write=False)
            return x
        for k in ("origin_state", "baseline_state", "forcing", "target"):
            record[k] = jax.tree_util.tree_map(readonly, record[k])
        return record
