"""A data-scale comparison must preserve the old windows, fields and predictions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.experiments import fresh_data_scale_cache as aligned
from src.models.mamba.v24_Ilya import baseline_cache as cache


def test_alignment_preserves_every_old_segment_with_gap_only_between_segments():
    full = pd.date_range("1982-01-01", "2022-12-31 18:00", freq="6h").values
    old = pd.date_range("2015-01-01", "2022-12-31 18:00", freq="6h").values
    raw = np.arange(2, len(old) - 2)
    years = pd.DatetimeIndex(old[raw]).year
    original = dict(anchor_indices=raw, anchor_times=old[raw],
                    split_train=np.flatnonzero(years <= 2021), split_val=np.flatnonzero(years == 2022))
    arrays, proof = aligned.aligned_anchor_arrays(full, old, original)
    assert proof["historical_anchors"] % 96 == 0
    assert proof["historical_segments"] > 0
    assert len(proof["omitted_boundary_anchor_indices"]) < 96
    for split in ("split_train", "split_val"):
        mapped = arrays[split]
        if split == "split_train":
            mapped = mapped[proof["historical_anchors"]:]
        np.testing.assert_array_equal(full[arrays["anchor_indices"][mapped]], old[raw[original[split]]])
        for offset in range(0, len(mapped) - 95, 96):
            np.testing.assert_array_equal(
                full[arrays["anchor_indices"][mapped[offset:offset + 96]]],
                old[raw[original[split][offset:offset + 96]]])
    # 484 original chunks remain: 106 train and 15 validation segments, four chunks each.
    assert (len(original["split_train"]) // 96 + len(original["split_val"]) // 96) * 4 == 484


def test_alignment_rejects_missing_original_clock_frame():
    old = np.arange(20) * np.timedelta64(6, "h") + np.datetime64("2015-01-01")
    raw = np.arange(2, 18)
    arrays = dict(anchor_indices=raw, anchor_times=old[raw], split_train=np.arange(8), split_val=np.arange(8, 16))
    with pytest.raises(ValueError, match="clock"):
        aligned.aligned_anchor_arrays(np.delete(old, 10), old, arrays, segment_steps=4)


def stores():
    times = np.arange(8) * np.timedelta64(6, "h") + np.datetime64("2015-01-01")
    weather = np.arange(16, dtype=np.float32).reshape(8, 2)
    def store(time, data):
        return SimpleNamespace(time=SimpleNamespace(values=time),
            coords={"time": time, "lon": np.array([0, 1], dtype=np.float32)},
            data_vars={"u": SimpleNamespace(dims=("time", "lon"), data=data),
                       "land": SimpleNamespace(dims=("lon",), data=np.array([1, 0], dtype=np.float32))})
    full_times = np.concatenate([times[:1] - np.timedelta64(6, "h"), times])
    return store(full_times, np.vstack([weather[:1] - 1, weather])), store(times, weather)


def test_overlap_checks_every_value_including_tail_and_static():
    full, old = stores()
    report = aligned.verify_prepared_overlap(full, old, block_steps=3)
    assert report["bitwise_equal"] and report["time_steps"] == 8
    assert report["variables"]["u"]["elements"] == 16
    full.data_vars["u"].data[-1, -1] += 1
    with pytest.raises(ValueError, match="overlap values differ"):
        aligned.verify_prepared_overlap(full, old, block_steps=3)
    full, old = stores()
    full.data_vars["land"].data[-1] = 1
    with pytest.raises(ValueError, match="static field"):
        aligned.verify_prepared_overlap(full, old)


def test_overlap_rejects_coordinate_dtype_change():
    full, old = stores()
    full.coords["lon"] = full.coords["lon"].astype(np.float64)
    with pytest.raises(ValueError, match="coordinate differs"):
        aligned.verify_prepared_overlap(full, old)


def test_overlap_bitwise_claim_distinguishes_signed_zero():
    full, old = stores()
    full.data_vars["u"].data[1, 0] = -0.0
    assert np.array_equal(full.data_vars["u"].data[1:], old.data_vars["u"].data)
    with pytest.raises(ValueError, match="overlap values differ"):
        aligned.verify_prepared_overlap(full, old)


def _manifest(items, shards=1):
    compatibility = {key: "same" for key in (*aligned.SEMANTIC_KEYS, *aligned.PREPARED_IDENTITY_KEYS)}
    compatibility.update(bptt_steps=2, segment_steps=4, truth_prefix_steps=1, ar_tail_k=1,
                         baseline_sha256="baseline", stats_sha256={"stats": "same"})
    result = dict(format=cache.FORMAT, compatibility=compatibility,
                  compatibility_sha256=cache.digest(compatibility),
                  variables={"u": dict(dims=["lon"], shape=[2], dtype="float32")},
                  coordinates={"lon": [0, 1]}, chunks=items, num_shards=shards,
                  source_paths={"prepared_root": "/toy-prepared"},
                  partial=False, full_chunk_count=len(items), prediction_bytes=len(items) * 16)
    result["manifest_sha256"] = cache.digest(result)
    return result


def _chunk(index, year, split="train", offset=0, shard=0):
    times = np.arange(2) * np.timedelta64(6, "h") + np.datetime64(f"{year}-01-01T00")
    return dict(id=index, split=split, segment_id=0, segment_offset=offset, chunk_index=offset // 2,
                raw_anchor_indices=[index * 2 + 2, index * 2 + 3],
                target_timestamps=[str(t) for t in times], shard=shard)


def _write_shard(root, manifest, shard):
    directory = root / f"shard_{shard:03d}"
    directory.mkdir(parents=True)
    files = {}
    items = [c for c in manifest["chunks"] if c["shard"] == shard]
    for item in items:
        chunk = directory / f"chunk_{item['id']:06d}"
        chunk.mkdir()
        path = chunk / "u.npy"
        np.save(path, np.full((2, 2), item["id"], dtype=np.float32))
        files[str(path.relative_to(directory))] = dict(sha256=cache.sha256(path), bytes=path.stat().st_size)
    cache.atomic_json(directory / "COMPLETE.json", dict(manifest_sha256=manifest["manifest_sha256"],
        chunk_ids=[c["id"] for c in items], files=files))


@pytest.fixture
def production(tmp_path):
    source_root = tmp_path / "original"
    root = tmp_path / "expanded"
    source = _manifest([_chunk(0, 2015), _chunk(1, 2022, "val")])
    cache.ensure_manifest(source_root, source)
    _write_shard(source_root, source, 0)
    cache.verify_cache(source_root)
    chunks = [_chunk(0, 1982), _chunk(1, 2015), _chunk(2, 2022, "val")]
    manifest = _manifest(chunks)
    cache.ensure_manifest(root, manifest)
    cache.atomic_json(root / "alignment.json", dict(format=aligned.PROOF_FORMAT,
        source_cache_root=str(source_root.resolve()), source_manifest_sha256=source["manifest_sha256"],
        manifest_sha256=manifest["manifest_sha256"], overlap={"bitwise_equal": True},
        imported_chunks=2, historical_chunks=1))
    return source_root, root, source, manifest


def test_shared_window_requires_same_recurrent_segment_position(production):
    _, _, source, manifest = production
    manifest["chunks"][1]["segment_offset"] = 2
    with pytest.raises(ValueError, match="recurrent segment"):
        aligned.match_source_chunks(manifest, source)


def test_missing_source_window_and_changed_stats_fail(production):
    _, _, source, manifest = production
    missing = copy.deepcopy(manifest)
    missing["chunks"].pop()
    with pytest.raises(ValueError, match="preserve every"):
        aligned.match_source_chunks(missing, source)
    manifest["compatibility"]["stats_sha256"] = {"stats": "changed"}
    with pytest.raises(ValueError, match="stats_sha256"):
        aligned.match_source_chunks(manifest, source)


def test_generate_only_earlier_chunks_and_hardlink_identical_old_predictions(production, monkeypatch):
    source_root, root, source, manifest = production
    monkeypatch.setattr(cache, "load_context", lambda _: "context")
    monkeypatch.setattr(cache, "build_manifest", lambda context, n: manifest)
    generated = []
    def generate(context, subset, target, shard, resume):
        cache.validate_manifest(subset)
        generated.extend(subset["chunks"])
        cache.ensure_manifest(target, subset)
        _write_shard(target, subset, shard)
    monkeypatch.setattr(cache, "generate_shard", generate)
    proof = aligned.generate_aligned_shard(Path("config"), source_root, root, 0)
    assert [c["id"] for c in generated] == [0]
    assert proof["imported_chunks"] == 2 and proof["historical_chunks"] == 1
    for new_id, original_id in [(1, 0), (2, 1)]:
        original = source_root / f"shard_000/chunk_{original_id:06d}/u.npy"
        linked = root / f"shard_000/chunk_{new_id:06d}/u.npy"
        assert original.stat().st_ino == linked.stat().st_ino
        np.testing.assert_array_equal(np.load(original), np.load(linked))
    assert aligned.verify_aligned_cache(root)["chunks"] == 3
    # A repeated worker verifies completed results without new prediction calls.
    assert aligned.generate_aligned_shard(Path("config"), source_root, root, 0)["resumed"]
    assert len(generated) == 1


def test_import_rejects_corrupted_old_prediction(production, monkeypatch):
    source_root, root, source, manifest = production
    monkeypatch.setattr(cache, "load_context", lambda _: "context")
    monkeypatch.setattr(cache, "build_manifest", lambda context, n: manifest)
    def generate(context, subset, target, shard, resume):
        cache.ensure_manifest(target, subset)
        _write_shard(target, subset, shard)
    monkeypatch.setattr(cache, "generate_shard", generate)
    np.save(source_root / "shard_000/chunk_000000/u.npy", np.full((2, 2), 999, dtype=np.float32))
    with pytest.raises(ValueError, match="Checksum"):
        aligned.generate_aligned_shard(Path("config"), source_root, root, 0)
    assert not (root / "shard_000").exists()


def test_source_ready_must_match_coverage(production):
    source_root, _, _, _ = production
    ready = aligned.read(source_root / "READY.json")
    ready["chunks"] += 1
    cache.atomic_json(source_root / "READY.json", ready)
    with pytest.raises(ValueError, match="complete verified"):
        aligned.read_source_cache(source_root)


@pytest.fixture
def historical_production(production):
    source_root, root, source, manifest = production
    history_root = root.parent / "historical"
    historical = _manifest([_chunk(0, 1982), _chunk(1, 2014)])
    cache.ensure_manifest(history_root, historical)
    _write_shard(history_root, historical, 0)
    cache.verify_cache(history_root)
    proof = aligned.read(root / "alignment.json")
    proof.update(historical_cache_root=str(history_root.resolve()),
                 historical_manifest_sha256=historical["manifest_sha256"],
                 imported_historical_chunks=1, generated_chunks=0)
    cache.atomic_json(root / "alignment.json", proof)
    return source_root, root, source, manifest, history_root, historical


def test_import_only_hardlinks_history_and_recent_without_gpu_generation(historical_production, monkeypatch):
    source_root, root, source, manifest, history_root, history = historical_production
    monkeypatch.setattr(cache, "load_context", lambda _: "context")
    monkeypatch.setattr(cache, "build_manifest", lambda context, n: manifest)
    monkeypatch.setattr(cache, "generate_shard", lambda *a, **kw: pytest.fail("Import-only initialized generation"))
    monkeypatch.setattr(cache, "make_predictor", lambda *a, **kw: pytest.fail("Import-only initialized GraphCast"))
    report = aligned.generate_aligned_shard(Path("config"), source_root, root, 0, historical_cache_root=history_root)
    assert report["generated_chunks"] == 0 and report["imported_historical_chunks"] == 1
    for new_id, original_id, original_root in [(0, 0, history_root), (1, 0, source_root), (2, 1, source_root)]:
        original = original_root / f"shard_000/chunk_{original_id:06d}/u.npy"
        linked = root / f"shard_000/chunk_{new_id:06d}/u.npy"
        assert original.stat().st_ino == linked.stat().st_ino
    assert not (root / ".historical").exists()
    assert aligned.verify_aligned_cache(root)["generated_chunks"] == 0


@pytest.mark.parametrize("key", aligned.PREPARED_IDENTITY_KEYS)
def test_historical_import_rejects_prepared_identity_drift(historical_production, key):
    _, _, source, manifest, _, historical = historical_production
    historical["compatibility"][key] = "drifted"
    with pytest.raises(ValueError, match="source/numerical identity"):
        aligned.match_historical_chunks(manifest, historical, aligned.match_source_chunks(manifest, source))


def test_historical_import_rejects_moved_raw_windows(historical_production):
    _, _, source, manifest, _, historical = historical_production
    historical["chunks"][0]["raw_anchor_indices"] = [3, 4]
    with pytest.raises(ValueError, match="window/segment identity"):
        aligned.match_historical_chunks(manifest, historical, aligned.match_source_chunks(manifest, source))


def test_import_requires_the_historical_cache_pinned_during_preparation(historical_production, monkeypatch):
    source_root, root, source, manifest, history_root, history = historical_production
    monkeypatch.setattr(cache, "load_context", lambda _: "context")
    monkeypatch.setattr(cache, "build_manifest", lambda context, n: manifest)
    with pytest.raises(ValueError, match="pinned alignment proof"):
        aligned.generate_aligned_shard(Path("config"), source_root, root, 0)


def test_historical_import_rejects_corrupted_prediction(historical_production, monkeypatch):
    source_root, root, source, manifest, history_root, history = historical_production
    monkeypatch.setattr(cache, "load_context", lambda _: "context")
    monkeypatch.setattr(cache, "build_manifest", lambda context, n: manifest)
    np.save(history_root / "shard_000/chunk_000000/u.npy", np.full((2, 2), 777, dtype=np.float32))
    with pytest.raises(ValueError, match="Checksum"):
        aligned.generate_aligned_shard(Path("config"), source_root, root, 0, historical_cache_root=history_root)
    assert not (root / "shard_000").exists()
