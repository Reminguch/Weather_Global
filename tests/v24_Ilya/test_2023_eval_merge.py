"""Scientific and completeness checks for the held-out daily-start report."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scripts.analyze_models.merge_v24_2023_eval import (
    bootstrap_analysis,
    loss_improvement,
    merge_shards,
    moving_block_counts,
    validate_protocol,
)


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


@pytest.fixture
def protocol():
    return {
        "protocol_hash": "protocol-frozen-before-test",
        "initialization_manifest_sha256": "date-manifest-hash",
        "initialization_times": [_iso(datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)) for i in range(365)],
        "target_steps": 40,
        "lead_hours": list(range(6, 241, 6)),
        "models": [{"model_id": model, "checkpoint_sha256": f"hash-{model}"} for model in ("M1", "M2", "M3")],
        "baseline": {"checkpoint_sha256": "hash-GC", "canonical_model_id": "M1", "repeatability_rtol": 1e-3, "repeatability_atol": 1e-6},
        "bootstrap": {"replicates": 100, "seed": 20260916, "block_days": 14, "sensitivity_block_days": [10, 21], "segmentation": "five_seasons"},
    }


def _shards(tmp_path, protocol, shard_count=1):
    paths = []
    for m, model in enumerate(protocol["models"]):
        for shard_index in range(shard_count):
            records = []
            for i in range(shard_index, 365, shard_count):
                time = protocol["initialization_times"][i]
                stamp = datetime.fromisoformat(time.replace("Z", "+00:00"))
                valid = [stamp + timedelta(hours=lead) for lead in protocol["lead_hours"]]
                records.append({
                    "ordinal": i, "initialization_time": time,
                    "initialization_id": stamp.strftime("%Y%m%dT%H%M%SZ"),
                    "input_times": [_iso(stamp - timedelta(hours=6)), time],
                    "valid_times": [_iso(value) for value in valid],
                    "initialization_year": 2023, "valid_years": [value.year for value in valid],
                    "baseline_per_step": [2.0 + i / 1000 + m * 1e-4] * 40,
                    "full_per_step": [(2.0 + i / 1000) * (0.8 + m / 10)] * 40,
                })
            shard = {
                "model_id": model["model_id"], "protocol_hash": protocol["protocol_hash"],
                "initialization_manifest_sha256": protocol["initialization_manifest_sha256"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "baseline_checkpoint_sha256": protocol["baseline"]["checkpoint_sha256"],
                "global_initialization_times": protocol["initialization_times"], "target_steps": 40,
                "evaluation_status": "complete", "metrics_complete": True,
                "anchor_shard_index": shard_index, "anchor_shard_count": shard_count,
                "evaluated_samples": len(records), "original_graphcast_loss_per_initialization": records,
            }
            path = tmp_path / f"{model['model_id']}_{shard_index}.json"
            path.write_text(json.dumps(shard))
            paths.append(path)
    return paths


def test_canonical_denominator_and_shard_invariance(tmp_path, protocol):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    b1, f1, audit = merge_shards(protocol, _shards(one, protocol))
    b2, f2, _ = merge_shards(protocol, list(reversed(_shards(two, protocol, 2))))
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(f1, f2)
    # Paired branches differ slightly, but every final denominator is M1.
    assert audit["baseline_repeatability"]["M3"]["max_absolute_difference"] > 0
    result = bootstrap_analysis(protocol["initialization_times"], b1[0], f1, block_days=14, replicates=40, seed=1)
    np.testing.assert_allclose(np.asarray(result["improvement_pct"])[:, 0], [20, 10, 0], atol=1e-12)


@pytest.mark.parametrize("mutation,match", [
    ("protocol", "protocol_hash"),
    ("checkpoint", "checkpoint_sha256"),
    ("baseline_identity", "baseline_checkpoint_sha256"),
    ("incomplete", "evaluation_status"),
    ("missing_init", "initialization records"),
    ("missing_lead", "baseline_per_step"),
    ("nan", "full_per_step"),
    ("time", "valid_times"),
    ("baseline_drift", "Baseline repeatability"),
])
def test_merge_rejects_invalid_artifacts(tmp_path, protocol, mutation, match):
    paths = _shards(tmp_path, protocol)
    payload = json.loads(paths[1].read_text())
    first = payload["original_graphcast_loss_per_initialization"][0]
    if mutation == "protocol":
        payload["protocol_hash"] = "another-protocol"
    elif mutation == "checkpoint":
        payload["checkpoint_sha256"] = "another-checkpoint"
    elif mutation == "baseline_identity":
        payload["baseline_checkpoint_sha256"] = "another-baseline"
    elif mutation == "incomplete":
        payload["evaluation_status"] = "in_progress"
    elif mutation == "missing_init":
        payload["original_graphcast_loss_per_initialization"].pop()
    elif mutation == "missing_lead":
        first["baseline_per_step"].pop()
    elif mutation == "nan":
        first["full_per_step"][0] = float("nan")
    elif mutation == "time":
        first["valid_times"][0] = first["valid_times"][1]
    elif mutation == "baseline_drift":
        first["baseline_per_step"][0] *= 1.1
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        merge_shards(protocol, paths)


def test_missing_and_duplicate_shards_rejected(tmp_path, protocol):
    paths = _shards(tmp_path, protocol, 2)
    with pytest.raises(ValueError, match="Missing shards"):
        merge_shards(protocol, paths[:-1])
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="Duplicate shard index"):
        merge_shards(protocol, paths + [duplicate])


def test_pooled_ratios_are_recomputed_for_each_draw():
    baseline = np.asarray([[1, 1], [9, 9]], dtype=float)
    full = np.asarray([[[0, 0], [9, 9]]], dtype=float)
    counts = np.asarray([[1, 1], [2, 0], [0, 2]], dtype=float)
    improvements = loss_improvement(baseline, full, counts)
    # Mean of per-start percentages would incorrectly give 50% for draw 1.
    np.testing.assert_allclose(improvements[:, 0, 0], [10, 100, 0])
    np.testing.assert_allclose(improvements[:, 0, 1], [10, 100, 0])
    # Nor may the rollout aggregate average different per-lead reductions.
    baseline = np.asarray([[1, 9], [1, 9]], dtype=float)
    full = np.asarray([[[0, 9], [0, 9]]], dtype=float)
    np.testing.assert_allclose(loss_improvement(baseline, full, counts)[0, 0], [10, 100, 0])


def test_date_blocks_preserve_seasons_and_all_lead_pairing(protocol):
    times = protocol["initialization_times"]
    counts = moving_block_counts(times, block_days=14, replicates=20, seed=123)
    np.testing.assert_array_equal(counts, moving_block_counts(times, block_days=14, replicates=20, seed=123))
    np.testing.assert_array_equal(counts.sum(axis=1), 365)
    for left, right in ((0, 59), (59, 151), (151, 243), (243, 334), (334, 365)):
        np.testing.assert_array_equal(counts[:, left:right].sum(axis=1), right - left)
    baseline = np.arange(1, 366)[:, None] * np.arange(1, 41)[None, :]
    full = np.stack([baseline * 0.8, baseline, baseline * 1.2])
    result = bootstrap_analysis(times, baseline, full, block_days=14, replicates=30, seed=2)
    # Constant improvements are exactly identified; identity is zero and a
    # constructed worsening model has a negative CI under every resample.
    for m, expected in enumerate((20, 0, -20)):
        np.testing.assert_allclose(result["pointwise_ci95_lower"][m], expected, atol=1e-11)
        np.testing.assert_allclose(result["pointwise_ci95_upper"][m], expected, atol=1e-11)
    assert result["rollout_degenerate_endpoints"] == [0, 1, 2]


def test_simultaneous_rollout_bounds_cover_max_standardized_deviation(protocol):
    times = protocol["initialization_times"]
    rng = np.random.default_rng(10)
    baseline = np.ones((365, 40))
    varying = np.repeat(np.sin(np.arange(365)[:, None] / 25), 40, axis=1)
    full = np.stack([1 + varying * 0.2, 1 - varying * 0.2, 1 + rng.normal(0, 0.1, (365, 40))])
    result = bootstrap_analysis(times, baseline, full, block_days=14, replicates=200, seed=3)
    assert result["rollout_simultaneous_critical_value"] > 0
    for lower, upper in zip(result["rollout_simultaneous_ci95_lower"], result["rollout_simultaneous_ci95_upper"]):
        assert lower < 0 < upper


def test_block_resampling_retains_weather_regime_dependence(protocol):
    times = protocol["initialization_times"]
    baseline = np.ones((365, 40))
    # Smooth weather-regime errors have much less independent information than
    # 365 unrelated starts; date blocks must reflect that in the interval.
    regimes = np.sin(np.arange(365) / 10)
    full = np.repeat((1 + 0.2 * regimes)[None, :, None], 40, axis=2)
    block = bootstrap_analysis(times, baseline, full, block_days=14, replicates=300, seed=8)
    iid = bootstrap_analysis(times, baseline, full, block_days=1, replicates=300, seed=8)
    block_width = block["pointwise_ci95_upper"][0][0] - block["pointwise_ci95_lower"][0][0]
    iid_width = iid["pointwise_ci95_upper"][0][0] - iid["pointwise_ci95_lower"][0][0]
    assert block_width > 2 * iid_width


def test_protocol_must_not_silently_drop_year_end(protocol):
    protocol["initialization_times"] = protocol["initialization_times"][:-10]
    with pytest.raises(ValueError, match="365"):
        validate_protocol(protocol)
