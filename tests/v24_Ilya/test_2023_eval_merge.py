"""Scientific and completeness checks for the held-out daily-start report."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scripts.analyze_models.merge_v24_2023_eval import (
    MODEL_IDS,
    SECONDARY_ENDPOINTS,
    VARIABLES,
    analyze,
    bootstrap_analysis,
    cadence_block_counts,
    cadence_bootstrap_analysis,
    descriptive_diagnostics,
    loss_improvement,
    merge_shards,
    moving_block_counts,
    named_endpoint_draws,
    validate_protocol,
    write_outputs,
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
            for i in range(shard_index, len(protocol["initialization_times"]), shard_count):
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
                if protocol.get("schema_version") == 2:
                    records[-1]["variable_mse"] = {
                        variable: {f"{branch}_per_step": [value * (v + 1) for value in records[-1][f"{branch}_per_step"]]
                                   for branch in ("baseline", "full")}
                        for v, variable in enumerate(VARIABLES)
                    }
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
            if protocol.get("schema_version") == 2:
                shard["original_graphcast_loss"] = {}
                for branch in ("baseline", "full"):
                    pooled = np.mean([record[f"{branch}_per_step"] for record in records], axis=0)
                    shard["original_graphcast_loss"].update({f"{branch}_per_step": pooled.tolist(),
                                                           f"{branch}_rollout": float(pooled.mean())})
                shard["variable_mse_metadata"] = {
                    variable: {"units": "physical_units", "spatial_weighting": "normalized_cosine_latitude",
                               "pressure_pooling": "surface" if v < 5 else "equal_levels"}
                    for v, variable in enumerate(VARIABLES)
                }
                shard["per_variable_per_step"] = {
                    variable: {f"rmse_{branch}": np.sqrt(np.mean([
                        record["variable_mse"][variable][f"{branch}_per_step"] for record in records
                    ], axis=0)).tolist() for branch in ("baseline", "full")}
                    for variable in VARIABLES
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


def test_advisory_merge_retains_drift_and_paired_ranking(tmp_path, protocol):
    protocol["numerical_policy"] = "advisory"
    paths = _shards(tmp_path, protocol)
    payload = json.loads(paths[1].read_text())
    for row in payload["original_graphcast_loss_per_initialization"]:
        row["baseline_per_step"] = (np.asarray(row["baseline_per_step"]) * 1.25).tolist()
    paths[1].write_text(json.dumps(payload))
    baseline, full, audit = merge_shards(protocol, paths)
    assert audit["numerical_status"] == "warning"
    assert audit["baseline_repeatability"]["M2"]["outside_tolerance_count"] == 365 * 40
    assert audit["paired_baseline_sensitivity"]["ranking_changed"]
    assert audit["paired_baseline_sensitivity"]["canonical_ranking"][0] == "M1"
    assert audit["paired_baseline_sensitivity"]["paired_ranking"][0] == "M2"


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


@pytest.fixture
def cadence_protocol(protocol):
    protocol.update(
        schema_version=2, initialization_stride_days=3,
        initialization_times=protocol["initialization_times"][::3],
        models=[{"model_id": model, "checkpoint_sha256": f"hash-{model}"} for model in MODEL_IDS],
        bootstrap={"replicates": 100, "seed": 20260916, "block_starts": 5,
                   "sensitivity_block_starts": [3, 7], "segmentation": "annual", "quarter_sensitivity": True},
        analysis={"primary_endpoint": "M1_vs_GC_rollout", "secondary_endpoints": list(SECONDARY_ENDPOINTS)},
    )
    exposed = [_iso(datetime(2023, month, 15, tzinfo=timezone.utc)) for month in (1, 2, 4, 5, 7, 8, 10, 11)]
    protocol["prior_exposure"] = {
        "initialization_times": exposed,
        "sampled_initialization_times": [t for t in protocol["initialization_times"] if t in exposed],
        "selection_influence": "none_reported",
    }
    return protocol


def test_cadence_protocol_freezes_grid_endpoints_and_exposure(cadence_protocol):
    times, models, _ = validate_protocol(cadence_protocol)
    assert len(times) == 122 and times[-1] == "2023-12-30T00:00:00Z"
    assert models == list(MODEL_IDS)
    assert cadence_protocol["prior_exposure"]["sampled_initialization_times"] == [
        "2023-02-15T00:00:00Z", "2023-07-15T00:00:00Z", "2023-11-15T00:00:00Z"]
    cadence_protocol["analysis"]["secondary_endpoints"].append("M2_vs_GC_rollout")
    with pytest.raises(ValueError, match="four secondary"):
        validate_protocol(cadence_protocol)


def test_cadence_bootstrap_preserves_quarters_and_original_grid(cadence_protocol):
    times = cadence_protocol["initialization_times"]
    kwargs = dict(stride_days=3, block_starts=5, replicates=100, seed=1)
    annual = cadence_block_counts(times, **kwargs)
    np.testing.assert_array_equal(annual.sum(axis=1), 122)
    np.testing.assert_array_equal(annual, cadence_block_counts(times, **kwargs))
    quarterly = cadence_block_counts(times, segmentation="calendar_quarters", **kwargs)
    for left, right in ((0, 30), (30, 61), (61, 91), (91, 122)):
        np.testing.assert_array_equal(quarterly[:, left:right].sum(axis=1), right - left)
    kept = [t for t in times if t not in cadence_protocol["prior_exposure"]["sampled_initialization_times"]]
    with pytest.raises(ValueError, match="complete, regularly spaced"):
        cadence_block_counts(kept, **kwargs)


def test_named_contrasts_pool_before_ratio_and_use_eligible_denominator():
    baseline = np.asarray([[1., 1.], [9., 9.]])
    full = np.repeat(baseline[None], 7, axis=0)
    full[0, 0] = 0
    counts = np.asarray([[1., 1.], [2., 0.], [0., 3.]])
    definitions, effects, differences = named_endpoint_draws(baseline, full, counts, list(MODEL_IDS))
    assert [d["endpoint_id"] for d in definitions if d["role"] == "secondary"] == list(SECONDARY_ENDPOINTS)
    np.testing.assert_allclose(effects[:, 0], [10, 100, 0])
    np.testing.assert_allclose(effects[:, 2], [10, 100, 0])
    np.testing.assert_allclose(differences[:, 2], [.5, 1, 0])
    np.testing.assert_array_equal(effects[:, 3:], 0)


def test_schema2_merge_and_variable_pooling_are_shard_invariant(tmp_path, cadence_protocol):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    b1, f1, audit1 = merge_shards(cadence_protocol, _shards(first, cadence_protocol))
    b2, f2, audit2 = merge_shards(cadence_protocol, list(reversed(_shards(second, cadence_protocol, 3))))
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(f1, f2)
    for branch in ("baseline", "full"):
        np.testing.assert_array_equal(audit1["variable_mse"][branch], audit2["variable_mse"][branch])
    diagnostics = descriptive_diagnostics(cadence_protocol, b1[0], f1, audit1["variable_mse"])
    for season, count in (("DJF", 30), ("MAM", 31), ("JJA", 30), ("SON", 31)):
        assert {r["initialization_count"] for r in diagnostics["exact_loss"] if r["season"] == season} == {count}
    annual = next(r for r in diagnostics["variable_rmse"] if r["season"] == "annual" and r["model_id"] == "M1" and r["endpoint"] == "rollout")
    assert annual["baseline_rmse"] == pytest.approx(np.sqrt(b1[0].mean()))
    assert annual["model_rmse"] == pytest.approx(np.sqrt(f1[0].mean()))


@pytest.mark.parametrize("mutation", ["missing_variable", "nan_mse", "missing_lead", "wrong_units", "bad_aggregate"])
def test_schema2_rejects_incomplete_or_inconsistent_variable_stats(tmp_path, cadence_protocol, mutation):
    paths = _shards(tmp_path, cadence_protocol)
    payload = json.loads(paths[0].read_text())
    record = payload["original_graphcast_loss_per_initialization"][0]
    variable = VARIABLES[0]
    if mutation == "missing_variable":
        record["variable_mse"].pop(variable)
    elif mutation == "nan_mse":
        record["variable_mse"][variable]["baseline_per_step"][0] = float("nan")
    elif mutation == "missing_lead":
        record["variable_mse"][variable]["full_per_step"].pop()
    elif mutation == "wrong_units":
        payload["variable_mse_metadata"][variable]["units"] = ""
    else:
        payload["per_variable_per_step"][variable]["rmse_full"][0] *= 2
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="[Vv]ariable"):
        merge_shards(cadence_protocol, paths)


def test_schema2_exact_exposure_mask_and_hierarchical_claims(cadence_protocol):
    baseline = np.ones((7, 122, 40))
    full = np.repeat(np.asarray([.8, .9, 1., 1.1, .9, 1., 1.1])[:, None, None], 122, axis=1)
    full = np.repeat(full, 40, axis=2)
    result = analyze(cadence_protocol, baseline, full)
    primary = result["primary"]
    assert primary["block_elapsed_days"] == 12
    assert primary["secondary_family"] == list(SECONDARY_ENDPOINTS)
    assert all(e["confirmatory_supported"] for e in primary["named_endpoints"][:5])
    assert not primary["named_endpoints"][-1]["confirmatory_supported"]
    exclusion = result["sensitivities"][-1]
    assert exclusion["initialization_count"] == 119
    assert exclusion["resampling_grid_initialization_count"] == 122
    assert exclusion["bootstrap_eligible_count_min"] < exclusion["bootstrap_eligible_count_max"]
    assert all(c["supported_in_all_sensitivities"] for c in result["claim_sensitivity"])
    full[0] = 1.05
    failed = cadence_bootstrap_analysis(cadence_protocol, baseline[0], full, block_starts=5)
    memory = failed["named_endpoints"][2]
    assert memory["interval_supports_positive"]
    assert not memory["confirmatory_supported"]


def test_schema2_exposure_sensitivity_masks_scores_without_compressing_dates(cadence_protocol):
    baseline = np.ones((7, 122, 40))
    full = baseline * .8
    exposed = np.asarray([t in cadence_protocol["prior_exposure"]["sampled_initialization_times"]
                          for t in cadence_protocol["initialization_times"]])
    full[:, exposed] = 100
    result = analyze(cadence_protocol, baseline, full)
    assert result["primary"]["named_endpoints"][0]["effect"] < 0
    sensitivity = result["sensitivities"][-1]
    assert sensitivity["named_endpoints"][0]["effect"] == pytest.approx(20)
    assert sensitivity["named_endpoints"][0]["absolute_loss_reduction"] == pytest.approx(.2)
    assert sensitivity["named_endpoints"][0]["ci95_lower"] == pytest.approx(20)


def test_schema2_simultaneous_family_uses_only_four_registered_endpoints(cadence_protocol):
    rng = np.random.default_rng(27)
    baseline = np.ones((122, 40))
    full = .9 + .1 * rng.normal(size=(7, 122, 40))
    settings = cadence_protocol["bootstrap"]
    counts = cadence_block_counts(cadence_protocol["initialization_times"], stride_days=3,
                                  block_starts=5, replicates=settings["replicates"], seed=settings["seed"])
    _, point, _ = named_endpoint_draws(baseline, full, np.ones((1, 122)), list(MODEL_IDS))
    _, draws, _ = named_endpoint_draws(baseline, full, counts, list(MODEL_IDS))
    sd = draws[:, 1:5].std(axis=0, ddof=1)
    critical = np.quantile(np.max(np.abs(draws[:, 1:5] - point[:, 1:5]) / sd, axis=1), .95)
    result = cadence_bootstrap_analysis(cadence_protocol, baseline, full, block_starts=5)
    assert result["simultaneous_critical_value"] == pytest.approx(critical)
    for i, endpoint in enumerate(result["named_endpoints"][1:5]):
        assert endpoint["simultaneous_ci95_lower"] == pytest.approx(point[0, i + 1] - critical * sd[i])
        assert endpoint["simultaneous_ci95_upper"] == pytest.approx(point[0, i + 1] + critical * sd[i])
    assert "simultaneous_ci95_lower" not in result["named_endpoints"][-1]


def test_zero_variable_error_is_valid_but_zero_denominator_ratio_is_undefined(cadence_protocol):
    baseline = np.ones((122, 40))
    full = np.ones((7, 122, 40))
    zeros = np.zeros((7, 122, 40, 1))
    mse = dict(variables=[VARIABLES[0]], metadata={VARIABLES[0]: {"units": "K"}}, baseline=zeros, full=zeros)
    result = descriptive_diagnostics(cadence_protocol, baseline, full, mse)
    assert all(row["baseline_rmse"] == row["model_rmse"] == 0 for row in result["variable_rmse"])
    assert all(row["rmse_reduction_pct"] is None for row in result["variable_rmse"])


def test_schema2_outputs_are_reviewable_json_csv_and_publication_figures(tmp_path, cadence_protocol):
    sources = tmp_path / "shards"
    sources.mkdir()
    baseline, full, audit = merge_shards(cadence_protocol, _shards(sources, cadence_protocol))
    variable_mse = audit.pop("variable_mse")
    result = analyze(cadence_protocol, baseline, full, variable_mse)
    result.update(audit)
    output = tmp_path / "report"
    images = tmp_path / "images"
    write_outputs(cadence_protocol, result, baseline, full, output, images, variable_mse)
    summary = json.loads((output / "loss_summary.json").read_text())
    assert summary["schema_version"] == 2
    assert len(summary["diagnostics"]["variable_rmse"]) == 5 * 7 * 41 * 11
    assert "122 initializations" in (output / "report.md").read_text()
    assert "untouched" in (output / "report.md").read_text()
    for name in ("named_endpoints.csv", "claim_sensitivity.csv", "per_initialization_variable_mse.csv", "variable_rmse.csv"):
        assert (output / name).stat().st_size > 0
    for stem in ("exact_loss_reduction_2023", "memory_contrasts_2023", "variable_rmse_2023"):
        for suffix in ("png", "pdf"):
            assert (images / f"{stem}.{suffix}").stat().st_size > 0
