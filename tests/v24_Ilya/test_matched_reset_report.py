"""Scientific pooling, paired uncertainty, and provenance guards for reset control."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scripts.analyze_models import merge_v24_2023_eval as helpers
from scripts.analyze_models import report_v24_matched_reset_2023 as report


@pytest.fixture
def protocols():
    times = [(datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat().replace("+00:00", "Z")
             for i in range(0, 365, 3)]
    reference = dict(protocol_hash="original-frozen-protocol", initialization_times=times,
        initialization_manifest_sha256="manifest-sha", target_steps=40, lead_hours=list(range(6, 241, 6)),
        initialization_stride_days=3, models=[dict(model_id=m, checkpoint=f"/{m}.pkl", checkpoint_sha256=f"checkpoint-{m}")
                                            for m in ("M1", "M1_reset")],
        baseline=dict(checkpoint_sha256="baseline-sha", repeatability_rtol=.001, repeatability_atol=1e-6),
        bootstrap=dict(replicates=120, seed=20260916),
        prior_exposure=dict(sampled_initialization_times=[times[15], times[65], times[106]]))
    protocol = dict(protocol_hash="new-protocol", checkpoint=dict(path="/reset.pkl", sha256="reset-sha"))
    return protocol, reference


def make_shard(protocol, reference, model_id="R_matched", shard_index=0):
    records = []
    for i in range(shard_index, len(reference["initialization_times"]), 8):
        stamp = helpers._utc(reference["initialization_times"][i])
        records.append(dict(ordinal=i, initialization_time=helpers._iso(stamp),
            initialization_id=stamp.strftime("%Y%m%dT%H%M%SZ"), initialization_year=2023,
            input_times=[helpers._iso(stamp-timedelta(hours=6)), helpers._iso(stamp)],
            valid_times=[helpers._iso(stamp+timedelta(hours=h)) for h in reference["lead_hours"]],
            valid_years=[(stamp+timedelta(hours=h)).year for h in reference["lead_hours"]],
            baseline_per_step=[2.] * 40, full_per_step=[1.8] * 40,
            variable_mse={v: {"baseline_per_step": [2.] * 40, "full_per_step": [1.8] * 40}
                          for v in helpers.VARIABLES}))
    return dict(model_id=model_id, protocol_hash=protocol["protocol_hash"],
        checkpoint_sha256=protocol["checkpoint"]["sha256"], baseline_checkpoint_sha256="baseline-sha",
        initialization_manifest_sha256="manifest-sha", global_initialization_times=reference["initialization_times"],
        target_steps=40, anchor_shard_count=8, anchor_shard_index=shard_index,
        evaluation_status="complete", metrics_complete=True, rs_reset_every_step=True,
        eval_mode="cold_full", residual_state_init_resolved="zero", warmup_steps=0,
        eval_feedback="full", baseline_branch="pure_baseline_self_rollout",
        reference_protocol_hash=reference["protocol_hash"], training_state_policy="reset_every_anchor",
        evaluator_source_digest="frozen-source-digest", evaluated_samples=len(records),
        original_graphcast_loss_per_initialization=records,
        original_graphcast_loss=dict(baseline_per_step=[2.]*40, full_per_step=[1.8]*40,
                                    baseline_rollout=2., full_rollout=1.8),
        variable_mse_metadata={v: dict(units="physical_units", spatial_weighting="normalized_cosine_latitude",
            pressure_pooling="surface" if v in helpers.VARIABLES[:5] else "equal_levels") for v in helpers.VARIABLES},
        per_variable_per_step={v: dict(rmse_baseline=[np.sqrt(2.)]*40,
                                     rmse_full=[np.sqrt(1.8)]*40) for v in helpers.VARIABLES})


def validate(payload, protocols, canonical_metadata=None):
    protocol, reference = protocols
    return report.validate_shard(payload, model_id="R_matched", shard_index=0, protocol=protocol,
        reference=reference, helpers=helpers, source_digest="frozen-source-digest", canonical_metadata=canonical_metadata)


def test_pools_losses_before_ratio_not_case_percentages(protocols):
    protocol, reference = protocols
    size = len(reference["initialization_times"])
    gc = np.repeat(np.r_[np.full(size//2, 2.), np.full(size//2, 100.)][:, None], 40, axis=1)
    baseline = np.repeat(gc[None], 3, axis=0)
    factors = np.r_[np.full(size//2, .9), np.full(size//2, .5)]
    full = np.stack([gc*factors[:, None], gc*.95, gc*.8])
    result = report.analysis(protocol, reference, helpers, baseline, full)
    expected = 100 * (1 - 25.9/51.)
    assert result["improvement_pct"][0][0] == pytest.approx(expected)
    assert result["improvement_pct"][0][-1] == pytest.approx(expected)
    assert result["contrasts"][0]["effect"][0] == pytest.approx(expected-20.)
    assert result["improvement_pct"][0][0] != pytest.approx(30.)


def test_bootstrap_preserves_paired_policy_covariance(protocols):
    protocol, reference = protocols
    size = len(reference["initialization_times"])
    baseline = np.ones((3, size, 40))*2
    varying = np.sin(np.arange(size)/8.)[:, None]*.15
    first = np.broadcast_to(1.4+varying, (size, 40)).copy()
    full = np.stack([first, first+.1, first+.2])
    result = report.analysis(protocol, reference, helpers, baseline, full)
    assert result["pointwise_ci95_upper"][0][0] > result["pointwise_ci95_lower"][0][0]
    contrast = result["contrasts"][0]
    np.testing.assert_allclose(contrast["effect"], 10., atol=1e-12)
    np.testing.assert_allclose(contrast["pointwise_ci95_lower"], 10., atol=1e-12)
    np.testing.assert_allclose(contrast["pointwise_ci95_upper"], 10., atol=1e-12)


def test_pilot_mask_keeps_original_cadence(protocols):
    protocol, reference = protocols
    baseline = np.ones((3, 122, 40))*2
    result = report.analyze(protocol, reference, helpers, baseline, baseline*.8)
    masked = result["sensitivities"][-1]
    assert masked["initialization_count"] == 119
    assert masked["resampling_grid_initialization_count"] == 122
    assert masked["excluded_initialization_times"] == reference["prior_exposure"]["sampled_initialization_times"]
    assert result["primary_endpoint"] == "M1_vs_R_matched_rollout"
    assert "separate planned" in result["analysis_scope"].lower()
    assert "confirmatory_supported" not in json.dumps(result)


def test_own_baseline_sensitivity_does_not_change_canonical_endpoint(protocols):
    protocol, reference = protocols
    baseline = np.ones((3, 122, 40))*2
    baseline[2] *= 1.1
    full = np.ones_like(baseline)*1.6
    result = report.analysis(protocol, reference, helpers, baseline, full)
    assert result["contrasts"][0]["effect"][0] == pytest.approx(0.)
    own = result["own_baseline_sensitivity"]
    assert own["improvement_pct"][2][0] == pytest.approx(100*(1-1.6/2.2))
    assert own["M1_vs_R_matched_effect"][0] < 0


def test_complete_shard_is_accepted(protocols):
    payload = make_shard(*protocols)
    records, metadata = validate(payload, protocols)
    assert len(records) == 16
    assert set(metadata) == set(helpers.VARIABLES)


@pytest.mark.parametrize("field,value", [
    ("evaluation_status", "running"), ("metrics_complete", False), ("rs_reset_every_step", False),
    ("training_state_policy", "carry"), ("checkpoint_sha256", "wrong-checkpoint"),
    ("reference_protocol_hash", "other-campaign"), ("evaluator_source_digest", "other-code"),
    ("eval_feedback", "baseline"), ("residual_state_init_resolved", "ckpt"),
])
def test_rejects_wrong_policy_provenance_or_incomplete_shards(protocols, field, value):
    payload = make_shard(*protocols)
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        validate(payload, protocols)


def test_rejects_duplicate_or_missing_initializations(protocols):
    payload = make_shard(*protocols)
    payload["original_graphcast_loss_per_initialization"][-1] = copy.deepcopy(payload["original_graphcast_loss_per_initialization"][0])
    with pytest.raises(ValueError, match="missing, duplicate"):
        validate(payload, protocols)


@pytest.mark.parametrize("mutation", ["lead", "nonfinite", "aggregate", "units"])
def test_rejects_misaligned_or_inconsistent_scores(protocols, mutation):
    payload = make_shard(*protocols)
    canonical_metadata = copy.deepcopy(payload["variable_mse_metadata"])
    record = payload["original_graphcast_loss_per_initialization"][0]
    if mutation == "lead":
        record["valid_times"][0] = record["valid_times"][1]
    elif mutation == "nonfinite":
        record["full_per_step"][2] = float("nan")
    elif mutation == "aggregate":
        payload["original_graphcast_loss"]["full_rollout"] = 1.7
    else:
        payload["variable_mse_metadata"][helpers.VARIABLES[0]]["units"] = "wrong-units"
    with pytest.raises(ValueError):
        validate(payload, protocols, canonical_metadata)


def test_missing_shards_fail_before_output(protocols, tmp_path):
    protocol, reference = protocols
    protocol["reference_campaign"] = str(tmp_path)
    with pytest.raises(ValueError, match="Missing or unexpected shards"):
        report.gather(tmp_path, protocol, reference, helpers, "frozen-source-digest")
    assert not (tmp_path / "report").exists()


def test_rejects_tampered_protocol_before_opening_dependencies(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps(dict(schema_version=1, protocol_hash="tampered")))
    with pytest.raises(ValueError, match="self hash"):
        report.load_workflow(tmp_path)
