"""Scientific pooling and paired-bootstrap contracts for extreme reports."""
import json

import numpy as np
import pytest

from scripts.analyze_models.report_v24_extremes_2023 import (
    BRANCHES, DOMAINS, EVENTS, METRICS, STAT_NAMES, THRESHOLDS, _check_record,
    circular_block_counts, endpoint_statistics, metrics_from_sums,
)
from scripts.analyze_models import report_v24_extremes_2023 as reporter


def case(hit, miss, false_alarm, correct_negative, absolute_error=0):
    result = np.zeros(15)
    result[:5] = [hit, miss, false_alarm, correct_negative, hit + miss + false_alarm + correct_negative]
    result[5:10] = result[:5]
    result[10] = absolute_error
    result[12] = absolute_error ** 2
    return result


def test_pool_contingencies_before_ratio_and_count_false_alarms():
    a = case(1, 0, 0, 9)
    b = case(0, 9, 1, 0)
    pooled = metrics_from_sums(a + b)
    assert pooled[METRICS.index("csi")] == 1 / 11
    assert pooled[METRICS.index("recall")] == .1
    assert pooled[METRICS.index("precision")] == .5
    assert pooled[METRICS.index("false_alarm_ratio")] == .5
    assert pooled[METRICS.index("csi")] != np.mean(metrics_from_sums(np.stack([a, b]))[:, 0])


def test_empty_events_remain_undefined_but_false_alarm_frequency_is_defined():
    empty = metrics_from_sums(case(0, 0, 0, 10))
    assert np.isnan(empty[[0, 1, 2, 3, 4, 7, 8, 9]]).all()
    assert empty[5] == empty[6] == 0
    false_alarm_only = metrics_from_sums(case(0, 0, 1, 9))
    assert false_alarm_only[0] == 0 and false_alarm_only[3] == 1
    assert np.isnan(false_alarm_only[7])


def test_block_resampling_preserves_pairing_and_fixed_case_count():
    counts = circular_block_counts(12, block_size=5, replicates=100, seed=7)
    assert (counts.sum(axis=1) == 12).all()
    assert np.array_equal(counts, circular_block_counts(12, block_size=5, replicates=100, seed=7))
    values = np.zeros((12, 1, 1, 1, 2, 15))
    for i in range(12):
        values[i, 0, 0, 0, :] = case(i + 1, 2, 3, 4, absolute_error=2)
    point, bounds, support, delta, delta_bounds, delta_support = endpoint_statistics(values, counts)
    assert np.all(delta == 0) and np.all(delta_bounds == 0)
    assert np.all(support == 100) and np.all(delta_support == 100)


def test_tail_mae_uses_observed_tail_area_not_forecast_event_area():
    values = case(1, 3, 8, 0, absolute_error=20)
    assert metrics_from_sums(values)[METRICS.index("tail_mae")] == 5


def test_record_contract_keeps_missing_rain_explicit_and_checks_climatology(tmp_path):
    spec = dict(protocol_hash="protocol", model_id="M1", checkpoint_sha256="checkpoint",
                baseline={"checkpoint_sha256": "baseline"})
    stamp = "2023-01-01T00:00:00Z"
    metadata = {key: spec[key] for key in ("protocol_hash", "model_id", "checkpoint_sha256")}
    metadata.update(status="complete", initialization_time=stamp, baseline_checkpoint_sha256="baseline",
                    provenance={"extremes_protocol": {"climatology_metadata": {"sha256": "climate"}}})
    stats = np.zeros((40, 4, 2, 2, 2, 15))
    stats[:3, 3] = np.nan
    valid = np.ones((40, 4), bool)
    valid[:3, 3] = False
    path = tmp_path / "record.npz"
    np.savez(path, stats=stats, valid_step_event=valid, metadata=json.dumps(metadata),
             events=EVENTS, thresholds=THRESHOLDS, domains=DOMAINS, branches=BRANCHES,
             statistics=STAT_NAMES, lead_hours=np.arange(1, 41) * 6,
             negative_tp6_counts=np.zeros((40, 3), int), negative_tp6_branches=["truth", "baseline", "full"])
    actual_stamp, actual, negative = _check_record(path, spec, [stamp], "climate")
    assert actual_stamp == stamp and np.isfinite(actual).all()
    assert not actual[:3, 3].any() and not negative.any()
    with pytest.raises(ValueError, match="climatology"):
        _check_record(path, spec, [stamp], "wrong")


def test_report_publishes_accuracy_event_tables_and_figures(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    stats = np.broadcast_to(case(1, 2, 3, 4, absolute_error=3), (122, 40, 4, 2, 2, 2, 15)).copy()
    stats[:, :3, 3] = 0
    loss = np.ones((122, 40, 2))
    loss[..., 1] = .8
    campaign = dict(workspace=str(tmp_path), models=[dict(id="M1", label="Main model")])
    arrays = {"M1": dict(stats=stats, loss=loss, negative_tp6_counts=np.zeros((122, 40, 3), int))}
    monkeypatch.setattr(reporter, "load_campaign", lambda unused: (campaign, arrays, []))
    payload = reporter.report(root)
    assert payload["status"] == "complete"
    assert payload["summary"][0]["mse_reduction_rollout_pct"] == pytest.approx(20)
    assert (root / "complete.json").is_file()
    out = tmp_path / "plots/analyze_models/data/resolution_eval/campaign"
    assert (out / "per_lead.csv").is_file() and (out / "endpoints.csv").is_file()
    assert (tmp_path / "plots/analyze_models/images/resolution_eval/campaign/extreme_skill.pdf").is_file()
