"""Matched extreme comparisons preserve paired dates and verifying exposure."""
import json

import numpy as np
import pytest

from scripts.analyze_models import report_v24_matched_reset_extremes as reporter


def statistics(hit=1., miss=2., false_alarm=3., correct_negative=4., absolute_error=6.):
    values = np.zeros(15)
    values[:5] = [hit, miss, false_alarm, correct_negative,
                  hit + miss + false_alarm + correct_negative]
    values[5:10] = values[:5]
    values[10] = absolute_error
    values[12] = absolute_error ** 2
    values[13] = 300 * (hit + miss)
    values[14] = 298 * (hit + miss)
    return values


def test_verifying_exposure_check_accepts_different_forecasts_but_rejects_changed_truth():
    reference = np.broadcast_to(statistics(), (2, 4, 2, 15)).copy()
    candidate = reference.copy()
    candidate[..., 0, 0] += 1
    candidate[..., 0, 1] -= 1
    candidate[..., 0, 5] += 1
    candidate[..., 0, 6] -= 1
    candidate[..., 1, 10] *= .8
    reporter.validate_truth_exposure(reference, candidate)
    candidate[..., 1, 13] += 1
    with pytest.raises(ValueError, match="Truth-selected"):
        reporter.validate_truth_exposure(reference, candidate)


def test_tables_pair_carry_with_reset_and_pool_loss_before_percentage():
    base = np.broadcast_to(statistics(), (2, 40, 4, 2, 2, 15)).copy()
    base[:, :3, 3] = 0
    arrays = {}
    for model, scale, losses in (("GC", 1., [1., 9.]), ("M1", .5, [.8, 4.5]),
                                ("R_matched", .75, [.9, 6.])):
        stats = base.copy()
        stats[..., 10] *= scale
        arrays[model] = dict(stats=stats, loss=np.broadcast_to(np.array(losses)[:, None], (2, 40)))
    counts = np.array([[2, 0], [0, 2], [1, 1]])
    leads, endpoints, accuracy = reporter.build_tables(arrays, counts)
    accuracy_row = next(row for row in accuracy if row["model"] == "M1"
                        and row["reference"] == "GC" and row["endpoint"] == "rollout")
    assert accuracy_row["loss_reduction_pct"] == pytest.approx(47.)
    paired = next(row for row in endpoints if row["model"] == "M1"
                  and row["reference"] == "R_matched" and row["endpoint"] == "day5"
                  and row["event"] == "hot" and row["threshold"] == "primary"
                  and row["domain"] == "global" and row["metric"] == "tail_mae")
    assert paired["value"] == 1.
    assert paired["reference_value"] == 1.5
    assert paired["difference"] == -.5
    assert paired["difference_ci95_lower"] == -.5
    rain = [row for row in leads if row["event"] == "rain" and row["lead_hours"] < 24]
    assert all(row["available"] is False and row["csi"] is None for row in rain)
    assert set(row["model"] for row in leads) == {"GC", "M1", "R_matched"}


def test_loss_coverage_rejects_reordered_or_duplicate_dates():
    rows = [dict(initialization_time=stamp, baseline_per_step=[1.] * 40,
                 full_per_step=[.9] * 40) for stamp in ("first", "second")]
    result = dict(original_graphcast_loss_per_initialization=rows)
    with pytest.raises(ValueError, match="order"):
        reporter._insert_losses({}, result, ["second", "first"])
    with pytest.raises(ValueError, match="Duplicate"):
        reporter._insert_losses({"first": None}, result, ["first", "second"])


def test_reset_loader_aligns_compact_ids_with_iso_timestamps(tmp_path):
    stamp, compact = "2023-01-01T00:00:00Z", "20230101T000000Z"
    output = tmp_path / "evaluate"
    extreme_dir = output / "extremes"
    extreme_dir.mkdir(parents=True)
    initials = tmp_path / "initializations.json"
    reporter.write_json(initials, {"initializations": [{"initialization_time": stamp}]})
    metadata = dict(protocol_hash="manifest", model_id="R_matched", checkpoint_sha256="reset",
                    baseline_checkpoint_sha256="gc", status="complete", initialization_time=stamp,
                    provenance={"extremes_protocol": {"climatology_metadata": {"sha256": "climate"}}})
    stats = np.broadcast_to(statistics(), (40, 4, 2, 2, 2, 15)).copy()
    stats[:3, 3] = np.nan
    valid = np.ones((40, 4), bool)
    valid[:3, 3] = False
    record = extreme_dir / f"{compact}.npz"
    np.savez(record, metadata=json.dumps(metadata), stats=stats, valid_step_event=valid,
             events=reporter.EVENTS, thresholds=reporter.THRESHOLDS, domains=reporter.DOMAINS,
             branches=("baseline", "full"), statistics=reporter.STAT_NAMES,
             lead_hours=np.arange(1, 41) * 6, negative_tp6_counts=np.zeros((40, 3), int),
             negative_tp6_branches=("truth", "baseline", "full"))
    metrics_path = output / "metrics.json"
    reporter.write_json(metrics_path, dict(evaluation_status="complete", evaluated_samples=1,
        target_steps=40, rs_reset_every_step=True, checkpoint_sha256="reset", baseline_checkpoint_sha256="gc",
        original_graphcast_loss_per_initialization=[dict(initialization_time=stamp,
            baseline_per_step=[1.] * 40, full_per_step=[.8] * 40)]))

    def pin(path):
        return dict(path=str(path), sha256=reporter.sha(path))

    complete = dict(status="complete", manifest_hash="manifest", reset_state_every_step=True,
                    initialization_ids=[compact], initialization_times=[stamp], metrics=pin(metrics_path),
                    records=[pin(record)])
    reporter.write_json(output / "complete.json", complete)
    manifest = dict(dates=[stamp], initialization_ids=[compact], manifest_hash="manifest",
                    checkpoint={"sha256": "reset"}, baseline={"checkpoint_sha256": "gc"},
                    climatology_metadata={"sha256": "climate"},
                    shards=[dict(initializations=pin(initials), count=1, output_dir=str(output))])
    arrays = reporter._load_reset(manifest, [])
    assert arrays["loss"].shape == (1, 40, 2)
    assert arrays["loss"][0, 0, 1] == .8
    complete["initialization_ids"] = [stamp]
    reporter.write_json(output / "complete.json", complete)
    with pytest.raises(ValueError, match="Worker dates"):
        reporter._load_reset(manifest, [])


def test_report_exports_only_three_models_and_paired_memory_contrast(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    stats = np.broadcast_to(statistics(), (122, 40, 4, 2, 2, 15)).copy()
    stats[:, :3, 3] = 0
    arrays = {model: dict(stats=stats, loss=np.full((122, 40), loss))
              for model, loss in (("GC", 1.), ("M1", .8), ("R_matched", .9))}
    manifest = dict(workspace=str(tmp_path), manifest_hash="manifest")
    monkeypatch.setattr(reporter, "load_campaign", lambda unused: (manifest, arrays, [], []))
    counts = reporter.circular_block_counts(122, replicates=20)
    monkeypatch.setattr(reporter, "circular_block_counts", lambda unused: counts)
    payload = reporter.report(root)
    assert {m["id"] for m in payload["models"]} == {"GC", "M1", "R_matched"}
    assert dict(model="M1", reference="R_matched") in payload["comparisons"]
    assert not any("pristine" in text or "explored" in text for text in payload["limitations"])
    out = tmp_path / "plots/analyze_models/data/resolution_eval/campaign"
    for name in ("report.json", "report.md", "endpoints.csv", "per_lead.csv", "accuracy.csv"):
        assert (out / name).is_file()
    images = tmp_path / "plots/analyze_models/images/resolution_eval/campaign"
    assert (images / "extreme_skill.pdf").is_file()
    assert (images / "memory_effect.pdf").is_file()
    complete = json.loads((root / "complete.json").read_text())
    assert complete["report_sha256"] == reporter.sha(out / "report.json")
