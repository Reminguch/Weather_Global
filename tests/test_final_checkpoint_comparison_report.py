"""Final raw/SWA reports compare matched training identities and pooled scores."""
from copy import deepcopy
import csv

import numpy as np
import pytest

from scripts.analyze_models import report_v24_raw_checkpoint_pilot as reporter
from scripts.experiments.run_v24_mgda import seal, unseal


def model_pair(pair="gcm_l000", family="gc_mamba", weight=0.):
    return [{"id": f"{pair}_{kind}", "label": f"{pair} {kind}", "model_family": family,
             "lambda": weight, "step": 2500, "training_updates": 2500,
             "checkpoint_kind": kind, "comparison_pair": pair,
             "run_config": {"path": f"/{pair}/run_config.json", "sha256": pair},
             "checkpoint": {"path": f"/{pair}/{kind}.pkl", "sha256": kind},
             "swa_steps": [1000, 1500, 2000, 2500] if kind == "swa" else []}
            for kind in ("raw", "swa")]


def scores(rollout, day10, power, day10_power):
    return {"summary": {"exact_gc_mse_rollout": rollout, "exact_gc_mse_day10": day10,
                        "power_error_rollout_pct": power, "power_error_day10_pct": day10_power}}


def test_paired_changes_use_pooled_raw_loss_and_preserve_finite_worsening():
    models = model_pair()
    series = {models[0]["id"]: scores(10., 4., 30., 50.),
              models[1]["id"]: scores(9., 4.1, 32., 45.)}
    row, = reporter.paired_summary(models, series, .2, .2)
    assert row["swa_mse_reduction_vs_raw_rollout_pct"] == pytest.approx(10.)
    assert row["swa_mse_reduction_vs_raw_day10_pct"] == pytest.approx(-2.5)
    assert row["swa_minus_raw_power_error_rollout_pp"] == 2.
    assert row["swa_minus_raw_power_error_day10_pp"] == -5.
    assert row["swa_mse_decrease_exceeds_baseline_repeat_spread_rollout"] is True
    assert row["swa_mse_decrease_exceeds_baseline_repeat_spread_day10"] is False
    assert row["swa_steps"] == "1000,1500,2000,2500"


@pytest.mark.parametrize("mismatch", ["run_config", "lambda", "training_updates", "model_family"])
def test_pair_rejects_different_training_identity(mismatch):
    models = model_pair()
    models[1][mismatch] = "different"
    with pytest.raises(ValueError, match="different training identities"):
        reporter.checkpoint_pairs(models)


def test_pair_requires_exactly_one_raw_and_one_swa_checkpoint():
    models = model_pair()
    with pytest.raises(ValueError, match="Incomplete"):
        reporter.checkpoint_pairs(models[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        reporter.checkpoint_pairs(models + [deepcopy(models[0])])


def test_zero_raw_loss_keeps_relative_change_undefined():
    models = model_pair()
    series = {models[0]["id"]: scores(0., 0., 30., 50.),
              models[1]["id"]: scores(1., 1., 32., 45.)}
    row, = reporter.paired_summary(models, series, .2, .2)
    assert row["swa_mse_reduction_vs_raw_rollout_pct"] is None
    assert row["swa_mse_reduction_vs_raw_day10_pct"] is None


def test_raw_swa_styles_share_colors_but_distinguish_weight_averaging():
    models = model_pair("M1", weight=None) + model_pair() + model_pair("gcm_l025", weight=.25)
    styles = reporter.figure_styles(models, {"public_gc": {}, **{m["id"]: {} for m in models}})
    for pair in ("M1", "gcm_l000", "gcm_l025"):
        assert styles[f"{pair}_raw"]["color"] == styles[f"{pair}_swa"]["color"]
        assert styles[f"{pair}_raw"]["linestyle"] == "-"
        assert styles[f"{pair}_swa"]["linestyle"] == "--"
    assert styles["gcm_l000_raw"]["color"] != styles["gcm_l025_raw"]["color"]
    old = [{"id": "M1_raw", "model_family": "gc_mamba"},
           {"id": "M1_swa", "model_family": "gc_mamba"}]
    old_styles = reporter.figure_styles(old, {m["id"]: {} for m in old})
    assert old_styles["M1_raw"]["color"] != old_styles["M1_swa"]["color"]
    assert reporter.checkpoint_pairs(old) == {}


@pytest.mark.parametrize("full_gc_positive_rate", [None, 3e-7, 1e-5])
def test_final_report_records_pairs_and_warnings_without_loss_gate(tmp_path, monkeypatch, full_gc_positive_rate):
    root, data, images = (tmp_path / name for name in ("campaign", "data", "images"))
    root.mkdir()
    models = model_pair("M1", weight=None) + model_pair()
    if full_gc_positive_rate is not None:
        models = model_pair("M1", weight=None) + model_pair("gc_l000", "full_gc", 0.) + model_pair("gc_l025", "full_gc", .25)
        for model in models[2:]:
            model["learning_rate"] = 3e-7 if model["lambda"] == 0 else full_gc_positive_rate
    manifest = seal(root / "manifest.json", {"models": models,
        "data_root": str(data), "image_root": str(images),
        "report_title": "Final 2500 raw/SWA comparison", "figure_title": "Final 2500",
        "report_limits": "Four-date final-checkpoint development comparison."})
    dates = [f"2022-{month:02d}-02T00:00:00Z" for month in (1, 4, 7, 10)]
    channels = [["2m_temperature", None], ["geopotential", 500], ["u_component_of_wind", 850]]
    channels += [["fixture", n] for n in range(80)]
    truth = np.broadcast_to(1., (40, 83, 180))
    items = {}
    for index, model in enumerate(models):
        # M1 SWA is deliberately worse; public-GC repeat differences exceed the
        # reference bound. Both remain reportable finite observations.
        full_loss = (2.1, 2.2, 1.8, 1.7, 1.9, 1.8)[index]
        baseline_loss = 2. + index * .05
        items[model["id"]] = {"model": model, "loss": {
            "baseline": np.full(40, baseline_loss), "baseline_rollout": baseline_loss,
            "full": np.full(40, full_loss), "full_rollout": full_loss}, "power": {
                "truth_power": truth, "prediction_power": np.broadcast_to(
                    np.array([1.1, 1.2 + index * .01])[None, :, None, None], (40, 2, 83, 180))}}
    monkeypatch.setattr(reporter, "load_models", lambda *args: (items, {"channels": channels}, dates, []))
    figure_titles = []
    def fake_figures(*args, figure_title):
        figure_titles.append(figure_title)
        return []
    monkeypatch.setattr(reporter, "make_figures", fake_figures)
    result = reporter.report(root)
    assert result["status"] == "complete"
    assert result["numerical_warnings"] > 0
    payload = unseal(data / "report.json")
    assert payload["manifest_hash"] == manifest["record_hash"]
    assert len(payload["raw_vs_swa"]) == len(models) // 2
    assert payload["raw_vs_swa"][0]["swa_mse_reduction_vs_raw_rollout_pct"] < 0
    assert payload["raw_vs_swa_csv"]["path"] == str(data / "raw_vs_swa.csv")
    assert figure_titles == ["Final 2500"]
    text = (data / "report.md").read_text()
    assert text.startswith("# Final 2500 raw/SWA comparison")
    assert "Matched raw versus SWA weights" in text
    assert "intermediate" not in text
    assert ("**Learning-rate caveat:**" in text) == (full_gc_positive_rate == 1e-5)
    with (data / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["checkpoint_kind"] == "raw"
    assert rows[2]["checkpoint_kind"] == "swa"
    assert rows[1]["training_updates"] == "2500"
