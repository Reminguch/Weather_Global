"""Scientific aggregation and endpoint identity for the split-MGDA report."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.analyze_models import report_v24_amse_split_mgda as reporting


def result(loss, rmse, baseline_loss=10.):
    per_step = np.tile([loss, 3 * loss], 20)
    field = {"rmse_full": np.tile([rmse, 3 * rmse], 20).tolist(),
             "rmse_baseline": np.full(40, 10.).tolist()}
    return {"target_steps": 40, "original_graphcast_loss": {
        "full_per_step": per_step.tolist(), "full_rollout": float(per_step.mean()),
        "baseline_per_step": np.full(40, baseline_loss).tolist(), "baseline_rollout": baseline_loss},
        "per_channel_per_step": {"temperature_level850": copy.deepcopy(field)},
        "per_variable_per_step": {"temperature": copy.deepcopy(field)}}


def test_accuracy_pools_losses_and_squared_errors_before_ratios():
    candidate, initial, baseline = result(1, 1), result(2, 2), result(5, 5)
    # A varying baseline makes mean-of-ratios differ from ratio-of-means.
    baseline["original_graphcast_loss"]["baseline_per_step"] = np.tile([1., 9.], 20).tolist()
    summary = reporting.accuracy_summary(candidate, initial, baseline)
    assert summary["loss_rollout"] == 2
    assert summary["loss_day10"] == 3
    assert summary["loss_improvement_vs_gc_pct"] == pytest.approx(60)
    assert summary["loss_improvement_vs_initialization_pct"] == pytest.approx(50)
    assert summary["physical_fields"][0]["rmse_rollout"] == pytest.approx(np.sqrt(5))
    assert summary["physical_fields"][0]["rmse_rollout"] != 2


def test_zero_reference_and_zero_power_are_explicit():
    assert reporting.ratio(1, 0) is None
    assert reporting.improvement(0, 0) is None
    amplitude, pattern = reporting.physical_components([0, 4, 4], [0, 0, 4], [0, 0, -4])
    np.testing.assert_array_equal(amplitude, [0, 4, 0])
    np.testing.assert_array_equal(pattern, [0, 8, 16])


def spherical_fixture(tmp_path, model="model"):
    directory = tmp_path / model
    folder = directory / "spherical"
    folder.mkdir(parents=True)
    stamps = ["2023-01-15T00:00:00Z", "2023-02-15T00:00:00Z"]
    spec = {"initialization_times": stamps, "target_steps": 40, "model_id": model,
            "protocol_hash": f"protocol-{model}", "checkpoint_sha256": f"checkpoint-{model}",
            "baseline": {"checkpoint_sha256": "gc"}}
    for index, stamp in enumerate(stamps):
        truth = np.ones((40, 1, 2))
        power = truth * (1 if index == 0 else 9)
        metadata = {"status": "complete", "initialization_time": stamp,
                    **{key: spec[key] for key in ("model_id", "protocol_hash", "checkpoint_sha256")},
                    "baseline_checkpoint_sha256": "gc"}
        np.savez(folder / f"sample{index}.npz", metadata=np.asarray(json.dumps(metadata)),
                 variables=np.asarray(["temperature_level850"]), units=np.asarray(["K"]),
                 degree=np.arange(2), lead_hours=np.arange(6, 241, 6),
                 truth_power=truth, baseline_power=truth * 4, full_power=power,
                 baseline_cross_power=truth * 2, full_cross_power=np.sqrt(power),
                 baseline_error_power=truth, full_error_power=(np.sqrt(power) - 1) ** 2,
                 full_grid_mse=np.ones((40, 1)))
    return directory, spec


def test_spherical_components_average_per_case_before_pooling(tmp_path):
    directory, spec = spherical_fixture(tmp_path)
    data = reporting.load_spherical(directory, spec)
    np.testing.assert_array_equal(data["pooled"]["full_power"], 5)
    np.testing.assert_array_equal(data["components"]["full_amplitude"], 2)
    assert not np.allclose(data["components"]["full_amplitude"], (np.sqrt(5) - 1) ** 2)
    np.testing.assert_allclose(data["components"]["full_pattern"], 0)
    assert data["degree"] == [0, 1]
    assert "full_grid_mse" in data["pooled"]
    spec["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        reporting.load_spherical(directory, spec)


def test_training_counts_only_canonical_complete_histories(tmp_path):
    for family in ("di16", "di32"):
        folder = tmp_path / family
        folder.mkdir()
        rows = [{"step": step, "amplitude_loss": 1., "pattern_loss": 2., "gc_loss": 3.,
                 "mgda_alpha": .5, "update_norm": 0 if step == 100 else .1}
                for step in range(1, 2001)]
        (folder / "train_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        # Shadow metrics must never enter the canonical summaries.
        (folder / "shadow").mkdir()
        (folder / "shadow/train_metrics.jsonl").write_text('{"step": 100, "update_norm": 999}\n')
    output = reporting.training_records(tmp_path)
    assert output["di16"]["steps"] == 2000
    assert output["di16"]["exact_zero_update_count"] == 1
    assert output["di16"]["small_update_count"] == 1
    with (tmp_path / "di16/train_metrics.jsonl").open("a") as stream:
        stream.write('\n{"step": 2000}\n')
    with pytest.raises(ValueError, match="canonical updates"):
        reporting.training_records(tmp_path)


def test_report_pairs_each_width_and_uses_one_baseline(tmp_path, monkeypatch):
    tasks, data = [], {}
    for family, starting_loss in (("di16", 2.), ("di32", 4.)):
        for step, fraction in ((0, 1), (500, .9), (2000, .8)):
            identity = f"{family}_step{step:05d}"
            tasks.append({"id": identity, "family": family, "step": step,
                          "paired_initialization_id": f"{family}_step00000"})
            directory, spec = spherical_fixture(tmp_path, identity)
            spec["chosen_idx"] = [1, 2]
            data[identity] = (directory, spec, result(starting_loss * fraction, fraction,
                                                    baseline_loss=10 if family == "di16" else 20), False)
    manifest = {"evaluation_role": "development", "task_specs": tasks,
                "canonical_mse_id": "di16_step00000", "output_dir": str(tmp_path / "evaluation")}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(reporting.evaluation, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(reporting.evaluation, "resolve_result", lambda _, index, stage: data[tasks[index]["id"]])
    monkeypatch.setattr(reporting.evaluation, "frozen_file", lambda p: {"path": str(p), "sha256": "fixture"})

    def zonal(directory, spec, result):
        spherical = reporting.load_spherical(directory, spec)
        return {**spherical, "wavenumber": np.array([0, 1]), "result": result}

    monkeypatch.setattr(reporting.evaluation, "load_spectral_result", zonal)
    monkeypatch.setattr(reporting.evaluation, "feature_rows", lambda _: [])
    destination = reporting.build_report(path)
    report = json.loads((destination / "split_mgda_report.json").read_text())
    assert report["primary_endpoint"] == "step02000"
    assert report["selection_allowed"] is False
    assert report["accounting"]["canonical_optimizer_updates"] == 4000
    di32 = next(record for record in report["records"] if record["id"] == "di32_step02000")
    assert di32["paired_initialization_id"] == "di32_step00000"
    assert di32["accuracy"]["spectral"]["loss_improvement_vs_initialization_pct"] == pytest.approx(20)
    # It uses designated GC loss=10, not this task's repeated loss=20.
    assert di32["accuracy"]["spectral"]["loss_improvement_vs_gc_pct"] == pytest.approx(36)
    assert any("drift" in warning for warning in report["warnings"])
    assert (destination / "rmse_rollouts.csv").is_file()
    assert (destination / "spherical_components.csv").is_file()
    assert (destination / "di32_step02000_spherical_pooled.npz").is_file()
    tasks[-1]["paired_initialization_id"] = "di16_step00000"
    with pytest.raises(ValueError, match="own-width"):
        reporting.build_report(path)


def test_overview_preserves_geopotential_and_restricts_plot_count():
    fields = [f"channel_{number}" for number in range(83)] + ["geopotential_level500", "temperature_level850"]
    result = reporting.overview_fields(fields)
    assert len(result) == 16
    assert "geopotential_level500" in result
    assert "temperature_level850" in result


def test_render_generates_absolute_first_and_separate_cohort_figures(tmp_path):
    directory, spec = spherical_fixture(tmp_path)
    spherical = reporting.load_spherical(directory, spec)
    stage = (directory, spec, result(1, 1), False)
    loaded = {"di16_step00000": {"task": {"family": "di16", "step": 0},
                               "stages": {"exact": stage, "spectral": stage},
                               "spherical": spherical,
                               "zonal": {**spherical, "wavenumber": np.array([0, 1])}}}
    path = tmp_path / "report.json"
    path.write_text("{}")
    images = tmp_path / "images"
    reporting.render(loaded, "di16_step00000", {}, images, path)
    figures = json.loads((images / "figures_manifest.json").read_text())["figures"]
    assert {row["cohort"] for row in figures} == {"exact", "spectral"}
    spectral_figures = [row for row in figures if "panel_order" in row]
    assert len(spectral_figures) == 2
    assert all(row["panel_order"][0] == "absolute_power" for row in spectral_figures)
    assert all(__import__("pathlib").Path(row["files"]["png"]).is_file() for row in figures)
