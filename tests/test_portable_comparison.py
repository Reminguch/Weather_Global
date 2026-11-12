"""Portable campaign counters, development-only selection and paired inference."""
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import run_v24_portable_comparison as portable
from scripts.experiments.frozen_runtime import freeze_source
from scripts.experiments.run_v24_mgda import pinned, read, seal, unseal, write


def package(tmp_path):
    root = tmp_path / "package"
    for relative in ("src/stub.py", "scripts/stub.py", "third_party/graphcast/graphcast/graphcast.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    baseline = root / "baseline.npz"; baseline.write_bytes(b"public GC")
    stats = root / "stats"; stats.mkdir()
    for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"):
        (stats / name).write_bytes(name.encode())
    data = root / "data.nc"; data.write_bytes(b"ERA5 fixture")
    cohort = root / "cohort.json"
    write(cohort, {"initializations": [{"initialization_time": "2022-01-01T00:00:00Z"}]})
    return root, baseline, stats, data, cohort


def training(tmp_path, package_root, baseline, stats, family, values):
    campaign = tmp_path / family
    tasks = []
    for value in values:
        identity = f"{family}_{value}"
        cfg = {"model_family": family, "architecture": {},
               "data": {"baseline_checkpoint": str(baseline), "stats_dir": str(stats)}}
        path = campaign / "configs" / f"{identity}.json"
        path.parent.mkdir(parents=True, exist_ok=True); write(path, cfg)
        task = {"id": identity, "logical_id": identity, "model_family": family,
                "lambda": value if family == "gc_mamba" else 0., "config": pinned(path)}
        if family == "full_gc":
            task["horizon"] = value
        tasks.append(task)
        for step in ([500, 1000, 1500, 2000] if family == "gc_mamba" else portable.FULL_GC_POINTS):
            ckpt = campaign / "runs" / identity / "checkpoints" / f"checkpoint_step{step:08d}.pkl"
            ckpt.parent.mkdir(parents=True, exist_ok=True); ckpt.write_bytes(f"{identity} {step}".encode())
    source = freeze_source(package_root, campaign / "source")
    manifest = seal(campaign / "manifest.json", {"experiment": str(campaign), "tasks": tasks,
        "source_root": str(campaign / "source"), "source_digest": source["source_digest"]})
    for task in tasks:
        step = 2000 if family == "gc_mamba" else 2500
        seal(campaign / "checks" / f"{task['id']}_total00002500.json", {
            "status": "complete", "manifest_hash": manifest["record_hash"], "task_id": task["id"],
            "completed_steps": step, "total_training_updates": 2500,
            "checkpoint": pinned(campaign / "runs" / task["id"] / "checkpoints" / f"checkpoint_step{step:08d}.pkl")})
    return campaign


def fake_swa(root, spec):
    path = root / "swa" / f"{spec['task']['id']}.pkl"
    path.write_bytes(b"SWA fixture")
    return pinned(path)


def test_portable_combined_prepare_and_checkpoint_counters(tmp_path, monkeypatch):
    root, baseline, stats, data, cohort = package(tmp_path)
    gcm = training(tmp_path, root, baseline, stats, "gc_mamba", [0., .25, .5, .75, 1.])
    gc = training(tmp_path, root, baseline, stats, "full_gc", [8, 12, 16, 20])
    monkeypatch.setattr(portable, "build_swa", fake_swa)
    target = tmp_path / "comparison"
    result = portable.prepare(target, root, [gcm, gc], data, cohort, 2022)
    assert len(result["models"]) == 34
    gcm_models = [m for m in result["models"] if m["model_family"] == "gc_mamba"]
    assert len(gcm_models) == 10
    assert {m["step"] for m in gcm_models} == {2000}
    assert {m["training_updates"] for m in gcm_models} == {2500}
    assert [m for m in gcm_models if m["lambda"] == .75 and m["checkpoint_kind"] == "swa"][0]["swa_steps"] == [500, 1000, 1500, 2000]
    gc_models = [m for m in result["models"] if m["model_family"] == "full_gc"]
    assert len(gc_models) == 24
    assert all(m["swa_steps"] == [1000, 1500, 2000, 2500] for m in gc_models if m["checkpoint_kind"] == "swa")
    assert portable.verify(target)["record_hash"] == result["record_hash"]
    assert "dabanin" not in str(result)


def test_rejects_wrong_final_counter_before_creating_comparison(tmp_path, monkeypatch):
    root, baseline, stats, data, cohort = package(tmp_path)
    gcm = training(tmp_path, root, baseline, stats, "gc_mamba", [0.])
    task = unseal(gcm / "manifest.json")["tasks"][0]
    proof = gcm / "checks" / f"{task['id']}_total00002500.json"
    value = unseal(proof); value.pop("record_hash"); value["completed_steps"] = 2500
    proof.unlink()
    seal(proof, value)
    target = tmp_path / "comparison"
    with pytest.raises(ValueError, match="final training proof"):
        portable.prepare(target, root, [gcm], data, cohort, 2022)
    assert not target.exists()


def development_rows():
    return [{"training_run_id": "k8", "checkpoint_kind": "raw", "checkpoint_step": step,
             "exact_gc_mse_rollout": loss} for step, loss in zip(portable.FULL_GC_POINTS, [5., 3., 3., 4., 6.])]


def test_selection_uses_2022_only_and_tie_prefers_earlier_step():
    report = {"status": "complete", "year": 2022, "summary": development_rows()}
    assert portable.selected_full_gc_steps(report, [{"id": "k8"}]) == {"k8": [1000, 2500]}
    report["year"] = 2023
    with pytest.raises(ValueError, match="2022"):
        portable.selected_full_gc_steps(report, [{"id": "k8"}])


def test_selection_does_not_accept_partial_or_nonfinite_scores():
    rows = development_rows()
    with pytest.raises(ValueError, match="five raw"):
        portable.selected_full_gc_steps({"status": "complete", "year": 2022, "summary": rows[:-1]}, [{"id": "k8"}])
    rows[0]["exact_gc_mse_rollout"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        portable.selected_full_gc_steps({"status": "complete", "year": 2022, "summary": rows}, [{"id": "k8"}])


def test_annual_intervals_preserve_paired_constant_effect():
    dates = [f"2023-01-{day:02d}T00:00:00Z" for day in (1, 4, 7, 10, 13, 16)]
    cases = [{"baseline_per_step": [10.] * 40, "full_per_step": [8.] * 40} for _ in dates]
    items = {"model": {"result": {"original_graphcast_loss_per_initialization": cases}}}
    result = portable.annual_intervals(items, dates, "model")
    assert result["block_starts"] == 5
    assert len(result["contrasts"]) == 2
    assert all(row["advantage_pp_of_gc_loss"] == row["ci95_low"] == row["ci95_high"] == 20. for row in result["contrasts"])


def test_annual_intervals_reject_irregular_dates():
    dates = [f"2023-01-{day:02d}T00:00:00Z" for day in (1, 4, 7, 10, 13, 17)]
    with pytest.raises(ValueError, match="regularly spaced"):
        portable.annual_intervals({}, dates, "unused")


def test_cohort_rejects_wrong_year_or_duplicate_dates():
    with pytest.raises(ValueError, match="year"):
        portable.dates_from({"initializations": [{"initialization_time": "2023-01-01T00:00:00Z"}]}, 2022)
    with pytest.raises(ValueError, match="unique"):
        portable.dates_from({"initializations": [{"initialization_time": "2022-01-01T00:00:00Z"}] * 2}, 2022)


def test_zarr_metadata_is_supported_and_changes_are_rejected(tmp_path):
    path = tmp_path / "era5.zarr"; path.mkdir()
    (path / ".zgroup").write_text('{"zarr_format": 2}\n')
    (path / ".zmetadata").write_text('{"metadata": {}}\n')
    record = portable.evaluation_data_record(path)
    assert record["kind"] == "zarr"
    portable.verify_evaluation_data(record)
    (path / ".zmetadata").write_text('{"metadata": {"changed": true}}\n')
    with pytest.raises(ValueError):
        portable.verify_evaluation_data(record)


def test_annual_preparation_selects_only_from_same_campaigns(tmp_path, monkeypatch):
    root, baseline, stats, data, cohort = package(tmp_path)
    gc = training(tmp_path, root, baseline, stats, "full_gc", [8])
    write(cohort, {"initializations": [{"initialization_time": "2023-01-01T00:00:00Z"}]})
    task = unseal(gc / "manifest.json")["tasks"][0]
    rows = development_rows()
    for row in rows:
        row["training_run_id"] = task["id"]
    development = tmp_path / "development.json"
    seal(development, {"status": "complete", "year": 2022, "summary": rows,
                       "training_campaigns": [pinned(gc / "manifest.json")]})
    monkeypatch.setattr(portable, "build_swa", fake_swa)
    result = portable.prepare(tmp_path / "annual", root, [gc], data, cohort, 2023, development_report=development)
    assert [(m["step"], m["checkpoint_kind"]) for m in result["models"]] == [(1000, "raw"), (2500, "raw"), (2500, "swa")]
    bad = unseal(development); bad.pop("record_hash"); bad["training_campaigns"] = []
    development.unlink()
    seal(development, bad)
    with pytest.raises(ValueError, match="different training campaigns"):
        portable.prepare(tmp_path / "wrong_annual", root, [gc], data, cohort, 2023, development_report=development)


def test_complete_report_plots_and_preserves_finite_drift_warning(tmp_path, monkeypatch):
    channels = [["2m_temperature", None], ["geopotential", 500], ["u_component_of_wind", 850]]
    channels += [[f"fixture_{i}", None] for i in range(80)]
    truth = np.ones((40, 83, 180))
    models, items = [], {}
    for kind, loss, baseline in (("raw", 8., 10.), ("swa", 7.9, 10.3)):
        model = {"id": f"gcm_{kind}", "label": f"GCM {kind}", "model_family": "gc_mamba",
            "lambda": 0., "horizon": 20, "step": 2000, "checkpoint_kind": kind,
            "training_updates": 2500, "training_run_id": "gcm", "comparison_pair": "gcm",
            "run_config": {"path": "same.json", "sha256": "same"}}
        if kind == "swa":
            model["swa_steps"] = [500, 1000, 1500, 2000]
        models.append(model)
        items[model["id"]] = {"model": model,
            "loss": {"baseline": np.full(40, baseline), "baseline_rollout": baseline,
                     "full": np.full(40, loss), "full_rollout": loss},
            "power": {"truth_power": truth, "prediction_power": np.stack([truth * 1.2, truth * 1.1], axis=1),
                      "error_power": np.stack([truth * .1, truth * .05], axis=1),
                      "cross_power": np.stack([truth, truth], axis=1)}}
    monkeypatch.setattr(portable, "load_results", lambda *args: (items, {"channels": channels}, ["2022-01-01", "2022-07-01"]))
    (tmp_path / "report").mkdir()
    manifest = {"models": models, "year": 2022, "count": 2, "record_hash": "fixture",
                "training_campaigns": [], "limits": "Synthetic verification only"}
    result = portable.report(tmp_path, manifest)
    assert result["status"] == "complete"
    saved = unseal(tmp_path / "report" / "report.json")
    assert saved["warnings"]
    assert saved["baseline_repeat_spread_rollout"] == pytest.approx(.3)
    assert saved["summary"][1]["mse_reduction_rollout_pct"] == pytest.approx(20.)
    assert saved["summary"][1]["power_error_rollout_pct"] == pytest.approx(10.)
    assert len(saved["figures"]) == 5 and all(Path(path).stat().st_size > 0 for path in saved["figures"])
    assert (tmp_path / "report" / "raw_vs_swa.csv").is_file()
