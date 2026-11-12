"""Portable filter campaigns retain fixed cohorts, filter policy and provenance."""
from datetime import datetime, timedelta
from pathlib import Path
import json

import pytest

from scripts.experiments import run_v24_portable_smoothing as portable


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    package = tmp_path / "package"
    for relative in ("src/fixture.py", "scripts/fixture.py", "third_party/graphcast/graphcast/graphcast.py"):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture\n")
    cohorts = {}
    for name, count, year in (("development32", 32, 2022), ("annual122", 122, 2023)):
        dates = [(datetime(year, 1, 1) + timedelta(days=3*i)).strftime("%Y-%m-%dT00:00:00Z")
                 for i in range(count)]
        path = package / "portable" / f"{name}.json"
        portable.write(path, {"initializations": [{"initialization_time": value} for value in dates]})
        cohorts[name] = path
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config = inputs / "run_config.json"
    portable.write(config, {"model_family": "full_gc", "architecture": {"resolution": 1.0}})
    paths = {"checkpoint": inputs / "checkpoint.pkl", "baseline_checkpoint": inputs / "baseline.npz"}
    for name, path in paths.items():
        path.write_text(name)
    stats = inputs / "stats"
    stats.mkdir()
    expected = [{"input": "baseline_checkpoint", "relative_path": "",
                 "sha256": portable.pinned(paths["baseline_checkpoint"])["sha256"]}]
    for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"):
        path = stats / name
        path.write_text(name)
        expected.append({"input": "stats_dir", "relative_path": name,
                         "sha256": portable.pinned(path)["sha256"]})
    portable.write(package / "portable/input_reference.json", {"files": expected})
    for year in (2022, 2023):
        data = inputs / f"data{year}.zarr"
        data.mkdir()
        (data / ".zmetadata").write_text("{}")
    monkeypatch.setattr(portable, "ROOT", package)
    kwargs = dict(checkpoint=paths["checkpoint"], run_config=config, model_id="full_gc_k08_raw",
                  development_data=inputs / "data2022.zarr", annual_data=inputs / "data2023.zarr",
                  development_manifest=cohorts["development32"], annual_manifest=cohorts["annual122"],
                  baseline_checkpoint=paths["baseline_checkpoint"], stats_dir=stats)
    return package, tmp_path / "fitted", kwargs


def test_prepare_preserves_cohort_partition_and_original_scientific_policy(fixture):
    package, root, kwargs = fixture
    manifest = portable.prepare(package, root, **kwargs)
    assert manifest["model_family"] == "full_gc"
    assert manifest["training_campaign"] is None
    assert [x["count"] for x in manifest["shards"]["calibrate"]] == [1, 11, 10, 10]
    assert [x["count"] for x in manifest["shards"]["evaluate"]] == [1, 18, 18, 17, 17, 17, 17, 17]
    assert manifest["policy"]["gaussian_widths_degrees"] == [0, .25, .5, 1, 2, 4, 8]
    assert manifest["policy"]["knots"] == [1, 10, 40, 179]
    assert manifest["policy"]["eta"] == .01
    assert manifest["policy"]["feedback"] == "raw forecast"
    assert manifest["policy"]["finite_drift"] == "warning"
    assert portable.verify(root, frozen=False)["record_hash"] == manifest["record_hash"]
    with pytest.raises(FileExistsError):
        portable.prepare(package, root, **kwargs)


def test_cohort_change_fails_before_creating_output(fixture, tmp_path):
    package, root, kwargs = fixture
    changed = portable.read(kwargs["development_manifest"])
    changed["initializations"] = changed["initializations"][1:]
    path = tmp_path / "changed.json"
    portable.write(path, changed)
    kwargs["development_manifest"] = path
    with pytest.raises(ValueError, match="Cohort differs"):
        portable.prepare(package, root, **kwargs)
    assert not root.exists()


def test_baseline_or_stats_change_fails_before_creating_output(fixture):
    package, root, kwargs = fixture
    kwargs["baseline_checkpoint"].write_text("wrong public GC weights")
    with pytest.raises(ValueError, match="Baseline/statistics differ"):
        portable.prepare(package, root, **kwargs)
    assert not root.exists()


def test_strength_sweep_accepts_full_gc_and_retains_fitted_filters(fixture, tmp_path):
    package, parent, kwargs = fixture
    original = portable.prepare(package, parent, **kwargs)
    portable.seal(parent / "filters.json", {"campaign_hash": original["record_hash"],
        "branches": ["GC", kwargs["model_id"]], "fixture_gains": [1., .9, .8]})
    portable.seal(parent / "report.json", {"campaign_hash": original["record_hash"], "status": "complete"})
    root = tmp_path / "strength"
    manifest = portable.prepare_sweep(package, root, parent)
    assert manifest["model_id"] == "full_gc_k08_raw"
    assert sum(x["count"] for x in manifest["shards"]["evaluate"]) == 122
    assert manifest["sweep_policy"]["profile_blend_alphas"] == [i / 10 for i in range(11)]
    assert manifest["sweep_policy"]["fixed_gaussian_widths_degrees"] == [0., .25, .5, .75, 1., 1.5, 2., 3., 4., 8.]
    assert portable.verify(root, sweep=True, frozen=False)["record_hash"] == manifest["record_hash"]
    filters = portable.unseal(root / "filters.json")
    assert filters["fixture_gains"] == [1., .9, .8]
    assert filters["origin_campaign_hash"] == original["record_hash"]


def test_unfinished_parent_cannot_start_strength_sweep(fixture, tmp_path):
    package, parent, kwargs = fixture
    manifest = portable.prepare(package, parent, **kwargs)
    portable.seal(parent / "filters.json", {"campaign_hash": manifest["record_hash"]})
    portable.seal(parent / "report.json", {"campaign_hash": "different", "status": "complete"})
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        portable.prepare_sweep(package, tmp_path / "strength", parent)


def test_worker_dispatch_uses_original_forecast_and_fit_functions(monkeypatch, tmp_path):
    manifest = {"shards": {"calibrate": [{"count": 1}]}}
    monkeypatch.setattr(portable, "verify", lambda *args, **kwargs: manifest)
    calls = []
    monkeypatch.setattr(portable.original, "run_worker", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    monkeypatch.setattr(portable.sys, "argv", ["driver", "calibrate", "--experiment", str(tmp_path), "--task", "0"])
    portable.main()
    assert calls == [((tmp_path, manifest, "calibrate", 0), {"observer_class": None})]
    calls.clear()
    monkeypatch.setattr(portable.original, "fit", lambda *args: calls.append(args) or {})
    monkeypatch.setattr(portable.sys, "argv", ["driver", "fit", "--experiment", str(tmp_path)])
    portable.main()
    assert calls == [(tmp_path, manifest)]


def test_strength_report_changes_only_hardcoded_model_prose(monkeypatch, tmp_path):
    from scripts.analyze_models import report_v24_smoothing_sweep
    path = tmp_path / "report.md"
    path.write_text("# GC/M1 smoothing strength tradeoffs\nfixed GC and M1 checkpoints.\nExact MSE=1.23\n")
    monkeypatch.setattr(report_v24_smoothing_sweep, "report", lambda *args: {"data_dir": str(tmp_path)})
    portable.report_sweep(tmp_path, {"model_id": "full_gc_k08"})
    assert "GC/full_gc_k08" in path.read_text()
    assert "fixed GC and full_gc_k08 checkpoints." in path.read_text()
    assert "Exact MSE=1.23" in path.read_text()
