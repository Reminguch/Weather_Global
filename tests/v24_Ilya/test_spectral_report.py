"""Pilot reports must preserve pairing and pool Fourier power before ratios."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_models.report_v24_spectral_pilot import load_spectra, pooled_curves, report
from src.models.mamba.v24_Ilya.spectral_diagnostics import zonal_spectral_metadata


def _rewrite(path, change):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    change(data)
    np.savez_compressed(path, **data)


def _metadata_change(data, **changes):
    metadata = json.loads(str(data["metadata"].item()))
    metadata.update(changes)
    data["metadata"] = np.asarray(json.dumps(metadata))


@pytest.fixture
def workflow(tmp_path, request):
    year = getattr(request, "param", 2022)
    protocol = {
        "protocol_hash": "frozen-protocol", "initialization_times": [f"{year}-01-15T00:00:00Z", f"{year}-07-15T00:00:00Z"],
        "lead_hours": [24, 72, 168, 240], "target_steps": 4,
        "spectral_variables": ["T2m", "TP6"], "spectral_metadata": zonal_spectral_metadata(),
        "models": [{"model_id": key, "checkpoint_sha256": f"checkpoint-{key}", "label": key}
                   for key in ("M1", "M2", "M3")],
        "baseline": {"canonical_model_id": "M1", "checkpoint_sha256": "graphcast-checkpoint",
                     "spectral_repeatability_rtol": 1e-3, "spectral_repeatability_atol": 1e-10},
        "output_image_dir": str(tmp_path / "figures"), "output_data_dir": str(tmp_path / "report"),
    }
    if year != 2022:
        protocol.update({
            "evaluation_year": year, "evaluation_period": f"{year} pilot",
            "scope": f"Exploratory {year} spatial Fourier pilot on two fixed starts; no full-year or preservation claim",
        })
    body = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    for model in protocol["models"]:
        model_id = model["model_id"]
        directory = tmp_path / "models" / model_id
        directory.mkdir(parents=True)
        for index, initialization in enumerate(protocol["initialization_times"]):
            # Start 1 has nine times the power of start 0. This makes averaging
            # per-start ratios provably wrong, while all fields remain finite.
            truth = np.broadcast_to(np.array([0., 1., .5, .2]) * (1 if index == 0 else 9), (4, 2, 4)).copy()
            truth[:, 1, :] *= 1e-12  # Precipitation powers need a small tolerance.
            baseline_amplitude = .8 + ({"M1": 0., "M2": 1e-5, "M3": 2e-5}[model_id])
            amplitude = {"M1": float(index), "M2": .8, "M3": .5}[model_id]
            metadata = {
                "model_id": model_id, "checkpoint_sha256": model["checkpoint_sha256"],
                "protocol_hash": protocol["protocol_hash"], "initialization_time": initialization,
                "baseline_checkpoint_sha256": protocol["baseline"]["checkpoint_sha256"],
                "spectral_metadata": protocol["spectral_metadata"],
                "status": "complete",
            }
            np.savez_compressed(
                directory / f"start{index}.npz", metadata=np.asarray(json.dumps(metadata)),
                lead_hours=np.asarray(protocol["lead_hours"]), variables=np.asarray(protocol["spectral_variables"]),
                wavenumber=np.arange(4), truth_power=truth,
                baseline_power=truth * baseline_amplitude ** 2,
                baseline_cross_power=truth * baseline_amplitude,
                baseline_error_power=truth * (1 - baseline_amplitude) ** 2,
                full_power=truth * amplitude ** 2, full_cross_power=truth * amplitude,
                full_error_power=truth * (1 - amplitude) ** 2,
            )
    sources = []
    for model in protocol["models"]:
        path = tmp_path / "models" / model["model_id"] / "evaluation.json"
        path.write_text(json.dumps({"model_id": model["model_id"], "evaluated_samples": 2}))
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (tmp_path / "exact_loss_validation.json").write_text(json.dumps({
        "status": "passed", "protocol_hash": protocol["protocol_hash"],
        "canonical_baseline_model_id": "M1", "evaluation_sources": sources,
        "models": [{"model_id": model["model_id"]} for model in protocol["models"]],
    }))
    return tmp_path


def test_ratios_and_correlations_use_pooled_power_and_canonical_baseline(workflow):
    protocol, k, pooled, audit = load_spectra(workflow)
    truth, curves = pooled_curves(pooled)
    np.testing.assert_allclose(truth[..., 1], [[5., 5e-12]] * 4)
    np.testing.assert_allclose(curves["M1"]["power_ratio"][..., 1:], .9)
    np.testing.assert_allclose(curves["M1"]["error_ratio"][..., 1:], .1)
    np.testing.assert_allclose(curves["M1"]["correlation"][..., 1:], np.sqrt(.9))
    np.testing.assert_allclose(curves["GraphCast"]["power_ratio"][..., 1:], .64)
    # M2's repeated baseline differs slightly; the report still uses M1 only.
    assert audit["repeatability"]["M2"]["baseline_power"]["max_absolute_difference"] > 0
    assert np.isnan(curves["M1"]["power_ratio"][..., 0]).all()
    assert np.isnan(curves["M1"]["correlation"][..., 0]).all()
    assert len(audit["sources"]) == 6


@pytest.mark.parametrize("key,value", [
    ("protocol_hash", "other-protocol"), ("checkpoint_sha256", "other-checkpoint"),
    ("baseline_checkpoint_sha256", "other-baseline"), ("model_id", "M3"),
    ("spectral_metadata", {"smoothing": "gaussian"}),
    ("initialization_time", "2022-03-15T00:00:00Z"),
    ("status", "running"),
])
def test_rejects_forecasts_with_different_provenance(workflow, key, value):
    _rewrite(workflow / "models/M2/start0.npz", lambda data: _metadata_change(data, **{key: value}))
    with pytest.raises(ValueError):
        load_spectra(workflow)


@pytest.mark.parametrize("mutation,match", [
    ("missing", "expected 2 initialization NPZs"),
    ("duplicate", "duplicate initialization"),
    ("lead", "lead_hours mismatch"),
    ("variables", "variables mismatch"),
    ("nan", "finite and real"),
    ("dc", "DC must be zero"),
    ("error", "spectral error identity"),
    ("cross", "Cauchy-Schwarz"),
    ("drift", "repeatability failed"),
    ("truth_drift", "truth_power repeatability failed"),
])
def test_rejects_incomplete_or_scientifically_inconsistent_records(workflow, mutation, match):
    path = workflow / "models/M2/start0.npz"
    if mutation == "missing":
        path.unlink()
    elif mutation == "duplicate":
        (workflow / "models/M2/start1.npz").write_bytes(path.read_bytes())
    else:
        def change(data):
            if mutation == "lead":
                data["lead_hours"][0] = 12
            elif mutation == "variables":
                data["variables"] = data["variables"][::-1]
            elif mutation == "nan":
                data["full_power"][0, 0, 1] = np.nan
            elif mutation == "dc":
                data["full_power"][0, 0, 0] = 1.
            elif mutation == "error":
                data["full_error_power"][0, 0, 1] += 1.
            elif mutation == "cross":
                data["full_cross_power"][0, 0, 1] *= 2
            elif mutation == "drift":
                for suffix, factor in (("power", .75 ** 2), ("cross_power", .75), ("error_power", .25 ** 2)):
                    data[f"baseline_{suffix}"] = data["truth_power"] * factor
            elif mutation == "truth_drift":
                # Valid individual statistics, but the forecast branches did
                # not see exactly the same target: reject even a tiny drift.
                for key in ("truth_power", "baseline_power", "baseline_cross_power", "baseline_error_power",
                            "full_power", "full_cross_power", "full_error_power"):
                    data[key] *= 1 + 1e-12
        _rewrite(path, change)
    with pytest.raises(ValueError, match=match):
        load_spectra(workflow)


@pytest.mark.parametrize("workflow", [2022, 2023], indirect=True)
def test_report_writes_interpretable_figures_csv_and_auditable_summary(workflow, monkeypatch):
    from matplotlib.figure import Figure

    # Capture actual figure labels so the regression covers the rendered plots,
    # not merely the metadata written beside them.
    labels = []
    figure_text = Figure.text

    def capture_text(self, x, y, s, *args, **kwargs):
        labels.append(s)
        return figure_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Figure, "text", capture_text)
    protocol = json.loads((workflow / "protocol.json").read_text())
    year = protocol.get("evaluation_year", 2022)
    summary = report(workflow)
    assert summary["status"] == "complete"
    assert summary["samples_per_model"] == 2
    assert len(summary["outputs"]["images"]) == 4
    for path in summary["outputs"]["images"]:
        assert Path(path).read_bytes().startswith(b"\x89PNG")
    with Path(summary["outputs"]["pooled_spectra_csv"]).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 4 * 2 * 3
    assert all(int(row["zonal_wavenumber"]) > 0 for row in rows)
    saved = json.loads((workflow / "report/summary.json").read_text())
    assert saved["protocol_hash"] == json.loads((workflow / "protocol.json").read_text())["protocol_hash"]
    assert "No uncertainty" in saved["limitations"][-1]
    assert len(saved["audit"]["sources"][0]["sha256"]) == 64
    assert saved["evaluation_year"] == year
    assert saved["evaluation_period"] == f"{year} pilot"
    assert sum(f"Exploratory {year} pilot" in label for label in labels) == 4
    if year == 2023:
        assert saved["scope"] == protocol["scope"]
        assert "not a full-year evaluation" in saved["limitations"][0]
        assert "2022" not in json.dumps(saved)
        assert "2022" not in " ".join(labels)
        assert "development" not in " ".join(saved["limitations"] + labels)
    else:
        assert "not independent of model selection" in saved["limitations"][0]


@pytest.mark.parametrize("workflow", [2023], indirect=True)
def test_rejects_mixed_year_protocol_before_reporting(workflow):
    path = workflow / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["initialization_times"][0] = "2022-01-15T00:00:00Z"
    body = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="All initialization times must match evaluation_year 2023"):
        report(workflow, make_plots=False)
    assert not (workflow / "report/summary.json").exists()


def test_failed_pairing_never_writes_complete_report(workflow):
    (workflow / "models/M3/start0.npz").unlink()
    with pytest.raises(ValueError):
        report(workflow, make_plots=False)
    assert not (workflow / "report/summary.json").exists()


def test_stale_exact_loss_validation_cannot_be_combined_with_spectral_results(workflow):
    (workflow / "models/M2/evaluation.json").write_text('{"evaluation_status":"running"}')
    with pytest.raises(ValueError, match="validation is stale"):
        report(workflow, make_plots=False)
    assert not (workflow / "report/summary.json").exists()


def test_protocol_changes_after_evaluation_are_detected(workflow):
    path = workflow / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["initialization_times"][0] = "2022-02-15T00:00:00Z"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol_hash does not match"):
        load_spectra(workflow)
