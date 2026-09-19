"""Frozen pilot provenance and paired exact-loss checks."""
import copy

import numpy as np
import pytest
import xarray as xr

from scripts.experiments import run_v24_spectral_pilot as workflow


def test_default_pilot_dates_and_data_store_are_2023():
    expected = [f"2023-{month:02d}-15T00:00:00Z" for month in (1, 2, 4, 5, 7, 8, 10, 11)]
    assert workflow.pilot_initializations() == expected
    assert workflow.default_data_path(2023).name == "arco_res1_levels13_2023_eval.zarr"
    assert workflow.default_data_path(2022).name == "wb2_res1_levels13_train.zarr"
    assert all(value.startswith("2022-") for value in workflow.pilot_initializations(2022))
    with pytest.raises(ValueError, match="Supported pilot years"):
        workflow.default_data_path(2024)


@pytest.fixture
def completed_pilot(tmp_path):
    times = workflow.pilot_initializations()
    protocol = {
        "protocol_hash": "frozen", "initialization_times": times, "target_steps": 40,
        "initialization_manifest_sha256": "dates",
        "models": [{"model_id": f"M{i}", "checkpoint_sha256": f"checkpoint{i}"} for i in range(1, 4)],
        "baseline": {"checkpoint_sha256": "baseline", "canonical_model_id": "M1",
                     "repeatability_rtol": 1e-3, "repeatability_atol": 1e-6},
    }
    base = np.arange(1., 321.).reshape(8, 40)
    for index, model in enumerate(protocol["models"]):
        # Accepted harmless baseline jitter must not change the comparison denominator.
        local_baseline = base * (1 + index * 1e-5)
        payload = {
            "protocol_hash": "frozen", "model_id": model["model_id"], "evaluation_status": "complete",
            "checkpoint_sha256": model["checkpoint_sha256"], "baseline_checkpoint_sha256": "baseline",
            "initialization_manifest_sha256": "dates", "elapsed_seconds": 1300, "peak_rss_gib": 10.,
            "original_graphcast_loss": {"baseline_per_step": local_baseline.mean(0).tolist(),
                                        "full_per_step": (base * .8).mean(0).tolist(),
                                        "baseline_rollout": float(local_baseline.mean()), "full_rollout": float(base.mean() * .8)},
            "original_graphcast_loss_per_initialization": [
                {"initialization_time": stamp, "baseline_per_step": row.tolist(), "full_per_step": (full * .8).tolist()}
                for stamp, row, full in zip(times, local_baseline, base, strict=True)
            ],
        }
        workflow.write_json(tmp_path / "models" / model["model_id"] / "evaluation.json", payload)
    return tmp_path, protocol


def test_pilot_uses_canonical_baseline_and_binds_source_hashes(completed_pilot):
    directory, protocol = completed_pilot
    validation = workflow.validate_exact_losses(directory, protocol)
    assert validation["status"] == "passed"
    assert [model["improvement_pct_rollout"] for model in validation["models"]] == pytest.approx([20., 20., 20.])
    for record in validation["evaluation_sources"]:
        workflow.check_file(record)


@pytest.mark.parametrize("defect", ["baseline", "dates", "lead", "partial", "aggregate", "rollout", "identity", "nan"])
def test_invalid_pilot_cannot_produce_validated_comparison(completed_pilot, defect):
    directory, protocol = completed_pilot
    path = directory / "models" / "M2" / "evaluation.json"
    payload = copy.deepcopy(workflow.read_json(path))
    if defect == "baseline":
        # Alter every baseline and the aggregates coherently: repeatability must catch it.
        for record in payload["original_graphcast_loss_per_initialization"]:
            record["baseline_per_step"] = [value * 1.2 for value in record["baseline_per_step"]]
        payload["original_graphcast_loss"]["baseline_per_step"] = [value * 1.2 for value in payload["original_graphcast_loss"]["baseline_per_step"]]
        payload["original_graphcast_loss"]["baseline_rollout"] *= 1.2
    elif defect == "dates":
        payload["original_graphcast_loss_per_initialization"].reverse()
    elif defect == "lead":
        for record in payload["original_graphcast_loss_per_initialization"]:
            record["full_per_step"].pop()
    elif defect == "partial":
        payload["evaluation_status"] = "partial"
    elif defect == "aggregate":
        payload["original_graphcast_loss"]["full_per_step"][0] *= 2
    elif defect == "rollout":
        payload["original_graphcast_loss"]["full_rollout"] *= 2
    elif defect == "identity":
        payload["checkpoint_sha256"] = "wrong checkpoint"
    else:
        payload["original_graphcast_loss_per_initialization"][0]["full_per_step"][0] = None
    workflow.write_json(path, payload)
    with pytest.raises((ValueError, AssertionError)):
        workflow.validate_exact_losses(directory, protocol)
    assert not (directory / "exact_loss_validation.json").exists()


def test_protocol_hash_binds_pilot_dates_and_field_selection():
    protocol = {"initialization_times": workflow.pilot_initializations(), "spectral_variables": ["T2m", "U10"]}
    protocol["protocol_hash"] = workflow.protocol_digest(protocol)
    assert workflow.protocol_digest(protocol) == protocol["protocol_hash"]
    protocol["spectral_variables"].append("V10")
    assert workflow.protocol_digest(protocol) != protocol["protocol_hash"]


def test_frozen_source_verification_rejects_post_freeze_edits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "evaluator.py"
    code.write_text("frozen evaluator\n")
    workflow.write_json(tmp_path / "source_hashes.json", {"evaluator.py": workflow.sha256(code)})
    metadata = tmp_path / "metadata.json"
    workflow.write_json(metadata, {"status": "fixture"})
    protocol = {
        "source_root": str(source), "source_hashes": workflow.frozen_file(tmp_path / "source_hashes.json"),
        "initialization_manifest": workflow.frozen_file(metadata), "data_metadata": workflow.frozen_file(metadata),
        "data_validation": workflow.frozen_file(metadata), "statistics": [],
    }
    protocol["protocol_hash"] = workflow.protocol_digest(protocol)
    workflow.write_json(tmp_path / "protocol.json", protocol)
    assert workflow.verify_protocol(tmp_path)["protocol_hash"] == protocol["protocol_hash"]
    code.write_text("changed evaluator\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        workflow.verify_protocol(tmp_path)


@pytest.mark.parametrize("year", [2022, 2023])
def test_coordinate_validation_detects_missing_input_without_reading_fields(tmp_path, monkeypatch, year):
    times = np.arange(np.datetime64(f"{year}-01-14T18:00"), np.datetime64(f"{year}-01-25T06:00"), np.timedelta64(6, "h"))
    names = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure",
             "total_precipitation_6hr", "temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind")
    dataset = xr.Dataset({name: ((), 0.) for name in names}, coords={
        "time": times, "lat": np.arange(-90, 91), "lon": np.arange(360), "level": [700, 850],
    })
    source = [dataset]
    monkeypatch.setattr(xr, "open_zarr", lambda *args, **kwargs: source[0])
    assert workflow.validate_data(tmp_path, [f"{year}-01-15T00:00:00Z"])["status"] == "passed"
    source[0] = dataset.isel(time=slice(1, None))
    with pytest.raises(ValueError, match="lacks inputs or targets"):
        workflow.validate_data(tmp_path, [f"{year}-01-15T00:00:00Z"])
