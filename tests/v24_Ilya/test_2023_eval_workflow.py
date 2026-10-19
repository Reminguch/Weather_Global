"""Scientific and resource gates for the fixed 2023 evaluation workflow."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import run_v24_2023_eval as workflow


def test_dates_and_shards_cover_calendar_exactly():
    dates = workflow.make_initializations(2023)
    assert len(dates) == 365
    assert dates[0] == "2023-01-01T00:00:00Z"
    assert dates[-1] == "2023-12-31T00:00:00Z"
    shards = [dates[i::workflow.SHARDS] for i in range(workflow.SHARDS)]
    assert set(map(len, shards)) == {15, 16}
    assert sorted(value for shard in shards for value in shard) == dates


def test_three_day_calendar_keeps_all_prespecified_starts_and_qualifies_actual_shards():
    dates = workflow.make_initializations(2023, 3)
    assert len(dates) == 122
    assert dates[0] == "2023-01-01T00:00:00Z"
    assert dates[-1] == "2023-12-30T00:00:00Z"
    assert workflow.shard_count(dates) == 8
    shards = [dates[i::8] for i in range(8)]
    assert set(map(len, shards)) == {15, 16}
    assert sorted(value for shard in shards for value in shard) == dates
    assert [date for date in dates if date in workflow.PRIOR_EXPOSURE_TIMES] == [
        "2023-02-15T00:00:00Z", "2023-07-15T00:00:00Z", "2023-11-15T00:00:00Z"]


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_invalid_initialization_stride_is_rejected(stride):
    with pytest.raises(ValueError, match="positive integer"):
        workflow.make_initializations(2023, stride)


def test_pilot_repetitions_are_independent_tasks_not_anchor_shards():
    protocol = {"schema_version": 2, "models": [{}] * 7, "anchor_shard_count": 8}
    assert [workflow.task_assignment(protocol, i, pilot=True) for i in range(14)] == [
        (model, repeat) for model in range(7) for repeat in range(2)]
    assert workflow.task_assignment(protocol, 55) == (6, 7)
    for task, pilot in ((-1, True), (14, True), (56, False)):
        with pytest.raises(ValueError, match="Invalid task"):
            workflow.task_assignment(protocol, task, pilot=pilot)
    protocol["schema_version"] = 1
    assert workflow.task_assignment(protocol, 6, pilot=True) == (6, 0)


def test_protocol_hash_detects_configuration_changes():
    protocol = {"models": [{"checkpoint_sha256": "abc"}], "bootstrap": {"block_days": 14}}
    protocol["protocol_hash"] = workflow.protocol_digest(protocol)
    assert workflow.protocol_digest(protocol) == protocol["protocol_hash"]
    protocol["bootstrap"]["block_days"] = 10
    assert workflow.protocol_digest(protocol) != protocol["protocol_hash"]


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    times = [f"2022-{month:02d}-15T00:00:00Z" for month in (1, 4, 7, 10)]
    protocol = {"protocol_hash": "frozen", "models": [{"model_id": f"M{i}", "checkpoint_sha256": f"checkpoint{i}"} for i in range(1, 4)],
                "pilot": {"initialization_times": times, "manifest": {"sha256": "dates"}},
                "baseline": {"checkpoint_sha256": "baseline", "repeatability_rtol": 1e-3, "repeatability_atol": 1e-6},
                "resources": {"gpu": {"walltime_seconds": 3720}}}
    monkeypatch.setattr(workflow, "verify_protocol", lambda path: protocol)
    base = np.arange(1., 161.).reshape(4, 40)
    payload = {"protocol_hash": "frozen", "evaluation_status": "complete", "peak_rss_gib": 10,
               "elapsed_seconds": 900., "sample_elapsed_seconds": [523., 104., 104., 104.],
               "original_graphcast_loss": {"baseline_per_step": base.mean(0).tolist(), "full_per_step": (base*.8).mean(0).tolist()},
               "original_graphcast_loss_per_initialization": [
                   {"initialization_time": stamp, "baseline_per_step": row.tolist(), "full_per_step": (row*.8).tolist()}
                   for stamp, row in zip(times, base)]}
    for model in protocol["models"]:
        model_payload = dict(payload, model_id=model["model_id"], checkpoint_sha256=model["checkpoint_sha256"],
                             baseline_checkpoint_sha256="baseline", initialization_manifest_sha256="dates")
        workflow.write_json(tmp_path / "pilot" / f"{model['model_id']}_00.json", model_payload)
        if model["model_id"] == "M2":
            saved = model_payload
    return tmp_path, saved


def test_pilot_requires_complete_paired_losses_and_measured_resources(pilot):
    directory, _ = pilot
    workflow.validate_pilot(directory)
    assert json.loads((directory / "pilot_validation.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("defect", ["baseline", "dates", "memory", "runtime", "partial", "aggregate", "identity"])
def test_failed_pilot_cannot_release_production(pilot, defect):
    directory, saved = pilot
    payload = copy.deepcopy(saved)
    if defect == "baseline":
        payload["original_graphcast_loss_per_initialization"][0]["baseline_per_step"][0] *= 2
    elif defect == "dates":
        payload["original_graphcast_loss_per_initialization"].pop()
    elif defect == "memory":
        payload["peak_rss_gib"] = 15
    elif defect == "runtime":
        payload["sample_elapsed_seconds"] = [600, 300, 300, 300]
    elif defect == "partial":
        payload["evaluation_status"] = "partial"
    elif defect == "identity":
        payload["model_id"] = "M3"
    else:
        payload["original_graphcast_loss"]["full_per_step"][0] *= 2
    workflow.write_json(directory / "pilot" / "M2_00.json", payload)
    with pytest.raises((ValueError, AssertionError)):
        workflow.validate_pilot(directory)
    assert not (directory / "pilot_validation.json").exists()


@pytest.fixture
def repeated_pilot(pilot, monkeypatch):
    directory, _ = pilot
    protocol = workflow.verify_protocol(directory)
    protocol.update(schema_version=2, initialization_stride_days=3,
                    initialization_times=workflow.make_initializations(2023, 3), anchor_shard_count=8)
    protocol["resources"]["gpu"]["mem_gib"] = 16
    qualification = directory / "numerical_qualification.json"
    workflow.write_json(qualification, {"status": "passed", "protocol_hash": "frozen",
                                       "runtime": {"gpu_kind": "NVIDIA A100 80GB PCIe", "driver_version": "610.57.04"}})
    protocol["numerical_qualification"] = {"path": str(qualification)}
    monkeypatch.setattr(workflow, "require_numerical_qualification", lambda *args: workflow.frozen_file(qualification))
    for model in protocol["models"]:
        original = workflow.read_json(directory / "pilot" / f"{model['model_id']}_00.json")
        observation = {
            "setup": {key: {"sha256": key} for key in ("baseline_params", "baseline_state", "statistics",
                                                       "baseline_model_config", "task_config", "residual_params", "residual_state")},
            "initializations": [{"initialization_time": stamp, "inputs": {"sha256": stamp}, "rng": {"sha256": "rng"}}
                                for stamp in protocol["pilot"]["initialization_times"]],
            "predictions": [{"initialization_time": stamp, "step_index": lead, "truth": f"{stamp}_{lead}",
                             "baseline": "baseline", "full": "full"}
                            for stamp in protocol["pilot"]["initialization_times"] for lead in range(40)],
        }
        for repeat in range(2):
            payload = dict(original, pilot_repeat_index=repeat,
                           devices=[{"platform": "gpu", "kind": "NVIDIA A100 80GB PCIe"}],
                           gpu_kind="NVIDIA A100 80GB PCIe", driver_version="610.57.04", gpu_uuid="GPU-fixture",
                           numerical_observation=observation, metrics_complete=True,
                           evaluated_samples=len(protocol["pilot"]["initialization_times"]))
            workflow.write_json(directory / "pilot" / f"{model['model_id']}_{repeat:02d}.json", payload)
    return directory, protocol


def test_two_repeats_qualify_baseline_and_full_and_record_immutable_evidence(repeated_pilot):
    directory, protocol = repeated_pilot
    workflow.validate_pilot(directory)
    gate = workflow.require_production_gate(directory, protocol)
    assert len(gate["models"]) == 6
    assert gate["gpu_kind"] == "NVIDIA A100 80GB PCIe"
    assert gate["driver_version"] == "610.57.04"
    assert gate["numerical_qualification"] == workflow.frozen_file(directory / "numerical_qualification.json")
    assert all(row["projected_shard_starts"] == 16 for row in gate["models"])
    assert all(row["projected_shard_seconds"] == 2148 for row in gate["models"])
    with pytest.raises(FileExistsError, match="overwrite"):
        workflow.validate_pilot(directory)


def test_advisory_release_survives_drift_and_absent_or_changed_diagnostics(repeated_pilot, monkeypatch):
    directory, protocol = repeated_pilot
    protocol["numerical_policy"] = "advisory"
    # Restore the real function overridden by the legacy strict fixture.
    monkeypatch.setattr(workflow, "require_numerical_qualification", lambda *_: None)
    path = directory / "pilot/M2_01.json"
    payload = workflow.read_json(path)
    for branch in ("baseline", "full"):
        key = f"{branch}_per_step"
        for row in payload["original_graphcast_loss_per_initialization"]:
            row[key] = (np.asarray(row[key]) * 1.03).tolist()
        payload["original_graphcast_loss"][key] = (np.asarray(payload["original_graphcast_loss"][key]) * 1.03).tolist()
    workflow.write_json(path, payload)
    (directory / "numerical_qualification.json").unlink()
    workflow.validate_pilot(directory)
    gate = workflow.require_production_gate(directory, protocol)
    assert gate["status"] == "complete_with_warnings"
    assert gate["integrity_passed"] and gate["resources_passed"]
    assert "numerical_qualification" not in gate
    workflow.write_json(directory / "numerical_qualification.json", {"status": "failed"})
    assert workflow.require_production_gate(directory, protocol) == gate
    # A changed artifact is still an integrity failure after release.
    payload["numerical_observation"]["setup"]["baseline_state"] = {"sha256": "changed"}
    workflow.write_json(path, payload)
    with pytest.raises(ValueError, match="Frozen input changed"):
        workflow.require_production_gate(directory, protocol)


def test_advisory_qualification_does_not_read_diagnostic_report(tmp_path):
    protocol = {"schema_version": 2, "numerical_policy": "advisory",
                "numerical_qualification": {"path": str(tmp_path / "missing.json")}}
    assert workflow.require_numerical_qualification(tmp_path, protocol) is None


@pytest.mark.parametrize("stored_input_changed", [False, True])
def test_generated_input_gate_requires_exact_stored_inputs(repeated_pilot, monkeypatch, stored_input_changed):
    directory, protocol = repeated_pilot
    protocol.update(numerical_policy="advisory", generated_input_identity="separate_v1")
    monkeypatch.setattr(workflow, "require_numerical_qualification", lambda *_: None)
    monkeypatch.setattr(workflow, "generated_input_variables", lambda *_: ["toa_incident_solar_radiation"])
    for path in sorted((directory / "pilot").glob("*.json")):
        payload = workflow.read_json(path)
        different = path.name == "M2_01.json"
        for row in payload["numerical_observation"]["initializations"]:
            def fingerprint(digest):
                return {"sha256": digest * 64, "leaves": [{"name": "field", "shape": [1], "dtype": "float32"}]}
            row.update(derived_input_variables=["toa_incident_solar_radiation"], derived_inputs_finite=True,
                       source_inputs=fingerprint("b" if different and stored_input_changed else "a"),
                       derived_inputs=fingerprint("b" if different else "a"))
        workflow.write_json(path, payload)
    if stored_input_changed:
        with pytest.raises(ValueError, match="Effective pilot baseline"):
            workflow.validate_pilot(directory)
    else:
        workflow.validate_pilot(directory)
        gate = workflow.require_production_gate(directory, protocol)
        assert gate["status"] == "complete_with_warnings"
        assert gate["integrity_passed"]
        assert sum(not row["passed"] for row in gate["derived_input_diagnostics"]) == 1


@pytest.mark.parametrize("defect", ["input", "truth", "nonfinite", "aggregate"])
def test_advisory_pilot_keeps_integrity_blocking(repeated_pilot, monkeypatch, defect):
    directory, protocol = repeated_pilot
    protocol["numerical_policy"] = "advisory"
    monkeypatch.setattr(workflow, "require_numerical_qualification", lambda *_: None)
    path = directory / "pilot/M2_01.json"
    payload = workflow.read_json(path)
    if defect == "input":
        payload["numerical_observation"]["initializations"][0]["inputs"] = {"sha256": "wrong"}
    elif defect == "truth":
        payload["numerical_observation"]["predictions"][0]["truth"] = "wrong"
    elif defect == "nonfinite":
        payload["original_graphcast_loss_per_initialization"][0]["full_per_step"][0] = float("nan")
    else:
        payload["original_graphcast_loss"]["full_per_step"][0] *= 2
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, AssertionError)):
        workflow.validate_pilot(directory)
    assert not (directory / "pilot_validation.json").exists()


@pytest.mark.parametrize("defect", ["full", "baseline", "missing", "gpu_kind", "driver", "uuid", "cpu", "repeat", "memory", "nan_memory", "elapsed", "metrics", "samples"])
def test_repeat_gate_rejects_numerical_identity_and_resource_failures(repeated_pilot, defect):
    directory, _ = repeated_pilot
    path = directory / "pilot/M2_01.json"
    payload = workflow.read_json(path)
    if defect in ("full", "baseline"):
        field = f"{defect}_per_step"
        for record in payload["original_graphcast_loss_per_initialization"]:
            record[field] = (np.asarray(record[field]) * 1.01).tolist()
        payload["original_graphcast_loss"][field] = (
            np.asarray(payload["original_graphcast_loss"][field]) * 1.01).tolist()
    elif defect == "missing":
        path.unlink()
    elif defect == "gpu_kind":
        payload["devices"][0]["kind"] = "different GPU"
    elif defect == "driver":
        payload["driver_version"] = "different driver"
    elif defect == "uuid":
        del payload["gpu_uuid"]
    elif defect == "cpu":
        payload["devices"][0]["platform"] = "cpu"
    elif defect == "repeat":
        payload["pilot_repeat_index"] = 0
    elif defect == "memory":
        payload["peak_rss_gib"] = 12.81
    elif defect == "nan_memory":
        payload["peak_rss_gib"] = float("nan")
    elif defect == "metrics":
        payload["metrics_complete"] = False
    elif defect == "samples":
        payload["evaluated_samples"] = 3
    else:
        payload["elapsed_seconds"] = 0
    if defect != "missing":
        # Also verify invalid non-finite reports are rejected if supplied externally.
        path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        workflow.validate_pilot(directory)
    assert not (directory / "pilot_validation.json").exists()


def test_pending_numerical_qualification_blocks_even_consistent_pilots(repeated_pilot, monkeypatch):
    directory, _ = repeated_pilot
    def pending(*args):
        raise ValueError("Numerical qualification is pending")
    monkeypatch.setattr(workflow, "require_numerical_qualification", pending)
    with pytest.raises(ValueError, match="pending"):
        workflow.validate_pilot(directory)
    assert not (directory / "pilot_validation.json").exists()


@pytest.mark.parametrize("defect", ["baseline_params", "baseline_state", "statistics", "baseline_model_config", "task_config",
                                   "residual_params", "residual_state", "inputs", "rng", "missing"])
def test_pilot_gate_checks_effective_inputs_and_parameters(repeated_pilot, defect):
    directory, _ = repeated_pilot
    path = directory / "pilot/M2_01.json"
    result = workflow.read_json(path)
    if defect == "missing":
        del result["numerical_observation"]
    elif defect in ("inputs", "rng"):
        result["numerical_observation"]["initializations"][0][defect]["sha256"] = "changed"
    else:
        result["numerical_observation"]["setup"][defect]["sha256"] = "changed"
    workflow.write_json(path, result)
    with pytest.raises(ValueError, match="fingerprints|Effective"):
        workflow.validate_pilot(directory)
    assert not (directory / "pilot_validation.json").exists()


@pytest.mark.parametrize("defect", ["missing", "empty", "missing_lead", "truth", "baseline", "full"])
def test_pilot_gate_requires_complete_truth_and_prediction_fingerprints(repeated_pilot, defect):
    directory, _ = repeated_pilot
    path = directory / "pilot/M2_01.json"
    result = workflow.read_json(path)
    observation = result["numerical_observation"]
    if defect == "missing":
        del observation["predictions"]
    elif defect == "empty":
        observation["predictions"] = []
    elif defect == "missing_lead":
        observation["predictions"].pop()
    else:
        del observation["predictions"][0][defect]
    workflow.write_json(path, result)
    with pytest.raises(ValueError, match="coverage|fingerprints"):
        workflow.validate_pilot(directory)
    assert not (directory / "pilot_validation.json").exists()


@pytest.mark.parametrize("artifact", ["qualification", "pilot"])
def test_production_rejects_changed_qualification_or_pilot_after_gate(repeated_pilot, artifact):
    directory, protocol = repeated_pilot
    workflow.validate_pilot(directory)
    path = directory / ("numerical_qualification.json" if artifact == "qualification" else "pilot/M1_00.json")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        workflow.require_production_gate(directory, protocol)


def test_runtime_projection_uses_actual_largest_shard(repeated_pilot):
    directory, protocol = repeated_pilot
    protocol["anchor_shard_count"] = 10  # Thirteen starts in the largest diagnostic shard.
    workflow.validate_pilot(directory)
    gate = workflow.read_json(directory / "pilot_validation.json")
    assert all(row["projected_shard_starts"] == 13 for row in gate["models"])
    assert all(row["projected_shard_seconds"] == 1836 for row in gate["models"])


@pytest.mark.parametrize("field", ["gpu_kind", "driver_version"])
def test_all_pilots_must_match_qualified_hardware_even_if_mutually_consistent(repeated_pilot, field):
    directory, protocol = repeated_pilot
    path = Path(protocol["numerical_qualification"]["path"])
    report = workflow.read_json(path)
    report["runtime"][field] = "different qualified hardware"
    workflow.write_json(path, report)
    with pytest.raises(ValueError, match="differs from numerical qualification"):
        workflow.validate_pilot(directory)
    assert not (directory / "pilot_validation.json").exists()


@pytest.mark.parametrize("field", ["gpu_kind", "driver_version"])
def test_production_rejects_hardware_drift_before_forecasting(repeated_pilot, monkeypatch, field):
    from src.models.mamba.v24_Ilya import evaluation
    directory, protocol = repeated_pilot
    workflow.validate_pilot(directory)
    for model in protocol["models"]:
        model.update(checkpoint="unused.pkl", run_config={})
    protocol["baseline"]["path"] = "unused_baseline.npz"
    monkeypatch.setattr(workflow, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(workflow, "check_file", lambda *args: None)
    hardware = {"gpu_kind": "NVIDIA A100 80GB PCIe", "driver_version": "610.57.04"}
    hardware[field] = "changed runtime"
    monkeypatch.setattr(workflow, "capture_runtime_hardware", lambda *args: hardware)
    monkeypatch.setattr(evaluation, "evaluate_v24_Ilya", lambda *args, **kwargs: pytest.fail("Forecast ran before hardware qualification"))
    with pytest.raises(ValueError, match="Production GPU kind or driver differs"):
        workflow.evaluate(directory, 0)
    assert not (directory / "shards").exists()


@pytest.mark.parametrize("changed", ["checkpoint_sha256", "architecture", "evaluator_family", "reset_state_every_step", "run_config", "baseline", "statistics"])
def test_three_day_protocol_cannot_change_prespecified_models_or_normalization(changed):
    models = [{"model_id": model_id, "checkpoint_sha256": "checkpoint", "architecture": {"width": 512},
               "evaluator_family": "v24_Ilya", "reset_state_every_step": False, "run_config": {"sha256": "config"}}
              for model_id in ("M1", "M2", "M3", "M1_reset", "C_carry", "C_reset", "R_reset")]
    reference = {"models": models, "baseline": {"checkpoint_sha256": "baseline"},
                 "statistics": [{"path": "/stats/mean_by_level.nc", "sha256": "stats"}]}
    reference["protocol_hash"] = workflow.protocol_digest(reference)
    protocol = copy.deepcopy(reference)
    workflow.validate_reference_identities(protocol, reference)
    if changed == "baseline":
        protocol["baseline"]["checkpoint_sha256"] = "changed"
    elif changed == "statistics":
        protocol["statistics"][0]["sha256"] = "changed"
    else:
        protocol["models"][0][changed] = "changed"
    with pytest.raises(ValueError, match="changed"):
        workflow.validate_reference_identities(protocol, reference)


def test_prepare_three_day_campaign_freezes_protocol_without_claiming_qualification(tmp_path, monkeypatch):
    from scripts.experiments import v24_2023_qualification as qualification
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    dataset = tmp_path / "dataset.zarr"
    pilot_data = tmp_path / "data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr"
    for path in (dataset, pilot_data):
        workflow.write_json(path / ".zmetadata", {"zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}})
    workflow.write_json(dataset / "stage_report.json", {"status": "complete"})
    workflow.write_json(dataset / "source_metadata.json", {"source": "fixture"})
    directory = tmp_path / "campaign"
    workflow.write_json(directory / "data_validation.json", {
        "status": "passed", "dataset": str(dataset),
        "staging_report_sha256": workflow.sha256(dataset / "stage_report.json"),
        "dataset_metadata_sha256": workflow.sha256(dataset / ".zmetadata"),
    })
    workflow.write_json(tmp_path / "data/graphcast/graphcast/dataset/anchor_manifests/v24_Ilya_res1/metadata.json", {})
    workflow.write_json(tmp_path / "baseline.pkl", {"baseline": "fixture"})
    for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"):
        workflow.write_json(tmp_path / "stats" / name, {"statistics": "fixture"})
    runs = [(tmp_path / "artifacts/checkpoints/v24_Ilya" / relative, checkpoint, "carry")
            for _, _, relative, checkpoint in workflow.MODEL_PATHS]
    runs.extend((tmp_path / "artifacts/checkpoints/v22_final" / relative, "swa_step02000-08000.pkl", policy)
                for relative, policy in (
                    ("res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k", "carry"),
                    ("res1_dm_7yr_k20_di_bcg_reset_every_anchor_20k_20260815/di16_bcg1_closed_sg_reset_every_anchor_20k", "reset_every_anchor")))
    for run, checkpoint, policy in runs:
        workflow.write_json(run / "run_config.json", {
            "data": {"baseline_checkpoint": "baseline.pkl", "stats_dir": "stats"},
            "derived": {"prepared_store_selection": {"time_end": "2022-12-31T18:00:00Z"}},
            "architecture": {}, "sequence": {"temporal_state_policy": policy},
        })
        workflow.write_json(run / "swa" / checkpoint, {"checkpoint": str(run)})
    reference = tmp_path / "reference.json"
    workflow.write_json(reference, {"fixture": "reference"})
    reference_checks = []
    monkeypatch.setattr(workflow, "validate_reference_identities", lambda protocol, old: reference_checks.append(old))
    monkeypatch.setattr(qualification, "audit_training_provenance", lambda protocol: {"status": "audited"})
    monkeypatch.setattr(workflow.importlib.metadata, "version", lambda *args: "fixture")
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *args, **kwargs: "revision")
    def snapshot(path):
        workflow.write_json(path / "source_hashes.json", {})
        return workflow.frozen_file(path / "source_hashes.json")
    monkeypatch.setattr(workflow, "snapshot_source", snapshot)
    workflow.prepare(directory, dataset, stride_days=3, reference_protocol=reference)
    protocol = workflow.read_json(directory / "protocol.json")
    assert reference_checks == [{"fixture": "reference"}]
    assert protocol["schema_version"] == 2
    assert protocol["initialization_stride_days"] == 3
    assert len(protocol["initialization_times"]) == 122
    assert protocol["anchor_shard_count"] == 8
    assert len(protocol["models"]) == 7
    assert protocol["pilot"]["repeats"] == 2
    assert protocol["resources"]["gpu"]["mem_gib"] == 16
    assert protocol["numerics"]["internal_predictors"] == "bfloat16"
    assert protocol["bootstrap"] == {"replicates": 10000, "seed": 20260916, "block_starts": 5,
                                     "sensitivity_block_starts": [3, 7], "segmentation": "annual", "quarter_sensitivity": True}
    assert protocol["analysis"]["primary_endpoint"] == "M1_vs_GC_rollout"
    assert len(protocol["prior_exposure"]["initialization_times"]) == 8
    assert len(protocol["prior_exposure"]["sampled_initialization_times"]) == 3
    assert protocol["numerical_qualification"]["required"] is True
    assert not Path(protocol["numerical_qualification"]["path"]).exists()
    assert "training_data_end" not in protocol["models"][0]
    assert protocol["models"][0]["prepared_store_end"].startswith("2022-")
    assert protocol["protocol_hash"] == workflow.protocol_digest(protocol)


@pytest.mark.parametrize("pilot, task, expected_filename, expected_samples, expected_shard", [
    (True, 3, "pilot/M2_01.json", 4, 0),
    (False, 11, "shards/M2_03.json", 122, 3),
])
def test_evaluate_uses_manifest_sizes_and_keeps_repeat_out_of_anchor_shard(
    tmp_path, monkeypatch, pilot, task, expected_filename, expected_samples, expected_shard,
):
    import jax
    from types import SimpleNamespace
    from src.models.mamba.v24_Ilya import evaluation
    protocol = {
        "schema_version": 2, "protocol_hash": "frozen", "target_steps": 40,
        "anchor_shard_count": 8, "initialization_times": workflow.make_initializations(2023, 3),
        "initialization_manifest": {"path": str(tmp_path / "dates.json")},
        "pilot": {"initialization_times": [f"2022-{m:02d}-15T00:00:00Z" for m in (1, 4, 7, 10)],
                  "manifest": {"path": str(tmp_path / "pilot_dates.json")}, "data_path": str(tmp_path / "pilot_data")},
        "models": [{"model_id": f"M{i}", "checkpoint": str(tmp_path / f"M{i}.pkl"),
                    "checkpoint_sha256": f"checkpoint{i}", "run_config": {}, "architecture": {}} for i in (1, 2, 3)],
        "baseline": {"path": str(tmp_path / "baseline.npz"), "checkpoint_sha256": "baseline"},
        "stats_dir": str(tmp_path / "stats"), "data_path": str(tmp_path / "data"),
        "forecast": {"eval_mode": "cold_full", "residual_state_init": "zero", "residual_alpha": 1.0,
                     "warmup_steps": 0, "input_duration": "12h", "seed": 0},
    }
    monkeypatch.setattr(workflow, "verify_protocol", lambda path: protocol)
    monkeypatch.setattr(workflow, "validate_runtime", lambda value: None)
    monkeypatch.setattr(workflow, "check_file", lambda value: None)
    monkeypatch.setattr(workflow, "require_production_gate", lambda *args: {"gpu_kind": "fixture GPU", "driver_version": "fixture driver"})
    monkeypatch.setattr(workflow, "capture_runtime_hardware", lambda *args: {
        "gpu_kind": "fixture GPU", "driver_version": "fixture driver", "gpu_uuid": "fixture UUID",
        "devices": [{"platform": "gpu", "kind": "fixture GPU"}],
    })
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu", device_kind="fixture GPU")])
    configs = []
    observers = []
    def evaluator(config, **kwargs):
        configs.append(config)
        observers.append(kwargs.get("numerical_observer"))
        config.out_json.write_text("{}")
        return {"evaluation_status": "complete"}
    monkeypatch.setattr(evaluation, "evaluate_v24_Ilya", evaluator)
    workflow.evaluate(tmp_path, task, pilot=pilot)
    assert configs[0].n_samples == expected_samples
    assert configs[0].anchor_shard_index == expected_shard
    assert configs[0].anchor_shard_count == (1 if pilot else 8)
    assert configs[0].val_year == (2022 if pilot else 2023)
    assert configs[0].model_id == "M2"
    result = workflow.read_json(tmp_path / expected_filename)
    assert result["driver_version"] == "fixture driver"
    assert result["gpu_uuid"] == "fixture UUID"
    if pilot:
        assert result["pilot_repeat_index"] == 1
        assert result["numerical_observation"] == observers[0].data
    else:
        assert observers == [None]
        assert "numerical_observation" not in result
    with pytest.raises(FileExistsError, match="overwrite"):
        workflow.evaluate(tmp_path, task, pilot=pilot)
