"""Metadata preflight must fail before source freezing or expensive data work."""
import hashlib
import json

import pytest

from scripts.experiments import run_v24_2023_eval as daily
from scripts.experiments import run_v24_spectral_pilot as spectral
from scripts.experiments import run_v24_amse_pair as pair
from scripts.experiments import run_v24_res1_batch4_lr as batch
from scripts.analyze_models import evaluate_v24_amse_pair as amse
from src.data_operations.loaders.zarr_metadata import frozen_zarr_metadata, verify_frozen_zarr_metadata


def metadata_store(path, version):
    path.mkdir(parents=True, exist_ok=True)
    metadata = ({"zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}}
                if version == 2 else {"zarr_format": 3, "consolidated_metadata": {
                    "kind": "inline", "metadata": {"temperature": {"node_type": "array"}}}})
    target = path / (".zmetadata" if version == 2 else "zarr.json")
    # Deliberately retain whitespace: old identities hash file bytes.
    target.write_text(json.dumps(metadata, indent=4) + "\n")
    return target


@pytest.mark.parametrize("version", [2, 3])
def test_identity_matches_existing_frozen_file_and_symlink_resolution(tmp_path, version):
    target = metadata_store(tmp_path / "store", version)
    link = tmp_path / "linked.zarr"
    link.symlink_to(target.parent, target_is_directory=True)
    record = frozen_zarr_metadata(link)
    assert record == spectral.frozen_file(target)
    assert record == {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


@pytest.mark.parametrize("version, metadata", [
    (2, {"zarr_consolidated_format": 2, "metadata": {"anything": {}}}),
    (2, {"zarr_consolidated_format": 1, "metadata": {}}),
    (2, {"zarr_consolidated_format": 1, "metadata": ["not a mapping"]}),
    (3, {"zarr_format": 2}),
    (3, {"zarr_format": 3}),
    (3, {"zarr_format": 3, "consolidated_metadata": []}),
    (3, {"zarr_format": 3, "consolidated_metadata": {"kind": "inline", "metadata": {}}}),
    (3, {"zarr_format": 3, "consolidated_metadata": {"kind": "external", "metadata": {"x": {}}}}),
    (3, ["not a mapping"]),
])
def test_invalid_or_unconsolidated_metadata_has_actionable_error(tmp_path, version, metadata):
    target = metadata_store(tmp_path, version)
    target.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"consolidated Zarr v{version} metadata"):
        frozen_zarr_metadata(tmp_path)


def test_invalid_v3_cannot_silently_fall_back_to_stale_v2(tmp_path):
    metadata_store(tmp_path, 2)
    (tmp_path / "zarr.json").write_text("{}")
    with pytest.raises(ValueError, match="consolidated Zarr v3"):
        frozen_zarr_metadata(tmp_path)


def test_missing_store_metadata_and_invalid_json_fail_distinctly(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset directory.*absent"):
        frozen_zarr_metadata(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="Missing consolidated Zarr metadata"):
        frozen_zarr_metadata(tmp_path)
    (tmp_path / ".zmetadata").write_text("{unfinished")
    with pytest.raises(ValueError, match="Invalid Zarr metadata JSON"):
        frozen_zarr_metadata(tmp_path)


def test_spectral_rejects_metadata_before_coordinate_reads_and_freezing(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.zarr"
    dataset.mkdir()
    workflow = tmp_path / "workflow"
    monkeypatch.setattr(spectral, "validate_data", lambda *args: pytest.fail("Opened dataset before metadata preflight"))
    monkeypatch.setattr(spectral, "snapshot_source", lambda *args: pytest.fail("Frozen invalid experiment"))
    with pytest.raises(FileNotFoundError, match="Missing consolidated Zarr metadata"):
        spectral.prepare(workflow, dataset)
    assert not workflow.exists()


@pytest.mark.parametrize("version", [2, 3])
def test_spectral_prepare_records_selected_metadata_format(tmp_path, monkeypatch, version):
    monkeypatch.setattr(spectral, "ROOT", tmp_path)
    monkeypatch.setattr(spectral, "MODEL_PATHS", (("M1", "fixture", "run", "swa.pkl"),))
    dataset = tmp_path / "dataset.zarr"
    metadata = metadata_store(dataset, version)
    run = tmp_path / "artifacts/checkpoints/v24_Ilya/run"
    spectral.write_json(run / "run_config.json", {
        "data": {"baseline_checkpoint": "baseline.pkl", "stats_dir": "stats"}, "architecture": {},
    })
    (run / "swa").mkdir()
    (run / "swa/swa.pkl").write_bytes(b"fixture")
    (tmp_path / "baseline.pkl").write_bytes(b"baseline")
    (tmp_path / "stats").mkdir()
    for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"):
        (tmp_path / "stats" / name).write_bytes(b"stats")
    monkeypatch.setattr(spectral, "validate_data", lambda *args: {"status": "passed"})
    monkeypatch.setattr(spectral.importlib.metadata, "version", lambda *args: "fixture")
    monkeypatch.setattr(spectral.subprocess, "check_output", lambda *args, **kwargs: "revision")
    monkeypatch.setattr(spectral, "snapshot_source", lambda *args: spectral.frozen_file(metadata))
    workflow = tmp_path / "workflow"
    spectral.prepare(workflow, dataset)
    protocol = spectral.read_json(workflow / "protocol.json")
    assert protocol["data_metadata"] == spectral.frozen_file(metadata)
    assert protocol["protocol_hash"] == spectral.protocol_digest(protocol)


def test_daily_validation_does_not_claim_v3_chunk_support(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.zarr"
    metadata_store(dataset, 3)
    monkeypatch.setattr(daily, "snapshot_source", lambda *args: pytest.fail("Frozen unvalidated experiment"))
    with pytest.raises(ValueError, match="requires Zarr v2 chunk validation"):
        daily.prepare(tmp_path / "workflow", dataset)


def test_daily_preflight_also_validates_pilot_metadata_before_freezing(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    monkeypatch.setattr(daily, "snapshot_source", lambda *args: pytest.fail("Frozen invalid pilot dataset"))
    dataset = tmp_path / "dataset.zarr"
    metadata = metadata_store(dataset, 2)
    daily.write_json(dataset / "stage_report.json", {"status": "complete"})
    workflow = tmp_path / "workflow"
    daily.write_json(workflow / "data_validation.json", {
        "status": "passed", "dataset": str(dataset),
        "staging_report_sha256": daily.sha256(dataset / "stage_report.json"),
        "dataset_metadata_sha256": daily.sha256(metadata),
    })
    pilot = tmp_path / "data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr"
    pilot.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Missing consolidated Zarr metadata.*wb2_res1"):
        daily.prepare(workflow, dataset)
    assert not (workflow / "source").exists()
    assert not (workflow / "protocol.json").exists()


def test_new_v3_metadata_cannot_supersede_unchanged_pinned_v2(tmp_path):
    metadata_store(tmp_path, 2)
    pinned = frozen_zarr_metadata(tmp_path)
    assert verify_frozen_zarr_metadata(tmp_path, pinned) == pinned
    metadata_store(tmp_path, 3)
    # The old check passes even though a reader now selects the v3 store.
    spectral.check_file(pinned)
    with pytest.raises(ValueError, match="dataset metadata changed"):
        verify_frozen_zarr_metadata(tmp_path, pinned)


@pytest.mark.parametrize("stage", ["exact", "spectral"])
@pytest.mark.parametrize("action", ["preflight", "specification", "verify"])
def test_amse_detects_format_switch_from_preparation_pins(tmp_path, monkeypatch, stage, action):
    for name in ("exact", "spectral"):
        metadata_store(tmp_path / name, 2)
    pins = {name: frozen_zarr_metadata(tmp_path / name) for name in ("exact", "spectral")}
    spectral.write_json(tmp_path / "dates.json", {"initializations": []})
    pilot = {"data_path": str(tmp_path / "spectral"), "data_metadata": pins["spectral"],
             "initialization_manifest": spectral.frozen_file(tmp_path / "dates.json")}
    pilot["protocol_hash"] = spectral.protocol_digest(pilot)
    spectral.write_json(tmp_path / "pilot.json", pilot)
    spectral.write_json(tmp_path / "source/snapshot.json", {})
    manifest = {
        "workspace": str(tmp_path), "entries": [{"name": name} for name in amse.ARMS],
        "exact_data_path": "exact", "spectral_protocol": "pilot.json",
        "evaluation_data_metadata": pins, "source_root": str(tmp_path / "source"),
        "source_digest": spectral.sha256(tmp_path / "source/snapshot.json"),
    }
    spectral.write_json(tmp_path / "manifest.json", manifest)
    assert amse.validate_evaluation_data(tmp_path) == pins
    metadata_store(tmp_path / stage, 3)
    spectral.check_file(pins[stage])
    monkeypatch.setattr(pair, "verify_source", lambda *args: None)
    with pytest.raises(ValueError, match="dataset metadata changed"):
        if action == "preflight":
            amse.validate_evaluation_data(tmp_path)
        elif action == "specification":
            amse.make_specification(tmp_path, 0, "step2000", stage)
        else:
            pair.verify(tmp_path)


def test_batch_runtime_detects_metadata_format_switch(tmp_path, monkeypatch):
    dataset = tmp_path / "data"
    metadata_store(dataset, 2)
    spectral.write_json(tmp_path / "source/snapshot.json", {})
    monkeypatch.setattr(batch, "ROOT", tmp_path / "source")
    monkeypatch.setattr(batch, "verify_source", lambda *args: None)
    manifest = {
        "source_root": str(tmp_path / "source"), "config_hashes": {},
        "source_manifest_sha256": spectral.sha256(tmp_path / "source/snapshot.json"),
        "eval_data": str(dataset), "eval_data_metadata": frozen_zarr_metadata(dataset),
    }
    batch.require_runtime(manifest)
    metadata_store(dataset, 3)
    with pytest.raises(ValueError, match="dataset metadata changed"):
        batch.require_runtime(manifest)


@pytest.mark.parametrize("workflow", [spectral, daily])
def test_saved_protocol_detects_metadata_format_switch(tmp_path, workflow):
    dataset = tmp_path / "data"
    metadata_store(dataset, 2)
    protocol = {"data_path": str(dataset), "data_metadata": frozen_zarr_metadata(dataset)}
    protocol["protocol_hash"] = workflow.protocol_digest(protocol)
    workflow.write_json(tmp_path / "protocol.json", protocol)
    metadata_store(dataset, 3)
    with pytest.raises(ValueError, match="dataset metadata changed"):
        workflow.verify_protocol(tmp_path)
