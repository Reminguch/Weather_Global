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
