"""Guard the explicit memory policy and bounded, resumable submission."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_v24_matched_reset_extremes as driver


def test_reset_configuration_uses_zero_memory_at_every_forecast_step(tmp_path):
    manifest = dict(shards=[dict(output_dir=str(tmp_path), count=16,
                    initializations=dict(path=str(tmp_path / "dates.json")))],
                    checkpoint=dict(path="reset.pkl"), baseline=dict(path="gc.npz"),
                    stats_dir="stats", data_path="test.zarr", architecture={}, reset_state_every_step=True)
    config = driver.make_eval_config(manifest, 0)
    assert config.reset_state_every_step is True
    assert config.residual_state_init == "zero"
    assert config.eval_mode == "cold_full"
    assert (config.target_steps, config.warmup_steps, config.n_samples) == (40, 0, 16)
    assert config.ckpt == Path("reset.pkl")
    manifest["reset_state_every_step"] = False
    with pytest.raises(ValueError, match="state policy"):
        driver.make_eval_config(manifest, 0)


def test_date_ids_use_full_utc_timestamps():
    assert driver.initialization_ids(dict(initializations=[
        dict(initialization_time="2023-01-01T00:00:00Z"),
        dict(initialization_time="2023-12-30T00:00:00Z")])) == ["20230101T000000Z", "20231230T000000Z"]


def test_submission_is_unthrottled_bounded_and_idempotent(tmp_path, monkeypatch):
    (tmp_path / "submissions").mkdir()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=str(100 + len(calls)), stderr="")
    monkeypatch.setattr(driver.subprocess, "run", run)
    manifest = dict(workspace=str(tmp_path), source_root=str(tmp_path / "source"), manifest_hash="frozen")
    assert driver.submit(tmp_path, manifest) == dict(worker="101", report="102")
    assert "--array=0-7" in calls[0]
    assert "--time=61" in calls[0]
    assert "--qos=gpu-short" in calls[0]
    assert not any(arg.startswith("--partition=") for arg in calls[0])
    assert "--dependency=afterok:101" in calls[1]
    assert driver.submit(tmp_path, manifest) == dict(worker="101", report="102")
    assert len(calls) == 2
    receipt = tmp_path / "submissions/worker.json"
    receipt.write_text(json.dumps(dict(status="submitting")))
    with pytest.raises(RuntimeError, match="inspect Slurm"):
        driver.submit(tmp_path, manifest)
