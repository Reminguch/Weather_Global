"""Memory policy provenance and archived-model fixed-date orchestration."""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from scripts.analyze_models import eval_v22_2023_manifest as legacy
from scripts.analyze_models.merge_v24_2023_eval import merge_shards
from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
from tests.v24_Ilya.test_2023_eval_merge import protocol, _shards
from tests.v24_Ilya.test_initialization_manifest import _eval_data, _manifest


def test_merge_requires_control_policy_and_every_control_date(tmp_path, protocol):
    protocol["models"].append({"model_id": "M1_reset", "checkpoint_sha256": "hash-M1",
                               "reset_state_every_step": True})
    paths = _shards(tmp_path, protocol)
    payload = json.loads(paths[-1].read_text())
    payload["rs_reset_every_step"] = False
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="memory reset policy"):
        merge_shards(protocol, paths)
    payload["rs_reset_every_step"] = True
    paths[-1].write_text(json.dumps(payload))
    baseline, full, _ = merge_shards(protocol, paths)
    assert baseline.shape == full.shape == (4, 365, 40)


def test_legacy_manifest_preserves_reset_and_sample_losses_across_shards(tmp_path, monkeypatch):
    starts = ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"]
    manifest = _manifest(tmp_path, starts)
    monkeypatch.setattr(legacy, "open_eval_dataset", lambda *args: _eval_data())
    monkeypatch.setattr(legacy, "load_graphcast_checkpoint", lambda *args: SimpleNamespace(
        task_config=None, model_config=None, params={}))
    monkeypatch.setattr(legacy, "load_stats", lambda *args: {
        "diffs_stddev_by_level": xr.Dataset({"2m_temperature": xr.DataArray(2.)})})
    predictor = SimpleNamespace(init=lambda *args: ({}, {}), apply=lambda *args: None)
    monkeypatch.setattr(legacy, "build_predictors", lambda *args: SimpleNamespace(baseline=predictor, residual=predictor))
    monkeypatch.setattr(legacy, "overlay_matching_params", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(legacy, "load_v22_final_checkpoint", lambda *args: SimpleNamespace(
        residual_params={}, residual_state={}, checkpoint_format="legacy_test"))
    monkeypatch.setattr(legacy, "_release_host_memory", lambda: None)

    def fields(value, steps):
        return xr.Dataset({"2m_temperature": (("batch", "time", "lat", "lon"),
            np.full((1, steps, 2, 1), value, np.float32))},
            coords={"batch": [0], "time": np.arange(1, steps+1)*np.timedelta64(6, "h"),
                    "lat": [-45., 45.], "lon": [0.]})

    def build(*args, indices, target_steps, **kwargs):
        inputs = fields(indices[0], 1)
        return inputs, fields(0, target_steps), fields(0, target_steps)

    def rollout(*, inputs, reset_state_every_step, target_steps, **kwargs):
        error = float(inputs["2m_temperature"].values.flat[0])+1
        return SimpleNamespace(baseline_prediction=fields(error, target_steps),
                               full_prediction=fields(error*(.5 if reset_state_every_step else .25), target_steps))

    monkeypatch.setattr(legacy, "build_eval_batch", build)
    monkeypatch.setattr(legacy, "run_v22_final_rollout", rollout)
    config = V24IlyaEvalConfig(ckpt=tmp_path/"model.pkl", out_json=tmp_path/"all.json",
        initialization_manifest=manifest, model_id="C_reset", input_duration=None,
        target_steps=2, eval_mode="cold_full", warmup_steps=0, reset_state_every_step=True)
    result = legacy.evaluate_legacy_manifest(config, {})
    assert result["architecture_id"] == "v22_final"
    assert result["rs_reset_every_step"] is True
    assert result["original_graphcast_loss"]["improvement_pct_rollout"] == pytest.approx(75)
    records = result["original_graphcast_loss_per_initialization"]
    sharded = []
    for i in range(2):
        shard = legacy.evaluate_legacy_manifest(dataclasses.replace(config, anchor_shard_count=2,
            anchor_shard_index=i, out_json=tmp_path/f"shard{i}.json"), {})
        sharded.extend(shard["original_graphcast_loss_per_initialization"])
    assert sharded == records
    for branch in ("baseline", "full"):
        np.testing.assert_allclose(np.mean([r[f"{branch}_per_step"] for r in records], axis=0),
                                   result["original_graphcast_loss"][f"{branch}_per_step"])
