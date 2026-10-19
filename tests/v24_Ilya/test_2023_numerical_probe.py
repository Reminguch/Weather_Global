"""Numerical probes validate evidence identities without requiring a GPU."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.analyze_models import diagnose_v24_2023_numerics as probe
from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig


def _protocol(tmp_path):
    times = [f"2022-{m:02d}-15T00:00:00Z" for m in (1, 4, 7, 10)]
    manifest = tmp_path / "pilot.json"
    manifest.write_text(json.dumps({"initializations": [{"initialization_time": t} for t in times]}))
    return {"schema_version": 2, "target_steps": 40, "models": [{"model_id": "M1"}],
        "pilot": {"manifest": {"path": str(manifest)}, "initialization_times": times},
        "forecast": {"warmup_steps": 0, "eval_mode": "cold_full", "residual_state_init": "zero"},
        "resources": {"gpu": {}}}


def test_cli_defaults_and_unsafe_or_incompatible_arguments():
    args = probe.parse_args(["--workflow", "/tmp/campaign", "--label", "nodeA_process1"])
    assert (args.repeats, args.model_id, args.mode) == (2, "M1", "interleaved")
    for extra in (["--label", "../unsafe"], ["--repeats", "0"], ["--model-id", "M2", "--mode", "standalone"]):
        with pytest.raises(SystemExit):
            probe.parse_args(["--workflow", "/tmp/campaign", "--label", "probe", *extra])


def test_frozen_four_dates_and_zero_state_contract_are_mandatory(tmp_path):
    protocol = _protocol(tmp_path)
    assert probe.validate_probe_protocol(protocol, "M1")["model_id"] == "M1"
    with pytest.raises(ValueError, match="Unknown"):
        probe.validate_probe_protocol(protocol, "unregistered")
    protocol["pilot"]["initialization_times"][0] = "2023-01-15T00:00:00Z"
    with pytest.raises(ValueError, match="2022"):
        probe.validate_probe_protocol(protocol, "M1")


def test_atomic_evidence_publication_never_replaces_or_publishes_nan(tmp_path):
    path = tmp_path / "probe.json"
    probe.write_new_json(path, {"original": 1})
    with pytest.raises(FileExistsError):
        probe.write_new_json(path, {"overwrite": True})
    assert json.loads(path.read_text()) == {"original": 1}
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        probe.write_new_json(invalid, {"loss": float("nan")})
    assert not invalid.exists()
    assert not list(tmp_path.glob(".probe-*.tmp"))


def test_gpu_identity_requires_one_80gb_device_and_selects_actual_uuid(tmp_path, monkeypatch):
    protocol = _protocol(tmp_path)
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu", device_kind="NVIDIA A100-SXM4-80GB")])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **k:
        "0, GPU-other, NVIDIA A100-SXM4-80GB, 550.54, 81920\n1, GPU-selected, NVIDIA A100-SXM4-80GB, 550.54, 81920\n")
    identity = probe.gpu_identity(protocol)
    assert identity["gpu_uuid"] == "GPU-selected"
    assert identity["driver_version"] == "550.54"
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="cpu")])
    with pytest.raises(ValueError, match="one visible GPU"):
        probe.gpu_identity(protocol)


def test_bridge_compares_pooled_losses_and_effects_without_historical_tolerance():
    old = {"original_graphcast_loss": {"baseline_per_step": [2.]*40, "full_per_step": [1.]*40}}
    new = {"original_graphcast_loss": {"baseline_per_step": [4.]*40, "full_per_step": [3.]*40}}
    comparison = probe.compare_development(old, new)
    assert comparison["improvement_pct"] == {"archived": 50., "rerun": 25.}
    assert comparison["improvement_difference_pp"] == -25.
    assert comparison["branches"]["baseline"]["rollout_difference"] == 2.
    assert comparison["criterion"] == "descriptive_only_no_historical_repeatability_tolerance"


def test_runner_publishes_distinct_same_process_evidence_and_refuses_reuse(tmp_path, monkeypatch):
    protocol = _protocol(tmp_path)
    protocol.update(protocol_hash="frozen-hash", source_root=str(probe.ROOT), stats_dir=str(tmp_path),
                    baseline={"path": str(tmp_path / "baseline.npz"), "checkpoint_sha256": "baseline-hash"})
    protocol["pilot"]["data_path"] = str(tmp_path / "data.zarr")
    protocol["models"][0].update(checkpoint=str(tmp_path / "model.pkl"), checkpoint_sha256="model-hash",
                                 run_config={"path": str(tmp_path / "run.json"), "sha256": "run-hash"}, architecture={})
    monkeypatch.setattr(probe.campaign, "verify_protocol", lambda _: protocol)
    monkeypatch.setattr(probe.campaign, "validate_runtime", lambda _: None)
    monkeypatch.setattr(probe.campaign, "check_file", lambda _: None)
    monkeypatch.setattr(probe, "gpu_identity", lambda _: {"gpu_uuid": "GPU-test", "gpu_kind": "A100-80GB", "driver_version": "driver"})

    def fake_baseline(config, observer):
        assert config.val_year == 2022
        return {"evaluation_status": "complete", "metrics_complete": True, "evaluated_samples": 4,
            "original_graphcast_loss_per_initialization": [
                {"initialization_time": t, "baseline_per_step": [1.]*40, "full_per_step": [1.]*40}
                for t in protocol["pilot"]["initialization_times"]]}

    monkeypatch.setattr(probe, "standalone_baseline", fake_baseline)
    args = probe.parse_args(["--workflow", str(tmp_path), "--label", "nodeA_run1", "--mode", "standalone"])
    paths = probe.run(args)
    assert [p.name for p in paths] == [f"nodeA_run1_M1_standalone_{i:02d}.probe.json" for i in range(2)]
    first, second = [json.loads(path.read_text()) for path in paths]
    assert first["pid"] == second["pid"]
    assert first["process_run_id"] == second["process_run_id"]
    assert first["protocol_hash"] == "frozen-hash"
    assert first["diagnostic_mode"] == "standalone"
    assert first["baseline_checkpoint_sha256"] == "baseline-hash"
    assert second["within_process_comparison"]["passed"] is True
    with pytest.raises(FileExistsError):
        probe.run(args)


def test_bridge_maps_legacy_indices_in_manifest_free_2022_slice(tmp_path):
    checkpoint = tmp_path / "selected.pkl"
    baseline = tmp_path / "baseline.npz"
    checkpoint.write_bytes(b"selected")
    baseline.write_bytes(b"baseline")
    run = tmp_path / "run"
    source = run / "eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json"
    source.parent.mkdir(parents=True)
    indices = list(range(1, 33))
    source.write_text(json.dumps({"chosen_idx": indices, "global_chosen_idx": indices,
        "target_steps": 40, "evaluated_samples": 32, "evaluation_status": "complete", "eval_mode": "cold_full",
        "eval_feedback": "full", "warmup_steps": 0, "residual_state_init": "zero", "rs_reset_every_step": False,
        "seed": 0, "ckpt": str(checkpoint), "ckpt_in": str(baseline)}))
    model = {"run_config": {"path": str(run / "run_config.json")}, "checkpoint": str(checkpoint), "checkpoint_sha256": probe.campaign.sha256(checkpoint)}
    protocol = {"project_root": str(tmp_path), "baseline": {"path": str(baseline), "checkpoint_sha256": probe.campaign.sha256(baseline)},
                "pilot": {"data_path": str(tmp_path / "data.zarr")}}
    # Extra pre-2022 positions must be discarded before applying old indices.
    times = pd.date_range("2021-12-31", periods=104, freq="6h")
    store = tmp_path / "data.zarr"
    (store / "time").mkdir(parents=True)
    hours = ((times.values.astype("datetime64[h]") - np.datetime64("1959-01-01", "h"))
             / np.timedelta64(1, "h")).astype("<i8")
    (store / "time/0").write_bytes(hours.tobytes())
    (store / ".zmetadata").write_text(json.dumps({"metadata": {
        "time/.zarray": {"shape": [104], "chunks": [104], "dtype": "<i8", "order": "C", "compressor": None, "filters": None},
        "time/.zattrs": {"units": "hours since 1959-01-01 00:00:00", "_ARRAY_DIMENSIONS": ["time"]},
    }}))
    config = V24IlyaEvalConfig(ckpt=checkpoint, ckpt_in=baseline, out_json=tmp_path / "result.json", val_year=2022, eval_mode="cold_full")
    record, archived, starts = probe.bridge_manifest(protocol, model, config, tmp_path / "manifest.json")
    assert record == probe.file_record(source)
    assert starts[0] == "2022-01-01T06:00:00Z"
    assert len(starts) == len(archived["chosen_idx"]) == 32


def test_standalone_matches_rng_and_feedback_without_residual_forward_calls(tmp_path, monkeypatch):
    from src.models.graphcast.training.core import model as core
    from src.models.mamba.v24_Ilya import checkpoint, data, evaluation, model, rollout

    @dataclasses.dataclass
    class Task:
        input_duration: str = "12h"

    def fields(value, step=1):
        return xr.Dataset({"2m_temperature": (("batch", "time", "lat", "lon"), np.full((1, 1, 2, 1), value, np.float32))},
            coords={"batch": [0], "time": [np.timedelta64(6*step, "h")], "lat": [-45., 45.], "lon": [0.]})

    records = [{"initialization_time": "2022-01-15T00:00:00Z", "dataset_index": 1,
                "valid_times": [f"lead{i}" for i in range(40)]}]
    monkeypatch.setattr(core, "load_graphcast_checkpoint", lambda _: SimpleNamespace(task_config=Task(), model_config=Task(), params={}))
    monkeypatch.setattr(core, "load_stats", lambda _: {"diffs_stddev_by_level": xr.Dataset({"2m_temperature": xr.DataArray(1.)})})
    monkeypatch.setattr(data, "open_eval_dataset", lambda *a: SimpleNamespace(dataset=xr.Dataset(coords={"lat": [-45., 45.]}), input_steps=2, time_step=pd.Timedelta("6h")))
    monkeypatch.setattr(data, "resolve_initialization_manifest", lambda *a, **k: records)
    monkeypatch.setattr(data, "build_eval_batch", lambda *a, **k: (fields(0), fields(0), fields(0)))
    monkeypatch.setattr(data, "iter_eval_steps", lambda *a, **k: ((fields(0, i+1), fields(0, i+1)) for i in range(40)))
    monkeypatch.setattr(checkpoint, "overlay_frozen_baseline_params", lambda *a: ({}, {}))
    monkeypatch.setattr(evaluation, "_release_host_memory", lambda: None)
    monkeypatch.setattr(jax, "jit", lambda f: f)
    seen = []

    def apply(params, state, key, inputs, targets, forcings):
        seen.append(np.asarray(key))
        value = float(inputs["2m_temperature"].values.flat[0])
        return fields(value + 1, len(seen)), state

    def forbidden(*a, **k):
        raise AssertionError("Residual forwards are forbidden in standalone mode")

    monkeypatch.setattr(model, "build_predictors", lambda *a: SimpleNamespace(
        baseline=SimpleNamespace(init=lambda *a: ({}, {}), apply=apply),
        residual=SimpleNamespace(init=forbidden, apply=forbidden)))
    monkeypatch.setattr(rollout, "shift_inputs_with_field", lambda previous, field, forcing, **k: field)
    config = V24IlyaEvalConfig(ckpt=tmp_path / "ckpt", out_json=tmp_path / "out.json", initialization_manifest=tmp_path / "manifest.json", eval_mode="cold_full")
    observer = probe.NumericalObserver()
    result = probe.standalone_baseline(config, observer)
    expected_rng = evaluation.initialization_rng_key(config.seed, records[0]["initialization_time"], stream="rollout")
    for actual in seen:
        expected_rng, expected_key, _ = jax.random.split(expected_rng, 3)
        np.testing.assert_array_equal(actual, expected_key)
    assert len(seen) == len(observer.data["forcings"]) == 40
    assert result["metrics_complete"] is True
    assert result["rs_reset_every_step"] is False
    np.testing.assert_allclose(result["original_graphcast_loss_per_initialization"][0]["baseline_per_step"], np.arange(1, 41)**2)
    assert result["original_graphcast_loss"]["baseline_per_step"] == result["original_graphcast_loss"]["full_per_step"]
