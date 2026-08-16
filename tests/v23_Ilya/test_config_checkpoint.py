from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.models.mamba.v23_Ilya.checkpoint import (
    load_v23_Ilya_checkpoint,
    load_v23_Ilya_training_checkpoint,
    overlay_frozen_baseline_params,
)
from src.models.mamba.v23_Ilya.config import (
    ARCHITECTURE_ID,
    V23IlyaArchitectureConfig,
)
from src.models.mamba.v23_Ilya.training.config import (
    BPTT_BACKEND,
    V23IlyaDistributedConfig,
    V23IlyaTrainConfig,
    load_training_config,
)


def _config() -> V23IlyaTrainConfig:
    return V23IlyaTrainConfig(
        prepared_root=Path("prepared"),
        anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline.npz"),
        output_root=Path("output"),
        run_name="endpoint",
        architecture=V23IlyaArchitectureConfig(),
        segment_steps=24,
        bptt_steps=24,
        ar_tail_k=20,
    )


def test_endpoint_defaults_and_roundtrip(tmp_path: Path) -> None:
    config = _config()
    assert config.loss_mode == "last_step"
    assert config.weather_tape_precision == "bf16"
    assert config.truth_prefix_steps == 4
    payload = config.to_dict()
    assert payload["objective"] == {"loss_mode": "last_step"}
    assert payload["memory"] == {
        "weather_tape_precision": "bf16",
        "bptt_backend": BPTT_BACKEND,
    }
    assert payload["distributed"] == {
        "mode": "single",
        "num_devices": 1,
        "per_device_batch_size": 1,
        "drop_incomplete_replica_group": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    assert load_training_config(path) == config


def test_frozen_baseline_overlay_accepts_and_records_source_only_head() -> None:
    initialized = {
        "encoder": {"w": np.zeros((2, 3), np.float32)},
        "decoder/grid": {"b": np.zeros((4,), np.float32)},
    }
    source = {
        "encoder": {"w": np.ones((2, 3), np.float32)},
        "decoder/grid": {"b": np.ones((4,), np.float32)},
        "decoder/mesh": {
            "w": np.ones((3, 4), np.float32),
            "b": np.ones((4,), np.float32),
        },
    }

    loaded, stats = overlay_frozen_baseline_params(initialized, source)

    assert set(loaded) == set(initialized)
    assert np.array_equal(loaded["encoder"]["w"], source["encoder"]["w"])
    assert np.array_equal(loaded["decoder/grid"]["b"], source["decoder/grid"]["b"])
    assert stats.copied == 2
    assert stats.ignored_source == (
        "decoder/mesh/b",
        "decoder/mesh/w",
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "missing_modules"),
        ({"module": {}}, "missing_params"),
        (
            {"module": {"w": np.zeros((3,), np.float32)}},
            "mismatched_shapes",
        ),
    ],
)
def test_frozen_baseline_overlay_rejects_incomplete_target_coverage(
    source: dict,
    message: str,
) -> None:
    initialized = {"module": {"w": np.zeros((2,), np.float32)}}
    with pytest.raises(ValueError, match=message):
        overlay_frozen_baseline_params(initialized, source)


@pytest.mark.parametrize("loss_mode", ["last_step", "all_steps"])
@pytest.mark.parametrize("tape_precision", ["bf16", "fp32"])
def test_supported_objective_and_tape_modes(loss_mode: str, tape_precision: str) -> None:
    config = V23IlyaTrainConfig(
        **{
            **_config().__dict__,
            "loss_mode": loss_mode,
            "weather_tape_precision": tape_precision,
        }
    )
    assert config.loss_mode == loss_mode
    assert config.weather_tape_precision == tape_precision


def test_config_rejects_unknown_keys_and_backend(tmp_path: Path) -> None:
    payload = _config().to_dict()
    payload["objective"]["unexpected"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown keys in objective"):
        load_training_config(path)

    payload = _config().to_dict()
    payload["memory"]["bptt_backend"] = "ordinary_scan"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bptt_backend"):
        load_training_config(path)


def test_data_parallel_config_is_strict_and_roundtrips(tmp_path: Path) -> None:
    distributed = V23IlyaDistributedConfig(
        mode="data_parallel",
        num_devices=4,
        per_device_batch_size=1,
        drop_incomplete_replica_group=True,
    )
    config = V23IlyaTrainConfig(
        **{**_config().__dict__, "distributed": distributed}
    )
    path = tmp_path / "distributed.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_training_config(path) == config
    assert config.distributed.global_batch_size == 4

    payload = config.to_dict()
    payload["distributed"]["unexpected"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown keys in distributed"):
        load_training_config(path)


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "unknown"},
        {"mode": "single", "num_devices": 4},
        {"mode": "data_parallel", "num_devices": 1},
        {"mode": "data_parallel", "num_devices": 4, "per_device_batch_size": 2},
        {
            "mode": "data_parallel",
            "num_devices": 4,
            "drop_incomplete_replica_group": False,
        },
    ],
)
def test_distributed_config_rejects_invalid_values(values: dict) -> None:
    with pytest.raises(ValueError, match="distributed|per_device|drop_incomplete"):
        V23IlyaDistributedConfig(**values)


def test_v23_owned_warm_start_parser_accepts_v22_payload(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pkl"
    payload = {
        "architecture_id": "v22_final",
        "schema_version": 1,
        "checkpoint_format": "v22_final_training_pickle",
        "checkpoint_kind": "training",
        "residual_params": {"module": {"w": np.ones((2,), np.float32)}},
        "residual_state": {"module": {"state": np.zeros((1,), np.float32)}},
    }
    path.write_bytes(pickle.dumps(payload))
    checkpoint = load_v23_Ilya_checkpoint(path)
    assert checkpoint.checkpoint_format == "v22_final_training_pickle"
    assert checkpoint.has_residual_state
    with pytest.raises(ValueError, match=f"not a {ARCHITECTURE_ID} checkpoint"):
        load_v23_Ilya_training_checkpoint(path)


def test_data_selection_fields_are_strict_and_roundtrip(tmp_path: Path) -> None:
    config = V23IlyaTrainConfig(
        **{
            **_config().__dict__,
            "time_start": "2021-01-01T00:00:00",
            "time_end": "2021-01-31T18:00:00",
            "allow_incomplete_prepared_store": True,
        }
    )
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_training_config(path) == config

    payload = config.to_dict()
    payload["data"]["unknown_partial_option"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown keys in data"):
        load_training_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"time_start": "2021-01-01T00:00:00"},
        {"time_end": "2021-01-31T18:00:00"},
        {"allow_incomplete_prepared_store": True},
        {"allow_incomplete_prepared_store": "true"},
    ],
)
def test_data_selection_fields_reject_invalid_combinations(overrides: dict) -> None:
    with pytest.raises(ValueError, match=r"data\.|provided together|requires"):
        V23IlyaTrainConfig(**{**_config().__dict__, **overrides})

