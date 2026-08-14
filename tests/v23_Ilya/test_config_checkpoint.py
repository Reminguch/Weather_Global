from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.models.mamba.v23_Ilya.checkpoint import (
    load_v23_Ilya_checkpoint,
    load_v23_Ilya_training_checkpoint,
)
from src.models.mamba.v23_Ilya.config import (
    ARCHITECTURE_ID,
    V23IlyaArchitectureConfig,
)
from src.models.mamba.v23_Ilya.training.config import (
    BPTT_BACKEND,
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
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    assert load_training_config(path) == config


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

