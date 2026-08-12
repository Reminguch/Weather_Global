from __future__ import annotations

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from src.models.mamba.v22_final.checkpoint import (
    SWA_CHECKPOINT_FORMAT,
    atomic_pickle_dump,
    build_v22_final_swa,
    load_v22_final_checkpoint,
    load_v22_final_training_checkpoint,
    training_checkpoint_payload,
)
from src.models.mamba.v22_final.training.config import load_training_config


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = ROOT / "configs/experiments/v22_final/res2_gc500k_k12_open.json"


def _payload(step: int, value: float) -> dict:
    config = load_training_config(REFERENCE_CONFIG).to_dict()
    return training_checkpoint_payload(
        completed_step=step,
        residual_params={"module": {"weight": jnp.full((2,), value)}},
        residual_state={"module": {"state": jnp.full((1,), value)}},
        optimizer_state={"count": jnp.asarray(step)},
        rng_key=jnp.asarray([0, step], dtype=jnp.uint32),
        training_cursor={"epoch": 0, "segment_index": 0, "segment_offset": 0},
        resolved_training_config=config,
        baseline_checkpoint_path="baseline.npz",
        baseline_checkpoint_fingerprint="baseline-hash",
        anchor_manifest_fingerprint="manifest-hash",
        parameter_overlay_metadata={"copied": 1},
    )


def test_training_checkpoint_is_resumable_and_evaluable(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pkl"
    atomic_pickle_dump(_payload(2, 3.0), path)
    training = load_v22_final_training_checkpoint(path)
    evaluation = load_v22_final_checkpoint(path)
    assert training.completed_step == 2
    np.testing.assert_array_equal(training.residual_params["module"]["weight"], 3.0)
    assert evaluation.checkpoint_kind == "training"


def test_legacy_and_swa_are_not_exact_resume_points(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as handle:
        pickle.dump({"residual_params": {"m": {"w": np.zeros(1)}}, "residual_state": {}}, handle)
    with pytest.raises(ValueError, match="not a v22_final"):
        load_v22_final_training_checkpoint(legacy)


def test_evaluator_loader_rejects_incompatible_versioned_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-architecture.pkl"
    payload = _payload(2, 3.0)
    payload["architecture_id"] = "not_v22_final"
    atomic_pickle_dump(payload, path)
    with pytest.raises(ValueError, match="architecture_id"):
        load_v22_final_checkpoint(path)


def test_training_loader_rejects_malformed_resume_payload(tmp_path: Path) -> None:
    path = tmp_path / "malformed.pkl"
    payload = _payload(2, 3.0)
    payload["residual_state"] = "not-a-state-tree"
    atomic_pickle_dump(payload, path)
    with pytest.raises(ValueError, match="residual_state"):
        load_v22_final_training_checkpoint(path)

    path = tmp_path / "missing-optimizer.pkl"
    payload = _payload(2, 3.0)
    payload["optimizer_state"] = None
    atomic_pickle_dump(payload, path)
    with pytest.raises(ValueError, match="optimizer_state"):
        load_v22_final_training_checkpoint(path)


def test_swa_averages_floating_params_and_preserves_earliest_state(tmp_path: Path) -> None:
    first = tmp_path / "step2.pkl"
    second = tmp_path / "step4.pkl"
    output = tmp_path / "swa.pkl"
    atomic_pickle_dump(_payload(2, 2.0), first)
    atomic_pickle_dump(_payload(4, 4.0), second)
    result = build_v22_final_swa([first, second], [2, 4], output)
    assert result["checkpoint_format"] == SWA_CHECKPOINT_FORMAT
    np.testing.assert_allclose(result["residual_params"]["module"]["weight"], 3.0)
    np.testing.assert_allclose(result["residual_state"]["module"]["state"], 2.0)
    assert load_v22_final_checkpoint(output).checkpoint_kind == "swa"
    with pytest.raises(ValueError, match="not an exact-resume"):
        load_v22_final_training_checkpoint(output)


def test_swa_allows_operational_step_and_cadence_changes(tmp_path: Path) -> None:
    first = tmp_path / "step2.pkl"
    second = tmp_path / "step4.pkl"
    output = tmp_path / "swa.pkl"
    first_payload = _payload(2, 2.0)
    second_payload = _payload(4, 4.0)
    second_payload["resolved_training_config"]["optimizer"]["max_steps"] = 60_000
    second_payload["resolved_training_config"]["optimizer"]["checkpoint_every"] = 1_000
    atomic_pickle_dump(first_payload, first)
    atomic_pickle_dump(second_payload, second)
    build_v22_final_swa([first, second], [2, 4], output)
