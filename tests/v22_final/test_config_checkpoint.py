from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.models.mamba.v22_final.checkpoint import (
    V22FinalCheckpoint,
    load_v22_final_checkpoint,
    resolve_residual_state,
    validate_param_tree_compatible,
)
from src.models.mamba.v22_final.config import V22FinalEvalConfig, parse_args
from src.models.mamba.v22_final.evaluation import select_eval_indices


def test_cli_defaults_to_zero_residual_state() -> None:
    config = parse_args(
        [
            "--ckpt",
            "checkpoint.pkl",
            "--eval-mode",
            "cold_bp",
            "--out-json",
            "result.json",
            "--temporal-d-inner",
            "64",
            "--temporal-bc-groups",
            "4",
        ]
    )
    assert config.residual_state_init == "zero"
    assert config.ckpt == Path("checkpoint.pkl")
    assert config.out_json == Path("result.json")
    assert config.temporal_bc_groups == 4
    assert config.architecture.temporal_bc_groups == 4
    assert not config.reset_state_every_step


def test_cli_enables_reset_state_every_step() -> None:
    config = parse_args(
        [
            "--ckpt",
            "checkpoint.pkl",
            "--eval-mode",
            "cold_full",
            "--out-json",
            "result.json",
            "--reset-state-every-step",
        ]
    )
    assert config.reset_state_every_step


def test_cli_rejects_removed_temporal_hidden_size() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--ckpt",
                "checkpoint.pkl",
                "--eval-mode",
                "cold_bp",
                "--out-json",
                "result.json",
                "--temporal-hidden-size",
                "128",
            ]
        )


def test_config_derives_rollout_mode() -> None:
    config = V22FinalEvalConfig(
        ckpt=Path("checkpoint.pkl"),
        out_json=Path("result.json"),
        eval_mode="warm_full_reset_state",
        warmup_steps=7,
        target_steps=3,
    )
    assert config.is_warm
    assert config.is_full_feedback
    assert config.reset_state_after_warmup
    assert config.effective_warmup_steps == 7
    assert config.total_rollout_steps == 10


def test_eval_index_selection_is_seeded_sorted_and_validated() -> None:
    candidates = np.arange(10, 30, dtype=np.int64)
    first = select_eval_indices(candidates, n_samples=5, seed=7, force_idx=None)
    second = select_eval_indices(candidates, n_samples=5, seed=7, force_idx=None)
    assert first == second == sorted(first)
    assert select_eval_indices(candidates, n_samples=5, seed=7, force_idx=17) == [17]
    with pytest.raises(ValueError, match="exceeds"):
        select_eval_indices(candidates, n_samples=21, seed=7, force_idx=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [("target_steps", 0), ("warmup_steps", -1), ("n_samples", 0), ("resolution", 0.0)],
)
def test_config_rejects_invalid_numeric_values(field: str, value) -> None:
    kwargs = {
        "ckpt": Path("checkpoint.pkl"),
        "out_json": Path("result.json"),
        "eval_mode": "cold_bp",
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        V22FinalEvalConfig(**kwargs)


def test_load_checkpoint_and_resolve_states(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pkl"
    payload = {
        "residual_params": {"module": {"weight": np.zeros((2, 3), dtype=np.float32)}},
        "residual_state": {"module": {"ssm_state": np.ones((1,), dtype=np.float32)}},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    checkpoint = load_v22_final_checkpoint(path)
    assert checkpoint.has_residual_state
    zero_state = {"module": {"ssm_state": np.zeros((1,), dtype=np.float32)}}
    resolved, label = resolve_residual_state("ckpt", zero_state, checkpoint)
    assert resolved is checkpoint.residual_state
    assert label == "checkpoint"


def test_missing_checkpoint_state_warns_and_falls_back_to_zero() -> None:
    checkpoint = V22FinalCheckpoint(
        residual_params={"module": {"weight": np.zeros((1,), dtype=np.float32)}},
        residual_state=None,
    )
    zero_state = {"module": {"ssm_state": np.zeros((1,), dtype=np.float32)}}
    with pytest.warns(RuntimeWarning, match="falling back to zero"):
        resolved, label = resolve_residual_state("ckpt", zero_state, checkpoint)
    assert resolved is zero_state
    assert label == "zero_missing_checkpoint_state"


def test_checkpoint_rejects_missing_params(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pkl"
    with path.open("wb") as handle:
        pickle.dump({"residual_state": {}}, handle)
    with pytest.raises(ValueError, match="residual_params"):
        load_v22_final_checkpoint(path)


def test_parameter_compatibility_reports_shape_mismatch() -> None:
    initialized = {"module": {"weight": np.zeros((2, 3), dtype=np.float32)}}
    loaded = {"module": {"weight": np.zeros((3, 2), dtype=np.float32)}}
    with pytest.raises(ValueError, match="mismatched_shapes"):
        validate_param_tree_compatible(initialized, loaded)
