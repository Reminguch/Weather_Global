from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.mamba.v24_Ilya.checkpoint import (
    TRAINING_CHECKPOINT_FORMAT,
    load_v24_Ilya_training_checkpoint,
)
from src.models.mamba.v24_Ilya.config import (
    ARCHITECTURE_ID,
    SCHEMA_VERSION,
    V24IlyaArchitectureConfig,
)
from src.models.mamba.v24_Ilya.training import runner
from src.models.mamba.v24_Ilya.training.config import (
    V24IlyaTrainConfig,
    V24IlyaTrainInvocation,
    parse_cli,
)
from src.models.mamba.v24_Ilya.training.data import TrainingCursor


def _config(tmp_path: Path) -> V24IlyaTrainConfig:
    return V24IlyaTrainConfig(
        prepared_root=tmp_path / "prepared",
        anchor_manifest_root=tmp_path / "anchors",
        baseline_checkpoint=tmp_path / "baseline.npz",
        output_root=tmp_path / "runs",
        run_name="finetune",
        architecture=V24IlyaArchitectureConfig(residual_initialization="fresh"),
        input_duration=None,
        segment_steps=24,
        bptt_steps=24,
        ar_tail_k=20,
        max_steps=1,
        warmup_steps=0,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--resume", "saved.pkl"],
        ["--init-from", "saved.pkl", "--baseline-validation-only"],
        ["--init-from", "saved.pkl", "--validation-compare", "compare.pkl"],
    ],
)
def test_zero_state_requires_training_init_from(tmp_path, arguments):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(tmp_path).to_dict()))
    with pytest.raises(ValueError, match="--zero-state-on-init requires --init-from"):
        parse_cli(["--config", str(path), "--zero-state-on-init", *arguments])


def test_zero_state_cli_default_and_opt_in(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(tmp_path).to_dict()))
    argv = ["--config", str(path), "--init-from", "saved.pkl"]
    assert not parse_cli(argv).zero_state_on_init
    assert parse_cli([*argv, "--zero-state-on-init"]).zero_state_on_init
    with pytest.raises(ValueError, match="--zero-state-on-init requires --init-from"):
        V24IlyaTrainInvocation(_config(tmp_path), path, zero_state_on_init=True)


@pytest.mark.parametrize("zero_state", [False, True])
def test_warmstart_first_chunk_preserves_params_and_resets_only_requested_state(
    tmp_path, monkeypatch, zero_state
):
    """Exercise the real runner through its first update and checkpoint save."""
    config = _config(tmp_path)
    config.baseline_checkpoint.write_bytes(b"baseline fixture")
    initialized_params = {"module": {"w": jnp.zeros((2,), dtype=jnp.float32)}}
    zero = {"module": {"state": jnp.zeros((1, 2), dtype=jnp.float32)}}
    loaded_params = {"module": {"w": np.full((2,), 7, dtype=np.float32)}}
    loaded_state = {"module": {"state": np.full((1, 2), 9, dtype=np.float32)}}
    source = tmp_path / "source.pkl"
    source.write_bytes(pickle.dumps({
        "architecture_id": ARCHITECTURE_ID,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
        "checkpoint_kind": "training",
        "residual_params": loaded_params,
        "residual_state": loaded_state,
        "optimizer_state": {"old_count": 999},
        "rng_key": np.array([123, 456], dtype=np.uint32),
        "training_cursor": {"epoch": 23, "segment_index": 62, "segment_offset": 0},
        "completed_step": 10_000,
    }))
    source_bytes = source.read_bytes()
    baseline = SimpleNamespace(
        task_config=None, model_config=None, params=initialized_params,
    )
    transform = SimpleNamespace(init=lambda *args: (initialized_params, zero))
    transforms = SimpleNamespace(
        residual_loss=transform, residual_predict=transform, baseline_predict=transform,
    )

    @dataclass
    class DataReport:
        frames: int = 1

    cursors = []

    def build_chunk(cursor, *args):
        cursors.append(cursor)
        return SimpleNamespace(
            input_frames=(), static_inputs=None, truths=(), forcings=(),
            data_report=DataReport(), next_cursor=TrainingCursor(segment_index=1),
        )

    training_data = SimpleNamespace(
        anchor_indices=np.array([0]), train_split=np.array([0]), val_split=np.array([1]),
        input_steps=2, time_step=np.timedelta64(6, "h"),
        segments=[0], validation_segments=[1], manifest_fingerprint="anchors-fingerprint",
        store=SimpleNamespace(
            build_batch_from_indices=lambda **kwargs: (None, None, None),
            selection_metadata={},
        ),
        build_chunk=build_chunk,
    )
    monkeypatch.setattr(runner, "validate_input_paths", lambda config: None)
    monkeypatch.setattr(runner, "load_graphcast_checkpoint", lambda path: baseline)
    monkeypatch.setattr(runner, "load_stats", lambda path: None)
    monkeypatch.setattr(runner, "validate_stats_coverage", lambda *args: None)
    monkeypatch.setattr(runner, "open_training_data", lambda *args: training_data)
    monkeypatch.setattr(runner, "build_model_configs", lambda *args: None)
    monkeypatch.setattr(runner, "build_training_transforms", lambda *args: transforms)
    monkeypatch.setattr(runner, "cast_physical_boundary_fp32", lambda value: value)
    monkeypatch.setattr(runner, "memory_contract", lambda **kwargs: {})
    monkeypatch.setattr(runner, "_device_memory_snapshot", lambda: None)
    observed = {}

    def train_step(params, state, optimizer_state, keys, *args):
        observed.update(params=params, state=state, optimizer_state=optimizer_state, keys=keys)
        return params, state, optimizer_state, jnp.array(1.), jnp.array(1.), jnp.array([1.])

    monkeypatch.setattr(runner, "make_train_step", lambda **kwargs: train_step)
    saved = runner.run_training(V24IlyaTrainInvocation(
        config, tmp_path / "config.json", init_from=source, zero_state_on_init=zero_state,
    ))
    np.testing.assert_array_equal(observed["params"]["module"]["w"], loaded_params["module"]["w"])
    np.testing.assert_array_equal(observed["state"]["module"]["state"], 0 if zero_state else 9)
    optimizer, _ = runner.build_optimizer(config)
    expected_optimizer_state = optimizer.init(initialized_params)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(observed["optimizer_state"]),
        jax.tree_util.tree_leaves(expected_optimizer_state), strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    expected_rng = jax.random.PRNGKey(config.seed)
    expected_rng, _ = jax.random.split(expected_rng)  # residual initialization
    expected_rng, _ = jax.random.split(expected_rng)  # baseline initialization
    np.testing.assert_array_equal(
        observed["keys"], jax.random.split(expected_rng, config.bptt_steps + 1)[1:],
    )
    assert cursors == [TrainingCursor(), TrainingCursor()]
    checkpoint = load_v24_Ilya_training_checkpoint(saved)
    assert checkpoint.completed_step == 1
    assert checkpoint.parameter_overlay_metadata["init_state_policy"] == (
        "zero" if zero_state else "checkpoint"
    )
    assert checkpoint.parameter_overlay_metadata["zero_state_on_init"] is zero_state
    metadata = json.loads((config.run_dir / "run_config.json").read_text())
    assert metadata["derived"]["parameter_overlay"] == checkpoint.parameter_overlay_metadata
    assert source.read_bytes() == source_bytes
