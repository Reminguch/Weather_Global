from __future__ import annotations

import dataclasses
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from src.models.mamba.v22_final.checkpoint import load_v22_final_training_checkpoint
from src.models.mamba.v22_final.training.config import (
    V22FinalTrainInvocation,
    V22FinalValidationConfig,
    load_training_config,
)
from src.models.mamba.v22_final.training.runner import run_training


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/experiments/v22_final"
RUN_GPU = os.environ.get("V22_FINAL_TRAIN_GPU_TESTS") == "1"
DETERMINISTIC_XLA_FLAG = "--xla_gpu_deterministic_ops=true"
LEGACY_TEMPORAL_HIDDEN_SIZE = 128
pytestmark = pytest.mark.skipif(
    not RUN_GPU,
    reason="set V22_FINAL_TRAIN_GPU_TESTS=1 on an A100 to run training parity tests",
)


@pytest.fixture(scope="module", autouse=True)
def _require_deterministic_gpu_reductions() -> None:
    if RUN_GPU and DETERMINISTIC_XLA_FLAG not in os.environ.get("XLA_FLAGS", ""):
        pytest.fail(
            "GPU parity requires XLA_FLAGS to include "
            f"{DETERMINISTIC_XLA_FLAG!r} before Python starts"
        )


@pytest.fixture(autouse=True)
def _release_compiled_graphs_between_tests():
    yield
    jax.clear_caches()


def _assert_tree_allclose(actual, expected) -> None:
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=1e-6,
            atol=1e-6,
        )


def _new_run(config, *, config_path: Path, resume: Path | None = None) -> Path:
    result = run_training(
        V22FinalTrainInvocation(
            config=config,
            config_path=config_path,
            resume=resume,
        )
    )
    assert result is not None
    return result


@pytest.mark.parametrize(
    ("config_name", "legacy_feedback", "label"),
    [
        ("res2_gc500k_k12_open.json", "baseline", "open"),
        ("res2_gc500k_k12_closed_sg.json", "closed_loop_sg", "closed"),
    ],
)
def test_legacy_v20_two_update_parity(
    tmp_path: Path,
    config_name: str,
    legacy_feedback: str,
    label: str,
) -> None:
    config_path = CONFIG_ROOT / config_name
    base = dataclasses.replace(
        load_training_config(config_path),
        validation=V22FinalValidationConfig(enabled=False),
    )
    legacy_root = tmp_path / "legacy"
    command = [
        sys.executable,
        "scripts/training/full_mamba_v20/train_mz_v20.py",
        "--prepared-root", str(base.prepared_root),
        "--residual-root", str(base.anchor_manifest_root),
        "--ckpt-in", str(base.baseline_checkpoint),
        "--stats-dir", str(base.stats_dir),
        "--out-dir", str(legacy_root),
        "--run-name", label,
        "--resolution", str(base.architecture.resolution),
        "--mesh-size", str(base.architecture.mesh_size),
        "--width", str(base.architecture.width),
        "--baseline-msg-steps", str(base.architecture.baseline_msg_steps),
        "--residual-msg-steps", str(base.architecture.residual_msg_steps),
        "--input-duration", base.input_duration,
        "--target-steps", "1",
        "--batch-size", "1",
        "--max-steps", "2",
        "--checkpoint-every", "1",
        "--lr", str(base.learning_rate),
        "--weight-decay", str(base.weight_decay),
        "--seed", str(base.seed),
        "--precision", base.precision,
        "--grad-clip", str(base.grad_clip),
        "--warmup-steps", str(base.warmup_steps),
        "--sequential-segment-steps", str(base.segment_steps),
        "--bptt-steps", str(base.bptt_steps),
        "--ar-tail-K", str(base.ar_tail_k),
        "--feedback-mode", legacy_feedback,
        "--temporal-location", base.architecture.temporal_location,
        "--temporal-hidden-size", str(LEGACY_TEMPORAL_HIDDEN_SIZE),
        "--temporal-d-inner", str(base.architecture.temporal_d_inner),
        "--temporal-d-state", str(base.architecture.temporal_d_state),
        "--temporal-d-conv", str(base.architecture.temporal_d_conv),
        "--temporal-dt-rank", base.architecture.temporal_dt_rank,
        "--temporal-layers", str(base.architecture.temporal_layers),
        "--no-temporal-stateful",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    with (legacy_root / label / "v13_residual_step1.pkl").open("rb") as handle:
        legacy_step_one = pickle.load(handle)
    with (legacy_root / label / "v13_residual_step2.pkl").open("rb") as handle:
        legacy_checkpoint = pickle.load(handle)
    legacy_log = json.loads((legacy_root / label / "train_log.json").read_text())

    new_config = dataclasses.replace(
        base,
        output_root=tmp_path / "new",
        run_name=label,
        max_steps=2,
        checkpoint_every=1,
    )
    new_path = _new_run(new_config, config_path=config_path)
    new_step_one = load_v22_final_training_checkpoint(
        new_config.run_dir / "checkpoints/checkpoint_step00000001.pkl"
    )
    new_checkpoint = load_v22_final_training_checkpoint(new_path)
    new_log = [
        json.loads(line)
        for line in (new_config.run_dir / "train_metrics.jsonl").read_text().splitlines()
    ]
    _assert_tree_allclose(new_checkpoint.residual_params, legacy_checkpoint["residual_params"])
    _assert_tree_allclose(new_checkpoint.residual_state, legacy_checkpoint["residual_state"])
    assert sum(
        int(leaf.size) for leaf in jax.tree_util.tree_leaves(new_checkpoint.residual_params)
    ) == 10_344_655
    for checkpoint in (legacy_step_one, {"residual_params": new_step_one.residual_params}):
        zero_head = checkpoint["residual_params"]["temporal_residual_head"]
        for leaf in zero_head.values():
            np.testing.assert_array_equal(np.asarray(leaf), 0.0)
    np.testing.assert_allclose(
        [record["loss"] for record in new_log],
        [record["loss"] for record in legacy_log],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [record["gradient_norm"] for record in new_log],
        [record["grad_norm"] for record in legacy_log],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("segment_steps", "validation_enabled"),
    [(64, False), (32, False), (64, True)],
)
def test_exact_resume_matches_continuous_training(
    tmp_path: Path,
    segment_steps: int,
    validation_enabled: bool,
) -> None:
    config_path = CONFIG_ROOT / "res2_gc500k_k12_open.json"
    base = dataclasses.replace(
        load_training_config(config_path),
        validation=V22FinalValidationConfig(enabled=False),
    )
    validation = V22FinalValidationConfig(
        enabled=validation_enabled,
        every_steps=2,
        num_segments=2,
        final_num_segments=2,
    )
    continuous = dataclasses.replace(
        base,
        validation=validation,
        output_root=tmp_path / "continuous",
        run_name=f"continuous-segment-{segment_steps}-{validation_enabled}",
        segment_steps=segment_steps,
        max_steps=4,
        checkpoint_every=4,
    )
    continuous_path = _new_run(continuous, config_path=config_path)

    split = dataclasses.replace(
        base,
        validation=validation,
        output_root=tmp_path / "split",
        run_name=f"split-segment-{segment_steps}-{validation_enabled}",
        segment_steps=segment_steps,
        max_steps=2,
        checkpoint_every=2,
    )
    split_path = _new_run(split, config_path=config_path)
    resumed = dataclasses.replace(split, max_steps=4, checkpoint_every=4)
    resumed_path = _new_run(resumed, config_path=config_path, resume=split_path)

    continuous_checkpoint = load_v22_final_training_checkpoint(continuous_path)
    resumed_checkpoint = load_v22_final_training_checkpoint(resumed_path)
    _assert_tree_allclose(resumed_checkpoint.residual_params, continuous_checkpoint.residual_params)
    _assert_tree_allclose(resumed_checkpoint.residual_state, continuous_checkpoint.residual_state)
    _assert_tree_allclose(resumed_checkpoint.optimizer_state, continuous_checkpoint.optimizer_state)
    np.testing.assert_array_equal(resumed_checkpoint.rng_key, continuous_checkpoint.rng_key)
    assert resumed_checkpoint.training_cursor == continuous_checkpoint.training_cursor
    continuous_log = [
        json.loads(line)
        for line in (continuous.run_dir / "train_metrics.jsonl").read_text().splitlines()
    ]
    resumed_log = [
        json.loads(line)
        for line in (split.run_dir / "train_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in resumed_log] == [1, 2, 3, 4]
    for field in ("loss", "gradient_norm", "learning_rate"):
        np.testing.assert_allclose(
            [record[field] for record in resumed_log],
            [record[field] for record in continuous_log],
            rtol=1e-6,
            atol=1e-6,
        )
    for field in ("epoch", "segment_index", "segment_offset"):
        assert [record[field] for record in resumed_log] == [
            record[field] for record in continuous_log
        ]
    if validation_enabled:
        continuous_validation = [
            json.loads(line)
            for line in (continuous.run_dir / "validation_metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        resumed_validation = [
            json.loads(line)
            for line in (split.run_dir / "validation_metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [
            (record["step"], record["role"], record["segment_ids"])
            for record in resumed_validation
        ] == [
            (record["step"], record["role"], record["segment_ids"])
            for record in continuous_validation
        ]
        np.testing.assert_allclose(
            [record["loss"] for record in resumed_validation],
            [record["loss"] for record in continuous_validation],
            rtol=1e-6,
            atol=1e-6,
        )


def test_validation_does_not_change_training_trajectory(tmp_path: Path) -> None:
    config_path = CONFIG_ROOT / "res2_gc500k_k12_closed_sg.json"
    base = load_training_config(config_path)
    disabled = dataclasses.replace(
        base,
        validation=V22FinalValidationConfig(enabled=False),
        output_root=tmp_path / "disabled",
        run_name="disabled",
        max_steps=2,
        checkpoint_every=1,
    )
    enabled = dataclasses.replace(
        disabled,
        validation=V22FinalValidationConfig(
            enabled=True, every_steps=1, num_segments=2, final_num_segments=2
        ),
        output_root=tmp_path / "enabled",
        run_name="enabled",
    )
    disabled_path = _new_run(disabled, config_path=config_path)
    enabled_path = _new_run(enabled, config_path=config_path)
    disabled_checkpoint = load_v22_final_training_checkpoint(disabled_path)
    enabled_checkpoint = load_v22_final_training_checkpoint(enabled_path)
    _assert_tree_allclose(
        enabled_checkpoint.residual_params, disabled_checkpoint.residual_params
    )
    _assert_tree_allclose(
        enabled_checkpoint.residual_state, disabled_checkpoint.residual_state
    )
    _assert_tree_allclose(
        enabled_checkpoint.optimizer_state, disabled_checkpoint.optimizer_state
    )
    np.testing.assert_array_equal(
        enabled_checkpoint.rng_key, disabled_checkpoint.rng_key
    )
    assert enabled_checkpoint.training_cursor == disabled_checkpoint.training_cursor
    records = [
        json.loads(line)
        for line in (enabled.run_dir / "validation_metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    fixed = [record for record in records if record["role"] == "fixed_checkpoint"]
    assert [record["step"] for record in fixed] == [1, 2]
    assert fixed[0]["segment_ids"] == fixed[1]["segment_ids"]
