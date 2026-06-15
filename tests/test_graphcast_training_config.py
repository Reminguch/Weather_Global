from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from src.models.graphcast.training.core.config import parse_args
from src.models.graphcast.training.core.model import derive_model_config_from_checkpoint, gc
from src.models.graphcast.training.core.segments import include_bptt_loss_step
from src.models.mamba.training.segments_training import parse_gc_mamba_args
from src.models.mamba.residual_mamba.training.config import parse_args as parse_residual_mamba_args


def test_vanilla_config_accepts_grad_accum_and_mp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "standard_training",
            "--processor-msg-steps",
            "8",
            "--batch-size",
            "8",
            "--grad-accum-steps",
            "6",
        ],
    )

    cfg = parse_args()

    assert cfg.processor_msg_steps == 8
    assert cfg.batch_size == 8
    assert cfg.grad_accum_steps == 6


def test_derived_model_config_preserves_mesh2grid_edge_normalization_factor() -> None:
    base_cfg = gc.ModelConfig(
        resolution=1.0,
        mesh_size=5,
        latent_size=512,
        gnn_msg_steps=16,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
        mesh2grid_edge_normalization_factor=0.6180338738074472,
    )

    model_cfg = derive_model_config_from_checkpoint(
        base_cfg,
        resolution=2.0,
        mesh_size=4,
        latent_size=128,
        gnn_msg_steps=6,
    )

    assert model_cfg.resolution == 2.0
    assert model_cfg.mesh_size == 4
    assert model_cfg.latent_size == 128
    assert model_cfg.gnn_msg_steps == 6
    assert model_cfg.mesh2grid_edge_normalization_factor == 0.6180338738074472


def test_grad_accum_rejects_temporal_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "standard_training",
            "--grad-accum-steps",
            "2",
            "--temporal-backbone",
            "mamba",
            "--temporal-d-inner",
            "4",
        ],
    )

    with pytest.raises(ValueError, match="vanilla GraphCast"):
        parse_args()


def test_grad_accum_rejects_sequential_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "standard_training",
            "--grad-accum-steps",
            "2",
            "--sequential-segment-steps",
            "30",
        ],
    )

    with pytest.raises(ValueError, match="sequential-segment-steps"):
        parse_args()


def test_segment_config_accepts_mp8() -> None:
    cfg = parse_gc_mamba_args(["--processor-msg-steps", "8"])

    assert cfg.base_cfg.processor_msg_steps == 8


def test_gc_mamba_zero_initializes_temporal_output_by_default() -> None:
    cfg = parse_gc_mamba_args(["--temporal-backbone", "mamba", "--temporal-d-inner", "4"])

    assert cfg.base_cfg.zero_init_temporal_out is True


def test_gc_mamba_segment_ar_requires_target_steps_less_than_bptt() -> None:
    with pytest.raises(ValueError, match="target-steps.*bptt-steps"):
        parse_gc_mamba_args(["--target-steps", "4", "--bptt-steps", "4", "--len-segment", "8"])


def test_gc_mamba_segment_one_step_can_equal_bptt_boundary_case() -> None:
    cfg = parse_gc_mamba_args(["--target-steps", "1", "--bptt-steps", "1"])

    assert cfg.base_cfg.target_steps == 1


def test_gc_mamba_autoregressive_loss_mode_defaults_to_tail_uniform() -> None:
    cfg = parse_gc_mamba_args([])

    assert cfg.autoregressive_loss_mode == "tail_uniform"


def test_gc_mamba_optimizer_group_args_default_to_single_lr_behavior() -> None:
    cfg = parse_gc_mamba_args([])

    assert cfg.base_cfg.lr == 1e-4
    assert cfg.base_cfg.graphcast_lr is None
    assert cfg.base_cfg.mamba_lr is None
    assert cfg.base_cfg.lora_lr is None
    assert cfg.base_cfg.lora_rank == 0
    assert cfg.base_cfg.lora_alpha == 1.0
    assert cfg.base_cfg.lora_scope == "processor_mlp"
    assert cfg.base_cfg.adamw_beta1 == 0.9
    assert cfg.base_cfg.adamw_beta2 == 0.999
    assert cfg.base_cfg.max_grad_norm is None


def test_gc_mamba_optimizer_group_args_accept_explicit_values() -> None:
    cfg = parse_gc_mamba_args(
        [
            "--graphcast-lr",
            "3e-7",
            "--mamba-lr",
            "3e-6",
            "--lora-lr",
            "4e-6",
            "--lora-rank",
            "4",
            "--lora-alpha",
            "4",
            "--adamw-beta1",
            "0.9",
            "--adamw-beta2",
            "0.95",
            "--max-grad-norm",
            "32",
        ]
    )

    assert cfg.base_cfg.graphcast_lr == 3e-7
    assert cfg.base_cfg.mamba_lr == 3e-6
    assert cfg.base_cfg.lora_lr == 4e-6
    assert cfg.base_cfg.lora_rank == 4
    assert cfg.base_cfg.lora_alpha == 4
    assert cfg.base_cfg.adamw_beta1 == 0.9
    assert cfg.base_cfg.adamw_beta2 == 0.95
    assert cfg.base_cfg.max_grad_norm == 32


@pytest.mark.parametrize("loss_mode", ["tail_uniform", "all_bptt_uniform"])
def test_gc_mamba_autoregressive_loss_mode_accepts_choices(loss_mode: str) -> None:
    cfg = parse_gc_mamba_args(["--autoregressive-loss-mode", loss_mode])

    assert cfg.autoregressive_loss_mode == loss_mode


def test_gc_mamba_autoregressive_loss_mode_rejects_invalid_choice() -> None:
    with pytest.raises(SystemExit):
        parse_gc_mamba_args(["--autoregressive-loss-mode", "prefix_magic"])


def test_gc_mamba_mamba_lora_requires_enabled_lora() -> None:
    with pytest.raises(ValueError, match="mamba_lora.*lora-rank"):
        parse_gc_mamba_args(["--trainable-part", "mamba_lora"])


def test_gc_mamba_mamba_lora_accepts_rank() -> None:
    cfg = parse_gc_mamba_args(["--trainable-part", "mamba_lora", "--lora-rank", "4"])

    assert cfg.base_cfg.trainable_part == "mamba_lora"
    assert cfg.base_cfg.lora_rank == 4


def test_residual_mamba_segment_ar_requires_target_steps_less_than_bptt() -> None:
    with pytest.raises(ValueError, match="target-steps.*bptt-steps"):
        parse_residual_mamba_args(
            [
                "--baseline-ckpt",
                "baseline.npz",
                "--target-steps",
                "4",
                "--bptt-steps",
                "4",
                "--len-segment",
                "8",
            ]
        )


def test_residual_mamba_autoregressive_loss_mode_defaults_to_tail_uniform() -> None:
    cfg = parse_residual_mamba_args(["--baseline-ckpt", "baseline.npz"])

    assert cfg.autoregressive_loss_mode == "tail_uniform"


@pytest.mark.parametrize("loss_mode", ["tail_uniform", "all_bptt_uniform"])
def test_residual_mamba_autoregressive_loss_mode_accepts_choices(loss_mode: str) -> None:
    cfg = parse_residual_mamba_args(
        ["--baseline-ckpt", "baseline.npz", "--autoregressive-loss-mode", loss_mode]
    )

    assert cfg.autoregressive_loss_mode == loss_mode


def test_residual_mamba_autoregressive_loss_mode_rejects_invalid_choice() -> None:
    with pytest.raises(SystemExit):
        parse_residual_mamba_args(
            ["--baseline-ckpt", "baseline.npz", "--autoregressive-loss-mode", "prefix_magic"]
        )


def test_include_bptt_loss_step_modes() -> None:
    assert not include_bptt_loss_step("tail_uniform", bptt_i=2, truth_prefix_steps=3)
    assert include_bptt_loss_step("tail_uniform", bptt_i=3, truth_prefix_steps=3)
    assert include_bptt_loss_step("all_bptt_uniform", bptt_i=0, truth_prefix_steps=3)
    assert include_bptt_loss_step("all_bptt_uniform", bptt_i=3, truth_prefix_steps=3)
    with pytest.raises(ValueError, match="Unknown autoregressive loss mode"):
        include_bptt_loss_step("prefix_magic", bptt_i=0, truth_prefix_steps=3)
