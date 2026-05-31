from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from src.models.graphcast.training.core.config import parse_args
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


def test_gc_mamba_segment_ar_requires_target_steps_less_than_bptt() -> None:
    with pytest.raises(ValueError, match="target-steps.*bptt-steps"):
        parse_gc_mamba_args(["--target-steps", "4", "--bptt-steps", "4", "--len-segment", "8"])


def test_gc_mamba_segment_one_step_can_equal_bptt_boundary_case() -> None:
    cfg = parse_gc_mamba_args(["--target-steps", "1", "--bptt-steps", "1"])

    assert cfg.base_cfg.target_steps == 1


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
