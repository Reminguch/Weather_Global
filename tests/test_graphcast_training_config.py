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
