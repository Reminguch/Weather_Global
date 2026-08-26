from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_models.verify_v24_Ilya_fp32_validation import (
    EXPECTED_BY_HORIZON,
    EXPECTED_LOSS,
    verify_fp32_validation,
)


def _write_result(path: Path, *, loss: float = EXPECTED_LOSS) -> None:
    path.write_text(
        json.dumps(
            {
                "baseline": {
                    "role": "zero_residual_baseline",
                    "loss": loss,
                    "loss_by_horizon": EXPECTED_BY_HORIZON,
                }
            }
        ),
        encoding="utf-8",
    )


def test_fp32_validation_gate_accepts_matched_loss(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    _write_result(path)
    assert verify_fp32_validation(path)["loss"] == EXPECTED_LOSS


def test_fp32_validation_gate_rejects_raw_bf16_regime(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    _write_result(path, loss=8.113585)
    with pytest.raises(ValueError, match="raw-BF16 baseline"):
        verify_fp32_validation(path)
