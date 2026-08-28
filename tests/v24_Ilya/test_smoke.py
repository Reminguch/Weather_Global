from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.models.mamba.v24_Ilya.training.smoke import (
    validate_smoke_artifacts,
    validate_smoke_records,
)


def _serial_records() -> list[dict]:
    return [
        {
            "step": step,
            "loss": 5.0 / step,
            "gradient_norm": 0.25,
            "learning_rate": 1e-6,
            "step_seconds": 600.0 if step == 1 else 7.5,
            "loss_by_horizon": {"1": 1.0, "20": 5.0},
        }
        for step in range(1, 6)
    ]


def _data_parallel_records() -> list[dict]:
    records = _serial_records()
    for index, record in enumerate(records):
        record.update(
            {
                "mean_loss": record["loss"],
                "averaged_gradient_norm": record["gradient_norm"],
                "lane_losses": [record["loss"] - 0.1, record["loss"] + 0.1],
                "lane_loss_by_horizon": {
                    "1": [0.9, 1.1],
                    "20": [4.9, 5.1],
                },
                "phase_seconds": {"forward": 2.0, "reverse": 4.0},
                "replica_divergence_checked": index == 0,
                "max_parameter_replica_divergence": 0.0 if index == 0 else None,
                "max_optimizer_replica_divergence": 0.0 if index == 0 else None,
            }
        )
    return records


@pytest.mark.parametrize("records", [_serial_records(), _data_parallel_records()])
def test_smoke_validator_accepts_serial_and_data_parallel(records: list[dict]) -> None:
    summary = validate_smoke_records(
        records,
        max_median_step_seconds=20.0,
        gpu_peak_mib=4096,
        max_gpu_peak_mib=72 * 1024,
        require_loss_by_horizon=True,
    )
    assert summary.median_steady_step_seconds == 7.5
    assert summary.gpu_peak_mib == 4096


def test_smoke_validator_requires_canonical_fields() -> None:
    records = _data_parallel_records()
    del records[0]["loss"]
    with pytest.raises(ValueError, match="missing canonical fields.*loss"):
        validate_smoke_records(records, max_median_step_seconds=20.0)


def test_smoke_validator_rejects_nonfinite_nested_diagnostics() -> None:
    records = _data_parallel_records()
    records[2]["lane_losses"][1] = math.nan
    with pytest.raises(ValueError, match="non-finite step3.lane_losses"):
        validate_smoke_records(records, max_median_step_seconds=20.0)


def test_smoke_validator_rejects_alias_mismatch() -> None:
    records = _data_parallel_records()
    records[1]["mean_loss"] += 1.0
    with pytest.raises(ValueError, match="does not match canonical loss"):
        validate_smoke_records(records, max_median_step_seconds=20.0)


def test_smoke_validator_checks_artifact_paths(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    checkpoint = tmp_path / "checkpoint.pkl"
    with pytest.raises(FileNotFoundError, match="missing smoke metrics"):
        validate_smoke_artifacts(
            metrics,
            checkpoint,
            max_median_step_seconds=20.0,
        )
