"""Backend-independent validation for v24_Ilya training smoke artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANONICAL_METRIC_FIELDS = (
    "step",
    "loss",
    "gradient_norm",
    "learning_rate",
    "step_seconds",
)


@dataclass(frozen=True)
class SmokeValidationSummary:
    expected_steps: int
    median_steady_step_seconds: float
    gpu_peak_mib: int | None


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{label}[{index}]")
        return
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite {label}={value}")


def _require_close(record: Mapping[str, Any], alias: str, canonical: str) -> None:
    if alias not in record:
        return
    alias_value = float(record[alias])
    canonical_value = float(record[canonical])
    if not math.isclose(alias_value, canonical_value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{alias}={alias_value} does not match canonical "
            f"{canonical}={canonical_value}"
        )


def validate_smoke_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_steps: int = 5,
    max_median_step_seconds: float,
    gpu_peak_mib: int | None = None,
    max_gpu_peak_mib: int | None = None,
    require_loss_by_horizon: bool = False,
) -> SmokeValidationSummary:
    """Validate the canonical metric contract for serial or DP smoke records."""

    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    if max_median_step_seconds <= 0 or not math.isfinite(max_median_step_seconds):
        raise ValueError("max_median_step_seconds must be positive and finite")
    observed_steps = [record.get("step") for record in records]
    required_steps = list(range(1, expected_steps + 1))
    if observed_steps != required_steps:
        raise ValueError(
            f"expected metric steps {required_steps}, got {observed_steps}"
        )

    for record in records:
        step = record["step"]
        missing = [field for field in CANONICAL_METRIC_FIELDS if field not in record]
        if missing:
            raise ValueError(f"step {step} is missing canonical fields {missing}")
        for field in CANONICAL_METRIC_FIELDS[1:]:
            _assert_finite(record[field], f"step{step}.{field}")
        if require_loss_by_horizon:
            components = record.get("loss_by_horizon")
            if not isinstance(components, Mapping) or not components:
                raise ValueError(
                    f"step {step} requires a non-empty loss_by_horizon mapping"
                )
        for optional_field in (
            "loss_by_horizon",
            "lane_losses",
            "lane_loss_by_horizon",
            "phase_seconds",
        ):
            if optional_field in record:
                _assert_finite(
                    record[optional_field], f"step{step}.{optional_field}"
                )
        _require_close(record, "mean_loss", "loss")
        _require_close(record, "averaged_gradient_norm", "gradient_norm")

        if record.get("replica_divergence_checked"):
            for field in (
                "max_parameter_replica_divergence",
                "max_optimizer_replica_divergence",
            ):
                if field not in record:
                    raise ValueError(
                        f"step {step} checked replica divergence but omitted {field}"
                    )
                value = float(record[field])
                if value != 0.0:
                    raise ValueError(f"step {step} has {field}={value}, expected 0")

    steady_records = records[1:] if len(records) > 1 else records
    median_seconds = statistics.median(
        float(record["step_seconds"]) for record in steady_records
    )
    if median_seconds > max_median_step_seconds:
        raise ValueError(
            f"steady median {median_seconds:.2f}s exceeds "
            f"{max_median_step_seconds:.2f}s"
        )

    if gpu_peak_mib is not None:
        if gpu_peak_mib < 0:
            raise ValueError("gpu_peak_mib must be non-negative")
        if max_gpu_peak_mib is not None and gpu_peak_mib > max_gpu_peak_mib:
            raise ValueError(
                f"GPU peak {gpu_peak_mib} MiB exceeds {max_gpu_peak_mib} MiB"
            )

    return SmokeValidationSummary(
        expected_steps=expected_steps,
        median_steady_step_seconds=median_seconds,
        gpu_peak_mib=gpu_peak_mib,
    )


def validate_smoke_artifacts(
    metrics_path: Path,
    checkpoint_path: Path,
    **kwargs: Any,
) -> SmokeValidationSummary:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing smoke metrics: {metrics_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing smoke checkpoint: {checkpoint_path}")
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return validate_smoke_records(records, **kwargs)
