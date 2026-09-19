"""Execution settings layered over the unchanged v24 training configuration.

The producer's trajectory schedule remains in the common configuration. The
stepwise backend preserves the maintained one-step model and loss operations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import V24IlyaTrainConfig, load_training_config


@dataclass(frozen=True)
class CachedExecutionConfig:
    cache_root: Path
    backend: str = "cached_stepwise"
    prefetch_batches: int = 0
    expected_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        if self.backend != "cached_stepwise":
            raise ValueError("Cached execution backend must be 'cached_stepwise'")
        if type(self.prefetch_batches) is not int or self.prefetch_batches not in (0, 1):
            raise ValueError("prefetch_batches must be 0 or 1")
        fingerprint = self.expected_manifest_sha256
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("expected_manifest_sha256 must be a lowercase SHA256 digest")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["cache_root"] = str(self.cache_root)
        return result


@dataclass(frozen=True)
class CachedTrainingConfig:
    common: V24IlyaTrainConfig
    execution: CachedExecutionConfig
    config_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        if self.common.feedback_mode != "baseline":
            raise ValueError("Cached training requires feedback_mode='baseline'")
        if self.common.loss_mode != "all_steps":
            raise ValueError("The first cached backend requires loss_mode='all_steps'")
        if self.common.distributed.mode != "single" or self.common.distributed.global_batch_size != 1:
            raise ValueError("The first cached backend requires single-device batch size 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "common": self.common.to_dict(),
            "execution": self.execution.to_dict(),
            "config_path": str(self.config_path),
        }


def load_cached_training_config(
    config_path: Path,
    *,
    cache_root: Path,
    prefetch_batches: int = 0,
    expected_manifest_sha256: str | None = None,
) -> CachedTrainingConfig:
    """Load the same JSON used by online training plus separate execution flags."""
    return CachedTrainingConfig(
        common=load_training_config(Path(config_path)),
        execution=CachedExecutionConfig(
            cache_root=Path(cache_root),
            prefetch_batches=prefetch_batches,
            expected_manifest_sha256=expected_manifest_sha256,
        ),
        config_path=Path(config_path),
    )
