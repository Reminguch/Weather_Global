from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CanonicalBatch:
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    forcings: Mapping[str, Any]
    coords: Mapping[str, Any]
    metadata: Mapping[str, Any]


def validate_canonical_batch(batch: CanonicalBatch) -> None:
    if not batch.inputs:
        raise ValueError("inputs must not be empty")
    if not batch.targets:
        raise ValueError("targets must not be empty")

