"""Canonical data contracts shared across model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CanonicalBatch:
    """Single canonical batch format used by shared pipelines."""

    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    forcings: Mapping[str, Any]
    coords: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _validate_mapping(name: str, mapping: Mapping[str, Any], *, allow_empty: bool) -> None:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"`{name}` must be a mapping, got {type(mapping)!r}.")

    if not allow_empty and not mapping:
        raise ValueError(f"`{name}` cannot be empty.")

    for key in mapping:
        if not isinstance(key, str):
            raise TypeError(f"`{name}` keys must be strings. Got key {key!r}.")


def validate_canonical_batch(batch: CanonicalBatch) -> None:
    """Validate only essential CanonicalBatch contract constraints."""

    if not isinstance(batch, CanonicalBatch):
        raise TypeError(f"`batch` must be CanonicalBatch, got {type(batch)!r}.")

    _validate_mapping("inputs", batch.inputs, allow_empty=False)
    _validate_mapping("targets", batch.targets, allow_empty=False)
    _validate_mapping("forcings", batch.forcings, allow_empty=True)
    _validate_mapping("coords", batch.coords, allow_empty=True)
    _validate_mapping("metadata", batch.metadata, allow_empty=True)
