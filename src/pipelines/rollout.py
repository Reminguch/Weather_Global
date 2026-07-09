"""Model-agnostic rollout entrypoints."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.data.contracts import CanonicalBatch, validate_canonical_batch
from src.models.base import ForecastModel, TrainState
from src.models.registry import build_model


def initialize_rollout(
    config: Mapping[str, Any], *, rng: Any, sample_batch: CanonicalBatch
) -> tuple[ForecastModel, TrainState]:
    """Build model and initialize state for rollout."""

    validate_canonical_batch(sample_batch)
    model = build_model(config)
    train_state = model.init(rng, sample_batch)
    return model, train_state


def rollout_batches(
    model: ForecastModel, train_state: TrainState, batches: Iterable[CanonicalBatch]
) -> list[Mapping[str, Any]]:
    """Run repeated prediction calls over a sequence of canonical batches."""

    outputs: list[Mapping[str, Any]] = []
    for batch in batches:
        validate_canonical_batch(batch)
        outputs.append(model.predict(train_state, batch))
    return outputs
