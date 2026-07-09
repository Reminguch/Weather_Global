"""Model-agnostic training entrypoints."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from src.data.contracts import CanonicalBatch, validate_canonical_batch
from src.models.base import ForecastModel, TrainState
from src.models.registry import build_model


def initialize_training(
    config: Mapping[str, Any], *, rng: Any, sample_batch: CanonicalBatch
) -> tuple[ForecastModel, TrainState]:
    """Build model and initialize train state."""

    validate_canonical_batch(sample_batch)
    model = build_model(config)
    train_state = model.init(rng, sample_batch)
    return model, train_state


def train_one_step(
    model: ForecastModel, train_state: TrainState, batch: CanonicalBatch
) -> tuple[TrainState, Mapping[str, Any], Mapping[str, Any]]:
    """Run one generic train step through the shared model contract."""

    validate_canonical_batch(batch)
    output = model.loss_and_predictions(train_state, batch, is_training=True)
    metrics = dict(output.metrics)
    metrics.setdefault("loss", output.loss)
    next_state = replace(
        train_state,
        model_state=output.next_model_state,
        step=train_state.step + 1,
    )
    return next_state, metrics, output.predictions
