"""Model-agnostic evaluation entrypoints."""

from __future__ import annotations

from typing import Any, Mapping

from src.data.contracts import CanonicalBatch, validate_canonical_batch
from src.models.base import ForecastModel, TrainState
from src.models.registry import build_model


def initialize_evaluation(
    config: Mapping[str, Any], *, rng: Any, sample_batch: CanonicalBatch
) -> tuple[ForecastModel, TrainState]:
    """Build model and initialize state for evaluation."""

    validate_canonical_batch(sample_batch)
    model = build_model(config)
    train_state = model.init(rng, sample_batch)
    return model, train_state


def evaluate_batch(
    model: ForecastModel, train_state: TrainState, batch: CanonicalBatch
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return metrics and predictions for one evaluation batch."""

    validate_canonical_batch(batch)
    output = model.loss_and_predictions(train_state, batch, is_training=False)
    metrics = dict(output.metrics)
    metrics.setdefault("loss", output.loss)
    return metrics, output.predictions
