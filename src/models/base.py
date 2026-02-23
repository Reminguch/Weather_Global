"""Shared model contracts for training, evaluation, and rollout pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from src.data.contracts import CanonicalBatch


@dataclass(frozen=True)
class TrainState:
    """Generic immutable model state carried between training steps."""

    params: Any
    model_state: Any
    optimizer_state: Any
    step: int
    rng: Any


@dataclass(frozen=True)
class LossAndPredictions:
    """Standardized output for one model forward-and-loss call."""

    loss: Any
    predictions: Mapping[str, Any]
    metrics: Mapping[str, Any]
    next_model_state: Any


class ForecastModel(Protocol):
    """Model interface used by model-agnostic pipelines."""

    def init(self, rng: Any, sample_batch: CanonicalBatch) -> TrainState:
        """Initialize model parameters/state for the given sample batch."""

    def loss_and_predictions(
        self, train_state: TrainState, batch: CanonicalBatch, *, is_training: bool
    ) -> LossAndPredictions:
        """Compute loss and predictions for a canonical batch."""

    def predict(self, train_state: TrainState, batch: CanonicalBatch) -> Mapping[str, Any]:
        """Compute predictions only for a canonical batch."""
