from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.data.contracts import CanonicalBatch
from src.models.base import LossAndPredictions, TrainState
from src.pipelines.train import train_one_step


class FakeForecastModel:
    def init(self, rng: Any, sample_batch: CanonicalBatch) -> TrainState:
        return TrainState(params={}, model_state={}, optimizer_state=None, step=0, rng=rng)

    def loss_and_predictions(
        self, train_state: TrainState, batch: CanonicalBatch, *, is_training: bool
    ) -> LossAndPredictions:
        del is_training
        target = np.asarray(batch.targets["temperature"])
        prediction = np.asarray(batch.inputs["temperature"])
        loss = float(np.mean((prediction - target) ** 2))
        return LossAndPredictions(
            loss=loss,
            predictions={"temperature": prediction},
            metrics={"loss": loss},
            next_model_state={"touched": True},
        )

    def predict(self, train_state: TrainState, batch: CanonicalBatch) -> Mapping[str, Any]:
        del train_state
        return {"temperature": np.asarray(batch.inputs["temperature"])}


def test_one_train_step_runs_and_increments_step() -> None:
    model = FakeForecastModel()
    train_state = model.init(
        rng=np.array([0, 1], dtype=np.uint32),
        sample_batch=CanonicalBatch(
            inputs={"temperature": np.array([[1.0]], dtype=np.float32)},
            targets={"temperature": np.array([[1.5]], dtype=np.float32)},
            forcings={},
            coords={},
            metadata={},
        ),
    )
    batch = CanonicalBatch(
        inputs={"temperature": np.array([[2.0]], dtype=np.float32)},
        targets={"temperature": np.array([[1.0]], dtype=np.float32)},
        forcings={},
        coords={},
        metadata={},
    )

    next_state, metrics, predictions = train_one_step(model, train_state, batch)

    assert next_state.step == train_state.step + 1
    assert "loss" in metrics
    assert "temperature" in predictions
    assert next_state.model_state["touched"] is True
