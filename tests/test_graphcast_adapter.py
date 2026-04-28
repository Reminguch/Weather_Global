from __future__ import annotations

import numpy as np
import pytest

from src.data_operations.contracts import CanonicalBatch
from src.models.graphcast.adapter import build_graphcast


def _tiny_fake_batch() -> CanonicalBatch:
    return CanonicalBatch(
        inputs={"temperature": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
        targets={"temperature": np.array([[1.5, 1.5, 1.5]], dtype=np.float32)},
        forcings={"time_progress": np.array([[0.1, 0.2, 0.3]], dtype=np.float32)},
        coords={"batch_id": np.array([0], dtype=np.int32)},
        metadata={},
    )


def test_graphcast_adapter_forward_tiny_fake_batch() -> None:
    model = build_graphcast(
        {
            "model": {
                "name": "graphcast",
                "graphcast": {"backend": "stub", "use_normalization": False},
            }
        }
    )
    batch = _tiny_fake_batch()
    train_state = model.init(rng=np.array([0, 1], dtype=np.uint32), sample_batch=batch)

    output = model.loss_and_predictions(train_state, batch, is_training=True)
    assert np.isfinite(float(np.asarray(output.loss)))
    assert output.predictions.keys() == batch.targets.keys()
    assert "loss" in output.metrics

    predictions = model.predict(train_state, batch)
    assert predictions.keys() == batch.targets.keys()


def test_graphcast_dependency_optional_for_real_backend() -> None:
    pytest.importorskip("graphcast")
    with pytest.raises(ValueError, match="model_config"):
        build_graphcast({"model": {"name": "graphcast", "graphcast": {"backend": "graphcast"}}})
