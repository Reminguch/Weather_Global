from __future__ import annotations

import pytest

from src.models.graphcast.adapter import GraphCastAdapter
from src.models.registry import build_model


def test_build_graphcast_model() -> None:
    config = {
        "model": {
            "name": "graphcast",
            "graphcast": {"backend": "stub", "use_normalization": False},
        }
    }
    model = build_model(config)
    assert isinstance(model, GraphCastAdapter)


def test_unknown_model_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"model": {"name": "does-not-exist"}})
