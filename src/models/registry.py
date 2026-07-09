"""Model registry entrypoint for backend selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.models.base import ForecastModel
from src.models.graphcast.adapter import build_graphcast


Builder = Callable[[Mapping[str, Any]], ForecastModel]

_MODEL_BUILDERS: dict[str, Builder] = {
    "graphcast": build_graphcast,
}


def _extract_model_name(config: Mapping[str, Any]) -> str:
    model_name = None

    model_section = config.get("model")
    if isinstance(model_section, Mapping):
        model_name = model_section.get("name")

    if model_name is None:
        model_name = config.get("model_name")

    if not isinstance(model_name, str) or not model_name.strip():
        available = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(
            "Model name missing in config. Set `config['model']['name']` "
            f"or `config['model_name']`. Available models: {available}."
        )
    return model_name.strip().lower()


def build_model(config: Mapping[str, Any]) -> ForecastModel:
    """Build model from config using registry mapping."""

    model_name = _extract_model_name(config)
    builder = _MODEL_BUILDERS.get(model_name)
    if builder is None:
        available = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}.")
    return builder(config)
