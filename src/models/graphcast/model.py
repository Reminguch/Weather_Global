"""GraphCast family model wrappers."""

from __future__ import annotations

from .training.core.model import (
    build_loss_transform,
    build_predictor,
    build_prediction_transform,
    gc,
    load_graphcast_checkpoint,
    load_stats,
    scalarize_loss,
    validate_stats_coverage,
)

FAMILY_NAME = "graphcast"

__all__ = [
    "FAMILY_NAME",
    "build_loss_transform",
    "build_predictor",
    "build_prediction_transform",
    "gc",
    "load_graphcast_checkpoint",
    "load_stats",
    "scalarize_loss",
    "validate_stats_coverage",
]
