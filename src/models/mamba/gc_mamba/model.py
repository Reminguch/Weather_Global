"""GraphCast+Mamba model wrappers."""

from __future__ import annotations

from src.models.graphcast.training.core.model import (
    build_loss_transform,
    build_prediction_transform,
    build_predictor,
    gc,
    load_graphcast_checkpoint,
    load_stats,
    scalarize_loss,
    validate_stats_coverage,
)

MODEL_NAME = "gc_mamba"


def build_gc_mamba_predictor(*args, **kwargs):
    return build_predictor(*args, **kwargs)


def build_gc_mamba_loss_transform(*args, **kwargs):
    return build_loss_transform(*args, **kwargs)


__all__ = [
    "MODEL_NAME",
    "build_gc_mamba_loss_transform",
    "build_gc_mamba_predictor",
    "build_loss_transform",
    "build_prediction_transform",
    "build_predictor",
    "gc",
    "load_graphcast_checkpoint",
    "load_stats",
    "scalarize_loss",
    "validate_stats_coverage",
]
