"""Canonical GraphCast family package."""

FAMILY_NAME = "graphcast"
TRAINING_MODULE = "src.models.graphcast.training.standard_training"
INFERENCE_MODULE = "src.models.graphcast.inference.inference_engine"

__all__ = ["FAMILY_NAME", "TRAINING_MODULE", "INFERENCE_MODULE"]
