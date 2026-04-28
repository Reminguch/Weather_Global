"""Canonical residual Mamba model package."""

MODEL_NAME = "residual_mamba"
TRAINING_MODULE = "src.models.mamba.training.segments_training"
INFERENCE_MODULE = "src.models.mamba.residual_mamba.inference.inference_engine"

__all__ = ["MODEL_NAME", "TRAINING_MODULE", "INFERENCE_MODULE"]
