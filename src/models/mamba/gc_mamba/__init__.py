"""Canonical GraphCast+Mamba model package."""

MODEL_NAME = "gc_mamba"
TRAINING_MODULE = "src.models.mamba.training.segments_training"
INFERENCE_MODULE = "src.models.mamba.gc_mamba.inference.inference_engine"

__all__ = ["MODEL_NAME", "TRAINING_MODULE", "INFERENCE_MODULE"]
