"""Canonical Mamba family package."""

FAMILY_NAME = "mamba"
TRAINING_MODULE = "src.models.mamba.training.segments_training"
INFERENCE_MODULE = "src.models.mamba.gc_mamba.inference.inference_engine"

__all__ = ["FAMILY_NAME", "TRAINING_MODULE", "INFERENCE_MODULE"]
