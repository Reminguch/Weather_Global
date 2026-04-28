"""Model family registry."""

from __future__ import annotations

from .graphcast import FAMILY_NAME as GRAPHCAST_FAMILY
from .mamba import FAMILY_NAME as MAMBA_FAMILY
from .mamba.residual_mamba import MODEL_NAME as RESIDUAL_MAMBA_MODEL

MODEL_FAMILIES = (
    GRAPHCAST_FAMILY,
    MAMBA_FAMILY,
    RESIDUAL_MAMBA_MODEL,
)

__all__ = ["MODEL_FAMILIES", "GRAPHCAST_FAMILY", "MAMBA_FAMILY", "RESIDUAL_MAMBA_MODEL"]
