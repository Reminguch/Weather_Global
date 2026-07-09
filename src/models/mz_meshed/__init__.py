"""Backward-compatibility shim.

The real implementation has moved to ``src/models/mz/meshed_mamba/``. Existing
imports like ``from src.models.mz_meshed import MZResidualMeshedMamba`` keep
working via this shim.
"""

from ..mz.meshed_mamba import (  # noqa: F401
    MZResidualMeshedConfig,
    MZResidualMeshedMamba,
    build_grid_mesh_projections,
)
