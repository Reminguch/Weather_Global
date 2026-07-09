"""MZ-residual with Grid → Mesh → Mamba → Mesh → Grid pathway.

The per-grid-point Mamba of ``mz_residual_mamba.py`` has no spatial
communication — each grid point is an independent temporal sequence. This
variant projects the grid onto an icosahedral mesh first, runs Mamba on the
(much smaller) set of mesh nodes, then projects back to the grid. The mesh
projection gives cross-grid "talk" while simultaneously shrinking Mamba's
parallel dimension, freeing budget for a much larger hidden size.
"""

from .mesh_ops import build_grid_mesh_projections  # noqa: F401
from .mz_meshed_mamba import (  # noqa: F401
    MZResidualMeshedConfig,
    MZResidualMeshedMamba,
)
