"""MZ-residual memory models (unified namespace).

Subpackages:
  * ``grid_mamba``   — per-grid-point simplified SSM (no spatial mixing).
  * ``meshed_mamba`` — Grid→Mesh→simplified SSM→Mesh→Grid (shared simplified
                        block; adds icosphere-mesh spatial communication).
  * ``full_mamba``   — Grid→Mesh→S6-style SSM (d_state>1 + input-dep B,C)→Grid.
  * ``legacy``       — frozen v1-teacher module kept for loading old pickles.

See ``README.md`` in this directory for the full map of variants, usage and
checkpoint compatibility.
"""

# Re-export the three active variants so callers can just do
#   from src.models.mz import MZResidualMamba, MZResidualMeshedMamba, ...
from .grid_mamba.mz_grid_mamba import (  # noqa: F401
    MZResidualConfig,
    MZResidualMamba,
    shift_residual_history,
)
from .meshed_mamba import (  # noqa: F401
    MZResidualMeshedConfig,
    MZResidualMeshedMamba,
    build_grid_mesh_projections,
)
from .full_mamba import (  # noqa: F401
    FullMambaBlock,
    MZResidualFullMambaConfig,
    MZResidualFullMambaMeshed,
)
