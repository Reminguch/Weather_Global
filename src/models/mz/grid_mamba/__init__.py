"""Per-grid-point simplified SSM MZ-residual (baseline variant).

Every grid point is an independent Mamba temporal sequence — no spatial
communication. This is the original MZ-residual implementation; the meshed
and full-mamba variants reuse ``_SelectiveSSMBlock`` from here.
"""

from .mz_grid_mamba import (  # noqa: F401
    MZResidualConfig,
    MZResidualMamba,
    _SelectiveSSMBlock,
    shift_residual_history,
)
