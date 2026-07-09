"""Backward-compatibility shim.

The real implementation has moved to ``src/models/mz/grid_mamba/mz_grid_mamba.py``.
This file re-exports the public symbols so that existing imports like
``from src.models.mz_residual_mamba import MZResidualMamba, ...`` continue to
work. New code should import from ``src.models.mz`` or the subpackages
directly.
"""

from .mz.grid_mamba.mz_grid_mamba import (  # noqa: F401
    MZResidualConfig,
    MZResidualMamba,
    _SelectiveSSMBlock,
    shift_residual_history,
)
