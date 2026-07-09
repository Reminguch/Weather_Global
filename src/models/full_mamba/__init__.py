"""Backward-compatibility shim.

The real implementation has moved to ``src/models/mz/full_mamba/``. Existing
imports like ``from src.models.full_mamba import MZResidualFullMambaMeshed``
keep working via this shim.
"""

from ..mz.full_mamba import (  # noqa: F401
    FullMambaBlock,
    MZResidualFullMambaConfig,
    MZResidualFullMambaMeshed,
)
