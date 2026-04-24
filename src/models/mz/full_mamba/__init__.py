"""Full Mamba (S6-style) block with d_state > 1 and input-dependent B, C.

Closer to Gu & Dao 2023 than the simplified _SelectiveSSMBlock used in
mz_residual_mamba / mz_meshed. Tier 1 improvements:
  * Diagonal A with per-channel-per-state-dim parameters: shape (D_inner, N).
  * Input-dependent B and C matrices (not just dt): B(u), C(u).
  * SiLU-gated output (standard Mamba, instead of sigmoid gate).
  * Optional gate path (the z-branch) and expansion factor.

NOT yet included (punted to later rounds):
  * Conv1d preprocessing (kernel=4 causal).
  * Parallel associative scan (we still use jax.lax.scan).
  * Multi-layer stacking (use --layers CLI as before).
"""

from .full_mamba_block import FullMambaBlock  # noqa: F401
from .mz_full_mamba_meshed import (  # noqa: F401
    MZResidualFullMambaConfig,
    MZResidualFullMambaMeshed,
)
