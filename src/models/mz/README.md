# MZ-Residual Memory Models

Unified namespace for the Mori–Zwanzig residual-memory modules used on top of
a frozen GraphCast baseline. Three active variants plus one legacy module live
under `src/models/mz/`:

```
src/models/mz/
├── README.md               ← this file
├── __init__.py             ← re-exports the three active variants
│
├── grid_mamba/             ── per-grid-point simplified SSM (no spatial mixing)
│   ├── __init__.py
│   └── mz_grid_mamba.py    ── MZResidualMamba + _SelectiveSSMBlock
│
├── meshed_mamba/           ── Grid → Mesh → simplified SSM → Mesh → Grid
│   ├── __init__.py
│   ├── mesh_ops.py         ── icosphere + KNN geometric projection
│   └── mz_meshed_mamba.py  ── MZResidualMeshedMamba (reuses _SelectiveSSMBlock)
│
├── full_mamba/             ── Grid → Mesh → S6-style SSM → Mesh → Grid
│   ├── __init__.py
│   ├── full_mamba_block.py ── FullMambaBlock (d_state>1, input-dep B,C, SiLU)
│   └── mz_full_mamba_meshed.py ── MZResidualFullMambaMeshed
│
└── legacy/                 ── kept only for loading old checkpoints
    └── mz_v1_teacher.py    ── MZResidualConfig / MZResidualMamba (pre-refactor,
                                flat Haiku param names)
```

---

## The three active variants at a glance

All three inherit the same MZ-residual formalism
(`u_{t+1} = G(u_t) + r_t(u_{≤t})` with frozen `G` = GraphCast baseline) and
the same training / eval pipeline (teacher / `target_rollout` / AR modes,
Option-2 state feedback, ΔMAE on 11 GraphCast target variables).

They differ in **how the Mamba block is wired spatially** and **how the SSM
itself is parameterised**:

| Variant | Spatial pathway | SSM block | d_state | Input-dep B, C | Params (typical, h=128) |
|---|---|---|---|---|---|
| **grid_mamba** | per-grid-point | `_SelectiveSSMBlock` (simplified) | 1 | no | ~30k |
| **meshed_mamba** | Grid→Mesh→SSM→Grid | `_SelectiveSSMBlock` (same as grid) | 1 | no | ~115k |
| **full_mamba** | Grid→Mesh→SSM→Grid | `FullMambaBlock` (S6-style) | 16 | **yes** | ~650k |

### grid_mamba (baseline variant)

- Each grid point is an independent Mamba temporal sequence.
- No spatial communication — each `(lat, lon)` learns its own per-channel
  residual dynamics.
- Cheapest and simplest. Used for the original r=4 experiments
  (`mz_r4_m3_i32_seg32_h16_*`).

Import:
```python
from src.models.mz.grid_mamba import MZResidualConfig, MZResidualMamba
```

### meshed_mamba

- Adds Grid → Mesh aggregation (fixed KNN with Gaussian weights, non-trained)
  before Mamba, and Mesh → Grid aggregation after.
- Mesh nodes are O(100)–O(10k) (mesh=2..5) vs grid points O(10k)–O(100k).
  So the per-Mamba parallel dimension shrinks dramatically; h can go bigger.
- The SSM block is still the simplified `_SelectiveSSMBlock` reused from
  `grid_mamba` (guarantees apples-to-apples comparison with grid variant).
- Grid points communicate through the mesh bottleneck → picks up large-scale
  coherent residual patterns (e.g. MSLP phase drift) that per-grid cannot.

Import:
```python
from src.models.mz.meshed_mamba import (
    MZResidualMeshedConfig, MZResidualMeshedMamba,
    build_grid_mesh_projections,
)
```

### full_mamba

- Same spatial pathway as `meshed_mamba`, but the temporal block is the
  closer-to-original S6 Mamba from Gu & Dao 2023:
  - `A_log` is a `(D_inner, d_state)` matrix, not a `(H,)` vector (diagonal
    per-channel per-state-dim with independent time constants).
  - `B(u)` and `C(u)` are **input-dependent** (computed by `x_proj`), not
    absorbed into fixed weights.
  - SiLU-gated output, expansion factor (typical `expand=2`).
- Addresses the known limitation of the simplified block that every
  per-channel Mamba has only one effective memory time constant; with
  `d_state=16` the model can mix 16 different decay rates per channel.
- Significantly more params (~5–6× of the meshed variant).

Import:
```python
from src.models.mz.full_mamba import (
    MZResidualFullMambaConfig, MZResidualFullMambaMeshed, FullMambaBlock,
)
```

---

## When to use which

| Goal | Recommended variant | Why |
|---|---|---|
| First sanity / cheapest sweep | `grid_mamba` (h=16) | Smallest, fastest to train |
| A/B test "mesh vs per-grid" | `meshed_mamba` vs `grid_mamba` at same h | Identical SSM block isolates the spatial effect |
| Push past plateau on strong baseline | `full_mamba` at d_state=16 | Multi-scale memory + input-dep selectivity |
| Loading old `mz_residual_stepN.pkl` ckpts with `~/`-free names | `legacy` | Haiku name shim for pre-refactor checkpoints |

All three variants expose the same public interface:
- `MZResidual*Config` (frozen dataclass)
- `MZResidual*` module with `__call__(seq, is_training)` (parallel teacher-forced)
  and `rollout_ar(current_state, ...)` (autoregressive / scheduled sampling).

So the training script can swap variants purely by changing the constructor.

---

## Shared scaffolding (reused across variants)

The simplified SSM block `_SelectiveSSMBlock` lives in
`grid_mamba/mz_grid_mamba.py` and is imported by `meshed_mamba` — these two
variants share the exact same SSM parameters and differ only in whether
there is a grid↔mesh projection wrapping the temporal scan.

The icosphere mesh + KNN projection utilities live in
`meshed_mamba/mesh_ops.py` (`build_grid_mesh_projections`) and are consumed
by both `meshed_mamba` and `full_mamba`.

---

## Checkpoint compatibility

- `grid_mamba` (current): Haiku names have `~/` prefixed segments
  (e.g. `mz_residual_mamba/~/input_proj/w`). Produced by every MZ run after
  the 2026-04 refactor.
- `legacy` (`mz_v1_teacher.py`): Haiku names are flat (e.g. `input_proj/w`).
  Used only to load checkpoints produced before the refactor (the early
  `mz_r4_m3_i32_seg32_h16_fullnorm` runs).
- Pass `--legacy-v1` to `infer_mz_save_tensors.py` to route through this
  path.

The meshed and full-mamba variants are both newer and use the v2 naming.

---

## Backward compatibility

For existing code that still imports from old paths
(`src.models.mz_residual_mamba`, `src.models.mz_meshed`,
`src.models.full_mamba`, `src.models.mz_residual_mamba_v1_teacher`), those
locations now contain thin shim modules that re-export everything from the
new `src/models/mz/` layout. Existing training / inference scripts keep
working unchanged; new code should import from `src.models.mz` or the
subpackages directly.
