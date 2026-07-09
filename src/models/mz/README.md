# MZ-Residual Memory Models

Unified namespace for the **Mori–Zwanzig residual-memory** modules. Each
variant is a small Mamba-style temporal model that runs **on top of a frozen
GraphCast baseline**, predicting the one-step residual `r_t = truth_t −
G(u_t)` and adding it back to the baseline forecast:

```
corrected_t = G(u_t)  +  MZ_model([u_t, r_{t-1}, ...])
              └─ frozen GraphCast (DeepMind .npz)
              └─ trainable residual-memory model, everything in this folder
```

All three active variants share the same training pipeline, loss, data
pipeline, checkpoint format and `rollout_ar` semantics — they differ in
**how the temporal block is wired spatially** and **how the SSM is
parameterised**. This README describes every file under
`src/models/mz/` in detail.

---

## Directory layout at a glance

```
src/models/mz/
├── README.md                               ← this file
├── __init__.py                             ← unified re-exports
│
├── grid_mamba/                             ── per-grid-point simplified SSM
│   ├── __init__.py
│   └── mz_grid_mamba.py                    ── MZResidualMamba + _SelectiveSSMBlock
│
├── meshed_mamba/                           ── Grid→Mesh→simplified SSM→Grid
│   ├── __init__.py
│   ├── mesh_ops.py                         ── icosphere + KNN projection utils
│   └── mz_meshed_mamba.py                  ── MZResidualMeshedMamba (reuses _SelectiveSSMBlock)
│
├── full_mamba/                             ── Grid→Mesh→S6-style SSM→Grid
│   ├── __init__.py
│   ├── full_mamba_block.py                 ── FullMambaBlock (d_state>1, input-dep B,C, SiLU)
│   └── mz_full_mamba_meshed.py             ── MZResidualFullMambaMeshed
│
└── legacy/                                 ── kept only for loading old checkpoints
    └── mz_v1_teacher.py                    ── pre-refactor MZResidualMamba (flat Haiku names)
```

---

## Per-file reference

### `src/models/mz/__init__.py`

Top-level unified entry point. Exports every public symbol of the three
active variants so that callers can do:

```python
from src.models.mz import (
    # grid_mamba
    MZResidualConfig, MZResidualMamba, shift_residual_history,
    # meshed_mamba
    MZResidualMeshedConfig, MZResidualMeshedMamba, build_grid_mesh_projections,
    # full_mamba
    MZResidualFullMambaConfig, MZResidualFullMambaMeshed, FullMambaBlock,
)
```

New code should prefer this import path (or the subpackage imports below)
over the legacy shims at `src/models/mz_residual_mamba.py` etc.

---

### `grid_mamba/`

Per-grid-point variant. Each `(lat, lon)` is an **independent** temporal
Mamba sequence — there is no spatial mixing. Also hosts the simplified
`_SelectiveSSMBlock` that `meshed_mamba/` reuses.

#### `grid_mamba/mz_grid_mamba.py`

**Classes / functions:**

| Symbol | What it is | How to use |
|---|---|---|
| `MZResidualConfig` | `@dataclass(frozen=True)` | Holds `input_size = 2 * F`, `output_size = F`, `hidden_size = H`, `layers`, `dropout`, `a_log_init`. Construct once, pass to `MZResidualMamba`. |
| `_SelectiveSSMBlock` | `hk.Module` | One simplified Mamba block. `__call__(x_btd, is_training)` for parallel teacher-forced scan; `.step(x_bd, h_prev_bh, is_training)` for one-step AR. Reused inside the meshed variant so both can share params. |
| `MZResidualMamba` | `hk.Module` | Wraps input_proj → `layers` × `_SelectiveSSMBlock` → residual_head. Exposes `__call__` (teacher-forced) and `rollout_ar` (AR with Bernoulli/deterministic TF mask + optional Option-2 state feedback). |
| `shift_residual_history` | helper | Build teacher-forced `prev_residual` by shifting truth residuals back one step (zero-pad first). Used when the network is called with `input = concat(current_state, shift(residual))`. |

**How the data flows inside `MZResidualMamba.__call__`:**

```
seq [T, B, lat, lon, 2F]
    → reshape to [B·lat·lon, T, 2F]     # every grid point becomes a "batch"
    → input_proj: 2F → H                # LN + Linear
    → stacked _SelectiveSSMBlock(×layers)
    → residual_head: H → F
    → reshape back to [T, B, lat, lon, F]
```

**AR rollout** (`rollout_ar`) adds Option-2 state-feedback: at intra-sample
step `s > 0` the network's `current_state` input becomes
`baseline_raw_{s-1} + pred_residual_n_{s-1} × (output_denorm / input_std)`
rather than the unobservable true state.

Import (both paths work):
```python
from src.models.mz.grid_mamba import MZResidualConfig, MZResidualMamba
# or the unified top-level
from src.models.mz import MZResidualConfig, MZResidualMamba
```

---

### `meshed_mamba/`

Spatial variant. Projects the grid onto an icosahedral mesh via a fixed
KNN Gaussian kernel, runs Mamba on the (much smaller) mesh node sequences,
projects back to the grid. Uses **the same `_SelectiveSSMBlock`** as
`grid_mamba/`, so the only difference from `grid_mamba` is the spatial
pre/post-processing.

#### `meshed_mamba/mesh_ops.py`

Pure-numpy geometric utilities, run **once at model construction time**:

| Symbol | Inputs | Returns | Purpose |
|---|---|---|---|
| `build_grid_mesh_projections(lat_deg, lon_deg, mesh_size, n_grid_neighbors=6, n_mesh_neighbors=3, sigma_scale=1.0)` | lat/lon grid vectors + icosphere level | `(arrays_dict, n_mesh_nodes)` — 4 fixed tensors: `g2m_indices[M, K_g2m] (int32)`, `g2m_weights[M, K_g2m] (f32)`, `m2g_indices[P, K_m2g] (int32)`, `m2g_weights[P, K_m2g] (f32)`, each row sums to 1. | Precompute the Grid→Mesh and Mesh→Grid KNN projection tensors. |

The projection is **fixed and non-trainable** — it's a geometric linear
operator, not a GNN. Keeping it fixed is what lets the Mamba on the mesh
side absorb all the extra capacity.

Internal helpers (not re-exported):
- `_latlon_to_xyz` — lat/lon degrees → unit 3D vectors.
- `_icosphere_nodes(mesh_size)` — uses `third_party/graphcast/icosahedral_mesh`
  to return mesh-node xyz on the unit sphere.

#### `meshed_mamba/mz_meshed_mamba.py`

| Symbol | What it is |
|---|---|
| `MZResidualMeshedConfig` | Same shape as `MZResidualConfig` + `use_mesh_pre_mlp`, `use_mesh_post_mlp` toggles. |
| `_grid_to_mesh_aggregate(x_b_p_h, g2m_idx, g2m_w)` | Gather `x` at `K` grid neighbours per mesh node, weighted sum. `(B, P_grid, H) → (B, M, H)`. |
| `_mesh_to_grid_aggregate(x_b_m_h, m2g_idx, m2g_w)` | Reverse direction. `(B, M, H) → (B, P_grid, H)`. |
| `MZResidualMeshedMamba` | Haiku module with `.__call__` (teacher) and `.rollout_ar` (AR with per-step TF mask + Option-2 feedback). Constructor takes the fixed projection arrays + `n_mesh_nodes`. |

**How the data flows inside `MZResidualMeshedMamba.__call__`:**

```
seq [T, B, lat, lon, 2F]
    → reshape to [T·B, P_grid, 2F]
    → input_proj (2F → H)                                 # per grid point
    → Grid→Mesh aggregate: (T·B, P_grid, H) → (T·B, M, H) # fixed KNN weighted sum
    → SiLU(mesh_pre)                                      # optional H→H MLP
    → reshape + transpose to [B·M, T, H]                  # per-mesh sequence
    → stacked _SelectiveSSMBlock(×layers)                 # Mamba over time
    → reshape back to [T·B, M, H]
    → SiLU(mesh_post)                                     # optional H→H MLP
    → Mesh→Grid aggregate: (T·B, M, H) → (T·B, P_grid, H)
    → residual_head (H → F)
```

**Why meshed is better at the same H:** the Mamba's parallel dimension
drops from O(10k–100k) grid points to O(100–10k) mesh nodes, so the same
hidden size costs much less memory and also forces grid points to "talk"
through the mesh bottleneck (picks up large-scale coherent residuals).

Import:
```python
from src.models.mz.meshed_mamba import (
    MZResidualMeshedConfig, MZResidualMeshedMamba,
    build_grid_mesh_projections,
)
```

---

### `full_mamba/`

Spatial pathway identical to `meshed_mamba/`, but the temporal block is a
closer-to-original S6-style Mamba with `d_state > 1`, input-dependent
`B(u)` / `C(u)` / `dt(u)`, SiLU gating, and an expansion factor. This is
the variant for when you suspect the simplified block is the bottleneck.

#### `full_mamba/full_mamba_block.py`

Defines `FullMambaBlock`. Drop-in replacement for `_SelectiveSSMBlock` at
the **temporal-block** level (same `__call__` and `.step` signatures) but
with a richer SSM underneath.

**Constructor:**

```python
FullMambaBlock(
    hidden_size,        # H (= d_model)
    d_state=16,         # N — per-inner-channel SSM state vector length
    expand=2,           # D_inner = H * expand (SSM operates on D_inner, not H)
    dropout=0.0,
    a_log_init_min=-3.0,   # Uniform(-3, -0.1) init for A_log → multi-scale
    a_log_init_max=-0.1,   # memory time constants (half-life ~1–14 steps).
)
```

**Trainable parameters per block:**

| Param | Shape | Role |
|---|---|---|
| `layer_norm/{scale,offset}` | `(H,)`, `(H,)` | Pre-norm |
| `in_proj/{w,b}` | `(H, 2·D_inner)`, `(2·D_inner,)` | Projects H → `[x_path, z_path]` |
| `x_proj/{w,b}` | `(D_inner, D_inner + 2N)`, `(D_inner + 2N,)` | Input-dependent `[dt_input, B, C]` |
| `dt_proj/{w,b}` | `(D_inner, D_inner)`, `(D_inner,)` | Produces raw `dt` (softplus'd) |
| `A_log` | `(D_inner, N)` | Per-(channel, state-dim) decay rate (diagonal A) |
| `D` | `(D_inner,)` | "Skip" (direct-pass) multiplier |
| `out_proj/{w,b}` | `(D_inner, H)`, `(H,)` | Projects back D_inner → H |

**Key diff vs `_SelectiveSSMBlock`:**
- `A` is `(D_inner, d_state)` diagonal, not a `(H,)` scalar vector. This
  gives every inner channel **multiple parallel memory time constants**
  (one per `d_state` slot). Fixes the "one decay per channel" bottleneck
  that made `a_log_init` sweeps in the simplified block essentially flat.
- `B` and `C` are **input-dependent** (computed by `x_proj` from the
  current time step's `x_path`), implementing true Mamba selectivity.
- Output uses **SiLU gating** (`y * silu(z_path)`) and an expanded inner
  dimension, matching Gu & Dao 2023 more closely.

**Forward shape:**
```
x [B·P, T, H]                     # P = mesh nodes (or grid points) in caller
  → LN → in_proj → [x_path, z_path]  both [B·P, T, D_inner]
  → SiLU(x_path)
  → x_proj: D_inner → D_inner + 2N   → split into dt_input, B, C
  → dt = softplus(dt_proj(dt_input))  [B·P, T, D_inner]
  → lax.scan over T:
       state ∈ [B·P, D_inner, N]
       decay = exp(dt[:, :, None] * A[None, :, :])   # A = -exp(A_log)
       input_term = dt[:, :, None] * B[:, None, :] * u[:, :, None]
       new_state = decay * state + input_term
       y_t = Σ_N new_state * C[:, None, :] → [B·P, T, D_inner]
  → y += D · x_path                                  # skip path
  → y *= SiLU(z_path)
  → out_proj: D_inner → H
  → residual connection: return x + y
```

`FullMambaBlock.step(x_bd, h_prev_bin, is_training)` is the one-step
autoregressive counterpart used by `rollout_ar`. Same params, same math,
just one `t`-slice at a time instead of `lax.scan`.

#### `full_mamba/mz_full_mamba_meshed.py`

| Symbol | What it is |
|---|---|
| `MZResidualFullMambaConfig` | Frozen dataclass: `input_size`, `output_size`, `hidden_size`, `d_state`, `expand`, `layers`, `dropout`, `a_log_init_min`, `a_log_init_max`, `use_mesh_pre_mlp`, `use_mesh_post_mlp`. |
| `_grid_to_mesh_aggregate` / `_mesh_to_grid_aggregate` | Same as in `meshed_mamba/mz_meshed_mamba.py` (pure gather + weighted sum). Kept here as small local helpers. |
| `MZResidualFullMambaMeshed` | Haiku module with `.__call__` (teacher-forced parallel) and `.rollout_ar` (AR with TF mask, Option-2 feedback, optional residual clipping). |

Essentially the exact same outer pipeline as `MZResidualMeshedMamba`, but
constructs `FullMambaBlock` internally instead of `_SelectiveSSMBlock`. So
swapping variants is a one-line change at construction time.

Import:
```python
from src.models.mz.full_mamba import (
    MZResidualFullMambaConfig, MZResidualFullMambaMeshed, FullMambaBlock,
)
```

---

### `legacy/`

#### `legacy/mz_v1_teacher.py`

The pre-refactor `MZResidualConfig` / `MZResidualMamba` module, kept only
because **old `mz_residual_stepN.pkl` checkpoints used flat Haiku parameter
names** (e.g. `input_proj/w` without the `~/` prefix that the post-refactor
modules produce). Load those old pickles by passing `--legacy-v1` to
`scripts/training/infer_mz_save_tensors.py`; it will import this module
instead of the current `grid_mamba` one.

Do **not** use this module for new training. It exists purely for
backward compatibility with a handful of old inference runs.

Import (rarely needed):
```python
from src.models.mz.legacy.mz_v1_teacher import MZResidualConfig, MZResidualMamba
```

---

## Choosing a variant

| Goal | Variant to use | Rationale |
|---|---|---|
| Smallest, fastest smoke / sanity | `grid_mamba` with `hidden_size=16` | ~1.5k params, works on any GPU |
| A/B "does mesh help?" at matched H | `meshed_mamba` vs `grid_mamba`, same `hidden_size` | Shares the `_SelectiveSSMBlock`, so only spatial operator differs |
| Push past the simplified block's plateau | `full_mamba` with `d_state=16, expand=2` | Multi-scale decay + input-dep `B, C` + SiLU, closer to Gu & Dao 2023 |
| Load an ancient `.pkl` before the 2026-04 refactor | `legacy.mz_v1_teacher` | Flat param names, no spatial mixing, teacher-forced only |
| Run new experiments | `full_mamba` or `meshed_mamba`, **never** `legacy` | Latest API, resume-friendly |

All three active variants share the same **outer API** (`__call__`,
`rollout_ar`) and checkpoint format, so training scripts swap them purely
by changing which `Config` + constructor is used.

---

## How training scripts pick a variant

The training scripts under `scripts/training/` already handle the
dispatch. Relevant CLI flags:

| Flag | Effect |
|---|---|
| default (no flag) | `grid_mamba.MZResidualMamba` (per-grid-point simplified SSM) |
| `--meshed --mz-mesh-size N` | `meshed_mamba.MZResidualMeshedMamba` (still simplified SSM) |
| `--meshed --mz-mesh-size N --full-mamba --d-state N --expand N` | `full_mamba.MZResidualFullMambaMeshed` (S6-style SSM) |
| `--full-variables` | Train on all 11 GraphCast target variables (F=83) instead of the 4-var default (MSLP/Z/U/V, F=40). Orthogonal to the variant above. |
| `--target-steps K --train-mode target_rollout` | K-step intra-sample AR rollout during training. |
| `--resume-from <path> --resume-step N` | Load an existing `mz_residual_stepN.pkl`; training continues at step N+1. Critical for curriculum fine-tuning (`K=1 → K=2 → K=4 → …`). |

Scripts that route through these flags:

- `scripts/training/train_mz_residual_memory.py` — original 4-variable
  entry point, supports `--meshed`.
- `scripts/training/train_mz_residual_memory_resume.py` — same + resume.
- `scripts/training/full_mz/train_full_mz.py` — adds the 11-variable
  `RESOLVED_VARIABLES_FULL` set.
- `scripts/training/full_mamba/train_mz_fullmamba.py` — adds
  `--full-mamba`, `--d-state`, `--expand`, `--a-log-init-min/max`,
  `--full-variables` (orthogonal) and resume support, wiring the
  `MZResidualFullMambaMeshed` branch of the model factory.

---

## Example: building each variant directly

```python
import jax, jax.numpy as jnp, haiku as hk
import numpy as np
from src.models.mz import (
    MZResidualConfig, MZResidualMamba,
    MZResidualMeshedConfig, MZResidualMeshedMamba,
    MZResidualFullMambaConfig, MZResidualFullMambaMeshed,
    build_grid_mesh_projections,
)

F = 83  # all 11 GraphCast targets on 13 pressure levels
lat, lon = np.linspace(-90, 90, 181), np.linspace(0, 359, 360)

# 1) grid_mamba
def grid_fn(seq, training):
    cfg = MZResidualConfig(input_size=2*F, output_size=F, hidden_size=16, layers=1)
    return MZResidualMamba(cfg)(seq, is_training=training)

# 2) meshed_mamba
proj, M = build_grid_mesh_projections(lat_deg=lat, lon_deg=lon, mesh_size=5)
def meshed_fn(seq, training):
    cfg = MZResidualMeshedConfig(input_size=2*F, output_size=F, hidden_size=128, layers=1)
    return MZResidualMeshedMamba(cfg, n_mesh_nodes=M, **proj)(seq, is_training=training)

# 3) full_mamba
def full_fn(seq, training):
    cfg = MZResidualFullMambaConfig(
        input_size=2*F, output_size=F, hidden_size=128,
        d_state=16, expand=2, layers=1,
    )
    return MZResidualFullMambaMeshed(cfg, n_mesh_nodes=M, **proj)(seq, is_training=training)

# Transform → init → apply (Haiku pure-functional pattern)
grid_tf = hk.transform(grid_fn)
rng = jax.random.PRNGKey(0)
dummy = jnp.zeros((16, 1, len(lat), len(lon), 2*F))
params = grid_tf.init(rng, dummy, False)
out = grid_tf.apply(params, rng, dummy, False)   # [16, 1, lat, lon, F]
```

---

## Backward-compatibility shims

For any existing code that still imports from the old paths, thin shims
live at:

- `src/models/mz_residual_mamba.py` → re-exports
  `grid_mamba.mz_grid_mamba` symbols.
- `src/models/mz_residual_mamba_v1_teacher.py` → re-exports
  `legacy.mz_v1_teacher`.
- `src/models/mz_meshed/__init__.py` → re-exports
  `meshed_mamba`.
- `src/models/full_mamba/__init__.py` → re-exports
  `full_mamba`.

So `from src.models.mz_residual_mamba import MZResidualMamba` and
`from src.models.full_mamba import FullMambaBlock` keep working without
changes. New code should prefer `from src.models.mz import ...` or the
subpackage imports (`from src.models.mz.full_mamba import ...`).

---

## Checkpoint compatibility

| Checkpoint produced by | Module to use for loading | Haiku name pattern |
|---|---|---|
| Any `grid_mamba` / `meshed_mamba` / `full_mamba` run (post-2026-04) | The matching variant in `src/models/mz/` | `mz_residual_mamba/~/input_proj/w`, etc. |
| Early pre-refactor runs (e.g. `mz_r4_m3_i32_seg32_h16_fullnorm`) | `legacy.mz_v1_teacher` (via `--legacy-v1`) | Flat, `input_proj/w` |

Haiku requires exactly-matching parameter names to load a pickle, so the
legacy shim is the only way to use those early checkpoints for inference.

---

## Extending

If you want to add a new variant (e.g. `FullMamba + Conv1d` preprocessing,
or a multi-layer hierarchical mesh), the pattern is:

1. Create a new subfolder under `src/models/mz/your_variant/`.
2. Define `YourConfig` + `YourModule` with the same public API as the
   existing variants (`__call__` teacher-forced, `rollout_ar` AR).
3. Reuse the geometric helpers from `meshed_mamba/mesh_ops.py` and the SSM
   blocks from `grid_mamba/` or `full_mamba/` where possible.
4. Add an entry to `src/models/mz/__init__.py` re-exporting the new
   symbols.
5. Add a CLI branch to the appropriate training script (typically
   `train_mz_fullmamba.py` since it already dispatches on multiple flags).
6. Commit, push, run an A/B against the closest existing variant.

Keep the per-variant `__init__.py` minimal (one-line re-exports) and the
module files self-contained.
