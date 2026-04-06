# Residual Memory Architecture — Design Report (2026-04-04)

## Problem

Previous stateful Mamba experiments (Phase 2) showed no improvement over baseline despite the SSM hidden state being correctly threaded across autoregressive rollout steps. After analysis, we identified that **the encoding path difference was confounding the comparison**.

When `temporal_backbone=mamba`, the model uses a different encoding path from baseline:

| | Baseline | Mamba (old) |
|---|---|---|
| Feature extraction | `_inputs_to_grid_node_features()` — concatenate all timesteps along channel dim | `_inputs_to_grid_node_features_by_time()` — encode each timestep separately |
| Grid2Mesh | Single pass on concatenated features | One pass per timestep |
| Mesh latent shape | `[mesh, batch, D]` | `[time, mesh, batch, D]` |
| Temporal processing | None (already fused in channel concat) | Mamba over time axis, then take last step |

This means old Mamba experiments differed from baseline in **two ways simultaneously**:
1. The Mamba temporal module itself
2. The encoding path (per-timestep vs concatenated)

We cannot attribute any performance difference (or lack thereof) to Mamba alone.

Additionally, the per-timestep encoding path may be inherently weaker: the baseline's channel concatenation allows the Grid2Mesh GNN to jointly process all timesteps in a single MLP pass, learning cross-timestep features (velocity, acceleration) directly. The per-timestep path forces the GNN to encode each frame in isolation, then relies on Mamba to recover temporal relationships — a harder optimization target.

## Solution: Residual Memory Architecture

New temporal location: `mesh_post_encoder_residual`

**Key design principle**: Use the exact same encoding path as baseline, then inject cross-rollout temporal memory as a residual.

### Data Flow

```
Input: [batch, time=2, lat, lon, variables]
         │
         ▼
  Concatenate timesteps along channel dim     ← SAME AS BASELINE
  → [grid, batch, C*2]
         │
         ▼
  Grid2Mesh GNN (single pass)                 ← SAME AS BASELINE
  → [mesh, batch, D]
         │
         ▼
  ★ Stateful Mamba Block (NEW)
  │
  │  Unsqueeze: [mesh, batch, D] → [1, mesh, batch, D]
  │  Reshape:   → [batch*mesh, 1, D]
  │
  │  Load h_prev from hk.get_state            (from previous rollout step)
  │  shape: [mesh, H]
  │  Tile: → [batch*mesh, H]
  │
  │  SSM step (single timestep):
  │    decay = exp(A * softplus(Linear(u)))
  │    h_new = decay * h_prev + (1 - decay) * u
  │    y = h_new ⊙ sigmoid(gate) + skip ⊙ u
  │
  │  Save h_new via hk.set_state              (for next rollout step)
  │  mean over batch → [mesh, H]
  │
  │  Output projection + residual:
  │    output = input + Linear(y)              ← RESIDUAL CONNECTION
  │
  │  Squeeze: → [mesh, batch, D]
  │
         ▼
  Mesh GNN Processor                           ← SAME AS BASELINE
  → [mesh, batch, D]
         │
         ▼
  Mesh2Grid GNN                                ← SAME AS BASELINE
  → prediction
```

### Why This Should Work Better

1. **Perfect isolation**: The ONLY difference from baseline is the Mamba residual block. If there's improvement, it's 100% from temporal memory. If there's no improvement, we conclusively know Mamba doesn't help.

2. **Cannot be worse than baseline**: The Mamba block uses a residual connection (`output = input + proj(mamba_output)`). In the worst case, it learns zero weights and degrades to an identity function = baseline.

3. **Right job for Mamba**: Instead of redundantly processing 2-4 input frames (which the channel concatenation already handles), Mamba focuses on its actual strength: accumulating information across rollout steps via the hidden state.

4. **Faster training**: No per-timestep encoding overhead. Each forward pass is baseline speed + one small Mamba block.

### Cross-Rollout State Flow (target_steps=4 example)

```
Sample start → _reset_ssm_state() → h = 0

Rollout step 1:
  input = [t-1, t] (concatenated)
  Grid2Mesh → mesh_latent [mesh, batch, D]
  Mamba: load h=0, process 1 frame, save h₁
  output = mesh_latent + mamba_residual₁
  → predict t+1

Rollout step 2:
  input = [t, t+1]
  Grid2Mesh → mesh_latent [mesh, batch, D]
  Mamba: load h₁, process 1 frame, save h₂
  output = mesh_latent + mamba_residual₂    ← h₂ carries info from step 1
  → predict t+2

Rollout step 3:
  input = [t+1, t+2]
  Mamba: load h₂ → save h₃                  ← h₃ carries info from steps 1-2
  → predict t+3

Rollout step 4:
  input = [t+2, t+3]
  Mamba: load h₃ → save h₄                  ← h₄ encodes full 24h trajectory
  → predict t+4

Next sample → h = 0
```

## Code Changes

### `graphcast.py` (`__call__` method)

```python
# NEW: residual memory uses baseline encoding + Mamba state injection
if self._temporal_backbone == "none" or self._temporal_location == "mesh_post_encoder_residual":
    # Same encoding as baseline
    grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
    (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(grid_node_features)

    # Inject cross-rollout memory via stateful Mamba
    if self._temporal_location == "mesh_post_encoder_residual":
        latent_mesh_nodes = self._run_temporal_mesh_block(
            latent_mesh_nodes, is_training=is_training)
else:
    # Old path: per-timestep encoding (kept for backwards compatibility)
    ...
```

### `temporal_mesh_mamba_stateful.py` (`TemporalMeshBlock.__call__`)

```python
def __call__(self, mesh_latent_tnbd, *, is_training):
    if mesh_latent_tnbd.ndim == 3:
        # [n_mesh, batch, D] → unsqueeze time=1, run Mamba (1 step + state), squeeze back.
        # Used by mesh_post_encoder_residual: same encoding as baseline,
        # Mamba only injects cross-rollout memory via hidden state.
        mesh_4d = mesh_latent_tnbd[None]  # [1, n_mesh, batch, D]
        out_4d = self._run_sequence(mesh_4d, is_training=is_training)
        return out_4d[0]  # [n_mesh, batch, D]
    ...
```

## Experiments Submitted

### Smoke Tests (500 steps each, job 6527483)

| Array ID | Name | Config | Purpose |
|----------|------|--------|---------|
| 0 | resmem_t2 | residual memory, target_steps=2 | Short rollout baseline comparison |
| 1 | resmem_t4 | residual memory, target_steps=4 | Medium rollout |
| 2 | resmem_t6 | residual memory, target_steps=6 | Longer rollout |
| 3 | baseline_t2 | no Mamba, target_steps=2 | Control group |

Common parameters: mesh_size=4, width=128, batch_size=1, input_steps=2, lr=1e-4, bf16

### What to Look For

- `resmem_t2` vs `baseline_t2`: Mamba effect with same rollout length. Expect similar or slightly better (Mamba has 1 rollout step of memory).
- `resmem_t4` and `resmem_t6`: Whether longer rollout unlocks more Mamba benefit. If residual memory works, performance should improve more relative to baseline as target_steps increases.
- Training speed: resmem should be only slightly slower than baseline (one extra Mamba block per step).

## Comparison with Previous Approach

| | Old (`mesh_post_encoder`) | New (`mesh_post_encoder_residual`) |
|---|---|---|
| Encoding path | Per-timestep (different from baseline) | Same as baseline |
| Mamba input | Multi-frame sequence [T, mesh, batch, D] | Single frame [mesh, batch, D] + hidden state |
| What Mamba does | Redundant temporal processing on 2-4 frames | Pure cross-rollout memory injection |
| Fair comparison | No (encoding path confounded) | Yes (only difference is Mamba block) |
| Speed | Slower (per-timestep GNN) | Fast (same as baseline + small overhead) |
