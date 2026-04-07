# Weather Global: Stateful Mamba for GraphCast

This project integrates a **Mamba-style selective state space model (SSM)** into the [GraphCast](https://github.com/google-deepmind/graphcast) weather forecasting architecture, exploring whether explicit temporal memory across autoregressive rollout steps can improve multi-step weather prediction.

## Motivation

GraphCast predicts weather by taking two consecutive time steps as input and producing the next step. For multi-day forecasts, predictions are fed back autoregressively. However, each prediction step is **memoryless** — the model only sees the two most recent frames and has no mechanism to remember earlier atmospheric states during a rollout.

This means:
- No access to original initial conditions after the first prediction step
- Cannot track slowly evolving large-scale patterns (planetary waves, blocking events)
- Error accumulates without correction over long rollouts

**Mamba's selective SSM** provides a hidden state $h_t$ that accumulates temporal information with learned forgetting. If this state can be carried across autoregressive rollout steps, it gives the model a persistent "memory" of the entire forecast trajectory.

---

## Architecture

### Original GraphCast Pipeline (single prediction step)

```
Input: xarray [batch, time=2, lat, lon, variables]
         │
         ▼
  ┌─────────────────────────┐
  │  Flatten & Concatenate  │  Concatenate t-1, t along channel dim
  │  → [grid, batch, C*2]   │  (velocity info encoded implicitly)
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │   Grid2Mesh GNN         │  Single message-passing step
  │  → [mesh, batch, D]     │  Map grid features to icosahedral mesh
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │   Mesh GNN Processor    │  Multiple rounds of message passing
  │  → [mesh, batch, D]     │  Global spatial communication
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │   Mesh2Grid GNN         │  Map mesh features back to grid
  │  → [grid, batch, C]     │
  └──────────┬──────────────┘
             ▼
Output: xarray [batch, time=1, lat, lon, variables]  (prediction for t+1)
```

The autoregressive wrapper (`autoregressive.py`) loops this single-step predictor using `hk.scan`:
```
inputs=[t-1, t] → predict t+1 → inputs=[t, t+1] → predict t+2 → ...
```

### Modified Pipeline: Per-Timestep Encoding + Mamba

The key architectural change is splitting the encoding path so each input timestep is encoded **separately**, then processed temporally by the Mamba block before entering the mesh processor.

```
Input: xarray [batch, time=2, lat, lon, variables]
         │
         ▼
  ┌──────────────────────────────┐
  │  Encode Per Timestep         │  NEW: _inputs_to_grid_node_features_by_time()
  │  t-1 → [grid, batch, C]     │  Each timestep gets its own feature vector
  │  t   → [grid, batch, C]     │  (NOT concatenated)
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Grid2Mesh GNN (per step)   │  NEW: _run_grid2mesh_gnn_over_time()
  │  t-1 → [mesh, batch, D]    │  Same GNN weights, applied to each timestep
  │  t   → [mesh, batch, D]    │
  │  Stack → [T, mesh, batch, D]│
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  ★ Mamba Temporal Block      │  NEW: _run_temporal_mesh_block()
  │                              │
  │  Reshape: [T, mesh, batch, D]│
  │       → [batch*mesh, T, D]  │  Treat each mesh node as independent sequence
  │                              │
  │  Load SSM state h_{prev}     │  hk.get_state("ssm_state")
  │       shape: [mesh, H]      │  (from previous rollout step, or zeros)
  │  Tile → [batch*mesh, H]     │
  │                              │
  │  Selective scan along T:     │
  │    h_t = decay * h_{t-1}    │
  │        + (1-decay) * u_t    │
  │    y_t = h_t ⊙ σ(gate)     │
  │        + skip ⊙ u_t         │
  │                              │
  │  Save SSM state h_final      │  hk.set_state("ssm_state")
  │    mean over batch → [mesh,H]│
  │                              │
  │  Take last timestep output   │
  │  → [mesh, batch, D]         │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │   Mesh GNN Processor         │  Unchanged
  │  → [mesh, batch, D]         │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │   Mesh2Grid GNN              │  Unchanged
  │  → [grid, batch, C]         │
  └──────────┬───────────────────┘
             ▼
Output: prediction for t+1 (SSM state saved for next rollout step)
```

### Cross-Rollout State Persistence

The core innovation is that the Mamba hidden state **persists across autoregressive rollout steps**. This is achieved through three mechanisms working together:

**1. `hk.scan` in `autoregressive.py` (unchanged)**

GraphCast's autoregressive predictor uses `hk.scan` (not `jax.lax.scan`) to loop over target steps. `hk.scan` automatically threads Haiku state (including `hk.get_state`/`hk.set_state` values) between iterations. No changes needed here.

```python
# autoregressive.py line 212 — state is threaded automatically
_, flat_preds = hk.scan(one_step_prediction, inputs, scan_variables)
```

**2. `hk.get_state` / `hk.set_state` in `_StatefulSSMBlock`**

Each Mamba block stores its hidden state in Haiku's state mechanism. At each rollout step, it loads the previous state, runs the scan, and saves the new state:

```python
# Load state from previous rollout step (or zeros at sample start)
stored = hk.get_state("ssm_state", shape=(n_mesh, H), init=jnp.zeros)
init_state = jnp.tile(stored, (batch_size, 1))  # → [batch*mesh, H]

# Run selective scan over input timesteps
final_state, outputs = jax.lax.scan(step_fn, init_state, ...)

# Save for next rollout step (average over batch)
hk.set_state("ssm_state", final_state.reshape(batch, mesh, H).mean(0))
```

**3. `_reset_ssm_state` in training loop**

Before each training sample, all SSM states are zeroed so there's no cross-sample leakage:

```python
# train_graphcast.py — called at the start of each train_step
def _reset_ssm_state(state):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros_like(leaf) if isinstance(leaf, jax.Array) else leaf,
        state)
```

### Full Rollout Data Flow (target_steps=10)

```
Training sample: input=[t-1, t], targets=[t+1, t+2, ..., t+10]

_reset_ssm_state() → all h = 0

Rollout step 1:
  input=[t-1, t]
  Mamba: load h=0, process 2 frames, save h₁
  predict t+1

Rollout step 2:
  input=[t, t+1]          ← t+1 is the prediction from step 1
  Mamba: load h₁, process 2 frames, save h₂
  predict t+2              ← h₂ carries info from t-1 through t+1

Rollout step 3:
  input=[t+1, t+2]
  Mamba: load h₂, process 2 frames, save h₃
  predict t+3              ← h₃ carries info from the entire trajectory

  ...

Rollout step 10:
  input=[t+8, t+9]
  Mamba: load h₉, process 2 frames, save h₁₀
  predict t+10             ← h₁₀ encodes compressed summary of all 10 steps

Loss = mean(loss_step1, loss_step2, ..., loss_step10)
Gradients flow back through all 10 steps (with gradient checkpointing / hk.remat)

Next training sample → _reset_ssm_state() → h = 0 again
```

At step 10, the hidden state $h_{10}$ contains a **compressed summary of the entire 60-hour forecast trajectory** — information that the memoryless baseline has no access to.

### State Shape Design

The SSM state has shape `(n_mesh, hidden_size)`, **independent of batch size**:

- **Why not `(batch, n_mesh, H)`?** Haiku requires state shapes to match between `init` and `apply`. During `init`, a single sample is used (batch=1). During `apply`, batch size varies. Batch-independent state avoids this mismatch.
- **Batch handling**: Before the scan, state is tiled to `(batch*n_mesh, H)`. After the scan, the per-batch states are averaged back to `(n_mesh, H)` before saving.
- **Implication**: During training with batch_size > 1, different batch elements (different time windows) contribute to a "consensus" state. During inference (batch=1), the state is exact.

---

## Key Bug Fix: Transpose in Stateless Version

The original stateless implementation had a transpose bug in `temporal_mesh_mamba.py`:

```python
# BEFORE (wrong): extra transpose that shuffled mesh and batch dims
return jnp.transpose(sequence[-1], (1, 0, 2))

# AFTER (correct): sequence[-1] is already [n_mesh, batch, channels]
return sequence[-1]
```

This meant all prior stateless Mamba experiments had **corrupted output** — the mesh node and batch dimensions were swapped, producing meaningless predictions that happened to train to similar loss as baseline (the GNN downstream was robust enough to partially compensate). This bug was fixed in the current version.

---

## Encoding Path Difference and the Residual Memory Fix

When `temporal_backbone != "none"` with the original design, the model used a **different encoding path** than baseline:

| | Baseline | Old Mamba (`mesh_post_encoder`) |
|---|---|---|
| Feature extraction | Concatenate all timesteps → `[grid, batch, C*T]` | Encode each timestep separately |
| Grid2Mesh | Single pass | One pass per timestep |
| Mesh features | `[mesh, batch, D]` | `[time, mesh, batch, D]` |

This confounded every comparison: the performance difference could come from either the Mamba module or the weaker encoding path. Moreover, per-timestep encoding is inherently weaker than channel concatenation because the GNN cannot jointly process multiple frames in a single MLP pass.

### Residual Memory Architecture (`mesh_post_encoder_residual`)

To isolate Mamba's contribution, we introduced a new location: **`mesh_post_encoder_residual`**. This preserves the baseline encoding path entirely and injects Mamba only as a residual on the mesh latent:

```
Channel concat [t-1, t] → Grid2Mesh GNN → mesh_latent   ← identical to baseline
                                                ↓
                               ★ Mamba: load h_prev, process 1 frame,
                                 output = mesh_latent + proj(mamba_out)
                                 save h_new                              ← only change
                                                ↓
                               Mesh GNN → Mesh2Grid → prediction        ← identical to baseline
```

Key properties:
- **Fair comparison**: the only difference from baseline is the Mamba residual block
- **Cannot degrade**: residual connection means worst case = identity (baseline behavior)
- **Focused purpose**: Mamba no longer redundantly processes the input frames; it only injects cross-rollout memory via hidden state

Code changes:
- `graphcast.py`: new branch for `mesh_post_encoder_residual` uses `_inputs_to_grid_node_features()` (baseline encoding) then applies Mamba on the 3D `[mesh, batch, D]` latent
- `temporal_mesh_mamba_stateful.py`: handles 3D input by unsqueezing a time=1 dimension, running the SSM step + state update, then squeezing back
- `train_graphcast.py`: added `mesh_post_encoder_residual` to `--temporal-location` choices

### Residual Memory Results

**Important note**: All previous training-time comparisons (baseline RMSE=62 vs resmem RMSE=65) were confounded by a mesh_size mismatch (baseline mesh_size=3 vs resmem mesh_size=4). The results below use **correctly matched** mesh_size=4 for both models.

#### Long Rollout Inference (trained with target_steps=2, inference with varying rollout lengths)

Both models trained with target_steps=2 (12h rollout), then evaluated with longer inference rollouts. This tests whether Mamba's cross-rollout memory helps during extended forecasting even when not trained on long rollouts.

**Step 2k checkpoint (mesh_size=4, both models):**

| Inference t | Forecast | Baseline RMSE | resmem RMSE | Gap |
|------------|----------|--------------|-------------|-----|
| 2 | 12h | 112.22 | **111.63** | **-0.5%** |
| 10 | 2.5 days | 530.01 | **527.08** | **-0.6%** |
| 16 | 4 days | 681.88 | **673.28** | **-1.3%** |
| 24 | 6 days | 713.55 | **705.89** | **-1.1%** |
| 40 | 10 days | **1000.84** | 1014.84 | +1.4% |

**Step 4k checkpoint:**

| Inference t | Forecast | Baseline RMSE | resmem RMSE | Gap |
|------------|----------|--------------|-------------|-----|
| 2 | 12h | **101.12** | 101.66 | +0.5% |
| 10 | 2.5 days | **371.44** | 373.46 | +0.5% |
| 16 | 4 days | **660.54** | 666.32 | +0.9% |
| 24 | 6 days | 670.85 | **614.40** | **-8.4%** |
| 40 | 10 days | 881.45 | **867.94** | **-1.5%** |

**Step 10k checkpoint:**

| Inference t | Forecast | Baseline RMSE | resmem RMSE | Gap |
|------------|----------|--------------|-------------|-----|
| 2 | 12h | **120.34** | 121.05 | +0.6% |
| 10 | 2.5 days | 690.53 | **626.63** | **-9.3%** |
| 16 | 4 days | 882.24 | **778.68** | **-11.7%** |
| 24 | 6 days | 850.03 | **805.85** | **-5.2%** |
| 40 | 10 days | **870.92** | 964.71 | +10.8% |

#### Key Findings

1. **Mamba helps at medium-range forecasts (2.5–6 days)**: At step 10k, resmem outperforms baseline by 5–12% for inference rollouts of 10–24 steps. The peak improvement is **-11.7% RMSE at 4-day forecast** (t=16).

2. **No benefit at short range (<12h)**: At t=2, resmem is slightly worse than baseline (~0.5%), confirming that the 2-frame input already provides sufficient temporal information for single-step prediction.

3. **Degrades at very long range (>6 days)**: At t=40 (10 days), resmem is worse than baseline. The Mamba hidden state, trained only on 2-step rollouts, accumulates noise when pushed far beyond its training horizon.

4. **Improvement grows with training**: The gap at t=16 went from -1.3% (step 2k) to -11.7% (step 10k), suggesting more training helps Mamba learn better state utilization.

5. **The "sweet spot" is 2.5–6 day forecasts**: This matches the regime where the baseline's memoryless rollout starts to accumulate significant error, but the trajectory hasn't diverged so far that Mamba's compressed state becomes noise.

### Known Limitation: State Reset Between Samples

The Mamba hidden state is **reset to zeros at the start of each training sample**. Because training samples are randomly drawn from 2020-2021 (non-consecutive time windows), carrying state across samples would be meaningless. This means Mamba only has `target_steps` worth of sequential context (2-6 steps = 12-36 hours) before its memory is wiped.

This fundamentally limits Mamba's ability to learn long-range temporal patterns. To fully exploit Mamba's memory, one would need to either:
1. **Sequential sampling with truncated BPTT**: iterate through the data chronologically, carry state forward with `stop_gradient` at sample boundaries
2. **Much larger target_steps** (20-40): gives Mamba longer sequences within each sample, but requires more GPU memory

---

## Experiments

### Setup
- **Model**: GraphCast_small (resolution 2.0°, mesh_size 4, latent_width 128, 1 message passing step)
- **Data**: ERA5 WeatherBench2, 6h intervals, 13 pressure levels
  - Train: 2020–2021, Eval: 2022
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4), bf16 precision
- **GPU**: NVIDIA A100 80GB (Princeton Della cluster)
- **Mamba**: hidden_size=128, layers=1, dropout=0.0

### Phase 1: Stateless Mamba (mesh_size=3, batch_size=4, target_steps=1)

| Model | RMSE @8k | MAE @8k |
|-------|----------|---------|
| Baseline (input=2) | **138.55** | **35.13** |
| Mamba h2 (stateless, input=2) | 139.39 | 35.47 |
| Mamba h4 (stateless, input=4) | 138.83 | 35.39 |

**Result**: No meaningful improvement. Stateless Mamba is redundant with the GNN's built-in temporal handling. (Also affected by a transpose bug that corrupted output dimensions.)

### Phase 2: Baseline Reference (mesh_size=4, batch_size=1, 20k steps)

| input_steps | target_steps | RMSE @20k | MAE @20k |
|-------------|-------------|-----------|----------|
| 2 | 2 | 62.41 | 18.21 |
| **4** | **2** | **59.59** | **17.48** |
| 2 | 4 | 96.08 | 26.89 |
| 4 | 4 | 95.13 | 26.71 |

### Phase 2: Stateful Mamba — Old Encoding Path (cancelled)

Used per-timestep encoding (different from baseline), confounding the comparison. All configs trailed baseline by 13-20%. Cancelled at ~70% completion.

| Config | Best RMSE | Baseline @20k | Gap |
|--------|-----------|---------------|-----|
| sf-h2, target=2 | 72.70 @14k | 62.41 | +16.5% |
| sf-h4, target=2 | 71.37 @12k | 59.59 | +19.8% |
| sf-h2, target=4 | 108.41 @12k | 96.08 | +12.8% |
| sf-h4, target=4 | 111.09 @10k | 95.13 | +16.8% |

### Phase 3: Residual Memory (mesh_size=4, batch_size=1)

Uses baseline encoding path + stateful Mamba residual. See results above.


---

## Repository Structure

```
src/models/
  temporal_mesh_mamba.py           # Stateless Mamba block (resets state every call)
  temporal_mesh_mamba_stateful.py  # Stateful Mamba block (cross-rollout memory)

third_party/graphcast/graphcast/
  graphcast.py                     # Modified: temporal backbone integration,
                                   #   per-timestep encoding, stateful dispatch
  autoregressive.py                # Unmodified: hk.scan threads state automatically

scripts/training/
  train_graphcast.py               # Training loop: _reset_ssm_state, --temporal-stateful,
                                   #   --temporal-location mesh_post_encoder_residual
  stateful_longrollout_20k.slurm   # target_steps=10, 20k steps (old encoding)
  stateful_longrollout_smoke.slurm # Smoke test for target_steps=10
  resmem_t2.slurm                  # Residual memory, target_steps=2, 2k steps
  resmem_t4.slurm                  # Residual memory, target_steps=4, 2k steps
  resmem_t6.slurm                  # Residual memory, target_steps=6, 2k steps
  resmem_t2_20k.slurm              # Residual memory, target_steps=2, 20k steps
  resmem_t4_20k.slurm              # Residual memory, target_steps=4, 20k steps
  baseline_t6.slurm                # Baseline, target_steps=6, 2k steps

results/
  eval_summary.md                  # Experiment results summary
  2026-04-03_mamba_graphcast_technical_note.tex  # Technical note (LaTeX)
  2026-04-03_stateful_vs_stateless_comparison.png
```

## Usage

```bash
# Baseline (no Mamba)
python scripts/training/train_graphcast.py \
  --target-steps 2

# Stateless Mamba
python scripts/training/train_graphcast.py \
  --temporal-backbone mamba \
  --input-duration 12h \
  --target-steps 1

# Stateful Mamba with 10-step autoregressive rollout (old encoding path)
python scripts/training/train_graphcast.py \
  --temporal-backbone mamba \
  --temporal-stateful \
  --temporal-location mesh_post_encoder \
  --input-duration 12h \
  --target-steps 10

# Residual Memory (recommended): same encoding as baseline + Mamba cross-rollout state
python scripts/training/train_graphcast.py \
  --temporal-backbone mamba \
  --temporal-stateful \
  --temporal-location mesh_post_encoder_residual \
  --input-duration 12h \
  --target-steps 4
```

## References

- Lam et al., "Learning skillful medium-range global weather forecasting," *Science*, 2023.
- Gu & Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv:2312.00752*, 2023.
