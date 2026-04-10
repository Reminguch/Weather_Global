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

| Inference t | Forecast | Baseline RMSE | resmem RMSE | RMSE Gap | Baseline MAE | resmem MAE | MAE Gap |
|------------|----------|--------------|-------------|----------|-------------|------------|---------|
| 2 | 12h | 112.22 | **111.63** | **-0.5%** | 32.05 | **31.93** | **-0.4%** |
| 10 | 2.5 days | 530.01 | **527.08** | **-0.6%** | 130.46 | **129.64** | **-0.6%** |
| 16 | 4 days | 681.88 | **673.28** | **-1.3%** | 171.69 | **169.62** | **-1.2%** |
| 24 | 6 days | 713.55 | **705.89** | **-1.1%** | 183.93 | **182.04** | **-1.0%** |
| 40 | 10 days | **1000.84** | 1014.84 | +1.4% | **268.91** | 272.54 | +1.4% |

**Step 4k checkpoint:**

| Inference t | Forecast | Baseline RMSE | resmem RMSE | RMSE Gap | Baseline MAE | resmem MAE | MAE Gap |
|------------|----------|--------------|-------------|----------|-------------|------------|---------|
| 2 | 12h | **101.12** | 101.66 | +0.5% | **28.40** | 28.56 | +0.6% |
| 10 | 2.5 days | **371.44** | 373.46 | +0.5% | **93.90** | 94.37 | +0.5% |
| 16 | 4 days | **660.54** | 666.32 | +0.9% | **169.78** | 171.38 | +0.9% |
| 24 | 6 days | 670.85 | **614.40** | **-8.4%** | 175.53 | **162.07** | **-7.7%** |
| 40 | 10 days | 881.45 | **867.94** | **-1.5%** | 226.70 | **222.22** | **-2.0%** |

**Step 10k checkpoint:**

| Inference t | Forecast | Baseline RMSE | resmem RMSE | RMSE Gap | Baseline MAE | resmem MAE | MAE Gap |
|------------|----------|--------------|-------------|----------|-------------|------------|---------|
| 2 | 12h | **120.34** | 121.05 | +0.6% | **35.81** | 36.21 | +1.1% |
| 10 | 2.5 days | 690.53 | **626.63** | **-9.3%** | 185.56 | **167.32** | **-9.8%** |
| 16 | 4 days | 882.24 | **778.68** | **-11.7%** | 206.32 | **186.15** | **-9.8%** |
| 24 | 6 days | 850.03 | **805.85** | **-5.2%** | 234.72 | **215.42** | **-8.2%** |
| 40 | 10 days | **870.92** | 964.71 | +10.8% | **245.73** | 263.76 | +7.3% |

#### Key Findings

1. **Mamba helps at medium-range forecasts (2.5–6 days)**: At step 10k, resmem outperforms baseline by 5–12% for inference rollouts of 10–24 steps. The peak improvement is **-11.7% RMSE at 4-day forecast** (t=16).

2. **No benefit at short range (<12h)**: At t=2, resmem is slightly worse than baseline (~0.5%), confirming that the 2-frame input already provides sufficient temporal information for single-step prediction.

3. **Degrades at very long range (>6 days)**: At t=40 (10 days), resmem is worse than baseline. The Mamba hidden state, trained only on 2-step rollouts, accumulates noise when pushed far beyond its training horizon.

4. **Improvement grows with training**: The gap at t=16 went from -1.3% (step 2k) to -11.7% (step 10k), suggesting more training helps Mamba learn better state utilization.

5. **The "sweet spot" is 2.5–6 day forecasts**: This matches the regime where the baseline's memoryless rollout starts to accumulate significant error, but the trajectory hasn't diverged so far that Mamba's compressed state becomes noise.

### Phase 4: Sequential Sampling with Truncated BPTT

The previous phases used **random sampling** — each training sample is drawn from a random time, and Mamba's state is reset to zeros before every sample. This means Mamba never sees long continuous sequences during training.

**Sequential sampling** fixes this: training data is divided into 30-day contiguous segments (120 time steps). Segments are shuffled across epochs (inter-segment i.i.d.), but within each segment, samples are iterated sequentially. Mamba's state carries forward within each segment via `stop_gradient` (truncated BPTT): the state values are preserved but gradients are cut at sample boundaries.

```python
# Random sampling (before): state = zeros every sample
state = _reset_ssm_state(state)        # h = 0

# Sequential sampling (new): state carries forward, gradient cut
state = _stop_grad_state(state)        # h preserved, grad detached
```

#### Training Evaluation Results

**Training eval (target_steps=2, state starts from zero):**

| Step | Baseline RMSE | resmem seq RMSE | Gap |
|------|--------------|-----------------|-----|
| 2k | **93.29** | 97.42 | +4.4% |
| 4k | **82.11** | 100.02 | +21.8% |
| 6k | **75.91** | 85.07 | +12.1% |
| 8k | **76.05** | 77.60 | +2.0% |
| 10k | **70.13** | 75.20 | +7.2% |

Sequential resmem is 2–22% worse than baseline on training eval, because during training the model learns to depend on accumulated state (within 30-day segments), but training eval resets state to zero.

#### Long Rollout Inference Results (Corrected, eval-only mode)

**Important correction**: Earlier inference results showing Mamba improvements of $-$27\% to $-$49\% were **incorrect**. They used a flawed `--max-steps 1` workaround that ran one extra training step before evaluation, introducing noise. The results below use the corrected `--eval-only` mode (no extra training).

All checkpoints at step 10k, mesh_size=4. Gap shows resmem relative to baseline (positive = resmem worse).

**Train target_steps=2:**

| Inf t | Forecast | Baseline RMSE | resmem seq RMSE | Gap |
|-------|----------|--------------|-----------------|-----|
| 2 | 12h | **70.16** | 75.19 | +7.2% |
| 10 | 2.5d | **240.56** | 272.75 | +13.4% |
| 16 | 4d | **341.67** | 381.57 | +11.7% |
| 24 | 6d | **442.58** | 481.08 | +8.7% |

**Train target_steps=4:**

| Inf t | Forecast | Baseline RMSE | resmem seq RMSE | Gap |
|-------|----------|--------------|-----------------|-----|
| 2 | 12h | **74.02** | 89.85 | +21.4% |
| 10 | 2.5d | **211.87** | 294.32 | +38.9% |
| 16 | 4d | **286.18** | 387.62 | +35.4% |
| 24 | 6d | **358.73** | 473.81 | +32.1% |

**Train target_steps=6:**

| Inf t | Forecast | Baseline RMSE | resmem seq RMSE | Gap |
|-------|----------|--------------|-----------------|-----|
| 2 | 12h | **80.62** | 99.01 | +22.8% |
| 10 | 2.5d | **207.45** | 278.05 | +34.0% |
| 16 | 4d | **271.49** | 352.67 | +29.9% |
| 24 | 6d | **332.39** | 416.54 | +25.3% |

**Train target_steps=8:**

| Inf t | Forecast | Baseline RMSE | resmem seq RMSE | Gap |
|-------|----------|--------------|-----------------|-----|
| 2 | 12h | **86.08** | 100.01 | +16.2% |
| 10 | 2.5d | **210.67** | 250.79 | +19.0% |
| 16 | 4d | **271.58** | 315.87 | +16.3% |
| 24 | 6d | **330.33** | 374.07 | +13.2% |

#### Conclusions (Corrected)

After fixing the evaluation bug, the results change dramatically:

1. **Sequential resmem does NOT outperform baseline.** Across all training target_steps (2, 4, 6, 8) and all inference rollout lengths (2, 10, 16, 24), baseline is consistently better by 7–39%.

2. **Longer BPTT windows do not help.** Increasing training target_steps from 2 to 4, 6, 8 does not improve resmem relative to baseline. In fact, train target_steps=4 is the worst configuration, and train target_steps=2 is the least bad.

3. **The earlier "49% improvement" was an evaluation artifact.** The `--max-steps 1` workaround ran one extra training step before eval, which introduced random parameter perturbations that happened to favor resmem in some configurations. With proper `--eval-only` mode, no such improvement exists.

4. **Training eval and inference eval now agree exactly** (e.g., baseline t=2 RMSE 70.13 vs 70.16; resmem t=2 75.20 vs 75.19), confirming the fix.

The most plausible explanations for why Mamba does not help:
- **No spatial communication in Mamba**: each of the 2,562 mesh nodes runs its own independent SSM. Temporal patterns that involve spatial motion (e.g., propagating fronts) cannot be captured.
- **State reset between eval samples**: during inference, the model processes each initial condition independently with state starting from zero. State only accumulates within one rollout (2–24 steps), never reaching the rich accumulated state seen during sequential training (up to 120 steps).
- **Training objective mismatch**: the training loss optimizes predictions at `target_steps` horizon, not long rollouts.
- **Mamba position**: placed after the encoder but before the Mesh GNN, so Mamba sees raw encoded features without spatial processing. An alternative would be placing Mamba inside or after the Mesh GNN (`mesh_processor_interleaved`), which has not been tested with sequential sampling.

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
