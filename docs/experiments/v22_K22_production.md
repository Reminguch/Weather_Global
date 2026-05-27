# v22 — Long-AR Mamba residual on GraphCast-small (K=22)

Run directory: `/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/`
Best checkpoint: `K22_from_v22K1_23k/v13_residual_step26000.pkl`
Date: 2026-05-23

---

## 1. What v22 actually is

v22 = **frozen GraphCast-small backbone** + **trainable residual head with full S6 Mamba** (interleaved inside the residual GNN processor).

| Layer | Status | Source of params |
|---|---|---|
| Frozen GraphCast-small encoder/processor/decoder | NOT trained | DeepMind pretrained ckpt (`GraphCast_small … 1979–2015 1deg`) |
| Residual head (encoder + 2-step processor + decoder + zero-head) | TRAINED | Fresh init + warm start from v22 K=1 ckpt |
| 2× Mamba S6 blocks inside residual processor (per insertion, 2 insertions) | TRAINED | Same warm start |
| 83×83 per-channel output linear (`temporal_residual_head`) | TRAINED | Same warm start |

Forward path: `input → frozen GC → residual_head(latent + Mamba) + zero-head → corrected output`. Residual head reads the latent from the frozen processor and adds a correction.

---

## 2. Architecture details (verified against ckpt)

### GraphCast backbone (frozen)
- Resolution: **1.0°** (181 × 360 grid, 65160 grid nodes)
- Mesh: icosahedral level **5** (10242 mesh nodes)
- Latent dim (`width`): **512**
- Baseline `gnn_msg_steps`: **16** (frozen)
- Pretrained DeepMind small ckpt; 94 overlay params copied to residual init, 46 fresh-init.

### Residual head (trainable, 11.09M params)
- `latent_size`: 512
- `gnn_msg_steps`: **2** (the residual processor has 2 message-passing steps)
- `hidden_layers`: 1
- `radius_query_fraction_edge_length`: 0.6
- `mesh2grid_edge_normalization_factor`: 0.618
- Architecture tag: `v9_corrected_GCResidualWithZeroHead`

### Temporal Mamba module — **full S6-style** (NOT a simplified gated SSM)
Source: `/home/lm8598/Weather_Global_experiments/src/models/mamba/modules/temporal_mesh_mamba_Ilya.py`
(NOT `src/models/temporal_mesh_mamba_stateful.py` — that file is a leftover/earlier prototype.)

**Frozen snapshot of the exact source code used for v22 K=22**:
`/scratch/gpfs/DABANIN/lm8598/Weather_Global/docs/model_source_snapshots/v20_v22_mamba_2026-05-23/`

Per Mamba block:

| Sub-module | Shape | Function |
|---|---|---|
| LayerNorm | (512,) scale + offset | pre-norm |
| `in_proj` linear | (512 → 256) | split into (x_path=128, z_path=128); 2× expand to D_inner=128 |
| **`conv1d`** depthwise causal | kernel=4, channels=128, bias=128 | local temporal mixing |
| `x_proj` linear | (128 → 64) | outputs (dt_input=32, B=16, C=16) — **selective B(x), C(x)** |
| `dt_proj` linear | (32 → 128) + bias | low-rank Δ projection (dt_rank=32) |
| `A_log` param | (128, 16) | SSM dynamics matrix, A = −exp(A_log) per (D_inner, d_state) |
| `D` param | (128,) | per-channel skip |
| `out_proj` linear | (128 → 512) | zero-init at start |

### Effective Mamba config

| Knob | Value | Notes |
|---|---|---|
| `hidden_size` (H = d_model) | 128 | input/output feature dim for the SSM core (residual stream is 512, Mamba sub-stream is 128 inside) |
| `d_inner` | 128 | resolves to `hidden_size` because `temporal_d_inner=null`; expand factor = 1 (NOT default 2) |
| `d_state` (N) | 16 | per-channel state modes |
| `d_conv` | 4 | causal conv kernel |
| `dt_rank` | 32 (auto = ⌈d_model/16⌉ where d_model=512) | low-rank Δ |
| `layers` (per insertion) | 2 | two Mamba blocks stacked |
| `bias` | False | |
| `conv_bias` | True | |
| `dropout` | 0.0 | |
| `zero_init_output` | True | initial residual contribution is zero |

### Where Mamba blocks live in the model

- `temporal_location` = **`mesh_processor_interleaved`** → Mamba block runs **after each message-passing step** of the residual processor
- Residual processor has `gnn_msg_steps=2 × num_processor_repetitions=1 = 2` insertion sites
- Per insertion: `temporal_layers=2` Mamba blocks
- Plus 1 extra at output: `temporal_residual_head` (per-channel 83→83 linear, 6,972 params)
- **Total Mamba blocks per AR step = 2 × 2 = 4**

### Per-(batch, mesh_node) persistent state

Across autoregressive steps, each Mamba layer maintains:
- `ssm_state`: shape `(B, N_mesh, D_inner=128, d_state=16)` = ~21 MB bf16 (B=1)
- `conv_cache`: shape `(B, N_mesh, D_inner=128, d_conv-1=3)` = ~3.9 MB bf16

Total state per AR step (all 4 blocks): ~100 MB bf16. BPTT=24 retains ≈ 2.4 GB activation + state memory.

---

## 3. Parameter counts (from ckpt)

| Component | Params |
|---|---|
| GraphCast residual head (GNN + MLPs, all non-temporal) | 10,231,891 |
| Temporal Mamba (2 insertions × 2 layers × 1 block) | 851,456 |
| `temporal_residual_head` (output 83×83) | 6,972 |
| **Total trainable residual params** | **11,090,319** |
| Frozen GraphCast-small backbone (not counted above) | ~5.5M (DeepMind ckpt) |

Per-insertion breakdown (e.g. `mesh_interleaved_temporal_r0_s0`):

| | Params |
|---|---|
| layer_norm_0 + layer_norm_1 | 2,048 |
| mamba_block_0 (A_log + D + conv1d + in_proj + out_proj + ssm/dt_proj + ssm/x_proj) | 211,840 |
| mamba_block_1 | 211,840 |
| **Per insertion total** | **425,728** |

---

## 4. Training hyperparameters

| Parameter | Value |
|---|---|
| `precision` | bf16 |
| `batch_size` | 1 |
| `lr` | 1e-4 |
| `weight_decay` | 1e-4 |
| `grad_clip` | 1.0 |
| `warmup_steps` | 200 |
| `seed` | 22 |
| `input_duration` | 12h (2 input steps) |
| `target_steps` | 1 (single-step forecast per sample) |
| `ar_tail_K` | **22** (autoregressive tail length) |
| `bptt_steps` | **24** (BPTT truncation window) |
| `sequential_segment_steps` | **96** |
| `max_steps` | 26000 |
| `checkpoint_every` | 500 |
| `resume_from` | `results/v22/K1_from_v15v2_20k/v13_residual_step23000.pkl` (start_step=23001) |

---

## 5. Headline result

At lead 240h on the 2022 eval set (lat-w skill score, vs DeepMind GraphCast-small baseline at 1°):

- **Total MSE-improvement (paper-formula): +12.69%**
- Per-variable lat-w RMSE-improvement (selected): 2m_T peak +3.8% at lead ≈48h, MSL peak +4.2%, Z500 +2.5%, Z850 +3.4%
- All 83 channels improve at short leads (1–24h); 24–30 channels remain better than baseline through 240h
- See `plots/4metrics_K40.png` and `plots/per_variable_improvement_K40.png`

K-scan story: K=1→22 production-train ckpts show a transition between K=16 and K=18 (the apparent K\* crossover) where long-lead improvement jumps. v22 K=22 sits well above K\*.

---

## 6. Audit notes (read before modifying)

- The **real Mamba source** is `src/models/mamba/modules/temporal_mesh_mamba_Ilya.py` (full S6).
- `src/models/temporal_mesh_mamba_stateful.py` is a **simpler earlier prototype** (lacks conv1d, x_proj, dt_rank, high-dim SSM state). It is **not** what v22 uses. Do not assume the model behavior from that file.
- The `temporal_d_state / d_conv / dt_rank / d_inner` config knobs ARE consumed; the translation happens in `src/models/mamba/residual_mamba/runtime.py` (sets `predictor._temporal_d_*` attrs) before `TemporalMeshConfig` is constructed.
- The frozen backbone uses `baseline_msg_steps=16` but Mamba blocks live inside the **trainable residual head** which has `residual_msg_steps=2`. So only **2 interleaved insertions**, NOT 16.

---

## 7. Plots in this directory

- `plots/4metrics_K40.png` — Total MSE-imp + per-var RMSE @ 240h
- `plots/per_variable_improvement_K40.png` — 12 per-variable improvement curves vs lead
- `plots/extreme_records_*` — extreme-event scoring
- `eval_jsons/v22_K22_K40.json` — raw eval (per-variable RMSE/MAE × 40 lead steps)

Eval protocol: 240 anchors × 40 lead steps (6h–240h), lat-weighted, vs frozen GraphCast-small baseline forecasts on identical anchors.
