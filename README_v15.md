# v15: paper-standard residual + Mamba pipeline

## Tl;dr
- v9-style architecture (full DeepMind GraphCast small as frozen baseline + Mamba interleaved processor + zero-init residual head).
- Trained on paper-standard ERA5 from WeatherBench2's cloud zarr `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr`, 2015-2021, validated on 2022.
- **Best K=1 number (paper-comparable): +2.904% mean lat-weighted RMSE improvement, 83/83 channels improved at step 14000 (epoch 11).** Final step 20000: +2.473%, 83/83.
- K=2..6 AR closed-loop rollout degrades sharply (+2.04% → -0.86%): teacher-forced single-step training does not generalise to multi-step rollout. v3-style K-curriculum training is the principled fix; v16 is the in-flight experiment that adds closed-loop AR inside the BPTT chunk.

---

## Models in this branch

| Variant | Train recipe | Eval at K=1 | Notes |
|---|---|---|---|
| **v15 v2** | v9 arch, teacher-forced K=1, 7 yr WB2-standard data, bptt=8, d_state=16 | **+2.473%** @20k, 83/83 ch | the production baseline |
| v15 v3 | v15 v2 + bptt 8 → 16 | cancelled (no win) | longer BPTT does not change train loss at all |
| v15 v4 | v15 v2 + Mamba d_state 16 → 32 | +2.347% @17k | doubling SSM state ≈ noise, K=1-only regime saturates at d_state=16 |
| **v16** | v15 v2 + **closed-loop AR within BPTT chunk** (frozen baseline + residual forwarded live each step; gradient flows back through the self-feed) | in flight | the K-curriculum-equivalent fix for AR rollout collapse |

---

## Data pipeline (this branch)

The legacy local `wb2_res1_levels13_1979_2021.zarr` is **non-standard**: lon-collapsed poles, ~0.5 K global RMSD vs the WB2 cloud zarr that GraphCast / WeatherBench2 papers use. We rebuilt the pipeline from scratch on the paper-standard data.

Steps (all in `src/data/`):
1. `download_wb2_6h.py` — pulls one date window from the WB2 cloud zarr and writes a local zarr in our convention (lat 90 → -90, 13 GraphCast pressure levels, 13 target vars).
2. `download_wb2_chunks.sh` — wrapper that downloads 2015-2022 in 16 × 6-month chunks (safe against login-node kills; each chunk is atomic and writes a `.done` marker).
3. `merge_zarrs.py` — concatenates the 16 chunk zarrs into one `wb2_res1_levels13_2015_2022.zarr`. Rounds lat/lon to integer to avoid 1e-13 float drift between sources.
4. `prepare_streaming_store.py` — converts the zarr to per-variable `.npy` memmaps for fast training.
5. `precompute_residual_targets.py` — runs the frozen GC1 baseline once per anchor, saves `r_target = truth − baseline_pred` to disk (used by v15 v2/v3/v4; not used by v16).

Sanity-checked the WB2 cloud zarr against the ARCO 1h zarr: 16 vars × 28 timestamps → max relative diff ≤ 8e-6 (float32 round-trip).

---

## Architecture (v15 v2 / v3 / v4 / v16)

```
inputs (lat-lon, time=2 -- 12 h window + forcings)
    │
    ▼
[1] grid2mesh_gnn          ← Encoder: lat-lon grid → icosphere mesh (DeepMind weights)
    │
    ▼
[2] mesh_gnn processor       ← Processor: 2 mesh GNN steps + Mamba interleaved
    │   { mesh_gnn_step → Mamba SSM block }  × 2
    │   (vs 16 mesh GNN steps in the frozen GC1 baseline)
    │
    ▼
[3] mesh2grid_gnn            ← Decoder: mesh → grid (DeepMind weights)
    │
    ▼
[4] temporal_residual_head   ← fresh zero-init Linear(83 → 83) on grid nodes
    │
    ▼
residual prediction r̂_t  (shape: lat × lon × 83 channels)
```

The frozen baseline (`baseline_msg_steps=16`) and the residual model (`residual_msg_steps=2`) share encoder/decoder weights. Only the residual model is trained.

Per-block Mamba defaults: `hidden=128, d_state=16, d_conv=4, dt_rank=auto, layers=2` (`mesh_processor_interleaved`).

---

## Training entry points

### v15 v2 — teacher-forced single-step (canonical)
```
sbatch /home/lm8598/Fermihubbardnumeric/slurm_pilots/v15_20k_v2.slurm
```
Uses `scripts/training/full_mamba_v13/train_mz_v13.py` with precomputed residual targets. ~0.77 s/step, 1276 steps/epoch.

### v15 v3 (bptt=16) — cancelled
```
sbatch /home/lm8598/Fermihubbardnumeric/slurm_pilots/v15_v3_bptt16.slurm
```

### v15 v4 (d_state=32)
```
sbatch /home/lm8598/Fermihubbardnumeric/slurm_pilots/v15_v4_dstate32.slurm
```

### v16 — closed-loop AR within BPTT chunk
```
sbatch /home/lm8598/Fermihubbardnumeric/slurm_pilots/v16_20k.slurm
```
Uses `scripts/training/full_mamba_v16/train_mz_v16.py`. Inside each 8-anchor BPTT chunk:
- Anchor 1 receives real ERA5 inputs.
- Anchors 2..8 receive the model's own corrected prediction (`baseline_pred + residual_pred`) shifted forward in the 12 h window.
- Gradient flows back through the self-feedback.
- `jax.checkpoint` is wrapped around the per-step forward so mesh-level activations are recomputed during backward → ~10× lower activation memory at the cost of ~2× compute.

Hyperparameters in line with v3: `--grad-clip 1.0 --warmup-steps 500 --lr 1e-4`. Expected ~3 s/step, 16 h wall (will likely TIMEOUT around step 18-19k; resume with `--resume-from`).

---

## Eval

### K=1 single-step (paper-comparable headline metric)
```
sbatch /home/lm8598/Fermihubbardnumeric/slurm_pilots/v15_v2_eval_all.slurm
```
Calls `scripts/training/full_mamba_v9/eval_v9.py`. Lat-weighted (cos(lat)/cos(lat).mean()) RMSE / MAE per channel + per-variable + per-level.

### K-step AR closed-loop rollout
```
sbatch /home/lm8598/Fermihubbardnumeric/slurm_pilots/v15_v2_eval_rollout_mae.slurm
```
Calls `scripts/training/full_mamba_v9/eval_v9_rollout.py`. Manual AR chain (one-step chunks via `graphcast.rollout.chunked_prediction_generator`); per-lead-time RMSE and MAE for K=1..6.

### Diagnostics (per-anchor or per-pixel)
- `diag_tp6h_errors.py` — dump raw truth / baseline / full fields for a variable (used for error distribution histograms, scatter plots, zonal spectra).
- `diag_var_timeseries.py` — per-anchor lat-weighted MAE/RMSE + spatial means across 2022 (for timeseries plots).
- `compute_historical_records.py` — per-pixel max/min over a train-year window (for extreme-record exceedance analysis à la WB2 Fig. 2).

---

## Headline numbers (v15 v2 on paper-standard 2022)

K=1 (6 h lead):
- **mean lat-weighted RMSE improvement over GraphCast small baseline: +2.473%** (step 20000) / **+2.904%** (step 14000 best by validation).
- **83/83 channels improve.**

K=1..6 AR rollout, step 20000:
| K | lead | mean RMSE improvement | mean MAE improvement | channels >0 (RMSE) |
|---|---|---|---|---|
| 1 | 6 h | +2.41% | +2.34% | 83/83 |
| 2 | 12 h | +2.03% | +2.15% | 79/83 |
| 4 | 24 h | +0.51% | +0.73% | 42/83 |
| 6 | 36 h | -0.86% | -0.64% | 14/83 |

Loss-of-skill at K≥3 is the same teacher-forced-only artefact v3 documented. Reason: the model's training-time inputs are always real ERA5 windows, never its own drifted predictions. v16 is the experiment that fixes this directly.

---

## Reports and plots in this branch

- `results/2026-05-12_v15_v2_step14000_plots/` — full plot set for the best K=1 ckpt: train loss vs epoch, eval improvement vs epoch, RMSE/MAE-vs-lead-time, zonal energy spectra (11 variables, K=1 and K=6), pixel-error distributions, signed bias and truth-vs-pred timeseries across 2022, paper-style extreme-record exceedance plot.
- `results/2026-05-12_v3_vs_v15_detailed.pdf` — detailed step-by-step contrast of v3's K-curriculum target_rollout training vs v15 v2's teacher-forced BPTT chain, with a fix proposal that v16 implements.
