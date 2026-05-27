# AR-Training-Lianghong — v20/v22 Long-AR Curriculum Snapshot

Branch owner: Lianghong Mo (lianghongmo0311@gmail.com)
Snapshot date: 2026-05-27
Base: `main` @ b505d04

This branch is a self-contained record of the **v20 / v22 long-AR
training curriculum** for the residual-Mamba GraphCast model. It bundles
the production trainer, the K-scan SLURM launchers, per-version
writeups, and the lightweight evaluation results (no checkpoints, no
training logs — just headline JSON metrics + key plots).

The runtime model code (frozen GraphCast-small backbone, residual head,
S6 Mamba module) is **not** included on this branch by design — it
lives in the main project tree at `/home/lm8598/Weather_Global_experiments/`.
See "Where the model code lives" below.

---

## 1. What v20 and v22 are

Both are the **same architecture** trained with progressively longer
autoregressive (AR) horizons:

| Layer | v20 | v22 | Trained? |
|---|---|---|---|
| Frozen GraphCast-small backbone (1°, mesh-5, 16 msg-steps) | same | same | NO |
| Residual head (latent=512, gnn_msg_steps=2) | same | same | YES |
| 4× Mamba S6 blocks (2 insertions × 2 layers, interleaved in residual processor) | same | same | YES |
| 83×83 per-channel output linear (`temporal_residual_head`) | same | same | YES |
| `--ar-tail-K` | **14** | **22** | — |
| `bptt_steps` | 16 | 24 | — |
| `sequential_segment_steps` | 64 | 96 | — |
| Total trainable params | 11.09M | 11.09M | — |

**Mamba S6** per block: hidden_size=128, d_inner=128 (expand=1),
d_state=16, d_conv=4, dt_rank=32 (auto), layers=2; per-`(batch,
mesh_node)` persistent `ssm_state` + `conv_cache`.

Forward path: `input → frozen GC → residual_head(latent + Mamba) +
zero-head → corrected output`.

---

## 2. Headline result — long-AR helps long-lead

Eval: 240 sample anchors × 40 forecast lead steps (6h → 240h),
lat-weighted RMSE, vs frozen GraphCast-small baseline on identical
anchors.

| Run | K | BPTT | Total MSE-imp @ 240h | Notes |
|---|---|---|---|---|
| v20 | 14 | 16 | **+3.56%** | sits *below* K\* |
| v22 | 22 | 24 | **+12.69%** | sits *above* K\* — long-horizon stabilization regime |
| Δ | +8 | +8 | +9.13 pp | same architecture, only K differs |

Per-variable peak RMSE-improvement (v22 K=22): 2m_T +3.8% @ 48h,
MSL +4.2%, Z500 +2.5%, Z850 +3.4%. All 83 channels improve at short
leads (1–24h); ~24–30 channels remain better than baseline through 240h.

See [`results/v22/plots/K22/4metrics_K40.png`](results/v22/plots/K22/4metrics_K40.png)
and [`results/v22/plots/K22/per_variable_improvement_K40.png`](results/v22/plots/K22/per_variable_improvement_K40.png).

---

## 3. K-scan and the K\* crossover

A 12-point K-scan (K = 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22) shows
a **transition between K = 16 and K = 18** above which 240h
MSE-improvement jumps:

- K ≤ 16 cluster near **+2 to +4%** improvement
- K ≥ 18 jump to **+8 to +13%**

Interpretation: the model's effective AR-rollout receptive field crosses
a threshold where Mamba's persistent SSM state begins tracking
multi-day-evolving patterns rather than just short-range corrections.
Short AR-tail training leaves the state untrained at long lead times so
improvement decays with lead; long AR-tail training trains the state to
be informative across the full forecast horizon.

Per-K eval JSON for all 12 K values is in
[`results/v22/eval_jsons/`](results/v22/eval_jsons/).

---

## 4. Training chain

```
DeepMind GraphCast-small ckpt (frozen, ~36M params)
            │ overlay 94 matching GC params
            ▼
v15 K=1 fresh init  ──────────────►  20,000 steps  (teacher forcing only, --ar-tail-K 0)
            │ resume_from v15 step20000
            ├────────────────────►  v20 K=14  (+3k K=1 spinup → +3k at ar-tail-K=14)
            └────────────────────►  v22 K=22  (+3k K=1 spinup → +3k at ar-tail-K=22)
```

Each downstream run is a small (~3000-step) AR-tail finetune on top of
the shared K=1 foundation. Total Mamba training: **23k K=1 steps +
3k K-production steps = 26k steps** for v20 and v22 alike — the only
difference between them is the K used in the production finetune.

### Naming convention (important)

`K` in run names is the *training AR horizon* (production K). The
corresponding CLI argument is `--ar-tail-K` with one off-by-one quirk:

- `K=1` (single-step, no AR rollout) ⇔ `--ar-tail-K 0`
- `K=N` for `N ≥ 2` ⇔ `--ar-tail-K N`

---

## 5. AR + truth loss aggregation

The trainer does **not** weight AR-step loss and truth-step loss
separately. For each anchor in a `bptt`-long chunk it computes a
per-anchor `MSE(residual_pred, truth − stop_grad(GC(current_inputs)))`
and aggregates by **uniform mean** across all `bptt` anchors:

```python
losses = []
for i in range(bptt):
    loss_i = MSE(residual_pred_i, truth_i - GC(current_inputs_i))
    losses.append(loss_i)
    if next_i < ar_start_static:   # first (bptt - K) anchors
        current_inputs = inputs_truth_list[next_i]      # truth-fed
    else:                          # last K anchors
        current_inputs = shift_with_state(current_inputs, baseline_pred, ...)  # AR-fed
return jnp.stack(losses).mean()
```

So the AR-tail's effective weight in the total loss is just `K / bptt`,
controlled via `--ar-tail-K`, not via a hand-tuned loss coefficient.
See `scripts/training/full_mamba_v20/train_mz_v20.py` (the train-step
loop near the bottom of the file).

---

## 6. What's in this branch

### Code

- [`scripts/training/full_mamba_v20/train_mz_v20.py`](scripts/training/full_mamba_v20/train_mz_v20.py)
  — production trainer, shared by both v20 and v22 (only CLI args differ)
- [`scripts/training/full_mamba_v20/eval_v20_rollout.py`](scripts/training/full_mamba_v20/eval_v20_rollout.py)
  — 240-anchor × 40-step rollout eval against frozen GC baseline
- [`scripts/training/full_mamba_v20/save_dense_preds_v20.py`](scripts/training/full_mamba_v20/save_dense_preds_v20.py)
  — dense per-grid prediction dumper
- [`scripts/training/full_mamba_v20/save_extreme_records_data_v20.py`](scripts/training/full_mamba_v20/save_extreme_records_data_v20.py),
  [`scripts/training/full_mamba_v20/test_save_extreme_records_v20_full_fb.py`](scripts/training/full_mamba_v20/test_save_extreme_records_v20_full_fb.py)
  — extreme-event scoring pipeline

### SLURM launchers

- [`slurm_pilots/v20_kscan/`](slurm_pilots/v20_kscan/) — v20 K-scan launchers (K=4, 6, 8, 10, 12, 14)
- [`slurm_pilots/v22_kscan/`](slurm_pilots/v22_kscan/) — v22 K-scan launchers (K=2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
- [`slurm_pilots/v22_K1.slurm`](slurm_pilots/v22_K1.slurm) — K=1 spinup

### Per-version writeups

- [`docs/experiments/v15_K1_baseline.md`](docs/experiments/v15_K1_baseline.md) — single-step pretrain (20k steps)
- [`docs/experiments/v20_K14_production.md`](docs/experiments/v20_K14_production.md) — K=14 production
- [`docs/experiments/v22_K22_production.md`](docs/experiments/v22_K22_production.md) — K=22 production + K-scan

### Results (lightweight)

- [`results/v20/`](results/v20/) — v20 K-scan eval JSONs (K=1,4,6,8,10,12,14) + key plots
- [`results/v22/`](results/v22/) — v22 K-scan eval JSONs (K=1,2,4,…,22) + key plots
- Each has its own `README.md` with architecture, hyperparameters, and per-variable numbers

Eval JSON schema: per-variable × per-pressure-level × per-lead-step
RMSE / MAE / improvement, lat-weighted.

**Not included** (intentionally, to keep the branch small): model
checkpoints (`*.pkl`), training-step loss histories (`train_logs/`),
dense per-grid prediction dumps, extreme-event raw record tensors.

---

## 7. Where the model code lives

The trainer above imports from the parent project tree:

- `src/models/mamba/modules/temporal_mesh_mamba_Ilya.py` — production
  full-S6 Mamba (per block: in_proj → depthwise causal conv1d →
  selective B(x)/C(x) via x_proj → low-rank dt_proj → SSM scan → out_proj;
  zero-init out_proj)
- `src/models/mamba/training/param_utils.py` — param-merging /
  warm-start helpers
- `src/models/graphcast/training/core/model.py` — frozen GC-small
  + residual head
- `src/data/prepared_array.py` — ERA5 batch builder
- `scripts/training/train_graphcast.py` — base trainer this one wraps
- `scripts/training/full_mamba_v9/train_mz_v9.py` — re-uses utilities
  (`DirectResidualNormalizer`, `scalarize_loss`, `_build_segments`)

A frozen snapshot of the exact Mamba source used for v20/v22 is at
`/scratch/gpfs/DABANIN/lm8598/Weather_Global/docs/model_source_snapshots/v20_v22_mamba_2026-05-23/`.

To actually run the trainer you need a checkout of the feature branch
(`feature/v2-mod-G-A-C` or descendants) where all of `src/models/...`,
`src/data/...`, and the GraphCast wrappers live. This branch is meant
as a documentation/snapshot bundle, not a standalone runnable repo.
