# v15 — K=1 (single-step) Mamba pretraining (foundation for v18-v22)

**Role**: the "K=1" pretraining stage from which all subsequent long-AR experiments
(v18, v20, v21, v22) resume. Trains GraphCast-small + full S6 Mamba on **single-step
prediction only** (no AR rollout, `--ar-tail-K 0`) for 20,000 steps.

## Naming convention note

This project uses **two related but distinct numbers**:

- **`K` in run names** (e.g. `K1_from_v15v2_20k`, `K22_from_v22K1_23k`) = the AR rollout
  horizon during *training*. K=1 means "single-step prediction, no AR rollout."
- **`ar_tail_K` CLI argument** = the number of AR-rollout anchors at the *tail* of each
  training chunk. `ar_tail_K=0` corresponds to K=1 (zero AR-tail anchors → all anchors
  teacher-forced → single-step training). `ar_tail_K=22` corresponds to K=22 (22 AR-tail
  anchors).

The off-by-one is awkward but the convention is consistent across v15/v20/v22 runs:
"K=1" ⇔ `ar_tail_K=0`, "K=N" for N≥2 ⇔ `ar_tail_K=N`.

Run directory: `/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2/`
Final ckpt: `v13_residual_step20000.pkl`
Trained: April–May 2026

---

## 1. What v15 is

v15 = **frozen GraphCast-small backbone** + **trainable residual head with full S6 Mamba**
trained with `ar_tail_K=0` (i.e., K=1 in run-name convention) — meaning each training sample uses ground-truth inputs for
the entire chunk; the model never sees its own predictions during training. This is the
standard "teacher-forcing" pretraining used in most weather ML pipelines.

It establishes a strong single-step residual corrector. The Mamba block's hidden state
still propagates across the chunk's 24 chunked steps via `hk.scan`, so the model does
learn short-range temporal recurrence — just not autoregressive cascading.

## 2. Architecture

Identical to v20/v22 (see `v20_K14_production.md` / `v22_K22_production.md` for full breakdown):

- Frozen GraphCast-small (1°, mesh-5, width=512, 16 msg-steps, ~36M frozen params)
- Trainable residual head (10.2M params): 2-step processor, latent=512
- Temporal Mamba S6 block (851k params): `hidden_size=128, d_state=16, d_conv=4,
  dt_rank=auto(32), layers=2, location=mesh_processor_interleaved`
- `temporal_residual_head` 83×83 per-channel output (7k params, zero-init)
- **Total trainable**: 11,090,319 params

Source code: `scripts/training/full_mamba_v20/train_mz_v20.py` (v20/v22 reused this script with K argument changes).

## 3. Training hyperparameters

| | v15 K=1 |
|---|---|
| `precision` | bf16 |
| `batch_size` | 1 |
| `lr` | 1e-4 (constant after 200-step warmup) |
| `weight_decay` | 1e-4 |
| `grad_clip` | 1.0 |
| `ar_tail_K` | **0** (no AR rollout — teacher forcing only) |
| `sequential_segment_steps` | 32 |
| `bptt_steps` | 8 |
| `max_steps` | 20000 |
| `checkpoint_every` | 1000 |

Note: v15's BPTT/segment sizes (8 / 32) are smaller than v20/v22 (24 / 96). Each v15 step
processes ~8 anchors vs v20/v22's ~24 anchors, so v15's step count is ~3× higher per
epoch, but each step is ~3× cheaper. The total compute is similar.

## 4. Trajectory

Fresh-init loss starts at ~0.51 (= pure baseline GraphCast loss, since `temporal_residual_head`
and Mamba `out_proj` are zero-init → residual contribution = 0 at step 1).

Loss plateaus around step 10000-15000 in the 0.47-0.49 range. Final ckpt at step 20000
reaches lat-weighted MSE ~0.47 (normalized).

## 5. Role in the AR curriculum

v15 step 20000 is the **canonical resume point** for the K-curriculum follow-ups:

```
DeepMind GraphCast-small ckpt (frozen baseline)
            ↓ overlay 94 matching GC params
v15 K=1 (20,000 steps)
            ↓ resume_from v15 step20000
            ├──→  v18, v19 (intermediate K-experiments)
            ├──→  v20 K=14 (3,000 more K=1 steps + 3,000 K=14 production)
            └──→  v22 K=22 (3,000 more K=1 steps + 3,000 K=22 production)
```

Because v15 already established 20k steps of single-step Mamba training, subsequent
versions need only a short K=1 spinup before switching to long-AR (K=14, K=22) production.

## 6. Why v15 matters

Going straight to fresh-init K=22 (no K=1 pretraining) typically **diverges**: at step 1,
random Mamba weights produce nonsensical residual corrections, the 22-step AR rollout
cascades the error, loss explodes. The K=1 pretraining is a curriculum step that:

1. Stabilizes Mamba's internal SSM state and gating in single-step regime
2. Teaches the conv1d kernel basic short-range temporal patterns
3. Bootstraps the per-channel output head to near-identity
4. Provides a non-pathological starting point for long-AR rollout

v23 (d_conv=8 ablation) had to retrain a similar K=1 stage from scratch because v15's
d_conv=4 conv1d.kernel shape `[128, 4]` does not match v23's expected `[128, 8]`.

## 7. Source code & launch

- Training script: `scripts/training/full_mamba_v20/train_mz_v20.py` (with `--ar-tail-K 0`)
- v15 used `bptt_steps=8 --sequential-segment-steps 32`; v20/v22 use `24 / 96`.
- The original v15 slurm launchers are in `slurm_pilots/` (e.g., `v15_20k_v2.slurm`,
  `v15_20k_v2_resume.slurm`).
