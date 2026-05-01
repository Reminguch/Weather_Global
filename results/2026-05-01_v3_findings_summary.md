# v3 K-curriculum + Mamba memory ablation — findings summary (2026-05-01)

Summary of the empirical results produced by the v3 trainer + ablation
infrastructure (commits `7986802`, `b5d0199`, `e4ccaeb` on
`feature/v2-mod-G-A-C`). Raw JSONs and plots referenced inline live under
this `results/` tree.

## A. K-curriculum eval Δ% across phases (per-phase plateau)

Each phase evaluates at horizon=K. Δ% is `(baseline_RMSE − corrected_RMSE) /
baseline_RMSE × 100`, lat-weighted (`*_RMSE_latw`), averaged across all
eval points within the phase.

| phase | step range | eval horizon | Z latw RMSE Δ% (mean ± std) | MSLP latw RMSE Δ% (mean ± std) | n eval points |
|-------|------------|--------------|-----------------------------|-------------------------------|---------------|
| K=2   | 3600–5400  | 2 | +1.25% ± 0.15 | ~+1.10% | 10 |
| K=4   | 5600–7200  | 4 | +1.66% ± 0.10 | +1.02% ± 0.09 | 9 |
| K=6   | 7200–8800  | 6 | **+2.09% ± 0.12** | **+1.41% ± 0.19** | 9 |

Each K-switch produces a step-jump within ~200 train steps; phases
internally plateau (train loss flat at the new floor for 1000+ steps,
eval Δ% bounces in a band).

**Caveat**: eval horizon = K, so longer leads (with bigger baseline error)
are averaged in at every K-switch boundary. Baseline RMSE itself jumps:
Z baseline latw RMSE = 32 m²/s² (K=2 phase) → 48 (K=4) → 59 (K=6) despite
identical baseline weights. Apples-to-apples comparison submitted as
slurm 7559631: K=4 step 7000 ckpt and K=6 step 8500 ckpt × eval horizon
{4, 6} → 2×2 cells to disentangle real-learning vs horizon-shift artifact.

Plots: `2026-05-01_K6_train_eval_curves/v3_K6_train_eval_curves.png`,
`v3_K_curriculum_stitched_dpct.png`,
`v3_K_curriculum_baseline_horizon_artifact.png`.

## B. D3 ablation matrix — Mamba memory directly verified at K=4

5-mode ablation on K=4 step 6000 (TF, horizon=4, num_segments=64). Same
matrix on K=2 step 5500 (TF, horizon=2). Raw JSONs in
`2026-04-30_mamba_memory_ablation/`.

Modes:
- NORMAL: full model
- RESET: SSM hidden state zeroed each step
- ZEROPREV: prev-residual encoder input zeroed
- NOFB: ZEROPREV + state_from_feedback path disabled
- BOTHOFF: RESET + ZEROPREV + state_fb off (per-step residual head, no
  temporal context)

### K=4 step 6000, +24h Z latw RMSE Δ%

| mode | Δ% | meaning |
|------|------|---------|
| NORMAL | **+1.881%** | full model |
| ZEROPREV | +1.770% | encoder-input direct contribution = 0.111pp |
| NOFB | +1.773% | total residual-feedback contribution = 0.108pp |
| RESET | +1.293% | **hidden carries 0.588pp = 31% of total** |
| BOTHOFF | +1.303% | floor / per-step residual head |

Decomposition:
- Hidden alone (`NOFB − BOTHOFF`) = **+0.469pp** = 81% of total temporal lift
- Total temporal lift (`NORMAL − BOTHOFF`) = +0.577pp at +24h Z

### Lead-time decomposition (Z latw RMSE Δ%, K=4 step 6000)

| lead | encoder residual contribution | Mamba hidden contribution |
|------|-------------------------------|---------------------------|
| +6h  | **0.65pp** (100% of correction) | **0.00pp** |
| +12h | 0.13pp | 0.00pp |
| +18h | 0.09pp | 0.31pp |
| +24h | 0.11pp | **0.59pp** (81% of correction) |

Two channels are clearly **complementary**: encoder residual = short-lead
patcher, Mamba SSM hidden = long-lead memory. Hidden contribution is
zero at +6h/+12h (SSM time axis hasn't built up enough state) and
dominates at +24h.

### K=2 step 5500 — verifies the curriculum direction

| | NORMAL | RESET | hidden contribution |
|---|--------|-------|---------------------|
| Z +6h | +1.45% | +1.45% | **0.00pp** |
| Z +12h | +1.50% | +1.43% | 0.07pp |
| Channel-mean MAE Δ% | +0.803% | +0.790% | −0.012pp (≈ 0) |

At K=2 (T=2, hidden has 1 step to build), hidden contribution is
essentially zero. K=4 unlocks it (31% on +24h Z). This is the direct
verification of the K-curriculum hypothesis.

## C. Per-lead lead-time signature (D1 — K=4 step 7000)

Raw: `2026-04-30_K4_perlead_diagnostic/perlead_K4_step7000_h4_tf.json`.

Per-variable latw RMSE Δ% by lead time. Variables that show
correction-grows-with-lead pattern are the ones Mamba hidden helps:

| variable | +6h | +12h | +18h | +24h | trend |
|----------|-----|------|------|------|-------|
| **geopotential** | +1.14% | +1.35% | +1.68% | **+1.88%** | ↑ (+0.73pp) |
| **mean_sea_level_pressure** | **−0.00%** | +0.66% | +1.00% | **+1.19%** | ↑ (+1.19pp) |
| temperature | +0.60% | +0.71% | +0.72% | +0.72% | ↑ |
| u_wind (upper) | +0.16% | +0.21% | +0.21% | +0.22% | ↑ |
| 10m_u_wind | +0.33% | +0.38% | +0.40% | +0.41% | ↑ |
| **vertical_velocity** | +0.08% | +0.04% | +0.03% | +0.03% | ↓ |
| **specific_humidity** | +0.82% | +0.80% | +0.71% | +0.68% | ↓ |
| **total_precipitation** | +0.57% | +0.56% | +0.22% | +0.15% | ↓ |
| Channel-mean MAE | +0.63% | +0.76% | +0.81% | +0.86% | ↑ |

Memory-friendly variables (Z, MSLP, T, winds): correction grows with lead
→ memory accumulating. Noisy/displacement-dominated variables (precip,
humidity, w): correction decays → no long-horizon benefit.

## D. K=4 plateau diagnosed as multi-task averaging trade-off

K=4 phase train_loss flat at 0.184 for 1500+ steps, grad_norm 0.013 ≪
grad_clip 1.0. Three diagnostics:

- **D4 (LR×3 from 1e-4 to 3e-4, 800 steps)**: train_loss 0.18363 →
  0.18600 (slight rise). **Not optimization-stuck.**
- **D5 (lead-weight power=1.0 only, 800 steps)**: train_loss 0.20427 →
  0.20691 (rises because long leads weigh more). Eval Z marginally
  better but precip stable. Lead-weighting alone doesn't unblock.
- **D1 per-lead (above)**: Z/MSLP long lead getting better, precip/humidity
  long lead getting worse → averaging cancels in train_loss.

**Conclusion**: plateau ≠ capacity ceiling, plateau = multi-task trade-off
under uniform-channel uniform-lead loss. Motivated the v3 three-group
loss + LW=1.0 setup which broke the plateau (K=6 phase Z mean +2.09% vs
K=4 +1.66%, grad_norm 0.013 → 0.063, 5×).

## E. Pending follow-ups (jobs in queue at time of writing)

- **7558339** — D3 ablation matrix on K=6 step 8500 (5 modes, h=6 TF).
  Will tell whether hidden contribution % grows from K=4's 31% to higher
  at K=6, or saturates.
- **7559631** — Apples-to-apples K=4/K=6 × horizon=4/6 (4 evals). Splits
  the K=4→K=6 +0.45pp Z gain into real-learning vs horizon artifact.
- **7558363** — K=8 training, resumes from K=6 step 8500, 2500 K=8 steps.
  Pushes the curriculum one more notch.

## Files in this push

```
results/2026-04-30_K4_perlead_diagnostic/
  perlead_K4_step7000_h4_tf.json

results/2026-04-30_mamba_memory_ablation/
  v3K2_step5500_h2_tf_{NORMAL,RESET,ZEROPREV,NOFB,BOTHOFF}.json
  v3K4_step6000_h4_tf_{NORMAL,RESET,ZEROPREV,NOFB,BOTHOFF}.json

results/2026-05-01_K6_train_eval_curves/
  summary.json
  v3_K6_eval_per_variable_Z_MSLP.png
  v3_K6_train_eval_curves.png
  v3_K6_train_per_group_Z_MSLP.png
  v3_K_curriculum_baseline_horizon_artifact.png
  v3_K_curriculum_stitched_dpct.png

results/2026-05-01_v3_findings_summary.md   <- this file
```
