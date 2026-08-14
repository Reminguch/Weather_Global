# res=1 residual-Mamba on frozen GraphCast — K-scan and current checkpoint

## 1. Figure: skill vs lead time, and the transition at K*

`cold_full_vs_baseline.png`
(source: `results/2026-07-02-v22cl-r1-allK-EMA-eval/plots/cold_full_vs_baseline.png`)

**Setup.** GraphCast-small (35.98 M params, 1.0 deg, 13 pressure levels, mesh 2to5)
is frozen. A residual adapter (a 2-message-step GraphCast encoder + temporal Mamba
blocks + zero-initialised head, ~10.4 M params) is trained closed-loop: the model's
own corrected forecast is fed back as the next input, with stop-gradient on the
correction. `K` is the number of autoregressive steps in the training rollout
(1 step = 6 h). Metric is paper-weighted normalised MSE improvement over the frozen
model: per-channel MSE divided by `diffs_stddev_by_level^2`, level weight
proportional to pressure, GraphCast per-variable weights, cos(lat)-weighted.
Evaluation is a cold-start 240 h rollout (40 steps), residual state zero-initialised.

**Left panel** — improvement vs lead time, one curve per training K.
**Right panel** — the same data re-sliced: improvement vs K, one curve per lead.

**What it shows.** There is a sharp transition. At 240 h the model is *worse* than
frozen GraphCast for K <= 8 and better for K >= 10:

| K | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| @240 h (%) | -5.4 | -8.8 | -5.8 | -1.3 | **+3.6** | +6.3 | +9.1 | +14.4 | +13.6 | **+19.6** | +19.4 |
| @24 h (%) | +6.4 | +7.1 | +7.2 | +7.2 | +6.9 | +6.9 | +6.5 | +6.1 | +5.7 | +4.9 | +5.0 |

so K* is between 8 and 10 (~2 days of rollout exposure). Short lead moves the
opposite way: 24 h skill decays slowly from +7.2 % to +4.9 % as K grows. Training
K therefore trades short-lead accuracy for long-lead accuracy, and the trade only
becomes profitable past K*.

**Three caveats, please read before quoting numbers.**

1. **This figure is trained on 7 years of ERA5, not 15.** The 15-year runs at the
   same architecture and K=20 reach **+24.7 %** at 240 h rather than +19.6 %. The
   7-year runs also become unstable late in training (they drop to ~0 % at step
   12k and 16k); the 15-year runs hold a plateau.
2. **Checkpoint provenance for this lineage is not fully verifiable.** The run
   directory records `start_step=14001` and its training log only covers steps
   14001-20000: the original stage-1 config and loss log were overwritten in place
   by a warm restart. lr / warmup / seed for steps 1-14000 cannot be confirmed from
   any surviving artifact. Treat the absolute numbers as indicative.
3. **The checkpoints are labelled "EMA" but are really SWA** (stochastic weight
   averaging) over a checkpoint window **tuned separately for each K**, so part of
   the K-trend is selection. A fixed-window version is the cleaner ablation.

A clean 15-year replication of the transition region is running now
(K = 4, 8, 10, 12, 16, single-axis, verifiable checkpoints); I will send that
version when it lands.

## 2. Current checkpoint

Clean lineage, config hash and git commit embedded in every checkpoint file:

```
/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res1_sizeladder_15yr/
    ladder_15yr_S128_h128_di128_ds16/v13_residual_SWA_step6k-11k.pkl      (422 MB)
```

15 years of ERA5, K = 20, lr 1e-4, seed 22, closed-loop with stop-gradient,
SWA over steps 6k-11k. Evaluated on a frozen 32-anchor validation split of 2022:

- **J_long (mean improvement over leads 120-240 h): +17.2 %**
- **@240 h: +24.7 %**

The Mamba here is d_hidden 128, d_inner 128, d_state 16 (~0.85 M of the adapter's
10.4 M params). A size ladder spanning 16x the state dimension changes the result
by 0.89 pp, i.e. within evaluation noise, so this is the recommended default.
