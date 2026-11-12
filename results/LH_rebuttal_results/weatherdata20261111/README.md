# Lianghong’s 2023 evaluation handoff

## Did we use Ilya’s 2023 setup?

> We followed your documented 2023 evaluation setup. We used the same 122 initialization dates, 40 six-hour lead times, native 1° grid and exact GraphCast loss against independently rolled public GraphCast Small. The results come from local runs on a quality-controlled ERA5 store. Exact parity with the frozen Della inputs, runtime and archived forecasts remains unverified. For the raw 2K extreme-event run, we rebuilt the documented 1991–2020 thresholds and scoring locally.

## Evaluation setup

| Component | Local evaluation |
| --- | --- |
| Forecast starts | 00 UTC every three days from January 1 through December 30, 2023, for 122 starts. [Exact date list](setup/initializations_2023_stride3.json) |
| Forecast horizon | 40 six-hour leads, from 6 through 240 hours. The last day-10 target is January 9, 2024 at 00 UTC. |
| Grid and verification | Native global 1° grid, 181 × 360 points and 13 pressure levels. The local ARCO ERA5 store uses nine NCAR-verified replacement chunks. [Truth audit](setup/native_truth_audit.json) |
| Baseline and rollout | Public 1° GraphCast Small with the original normalization statistics and its own autoregressive trajectory. GC–Mamba starts with zero recurrent state and feeds back its corrected forecast. Carry runs retain memory between steps. Fig. 6 controls use the specified reset policy. |
| Loss | Original normalized, weighted GraphCast MSE evaluated in FP32, with BF16 network compute. Reduction is `100 × (1 − pooled model loss / pooled public GC loss)`. The rollout pools all 122 starts and 40 leads. Day 10 pools all 122 starts at 240 hours. |
| Uncertainty | Pointwise 95% exact-loss intervals use 10,000 paired non-wrapping moving-block draws of five starts, seed 20260916. The five-seed figure shows sample SD across training seeds. |

The date list, verification-store metadata hash, public checkpoint hash and normalization hashes are recorded together in the [machine-readable setup](setup/evaluation_protocol.json). Local comparison figures identify the public GraphCast trajectory used as their denominator. The archived Della protocol reference is recorded in the same file.

## 1. GraphCast fine-tuning

The three local runs update all GraphCast parameters with the standard loss (λ = 0), starting from the public checkpoint. Training uses 2015–2021 ERA5, batch four, 96-step segments, 24-step BPTT with a 20-step autoregressive tail, and full weather-feedback gradients. The results below use each run’s native checkpoint at **2,500 total optimizer updates**.

| Peak learning rate | Rollout loss reduction | Day-10 loss reduction |
| --- | ---: | ---: |
| `3e-7` | 4.19% | 4.79% |
| `1e-5` | 3.62% | 6.85% |
| `3e-5` | −2.90% | 2.52% |

The `3e-5` day-10 paired interval spans zero. The 2023 results are fixed-checkpoint comparisons. Local 2022 development verification remains incomplete. The optimizer and sampler were reconstructed locally from the documented recipe and related frozen configurations.

The [fine-tuning folder](01_graphcast_finetuning/README.md) contains full per-start results, paired intervals, configurations, checkpoint references, resource measurements and the 2,500-total-update comparison. In that figure, GCFT has 2,500 total updates and each M1 model has 500 warm-up plus 2,000 main-stage updates.

## 2. Extreme events

The [extreme-event folder](02_extreme_events/README.md) contains the 122-start evaluation of local seed 22 at raw main step 2,000, with an independent public GraphCast baseline. The locally rebuilt percentile protocol uses 1991–2020 ERA5, a ±30-day calendar window at each UTC hour, q98/q2 primary thresholds and q95/q5 sensitivity thresholds. Scores use spherical cell areas for global and land results. Rainfall uses 24-hour accumulation starting at lead 24 hours. Event intervals use 2,000 paired circular date-block draws of five starts, seed 20261106.

At day 1, global cold and wind CSI improved. At day 10, global heat, wind and rain CSI declined. The day-10 global cold CSI interval spans zero. A direct comparison with Ilya’s archived SWA forecast still requires the original Della outputs and a common threshold/scorer check. The S128 K-scan remains pending.

The two record-exceedance figures use a separate diagnostic. They bin observed heat, cold and wind records against 2015–2021 local monthly maxima or minima, and show RMSE and bias at 24 and 240 hours.

## 3. Training-seed replication

The [seed-replication folder](03_seed_replication/README.md) contains five locally trained M1 carry/carry runs, seeds 22–26, and their SWA evaluations. Each run uses 500 warm-up plus 2,000 main-stage updates. SWA averages main checkpoints 500, 1,000, 1,500 and 2,000. Across the five runs, exact GC loss reduction averages **14.17%** over the rollout (sample SD 0.43 percentage points) and **21.03%** at day 10 (sample SD 0.75 percentage points).

Seed 22 was retrained locally. The package also contains the local seed-22 Fig. 6 controls and the raw-checkpoint comparisons requested later. The seed-22 checkpoints from the 2,250–3,000-step continuation had lower pooled and day-10 reductions than the seed-22 SWA. The original proposed reset/reset repeats at seeds 23 and 24 remain pending.

## Files

The archive contains scores, figures, configurations and provenance. Model checkpoints are identified by their original paths and SHA-256 hashes in the task folders. `SHA256SUMS` records the checksums of the files in this handoff.
