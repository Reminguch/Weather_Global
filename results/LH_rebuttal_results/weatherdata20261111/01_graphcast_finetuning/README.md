# Standard-loss GraphCast fine-tuning

All three local runs started from public GraphCast-small at 1° with 13 pressure levels and updated the full model on 2015–2021. They used seed 22, batch size 4, 24-step training windows with K20 feedback, 96-step segments and full weather-feedback gradients. The checkpoints below are the native weights at update 2,500. The local optimizer, sampling order and runtime were reconstructed from the available v24 recipe. The Della training configuration and checkpoint were unavailable locally. [Configurations and checkpoint identities](manifest.json)

## Final 2023 evaluation

Positive percentages indicate lower exact GraphCast loss than the paired public GraphCast rollout. The intervals are paired 95% date-block intervals for each fixed checkpoint. GPU hours cover training and compilation. GPU memory is the JAX allocator high-water mark, and host memory is peak RSS.

| Peak LR | Rollout model loss | Rollout reduction (95% CI) | Day-10 model loss | Day-10 reduction (95% CI) | GPU hours | GPU / host GiB |
|---|---:|---:|---:|---:|---:|---:|
| `3e-7` | 10.87072 | +4.19% [3.07, 5.02] | 28.12281 | +4.79% [3.18, 5.98] | 9.44 | 50.92 / 223.31 |
| `1e-5` | 10.93935 | +3.62% [2.21, 4.58] | 27.50258 | +6.85% [4.75, 8.51] | 8.83 | 50.93 / 223.32 |
| `3e-5` | 11.68110 | −2.90% [−4.88, −1.66] | 28.83628 | +2.52% [−0.24, 4.45] | 8.82 | 50.93 / 223.32 |

The `3e-7` run is the standard-loss λ=0 reference specified in the handoff. The positive-λ full-GC configurations in the handoff use `1e-5`. [Final summaries](final_2500_summary.json) · [Full `3e-7` report](evaluations/lr3em7_step2500_2023.json) · [Full `1e-5` report](evaluations/lr1em5_step2500_2023.json) · [Full `3e-5` report](evaluations/lr3em5_step2500_2023.json)

The reports retain `baseline_per_step` and `full_per_step` for every UTC initialization under `original_graphcast_loss_per_initialization`. They use the common 122 starts and 40 six-hour leads with the exact weighted GraphCast loss. The local evaluation uses repaired native ERA5 verification fields. NCAR pressure-level fields replace stale ARCO values at 06, 12 and 18 UTC on 10 November 2023, affecting nine start–lead pairs and zero initialization frames. Each report includes an exclusion sensitivity for those nine pairs. The historical Ilya M1 SWA summary in the reports is an unpaired reference. [Initialization manifest](initializations_2023_stride3.json) · [Data and metric provenance](manifest.json)

## Intermediate checkpoints and M1 reference

The 2023 day-10 progression covers updates 500, 1,000, 1,500 and 2,000 for all rates and update 2,500 for `3e-7`. It uses a day-10-only evaluation path. The table above comes from the all-lead evaluator, which gives slightly different values for the same checkpoint. [Intermediate day-10 summary](intermediate_day10_2023_summary.json)

The equal-total figure compares the `3e-7` checkpoint at 2,500 updates with five raw M1 carry checkpoints after 500 warm-up and 2,000 main-stage updates each. Day-10 reduction is 4.79% for fine-tuned GraphCast and 20.13% for M1, with a sample SD of 0.75 percentage points across M1 seeds. [Figure (PDF)](gcft_vs_m1_seeds_2500_total_lead_time.pdf) · [Plotted values](gcft_vs_m1_seeds_2500_total_lead_time.json)

## 2022 development evidence

The Della handoff reports that the original 500-update runs at `1e-5` and `3e-5` increased 2022 development rollout loss by 13.90% and 52.25%. Their exact training configurations and checkpoints were unavailable locally. The local development run stopped at 42 of 119 starts for `3e-7` when evaluation switched to the 2023 cohort. The local 2023 milestones are final-cohort measurements. [Development provenance](manifest.json)
