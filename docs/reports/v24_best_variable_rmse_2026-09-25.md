# Best-model RMSE by forecast lead — 2026-09-25

The requested RMSEs are already calculated. No new evaluation is needed for the saved 32-start 2022 development set or the eight-start 2023 pilot.

Model: 1° v24 `di16_fast_mamba`, Mamba1/BCG1, residual width512/MP2, threefold Mamba learning rate, SWA updates 500/1000/1500/2000. This is the selected model in the current paper figures and the September 24 recipe comparison. Best means highest observed aggregate exact GraphCast loss reduction in the compared completed recipes, not proven superiority or best at every variable/lead. Its 16.3434% aggregate loss reduction is close to di32 joint SWA (16.2940%).

Both evaluations use 40 six-hour leads, native 1° verification, cold full-feedback rollouts, zero initial recurrent memory, state carry thereafter, and their own paired frozen GraphCast-small baseline. The 2022 starts were used for model selection. The 2023 pilot has eight preselected dates: January 15, February 15, April 15, May 15, July 15, August 15, October 15 and November 15, all at 00 UTC. These datasets are reported separately; neither has variable-level confidence intervals here.

RMSE reduction (%) = `100 × (1 − model RMSE / baseline RMSE)`. Positive values mean lower error; negative values mean degradation. RMSE pools squared errors across starts before taking the square root and uses normalized cosine-latitude weights. Upper-air group results additionally pool squared errors equally across all 13 pressure levels; they are not pressure-weighted or a single standard-level score. Exact GraphCast loss uses a different weighting/normalization and is reported separately.

**2022 development · 32 starts: RMSE reduction (%)**

| Variable | Day 1 | Day 3 | Day 5 | Day 7 | Day 10 |
|---|---:|---:|---:|---:|---:|
| 2 m temperature | -0.14 | -0.19 | 2.96 | 6.32 | 10.98 |
| Mean sea-level pressure | 0.33 | -0.08 | 3.87 | 8.80 | 12.42 |
| 10 m zonal wind (u) | -0.03 | 0.12 | 5.56 | 11.37 | 17.06 |
| 10 m meridional wind (v) | -0.26 | 0.33 | 5.78 | 11.90 | 15.18 |
| 6 h precipitation | -0.11 | 0.87 | 4.54 | 8.66 | 13.44 |
| Geopotential | -0.57 | -0.53 | 2.23 | 6.16 | 10.21 |
| Air temperature | 0.05 | 0.08 | 2.59 | 7.24 | 11.22 |
| Specific humidity | -0.03 | 1.07 | 4.64 | 8.48 | 12.83 |
| Zonal wind (u) | -0.15 | -0.31 | 3.28 | 8.60 | 13.35 |
| Meridional wind (v) | -0.14 | -0.25 | 3.69 | 8.75 | 13.66 |
| Pressure vertical velocity | -0.21 | 0.09 | 3.46 | 7.67 | 11.60 |

All 11 variable groups improve at day 10; 82/83 individual channels improve. Day-10 exceptions: `geopotential_level50` (-1.62% RMSE reduction).
Exact original GraphCast loss reduction, a separate metric: 16.3434% over all leads and 23.6831% at day 10.

**2023 pilot · 8 starts: RMSE reduction (%)**

| Variable | Day 1 | Day 3 | Day 5 | Day 7 | Day 10 |
|---|---:|---:|---:|---:|---:|
| 2 m temperature | 0.20 | 0.82 | 2.70 | 6.79 | 12.66 |
| Mean sea-level pressure | 0.59 | 0.54 | 1.79 | 7.06 | 7.95 |
| 10 m zonal wind (u) | 0.33 | -0.01 | 4.07 | 10.41 | 12.14 |
| 10 m meridional wind (v) | -0.09 | -0.13 | 4.01 | 9.97 | 12.30 |
| 6 h precipitation | 0.02 | 0.79 | 3.22 | 5.91 | 9.30 |
| Geopotential | 0.17 | 0.04 | 0.35 | 4.87 | 11.29 |
| Air temperature | 0.14 | 0.09 | 1.50 | 4.36 | 10.32 |
| Specific humidity | 0.27 | 1.00 | 4.11 | 8.05 | 11.72 |
| Zonal wind (u) | 0.05 | -0.18 | 1.94 | 6.42 | 11.65 |
| Meridional wind (v) | -0.03 | -0.29 | 1.94 | 6.55 | 11.20 |
| Pressure vertical velocity | 0.04 | -0.03 | 2.05 | 4.77 | 6.43 |

All 11 variable groups improve at day 10; 82/83 individual channels improve. Day-10 exceptions: `temperature_level50` (-4.99% RMSE reduction).
Exact original GraphCast loss reduction, a separate metric: 13.7640% over all leads and 20.4459% at day 10.

**Absolute day-10 RMSE**

| Variable | Unit | 2022 baseline | 2022 model | 2023 baseline | 2023 model |
|---|---|---:|---:|---:|---:|
| 2 m temperature | K | 2.568 | 2.286 | 2.7 | 2.359 |
| Mean sea-level pressure | hPa | 7.161 | 6.271 | 6.971 | 6.417 |
| 10 m zonal wind (u) | m/s | 4.307 | 3.572 | 4.231 | 3.718 |
| 10 m meridional wind (v) | m/s | 4.478 | 3.798 | 4.498 | 3.945 |
| 6 h precipitation | mm | 2.355 | 2.038 | 2.649 | 2.402 |
| Geopotential | m²/s² | 774.7 | 695.7 | 786.9 | 698 |
| Air temperature | K | 3.104 | 2.756 | 3.079 | 2.762 |
| Specific humidity | g/kg | 1.076 | 0.9376 | 1.079 | 0.9524 |
| Zonal wind (u) | m/s | 8.692 | 7.531 | 8.612 | 7.609 |
| Meridional wind (v) | m/s | 9.059 | 7.821 | 8.94 | 7.938 |
| Pressure vertical velocity | Pa/s | 0.154 | 0.1361 | 0.1744 | 0.1632 |

**Individual standard-level channels: RMSE reduction (%)**

| Channel | 2022 day 5 | 2022 day 10 | 2023 day 5 | 2023 day 10 |
|---|---:|---:|---:|---:|
| geopotential_level500 | 3.16 | 11.93 | 0.90 | 11.13 |
| temperature_level850 | 3.07 | 12.25 | 2.00 | 11.96 |
| specific_humidity_level700 | 4.59 | 13.23 | 3.21 | 10.63 |
| u_component_of_wind_level850 | 4.70 | 16.28 | 2.74 | 11.89 |
| v_component_of_wind_level850 | 4.94 | 14.55 | 3.22 | 11.39 |
| vertical_velocity_level500 | 3.31 | 11.98 | 2.19 | 7.26 |

**Evaluation still needed for full-year confirmation**

The separate 365-start 2023 campaign is incomplete. Slurm accounting checked on September 25 shows preflight tasks `14077246_0–2` timed out (1:02 limits; elapsed about 1:07). Gate `14077255`, production array `14077262_[0–68]`, and report `14077269` were cancelled. Each saved preflight partial JSON has zero completed starts. There is no completed full-year result in that campaign directory and these jobs are no longer queued. Recover the preflight, validate runtime/numerical settings, and rerun the dependent campaign to obtain full-year evidence. This report did not submit or modify jobs.

**Saved outputs**

- [All-variable RMSE-reduction curves (PDF)](../../plots/analyze_models/images/resolution_eval/v24_best_variable_rmse_20260925/rmse_reduction_vs_lead.pdf)
- [Daily RMSE-reduction heatmap (PDF)](../../plots/analyze_models/images/resolution_eval/v24_best_variable_rmse_20260925/daily_rmse_reduction.pdf)
- [Absolute RMSE curves (PDF)](../../plots/analyze_models/images/resolution_eval/v24_best_variable_rmse_20260925/absolute_rmse_vs_lead.pdf)
- [11 variables × 40 leads × 2 datasets (CSV)](../../plots/analyze_models/data/resolution_eval/v24_best_variable_rmse_20260925/per_variable_all_leads.csv)
- [83 channels × 40 leads × 2 datasets (CSV)](../../plots/analyze_models/data/resolution_eval/v24_best_variable_rmse_20260925/per_channel_all_leads.csv)
- [Integer-day table (CSV)](../../plots/analyze_models/data/resolution_eval/v24_best_variable_rmse_20260925/per_variable_integer_days.csv)
- [Improved variable/channel counts (CSV)](../../plots/analyze_models/data/resolution_eval/v24_best_variable_rmse_20260925/improved_counts.csv)
- [Source hashes, checkpoint identity, protocol and exact losses](../../plots/analyze_models/data/resolution_eval/v24_best_variable_rmse_20260925/sources.json)

Validation: checked complete 32/8-start outputs, matching checkpoint and baseline identities, checkpoint SHA-256 against the frozen 2023 protocol, all 7,520 variable/channel lead reductions against paired RMSEs, and group RMSEs against equally pooled pressure-level MSEs. Source evaluation files are unchanged.

Reproduce: `source scripts/graphcast_env.sh && python scripts/analyze_models/report_v24_best_variable_rmse.py`.

Sources: [development_2022_n32](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json); [pilot_2023_n8](../../plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/models/M1/evaluation.json); [September 24 recipe selection](../experiments/v24_res1_ar_sweep_2026-09-24.md).
