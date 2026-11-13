# Non-SWA M1 and existing lambda checkpoints: fast MSE/spectrum pilot

**November 12 correction:** full-GC peak-LR `1e-5` runs are excluded from
scientific comparisons by explicit user instruction. Their original sealed
measurements remain archival evidence; use only the full-GC `3e-7` lambda0
result here. See the [comparison policy](v24_full_gc_comparison_policy_2026-11-12.md).

Updated: **2026-11-11, America/New_York**. Status: both pilots completed and inspected.
This is an evaluation of saved checkpoints. The separate four-fresh-run
full-GC AR-horizon training campaign remains on hold.

## Completed pilot results — November11

All20 unique model evaluations and both reports completed. Each model used
four matched2022 starts and40 six-hour leads. Checkpoint/source hashes and
coverage were verified from the sealed reports; the historical report reuses
the identical M1 raw/SWA outputs. These are four-date diagnostics without
confidence intervals, not annual or training-seed conclusions.

**Removing SWA worsens both M1's MSE and aggregate spherical power mismatch
on this pilot.** SWA lowers MSE2.677% relative to raw M1; it does not eliminate
the power mismatch relative to ERA5.

| Reference | Exact rollout MSE | Reduction vs public GC | Spherical power error | Day10 MSE |
| --- | ---: | ---: | ---: | ---: |
| Public GC | 11.133143 | 0% | 61.571% | 28.725877 |
| M1 raw main-step2000 | 9.424010 | 15.352% | 75.943% | 22.180490 |
| M1 SWA500–2000 | 9.171727 | 17.618% | 71.886% | 21.857055 |

Fresh models, all raw step1250:

| λ | GC–Mamba MSE | GC–Mamba power error | Full-GC MSE | Full-GC power error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.8511 | 64.321% | 10.6106 | 63.972% |
| 0.25 | 10.9865 | 63.440% | Excluded: LR1e-5 | Excluded |
| 0.5 | 11.0667 | 62.843% | Excluded: LR1e-5 | Excluded |
| 0.75 | 11.0064 | 63.112% | Excluded: LR1e-5 | Excluded |
| 1 | 11.0951 | 64.801% | Excluded: LR1e-5 | Excluded |

Fresh GC–Mamba has substantially higher MSE than M1 at this interim endpoint,
despite lower aggregate power mismatch. Full-GC λ0 improves MSE4.693% versus
public GC. Positive-lambda full-GC results were excluded on November12; they
cannot establish a scientific lambda comparison. No positive-lambda full-GC
result at the accepted3e-7 rate is available. Training completion and scientific
improvement remain separate; this documentation change does not alter a
training schedule.

Historical raw endpoints, with each training history kept separate:

| Group/model | Exact rollout MSE | Spherical power error |
| --- | ---: | ---: |
| September24 MSE control, step2000 | 9.3796 | 74.865% |
| September24 λ0.1, step2000 | 9.5805 | 76.996% |
| September19 MSE control, step2000 | 9.4363 | 76.267% |
| September19 λ1, step2000 | 10.7817 | 76.776% |
| October exact-grid MSE control, step500 | 9.1853 | 73.093% |
| October spectral λ0, step500 | 9.1872 | 73.598% |
| October λ0.25, step500 | 9.2605 | 72.010% |
| October λ0.5, step500 | 9.4624 | 71.015% |

Octoberλ0.25 improves both MSE and power mismatch relative to **raw** M1;
M1 SWA remains slightly better on both. Octoberλ0.5 improves power matching
further, with higher MSE. Septemberλ0.1/λ1 do not improve the spherical
aggregate relative to their controls here. This differs from the cached
eight2023-start **zonal** result below: dates, geographic domain, channels and
spectral operator differ. Do not transfer the earlier zonal conclusion to
all-channel spherical fidelity.

**Numerical diagnostics:** repeated public-GC rollout loss differs by up to
0.0793851 (0.713% of canonical GC loss); day10 bounds from observed repeats are
0.471028 fresh and0.294249 historical. Fresh GC–Mamba λ0.5/λ1 rollout decreases
are smaller than that observed baseline spread. M1's raw→SWA MSE difference
0.252283 exceeds it. These are warnings and descriptive comparisons against
baseline repeats, not full model-repeatability estimates or acceptance gates.
All values are finite;22 fresh and18 historical reference warnings are retained,
with no excluded zero-truth-power bins.

Reports and figures:

- [Current eligible comparison table and figures](../../plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110_eligible_20261112_v2/report.md): full-GC3e-7 lambda0, GC–Mamba and public-GC/M1 references.
- [Current eligible MSE/power tradeoff](../../plots/analyze_models/images/resolution_eval/v24_raw_checkpoint_pilot_20261110_eligible_20261112_v2/mse_power_tradeoff.png).
- [Archival fresh table and figures — includes excluded full-GC runs; do not use as a comparison](../../plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110/report.md).
- [Historical complete table and all figures](../../plots/analyze_models/data/resolution_eval/v24_legacy_raw_checkpoint_pilot_20261110/report.md).
- [Archival fresh MSE/power tradeoff — superseded for comparisons](../../plots/analyze_models/images/resolution_eval/v24_raw_checkpoint_pilot_20261110/mse_power_tradeoff.png).
- [M1/fresh GC–Mamba power spectra](../../plots/analyze_models/images/resolution_eval/v24_raw_checkpoint_pilot_20261110/power_spectrum_gc_mamba.png)
  and [power ratios](../../plots/analyze_models/images/resolution_eval/v24_raw_checkpoint_pilot_20261110/power_ratio_gc_mamba.png).
- [M1/October power spectra](../../plots/analyze_models/images/resolution_eval/v24_legacy_raw_checkpoint_pilot_20261110/power_spectrum_october_500_update_continuation.png).

Observed resource use:14m44s–15m46s per evaluation,8.75GiB maximum host RSS,
5.064 summed GPU-hours for20 tasks. CPU reports completed in49/52seconds.
Jobs17249273/17249274 and17250938/17250939 all completed with exit0.

## Original checkpoint choices and protocol

Use principal M1's final raw
[`checkpoint_step00002000.pkl`](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/checkpoints/checkpoint_step00002000.pkl):
2,000 main-stage updates after 500 warm-up updates, or 2,500 updates total.
It is the predetermined final endpoint, not a checkpoint selected by this pilot.
Include the historical SWA500–2000 as a separate reference.

All ten selected fresh scalar-lambda arms have completed their first training
part through update **1,250**. Their update-2,500 continuations were pending at
inspection. Use `checkpoints/checkpoint_step00001250.pkl` for every arm, so the
pilot need not wait for training. These lambda checkpoints are already non-SWA.

| Family | λ | Run directory under the fresh campaign's `runs/` |
| --- | ---: | --- |
| GC–Mamba | 0 | `gcm_s22_l000_lr0` |
| GC–Mamba | 0.25 | `gcm_s22_l025` |
| GC–Mamba | 0.5 | `gcm_s22_l050_profile` |
| GC–Mamba | 0.75 | `gcm_s22_l075` |
| GC–Mamba | 1 | `gcm_s22_l100` |
| Full GC | 0 | `gcft_s22_l000_lr3em7_fresh_20261107` |
| Full GC | 0.25 | `gcft_s22_l025` |
| Full GC | 0.5 | `gcft_s22_l050_profile` |
| Full GC | 0.75 | `gcft_s22_l075` |
| Full GC | 1 | `gcft_s22_l100` |

The [fresh training campaign](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107)
records the actual configurations. GC–Mamba uses peak LR3e-5 throughout.
**Full GC λ0 uses peak LR3e-7, while positive λ uses1e-5; this is not a fixed-LR
lambda-only comparison.** The user subsequently clarified **both sets**:
the older September/October AMSE continuations are included in a separate
extension below, with their objective/training histories labeled separately.

## Cohort and figures

Take the earliest available development32 initialization in each calendar
quarter: **2022-01-02 00Z, April3 06Z, July7 18Z, October7 06Z**. Every model
forecasts the same40 six-hour leads, with cold zero Mamba state and full weather
feedback. There is no output smoothing. Four starts provide a quick diagnostic,
not a replacement for the planned32/122-start development/annual evaluations.

The report produces:

- Exact original GraphCast weighted MSE by forecast lead, plus rollout and day10
  tables and reductions against one common public-GC denominator.
- Physical spherical forecast power and ERA5 power ratios for T2m, Z500 and U850
  at days5 and10, split into GC–Mamba/full-GC panels with M1 raw/SWA references.
- Rollout/day10 MSE-versus-power-mismatch plots. Powers are pooled over dates
  before computing `100 * abs(Pforecast / Ptruth - 1)`, then averaged equally
  across83 channels, degrees1–179 and the reported leads with a shared valid mask.
- All83 channels/40 leads of forecast, truth, error and cross power in per-date
  and pooled NPZ artifacts. Finite reference differences retain values/bounds
  and warnings; whether loss decreases exceed observed baseline repeat spread
  is reported separately. Nonfinite values and integrity failures are errors.

## Reused M1 evidence

These are existing **32-start2022** results, not new four-start pilot results.

| M1 checkpoint | Exact rollout MSE | Paired public-GC MSE | Paired rollout reduction |
| --- | ---: | ---: | ---: |
| Raw main step2000 | 9.167232 | 10.818541 | 15.2637% |
| SWA500–2000 | 9.064574 | 10.835462 | 16.3434% |

Source: [raw result](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/step2000.json)
and [SWA result](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json).
The finite baseline difference is a numerical warning; these paired reductions
use their respective saved denominators. The new pilot uses a common denominator.

## Execution and artifacts

[Pilot campaign](../../artifacts/evaluations/v24_raw_checkpoint_pilot_20261110),
[driver](../../scripts/experiments/run_v24_raw_checkpoint_pilot.py),
[Slurm wrapper](../../scripts/experiments/v24_raw_checkpoint_pilot.slurm), and
[reporter](../../scripts/analyze_models/report_v24_raw_checkpoint_pilot.py).
The sealed manifest pins all checkpoint/config hashes, cohort and an immutable
runtime snapshot; the Git revision alone does not identify this dirty workspace.

Twelve unthrottled array elements each request one80GB GPU,4CPUs,16GiB RAM and
62minutes. The62-minute allocation avoids the documented ≤1-hour gputest
routing restriction; expected execution is roughly10–20minutes per model,
excluding queue time. Job17022624 measured9m57s/9.17GiB for one start with more
expensive smoothing statistics;17024230_1 measured33m03s/9.88GiB for11 starts.
A dependent CPU report requests2CPUs/8GiB/15minutes. Focused driver checks:
12 passed; synthetic report rendering and metadata validation passed.

Result destinations, populated after evaluation:
`plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110/`
(`report.md`, `report.json`, `summary.csv`, `lead_curves.csv`) and the matching
`plots/analyze_models/images/resolution_eval/v24_raw_checkpoint_pilot_20261110/`
directory (MSE curves, power spectra/ratios and tradeoff, PNG/PDF).

Submission records and exact commands live under the campaign's `submissions/`.
Do not infer completion from submission or regenerate the frozen source.

Submitted at13:52 Eastern: evaluation array **17249273**, dependent report
**17249274**. Initial queue check showed both pending. Monitor with
`squeue -j 17249273,17249274`.
Frozen runtime digest:
`8203142838c3c60027df2f816a15ab11cb1da0cfb1e57f2e23ec2fe09d1b5324`.

## Both sets: historical AMSE raw checkpoints

The follow-up answer requests both the latest fresh sweep and older AMSE runs.
The fresh campaign/snapshot/jobs above are preserved. The historical extension
uses the **same four2022 starts and40leads**, reusing the pending M1 raw/SWA
evaluations rather than forecasting them again. Eight additional models:

| Training history | Unaveraged endpoints | Saved update |
| --- | --- | ---: |
| September24 | λ0.1 and its exact-grid MSE control | 2000 |
| September19 | λ1 and its exact-grid MSE control | 2000 |
| October4 continuation | λ0 (spectral MSE), λ0.25, λ0.5, and exact-grid MSE control | 500 |

The September runs use the principal M1 warm-up initializer. October starts
from September24's MSE **SWA**, although the evaluated October endpoints are
raw/unaveraged. Spectral λ0 and the exact-grid MSE control are distinct
objectives. These histories are not one controlled five-point lambda sweep,
and their spectral interpolation objective differs from the fresh campaign.

Campaign: `artifacts/evaluations/v24_legacy_raw_checkpoint_pilot_20261110/`.
It inherits the original pilot's frozen computational code and overlays only
the preparation/evaluation orchestration and reporting. Figures are separated
by training history. Only eight new GPU tasks are needed, each with the same
1GPU/4CPU/16GiB/62min allocation; report waits for those plus the two original
M1 tasks. Its report/image destinations use the same name under the respective
resolution_eval directories.

Submitted historical extension: **17250938 →17250939**. The report additionally
depends on original M1 tasks17249273_0 and17249273_1. Frozen source digest:
`5c714cfb5e73c34f5c43a32ce93c3cfcff5892a46b070c4810b01e27cb9061c5`.
Validation:15 driver/preparation tests and6 report-reuse tests passed; separate
history plot layouts were rendered and inspected. Monitor both campaigns with
`squeue -j 17249273,17249274,17250938,17250939`.

### Immediately available cached results

[Historical raw MSE/spectrum report](../../plots/analyze_models/data/resolution_eval/amse_lambda_raw_cached_20261110/report.md)
and [overview figure](../../plots/analyze_models/images/resolution_eval/amse_lambda_raw_cached_20261110/mse_and_power_overview.png)
reuse **eight identical2023 starts/40leads**, with no new inference. All eight
model endpoints are raw. MSE and spectra use the same forecasts. These are
**zonal Fourier powers over nine fields at30–60° latitude in both hemispheres**,
not the new all-channel spherical pilot. Existing32-start2022 exact losses are
saved separately and are not mixed into these figures.

| Group/model | Exact rollout MSE | Zonal power mismatch |
| --- | ---: | ---: |
| October spectral λ0 | 9.7746 | 56.813% |
| October λ0.25 | 9.8881 | 55.090% |
| October λ0.5 | 10.1328 | 53.391% |
| September24 MSE control | 9.9584 | 57.245% |
| September24 λ0.1 | 10.0607 | 55.327% |
| September19 MSE control | 9.9148 | 57.211% |
| September19 λ1 | 12.2584 | 27.575% |

Within each history, positive λ improves the aggregate power mismatch while
increasing exact MSE relative to its control. September λ1 also worsens MSE
versus its public-GC reference11.2122. The table is descriptive eight-date
evidence, not annual or training-seed robustness. Provenance records all
checkpoint/forecast hashes, finite reference differences, bounds and whether
loss decreases exceed observed repeated-baseline spread.
