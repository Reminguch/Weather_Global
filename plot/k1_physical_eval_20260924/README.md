# Six-hour K=1 forecasts in physical units

> **The current training loss differs from the NeuralGCM paper.**
> Geopotential contributes **95.06%** of baseline custom loss;
> its 1–7 hPa levels alone contribute **88.81%** of the total.
> Aggregate loss improvement does not imply improvement across weather variables.
> See the [loss mismatch explanation](../README.md#loss-mismatch-and-geopotential-dominance).

**Improving upper-air results:** [geopotential at 1–30 hPa and specific humidity
at 1–5 hPa, with all twelve individual comparisons and all four configurations](../upper_air_physical_eval_20260924/README.md).

![Seven-variable comparison](r2p8_w128_di16/all_variables_global_timeseries.png)

**Three curves in every time-series panel:** ERA5 truth is black with circles,
frozen NGCM baseline is blue dashed with squares, and residual NGCM is orange
dash-dot with triangles. The main figure shows global Gaussian-area means.
Each panel names its variable, pressure level, and physical unit.

The horizontal axis is **physical time in hours from a fixed start**, with
one forecast point every **6 h**. The main window starts at
`2022-01-10T00 UTC` and covers 72 hours.
Dates are retained in the CSV metadata; the plotted ticks are 0, 6, 12, …, 72.

**K=1 semantics:** the point at offset 12 h is the six-hour forecast initialized
from ERA5 at offset 6 h. The physical state is reinitialized at each origin.
These curves are successive one-step forecasts, not a free-running 72-hour rollout.
Recurrent memory follows the production validation rule: chronological carry
with resets every 96 records and at time gaps.

The model has pressure-level temperature, including T850, but no direct T2m
output. [NeuralGCM Fig. 4a](https://arxiv.org/pdf/2311.07222#page=12) also
plots global mean temperature at **850 hPa**. Its long climate integrations
are a different forecasting protocol from the six-hour evaluations here.

## Current checkpoints

| Configuration | Epoch | Update | Evaluation job | Validation records |
| --- | ---: | ---: | --- | ---: |
| r2p8_w128_di16 | 7 | 2,968 | 14366171 | 1,455 |
| r2p8_w128_di32 | 7 | 2,968 | 14366172 | 1,455 |
| r2p8_w256_di16 | 5 | 2,120 | 14366173 | 1,455 |
| r2p8_w256_di32 | 5 | 2,120 | 14366184 | 1,455 |

All models use the frozen `20260924_feedback_v2` source and their recorded
checkpoint hashes. Checkpoint choices were frozen before physical evaluation.
This is the **2022 validation set**, not the held-out 2023 test set.

## Physical forecast errors

Global spatial RMSE uses every grid cell and all 1,455 validation forecasts,
with Gaussian latitude weights and uniform longitude weights, then takes
the square root. It is not the RMSE of the global-mean time series.
The three-day curves are illustrative; the table uses the full year.

| Variable / pressure | Unit | NGCM | w128 / di16 | w128 / di32 | w256 / di16 | w256 / di32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T850 | K | 0.48309 | 0.5937 | 0.5947 | 0.58878 | 0.58595 |
| Z500 | m²/s² | 31.375 | 35.744 | 35.569 | 36.544 | 36.701 |
| U850 | m/s | 0.93846 | 1.0104 | 1.008 | 1.0175 | 1.0242 |
| V850 | m/s | 0.99813 | 1.1907 | 1.1826 | 1.1811 | 1.1875 |
| Q700 | g/kg | 0.37904 | 0.4759 | 0.47433 | 0.46802 | 0.46894 |
| CI250 | g/kg | 0.0084433 | 0.012551 | 0.012295 | 0.011112 | 0.011096 |
| CL850 | g/kg | 0.01835 | 0.025986 | 0.026111 | 0.025558 | 0.025378 |

[Physical RMSE chart](physical_rmse.png) · [All 37 levels, all seven fields](physical_metrics_all_levels.csv)

## Why the aggregate improvement is so large

For `r2p8_w128_di16`, the custom objective falls from
1.138953 to 0.177667, a **84.40%** reduction.
Geopotential contributes **95.06%** of baseline loss.
The 1, 2, 3, 5, and 7 hPa geopotential levels alone contribute
**88.81%** of the total baseline objective.
This concentrates the headline metric on upper-atmosphere geopotential.
It does not mean that temperature, wind, humidity, or clouds improve by that percentage.
Use the per-variable physical errors above to assess forecast skill.

[Objective contributions by field and pressure](objective_contributions.csv) ·
[Source identities, checks, and complete reports](summary.json)

## Where geopotential improves

![Improved geopotential levels](r2p8_w128_di16/geopotential_improved_levels.png)

These eight levels are the levels with improved full-year geopotential RMSE
for the representative checkpoint. They were selected after the error audit,
as requested, and are explicitly labeled; the 500 hPa comparison remains above.
The same eight levels are also plotted for every other configuration.

| Pressure (hPa) | Baseline RMSE (m²/s²) | Residual RMSE (m²/s²) | RMSE reduction |
| ---: | ---: | ---: | ---: |
| 1 | 30,070.34 | 4,770.43 | 84.14% |
| 2 | 20,535.48 | 2,021.87 | 90.15% |
| 3 | 15,485.94 | 1,499.61 | 90.32% |
| 5 | 9,498.43 | 1,072.88 | 88.70% |
| 7 | 5,993.53 | 735.21 | 87.73% |
| 10 | 2,748.21 | 511.39 | 81.39% |
| 20 | 1,367.65 | 851.58 | 37.73% |
| 30 | 1,771.89 | 617.53 | 65.15% |

[Loss contributions and RMSE reductions](r2p8_w128_di16/geopotential_loss_breakdown.png)

Geopotential accounts for 19.13%
of the residual model objective, compared with 95.06% for baseline.
The stacked bars show contributions to the actual custom objective,
including its field scaling and vertical weighting; they are not physical-energy shares.

**This is not the original NeuralGCM training objective.** The original
[G.3–G.4 definition](https://arxiv.org/pdf/2311.07222#page=41) uses 24-hour
difference standard deviations and includes spectral, bias, and internal-state
loss terms. Our residual objective uses six-hour change statistics and decoded
weighted MSE. It borrows the amplitude factors (geopotential 2, humidity 0.66,
cloud species 0.05), but does not fully reproduce the original loss.
The reported 95% share is an empirical imbalance of this custom objective,
not a prescribed NeuralGCM weighting. The subsequent
[normalization audit](../loss_alignment_audit_20260924/README.md) finds that
24-hour scales alone do not remove this imbalance. These physical results
still describe checkpoints trained with the existing custom objective.

## Individual figures and data

The main figure uses `r2p8_w128_di16`. Each configuration has the same
seven individual global-mean figures and two location-specific winter/summer
comparison grids. The latter use the nearest Gaussian grid cells to New York
(40.464°N, 73.125°W) and Beijing (40.464°N, 115.312°E), not station data.
The winter and summer windows start at 2022-01-10 00 UTC and 2022-07-10 00 UTC.

| Variable | PNG | PDF | SVG |
| --- | --- | --- | --- |
| Temperature at 850 hPa | [PNG](r2p8_w128_di16/T850_global_timeseries.png) | [PDF](r2p8_w128_di16/T850_global_timeseries.pdf) | [SVG](r2p8_w128_di16/T850_global_timeseries.svg) |
| Geopotential at 500 hPa | [PNG](r2p8_w128_di16/Z500_global_timeseries.png) | [PDF](r2p8_w128_di16/Z500_global_timeseries.pdf) | [SVG](r2p8_w128_di16/Z500_global_timeseries.svg) |
| Eastward wind at 850 hPa | [PNG](r2p8_w128_di16/U850_global_timeseries.png) | [PDF](r2p8_w128_di16/U850_global_timeseries.pdf) | [SVG](r2p8_w128_di16/U850_global_timeseries.svg) |
| Northward wind at 850 hPa | [PNG](r2p8_w128_di16/V850_global_timeseries.png) | [PDF](r2p8_w128_di16/V850_global_timeseries.pdf) | [SVG](r2p8_w128_di16/V850_global_timeseries.svg) |
| Specific humidity at 700 hPa | [PNG](r2p8_w128_di16/Q700_global_timeseries.png) | [PDF](r2p8_w128_di16/Q700_global_timeseries.pdf) | [SVG](r2p8_w128_di16/Q700_global_timeseries.svg) |
| Cloud ice at 250 hPa | [PNG](r2p8_w128_di16/CI250_global_timeseries.png) | [PDF](r2p8_w128_di16/CI250_global_timeseries.pdf) | [SVG](r2p8_w128_di16/CI250_global_timeseries.svg) |
| Cloud liquid water at 850 hPa | [PNG](r2p8_w128_di16/CL850_global_timeseries.png) | [PDF](r2p8_w128_di16/CL850_global_timeseries.pdf) | [SVG](r2p8_w128_di16/CL850_global_timeseries.svg) |

| Configuration | Global overview | New York | Beijing | Time-series CSV |
| --- | --- | --- | --- | --- |
| r2p8_w128_di16 | [Figure](r2p8_w128_di16/all_variables_global_timeseries.png) | [Figure](r2p8_w128_di16/all_variables_new_york.png) | [Figure](r2p8_w128_di16/all_variables_beijing.png) | [CSV](r2p8_w128_di16_timeseries.csv) |
| r2p8_w128_di32 | [Figure](r2p8_w128_di32/all_variables_global_timeseries.png) | [Figure](r2p8_w128_di32/all_variables_new_york.png) | [Figure](r2p8_w128_di32/all_variables_beijing.png) | [CSV](r2p8_w128_di32_timeseries.csv) |
| r2p8_w256_di16 | [Figure](r2p8_w256_di16/all_variables_global_timeseries.png) | [Figure](r2p8_w256_di16/all_variables_new_york.png) | [Figure](r2p8_w256_di16/all_variables_beijing.png) | [CSV](r2p8_w256_di16_timeseries.csv) |
| r2p8_w256_di32 | [Figure](r2p8_w256_di32/all_variables_global_timeseries.png) | [Figure](r2p8_w256_di32/all_variables_new_york.png) | [Figure](r2p8_w256_di32/all_variables_beijing.png) | [CSV](r2p8_w256_di32_timeseries.csv) |

CSV time-series columns use Kelvin, m²/s² for geopotential, m/s for wind,
and g/kg for humidity/clouds. `physical_metrics_all_levels.csv` keeps the
original SI units, including kg/kg for humidity/clouds. Z500 here denotes
geopotential, not geopotential height; divide by 9.80665 to convert to metres.

## Validation and reproduction

An independent eight-date GPU preflight checked live/cache equality, production
forward-pass and recurrent-memory parity, and independent NumPy float64 error
reductions. Every complete replay independently reproduced its saved K=1 loss.
No training or backbone weights were modified. GPU jobs used `gpu-test`,
`gputest`, one A100 GPU, and a one-hour limit. Original ERA5 fields were
conservatively regridded to the common 128 × 64 Gaussian verification grid.

Redraw from the committed CSV files using Python, NumPy and Matplotlib:

```bash
python plot/plot_k1_physical.py
```

Refresh the data from completed evaluation reports:

```bash
python plot/plot_k1_physical.py --results-root logs/neuralgcm_physical_eval_20260924
```

The evaluation source is
[evaluate_neuralgcm_k1_physical.py](../../scripts/diagnostics/evaluate_neuralgcm_k1_physical.py)
and its [Slurm launcher](../../scripts/diagnostics/run_neuralgcm_k1_physical.sbatch).
