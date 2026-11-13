# City forecast examples

The four original PDFs remain available here as archived assets.
`main_v2.tex` and `main_v3.tex` use one compact appendix figure,
`cities_Paris_NYC_20220115_compact.pdf`: Paris and New York City on January 15,
with rows for 2 m temperature, 850 hPa zonal wind, and six-hour precipitation.
This reduces the displayed city examples from 60 panels to six. Four of the
six panels have lower corrected MAE over 120–240 h; Paris temperature and
New York precipitation have higher MAE. The curves and unit conversions
are unchanged from the corresponding original panels.

Regenerate the compact figure with `generate_compact.py --npz <export.npz>`.
The adjacent `.sources.json` records the input hash and per-panel MAEs.
The compact figure uses `../paper_style.py` on its 6.5-inch export canvas:
9-point axis labels, ticks and legend text, and 11-point titles.
See `../FIGURE_TYPOGRAPHY.md` for sizes at the ICLR manuscript's display width.
The 5.4-inch height, wrapped precipitation label, and three-day date ticks
give the larger text room without changing any forecast trajectory.

The original four figures show independent ten-day forecasts from 00 UTC on 15 January, April,
July, and October 2022 at the nearest 1-degree grid points to Paris, New York
City, and London. The correction is trained separately on ERA5 2007–2021
(K=20, Mamba inner dimension 128, SWA at updates 6k–11k). This is a different
checkpoint from the seven-year correction used for the main v3 results.

The export records `mode=cold_full_two_branch_zero_state`: both forecasts
start from ERA5 at -6 h and 0 h, follow their own predicted fields, and the
correction starts with zero memory. The dotted line is at 0 h; one input
state is before it and the other lies on it. Precipitation is not clipped.

## Caption audit

`mae_120_240h.csv` records the mean absolute error against ERA5 over the 21
six-hour forecast leads from 120 through 240 h, inclusive. This diagnostic
uses individual grid points and is distinct from spatially aggregated RMSE.
Temperature MAE has the same numerical value in K and degrees Celsius.
Precipitation errors are converted from metres to millimetres.

| Start | Lower corrected MAE | Higher corrected MAE | Exact ties |
| --- | ---: | ---: | ---: |
| 2022-01-15 | 11 | 4 | 0 |
| 2022-04-15 | 9 | 6 | 0 |
| 2022-07-15 | 12 | 3 | 0 |
| 2022-10-15 | 13 | 2 | 0 |

April's Paris 10 m meridional-wind MAEs are 2.00200314926 m/s (GraphCast)
and 2.00130918365 m/s (corrected). Both round to 2.00 m/s, yielding eight
improvements, six deteriorations, and one tie at two decimal places.
The original April caption counted full-precision improvements and explained this rounding.

## Provenance

- Export: `Weather_Global_experiments/results/2026-09-25-headline-figures/cities_allvars_S128_K20_15yr_SWA6k-11k_15th.npz`
- SHA-256: `6eab01e6d237453d87b84327039a022776497e859f673a2851b6917bec7afce8`
- Plot generator: `plot_cities.py` in the same source directory.
- Checkpoint recorded by the export: `/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res1_sizeladder_15yr/ladder_15yr_S128_h128_di128_ds16/v13_residual_SWA_step6k-11k.pkl`
- The four paper PDFs preserve the source curves. Approved terminology changes
  to their labels are recorded in `../../tools/figure_label_changes.json`;
  the original PDFs are retained in `../../tools/figure_label_sources/`.

For each city and variable, select 850 hPa where applicable and compute
`mean(abs(prediction - truth))` across those 21 leads. Compare the original
errors for counts; rounding is applied only to the two-decimal display columns.
