# Day-5 and day-10 spectra in Figure 7

Figure 7 uses two columns (day 5 and day 10) and two rows (amplitude ratio
and coherence). Each panel includes raw GraphCast, GraphCast plus its
development-fitted optimal output filter, carry/carry (M1), and reset/reset.
The columns use matching axis limits for each metric. The original curve
colors and complementary coherence dashes for raw and filtered GraphCast
are retained, without markers or overlap annotations.

The figure reads the existing `weighted_spectral_curves.csv` from the frozen
36-date archive `../v4_appendix_e_matched36_20261114/`. It uses the 120-hour
and 240-hour curves, each with total wavenumbers 1–179. All model comparisons
use the same 36 initialization dates, 1 January–16 April 2023. No forecasts,
filters, metrics or uncertainty estimates were recalculated.

The CSV is a byte-identical source copy; `manifest.json` pins the data,
plotting script and figure outputs. The original day-5-only figure and all
previous numerical archives remain unchanged. The active manuscript asset is
`arxiv_pics/fig7_smoothing_day5_day10_v4_matched36.pdf`, included at full text
width. The manuscript references and caption were updated in red.

To reproduce with NumPy and Matplotlib, run from this directory:

```sh
python plot_v4_spectra_day5_day10_20261114.py --data weighted_spectral_curves.csv --output /tmp/spectra-day5-day10
```
