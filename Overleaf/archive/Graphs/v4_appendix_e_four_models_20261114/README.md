# Appendix E: four models on a common cohort

Every active Appendix E table/figure includes raw GraphCast, development-filtered
GraphCast, raw carry/carry (M1) and matched raw reset/reset. Only four forecast
dates currently have completed spectra for all models: 2023-01-01, 2023-04-01,
2023-07-03 and 2023-10-01 at 00 UTC. All 40 six-hour leads are available. The
GraphCast filter was fitted on 32 development starts in 2022; memory-policy outputs
are unchanged and unfiltered. Filtered M1 is not a comparator.

The five new figure PDFs supply Figures 11–15. Existing Figures 16–18 use the same
four dates and models and are reused unchanged. Table 7 contains exactly four
rows from this common cohort. All scores are fixed-checkpoint point estimates;
no four-date bootstrap or training-seed uncertainty is estimated. The principal
122-date memory comparison and its uncertainty are unchanged. Earlier 92-date
filtering results/intervals remain in `../v4_smoothing_20261114/` and are not
combined with this four-date sample.

Exports cover exact losses, weighted amplitude/coherence, all 83 channels at 6 hours,
day 5 and day 10, selected-field day-5 curves and Table 7 scores. The manifest pins
raw sources, filters, scripts and results; numerical repeat differences are
preserved. `archive_index.json` pins files relative to this portable archive.

Reproduce from the parent Weather_global workspace in its GraphCast environment:
`scripts/analyze_models/plot_v4_appendix_e_four_models_20261114.py`.
It imports `plot_v4_smoothing_paper_20261114.py`, copied alongside here. No new
training, forecasts, filter fitting or bootstrap draws were performed.
Future extensions must retain all four models on an identical completed cohort.

Table 7 now expresses both diagnostics as percentage reductions relative to raw
GraphCast, using `100 * (1 - model_score / raw_GraphCast_score)` separately for
loss and power mismatch. Positive values indicate improvement; negative values
indicate deterioration. The percentages describe distinct metrics. Current
table entries are in `table7_relative_to_graphcast.csv`; its JSON companion
records the exact formula and source hash. `table7_accuracy_power.csv` and the
original manifest retain the underlying absolute scores unchanged.
