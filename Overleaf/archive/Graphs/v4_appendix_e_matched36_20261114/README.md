# Appendix E: fixed 36-date diagnostic snapshot

This archive supplies the revised ICLR v4 Appendix E, Table 7 and Figures 11–18.
It preserves 36 common 00 UTC initializations, every third day from 1 January
through 16 April 2023. It is an interim January–April diagnostic, separate from
the principal 122-date evaluation. Future completion of the larger spectral
campaign must produce a separate archive; do not overwrite this snapshot.

Every presented comparison includes raw GraphCast, development-filtered
GraphCast, original carry/carry (M1) and matched reset/reset. Both correction
models are unfiltered and have not been amplitude matched. No filtered M1,
carry/reset diagnostic, new filters, forecasts or uncertainty estimates enter
this revision. The existing GraphCast filter was fitted on 32 dates in 2022.

`data/comparison_scores.csv` contains all four models for the rollout, day 5
and day 10, including exact GraphCast-weighted MSE, absolute power mismatch,
and the separate percentage reductions relative to raw GC. Positive reduction
means improvement; these two percentages measure different quantities.
`table7_rollout_scores.csv` is a convenience rollout-only subset; the revised
Table 7 uses all three windows from `comparison_scores.csv`.

`loss_by_lead.csv`, `weighted_spectral_curves.csv`, `feature_day5_curves.csv`
and `physical_power_selected_fields.csv` preserve every plotted numerical
curve. `all_channels_selected_leads_curves.npz` retains all 83 channels at
6, 120 and 240 hours, including original forecast/truth powers, amplitudes and
coherences. The JSON metadata inside it defines array axes, channels and IDs.
The accompanying CSV includes both equal-degree and truth-power-weighted
mismatch scores. `summary.json` records numerical repeatability diagnostics;
finite rerun drift remains a warning, not a rejection criterion.

`sources/` preserves exact plotting helpers, the pooling script, cohort and
source metadata. `source_locations.json` maps workspace originals to archive
copies and SHA256 pins (the filter JSON is losslessly gzip-compressed). Full 40-lead pooled moments remain preserved under
`artifacts/evaluations/v24_matched_spectra_completed_20261114/` in the parent
Weather_global workspace; their hashes are pinned, and that pooling script
reuses its saved cohort instead of discovering subsequently completed dates.
The distinct full-campaign report cannot replace these files silently.

Reproduce in the parent GraphCast environment with
`scripts/analyze_models/plot_v4_appendix_e_matched36_20261114.py`. The archived
scripts preserve workspace source paths; the exported curves above are also
portable numerical sources for rebuilding the figures without new forecasts.
Figure 16 is the compact two-panel percentage summary; Figures 17–18 retain
physical field units at days 5 and 10. Active PDFs are in `arxiv_pics/`.
`archive_manifest.json` pins this archive and those active figure PDFs.

Interpretation: carry/carry has 2.84% lower day-5 MSE than reset/reset with
68.66% versus 68.16% power mismatch. Its 10.20% day-10 advantage accompanies
84.47% versus 75.74% mismatch. This supports a lead-dependent accuracy benefit,
not identical spectral fidelity or uniform dominance of reset/reset. Carry
also outperforms the tested filtered-GC baseline on both aggregate scores.
