# Section 4.6: table and spectral motivation

The user requested replacing the four-model percentage bar chart (previous
Figure 8) with a table and explaining why spectral fidelity is an additional
important measure alongside MSE. The opening now motivates the output-smoothing
control through the possible loss of spatial detail when MSE improves.

The existing Appendix E percentage table is now Table 2 in Section 4.6. It
replaces the bar chart without duplicating the table in the appendix. It has
four model rows and paired MSE/power-mismatch reductions for the rollout,
day 5 and day 10. Both columns explicitly denote reductions relative to raw GraphCast; zero is
baseline performance, not zero error. Negative spectral reductions indicate
worse power matching. All 24 numerical cells are unchanged. `table_source.tex`
preserves its LaTeX; `comparison_scores.csv` is the fixed numerical source.

The 36-date snapshot in `../v4_appendix_e_matched36_20261114/` is unchanged.
No forecasts, filters, data pools or uncertainty calculations were rerun.
`figure_mapping.json` records byte-identical copies for the figure numbers
following removal of the bar chart. The original PDFs, including that chart,
remain available, and the original arXiv manuscript is unchanged.
