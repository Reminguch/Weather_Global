# Figures used by the arXiv manuscript

## Current v5 spectral data (November 15, 2026)

The five `_v5_matched122` assets for Figures 6 and 10–13 now use the same
122 validated starts (January 1–December 30, 2023) for GraphCast, filtered GraphCast,
M1 and R1, at all 40 six-hour leads. Compact layouts and
M1/R1 labels are retained; the removed physical-power figures remain removed.
Table 1 and both v5 sources are synchronized with these plots.
Data, source snapshots and reproduction checks are in
[`v5_spectral_matched122_20261115`](../archive/Graphs/v5_spectral_matched122_20261115/).
The earlier 36- and 70-date archives remain preserved separately.

## Original arXiv assets

These PDFs are unchanged copies of the original assets. Filenames follow the
figure numbers in `arxiv_versoin.tex`. Figure 3 is drawn directly in LaTeX.

| Figure | Current PDF | Preserved original |
|---|---|---|
| 1a | [fig1_GC_mamba.pdf](fig1_GC_mamba.pdf) | [Pics/gc_mamba_architecture/presentation/03_temporal_rollout.pdf](../archive/Pics/gc_mamba_architecture/presentation/03_temporal_rollout.pdf) |
| 1b | [fig1_improvement.pdf](fig1_improvement.pdf) | [Graphs/arxiv_2023/overview_heatmap.pdf](../archive/Graphs/arxiv_2023/overview_heatmap.pdf) |
| 1c | [fig1_channel_coverage.pdf](fig1_channel_coverage.pdf) | [Graphs/arxiv_2023/overview_coverage.pdf](../archive/Graphs/arxiv_2023/overview_coverage.pdf) |
| 2 | [fig2_correction_branch.pdf](fig2_correction_branch.pdf) | [Pics/gc_mamba_architecture/presentation/02_correction_branch.pdf](../archive/Pics/gc_mamba_architecture/presentation/02_correction_branch.pdf) |
| 4 | [fig4_forecast_skill.pdf](fig4_forecast_skill.pdf) | [Graphs/arxiv_2023/forecast_skill.pdf](../archive/Graphs/arxiv_2023/forecast_skill.pdf) |
| 5 | [fig5_common_grid_comparison.pdf](fig5_common_grid_comparison.pdf) | [Graphs/common_grid_2023_public_checkpoints_ens/fig5_public_checkpoints_2023_repaired_vs_ens.pdf](../archive/Graphs/common_grid_2023_public_checkpoints_ens/fig5_public_checkpoints_2023_repaired_vs_ens.pdf) |
| 6 | [fig6_memory_policies.pdf](fig6_memory_policies.pdf) | [Graphs/arxiv_2023/memory_policies.pdf](../archive/Graphs/arxiv_2023/memory_policies.pdf) |
| 7 | [fig7_training_horizon_closed_loop.pdf](fig7_training_horizon_closed_loop.pdf) | [Graphs/main_v3/training_ar_horizon_15yr_closedloop.pdf](../archive/Graphs/main_v3/training_ar_horizon_15yr_closedloop.pdf) |
| 8 | [fig8_training_horizon_open_loop.pdf](fig8_training_horizon_open_loop.pdf) | [Graphs/main_v2/training_ar_horizon.pdf](../archive/Graphs/main_v2/training_ar_horizon.pdf) |
| 9 | [fig9_common_grid_all_variables.pdf](fig9_common_grid_all_variables.pdf) | [Graphs/common_grid_2023_public_checkpoints_ens/fig9_public_checkpoints_2023_repaired_vs_ens.pdf](../archive/Graphs/common_grid_2023_public_checkpoints_ens/fig9_public_checkpoints_2023_repaired_vs_ens.pdf) |
| 10 | [fig10_city_forecasts.pdf](fig10_city_forecasts.pdf) | [Graphs/city_traces/cities_Paris_NYC_20220115_compact.pdf](../archive/Graphs/city_traces/cities_Paris_NYC_20220115_compact.pdf) |

Plotting scripts, source data, provenance metadata, and editable architecture
artwork remain alongside the preserved originals in `archive/`.

## ICLR v4 replacements (November 13, 2026)

These four files are used only by `v4.tex`; original arXiv assets above remain
unchanged. Panels retain their scientific line colors and use no colored
revision frames in the manuscript.

| Figure | v4 PDF | Analysis |
|---|---|---|
| 4 | [fig4_forecast_skill_v4_seeds.pdf](fig4_forecast_skill_v4_seeds.pdf) | Five fresh SWA training seeds: individual curves, mean and sample SD. |
| 5 | [fig5_common_grid_comparison_v4_ci.pdf](fig5_common_grid_comparison_v4_ci.pdf) | Original selected M1 and fixed public models, with paired pointwise 95% date-bootstrap intervals. |
| 6 | [fig6_memory_policies_v4_matched_only.pdf](fig6_memory_policies_v4_matched_only.pdf) | Carry/carry versus reset/reset: original matched SWA/date intervals beside local raw-checkpoint replication with five carry seeds and one reset seed. |
| 10 (originally 9) | [fig10_common_grid_all_variables_v4_ci.pdf](fig10_common_grid_all_variables_v4_ci.pdf) | Figure 5 analysis extended to all 12 diagnostics. |

Numerical exports, source identities, and plotting source snapshots are in
[`archive/Graphs/v4_seed_uncertainty_20261113/`](../archive/Graphs/v4_seed_uncertainty_20261113/)
and
[`archive/Graphs/v4_common_grid_uncertainty_20261113/`](../archive/Graphs/v4_common_grid_uncertainty_20261113/).
The original source data live in the Weather_global workspace, at
`results/LH_rebuttal_results/weatherdata20261111/03_seed_replication/`
and `plots/analyze_models/data/resolution_eval/public_checkpoints_2023_repaired_ens_20261018/`.
Reproduction scripts are `scripts/analyze_models/plot_v4_seed_replication.py`
and `scripts/analyze_models/plot_v4_common_grid_uncertainty.py` in that workspace.

Seed summaries are computed after pooling each run's errors; they are not
forecast ensembles or confidence intervals. The local seed 22 is a fresh run,
not the historical selected checkpoint. Figure 6 never mixes raw and SWA
weights within a panel, and its five-seed band covers carry/carry only.
Local native-grid truth repairs and historical provenance limits are recorded
in the seed summary and Appendix A.4.

The November 14 Figure 6 revision omits the carry/reset diagnostic and reuses
the retained curves and intervals without recalculation. Its source, exports
and manifest are archived in
[`archive/Graphs/v4_memory_two_policies_20261114/`](../archive/Graphs/v4_memory_two_policies_20261114/).
The earlier `fig6_memory_policies_v4_seeds.pdf` and complete original exports
remain available for provenance.

Common-grid intervals use 10,000 joint nonwrapping five-start block draws
on the original regular 122-date grid, seed 20260916. Existing ENS masks are
applied after resampling, and MSE is pooled before square roots and ratios.
Intervals are conditional on fixed checkpoints and the observed year; they
are pointwise, not simultaneous or training-seed intervals. Paired day-10
M1-versus-GC025 reductions are exported separately. No new forecasts were run.

## ICLR v4 smoothing revision (November 14, 2026)

New manuscript captions are red; scientific curve colors are preserved.
Figures use plain `\includegraphics` without colored revision frames, retaining
the latest Overleaf widths (Figures 4 and 6 at 80% text width).
Section 4.6 explains the spectral motivation, literature and output-filter
control, with day-5 and day-10 spectra in Figure 7 and the four-model percentage comparison
in Table 2. The previous Figure 8 bar chart and duplicate appendix table have
been replaced by this main-text table. Appendix E retains detailed methods,
per-field and multilead plots. Every
smoothing comparison uses the same frozen 36 initializations (1 January–16
April 2023) and all four predictors. Both memory policies retain their original
spectral amplitudes. No diagnostic uncertainty bands were added.

| v4 figure | PDF | Scope |
|---|---|---|
| 7 | [fig7_smoothing_day5_day10_v4_matched36.pdf](fig7_smoothing_day5_day10_v4_matched36.pdf) | Section 4.6: day 5 (left) and day 10 (right), amplitude ratio above coherence. |
| 8 | [fig8_training_horizon_closed_loop_v4.pdf](fig8_training_horizon_closed_loop_v4.pdf) | Original Figure 7, unchanged data and sizing. |
| 9 | [fig9_training_horizon_open_loop_v4.pdf](fig9_training_horizon_open_loop_v4.pdf) | Original Figure 8, unchanged data and sizing. |
| 10 | [fig10_common_grid_all_variables_v4_ci.pdf](fig10_common_grid_all_variables_v4_ci.pdf) | Original Figure 9, unchanged common-grid intervals. |
| 11 | [fig11_city_forecasts_v4.pdf](fig11_city_forecasts_v4.pdf) | Original Figure 10, unchanged city forecasts. |
| 12 | [fig12_smoothing_loss_v4_matched36.pdf](fig12_smoothing_loss_v4_matched36.pdf) | Exact native-grid loss over 40 leads, one panel with all four models. |
| 13 | [fig13_smoothing_feature_amplitude_v4_matched36.pdf](fig13_smoothing_feature_amplitude_v4_matched36.pdf) | Day-5 T2m/Z500/Q700/TP6h amplitudes. |
| 14 | [fig14_smoothing_feature_coherence_v4_matched36.pdf](fig14_smoothing_feature_coherence_v4_matched36.pdf) | Same fields and dates, coherence separately. |
| 15 | [fig15_smoothing_lead_spectra_v4_matched36.pdf](fig15_smoothing_lead_spectra_v4_matched36.pdf) | Six-hour/day-5/day-10 amplitude and coherence. |
| 16 | [fig16_memory_power_day5_v4_matched36.pdf](fig16_memory_power_day5_v4_matched36.pdf) | Day-5 physical power spectra, four models plus ERA5. |
| 17 | [fig17_memory_power_day10_v4_matched36.pdf](fig17_memory_power_day10_v4_matched36.pdf) | Same physical power comparison at day 10. |

The source snapshot for Table 2, Figure 7 and Figures 12–17 remains in
[`archive/Graphs/v4_appendix_e_matched36_20261114/`](../archive/Graphs/v4_appendix_e_matched36_20261114/).
The table source and figure-number mapping for the latest revision are in
[`v4_smoothing_table_20261114/`](../archive/Graphs/v4_smoothing_table_20261114/).
All renamed figures retain their original bytes and inclusion widths. The
previous Figure 8 bar chart remains available as
`fig8_accuracy_power_summary_v4_matched36.pdf` for provenance.
The fixed numerical archive itself has not been changed.
Figure 7 adds day 10 beside day 5, using the existing weighted spectral curves
on the same fixed 36 dates. Both columns use common axes for each metric and
retain all four models. Its source export, plotting script and provenance are
in [`v4_spectra_day5_day10_20261114/`](../archive/Graphs/v4_spectra_day5_day10_20261114/).
The earlier day-5-only PDF remains available for provenance.
Figure 12 removes the second loss-reduction panel while retaining the same
absolute-loss curves and inclusion width. Its source and numerical export are
in [`v4_smoothing_loss_only_20261114/`](../archive/Graphs/v4_smoothing_loss_only_20261114/).
The earlier two-panel PDF remains available for provenance.
The archive retains all 83 channels at the selected leads, every plotted curve,
the complete cohort IDs and SHA256 source/output pins. Reproduce in the parent
GraphCast environment with
`scripts/analyze_models/plot_v4_appendix_e_matched36_20261114.py`.
The source cohort is fixed explicitly; future completed dates must not overwrite
this snapshot. No forecasts, filter fitting or uncertainty resampling were run
for this manuscript update.

Historical four-date sources remain in
[`v4_appendix_e_four_models_20261114/`](../archive/Graphs/v4_appendix_e_four_models_20261114/)
and [`v4_memory_spectra_appendix_20261114/`](../archive/Graphs/v4_memory_spectra_appendix_20261114/).
The earlier 92-date filtering-only comparison and its intervals remain in
[`v4_smoothing_20261114/`](../archive/Graphs/v4_smoothing_20261114/).
Those intervals are not transferred to the current matched 36-date comparison.
Earlier figure filenames and original arXiv manuscript paths remain valid.
