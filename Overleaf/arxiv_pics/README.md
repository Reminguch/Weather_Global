# Figures used by the arXiv manuscript

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
unchanged. Changed panels receive blue frames in the manuscript while preserving
their scientific line colors.

| Figure | v4 PDF | Analysis |
|---|---|---|
| 4 | [fig4_forecast_skill_v4_seeds.pdf](fig4_forecast_skill_v4_seeds.pdf) | Five fresh SWA training seeds: individual curves, mean and sample SD. |
| 5 | [fig5_common_grid_comparison_v4_ci.pdf](fig5_common_grid_comparison_v4_ci.pdf) | Original selected M1 and fixed public models, with paired pointwise 95% date-bootstrap intervals. |
| 6 | [fig6_memory_policies_v4_seeds.pdf](fig6_memory_policies_v4_seeds.pdf) | Original matched SWA policies and date intervals beside local raw-checkpoint replication: five carry seeds, one seed per reset control. |
| 9 | [fig9_common_grid_all_variables_v4_ci.pdf](fig9_common_grid_all_variables_v4_ci.pdf) | Figure 5 analysis extended to all 12 diagnostics. |

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

Common-grid intervals use 10,000 joint nonwrapping five-start block draws
on the original regular 122-date grid, seed 20260916. Existing ENS masks are
applied after resampling, and MSE is pooled before square roots and ratios.
Intervals are conditional on fixed checkpoints and the observed year; they
are pointwise, not simultaneous or training-seed intervals. Paired day-10
M1-versus-GC025 reductions are exported separately. No new forecasts were run.
