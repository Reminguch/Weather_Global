# Main v3

`main_v3.tex` uses the seven-year correction-training setup (ERA5 2015–2021)
as its reference. The pretrained GraphCast-small backbone remains frozen and
is the forecast baseline. Build from the repository root with
`bash build_v3.sh`; the clean PDF is `main_v3.pdf`, and
auxiliary files and logs go to `build/main_v3/`.

The selected V24 checkpoint supplies the main exact-loss and T2m RMSE figure,
the variable-group heatmap, and the channel-coverage curve on the same 32
2022 development starts. Figure 1 combines the temporal rollout with compact
heatmap and channel-coverage panels; the correction-branch diagram appears
separately in Methods.

Sections 4.2--4.3 retain the headline loss reductions, one physical-unit
temperature example, the variable-group range, and the day-ten channel
exception. Appendix "RMSE by variable group" provides the 11-variable table:
absolute day-ten RMSEs and RMSE reductions at days five, seven, and ten.
Values come from `Graphs/main_v3/breadth_7yr_all_leads.csv`; humidity,
sea-level pressure, and precipitation are displayed in g/kg, hPa, and mm.
Percentages are calculated before rounding and remain distinct from loss
reductions. The checkpoint/protocol overview is deferred.

The main memory comparison uses the seven-year carry/carry, carry/reset, and
separately trained reset/reset controls. Its approximate reconstructed loss is
explicitly distinguished from V24's exact GraphCast loss and RMSE diagnostics.
Section 4.5 reports one $B,C$ group and the 2k–8k SWA window; the appendix
comparison across additional group settings and averaging windows has been
removed. City trajectories now supplement the main results with a separately trained
fifteen-year correction (ERA5 2007–2021, K=20, inner dimension 128, SWA 6k–11k).
One six-panel figure in the appendix "Forecast examples at city locations"
shows Paris and New York City on January 15, 2022, with rows for temperature,
850 hPa zonal wind, and six-hour precipitation. This replaces the four seasonal
figures (60 panels). The Results cross-reference and appendix text describe
both improvements and larger local errors: four of the six panels have lower
MAE over 120–240 h, while Paris temperature and New York precipitation have
higher MAE. This pointwise diagnostic remains distinct from aggregate RMSE.
The original seasonal figure assets remain available on disk.

Section 4.4 compares GraphCast--Mamba, GraphCast-small, and the higher-resolution
GraphCast reference using a single row of Z500, T2m, and U850 panels. These are
direct crops from the existing common-grid figure with a shared model legend;
the curves are unchanged. Appendix D retains the full variable comparison.
Optimizer studies and training-horizon diagnostics remain in the appendix.
Appendix B retains the training-horizon comparison; its former variable-breadth
subsection (B.2) and channel-improvement figure have been removed. A new B.2
adds the closed-loop fifteen-year training-horizon sweep
(`Graphs/main_v3/training_ar_horizon_15yr_closedloop.pdf`, K = 1, 4, 8, 10, 12,
16, 20). Its K = 1 run conditions all 24 window steps on ERA5, as does the
archived sweep's K = 1.

The main training subsection keeps initialization, memory and gradient handling,
the averaged objective, and the 2,500-update training budget. Appendix A,
"Training details and model selection," collects the reported configuration
(A.1), complete loss definition (A.2), Mamba capacity and learning-rate selection
(A.3), optimizer configuration study (A.4), and checkpoint averaging and model
selection (A.5). A.3 motivates separate temporal learning rates using the
width-scaling results of Yang et al. (2022) and reports all six configurations
in one table; the scaling argument is an empirical motivation for this
architecture, not a prescribed multiplier.

The Mamba description assumes familiarity with the architecture and focuses on
mesh integration, configuration choices, and memory carried between forecasts.
Collaborator annotations are preserved in the source and hidden in the PDF.
The final-version merge removes revision colors and reports the principal
day-10 loss reduction consistently as 23.68%, with a 10.77-percentage-point
difference in the memory comparison, following the supplied final text.
The memory comparison figure uses 80% of the text width.
The AI-use statement
still asks the authors, in a source comment, to verify the study-wide history.

Seven-year figure sources, numerical exports, and reproduction instructions are
in [Graphs/main_v3/README.md](Graphs/main_v3/README.md). Earlier unused figure
assets remain available on disk. `main_v2.tex`, `main_v2.pdf`, and
`introduction_draft.tex` are unaffected by this revision.

The final appendix, "Additional ablations on fifteen years of training data,"
combines the supplied spatial-depth and memory/size figures into panels (a,b).
Its prose and caption are blue at the author's request. All original curves
and markers are retained, with K=0 relabeled as the paper's teacher-forced K=1
reference and the size-study legend reporting actual Mamba dimensions.
The caption distinguishes four-initialization monitoring from the separate
32-initialization SWA evaluation and uses approximate reconstructed loss.
Original assets and a portable generator are preserved under
`Graphs/main_v3/ablation_15yr/`.
