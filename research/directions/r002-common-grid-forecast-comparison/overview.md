# r002 — How does the selected correction compare with higher-resolution GraphCast?

Category: evaluation
Status: active
Updated: 2026-10-19 (America/New_York)
Related: [matched memory training](../r001-matched-memory-training/overview.md)

## Question and hypothesis

Does the selected seven-year GraphCast–Mamba checkpoint improve long-lead
physical-unit RMSE relative to official 0.25° GraphCast when both and the
paired 1° backbone are evaluated on identical 2023 initializations against
the same 1.5° verification truth?

## Current understanding

**October 19 paper verification (e011):** Fresh hashes of the current GC025
checkpoint and all three normalization files match live official DeepMind
object metadata. Its 0.25°/37-level, mesh-6, width-512, 16-step architecture
matches the released main-paper model. This establishes release identity,
not reproduction of the paper's 2018 native-grid scorecard: the current
comparison uses 2023 starts and 1.5° verification. Re-pooling the saved 122
matched 2022 starts confirms day-3 RMSE above the unidentified DeepMind
archive by 11.758% Z500, 11.991% T2m, 7.166% U850 and 6.744% Q700.
The existing one-start upstream audit differs by at most 1.8431% in RMSE;
it does not establish full numerical parity. No new forecasts were run.
See [e011](experiments.md#e011) for the fresh audit artifact and limits.
Follow-up reading found another protocol difference: paper Eq. 20 averages
per-initialization spatial RMSE, whereas our WB2 comparison pools MSE before
taking a square root. A proposed small numerical reproduction targets only
the GraphCast Z500 curve in Figure 2a, initially at 12/24/48 h on 24
seasonally spread 2018 starts at 06/18 UTC. This pilot is not yet run and
would not establish full-year parity. Supplement Figure 50 provides an even
smaller qualitative single-start check (2018-11-23 12 UTC).

**October 19 figure revision:** Figure 5 now shows Q700/T2m/U850 at the
user's request (92–105 matched starts per point); its Q700 day-10 RMSE
reduction versus public GC025 is 7.70%. Numerical data and Figure 9 are
unchanged. Updated captions/text and the checked PDF were pushed as `1b32b6c`.

**Latest October 18 manuscript update (e010):** The user approved the
ENS-relative previews for Figures 5/9. Section 4.4 and Appendix C now use the
selected M1/public-checkpoint 2023 comparison. On the matched samples M1 has
lower RMSE than public GC025 on all 12 diagnostics at every lead from 3.5 to
10 days; day-10 reductions span 7.70–11.42% (Z500 8.48%, T2m 10.58%, U850 9.83%).
The checkpoint audit establishes that GC025 is higher-resolution, not a
larger-parameter model: 36.46M stored parameters versus 46.33M for M1 total.
It uses about 16× the horizontal grid points and 37 vs 13 pressure levels.
The user clarified that “larger” refers to memory and FLOPs. The published
text emphasizes the larger spatial computational workload and retains the
common-grid interpretation, without claiming measured cost ratios. The
update was pushed to Non-Markovian-Weather/main as `098e948` after preserving
two newer Overleaf formatting edits and checking the compiled PDF.

The user requested the historical
Figure5/9 style with an ENS baseline. Repaired 2023 ENS-relative previews are
complete under
`plots/analyze_models/images/resolution_eval/public_checkpoints_2023_repaired_ens_20261018/`.
They show100 × model RMSE / matched ENS-mean RMSE, with ENS=100 and the same
panel layout and line colors. ENS availability permits91–108 starts per
diagnostic/lead (91–103 in Figure5); every model uses exactly the same dates
at each point. These are conditional subsets of the122 candidate starts,
not a complete122-start ENS comparison. All42 available truth-affected ENS
scores were repaired, with original scores reproduced exactly and missing
entries preserved. Independent validation checked all187720 matched CSV
rows and the pooling, ratios, hashes and plots. At day10, M1 is107.36%,
107.26% and106.34% of ENS RMSE for Z500/T2m/U850 respectively: it does not
outperform ENS at that lead. The validated full122 model-only comparison
below remains intact; its confidence intervals are not transferred to these
different ENS-matched subsets.

**Repaired model-only comparison (e009):** Independent NCAR truth recovery, the three
targeted reruns, the full 122-start merge and independent rescoring are complete.
Every model has 122 samples at every lead and diagnostic. GPU jobs15802505
and15802506 succeeded. Report job15802647 stopped at its source guard after
a metadata-only plotter edit; a separately recorded CPU recovery completed
without changing the plotting function or forecast results.
The November anomaly disappears for all three models. Numerical rerun
variation affects pooled RMSE by at most 0.02565%, separately measured from
the truth repair. Day10 M1 physical-RMSE reductions versus public GC025 are
7.6795–11.3843% across the twelve diagnostics, including Z500 8.4446%,
T2m 10.2386% and U850 10.0592% on the common 1.5-degree grid.
Previews are under
`plots/analyze_models/images/resolution_eval/public_checkpoints_2023_repaired_20261018/`.
Validation checked all 175680 CSV cells and independently recomputed 8640
old/new-truth scores from 360 retained forecasts. The campaign is
`artifacts/evaluations/graphcast_common_grid_2023_truth_repair_20261018`;
its repaired summary hash is `dcda10d7...e4d5e81`. Manuscript incorporation
awaits review; native GraphCast-loss repair remains separate.

**Prior October18 result (e008):** PrincipalM1/public-checkpoint2023 previews
are complete after user-approved identical data-quality handling: all122
starts retained,121--122 paired valid samples per diagnostic/lead. The spike
was traced to stale ARCO source fields, corroborated by raw copies and
independent NCAR samples. Independent pooling and masked-bootstrap checks
passed. New physical-RMSE previews are under
`plots/analyze_models/images/resolution_eval/public_checkpoints_2023_qc_20261018/`.
Day10 reductions versus publicGC025 remain7.6785--11.3892% across12 diagnostics.
At the user's request, figures/captions omit exclusion detail and methods
retain the sample-count range; full audit details remain in reproducibility
files. The approved68-date manuscript assets remain unchanged pending review.
Native-loss sensitivity is quantified (M1 14.4240% to14.4316%); original
native manuscript values are unchanged.

**October18 update:** The same-start2022 archive check is complete (e006).
DeepMind's actual`graphcast_v2/2022` forecasts score close to the published
benchmark, whereas the public1979–2017 checkpoint rerun has higher early-lead
RMSE on the same starts and truth. The remaining difference is between model
runs; weights versus numerical configuration is unresolved. Earlier references
below to only2018/2020 forecasts describe the initial search, superseded by
the recovered2022`graphcast_v2` archive.

The user authorized a separate historical K22 re-evaluation on69 daily00UTC
starts in five2022 periods, with full10-day forecast windows disjoint from
the recovered32 development starts and four numerical pilots (e007).
Actual DeepMind GC025 and IFS ENS-mean archived forecasts are scored on
matching dates. The frozen campaign is
`artifacts/evaluations/k22_common_grid_2022_recovery_20261018`. All69 inputs
were staged; an extra-truth writer argument was repaired in a new frozen
recovery, preserving the failed attempt and reusing the inputs. Data gate
15792878 → pilot15792879 → nine-shard array15792880 → report15792881
are submitted; staging of69 inputs/456 truth times is complete and the
pilot and all nine production tasks completed. DeepMind's archive lacks
Dec31, so the user approved excluding that date from all four models.
The completed68-date analysis and Figure5/9 analogs are in
`artifacts/evaluations/k22_common_grid_2022_matched68_20261018` (e007).
Independent CSV pooling and matched truth/lead coverage checks passed, and
both figures were visually inspected. GC025 has lower short-lead errors;
K22 has lower day10 RMSE on all12 diagnostics(5.465–10.465%). Z500/T2m/U850
day10 reductions are9.388/8.120/9.866%, with sustained positive point
estimates from days6.5/7.0/5.5. This is supplementary evidence on five
seasonally uneven periods, not selected-M1, untouched-year or annual skill.
The user approved incorporating the68-date figures into
`Overleaf/arxiv_versoin.tex`. Figures5/9, their captions, Section4.4 and
AppendixC now document this evaluation, including the incomplete historical
checkpoint provenance. The compiled PDF and changed pages were inspected;
figure-source hashes and dates accompany the copied assets.

The paper's historical common-grid panels use a different K=22 correction
and 2022 forecasts. They do not establish the selected model's relative
performance. The user authorized a new 2023 comparison on 2026-10-15.

The principal checkpoint and 122-start manifest are fixed by the completed
October 9 evaluation. New common-grid metrics require new rollouts because
the existing outputs retain scalar errors rather than forecast fields.
Live official WeatherBench2 listings contain GraphCast forecasts for 2018
and 2020, not 2023. The official 0.25°/37-level checkpoint is available locally.
All 122 starts and 40 leads are complete for all three models, and the final
report was generated. Its day-10 M1 RMSE reductions versus the 0.25° reference
are 7.68–11.39% across twelve diagnostics. These are computed results whose
scientific interpretation remains under validation after the user questioned
the discrepancy with the historical figure.

The archived curve comparison shows substantially earlier crossovers in the
new results, especially Z500 (about day 8 historically versus day 1.25 now).
The main discrepancy is the 0.25° reference's skill relative to the 1°
backbone: at day 3, Z500 changes from approximately 15% lower RMSE in the
archived figure to 2.97% higher RMSE now. Historical values are image-derived
approximations; the archived figure is not an independently verified control.
Normalization files match the earlier 37-level setup, and 1°→1.5° remapping
weights match the historical implementation exactly. Native M1 loss reduction
is 14.4793% versus the original 14.4240%. No cause for the reference discrepancy
has been established. Do not treat successful jobs or plotting tests as an
independent validation of the 0.25° forecasts.

The archived green reference source has now been recovered: the public
WeatherBench `graphcast_vs_era5_240x121_2022.nc` scores, divided by the public
2022 ENS-mean scores. Local archived files match the public bucket checksums;
all twelve raster curves agree within 0.06–0.13 percentage points RMS on the
ENS scale over days 1–9.75. The new reference checkpoint separately matches
DeepMind's public 1979–2017, 37-level 0.25° file exactly. These are different
forecast sources and evaluation years; the exact checkpoint behind the public
2022 scores is not specified in the recovered score file. Do not conflate it
with the separately found local operational-GraphCast experiment.

An independent rollout using the unmodified upstream GraphCast package,
direct checkpoint parameters, and upstream autoregressive/chunked prediction
completed as job `15657532` on the January 1, 2023 input. All 18 input fields
match the frozen evaluator bitwise in the CPU preparation check. Upstream RMSE
differs from production by at most 1.8431% across 12 diagnostics × 40 leads;
day-3 Z500 is 1.3069% lower and day-10 Z500 0.0673% lower. This limited test
does not reproduce a large rollout error; it is not bitwise forecast parity
or a full-year validation. CPU/GPU preparation differs in solar-radiation
bits, and forecast differences have not been attributed to an individual
numerical mechanism. All three statistics files also match the official
DeepMind bucket exactly. The cause of the full archived/new skill gap remains
unresolved; do not claim a proven incorrect model or a proven training-cutoff
effect. No manuscript figures or production results were changed. See e002,
`audits/reference_provenance.json`, and
`audits/stock_rollout_20230101T000000Z/result.json`.

## Interpretation limits

This compares fixed predictors, not resolution in isolation: the checkpoints
also differ in pressure levels, mesh, and training coverage. The original
2023 pilot exposure remains disclosed. The primary 1° exact GraphCast loss
evaluation remains separate; common-grid physical RMSE is a secondary metric.
Results do not answer the matched memory-training question in r001.

## Next decision

Figure 2a's 24-start reproduction passed all 480 independent score checks
(e012), using existing native 0.25°/37-level 2018–2019 ERA5, the verified
public checkpoint and upstream inference. At days 1/2/3/5, mean spatial RMSE
is within 0.8% of the approximate digitized paper curve; day 10 is 4.45% lower.
This is a sampled check, not annual parity. Full 730-start array `15870083`
is running; report `15870088` and corrected plot `15870456` are dependent.
Inspect the full-year campaign's summary and plot after completion before
concluding agreement with the paper.

The approved e010 ENS-relative Figure 5/9 comparison replaces the historical
68-date K22 figures. Broader claims require additional checkpoints, years,
and native-scale or controlled-compute evaluation; these are not established
by the present fixed-model comparison. The archive's precise checkpoint
remains a separate unresolved provenance question (e006).
If updating native-loss figures/tables, reconcile their corresponding
data-quality sensitivity separately and consistently. Do not select
checkpoints to reproduce an expected ranking.

Live execution state:
`artifacts/evaluations/graphcast_common_grid_2023_20261015/control/state.json`.
The generated report is in that campaign's `report/report.md`; the archive
comparison is in `audits/archive_comparison.json`.

## Records

- [Attempts and evidence](experiments.md)
