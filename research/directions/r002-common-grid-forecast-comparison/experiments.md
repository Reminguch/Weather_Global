# Attempts and evidence

<a id="e001"></a>
## e001 — Fixed-model 2023 common-grid comparison

Event date: 2026-10-15
Recorded: 2026-10-15 (America/New_York)
Execution: completed (forecast and report jobs)
Outcome: inconclusive (independent reference validation pending)

**Question and change:** Replace the historical K=22/2022 reference with a
new comparison of the selected principal checkpoint and its 1° backbone to
official 0.25° GraphCast on the same 122 initialization dates.

**Setup:** 00 UTC every third day from 2023-01-01 through 2023-12-30;
40 six-hour leads through 240 h, with truth extending through 2024-01-09.
All three models score against the same native-0.25° ERA5 fields conservatively
remapped to 1.5°. Twelve historical diagnostics are Z500, Z850, T500, T850,
T2m, MSL, Q700, Q850, U850, V850, U10, and V10. Spatial scoring uses
normalized cosine-latitude weights. Pool per-start physical MSE before square
root. Paired 10,000-draw five-start-block intervals are pointwise; three/seven
start block and prior-pilot exclusion sensitivities are retained.

**Fixed checkpoints:** M1 is
`artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/swa/swa_step00500-02000.pkl`,
SHA256 `b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0`.
Its paired backbone is official GraphCast_small, trained on ERA5 1979–2015.
The high-resolution checkpoint is official GraphCast, 0.25°, 37 pressure
levels, mesh 2to6, trained on ERA5 1979–2017.

**Provenance:** The original campaign is
`artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009`, protocol
hash `51bee2021076fffeabb58ca8bc86c9c08d951a43879e1b75cd62e6da431fe065`.
M1 uses that campaign's frozen evaluator and numerical policy: BF16 predictor,
FP32 physical fields/state boundaries, triton GEMM disabled; finite numerical
drift is reported under its existing advisory policy. New scripts are frozen
separately before submission; a dirty working-tree commit alone is insufficient
to identify the run. Native-grid scores are retained as a bridge, separately
from the common-grid comparison.

**Data evidence:** Official WeatherBench2 GCS listing on 2026-10-15 has only
2018 and 2020 GraphCast archives. Existing common-grid figure regeneration
copies archived raster artwork and cannot produce new numerical results.
The local ARCO staging report for the earlier 1° dataset records approximately
984.6 GB transferred in 2,324 seconds with eight workers on della-vis1; this
is a planning reference, not a timing guarantee for the new data.

**Resource evidence:** Principal evaluation job `15297331_0` completed in
23:20 at 7.37 GiB CPU peak for 16 starts. Comparable 0.25° paired-model
streaming evaluations `13160779_0` and `14008384_15` took about 3:02–3:15
at 60.5–60.6 GiB CPU peak for eight starts. A baseline-only smoke will
determine the new 0.25° production request. No array concurrency throttle.

**Frozen campaign:**
`artifacts/evaluations/graphcast_common_grid_2023_20261015`, protocol hash
`b69bf34e002a47cc12b17a6ffda91e377834412d31598e6bb6a20dea4fda5228`, source digest
`78cd38710267ac2a4063b481ae49e6a0a6e4aa00a018e2196ad6b0151d3309e2`.
The targeted scorer, inference-runner, and merge suite passed 39 tests.
The separate continuation controller passed another 21 tests (60 total);
its CPU merge wrapper also passed shell syntax validation.

**Staging and parity evidence:** The first initialization plus its 40 truth
times completed on della-vis1 in 69.9 seconds, transferring 22.07 GB.
The full 122-input/1,492-truth campaign is staging with eight workers under
configuration SHA256
`17dbfe987adadd1b9fa037a7700de770b3ddbbd7e23195b32df6d6940ba4a5e4`.
`audits/input_parity_first.json` records exact equality for all 11 dynamic
variables at both input times on shared coordinates and 13 pressure levels;
land–sea mask also matches. Surface geopotential differs only at the two
pole rows (720 points; maximum 363.666 m²/s²); every nonpolar point matches.
The old 1° static field came from its original WeatherBench2 store, whereas
the new native field comes from ARCO. Its forecast effect is unmeasured.

**Pilot submissions:** M1 plus paired 1° GraphCast job `15634901`, 16 GiB;
native 0.25° GraphCast job `15634902`, 96 GiB. Both request one A100 80 GB,
eight CPUs, 1:02 walltime, and one January 1 forecast. Submission receipts
and logs are under the frozen campaign. Initial restricted-network sbatch
attempts exited without creating jobs; retries with scheduler access created
these two recorded jobs.

**Automatic continuation:** The separately frozen controller is running on
della-vis1 and waiting for staging process `1882138`. Its read-only control
code and hash manifest are in the campaign's `control/`; live status is
`control/state.json` and the log is `logs/controller.log`. It verifies all
1,614 files and both complete pilots, derives the 0.25° resources from
observed timing and memory, submits eight M1 and sixteen 0.25° shards, and
queues a CPU merge after both arrays succeed. It stops on failed validation
or ambiguous submission rather than launching duplicates. The full-array
outputs and scientific comparison are still pending.

**Independent review:** Native 0.25° input times, 37-level statistics,
forcing timestamps, checkpoint overlay, precision, and independent feedback
were reviewed against the vendored interfaces; no blocking issue was found.
Verification truth enters only scoring. Changing static time coordinates can
cause per-lead compilation during the first forecast; the pilot therefore
determines the production budget. A successful one-start smoke is not
scientific evidence about relative forecast skill.

**Completed M1 pilot:** Job `15634901` finished its rollout in 487.6 s
(approximately 517 s evaluator elapsed), with 6.77 GiB peak host RAM.
Slurm accounting records 537 s allocation elapsed and 7.06 GiB peak RSS;
the controller uses the larger accounting measurements for its budget. All
40 truth hashes and 960 common-grid MSE entries passed the integrity audit.
Native replay is not exact: January 1 mean baseline loss changed from
11.30724662 to 11.21694281 (−0.7986%), and M1 loss from 9.51575155 to
9.53819509 (+0.2359%). Maximum relative drift across leads was 1.3459%
for the baseline and 0.4795% for M1; both exceeded the historical 0.1%
replay tolerance on 26/40 leads. This is an advisory warning under the
unchanged numerical policy, not a checkpoint-identity failure. The cause is
unproven. The smoke uses a 42-timestamp data window; production uses the
full initialization manifest as in the original campaign. Full production
native scores must be retained for comparison, and the drift disclosed.

**Replay investigation:** `audits/replay_context_first.json` compares the
42-timestamp smoke context with the full 1,494-timestamp original context.
All five generated forcing variables, including solar radiation, are bitwise
identical on CPU for the first initialization's two inputs and 40 future
leads. This does not support context length as the explanation and does not
establish GPU repeatability. Historical V24 executions varied by up to 3.24%
at individual January 1 leads, but their mean baseline losses span
11.29140–11.37272; the fresh mean 11.21694 lies below that observed range.
Do not describe the discrepancy as proven ordinary numerical noise.

**Observation:** No forecast comparison result yet. Existing manuscript and
previous campaigns are preserved. Production must wait for complete staging
and successful, provenance-checked pilots.

**Decision / next action:** Finish the 0.25° pilot and full data staging.
The running controller will validate both prerequisites, submit production,
and merge only complete matched coverage. Inspect the resulting full-year
native replay evidence alongside the common-grid comparison before drawing
scientific conclusions.

### 2026-10-15 — Completed campaign and user-requested archive comparison

Production arrays `15642243` (8 M1/GC1 shards) and `15642244` (16 GC025
shards) and report job `15642245` completed successfully. Every model has
122 starts and 40 leads. The report computes 7.68–11.39% lower day-10 RMSE
for M1 than GC025 across twelve diagnostics. This confirms execution and
aggregation, not independent scientific validation of the reference forecasts.

After the user challenged the size of the advantage, compared the archived
PDF bitmap with new numerical results. The audit script
`scripts/analyze_models/audit_common_grid_archive.py` preserves the archived
files and labels image-derived values as approximate. Outputs:
`audits/archive_comparison.json` and
`plots/analyze_models/images/resolution_eval/common_grid_2023_20261015/audit_archive_comparison.{png,pdf}`.

| Diagnostic | Archived approximate day-10 RMSE reduction | New exact day-10 RMSE reduction | Archived → new sustained positive crossover |
| --- | ---: | ---: | --- |
| Z500 | 5.6% | 8.46% | day 8 → day 1.25 |
| T2m | 8.1% | 10.24% | day 6.75 → day 3.5 |
| U850 | 9.1% | 10.07% | day 5.5 → day 3 |

Crossovers refer to point estimates on six-hour leads, not significance.
Ratios between models cancel a shared ENS denominator, so normalization
cannot explain these relative-curve differences. The archive uses a different
K=22 correction and is labeled 2022; its high-resolution forecast provenance
has not been recovered, so it cannot be assumed correct or directly matched.

The largest discrepancy is the reference's early/medium-lead skill. At day 3,
GC025 versus GC1 Z500 RMSE reduction is about +15.12% in the image but −2.97%
in the new data. T2m and U850 also lose their earlier reference advantage.
No confirmed implementation defect or causal explanation has been found.

Checks completed: all three normalization files in `stats` and
`stats_graphcast_37` have identical SHA256 values; latitude and longitude
remapping matrices match the old WB2 helper exactly (maximum differences
both zero); the new M1 native loss reduction is 14.4792985% overall and
21.8081447% at day 10, versus 14.4240196% and 21.6584261% originally. The
original comparison selected exactly `M1_00.json`–`M1_07.json`, excluding
the separately named inference-reset shards. These checks do not establish
equivalence to an independent official 0.25° rollout.

Decision: keep the new higher-resolution claims provisional. Next check is
an independent stock GraphCast rollout with identical checkpoint, input
fields, precision, and forcing, followed by archived-reference provenance
recovery. Preserve existing outputs rather than changing them toward an
expected model ranking.

### e002 — 2026-10-15: Identify the reference mismatch and independently audit inference

Execution: completed (`15657532`; January 1, 2023, all 40 six-hour leads)
Outcome: inconclusive (reference source recovered; large rollout error not
reproduced on this initialization; full archived/new gap not causally explained)

The user explicitly requested identification of the mismatch, then asked
whether the 0.25° reference is DeepMind GraphCast. Confirmed the new checkpoint
is exactly the official DeepMind public artifact: byte size 145,990,258 and
MD5 `w+1bpwpEaR1SorPLZ3Gojw==`, checked against the public `dm_graphcast`
bucket metadata. It is the 37-level, ERA5 1979–2017, mesh-6 model. Official
metadata is retained in `audits/deepmind_checkpoint_listing.json`.

Recovered the old green curves' numerical source under the shared project:
`della:/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/wb2_scores/graphcast_vs_era5_240x121_2022.nc`
and `ens_mean_vs_era5_240x121_2022.nc`. Both match the current public WeatherBench
bucket byte counts and MD5 values. GraphCast score MD5 is
`y73Cglyh1vtvzB+u6at1UA==`. Copies are retained in `audits/recovered_wb2/`.
Script `scripts/analyze_models/audit_common_grid_provenance.py` matches all
twelve green curves against 100 × published GraphCast RMSE / published ENS
RMSE. Raster-versus-score RMS disagreement over days 1–9.75 is 0.060–0.128
percentage points, consistent with raster reading precision. Numerical
evidence is `audits/reference_provenance.json`.

| Diagnostic, day 3 | Recovered 2022 GraphCast RMSE | New 2023 GraphCast RMSE |
| --- | ---: | ---: |
| Z500 (m² s⁻²) | 122.412654 | 139.813840 |
| T2m (K) | 0.808285 | 0.924205 |
| U850 (m s⁻¹) | 1.901066 | 2.062023 |

This proves a reference-source/year mismatch; it does not identify its causal
contribution to skill. The 2022 score file has no checkpoint ID or initialization
manifest. The public WeatherBench FAQ describes a 1979–2019 GraphCast version,
but its text does not uniquely identify the recovered 2022 scores. Do not
assert the exact 2022 training cutoff from that FAQ alone. The public forecast
listing still contains only 2018 and 2020, despite the published 2022 scores.

A separate shared-project `fengwu/gc025_rollout_score.py` and
`results/gc_025_baseline_eval/gc_operational_025_eval32_2022.json` use the public
13-level operational model. This initially suggested a model-family mismatch,
but it is not the recovered source of the twelve green curves. Preserve this
distinction; the earlier intermediate suggestion was not a confirmed cause.

Independent inference implementation:
`scripts/analyze_models/audit_common_grid_stock.py`, with a job-local source
copy in `audits/stock_source/`. Uses an isolated official GraphCast checkout
at `08cf73625c9d12bd9aaa038868bcb2fe488f2a22`, direct checkpoint parameters,
upstream Bfloat16Cast/InputsAndResiduals/autoregressive.Predictor and
chunked_prediction_generator. No project model wrappers are imported. Same
native staged input and common-grid scorer; future meteorological truth is
only used for verification. Per-lead common fields and first-lead native
diagnostics are retained for inspection. CPU checks establish all 18 input
fields match the frozen production preparation bitwise, including all 37
levels, precipitation, forcings, and static fields. Evidence:
`audits/stock_prepare_20230101T000000Z.json` and
`audits/stock_frozen_input_parity.json`.

Upstream source comparison found only inactive Mamba extensions, unrelated
multi-device API changes, and a longitude-only rotation compatibility change
(GraphCast requests both latitude and longitude rotations). No identified
active model-math defect. This is source inspection, not numerical equivalence.

Budget: one A100-80, 8 CPUs, 48 GiB CPU RAM. Comparable production pilot
`15634902_0` completed in 32m14s with 27,730,656 KiB MaxRSS; production shard
`15642244_0` completed in 37m49s with 32,424,728 KiB. Initially submitted
`15657342` with 62 minutes, but the queue estimated a two-hour wait. Moving a
job between partitions was rejected by the scheduler; cancelled only that
unstarted audit and resubmitted with 45 minutes, letting the cluster select
gputest. Explicit partition requests are rejected by this cluster.
`15657501` failed after 14 seconds before inference because its clean-source
guard counted generated `__pycache__` files. Preserved its source as
`audits/stock_source/attempt1_source.py`; changed the guard to check tracked
source modifications, and submitted `15657532`, which started on della-l07g3.
Logs: `audits/stock_logs/15657532.{out,err}`. Original production snapshot,
forecasts, report, and manuscript figures are unchanged.

Public provenance sources: [official WeatherBench score object](https://storage.googleapis.com/weatherbench2/benchmark_results/graphcast_vs_era5_240x121_2022.nc),
[ENS score object](https://storage.googleapis.com/weatherbench2/benchmark_results/ens_mean_vs_era5_240x121_2022.nc),
[WeatherBench model FAQ](https://sites.research.google/weatherbench/faq/),
[upstream GraphCast revision](https://github.com/google-deepmind/graphcast/tree/08cf73625c9d12bd9aaa038868bcb2fe488f2a22).

Completed independent audit: `15657532` succeeded in 4m08s, MaxRSS 7,687,896
KiB (7.33 GiB); Python reports 7.39 GiB peak. Forecast plus streaming scoring
took 70.86 seconds; setup includes a slow solar-forcing compilation. All
40 common-field files and first-lead native diagnostics are saved under
`audits/stock_rollout_20230101T000000Z/`; `result.json` contains 480 scores and
their production comparisons. Maximum absolute relative RMSE difference over
12 diagnostics × 40 leads is 1.8431%. At day 3, upstream-minus-production
RMSE is −1.3069% (Z500), −0.2754% (T2m), and −0.4730% (U850). At day 10,
the corresponding changes are −0.0673%, −0.4205%, and +1.1740%. The upstream
path therefore reproduces the broad error scale on this start and does not
expose a large implementation failure. These are not bitwise-identical
forecasts and this one start cannot establish full-year equivalence.

The GPU preparation hashes differ from CPU only for incident solar radiation;
all 17 remaining fields match exactly. The CPU frozen/upstream preparation
comparison still matches all 18 fields. We have not isolated whether forecast
differences arise from GPU arithmetic, compile shape, solar generation, or
another small numerical difference. Do not label them proven ordinary noise.
All three local normalization files were additionally verified byte-for-byte
against the public DeepMind `stats/` objects; retained official metadata in
`audits/deepmind_stats_listing.json` and identities in `reference_provenance.json`.

Next decision: resolve the exact 2022 benchmark checkpoint, or validate against
an independently archived prediction with matched checkpoint and timestamps.
Do not substitute the unrelated operational checkpoint merely to obtain a
desired ranking. Keep the new full-year superiority claims provisional.

### e003 — 2026-10-16: Matched 2023 ENS normalization and figure analogs

Execution: completed (login-node CPU scoring; no Slurm job)
Outcome: partial (ENS denominator recovered for 46,930/58,560 entries;
reference gap localized to early/medium leads)

The user requested 2023 analogs of main_v3 Figs. 5 and 9 with the archived
"RMSE vs ENS" axis. WeatherBench2 publishes ENS scores only for 2018, 2020 and
2022. Its public `ifs_ens/2016-2024-240x121_equiangular_with_poles_conservative_mean.zarr`
has no zarr metadata or coordinates and lacks about a fifth of its chunks.
`scripts/analyze_models/score_ens_common_grid_2023.py` (SHA256
`fa4c8dc5232623447210608b53d29d86e2313f421673d26399bcdedf15289388`) borrows
the layout from the complete 2018–2022 sibling; 71 overlapping chunks across
all nine variables agree to 6.7e-7 relative difference. It scores with the
frozen campaign `common_grid_metrics.score_step` against the same staged truth,
checking every truth hash against the M1 shards. The ENS grid equals the truth
grid. Absent chunks are null; nothing is substituted. Output:
`ens/ens_mean_mse.json` (SHA256 `1f9c8b58bcbcb742ac2b678893479e74a041827a1736eecdf876ab6ac5063e14`).
No start is complete in all variables; per (diagnostic, lead) 91–108 starts
have ENS data. Every model is pooled over exactly those starts.

`scripts/analyze_models/plot_common_grid_2023.py` writes GraphCast-relative and
ENS-relative versions under
`plots/analyze_models/images/resolution_eval/common_grid_2023_20261015/`.
Manuscript figures are unchanged.

| % of ENS RMSE | Archived 2022 M1 / GC1 / GC025 | New 2023 M1 / GC1 / GC025 |
| --- | --- | --- |
| Z500 day 3 | 105.8 / 107.3 / 91.1 | 100.8 / 101.8 / 104.6 |
| Z500 day 10 | 110.6 / 124.7 / 117.2 | 107.3 / 121.1 / 117.3 |
| T2m day 3 | 80.9 / 82.2 / 75.5 | 84.3 / 84.9 / 84.7 |
| T2m day 10 | 109.5 / 121.7 / 119.1 | 107.3 / 119.8 / 120.0 |
| U850 day 3 | 94.5 / 95.5 / 90.6 | 96.9 / 96.7 / 97.4 |
| U850 day 10 | 107.7 / 123.8 / 118.5 | 106.3 / 122.9 / 117.9 |

Archived values are raster digitizations; archived M1 is the K=22 model on a
different start set. At day 10 the local GC025 matches published 2022
GraphCast relative to ENS within about 1 point. At day 3 it is 7–13 points
worse, while both 1° curves stay within about 6 points of the archive. The
remaining discrepancy is therefore concentrated at early/medium leads, a
pattern consistent with, but not proof of, a different reference checkpoint
or an input/initial-condition difference. ENS itself changed resolution in
2023 (IFS 48r1), so year-to-year ENS ratios are not strictly comparable.

Next decision: the WB2 2020 archive check (score archived forecasts with our
scorer; compare our pipeline field-by-field on matched 2020 starts) remains
the discriminating test for early-lead GC025 skill.

### e004 — 2026-10-16: Inspect curly 2022 smoke curves

Execution: completed (local scalar-score inspection; no jobs submitted)
Outcome: supported for sampling/normalization explanation of the visual shape;
overall reference skill discrepancy remains inconclusive.

The figures in `plots/analyze_models/images/resolution_eval/common_grid_2022_20261016/`
with `_smoke` suffix use only `pilot/gc025_00.json`: one initialization,
2022-01-01 00 UTC, not the intended 122-start annual manifest. The published
WB2 file has no initialization dimension or attributes identifying its exact
sample set. Both sources have the same ordered 40 leads, 6–240 hours.
The plotter uses straight line segments and divides both RMSE curves by the
same published 2022 ENS RMSE; no spline interpolation is involved.

Raw single-start Z500 RMSE increases at all 39 transitions. T2m decreases at
4 and U850 at 2. Thus nonmonotonic Z500 ENS-relative ratios do not imply a
nonmonotonic raw error or rollout failure. Day-3 single-start differences
versus published WB2 are +2.00% Z500, +12.66% T2m, +9.38% U850; these are
not estimates of annual skill differences. Single-start sampling is a strong
explanation for the irregular shape, not proof that all implementation is correct.

Saved `gc025_sampling_diagnostic.png` in the same image directory: raw RMSE
and individual/pooled 2023 scores on the same ENS denominator. The 2023
comparison is explicitly a different-year illustration, not a matched control.
It exposed a separate outlier: 2023-11-09 Z500 RMSE at 30/36/42 hours is
212.65/354.58/502.14, then 85.68 at 48 hours; the cross-start median at
42 hours is 69.85. Cause unknown. Do not discard this record or attribute
it to model instability without inspecting prediction/truth fields.

Next: obtain the full 2022 aggregate before interpreting annual mismatch;
audit the November 9, 2023 truth and forecast fields at 30–48 hours separately.
Existing figures, evaluator code, and scores were preserved.

### e005 — 2026-10-16: Twelve monthly 2022 starts

Execution: submitted (data readiness 15708704; GPU array 15708705_1–11;
dependent report 15708711)
Outcome: pending

User authorized the proposed 12-start seasonal test. Selected the first
existing annual-manifest start in each month: Jan 1, Feb 3, Mar 2, Apr 1,
May 1, Jun 3, Jul 3, Aug 2, Sep 1, Oct 1, Nov 3, Dec 3, all 00 UTC 2022.
Manifest: `artifacts/evaluations/graphcast_common_grid_2022_20261016/initializations_monthly12.json`.
The original January pilot is reused by symlink as `monthly12/gc025_00.json`;
array indices 1–11 use shard count 12 and the monthly manifest override.
Frozen evaluator, checkpoint, protocol and original results remain unchanged.

The existing full-year stager was confirmed active on della-vis1; 51/492
required monthly input/truth files were ready at submission. A CPU readiness
job waits for file/receipt sizes; the evaluator subsequently verifies hashes.
Forecasts depend on its successful completion, so GPUs do not wait for downloads.
The readiness gate fails if staging reports failure or exceeds 11.5 hours;
downstream jobs then remain blocked for diagnosis rather than running incomplete data.

Resource evidence: January job 15705167_0 completed in 29m50s with batch
MaxRSS 27,428,492 KiB (26.16 GiB). Each new one-start job requests 40 GiB,
45 minutes and one A100-80GB; array has no concurrency throttle. The gate
requests 1 CPU/1 GiB/12h; scalar-score report requests 1 CPU/4 GiB/15min.
Submission record: `submissions/monthly12.json` in the 2022 workflow.

After all forecasts succeed, the report validates exact 12-start coverage and
40 leads, pools MSE before square root, and writes `*_monthly12` figures/JSON
in the existing 2022 image directory. This intermediate seasonal sample can
test persistence of the shape/gap but does not establish annual WB2 parity.

### e006 — 2026-10-16: Same-start comparison with DeepMind's archived 2022 GraphCast

Execution: completed (login-node CPU scoring; full-year GC025 array `15709551`
and monthly array `15708705` both completed)
Outcome: supported — the GC025 early-lead gap is a model-run difference, not
data, scoring, year, or sampling; checkpoint versus numerics is unresolved.

Data checks. Staged 2022-06-03 inputs (both times, 13 shared levels, all 13
variables including 6 h precipitation and statics) are bitwise identical to
WB2 `era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`.
Liangong's (`lm8598/.../fengwu`) `truth15` Z500/T850/Q700 equal our common
truth to float32 rounding after his N→S flip; his ICs come from that WB2
store. His `gc025_rollout_score.py` runs the 13-level GraphCast_operational
(ERA5-HRES 1979–2021) with Bfloat16Cast on ERA5 inputs; its 6 h Z500 RMSE is
77% above published, so it is not a valid reference for either model.

`scripts/analyze_models/score_wb2_graphcast_archive_2022.py` (SHA256
`f92025ad3b5226a1e7e5c0bfa0ad27fa73d6444492322fec2890f39881d1f37e`) scores
WB2 `graphcast_v2/2022-240x121_equiangular_with_poles_conservative.zarr`
(DeepMind 37-level, ERA5-initialized; internal model_xid 50432171) on our
122 starts with the frozen scorer and hash-checked truth. Output:
`wb2_archive/graphcast_v2_2022_mse.json` (SHA256 `729c54889639a9373939892b765a891f6e4277ee4243c8a5fa0da3b80224c243`).

Pooled archive RMSE matches the published 2022 score file within about 1%
at all checked leads, validating our truth and scorer. Our public 1979–2017
checkpoint versus the archive on identical starts (% RMSE):

| | 6 h | Day 1 | Day 3 | Day 5 | Day 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Z500 | +6.2 | +13.8 | +11.8 | +7.3 | +1.4 |
| T850 | +2.3 | +9.6 | +12.2 | +6.5 | +0.4 |
| T2m | +9.5 | +13.2 | +12.0 | +8.0 | +3.3 |
| U850 | +1.8 | +5.6 | +7.2 | +3.8 | −0.3 |
| Q850 | +3.2 | +5.0 | +7.5 | +6.1 | +1.2 |

Day-1 paired start bootstrap intervals exclude zero for all twelve
diagnostics (Z500 +12.6 to +14.9%). Ours is worse on 118/122 starts at
Z500 day 3. With inputs, truth and scorer identical, the remaining
differences are checkpoint weights and inference numerics (our bf16 GPU
versus DeepMind's unrecorded configuration). The archive does not identify
its checkpoint. Do not call it the public 1979–2017 model.

Next: an FP32 rollout of a few 2022 starts, saving 6 h fields, compared
against archive fields. Near-identical 6 h fields would implicate numerics;
a structured 6 h difference persisting in FP32 would implicate weights.

<a id="e007"></a>
### e007 — 2026-10-18: Historical K22 comparison away from recovered development windows

Execution: completed (69 local forecasts; matched68-date analysis and figures)
Outcome: supported for lower long-lead K22 physical RMSE on these five periods;
does not establish selected-M1, annual or untouched-year performance.

The user authorized rebuilding Figures5/9 using dates away from development,
then accepted the proposed daily schedule. Campaign:
`artifacts/evaluations/k22_common_grid_2022_20261018`; protocol hash
`fabbf725477c373a3f15efa6cea865fa20e5924ceca76a5c04df8d6956d8f2dc`.
The full source, checkpoint copy and date manifest are frozen separately.

The 69 starts are00UTC daily in these inclusive2022 ranges: Feb7–19(13),
Mar14–23(10), Apr27–May7(11), May29–Jun12(15), Dec12–31(20). Each
input/verification interval[start−6h,start+240h] must be disjoint from all
32 recovered development forecasts and four numerical pilots. The prior
24-start count considered only the existing every-third-day schedule.

Recovered K22 checkpoint, EMA and alpha sweeps all use seed0 and32 samples
from the1395 eligible2022 indices with64 future steps; reproducing that
sampling rule gives exactly the principal development32 indices. Evidence
copies and hashes are in`audit/` and`development_audit.json`. This establishes
separation from recovered development windows; it cannot exclude unrecorded
exposure and does not make2022 an untouched test year. The five remaining
periods are seasonally uneven and provide limited independent weather.

The historical checkpoint path`lm8598/.../results/v22cl_r1_EMA/K22_ema.pkl`
resolves to`v22cl_r1_K22_ema_5k_10k.pkl`; its contents contain params/state
only. The filename does not establish the exact averaging recipe. The fixed
copy is`checkpoints/K22_swa_5k_10k.pkl`; protocol records its hash. The
evaluator uses the actual legacy V9 residual factory/temporal attachments,
independent full-feedback trajectories and zero-initialized temporal memory;
future truth enters only common-grid scoring. Parameters are checked against
the initialized tree before forecasts. This remains a separate historical
configuration, not the selected principal M1 checkpoint.

Inputs are1°/13-level subsamples of native0.25° ERA5, reusing cached native
inputs when available. All models score the same hash-checked native0.25°
ERA5 truth conservatively remapped to1.5°. The actual DeepMind
`graphcast_v2/2022` archive and complete2018–2022 IFS ENS-mean archive are
scored on precisely these dates. No yearly aggregate or locally rerun GC025
is substituted. Missing reference chunks fail the pipeline instead of
selecting dates by forecast performance. Five- and ten-day paired moving
blocks stay within each contiguous date window; intervals are pointwise
and conditional on these periods. Per-start physical MSE is saved; pool MSE
before taking square roots. These are explicitly RMSE diagnostics.

Resource evidence: matched legacy K22 pilot15750888 completed8m31s at
10.87GiB host RAM. New pilot requests16GiB/20min; nine production shards
of7–8 starts request16GiB/45min, unthrottled. Data readiness precedes GPU
work, successful pilot precedes production, and report follows all shards.
Internet-enabled staging/reference scoring runs on della-vis1, PID1317838,
with failure markers and log`logs/stage_and_references.log`. CPU readiness
and report jobs have bounded waits. Submission receipts are in`submissions/`.

Validation before submission: all four Python files parse; shell scripts pass
`bash -n`; a synthetic two-period bootstrap check preserves each period's
sample allocation and never resamples across the temporal gap. GPU checkpoint
compatibility, complete data and reference coverage, and the scientific
comparison remain pending. No manuscript figure has been replaced.

Next: inspect the pilot and staged input parity, verify all69 starts×40
leads for all four sources, review both uncertainty sensitivities and the
generated Figure5/9 analogs before incorporating results into the article.

Preflight evidence after submission: the first input(Feb7) passed exact time,
grid,13-level, finite-value and float32 preparation checks
(`audit/first_input_validation.json`). One6-hour lead from both public
references scores successfully for all12 diagnostics against the same truth
(`audit/reference_first_lead_validation.json`). An isolated synthetic69-start
report check recovered known RMSE ratios and paired intervals, generated
all four PNG/PDF artifacts and rejected a mismatched truth digest; its
script/result are retained in`audit/check_report.py` and`report_validation.json`.

Source audit found the current legacy factory's stateful-default handling and
GraphCast temporal configuration differ from the shared Oct17 pilot checkout.
The normalizer is byte-identical. For the actual legacy128-inner, two-layer,
one-group temporal path, a shared-parameter40-step CPU comparison against
that checkout gives exactly zero maximum output/state difference in both
float32 and bfloat16(`audit/legacy_temporal_forward_parity.json`). This
isolates temporal-module equivalence under the checked settings; it is not
full-GPU forecast parity. No source snapshot or forecast output was modified.

### October18 — Staging repair and preserved recovery

The first stager completed all69 input files, then failed at the four extra
truth times because the caller passed an integer hour to`write_truth`, which
requires the complete hour-lookup mapping. Data gate15792340 failed; pilot
15792341, production15792342 and report15792343 were automatically cancelled
without running. This is an execution failure, not a scientific result.

Corrected the caller in the working tree and tested the writer through a
synthetic constant native field: all12 common-grid diagnostics, timestamp
identity and readiness checksum passed. Frozen failed source/results remain
unchanged. The new campaign
`artifacts/evaluations/k22_common_grid_2022_recovery_20261018` has protocol
hash`6fcbebf34fc60fdd7ec08a8b8d8638eeb59a187ac79b25478e760f09f294bfdf`.
It records its parent protocol and symlinks all69 existing verified inputs;
dates, checkpoint, metrics and requested resources are identical. Staging/
reference scoring restarted on della-vis1 as PID1454770. Replacement chain:
15792878 →15792879 →15792880_0–8 →15792881. Only extra truth and the matched
reference scores remain to be generated before scientific results exist.

Recovery staging is now complete:69 input files and456 distinct verification
times, including four newly generated truth times. Data gate15792878
completed57s with exit0. GPU pilot15792879 is running on della-l07g4;
production and report retain their successful-completion dependencies.
Full reference scoring is active on della-vis1. The full69-date model
comparison remains pending.

First four starts have now been scored for both archived references. Pilot
startup has no reported error as of its first running minute; no completed
GPU sample yet. A separate optional read of the historical1° Zarr inputs
stalled and was stopped, with no parity conclusion. The independently
completed first-input finite/grid/time/float32 checks remain valid; do not
claim bitwise parity with that historical store from this attempt.

### October18 — Completed matched68-date comparison and Figure5/9 regeneration

Pilot15792879 completed7m52s; all nine production tasks15792880_0–8 completed
in9m01s–9m47s, all exit0. The reference scorer completed68 starts, then
failed because DeepMind's archive has no unique forecast for Dec31,2022
00UTC. Its failure marker correctly stopped report15792881 after8s.
Earlier planning incorrectly assumed that archive's end date was inclusive.

The user explicitly approved using the68 matching dates for all models.
Only Dec31 is excluded, solely for reference availability, independently
of forecast performance. Existing69-start forecasts and partial references
remain unchanged. All68 retained dates have complete40-lead scores for
K22, its GC1° backbone, archived DeepMind GC025 and ENS. December's final
range is now Dec12–30(19 starts); the other four periods are unchanged.

The separate frozen CPU analysis is
`artifacts/evaluations/k22_common_grid_2022_matched68_20261018`, protocol
hash`b4c66901988e0ccf15dbc40c67d236ed853c72d553da9b37b6d5586bf2b8a228`.
It records the parent score protocol, original69-start coverage, the
availability exclusion and new68-start manifest. The report code now derives
sample counts from the manifest; parent forecasts/references are reused
without changing their recorded protocol hashes. No GPU rerun occurred.

Completed artifacts are`report/{summary.json,report.md,validation.json}`;
CSV and Figure5/9 PNG/PDF outputs are under the corresponding
`plots/analyze_models/{data,images}/resolution_eval/k22_common_grid_2022_matched68_20261018/`
directories. Visual inspection confirms legible labels, matching68-date
titles and all12 panels. Independent regrouping of all130560 CSV rows
reproduces the summary RMSE to2e-14 relative tolerance; each model has exactly
68 starts×40 leads×12 diagnostics, without duplicate entries or Dec31.
Reference truth hashes and lead schedules match the local forecast records.

|Diagnostic|Day10 K22 RMSE reduction vs archived GC025|Sustained positive point estimate from|
|---|---:|---:|
|Z500|9.388%|Day6.5|
|T2m|8.120%|Day7.0|
|U850|9.866%|Day5.5|

GC025 has lower short-lead error; K22 has lower day10 RMSE on all12
diagnostics, with reductions5.465–10.465%. Five- and ten-day conditional
block intervals have positive lower endpoints at day10 for all12 diagnostics.
These are pointwise intervals conditional on five seasonally uneven periods,
with limited independent weather; do not turn them into an annual or
selected-M1 superiority claim. The archive-versus-public-checkpoint numerical
question from e006 remains distinct and unresolved. Manuscript source figures
have not been replaced by this regeneration.

### October18 — Remove uncertainty shading from Figure5/9 previews

At the user's request, regenerated both matched68-date PDF/PNG figures with
point-estimate curves only. The working report script now has a `plots` phase
that reads the completed summary without rescoring or recomputing intervals.
Saved uncertainty estimates, scores, dates and frozen source are unchanged;
`plot_metadata.json` alongside the figures records the summary and plotter
hashes and the absence of uncertainty shading. Manuscript files remain pending
the user's review of these previews.

### October18 — Incorporate approved Figure5/9 replacements into the arXiv draft

The user approved replacing Figures5/9 and explaining their construction.
`Overleaf/arxiv_versoin.tex` now uses the exact unshaded PDF previews, copied
with their PNGs into `Overleaf/Graphs/common_grid_2022_matched68/`.
The accompanying `figure_sources.json` records hashes, the checkpoint and
archive identities, all68 dates, the recovered exclusion origins, scoring
settings, source digest and regeneration command. Historical assets used by
other manuscript drafts remain intact.

Section4.4 and both captions now identify the historical K22 experiment and
matched ENS-mean normalization. AppendixC specifies the complete date windows,
Dec31 availability exclusion, independent forecast feedback, zero initial
memory, native input subsampling, reference stores, conservative verification
remapping, pooled physical RMSE and conditional bootstrap procedure. It
discloses the incomplete K22 averaging/selection history and unidentified
exact checkpoint behind the DeepMind archive, and limits claims to these
five periods rather than annual or selected-M1 performance.

Independent methods review found the text consistent with the frozen protocol.
pdfLaTeX/BibTeX and resolved-reference passes completed; Figures5/9 remain on
pages9/24 and methods on pages24–25 of the27-page PDF. Rendered pages were
visually inspected. No new LaTeX warnings or overfull boxes were introduced;
the existing1.59732pt box at lines115–123 remains. The build and before-edit
backup are in `Overleaf/build/arxiv_common_grid68_20261018/`.

### October18 — Public-checkpoint comparison preview; incorporation deferred

Reused the publicGC025 122-start2022 scores and the approved68-start campaign
on their24 shared dates. All truth hashes, valid times and leads match.
`scripts/analyze_models/plot_public_gc025_archive_comparison.py` recomputes
GC1, publicGC025, archivedGC025 and ENS errors on those24 dates. Outputs are
under `plots/analyze_models/{data,images}/resolution_eval/public_gc025_archive_matched24_20261018/`.
Independent pooling of all46080 CSV cells reproduces the saved RMSE.

At day3, publicGC025 versusGC1 RMSE differences are +1.69/−0.05/+0.94%
for Z500/T2m/U850; versus the archive they are +10.68/+10.79/+7.26%.
Similarity to GC1 is limited to middle leads; at day10 publicGC025 has
6.05/1.39/5.60% lower RMSE than GC1. The archive's exact weights and inference
configuration remain unidentified; do not claim proven different weights.

The user initially chose a separate AppendixC figure, then explicitly asked
to see it before addition. Manuscript source was restored exactly to the
before-comparison version, and the root PDF was never replaced. The new plot
and proposed text remain previews only, pending review. Proposed text/build
are in `Overleaf/build/arxiv_gc025_provenance_20261018/`; unused asset copies
are in `Overleaf/Graphs/gc025_provenance_matched24/`. Approved68-date Figures5/9
remain in the manuscript unchanged.

### e008 — October18: Switch to principal M1/public checkpoints; validate repeated truth

Execution: completed. Outcome: supported for the specified common-grid
comparison on eligible paired samples; manuscript incorporation pending review.

The user selected the main M1 correction and all122 original2023 starts for
the proposed replacement comparison, and asked to finish validation first.
The24-date historicalK22/public-checkpoint alternative was stopped before
creating any new files. Existing approved68-date manuscript Figures5/9 remain
unchanged pending validated replacement previews.

The November9 Z500 spike is shared by all three models: at30/36/42h, public
GC025 RMSE is212.65/354.58/502.14, GC1 is212.16/354.99/505.90, and M1 is
210.63/355.26/507.87 m2 s-2. The verification Z500 arrays for November10
06/12/18UTC are bitwise identical despite distinct timestamps. The separate
native1degree scoring also spikes, so this is not evidence of a defect
specific to GC025 inference.

`scripts/analyze_models/audit_common_grid_truth_repeats.py` checked all1492
common-truth NetCDF files, their receipts and their hashes against every
forecast record. Exact repeated complete fields occur only at those three
timestamps, for Z500,Z850,T500,Q700,Q850. This flags9 lead/start pairs per
model: November3 at174/180/186h, November6 at102/108/114h, and November9 at
30/36/42h. The audit does not exclude scores or identify replacement values.
Its evidence is `artifacts/evaluations/graphcast_common_grid_2023_20261015/audits/truth_repeats_20261018.json`.

Fresh read-only checks confirmed both public checkpoint hashes and all three
normalization hashes against the frozen protocol; checkpoint sizes/MD5 also
match the previously saved official bucket metadata. No additional active
inference-path defect was found. The existing independent January1 upstream
rollout remains a limited single-start check. The next action is to compare
the repeated fields against live ARCO and separate raw ERA5 files, delimit
the source defect, then repair or explicitly handle invalid verification
before generating the new122-start previews. No scores or manuscript assets
have been changed and no new GPU jobs have been submitted at this point.

#### Completed source audit and approved analysis

Live ARCO Z500/Z850 at November10 06/12/18UTC match01UTC bitwise. The separately
stored raw daily Z500 NetCDF repeats01--23UTC too. Independent NCAR samples at
06/12UTC evolve normally, corroborating an ARCO archive defect. NCAR fields
were not adopted as replacements. SourceURLs, checksums, sampled fields and
native-data checks are under the campaign's `audits/truth_repair_20261018/`.

The user approved identical diagnostic/time exclusions for all three models.
All122 forecast starts remain; five diagnostics use121 paired valid samples
at leads30/36/42/102/108/114/174/180/186h and all other cells use122. This is45
of58560 diagnostic cells per model. No selected input frame overlaps the
invalid hours. Original scores, source files and manuscript assets remain
intact. No GPU reruns or Slurm submissions were needed.

New analysis and Figure5/9 previews are under
`plots/analyze_models/{data,images}/resolution_eval/public_checkpoints_2023_qc_20261018/`,
generated by `scripts/analyze_models/plot_public_checkpoints_2023_qc.py` and
`scripts/analyze_models/bootstrap_public_checkpoints_2023_qc.py`. The CSV
retains all175680 original scores plus eligibility flags. Independent pooling
agrees within5.73e-16 relative error. Masked paired-block bootstrap, including
block-length and pilot sensitivities, was independently checked; every day10
interval remains bitwise identical to the original report.

The figures show physical-unit RMSE on the common1.5degree grid without ENS
normalization or shading. Day10 M1 RMSE reductions versus publicGC025 remain
7.6785--11.3892% across12 diagnostics: Z5008.4640%, T2m10.2420%, U85010.0657%.
All day10 lower confidence bounds remain positive across the prescribed
sensitivities. This compares fixed public models with different training,
levels and mesh; it does not isolate resolution or establish skill versus
the unidentified WB2 checkpoint. The upstream inference check remains a
limited single-start check, not proof of full-year equivalence.

The user requested omission of exclusion detail from the presentation.
Figures/captions omit it; proposed methods retain only the accurate
``121--122 matched valid samples per lead and diagnostic.'' Full details
remain in reproducibility files. `figures_methods_preview.tex` beside the
images contains proposed captions/methods. Root arXiv source/PDF is unchanged.

The same source defect affects native-loss verification. A separate whole-step
sensitivity omits9 of4880 loss cells identically and retains122 starts:
M1 reduction14.42401956% becomes14.43160406%, matched reset9.16266207% becomes
9.16818853%, and their advantage5.26135748pp becomes5.26341553pp. Day10 is
unchanged. These remain sensitivity estimates, not replacements for native
manuscript values; saved aggregate losses cannot isolate affected levels.
