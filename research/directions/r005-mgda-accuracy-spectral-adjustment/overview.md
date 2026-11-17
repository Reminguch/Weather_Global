# r005 — Can MGDA improve spectral fidelity while preserving exact GC accuracy?

Category: training
Status: active
Updated: 2026-11-15 (America/New_York)

## Question and hypothesis

Can a continuation from a good MSE-trained residual model improve spectral
fidelity without losing exact GraphCast grid accuracy? Test MGDA on exact GC
grid MSE and the nonnegative spectral adjustment
`AMSE_spectral - MSE_spectral`, with measured finite-step diagnostics and a
fixed training budget followed by forecast evaluation.

Both terms use the existing full-field difference-standard-deviation, variable
and pressure weights. The adjustment uses matching spectral MSE, never grid MSE,
and is computed directly to avoid cancellation. Their sum is a grid-MSE-based
adjusted objective; it is not exactly the existing spherical AMSE.

## Motivation and comparison

The earlier GC/power MGDA continuation used a different power-error surrogate.
The October 12/13 amplitude/pattern split optimized A and C, with GC loss only
diagnostic. That split substantially degraded forecast skill in both arms.
Its first-order checks did not enforce decreases after finite parameter updates.
See the preserved [diagnosis](../../../current_experiments.md#split-amse-failure-diagnosis--2026-10-17)
and [original split protocol](../../../docs/experiments/v24_amse_split_mgda_submission_plan_2026-10-12.md).

The new pilot compares MGDA, MSE-only, and a fixed sum of the new pair. All use
the same pair-gradient RMS preconditioning rule, so the control does not change
to a single-gradient second-moment estimator.

## Current understanding

November 15 [authorized presentation revision](experiments.md#v5-variable-spectral-presentation-20261115):
both v5 manuscripts now retain combined 83-channel curves in Section 4.6,
with separate day-5/day-10 amplitude/coherence illustrations for T2m, Z500,
U850 and Q100 in Appendix C. The selected fields' native-grid RMSE reductions
versus R1 are 1.37/2.02/1.76/0.84% at day 5 and 4.98/4.60/5.56/-1.01%
at day 10. This is explicitly an illustrative post-hoc selection, including
an adverse case, not a best/worst ranking. Z500 replaces T400 in the manuscript;
the exploratory T400 artifacts remain available. The sqrt(0.75) reference
applies to power deficit only within an individual channel and scale; its
combined-curve interpretation is weighted amplitude. Original equal-degree
mismatch and full-channel exports are retained, including the all-83-channel
day-10 deterioration versus R1 under that metric. No forecasts were rerun.
The [follow-up reference explanation](experiments.md#v5-spectral-reference-explanation-20261115)
ties the illustrative sqrt(0.75) line to Subich Section 3.1 and makes the
day-5/day-10 contrast explicit: first sampled crossings are 16/16 and 7/12
for M1/R1. Nearby 20% and 30% reference choices preserve that qualitative
contrast. Full curve proximity supports day-5 similarity; a shared crossing
alone does not establish equivalence.
The authorized appendix follow-up interprets ranges below the line for the
four illustrated fields: U850 gives the day-10 accuracy/attenuation tradeoff,
T2m only isolated dips, Z500 excess fine-scale power with low coherence, and
Q100 an adverse accuracy result despite similar attenuation.

November 15 [per-channel audit and illustration selection](experiments.md#v5-variable-spectral-audit-20261115):
the matched 122-date pools contain exact native-grid MSE for all 83 channels,
so accuracy can be compared with spectra without substituting truncated
spectral error. M1 lowers RMSE versus R1 in 81/83 channels at both day 5 and
day 10. With the supplementary truth-power-weighted mismatch, 31 channels
improve both metrics at day 5 and eight at day 10; the other improved-accuracy
channels show a spectral tradeoff. Under the original equal-degree mismatch,
all 83 channels worsen at day 10. Neither weighting should be presented as a
complete measure of forecast quality: relative-error averaging emphasizes
weak tails, while power weighting emphasizes energetic and potentially static
large-scale structure. Post-hoc examples T2m, T400, U850 and Q100 show a
power-weighted joint improvement, an earlier joint improvement in both
mismatch definitions, an accuracy/attenuation tradeoff, and a modest day-10
deterioration, respectively. Full channel tables accompany the figures.
Per-channel 20/25/30% power-deficit crossing times are diagnostic, with
recrossings and first-lead censoring retained. The manuscript was unchanged
at that exploratory stage; the later authorized revision is linked above.

November 15 [annual completion and v5 replacement](experiments.md#v5-spectra-matched122-20261115):
the matched spherical evaluation is complete for all 122 standard 2023 starts,
40 six-hour leads and 83 channels. All nine reset tail jobs and their annual
report succeeded on November 14; the November 15 audit verified the exact
date union, pooled-file digest and finite arrays. The current optimal-filter
GC/M1 campaign likewise contains 122 dates. Both v5 sources now use this
separately archived annual cohort in place of the frozen 70-date comparison.
The completed replot shows M1's exact-loss reductions versus R1 of 2.85% at
day 5, 5.76% across the rollout and 9.26% at day 10. Power mismatch is higher
by 0.50, 1.51 and 7.96 percentage points, respectively. M1 improves both
aggregate metrics versus filtered GC, while raw GC retains the smallest
power mismatch. Thus the annual result supports an accuracy/spectral-fidelity
tradeoff, not a memory advantage at identical spectra. The four comparators
and all five plots use the same annual cohort. Finite reset rerun drift
remains advisory: pooled loss differs by +0.0002531
(about 0.00244%) from the archived reset evaluation, while the maximum local
relative difference is 3.29457% against a 0.1% reference. Annual coverage does
not add seed replication or spectral confidence intervals. The dated entries
below preserve the earlier interim evidence and submission history.
Both v5 PDFs compile with zero LaTeX warnings, and the checked annual table,
captions and figures are readable without clipping or overlap; the clean main
text remains ten pages. See the
[manuscript validation](../../../artifacts/manuscripts/v5_spectral_matched122_20261115/validation.json).

November 14 [annual coverage audit and reset tail extension](experiments.md#matched-reset-spectra-tail30-20261114)
corrects the scope of the earlier 92-date explanation: raw GC/M1 spherical
moments already exist for all 122 annual starts in the November 7 and 9
smoothing campaigns. Only the newer optimal-filter rerun stops at 92 saved
dates. The correct matched-reset October 4–December 30 spectra remain absent
from the searched outputs. The user authorized nine GPU-test jobs, three or
four dates each and 50 minutes per job, to collect those 30 dates. Submitted
array `17574447_0–8` is followed by CPU report `17574448`; the initial check
found all nine forecasts pending and the report waiting on its dependency.
The original 92-date reset
campaign and report `17572314` have now completed with all eight shard checks.
Reuse those 92 reset dates for the eventual annual report and retain the frozen 70-date
v5 record until a separately validated matched annual comparison is available.

November 14 [v5 matched 70-date update](experiments.md#v5-spectra-matched70-20261114)
replaces the active 36-date spectral plots, table and numerical discussion in
both v5 sources at the user's request. All four predictors use the same
January 1–July 27 starts. M1's exact-loss reduction versus R1 is 2.92% at
day 5, 5.93% over the rollout and 9.70% at day 10; day-5 power mismatch is
69.50% versus 68.97%. The tradeoff interpretation remains, without a new
uncertainty claim. Historical 36-date artifacts are preserved. The 122-start
annual protocol remains the target; the current 92-date spectral campaign
does not include the final 30 reset dates.

November 14 [final GPU-test wave](experiments.md#matched-reset-spectra-wave3-20261114):
70/92 dates are complete and validated. Released GPU-test tasks `17572313_6–7`
now cover the last 22 dates at 50 minutes, followed by report `17572314`.
Pending originals `17553196_6–7` and report `17560837` were replaced safely.
Completed spectra, frozen computation and fixed 36-date manuscript artifacts
are unchanged. The initial check found task 6 configuring and task 7 pending;
both subsequently completed in 23m42s/23m28s, followed by the successful 92-date
report. No annual 122-date scientific comparison follows from that milestone.

November 14 [Section 4.6 smoothing explanation](experiments.md#smoothing-section46-20261114):
the user replaces the brief Section4.5 paragraph with a dedicated Section4.6.
The main text now explains spectral relevance, cites GraphCast/AMSE, outlines
the output-filter procedure and shows day-5 spectra in Figure 7 plus the
four-model percentage summary in Table 2. Both metrics are explicitly labeled
as reductions relative to raw GraphCast. Appendix E retains detailed methods.
The fixed36 data and all figure bytes are unchanged; day5supports the design
with a small aggregate spectral cost, while day10has higher absolute MSE and
lower coherence. Space optimization is deferred by the user.

November 14 [36-date Appendix E revision](experiments.md#appendix-e-matched36-20261114):
`Overleaf/v4.tex` now uses the fixed January 1–April 16 cohort throughout
Appendix E, its table and all eight figures. The main text retains one short
red paragraph. Day-5 carry/carry improves MSE by 2.84% versus reset/reset with
0.50 percentage points higher aggregate power mismatch; the larger day-10
spectral cost remains explicit. All four comparators are retained. The 36-date
data, source pins and plotting exports are archived separately and must not
be overwritten when the larger campaign completes. Main 122-date results
and their uncertainty are unchanged.

November 14 [second GPU-test wave](experiments.md#matched-reset-spectra-wave2-20261114):
completed coverage remains 36/92 dates. New `17560836_3–5` covers 34 dates on
GPU-test at 50 minutes; retained `17553196_6–7` covers 22 dates on regular GPU
at 61 minutes. Report `17560837` follows all remaining tasks. Replacements
are released, completed outputs and frozen computation are unchanged, and
no new spectra from this wave are claimed.

November 14 [same-cohort 36-date comparison](experiments.md#matched-spectra-36-20261114)
uses newly completed full-campaign reset shards 00–02 and re-pools the matching
GC/M1/filter moments on exactly those dates. Every updated loss, spectrum,
amplitude/coherence plot and per-feature plot uses the same January 1–April 16
sample. Carry/carry has 2.84% lower day-5 MSE than reset/reset, with similar
aggregate power mismatch (68.66%/68.16%). Its rollout/day-10 MSE advantages
are 6.01%/10.20%; day-10 mismatch differs more (84.47%/75.74%). These point
estimates do not establish an advantage at identical power or annual coverage.
The four-date preview remains historical; manuscript results are not changed
by this plot-only update. The new plot directory is
`v24_literature_spectral_matched36_20261114`, including both feature leads.

November 14 [day-10 per-feature extension](experiments.md#feature-spectra-day10-20261114)
adds all day-10 power, amplitude and coherence counterparts to the existing
four-date `by_feature` folder. Every plot includes GC, filtered GC, M1 and
reset/reset. Three seven-page atlases cover all 83 channels; best/worst fields
are ranked separately at each lead, while familiar fields remain fixed.
The saved 240-hour moments are reused and day-5 outputs are preserved.

November 14 [accelerated reset spectra](experiments.md#matched-reset-spectra-fast-20261114):
`17553195_0–2` on GPU-test (50 minutes; 36 dates) and `17553196_3–7` on regular
GPU (61 minutes; 56 dates) replace pending `17494525/17494526`, with report
`17553202` after both arrays. Frozen computation and the full 92-date cohort
are unchanged. All tasks remained pending at final inspection; estimated
execution is 25–30 minutes per shard after starting, excluding queue delay.
The three GPU-test shards have subsequently completed; the larger interim
36-date result is described above. The full 92-date result remains incomplete.

November 14 [required Appendix E comparators](experiments.md#appendix-e-four-models-20261114):
the user requires reset/reset in every Appendix E experiment, table and plot,
and excludes filtered M1 from the active comparison. Use raw GraphCast,
development-filtered GraphCast, raw carry/carry (M1), and raw matched reset/reset
on identical dates. Only four dates currently have completed spectra for all
four; the larger reset spectral campaign has no completed statistics/report.
All active Appendix E comparisons therefore use that four-date cohort and are
labelled descriptive point estimates. Figures 11–15 are regenerated; 16–18
already satisfy the comparator/cohort requirement. Prior 92-date filtering
results and intervals remain archived, not mixed into these four-date plots.
The principal 122-date memory-policy results and uncertainty are unchanged.
Future Appendix E extensions must retain all four comparators on a common
completed cohort, rather than append a reset row from a different sample.
Table 7 now reports each metric as a percentage reduction relative to raw GC;
negative values indicate deterioration. The underlying scores are unchanged,
and the two normalized metrics remain distinct quantities.

November 14 [appendix layout](experiments.md#smoothing-appendix-layout-20261114)
shows the original carry/carry and reset/reset spectra together at days 5 and
10, alongside their matched four-date native losses. All detailed smoothing
methods, results and figures are now in Appendix E, with one short main-text
summary. The 92-date filtering and four-date memory cohorts remain separate.
Carry/carry's lower rollout and day-10 MSE accompanies greater attenuation;
the displayed forecasts have not been amplitude-matched. The new physical
power plots retain field-specific units and ERA5 reference. No new forecasts
or training were run.

November 14 [equal-power diagnostic](experiments.md#equal-power-memory-20261114)
directly tests the proposed lower-MSE-at-identical-spectrum interpretation on
the four matched preview dates. Positive rescaling to a symmetric common
forecast-power target gives carry/carry 0.9131% higher weighted represented
spectral MSE over the rollout, 0.3595% at day 5 and 0.6048% at day 10. The
ordering persists for carry-power, reset-power and truth-oracle targets and
when degree zero is included. These small descriptive differences do not
establish significance or a general absence of memory benefit. The raw
native-grid and full-cohort memory comparisons remain distinct. The current
data do not support stating that carry/carry is better at the same spectrum.

November 14 [red manuscript revision](experiments.md#smoothing-manuscript-20261114)
reports the better measured accuracy/power-mismatch tradeoff relative to the
tested filtered-GC baseline, while preserving the raw-GC spectral advantage
and the separate scope of the four-date memory preview. A new paired interval
for raw M1 versus filtered GC is 4.2643% [3.3738,5.3438]% lower rollout loss;
day 10 is 4.6311% [3.1632,6.5746]%. These reuse saved 92-date native losses
and the frozen reporter's 10,000 five-start block draws. No spectral CI or
memory-related Pareto improvement is inferred. The full 83-channel day-5
spectral export, fixed-checkpoint figures and code are archived with the paper.

November 13 [per-feature spectral preview](experiments.md#feature-spectra-20261113)
resolves day 5 into all 83 variable/pressure channels on the same four dates.
T2m preserves power relatively well; precipitation and mid-atmosphere vertical
velocity show substantial deficits. Z500 and T850 retain most large-scale
power but have large relative small-scale errors. Rankings depend on scale
weighting: equal-degree relative mismatch and ERA5-power-weighted mismatch
are both retained, with separate amplitude, physical-power and coherence plots.
M1 has lower equal-degree power mismatch than raw GC in only 3/83 channels,
and than reset in 30/83. These descriptive counts do not establish significance
or spectral superiority; they do not measure the exact GraphCast loss.

November 13 [matched reset/reset spectral extension](experiments.md#matched-reset-spectra-20261113)
adds the original matched reset SWA to the exact-loss figure on the same 92
dates. Its archived evaluation contains no spectra, so the other six figures
await a raw spectral rerun with the existing checkpoint. Original jobs
**17494525/17494526** were superseded on November 14 by arrays
**17553195/17553196** and report **17553202**, which will regenerate all seven
plots automatically. Whether memory improves the
accuracy/amplitude/coherence tradeoff remains unmeasured; no filtered-reset
variant, retraining or filter fitting is included.
Four-date preview **17495398** completed in **14m35s** on `gpu-test` within the
user-requested 59-minute limit. All seven figures now include reset/reset in a
separate preliminary folder. M1 has1.60% lower rollout loss but worse power
matching; day5 curves are close and do not show M1 dominance. Finite reset
rerun drift (~0.29% pooled) is retained. The full92-date comparison remains queued.

November 13 [literature-style spectral figures](experiments.md#literature-spectral-20261113)
reuse the pinned 92-date optimal-filter subset. New AMSE-style weighted amplitude
and coherence panels, GraphCast-style spectral RMSE and physical spectra, and
separate exact native-grid loss curves retain 6-hour, 5-day and 10-day evidence.
At day 5 raw M1 improves exact loss by 2.2379% against filtered GC; filtering
both gives 1.6333%. Amplitude deficits and modest coherence changes remain
distinct. These are fixed-checkpoint point estimates, not five-seed spectra.
The [presentation plan](../../../docs/reports/v24_spectral_presentation_plan_2026-11-13.txt)
keeps the day-5 diagnostic prominent and the full lead range available. Original
AMSE figures were extracted for reference; its reported ensemble-mean RMSE is
not a directly comparable deterministic GraphCast loss. No jobs or manuscript
files changed.

November 13 [training-duration and K synthesis](experiments.md#training-steps-k-20261113)
reuses completed evaluations. M1 at fixed K20 has non-monotonic raw MSE and
spherical power mismatch at total updates 1,000/1,250/2,500 on four 2022 dates.
The older relative-power MGDA continuation lowers zonal mismatch 14.750% but
raises exact MSE 1.144% after 2,000 additional updates on eight 2023 dates.
The 32-date GC–Mamba K sweep supports an MSE comparison, but has no matched
saved spectra across K; fixed BPTT24 also changes teacher-prefix length.
See the [summary and four figures](../../../plots/analyze_models/data/resolution_eval/v24_training_steps_and_k_20261113/REPORT.md).
These are separate single-seed cohorts and spectral definitions, not a joint
training-step × K response surface or a new full-GC result. No jobs changed.

November 13 power-spectrum addition to [e015](experiments.md#e015): on the
same 92 starts, filtering improves MSE but worsens amplitude fidelity. Rollout
power error is 59.4141% / 78.3995% for raw / filtered GC and 67.5267% / 77.6371%
for raw / filtered M1. This equals the mean absolute power-ratio error across
channels, positive degrees and leads, after pooling dates. Raw M1 beats filtered
GC on both aggregate metrics at every lead; versus raw GC it has lower MSE but
higher power error. The evidence does not establish a joint raw M1 improvement
or universal fine-scale power recovery. See the
[joint report and figures](../../../plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113/REPORT.md).
The full-year power-report job `17470053` waits for annual report `17430311`.

November13 interim update to [e015](experiments.md#e015): fitting on all32
2022 starts completed;92/122 annual starts are evaluated (January1–October1).
Optimal filtering narrows the gap: independently filtered M1 still reduces
rollout loss4.9850% versus filtered GC (paired95% interval4.1243–5.9559%) and
day10 loss8.3204%. GC/M1 own-raw filtering gains10.8808%/0.7528% exceed observed
aggregate rerun differences about0.04%. Thirty October–December starts remain
queued, so this is not the full2023 result. See the
[interim readout](../../../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/interim_20261113/REPORT.md).

November 12 smoothing correction, [e015](experiments.md#e015): the user requested
GraphCast-style degree-wise optimal filtering, fitted on 2022 and evaluated on
v4's 122-start 2023 cohort. The earlier Gaussian/four-band controls do not
establish this flexible-filter comparison. Independent per-lead/channel/degree
least squares and pure spherical reconstruction are implemented; GC/M1 chain
17430308 → 17430309 → 17430310 → 17430311 is submitted and pending. Raw/projection
controls, canonical v4 exact-loss denominators, and separate spectral scores
are included. Scientific outcome remains pending; inspect the completed report
before concluding whether M1's advantage survives this filtering class.

Latest November12 decision, [e014](experiments.md#e014): the user requires
**constant3e-7 from the first update**, with zero warmup and no cosine decay.
All five full-GC lambdas0/0.25/0.5/0.75/1 therefore start fresh from public GC;
the earlier lambda0 cosine checkpoint cannot be reused for this schedule.
Arrays17429091→17429092, history17429093 and completion17429094 are submitted.
There is no new constant-rate training/evaluation result. The four-arm cosine
restart e013 and its downstream jobs, plus oldlambda0 continuation/history,
were canceled while pending. Existing cosine checkpoints and their measured
scores remain historical, not evidence for the constant-rate recipe.

GC–Mamba training/history17406861/17406862/17406863 is unchanged. New mixed
raw/SWA and annual pipelines use the five new constant-rate full-GC arms and
the existing M1-recipe GC–Mamba campaign; both depend on17429092:17406862.
Submitted mixed chain17429215→17429216→17429217 and annual controller17429218
implement those replacement dependencies; no new evaluation result is claimed.
The full-GC1e-5 exclusion remains in force. Cross-family recipes still differ
in feedback gradients, loss arithmetic, optimizer schedule and data order.
The [current protocol](../../../docs/experiments/v24_full_gc_constant_lr3em7_2026-11-12.md)
records the fresh initialization, source-preservation evidence and current jobs.
Lianghong's corrected portable package also uses constant3e-7/no-warmup for
her four fresh full-GC horizons; no remote execution is claimed.

### Historical November12 decisions superseded by e014

Earlier November12 activation: the user approved four fresh full-GC positive-lambda
runs at3e-7. Arrays17421838→17421839 now replace the rejected continuations;
lambda0 retains17049163_0 at the same rate. Thus the active full-GC sweep uses
peak3e-7 for all five lambdas. Old high-rate checkpoints remain excluded and
are not initializers. GC–Mamba training/history17406861/62/63 is unchanged.
At that earlier activation, shared downstream controllers were rebuilt around the three actual training
sources: raw/SWA17422018→17422019→17422020 and annual17422049 are submitted.
New training is pending; the eligible existing3e-7 lambda0 evidence
below remains the only low-rate full-GC result. See [e013](experiments.md#e013)
and the [superseded cosine protocol](../../../docs/experiments/v24_full_gc_positive_lambda_lr3em7_restart_2026-11-12.md).

November12 user correction: **exclude the November full-GC peak1e-5
lineages from all scientific comparisons**; retain their artifacts only for
provenance/explicit diagnosis. Future full-GC work uses peak3e-7. At that rate,
only lambda0 has results: exact rollout-loss reduction4.6434% at500 updates
on32 development dates, and4.6934% at1250 onfour dates. No positive-lambda3e-7
result is available. Submitted positive-lambda full-GC continuations still
use1e-5; a replacement decision is pending. This supersedes older statements
below treating those high-rate arms as eligible comparisons. GC–Mamba's
M1-matched rate is unchanged. See the [comparison policy](../../../docs/experiments/v24_full_gc_comparison_policy_2026-11-12.md).

November12 activation, superseding the earlier two-family review: the user
approved restarting **GC–Mamba only**, leaving full GC/DeepMind unchanged.
All five fresh GC–Mamba lambda arms now use M1's LR, two-stage schedule, native
BF16 base loss, stopped weather gradients and historical deterministic order.
Array17406861 runs initial500 plus main750; array17406862 exactly resumes
main750→2000, totaling2,500 updates. History17406863 and finish17406864 follow.
These jobs are submitted/pending; no new scientific result is available.

Old GC–Mamba17062358/59/60 and obsolete joint final17353172/73/74 were cancelled.
Full-GC17049163/64/65 and saved-checkpoint diagnostic17404694→17404732 remain
unchanged. New mixed final chain17406897→17406898→17406899 compares the new
GC–Mamba endpoints with unchanged full-GC endpoints, raw and SWA, on the same
four development dates. Its materialization waits for17049163 and17406862.
The new finish hook also retains development fitting and annual smoothing,
with a separate join controller combining the two campaigns' actual reports.

This is a comparison of two documented training recipes, not an
identical-training-protocol architecture experiment: full GC retains full
weather gradients, FP32 grid loss, its continuous schedule and original rates.
The prior review is preserved in [e012](experiments.md#e012); the
[protocol](../../../docs/experiments/v24_m1_matched_lambda_restart_2026-11-12.md)
records the narrower authorization and actual activation. The separate
full-GC horizon-training hold remains in place.

November12: the user requests M1 versus November lambda0 at accessible matched
intermediate budgets. Four new forecasts17404694_2–5 and report17404732 compare
all three methods at1000 and1250 total updates, reusing completed1250/reference
forecasts. M1 main500/750 correspond to total1000/1250. Evaluation settings and
full-GC weight loading passed inspection; training recipes differ in weather
gradients, schedule, loss arithmetic and order. Existing32-start results at500
total updates give M1/full-GC/GC–Mamba MSE9.8199/10.3373/10.3144, so the earlier
gap is already present at equal budgets, though those saved evaluations used
their respective source versions. The new frozen-runtime comparison is pending.
[Protocol and audit](../../../docs/experiments/v24_matched_budget_comparison_2026-11-12.md).
The diagnostic remains unchanged; its previously queued final2500 production
chain was replaced by the later GC–Mamba-only activation above.

Historical November11 follow-up, superseded for future production by the
November12 mixed chain above: the user requests the final2500-update comparison,
including raw and SWA weights, MSE and actual power spectra. The ten existing
continuations are still queued; dependent materialization17353172 →
evaluation17353173 → report17353174 is now submitted. Fixed SWA source steps
are1000/1500/2000/2500, aligned with M1's total-update window. The same four
development dates and completed M1 references are retained. Final fresh-model
results remain pending; [protocol and completed M1-only comparison](../../../docs/experiments/v24_final_checkpoint_comparison_2026-11-11.md).

November11: both e011 pilots completed and were verified. On the matched
four2022 starts/40leads, M1 raw/SWA MSE is9.4240/9.1717 (15.35%/17.62%
reduction versus publicGC11.1331), and spherical power error75.94%/71.89%.
Thus removing SWA worsens both aggregate metrics here. Fresh1250 models have
higher MSE than M1; fullGCλ1 reduces power error to58.54% versus GC61.57%
at12.61% worse MSE. Historical Octoberλ0.25 improves both metrics versus raw
M1, but M1 SWA remains slightly better on both. Septemberλ0.1/λ1 do not improve
the spherical aggregate versus their controls, unlike the earlier2023 zonal
result: metric/domain/cohort differences limit that earlier conclusion.
Observed baseline rollout spread0.079385 is advisory; small freshGC–Mamba
λ0.5/λ1 decreases do not exceed it. See [full completed results and figures](../../../docs/experiments/v24_raw_checkpoint_pilot_2026-11-10.md).
This four-date diagnostic does not establish annual/seed robustness or release
the separate full-GC horizon training hold. Dated submission history follows.

November10: [e011](experiments.md#e011) submits a fast MSE/power pilot for
principal M1 without SWA and all ten existing fresh scalarλ checkpoints.
M1 uses final raw main-step2000 (2500 total); the fresh arms share raw1250.
Historical M1 SWA is a separate reference. Four quarterly2022 starts ×40leads
keep evaluation bounded; array17249273 and report17249274 are initially pending.
Existing32-start M1 raw/SWA exact rollout reductions are15.2637%/16.3434%
against their respective saved GC baselines, with finite baseline drift noted.
New spectra and lambda skill are pending. This does not release the separate
full-GC horizon training hold. See the [checkpoint/protocol record](../../../docs/experiments/v24_raw_checkpoint_pilot_2026-11-10.md).

November10 follow-up includes both lambda sets. Historical extension17250938
adds eight raw endpoints, with report17250939 reusing the original M1 tasks.
Cached eight2023-start non-SWA results are already [plotted](../../../plots/analyze_models/data/resolution_eval/amse_lambda_raw_cached_20261110/report.md):
within October, λ0→0.5 changes exact MSE9.7746→10.1328 and zonal power error
56.81%→53.39%. Separate September λ1 substantially improves power matching
but worsens MSE versus public GC. The histories/objectives differ; those zonal
figures do not substitute for the pending matched four-date spherical comparison.

November9 smoothing extension: the user authorized three prespecified strength
series for public GC and principal M1 on the same122 annual starts. Gaussian
widths0–8° and blends α=0:0.1:1 toward the existing development-fitted Gaussian
and four-band profiles will expose the MSE/power tradeoff. Evaluation array
17213596 and report17213597 are submitted, initially pending. No new scientific
result yet. Exact native-grid and actual-correction spectral moments allow
offline strength refinement; no refitting or winner selection uses2023. See
[e010](experiments.md#e010) and the
[sweep protocol](../../../docs/experiments/v24_smoothing_strength_sweep_2026-11-09.md).

November9: the new full-GC K20 `3e-7` run reached1250 updates. Its update500
checkpoint reduced exact rollout loss4.6434% and day10 loss5.3090% against
public GC on the32-start2022 development cohort. The user requested K8/K12/K16/K20
at2500 updates, evaluations every500, SWA and spectra. A fixed-window comparison
has been technically prepared, reusing K20; new jobs are not yet submitted.
On user request submission is on hold: the draft does not yet include principal
M1 in its outputs or a matched2023 comparison. See the reviewer audit in the plan.
See [e009](experiments.md#e009) and the
[experiment plan](../../../docs/experiments/v24_full_gc_ar_horizons_2026-11-09.md).

November9: the completed122-start GC/M1 smoothing comparison shows a clear
accuracy/power-fidelity tradeoff. Raw GC, raw M1 and band-filtered GC have mean
relative power errors59.47%,67.57% and75.70%, respectively; their exact rollout
loss reductions versus raw GC are0%,14.38% and9.39%. Band-filtered M1 reaches
15.33% loss reduction with73.90% power error. M1 therefore improves loss beyond
these smoothing baselines, but does not improve the overall power score relative
to raw GC. Physical spectra show scale/channel dependence, including high-degree
excesses. See [the recorded analysis](experiments.md#e008-smoothing-spectra-20261109)
and its figures; these are point estimates without new spectral bootstrap
intervals. The full fresh λ-model comparison remains separate.

November7 scalar follow-up: full-GC profiling and both500-update LR calibrations
completed on one80GB GPU. Exact dev32 loss increased13.8958% at peak1e-5 and
52.2477% at3e-5 relative to the corresponding public-GC forecasts. The user
authorized a fresh public-GC λ0 run at provisional peak3e-7, retaining pending
job17049162_0 and its continuation17049163_0. This replaces the λ0 production
resume only; other λ arms remain1e-5, so the revision is not a fixed-LR λ-only
comparison. See the newer lower-LR result above and the
[dated e008 revision](experiments.md#e008-fresh-lr3em7-20261107) and
[current handoff](devel.md). Older MGDA experiments below remain separate.

The first qualification (15787947) failed on October18 after6m41s because
finite repeatability noise exceeded a hard threshold; release15787948 was
cancelled without training. This is an execution failure, not a scientific
result. On October19 the user reaffirmed that large finite variance must be
a saved warning and authorized a rerun. The [recovery attempt](experiments.md#e002)
uses the same original width512/di16 MSE parent and three200-update arms.
Qualification is discarded and does not consume production updates.
Recovery qualification15877833 passed and release15877834 launched training
on October19. All three arms subsequently stopped on exhausted measured-loss
backtracking: MGDA after46 accepted updates, MSE control after1, fixed sum
after31. Evaluations and report were cancelled by dependency; no final forecast
comparison exists. Finite variance remained advisory throughout. See the
[October23 outcome record](experiments.md#2026-10-23--recovery-outcome-inspected).

On October28 the user requested completing finite training without repeated
loss-decrease gates. [e003](experiments.md#e003) uses a single scheduled update
per batch; finite increases, stalls, direction discrepancies and variance are
recorded warnings. Raw repeats and before/after objectives remain available.
Nonfinite computation and corrupt/mismatched inputs still stop. The successful
e002 qualification supplies resource measurements, avoiding another GPU gate.
This broader standing preference is saved in [AGENTS.md](../../../AGENTS.md).

The [detailed failure diagnosis](experiments.md#diagnosis-20261028) finds
MSE repeat variation typically63× the local predicted decrement, correct
recorded MGDA coefficients, and a structural difference between stopped-weather
gradients and regenerated-rollout measurements. BF16 graph reductions are a
plausible numerical contributor, not an isolated cause. The next numerical
repair to consider is FP32 processor/decoder aggregation; true rollout-gradient
training required extending the existing short weather-feedback backend to the
two-objective case. That extension is now implemented separately; neither
change is a prerequisite for e003 to finish.

The requested [shared full-BPTT plan](devel.md#planned-shared-full-bptt-implementation--2026-10-28)
targets24-step windows, K20 and96-step segments for a matched first comparison.
Both objectives now differentiate memory and generated weather over the same
window. Memory values carry between windows with detached history; weather
restarts from observations each window. The [e004 campaign](experiments.md#e004)
completed all three200-update arms and matched evaluation on October29; its
report was inspected November2. MGDA retained eight observation updates and
resumed for192. The existing e003 snapshot is unchanged.

On the eight2022-origin,40-six-hour-lead development cohort, MGDA exact GC loss
rose0.4275% relative to the original parent, while secondary mean spectral power
error fell from44.5751% to43.2507%. Fixed sum showed a similar tradeoff: exact
loss+0.2412%, power error43.5225%. MSE control changed exact loss by−0.0499%
and power error to44.6026%. Coherence was slightly lower for all three arms.
The [completed result](experiments.md#e004-completion-20261102) demonstrates valid
execution and a spectral/accuracy tradeoff, not a joint accuracy-preserving
improvement. The hypothesis remains inconclusive on this small single-seed
development cohort; no statistical significance is claimed.

## Next decision

On November2 the user explicitly approved two follow-ups: continue the existing
MGDA and matched MSE-control trajectories to2000 total updates (e005), and train
a fresh matched pair for2500 updates using DeepMind spatial initialization and
the best setup's Mamba1/seed22/zero-output recipe (e006). Fresh MGDA starts at the
first update, with its own warm-up/decay schedule; the continuation preserves
the1e-6 schedule and optimizer state. Both retain full BPTT and the current
exact-GC/AMSE-excess pair. These are training-strategy comparisons rather than
an isolated initialization ablation. No new skill result is available yet.

The user subsequently requested exactly two execution parts per arm to reduce
repeated queue waits. The original pending chains were cancelled before any
training, and replacements submitted: e005 stops at1100 then2000 total updates;
e006 at1250 then2500. Optimization schedules and exact-resume state are unchanged.

Inspect completion and matched evaluation before judging either strategy.
Exact GC loss remains primary; spectral power/coherence are secondary. Finite
warnings remain advisory. No further extension, di32 arm or SWA is included.

## Records

November6: [e007](experiments.md#e007) extends relative-power MGDA SWA and
main-text M1 to the full122-start/40-lead2023 set, with paired vanillaGC.
Evaluation16982943 and dependent report16982944 are submitted. The requested
diagnostic is absolute power mismatch divided by vanillaGC's absolute power
mismatch, not forecast-error power. Exact GC MSE is reported alongside it.
No annual MGDA result is available until completion.

November6 saved-evaluation synthesis: relative-power MGDA SWA improves zonal
power error14.02% relative to main-text M1, but has0.98% higher matched32 MSE.
The full-BPTT200 pilot improves power error2.97% relative to its di16_joint
parent while worsening MSE0.43%; its power error remains23.77% higher than
original GC. Main-text M1 is di16_fast_mamba, not this pilot parent. Historical
and newer evaluation cohorts differ, so the
[comprehensive comparison](../../../plots/analyze_models/data/resolution_eval/mgda_comparison_20261106/comparison.md)
keeps cohorts separate and does not claim a newer-pilot-versus-M1 ranking.

November6 scalar-lambda follow-up: existing matched500-update spectral
lambda0/0.25/0.5 continuations already demonstrate an accuracy–power tradeoff.
Lambda0.5 gives9.786% exact-GC rollout MSE reduction on eight2023 starts,
versus13.137% for its matched grid-MSE control, while reducing the power-error
score from56.494% to53.405%. Earlier lambda0.1/1 have different training
histories and are shown separately in the
[retrieved comparison](../../../plots/analyze_models/images/resolution_eval/amse_lambda_comparison_20261106/mse_and_power_overview.png).
These use spectral-MSE interpolation, not the new exact-grid-plus-excess loss;
they do not constitute a full-GC fine-tuning comparison.

November6 collaborator plan: [fresh scalar λ comparison and smoothing](../../../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md)
now specifies fresh GC–Mamba and full-GC fine-tuning at λ=0/0.25/0.5/0.75/1,
using exact grid MSE plus the stable AMSE excess, full weather BPTT and a common
2,500-update budget. Independent development-fitted Gaussian and four-band
attenuation baselines test how much skill comes from smoothing. This is a
new matched comparison, not a reproduction of stopped-feedback M1. The original
[dated decision](experiments.md#scalar-plan-20261106) records the proposal.

November7: the user explicitly requested implementation and launch. The
[e008 campaign](experiments.md#e008) is implemented and submitted: initial
training profiles 17022621, family-independent controllers 17022622/17022623,
and GC/M1 smoothing 17022624→17022625. All were queued at submission verification.
Full-GC GPU memory feasibility and forecast outcomes remain pending. Automatic
continuation covers LR calibration, all ten fresh trajectories, output-filter
fitting and annual evaluation. Full fine-tuning is attempted before considering
LoRA; no silent method substitution or finite-improvement gate is present.

- [Original attempt](experiments.md#e001) and [warning-only recovery](experiments.md#e002)
- [Advisory-decrease continuation](experiments.md#e003)
- [Shared full-BPTT campaign](experiments.md#e004)
- [Longer continuation](experiments.md#e005) and [fresh initialization](experiments.md#e006)
- [Implementation and resumption](devel.md)
