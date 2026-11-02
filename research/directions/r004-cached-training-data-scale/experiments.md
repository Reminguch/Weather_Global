# Attempts and evidence

<a id="e001"></a>
## e001 — Seven versus forty years of fresh cached pretraining

Event date: 2026-10-15
Recorded: 2026-10-15 (America/New_York)
Execution: interrupted (forty-year arms superseded on2026-10-28 before training)
Outcome: inconclusive (seven-year controls completed; larger-data comparison unavailable)

**User clarification:** The hypothesis is the benefit of larger data, not a
width comparison restricted to the ready seven-year cache. This supersedes
[r003/e001](../r003-fresh-residual-width/experiments.md#e001), never submitted.

**Design:** Cross widths128/512 with seven/forty open-loop training years.
Start all residual spatial/temporal parameters fresh; share initial weights
within each width. Hold20k open-loop updates and checkpoint selection fixed,
then use common2015–2021 data for2k closed-loop fine-tuning. The
[protocol](overview.md) defines the controlled data superset, segment sampling,
evaluation, fixed-budget interpretation and limits.

**Inspected evidence:** The
[cache READY marker](../../../data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20/READY.json)
records484 complete chunks over8 shards. The
[existing larger-data campaign](../../../artifacts/checkpoints/v24_Ilya/res1_cached_40yr_w512_di64_20261014_v5/campaign.json)
specifies1982–2021 training and2022 validation, but warm-starts from the earlier
4k trained width512 parent and uses batch4/5k updates. It is unchanged and is
not a fresh control. The
[download status](../../../artifacts/data_staging/res1_1982_2022_recovery_20261014/download_status.json)
inspected on2026-10-15 reports2001 complete and2002 in progress; the planned
full-cache directory and READY marker are absent. This describes a mutable
status snapshot. Initialization and historical result evidence remains in
r003/e001 and is not new evidence for the larger-data hypothesis.

**Design-time provenance and state:** This correction created research notes only. No
executable configuration or immutable new runtime has been frozen; no jobs
were submitted or modified. The dirty working tree does not identify an
executable experiment until preparation records its snapshot.

**Interpretation:** New evidence pending. This estimates the benefit of
additional cached pretraining years with common downstream fine-tuning.
Expanding both stages or matching epoch counts estimates different effects.

**Original planned next action:** Reuse the completed width128 seven-year control. On execution
request, verify full data and shared cache segments, preserve its sampling
protocol, and prepare the three missing fresh training lineages.

### 2026-10-15 — Reuse the existing seven-year training

The user pointed out that seven-year training already exists. Rechecked the
fresh width128 di64 config and initialization audit:20k open-loop updates,
zero copied residual tensors, di64/BCG1/rank32. Its saved continuation loads
the14k parent and completes2k closed-loop updates, with10.7846% exact pooled
loss reduction on the32-start2022 evaluation. Reuse this trained control;
only its common2023 evaluation may remain.

The earlier proposal to add segment shuffling is withdrawn. The saved runtime
advances segment_index chronologically and wraps at each epoch. Preserve that
policy in new arms and report partial-epoch/year exposure instead of creating
an unnecessary sampler mismatch. Three new lineages are needed: width128/40y,
width512/7y fresh and width512/40y fresh. The completed width512/7y overlay
lineage remains supplementary because its initialization and training settings
differ. No jobs were submitted or changed.

### 2026-10-15, 21:15 EDT — Three missing lineages submitted

**Authorization:** The user requested “submit all three and make a proper
note (where?)”. This record and the linked overview are the durable scientific
record; implementation/recovery details are in [devel.md](devel.md).

| Arm | Initial qualification job | Current state | Prerequisites |
| --- | --- | --- | --- |
| Fresh width512, 2015–2021 | **15656209** | Pending, Priority | GPU scheduling |
| Fresh width128, 1982–2021 | **15656210** | Pending, Dependency | Aligned cache verified |
| Fresh width512, 1982–2021 | **15656211** | Pending, Dependency | Aligned cache verified and shared width512 initialization qualified |
| Fresh width128, 2015–2021 | Reused completed control | Open20k and closed2k complete | New common2023 evaluation waits for all finals |

Supporting jobs: scheduler compatibility bridge **15656204**, data/cache wait
**15656205**, aligned-data verification/index construction **15656206**, CPU
cache import array **15656207_[0–39]**, and cache verification **15656208**.
All eight submissions were confirmed in `squeue` and `sacct`; no production
updates have run. Wait/bridge jobs are pending a maintenance reservation.
At inspection the scheduler estimated23:40:47EDT for the width512/7y
qualification; this is a mutable estimate, not a promised start time. The
download status records2002 complete and2003 in progress (64/1460 time steps).

**Exact protocol:** di64, B/C groups1, explicit rank32, residual processor2,
batch1, K20/BPTT24/segment96, seed22, chronological segments with within-segment
memory carry. All residual spatial and temporal weights start fresh. The two
data arms at each width share identical numerical step-zero state. Open-loop
training uses20,000 updates, **200-update warmup** to constant1e-4, checkpoint
every1,000 and full15-segment2022 validation every2,000. The earlier draft's
“without warmup” was wrong; actual historical control configuration is the
source of truth. Select the lowest exact original GraphCast validation loss
on the same ten-point grid, breaking ties toward the earlier step. Closed-loop
fine-tuning uses the common2015–2021 split, fresh optimizer/RNG/cursor and zero
memory,2,000 updates,200-update warmup to1e-5; final2k is primary. See the
[protocol](overview.md) for fixed-update interpretation and limits.

**Control provenance:** Reuse the fresh width128 di64 control in
`res1_cached_w128_di_sweep_recovery_20261007/di64_carry`: its initialization
audit records zero copied residual tensors. Its selected open parent is
step14,000; its final is step2,000 under
`res1_cached_w128_closed_loop_k20_20261013/runs/di64`. These exact files and
configs are pinned in the new manifest. The earlier width512 spatial-overlay
run is not one of these four matched arms.

**Data construction:** Reuse the existing producer's2,008 earlier historical
chunks and the original seven-year cache's484 chunks by hardlink, retaining
all424 original train chunks and60 validation chunks. Expected large-data
training size is2,432 chunks/608 segments;18 transition anchors are omitted
to preserve the old restart boundaries. Before release, compare all shared
prepared weather fields bitwise and verify source identity, timestamps,
normalization, baseline and prediction checksums. No extra predictions are
generated by the new import array. Readiness and overlap checks remain pending.

**Frozen execution:** Campaign root
[`res1_fresh_data_scale_20261015_v2`](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/manifest.json).
Workflow ID: `d39843ee2e97a74077fb9d3396a3185d57a778234282379937cf6e674c2c09ae`.
Source digest: `fec9fb39d216897a4f67363a5d25706dba6fabc49199f7331eba8caacb075dbe`.
The snapshot restores historical training arithmetic while retaining bounded
and year-sharded prepared I/O. October7/October13 training differences are
inactive AMSE additions for this MSE/AdamW experiment. New helpers and the
controller are frozen with configuration/data/checkpoint hashes. The first
preparation without `_v2` was superseded before submission; it has no jobs.

**Validation and repairs:** **84 targeted tests passed** (13 controller,
18 cache,20 training,16 evaluation,17 scheduler bridge). Real CPU checks
used the historical frozen runtime and actual saved initialization/cache:
fresh state transfer, provenance, native checkpoint projection/validation,
and the production startup gate passed, stopping before training updates.
Shell syntax and all eight final Slurm test-only requests passed. Pre-submit
review corrected missing explicit cache hashes and legacy advisory-gate
arguments, plus format-aware Zarr v2/v3 evaluation metadata. Initial GPU
test-only requests exposed Della's rejection of explicit `--partition=gpu`;
the final requests omit it and use valid QoS. The shared older producer has
the same scheduler issue, so a separately frozen compatibility bridge repairs
only definite submission rejections after successful stages, preserving its
source, science, existing job IDs and duplicate-submission guards. No existing
job was cancelled, retrained or modified during submission.

**Automatic next stages:** Each successful qualification releases five4k
exact-resume open-loop jobs, selection, closed2k, and matched32 parent/final
2022 evaluation. All four final checkpoints are then pinned together before
eight-shard/model evaluation on122 common2023 starts and40 six-hour leads.
The report uses a newly evaluated width128/7y canonical baseline, paired
start-block bootstrap with simultaneous95% intervals for the two within-width
data effects, and secondary width interaction/day10 results.2023 has prior
project exposure and is not an untouched test set. One seed cannot establish
training-seed robustness.

**Artifacts:** [Initial commands and IDs](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/initial_submissions.json),
[live stage ledger](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/state.json),
[pre-submit checks](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/checks/pre_submit.json).
Stage evidence will appear under `reports/`, checkpoints/metrics under `arms/`,
and the final comparison under `results/` in that campaign root.

**Observation / interpretation:** Submitted and waiting; no new scientific
results yet. Passing implementation checks does not support or refute the
larger-data hypothesis. Preserve the planned contrast and inspect results
after automatic training and evaluation complete.

### 2026-10-17, 17:25 EDT — Width512 seven-year arm reaches12k; larger-data preparation still queued

Read-only status check: qualification15656209 completed in37m08s and released
the intended pipeline. Open jobs15659892,15659893 and15659894 completed the
4k,8k and12k boundaries in11h35m48s,11h36m21s and11h37m34s. All three signed
stage proofs pass; latest saved checkpoint and training metrics both end at
12,000 updates. The12k stage finished at15:06EDT today. Next stage15659895
(to16k) is pending priority; the current scheduler estimate is21:56:09EDT,
subject to change. Stage15659896 (to20k), selection15659897, closed-loop
training15659898 and evaluation15659899_[0–1] remain dependency-pending.

Exact cached2022 validation improvement at updates2k/4k/6k/8k/10k/12k is
0.610/0.908/0.940/1.390/0.712/0.607 percent. Best is8k, exact full-rollout loss3.693784.
The width128 control on the identical15-segment validation subset gave1.754%
at8k and1.747% at12k. The newer width512 run therefore has not improved on
that control at those checkpoints, and its validation has worsened after8k.
This is intermediate cached validation, not the prespecified final closed-loop
forecast result; retain the planned20k checkpoint search. Recent end-to-end
update time is10.16s, consistent with roughly23–24 additional GPU-hours for
the remaining open-loop work, excluding scheduling, fine-tuning and evaluation.

The source-download status now reports complete through2022 at
2026-10-17T08:16:46Z (04:16:46EDT). However, original producer gate15575578,
new wait15656205 and compatibility bridge15656204 are still pending
`ReqNodeNotAvail, Reserved for maintenance`. No prepared-stream, original
forty-year cache or aligned-cache root exists yet. Consequently qualifications
15656210/15656211 have not started. No new closed-loop or forecast evaluation
artifacts exist, and no larger-data result is available. No jobs or resource
requests were changed during this status check.

Evidence: live `squeue`/`sacct`,
[12k stage proof](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/reports/open_w512_7y_12000.json),
[validation records](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/arms/w512_7y/open/cached/validation_metrics.jsonl),
[best-validation marker](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/arms/w512_7y/open/cached/best_validation.json),
and [download status](../../../artifacts/data_staging/res1_1982_2022_recovery_20261014/download_status.json).


### 2026-10-23 — Both seven-year finals available; forty-year arms blocked by storage

Read-only reconstruction on October23 from saved stage proofs, exact evaluation
JSONs and live Slurm accounting. No jobs, executable sources or configurations
were changed. Events below occurred October18–20 (EDT), rather than today.

**All four arms:** width128/7y reuses completed open20k, selected14k and closed2k;
width512/7y has now completed open20k, selected8k and closed2k. Width128/40y
qualification15656210 and width512/40y qualification15656211 have not started.
The forty-year prepared-data, producer-cache and aligned-cache roots are absent.
The campaign has no2023 evaluation outputs or submitted2023 stages; its automatic
all-four-final freeze has not been reached. No data-size effect can be estimated.

**Matched2022 forecasts:** verified identical32 chosen starts,40 six-hour leads,
cold zero memory, full corrected feedback, baseline checkpoint and evaluation
metadata. To compare widths, use the existing width128 campaign's canonical
baseline (`evaluation/di32_step00000.json`, exact pooled loss10.824076081370004)
for every row. Positive reduction is better; these are exact original GraphCast
weighted losses, pooled before taking a ratio.

| Arm / checkpoint | Model loss | Rollout reduction | Day10 reduction |
| --- | --- | --- | --- |
| width128/7y selected14k open parent |14.271972605 |-31.853957% |-31.400884% |
| width128/7y closed2k final |9.656741653 |10.784610% |16.848425% |
| width512/7y selected8k open parent |14.236091947 |-31.522468% |-28.421492% |
| width512/7y closed2k final |10.126636726 |6.443408% |7.875807% |

Width512's own saved baseline gives6.469032% rollout and7.955506% day10
reduction; the table consistently uses the common baseline. Repeated-baseline
pooled differences from that reference are+0.041278/+0.020342/-0.032182/+0.027396%
for the four rows; maximum absolute lead differences are0.166698/0.282257/
0.229865/0.090902%. These finite discrepancies are warnings, not exclusions.
Width128's final leads width512 by4.341203pp on this development cohort and
single training seed. No paired confidence interval or training-seed replication
was computed here. Both cached parents perform poorly with corrected feedback;
closed-loop fine-tuning improves them. Cached-validation gains must not be
confused with these continuous ten-day forecasts.

**Width512 training:** all five4k jobs15659892–15659896 completed; open20k
finished October18 at18:25:56EDT. Selection15659897 chose8k (exact validation
loss3.693784196,1.390329% reduction). Closed15659898 completed2k October19 at
00:48:24EDT; evaluations15659899_0/1 completed by01:52:50EDT and each records
complete32-sample metrics and passing integrity. Open-stage elapsed sums to
58h17m55s, closed to6h18m48s (64h36m43s training, excluding qualification,
evaluation and queueing). Qualification peak host memory was13.6007GiB.

The fixed cached-validation grid at2k/4k/6k/8k/10k/12k/14k/16k/18k/20k gives
0.610064/0.907689/0.939875/1.390329/0.711772/0.606815/1.158203/0.212021/
-2.724082/-1.135567% reduction over15 common validation segments. Final20k
loss is3.788400835. Continuing beyond8k did not improve the selected validation
score; the completed search budget and fixed20k endpoint are both retained.

**Numerical warning:** width512 qualification passed all integrity checks with
`numerical_policy=advisory`. Independent GPU restart/replay state relative-L2
drift0.004281471 exceeded the diagnostic reference0.001 (max absolute0.03125,
reference state magnitude3.40625, cosine0.9999908345). Parameter relative-L2
was5.6268e-9 and optimizer3.0268e-6; exact serialization, cursor and RNG passed.
The diagnostic replay `passed=false` is advisory; overall qualification passed.
No hard finite-variance or restart-drift gate was applied.

**Forty-year blocker:** download gate15575578 completed October20 at16:11:45EDT.
Producer prepare pilot15905675 then failed after39s: `Insufficient storage:
2.314 TiB free; 2.680 TiB needed including 256 GiB reserve`. The shortfall at
that check was0.366TiB (about375GiB); current `df -h` still rounds available
space to2.3T. No data was prepared. Wait15656205 and bridge15656204 failed
at16:13EDT after detecting that failure. Alignment15656206 is pending
`DependencyNeverSatisfied`; cache import15656207_[0–39], verify15656208 and
both40y qualifications remain dependency-pending. They will not recover merely
by waiting for GPU priority. The bridge correctly refuses to retry a failed
producer stage. The separate warm-start40y experiment shares this blocker and
is not a substitute for either fresh arm.

**Next action:** resolve storage capacity or prepare an explicitly inspected
storage-efficient recovery, then recover the producer and dependent cache chain
while preserving source/data identity and matched initializations. Do not blindly
resubmit the blocked jobs or modify frozen snapshots. Once both40y finals exist,
release the prescribed common2023 comparison. This report authorizes no new jobs.

**Evidence:** [campaign ledger](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/state.json),
[selection](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/arms/w512_7y/selection.json),
[closed proof](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/arms/w512_7y/closed.json),
[parent evaluation](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/arms/w512_7y/evaluation/2022_parent.json),
[final evaluation](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/arms/w512_7y/evaluation/2022_final.json),
[qualification / replay warning](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/arms/w512_7y/qualification.json),
[width128 canonical summary](../../../artifacts/checkpoints/v24_Ilya/res1_cached_w128_closed_loop_k20_20261013/results/summary.csv),
[producer failure ledger](../../../artifacts/checkpoints/v24_Ilya/res1_cached_40yr_w512_di64_20261014_v5/state.json).

<a id="e002"></a>
## e002 — Seven versus twenty years after the forty-year storage failure

Event date: 2026-10-28
Recorded: 2026-10-28 (America/New_York)
Execution: running (replacement pipeline submitted; initial jobs queued)
Outcome: pending

**Authorization and change:** The user requested “let's use 20 years instead”.
Use2002–2021 for twenty training years and2022 for validation, preserving the
completed seven-year controls and all existing forty-year artifacts. This
supersedes the unstarted forty-year arms from e001. The scientific question
remains added-data benefit at each width, with a smaller historical extension.

**Controls and fixed protocol:** Reuse width128/7y selected open14k and final
closed2k, plus width512/7y selected open8k and final closed2k. Both have completed
20k open-loop search budgets. Only width128/20y and width512/20y are new training
lineages. Within each width use the exact saved fresh step-zero weights from
its seven-year control; retain di64/BCG1/rank32/MP2, batch1, seed22, K20/BPTT24/
segment96, chronological segments, open20k with200-step warmup to1e-4 and
selection on the same ten-point2022 exact-loss grid. Follow with common2015–2021
closed2k,200-step warmup to1e-5, fresh optimizer/RNG/cursor and zero memory.
Bounded training continues through finite weak metrics; numerical drift is
advisory. Nonfinite values, corrupt/mismatched data/checkpoints and unusable
execution remain errors.

**Data and storage:** Reuse the complete1982–2022 downloaded Zarr without
redownloading or copying it. Prepare only2002–2022 (30,680 six-hour frames),
using dedicated new roots. The provisional conservative payload scales the old
1.23TiB prepared plus1.20TiB cache estimate by30,680/59,900, then adds the
unchanged256GiB reserve: **1.494614TiB** total. Available space at design check
is2.212723TiB. This is an estimate, refined after cache manifest construction;
existing resource checks remain active. No deletion is needed. Preserve all
original seven-year train/validation windows and shared weather identity in
the aligned cache; final segment/chunk counts must come from its manifest.

**Evaluation:** Preserve common32-start2022 parent/final diagnostics and the
four-final freeze before122-start2023 evaluation,40 six-hour leads and cold
zero memory with corrected feedback. The primary comparisons become twenty-
minus-seven-year exact pooled original GraphCast loss reduction at each width,
against one common baseline, with simultaneous paired start-block intervals.
One training seed and previously inspected2023 data limit generality.

**Implementation / release:** Reuse valid seven-year qualification and resource
evidence; bind new cache/source/initialization identities in startup checks
without repeating70-update scientific qualification jobs. Build a new frozen
producer ending at verified cache, and a separate new consumer campaign for
the two missing lineages. Preserve old snapshots. Scheduler commands omit the
rejected explicit GPU partition. Submission IDs and validation evidence will
be recorded below after confirmation; no new jobs are claimed by this entry.


### 2026-10-28 — Twenty-year replacement submitted; old pending chain superseded

Confirmed in both Slurm queue and accounting after submission. Event time is
October28 at16:53EDT. The new first stage is queued; no new weather
arrays, cache predictions or twenty-year training updates are claimed yet.

| Stage | Job ID | Confirmed status |
| --- | --- | --- |
| Existing-download verification; then automatic annual preparation/cache producer | **16392462** | Pending |
| Prepared/cache wait, afterok16392462 | **16392463** | Pending dependency |
| Aligned weather/cache construction | **16392464** | Pending dependency |
| Hardlink import array,20 CPU shards, unthrottled | **16392465_[0–19]** | Pending dependency |
| Aligned cache verification | **16392466** | Pending dependency |
| Width128/20y CPU startup using reused qualification | **16392467** | Pending dependency |
| Width512/20y CPU startup using reused qualification | **16392468** | Pending dependency |

The jobs named `qualify_*` are CPU `startup` stages with zero optimizer probe
updates, not new70-update GPU qualification. Each successful startup releases
five4k exact-resume GPU training stages, selection, common seven-year closed2k
and matched32 parent/final2022 evaluation. Then all four finals are pinned before
2023 evaluation and the paired seven-versus-twenty-year comparison. Training IDs
will be added automatically to the new consumer ledger after startup.

**Provenance:** New producer
[`res1_cached_20yr_precompute_20261028`](../../../artifacts/checkpoints/v24_Ilya/res1_cached_20yr_precompute_20261028/campaign.json)
workflow `41bb961a7ed9749854fd2795fee9ffea716fab3835842b90bb24e667166dd891`,
source digest `ee3d2ab3bef164c9eb47c4d1451237e7fb27e18357b072066a07d53ec16751e4`.
It stops after verified cache, uses a single full40GB cache pilot, selects the GPU
partition automatically and requests valid QoS. It rechecks the already complete
source download; it launches no new download process. New consumer
[`res1_fresh_data_scale_20yr_20261028`](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/manifest.json)
workflow `a35967f36990b13382bccd6eb6487352d4163bbbaecd31e6f0f5758018b358eb`,
source digest `cbacdde2453c695749581bef06b97844889f11eb9c8c6bf8c2137d9397f880cf`.
Both manifests pin a snapshot of the dirty checkout; the consumer restores
historical control training arithmetic and pins both completed controls.

**Validation:**52 producer campaign/anchor/resume tests,77 consumer/cache/
training/evaluation tests,34 distinct baseline-cache/qualification tests and8
annual preparation tests passed (171 distinct targeted tests). Nine Slurm
`--test-only` requests passed. Real saved-config/checkpoint/environment reuse
checks pass for both widths using the new frozen arithmetic. Completed-download
verification and the full conservative initial storage check pass. New cache
startup still awaits the actual new prepared/cache artifacts; it was not
replaced by a synthetic readiness claim. Zarr asynchronous I/O stalled under
local sandbox socket restrictions; the same annual tests passed outside that
sandbox in14s. No production stage failed from that local tooling issue.

**Data accounting:** A calendar alignment against the actual saved seven-year
anchors projects197 additional historical segments plus106 retained train
segments:1,212 complete training chunks, with60 original validation chunks.
It omits78 transition anchors to retain old restart boundaries. These are
calendar-derived projections; final counts and bitwise overlap must be verified
by the generated manifest.20k updates correspond to approximately16.5 versus
47.2 passes over complete twenty-/seven-year chunks. Chronological year exposure
will be recorded by production. Initial disk estimate is692,681,312,027 prepared
bytes plus675,786,645,880 cache bytes plus274,877,906,944 reserve bytes; the
verified check found2,427,696,709,632 bytes free. Source data was not copied or
deleted. The two widths share the same cache.

**Supersession:** After confirming all five old submissions were still pending,
cancelled15656206,15656207_[0–39],15656208,15656210 and15656211; accounting
confirms cancellation without execution. Disabled automatic release in the old
fresh-consumer mutable ledger and linked the replacement. All old source
snapshots, logs, checkpoints and completed seven-year controls remain preserved.
No unrelated live jobs or separate warm-start experiment settings were changed.
The prior forty-year source/storage failure remains part of e001 history.

**Evidence:** [Submission IDs and commands](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/initial_submissions.json),
[new consumer ledger](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/state.json),
[new producer ledger](../../../artifacts/checkpoints/v24_Ilya/res1_cached_20yr_precompute_20261028/state.json),
[storage/download verification](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/checks/preparation.json),
[real protocol reuse](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/checks/reusable_protocol.json),
[scheduler preflight](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/checks/scheduler_preflight.json),
[confirmed queue/accounting](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/checks/post_submission.json),
[old job cancellation record](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20yr_20261028/superseded_40yr_jobs.json).

**Interpretation / next action:** Implementation and submission completed;
scientific outcome pending. Inspect the first producer stage and then its
measured preparation/cache progression. Successful execution should continue
to final evaluation even with finite loss increases or numerical warnings.


### 2026-10-28,16:54 EDT — New producer advances successfully

New download verification16392462 completed and published a passing signed
report at16:54:09EDT, then automatically submitted preparation pilot
**16392486** (4CPUs,32GiB,4h). Live queue confirms that pilot pending and
consumer wait16392463 running. Thus the replacement has progressed beyond
submission and passed the prerequisite that leads to preparation; annual data
construction and training remain unfinished. No new source download occurred.
