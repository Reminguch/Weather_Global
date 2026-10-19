# Attempts and evidence

<a id="e001"></a>
## e001 — Seven versus forty years of fresh cached pretraining

Event date: 2026-10-15
Recorded: 2026-10-15 (America/New_York)
Execution: running (submitted; initial jobs queued)
Outcome: pending

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
