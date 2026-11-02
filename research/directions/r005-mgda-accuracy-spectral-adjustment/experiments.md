# Attempts and evidence

<a id="e001"></a>
## e001 — Exact GC MSE and spectral AMSE-adjustment pilot

Event date: 2026-10-18
Recorded: 2026-10-18 (America/New_York)
Execution: failed
Outcome: execution-failed

**Question and change:** Test whether MGDA can improve spectral fidelity without
sacrificing exact GC accuracy. Replace the failed A/C pair with exact GC grid
MSE and the nonnegative AMSE spectral adjustment. Add actual post-update
rollout checks and backtracking rather than relying on gradient dot products.

**Setup:** Original width512/di16, residual MP2, Mamba1/BCG1/two layers,
state16/conv4/automatic rank. Initialize every residual parameter from
`res1_batch4_temporal_lr_20260908/runs/di16_joint/swa/swa_step00500-02000.pkl`,
SHA256 `25a0c980a3ae2f64d45bdfab95c876738dd9e8817edccfef3fc810a2dd47545b`.
The baseline remains frozen. All three arms have fresh optimizer, RNG, cursor
and zero initial memory; seed22, batch4, K20/BPTT24/segment96, all-step
supervision and within-segment memory carry. Retain original training inputs
and order (2015–2021 training, 2022 development evaluation).

**Objectives and controls:** `L1=exact_GC_grid_MSE`,
`L2=AMSE_spectral-MSE_spectral>=0`; full83-channel GC normalization and weights,
T179, lambda1, epsilon1e-12. No separate objective scaling or truth-power
normalization. Directions: MGDA; MSE-only; fixed sum `L1+L2`.
All arms update the second moment with `0.5*(g1**2+g2**2)`, decay0.98,
epsilon1e-8, spatial multiplier1/Mamba3 inside the metric, no first moment or
weight decay, global clip1.

**Acceptance:** Initial trial LR1e-6, at most eight halvings (nine candidates).
Measure three unchanged forward repeats each batch. Require decreases above
the maximum of absolute floor1e-8, relative floor1e-7, and four times the
observed repeatability span. Reject excessive repeatability noise above
`max(1e-8,1e-3*abs(loss))`. MGDA must lower both raw objectives; MSE control
must lower L1; fixed-sum control must lower L1+L2. All candidate evaluations
regenerate the complete trajectory from the identical original state and keys.
RMS moments are proposed once, and only the accepted parameters and regenerated
end memory are committed. Exhausted search aborts and records the failed batch.

**Stages:** Four fixed qualification batches test fixed-input baseline-feedback
and closed-loop candidates without applying updates. Passing qualification
releases three independent 200-update arms, checkpoint every50. Evaluate only
parent and final endpoints on Jan/Feb/Apr/May/Jul/Aug/Oct/Nov15, 2022,
00UTC, each40 six-hour leads, cold/full feedback and zero initial memory.
Both exact and spectral stages use those same eight origins. Final200 is
primary; no checkpoint selection or 2023 evaluation is included.

**Provenance:** Campaign root:
`artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/`.
Full current executable source and vendored GraphCast are frozen, with the
historical parent prepared-array loader restored and explicitly fingerprinted.
The old source archive omitted dependencies and environment, so exact
bitwise historical reproduction is not claimed. The campaign manifest pins
parent/config/data metadata/stats/evaluation origins and frozen-source digest.
Frozen source digest:
`c9fc590940a97931ebdc22122966071cb8800f82ee7ee1e4e1cd66611ffa61c4`.
Qualification15787947 and afterok release15787948 submitted at12:28EDT,
October18. Both are pending; training/evaluation IDs are created conditionally
by the release stage after successful qualification. The execution field
includes this queued stage and does not assert GPU work has started.

**Evidence:** [Driver](../../../scripts/experiments/run_v24_mse_amse_excess_pilot.py),
[losses](../../../src/models/mamba/v24_Ilya/training/mse_amse_excess_loss.py),
[acceptance](../../../src/models/mamba/v24_Ilya/training/mgda_acceptance.py).

**Observation:** 189 targeted regression checks passed, with32 campaign checks
also passing (counts overlap). Frozen-source/input verification, finite parent
parameter load, real historical prepared-loader startup, all three configuration
parses, eight-origin coverage, shell syntax and Slurm test-only preflight pass.
GPU qualification and scientific evaluation are pending.

**Decision / next action:** Qualification and its conditional release pipeline
are submitted. Review the pilot before authorizing any larger campaign.
Existing experiments remain preserved.

### 2026-10-18, 12:28 EDT — Submission

- Qualification: one A10080,8CPUs,320GiB host RAM,4h. Historical same-shape
  di16 job15515799 used234.71GiB and22h53m for1,900 updates.
- Release:2CPUs,8GiB,15min, afterok15787947. It requests unthrottled three-arm
  training only after passing the numerical proof. Training memory is rounded
  from1.35×max(historical, measured peak); time from200×measured p90×1.4 plus
  15min startup, bounded to512GiB/48h. Evaluation arrays are unthrottled,
 16GiB/62min each; CPU report12GiB/30min.
- [Qualification submission](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/submissions/qualify.json),
  [release submission](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/submissions/release.json),
  [CPU/runtime preflight](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/checks/pre_submit.json),
  [scheduler preflight](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/checks/scheduler_preflight.json).
- An interrupted preparation under restricted external-filesystem access is
  retained in the sibling `_preparation_sandbox_attempt` directory. It has no
  submitted jobs and is not a scientific run. The final campaign was prepared
  afresh with verified access to the unchanged historical Zarr v3 store.

### 2026-10-19 — Failure inspected and policy corrected

Slurm confirms qualification15787947 failed October18 at12:37EDT after6m41s;
dependent release15787948 was cancelled two seconds later. The first unchanged
forward repeats were GC MSE `[3.51186037,3.50442004,3.51004481]` and spectral
adjustment `[0.95065224,0.94967568,0.95088506]`. Their spans were0.212%/0.127%
of the first repeats, exceeding the0.1% hard ceiling. No production updates or
scientific evaluations occurred. Original source, logs and
[failure record](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/failures/qualify_0_15787947.json)
remain preserved. The user reaffirmed their prior instruction: large finite
variance is a warning, must be saved, and must not block a rerun.

<a id="e002"></a>
## e002 — Recovery with advisory repeatability variance

Event date and recorded: 2026-10-19 (America/New_York)
Execution: failed
Outcome: execution-failed

**Authorization and change:** User explicitly requested saving the warning-only
variance policy and rerunning. Standing project instruction saved in
[AGENTS.md](../../../AGENTS.md). `repeatability_policy=warn` is explicit in all
four recovery configurations and is the maintained default. The0.1% limit
remains a reported warning threshold. The original4× span is saved as an
interpretation diagnostic, not an indirect acceptance gate. Candidates still
must decrease their arm's measured objective(s) beyond the absolute1e-8 and
relative1e-7 floors; nonfinite values and integrity failures remain errors.
No claim of decrease above numerical noise is made for an accepted noisy step.

**Setup:** Same parent checkpoint, objectives, pair RMS, seed22, K20/batch4,
four discarded qualification batches, three200-update arms and eight seasonal
2022 evaluation starts as e001. No2k extension or di32 expansion. Recovery root:
`artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/`.
The source comes from e001's immutable snapshot with only the acceptance helper,
training step, configuration parser and campaign driver patched. The driver
records the parent manifest, source digest and changed files. Historical
prepared-loader behavior is retained.

**Saved evidence:** `backtracking_trials.jsonl` and qualification records retain
all three raw repeats, spans, warning bounds/flags, advisory noise margins and
`accepted_decrease_above_noise`; scalar flags also appear in train metrics.
Exhausted backtracking retains these diagnostics in failure records.

**Validation:**104 targeted tests passed, including the exact recorded failing
variance, all three accepted noisy-update paths, retained warning evidence,
nonfinite/increasing-loss rejection, configuration parsing and campaign release
checks. Modified-file whitespace and Slurm shell syntax checks pass.

**Resources:** Retain qualification1×A10080/8CPUs/320GiB/4h and dependent
release2CPUs/8GiB/15min. The original failed qualification used31.60GiB before
gradient computation, so it is not a full-step memory estimate. Comparable
same-shape training15515799 used234.71GiB; production requests still derive
from measured full qualification cost and that historical floor.

**Submission:** Qualification **15877833** and dependent release **15877834**
submitted October19 at20:00EDT. Both are confirmed pending; release waits for
qualification. The execution field includes this queued stage and does not
assert GPU work has begun. Passing qualification conditionally submits the
three training arms, exact/spectral evaluations and report.

**Provenance and preflight:** Frozen source digest
`8d9cc13cf98856ea84b4bbc651c287b69badcfb88943a084f9dbf76879c5db93`.
Verified that only the four declared files differ from e001, the initialization
is identical, and all configs differ only by output paths and the explicit
warning policy. The actual prior failing repeats now produce a warning.
Slurm test-only checks pass for qualification and release.
[Preflight evidence](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/checks/recovery_pre_submit.json),
[qualification submission](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/submissions/qualify.json),
[release submission](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/submissions/release.json).

**Next action:** Inspect qualification15877833 and its saved warnings, then
the three final200-update comparisons. No scientific result yet.

### 2026-10-23 — Recovery outcome inspected

Events occurred October 19; inspected and recorded October 23. Qualification
15877833 completed in 40m20s and passed all four batches; release15877834
completed and submitted training array15884371. The advisory policy allowed
qualification and training to proceed with finite variance warnings.

| Arm | Accepted updates / 200 | Failed attempted update | Accepted updates with variance warning | Accepted decreases above advisory noise margin |
| --- | ---: | ---: | ---: | ---: |
| MGDA | 46 | 47 | 33 | 0 |
| MSE control | 1 | 2 | 0 | 0 |
| Fixed sum | 31 | 32 | 25 | 1 |

All three stopped with `backtracking_exhausted`: all nine finite candidates
failed their arm's measured-objective-decrease requirement. MGDA and MSE control
had increased measured MSE in every rejected trial; fixed sum had increased
measured summed loss in every rejected trial. The blocking thresholds were
only the absolute/relative floors (approximately 1e-7 to 3.5e-7), not the
advisory noise margins. Each failed batch has `repeatability_policy=warn` and
`noise_exceeded=[false,false]`; larger finite variance had been logged and
accepted on earlier updates. These observations do not establish noise-free
loss increases: most accepted decreases were also below the advisory margin.

Scheduler states:15884371_0 failed at22:11:52EDT after1h29m17s;
15884371_1 failed at21:02:24EDT after18m17s;
15884371_2 failed at21:46:48EDT after1h00m04s. Exact evaluation15884372,
spectral evaluation15884373 and report15884374 were cancelled by dependencies
at22:11:53EDT. No endpoint evaluation results or final report exist, and no
production checkpoint was found (all arms stopped before the first50-update
checkpoint). Qualification measured58.36GiB peak RSS and79.64s steady p90.

Evidence: [qualification](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/checks/qualification.json),
[MGDA failure](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/runs/mgda/failed_update.json),
[MSE-control failure](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/runs/mse_control/failed_update.json),
[fixed-sum failure](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_recovery_20261019/runs/fixed_sum/failed_update.json).
Update counts and warning counts were read from each run's `train_metrics.jsonl`.

The warning-only fix worked, but the scientific comparison remains inconclusive.
Next decision: inspect measured acceptance under repeated-forward variability
before proposing another recovery. This status check changed only research
records; it did not modify code, acceptance policy or scheduler jobs.

<a id="e003"></a>
## e003 — Complete bounded training with advisory decrease diagnostics

Event date and recorded: 2026-10-28 (America/New_York)
Execution: running (MGDA started; controls queued)
Outcome: pending

**User decision:** Let the experiment finish despite finite lack of descent;
avoid excessive checks and save this as a general recommendation. Updated
[AGENTS.md](../../../AGENTS.md) to make finite increases, stalls, variance and
weak intermediate metrics advisory by default, while retaining invalid-execution
stops. This supersedes e002's strict measured-decrease gate.

**Reason:** e002's failures were deliberate `backtracking_exhausted` exits.
At MGDA attempt47, unchanged-model MSE had a0.003202 span; the best candidate's
apparent increase was0.000283. Each candidate was measured once against a
three-repeat reference mean. None of46 accepted MGDA steps cleared the4× span
diagnostic. This supports concern that the strict test was sensitive to noise.
The stopped-weather-feedback gradient also omits dependencies present when
regenerating the forecast. Noise and this gradient mismatch are plausible
contributors; their individual causal contributions remain unisolated.

**Change:** `mgda_backtracking.decrease_policy=warn` is now the default and is
explicit in this campaign. Evaluate and commit the one finite scheduled update
at LR1e-6. No candidate search or selection based on noisy measured improvement.
Before/after objectives, raw repeated losses, variance bounds, `measured_decrease`,
`decrease_warning`, `descent_valid`, `stalled` and above-noise diagnostics remain
in the history/metrics. Finite direction discrepancies or zero displacement do
not block the budget. Optimizer moments advance once and recurrent memory comes
from the committed candidate. Nonfinite gradients, moments, parameters, losses
or candidate state still stop; source/checkpoint/data integrity checks remain.
The final report retains warning counts and separates completion from skill.

**Setup:** Same width512/di16 original MSE parent, objectives, pair RMS,
seed22, batch4/K20/BPTT24, three200-step arms and eight matched2022 starts ×40
leads. e002 produced no saved production checkpoint, so every arm restarts from
the prescribed parent. No new2k/di32 expansion. Campaign root:
`artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_advisory_20261028/`.
Frozen source preserves e002 except four declared files: config parser,
acceptance helper, training step and campaign driver.

**Proportionate verification:**122 targeted cases pass after correcting a
test-fixture hash (121 initially passed, then all38 campaign cases passed).
Coverage includes committing a finite overshoot/tie once, truthful warning
records, candidate-state/RMS ownership, nonfinite rejection and downstream
validation accepting advisory increases. Historical strict tests are explicit;
they do not set the production policy.

**Resources and pipeline:** Reuse successful qualification15877833 as a pinned
historical resource measurement, not a claim of new numerical qualification.
No additional GPU qualification job. CPU release launches the unthrottled
three-arm training array, exact/spectral evaluation arrays and final report.
Each training arm requests320GiB/8CPUs/oneA10080/390min. Prior MGDA training
reached130.69GiB by46 updates and same-shape long training15515799 reached
234.71GiB; the existing320GiB budget retains headroom for200 steps. Time uses
the measured79.64s p90 with1.4× headroom and startup allowance, even though
removing retries should reduce cost. Evaluation requests remain16GiB/62min;
report12GiB/30min.

**Submission:** CPU release **16391359** accepted at16:11EDT on October28;
initial scheduler state pending. Frozen source digest:
`e0751783bf520e26a555a23c9748d58aed5a2bc0dcd331930dfd8aabf580e700`.
The [pre-submit record](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_advisory_20261028/checks/advisory_pre_submit.json)
confirms unchanged parent/model/data, the four-file source overlay, scheduler
resource acceptance, and that all three historical failing finite steps now
continue with explicit warnings. Release completed successfully in6s. Training
array **16391367** has all three arms queued; exact evaluation **16391368**,
spectral evaluation **16391369** and report **16391370** are pending their
dependencies. No new qualification job was submitted.

**Next:** Let the bounded pipeline finish and inspect its final evaluation.
No scientific result is claimed from this change.

<a id="diagnosis-20261028"></a>
### 2026-10-28 — Detailed e002 failure diagnosis while e003 continues

The user requested a deeper explanation and concrete repair options. This is
an analysis of the existing failures, not a new training attempt or gate.
At inspection, e003 MGDA16391367_0 is running; controls16391367_1/2 remain
pending priority, and evaluations/report await dependencies.

**Observed evidence:** All three e002 terminal exceptions are
`backtracking_exhausted`; their nine candidate loss pairs are finite. The MSE
control already failed on attempted update2, independently of MGDA mixing.
MGDA stopped on47 and fixed sum on32. No production failure here was an OOM,
timeout, nonfinite gradient or variance-bound rejection.

| Arm | Accepted updates | MSE repeat span at failure | Nominal MSE change versus repeat mean | Best MSE change among nine candidates |
| --- | ---: | ---: | ---: | ---: |
| MGDA | 46 | 0.00320172 | +0.00270192 | +0.00028340 |
| MSE control | 1 | 0.00199294 | +0.00095367 | +0.00033808 |
| Fixed sum | 31 | 0.00210714 | +0.00621541 | +0.00077446 |

All values are raw exact GC loss units; positive change means worse. Fixed
sum was judged on MSE + adjustment, not MSE alone: at its nominal step the
adjustment fell0.00137170 but the sum still rose0.00484371. MGDA's nominal
adjustment fell0.00018303. Shrinking LR from1e-6 to3.90625e-9 did not yield a
smooth approach to the reference: terminal MGDA MSE rose0.00572364 even at the
smallest step. Three repeats are an observed range, not a confidence interval,
and do not establish that every increase is noise.

Across46 accepted MGDA updates, the median within-update ratio of repeat span
to the local first-order decrease (`g dot (old_params - new_params)`) is
**63.14 for MSE** and **11.57 for adjustment**. The corresponding separate
medians are MSE span0.00500047 and local MSE decrease0.00010662. None of these
46 accepted steps exceeded the stored4× span diagnostic. Eighteen updates
required halving, including two accepted at LR/256. These accepted losses are
selected by the old rule and must not be treated as unbiased evidence of skill.
An especially clear comparison issue occurs at MSE-control failure: the nominal
candidate's MSE3.23540235 is below the first unchanged evaluation3.23546648,
yet above the three-repeat mean3.23444867. The pass/fail sign depends on which
unchanged reference is used.

**Optimizer audit:** All46 recorded MGDA directions passed local descent,
with no zero displacement. The mixing coefficients agree with the independent
metric formula within9.11e-8 absolute error; alpha ranges0.710–0.962, with
median0.857. Thirty batches have opposing raw gradients, but a common local
descent direction was still obtained. There is no recorded evidence of an
MGDA coefficient/sign bug or a Pareto stall. This audit covers saved metrics;
the failed batch's full gradient tensors were not saved. Fixed-sum local MSE
directions were uphill on13/31 accepted updates, which is permitted by its sum
objective and is a scientific tradeoff to evaluate, not an execution error.

**Two separate mechanisms:**

1. Numerical sensitivity: both predictors use `Bfloat16Cast`. Parameters and
   the final loss/tape are FP32, but the predictor views parameters and computes
   activations in BF16. GraphCast enables FP32 aggregation for grid-to-mesh,
   but explicitly disables it for mesh processing and mesh-to-grid. Those
   reductions use `segment_sum`; installed JAX documentation permits
   nondeterministic ordering for overlapping scatter updates. BF16 graph sums
   and subsequent autoregressive amplification are a concrete leading
   hypothesis for the repeat variation. Fixed RNG, zero dropout and unchanged
   inputs rule out deliberate sample/dropout resampling in these repeats.
   Low precision by itself does not cause run-to-run randomness, and no GPU
   kernel or precision ablation has isolated the actual contributor.
2. Gradient/measurement mismatch: the explicit reverse tape differentiates
   through residual parameters and recurrent memory, while holding weather
   frames and baseline predictions fixed. Candidate forwards regenerate both
   the weather trajectory and the baseline on that trajectory. Hence its
   local descent test concerns a surrogate, not the full regenerated rollout
   derivative. A four-step CPU toy using the actual tape confirms agreement
   with stopped-feedback AD, while full-feedback AD differs1.63%/1.84% for
   the toy's two losses with identical forward values. These percentages are
   illustrative, not estimates for the failed weather model or batch.

**Repair priorities:** The current e003 already fixes the abort: commit finite
scheduled updates, preserve warnings, and complete matched evaluation. Further
LR reduction is not supported as the first repair by these noisy searches.
For numerical stability, the first targeted candidate is FP32 accumulation in
mesh-processor and decoder graph reductions for both predictors. This changes
arithmetic, not parameter shapes; improvement is a hypothesis to measure, not
a promise of determinism. Full FP32 prediction is a more expensive follow-up
if needed. Casting the final loss to FP32 again cannot repair upstream BF16
rounding because that loss is already FP32.

If a true rollout gradient is needed, adapt the existing
`training/weather_feedback_step.py` backend to two objectives. It already
propagates weather cotangents and the frozen baseline's input Jacobian, but
supports short2/4/8-step scalar-loss training; current MGDA configuration does
not support this mode. It is a separate implementation/experiment, not a flag
that makes the current24-step pilot exact. For a cheaper derivative diagnostic,
replay a fixed weather tape and baseline, letting recurrent memory depend on
candidate parameters, then compare to the separately regenerated rollout.
Keep all finite discrepancies advisory. Let e003 finish before changing its
precision, gradient definition or LR.

**Reproduction and artifacts:**
[analysis script](../../../scripts/analyze_models/diagnose_v24_mgda_failure.py),
[hashed input record and numeric diagnosis](../../../artifacts/analysis/v24_Ilya/mgda_failure_20261028/diagnosis.json),
[failed line-search plot](../../../artifacts/analysis/v24_Ilya/mgda_failure_20261028/failed_line_search.png).
The CPU analysis completed successfully in the GraphCast environment after
adding the repository root to its import path. No production source, submitted
snapshot, job configuration or scheduler dependency was changed for this
diagnosis. There is still no final forecast-skill conclusion.

### 2026-10-28 — Shared full-BPTT implementation plan

The user clarified that MSE and spectral adjustment should use uniform BPTT
and requested an implementation plan plus horizon/segmentation recommendations.
Recommended first comparison: BPTT24/K20/segment96, batch4 and all-step losses,
with both objective gradients passing through recurrent memory and generated
weather inside the same24-step window. Carry detached memory across the four
windows in a segment; reset weather from observations each window and reset
memory at the segment boundary. Preserve the four teacher-input steps rather
than silently change to K23. This is a planned extension, not a launched run.

The [development plan](devel.md#planned-shared-full-bptt-implementation--2026-10-28)
specifies a shared scalar/vector weather-tape engine, two sequential reverse
sweeps, existing paired-RMS/advisory commits, config/validation/resume changes,
three focused validation groups and one resource measurement at the actual
24-step shape. Existing scalar full-feedback2/4/8 runs completed100 updates;
the8-step [completion proof](../../../artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004/checks/feedback_differentiable_b8_complete.json)
records19.28s p90 and76.35GiB peak host RSS. That evidence does not establish
24-step dual-gradient resources. Proposed resource-measurement updates count
toward the bounded training budget, and finite diagnostics remain advisory.
No production code, current job or frozen snapshot was changed for this plan.

<a id="e004"></a>
## e004 — Shared full BPTT for both MGDA objectives

Started: 2026-10-28. Completed: 2026-10-29. Inspected: 2026-11-02.
Execution: completed. Outcome: inconclusive for improving spectral fidelity
while preserving exact GC accuracy.

**Question:** Does differentiating both exact GC MSE and spectral adjustment
through the same generated weather and memory paths improve the matched
comparison? Retain BPTT24/K20/segment96, effective batch4, seed22, LR1e-6,
predictor precision and the original width512/di16 parent. Change the feedback
derivative, without claiming exact arithmetic or guaranteed loss decreases.
The prescribed parent SHA256 remains
`25a0c980a3ae2f64d45bdfab95c876738dd9e8817edccfef3fc810a2dd47545b`.

**Implementation:** `weather_feedback_step.py` now provides one shared
scalar/vector forward tape and independent sequential reverse sweeps. Teacher
observations stay constants; only generated predictions actually used as later
inputs receive weather cotangents. Both overlapping weather edges and frozen
GraphCast input derivatives are included. `mgda_step.py` reuses paired RMS,
the three directions and one finite scheduled candidate replay/commit. Exact
GC MSE remains the primary scalar validation/logged loss. Finite variance,
increases and stalls stay advisory. Same-mode exact resume works; switching
feedback mode is rejected as an exact resume.

**Prepared artifact:**
[campaign manifest](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/manifest.json).
Source digest:
`b71969ce37d1cf47b231d93719e24a21b01d01503717fe785a19baf6499cfd13`.
Preparation freezes all current runtime files and restores only the historical
prepared-array loader (`55caf0933456103cffb579f57d7d8761d91640bafbdbd321d68300ba4a624945`).
The runtime hashes were compared against the final implementation. No e003
submitted source, job or checkpoint was modified.

**Continuation/resource contract:** `observe` runs eight real MGDA updates with
the production200-update config and `--stop-after-steps 8`. Its checkpoint,
config, histories and timing/memory observations are sealed. Observation copies
remain immutable as live training histories grow. Only the verified step-eight
MGDA output may be reused; controls start from the parent with fresh optimizer,
RNG, cursor and zero memory. Release schedules MGDA192 more updates and two
200-update controls, then final200-update proofs and eight2022-origin,
40-six-hour-lead parent/control/candidate evaluations. Exact GC loss is reported
before spectral diagnostics. The initial observation request is320GiB/240min,
8 CPUs and one80GiB GPU. Final resources use the actual production measurement
with historical long-run host memory as a floor. No shorter horizon or smaller
effective batch is substituted automatically.

**Evidence and limits:** All230 focused CPU cases pass; the
[validation record](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/checks/implementation_validation.json)
identifies the tested files and hashes. Coverage includes independent derivatives
(including24/prefix4), frozen weights/active input Jacobian, truncation and
segment reset, exact RNG/cursor/optimizer advancement, candidate state replay,
all directions, warning/nonfinite policies, same-mode checkpoint continuation,
legacy compatibility and campaign provenance/dependencies. Slurm shell syntax
and the prepared dependency dry run pass. This establishes implementation
behavior on deterministic fixtures; full-size GPU capacity, training outcomes
and forecast skill remain unmeasured. No job had been submitted at preparation.

### 2026-10-28, 18:50 EDT — Submission

The user explicitly approved submission. The frozen driver submitted observation
job16399850 and release job16399851 at18:50:37EDT (22:50:37UTC). At18:50:48EDT,
Slurm confirmed observation pending and release pending with
`afterok:16399850` unfulfilled. A follow-up queue check reported observation
pending for `Priority` and release pending for `Dependency`. Submission completed
without stderr. Records:
[observation](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/submissions/observe.json)
and [release](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/submissions/release.json).
Observation requests8 CPUs,320GiB,240min and one80GiB GPU; release requests
2 CPUs,8GiB and15min. Release will size and submit the continuation, controls
and evaluations only after the observed execution completes validly. No
production update count, resource result or forecast-skill result is yet
available. The existing e003 experiment and all frozen source are unchanged.

<a id="e004-completion-20261102"></a>
### 2026-11-02 — October29 completion and results inspected

All stages completed. Observation16399850 ended October28 at19:59:51EDT after
38m58s; release16399851 ended at20:01:11EDT. Training16402493_0/1/2 finished
October29 at03:38:26/04:00:10/03:58:48EDT, respectively. Each training proof
records200 updates, with MGDA using its original eight plus192 resumed updates.
Exact16402494_0–3 and spectral16402495_0–3 completed, and report16402496 ended
October29 at04:41:00EDT. The
[final report](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/evaluation/report/pilot_report.json)
has `status=complete` and the prescribed eight2022 origins×40 six-hour leads.

| Arm | Exact GC loss | Loss change vs parent | Secondary power error (%) | Secondary coherence |
| --- | ---: | ---: | ---: | ---: |
| Parent | 8.903335 | 0 | 44.5751 | 0.496042 |
| MGDA | 8.941392 | +0.4275% | 43.2507 | 0.495513 |
| MSE control | 8.898894 | −0.0499% | 44.6026 | 0.495666 |
| Fixed sum | 8.924807 | +0.2412% | 43.5225 | 0.495561 |

Positive exact-loss change is worse. Power/coherence are equal means over the
declared feature/day1,3,7,10 cells, not the canonical GC loss. MGDA improves
power error while slightly degrading exact GC accuracy; fixed sum shows the
same tradeoff. No joint win is demonstrated. This is a small single-seed
development cohort, with no significance claim.

Finite warnings remained advisory. MGDA/MSE-control/fixed-sum recorded
88/79/86 decrease warnings and142/137/142 repeatability warnings, with zero
zero-update events in all arms. Completion is execution evidence, not proof of
the scientific hypothesis. No new submission or automatic extension is needed
or authorized; review the observed tradeoff before choosing another experiment.
