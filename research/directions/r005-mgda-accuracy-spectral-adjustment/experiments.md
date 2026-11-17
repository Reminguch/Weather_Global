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

<a id="e005"></a>
## e005 — Extend the existing full-BPTT trajectories

Date:2026-11-02. Execution: submitted, pending priority. Outcome: pending.

The user explicitly authorized continuation after reviewing e004. Extend MGDA
and its matched paired-RMS MSE control from200 to2000 total updates at unchanged
LR1e-6. Copy the actual checkpoints and histories into a separate campaign,
retaining parameters, optimizer moments, RNG, recurrent memory and cursor.
Preserve e004. Keep24/K20/96, batch4, seed22, full weather gradients, precision
and the exact-GC/AMSE-excess pair. This adds1800 updates per arm; the first200
are not repeated. No fixed-sum arm or checkpoint averaging is included.

[Campaign manifest](../../../artifacts/checkpoints/v24_Ilya/res1_mgda_full_bptt_continuation_20261102/manifest.json).
Training arrays16672907/16672908/16672910/16672911/16672912 stop at
600/1000/1400/1800/2000. Exact evaluation16672913, spectral16672914 and
report16672915 follow. Indices0/1 mean MGDA/MSE control. Matching indices use
`aftercorr` between stages, preserving independent training progression.

Resources use completed full-feedback jobs16402493_0/1: p90 end-to-end latency
137.356s/update and historical host-RSS floor234.71GiB.400-update stages request
384GiB/1320min/8CPUs/one80GB GPU; the last200 use690min. Final evaluation uses
same eight2022 starts and40 six-hour leads. Reuse the verified best-MSE
reference evaluations, report exact GC loss first and retain comparison with
each arm's step200 result. Finite increases, variance and stalls remain warnings.

<a id="e006"></a>
## e006 — Fresh DeepMind-initialized full-BPTT training

Date:2026-11-02. Execution: submitted, pending priority. Outcome: pending.

The user approved the best setup's initialization recipe: copy compatible
DeepMind GraphCast spatial parameters, initialize Mamba using Mamba1/seed22,
and zero the temporal/residual output heads. Internal randomness follows that
same seeded recipe. No trained Mamba/residual checkpoint is loaded; the separate
GraphCast baseline remains frozen. MGDA and matched MSE control start with fresh
optimizer, RNG, cursor and zero recurrent memory. MGDA applies from update1.

Train2500 updates with24/K20/96, batch4, full feedback and the same objective pair.
Use the previously exercised paired-RMS rate range:100-update warm-up1e-6→1e-5,
cosine decay to3e-6 by2000, then constant. Spatial/Mamba multipliers1/3, clipping1,
RMS decay.98, epsilon1e-8, no weight decay. This differs from the best historical
model's AdamW schedule and500-update MSE pretraining. e005/e006 compare training
strategies; different rates/histories prevent an initialization-only claim.
The October7 relative-power surrogate is not substituted. No SWA or extra seeds.

[Campaign manifest](../../../artifacts/checkpoints/v24_Ilya/res1_mgda_full_bptt_fresh_20261102/manifest.json).
Training arrays16672971/16672972/16672973/16672974/16672975/16672981/16672982 stop
at400/800/1200/1600/2000/2400/2500. Exact16672983, spectral16672984 and
report16672985 complete the bounded chain. Resource requests match e005 for
400-update stages; the last100 use360min. No discarded numerical qualifier,
automatic failure retry or further extension is included.

**Shared implementation and submission evidence:** Both source snapshots have
digest `f9537816c104ff2c5bf381ae639b70ae73b60331ee752e741bb7fbd369146615`.
They preserve e004's runtime and historical prepared loader except the new
campaign driver/wrapper and allowance for exact-resume output relocation.
104 focused CPU cases passed, covering bitwise relocated-resume equality,
feedback-mode mismatch rejection, matched configs, stage counts/dependencies,
source preservation, tamper rejection and finite-warning behavior. Preparation
validated both actual step200 checkpoint configurations. Parent evaluation
reuse and Slurm's resource test passed. On November2 at15:53:52EST, Slurm showed
both initial arrays pending priority and later stages/reports pending their
specified dependencies. Submission records are retained under each campaign's
`submissions/`. Successful submission is not a training or forecast-skill result.

### November 2 resubmission: exactly two parts per arm

The user requested two execution parts to reduce repeated queue waits. All old
e005/e006 jobs above were still pending and were cancelled at16:00:13EST;
no training progress was lost. Those campaign snapshots and submission records
remain historical evidence, with `superseded.json` linking their replacements.

- e005 replacement: [manifest](../../../artifacts/checkpoints/v24_Ilya/res1_mgda_full_bptt_continuation_two_parts_20261102/manifest.json).
  Training16673752→16673753 performs900+900 new updates, stopping at1100/2000.
  Exact evaluation16673754, spectral16673755, report16673756.
- e006 replacement: [manifest](../../../artifacts/checkpoints/v24_Ilya/res1_mgda_full_bptt_fresh_two_parts_20261102/manifest.json).
  Training16673757→16673758 performs1250+1250 updates, stopping at1250/2500.
  Exact evaluation16673759, spectral16673760, report16673761.

Both training arrays retain indices0/1 for MGDA/matched MSE control and use
`aftercorr` between parts. Initialization, optimizer schedules, total updates,
window boundaries and evaluation are unchanged. Each part requests384GiB,
8CPUs and one80GB GPU; continuation requests49h and fresh67.5h, within the72h
limit. Both revised Slurm resource tests passed. Nine focused campaign tests
passed after the split change; the prior104-case implementation evidence was
reused, not rerun. Both new frozen source digests are
`39f6cadbdf213f007497d796e483f9a61c772d04931606082426a2d7aa527d3b`.

Post-submission Slurm inspection confirmed initial arrays pending priority
and subsequent training/evaluation/report jobs pending dependencies. Launch
records are saved in each replacement's `checks/launch_validation.json`.
Training and scientific outcomes remain pending.

### 2026-11-06 — Comprehensive saved-evaluation comparison

Recomputed all26 available checkpoint/reference rows across relative-power
MGDA, amplitude/pattern di16/di32, stopped-feedback MSE/excess200 and shared
full-BPTT200, including matched MSE controls and fixed-sum controls where run.
[Report and full tables](../../../plots/analyze_models/data/resolution_eval/mgda_comparison_20261106/comparison.md),
[rollout figure](../../../plots/analyze_models/images/resolution_eval/mgda_comparison_20261106/rollout_comparison.png),
and [reproduction script](../../../scripts/analyze_models/report_mgda_comparison_20261106.py).
Source hashes, all40-lead metrics and finite numerical discrepancies are saved
beside the report. All eight existing pilot MSE/power summaries reproduce to
1e-10. Completed evaluation status, checkpoint identities, dates, lead coverage,
matched32 indices and finite metrics were checked; no GPU work was launched.

Important reference correction: the main-text best model M1 is
`di16_fast_mamba`, SHA256
`b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0`.
The newer pilot parent is `di16_joint`, a different trained checkpoint.
M1 gives16.3434% exact loss reduction on matched32/2022; its independent
122-start2023 main-text result remains14.4240%. Relative-power MGDA SWA
gives15.5244% on matched32, i.e.0.9790% more MSE than M1, while reducing
power error14.0167% relative to M1 on the eight2023 spectral starts.
Against original GC its power-error reduction is only1.3623%; raw2000 gives
1.9238%. The amplitude/pattern endpoints reduce power error52–53% vs GC
but worsen matched32 MSE29–36%.

On eight2022 origins, full-BPTT MGDA200 gives16.2531% MSE reduction vs GC,
but23.7700% higher power error. Relative to its trained parent, MSE worsens
0.4275% and power error improves2.9712%. Power error here is pooled zonal
relative-power error across nine fields, positive modes, days1/3/7/10 and
30–60-degree latitude bands; it is distinct from spherical training AMSE.
Historical MSE and power headline cohorts differ (32/2022 versus8/2023);
the figure also shows historical exact MSE on the same8/2023 spectral cohort.
No matched M1 evaluation was located for the newer8/2022 origins, so no
direct newer-pilot-versus-M1 ranking is established. The failed split report's
finite truth-power differences are retained as warnings, not omitted results.
No new significance claim is made. Long e005/e006 first parts remained pending
in the November6 scheduler inspection and have no scientific results yet.

### 2026-11-06 — Actual power spectra reproduced after metric interpretation query

Reproduced actual zonal power and forecast/ERA5 ratios at days1/3/7/10 for
all nine fields, separately for historical8/2023 and current8/2022 cohorts.
Script: `scripts/analyze_models/plot_mgda_actual_spectra_20261106.py`.
Plots are beside the comprehensive comparison under
`plots/analyze_models/images/resolution_eval/mgda_comparison_20261106/`:
`actual_spectra_{historical,current}_all_fields.pdf`, selected-field PNGs,
and `power_ratio_day10_{historical,current}.{png,pdf}`. Raw pooled powers,
CSV curves, source hashes and `power_metric_audit.csv` are beside the summary.

Independent original pilot diagnostic CSV reproduces GC49.40950934% and
M1 56.68124469% mean relative power error to rtol1e-12. Thus the prior14.72%
relative increase is arithmetically correct, but represents a7.27-percentage-
point increase in this particular error score, not uniform power loss.
Raw log-scale spectra retain similar shapes and large-scale power, especially
at short leads. Equal weighting of180 modes gives weak high-wavenumber bins
substantial influence. This does not fully explain late-lead differences:
M1's day10 mean relative power error over modes1–10 is50.78% vsGC16.57%,
and modes11–60 is71.93% vs39.59%, averaged across nine fields. Day1 aggregate
M1 error44.90% is slightly better thanGC46.32%, whereas day10 is67.12% vs50.67%.
Plots of T850 in both cohorts were inspected. No computation error was found;
interpretation is refined to distinguish visible spectral shape, relative-bin
power matching and forecast accuracy. All quantities remain exploratory.

User then requested original GraphCast as the unit reference. Added
`plot_mgda_gc_relative_power_20261106.py`: forecast power divided by canonical
GC power at each mode/lead, with GC=1 and ERA5/GC shown as the target. Saved
`gc_relative_power_{historical,current}_all_fields.pdf`, selected-field PNGs,
day10 grids and NPZ ratios beside the preceding comparison. This is a power
ratio (above1 means more power), not an accuracy improvement or error ratio.

### 2026-11-06 — User clarified intended spectral-error metric

The requested quantity is `full_error_power / baseline_error_power`, not
forecast-power/GC-power and not relative mismatch between forecast and truth
powers. Corrected `plot_mgda_gc_relative_power_20261106.py` to pool the saved
error powers over starts before division. Error power includes amplitude and
phase via `P_forecast + P_truth - 2*cross_power`; this identity is checked
against saved spectra. GC=1; below1 is better. A canonical GC denominator is
shared within each cohort. Keep original nine-field/day1/3/7/10 PDF paths so
the user's open file displays the requested metric; old plots/data are retained
under `superseded_raw_power_ratio/`. Export per-field rollout/day10 error
reductions by summing positive-mode error powers before ratios, without mixing
physical units across fields. Prior power-mismatch percentages describe a
different diagnostic and must not be presented as this spectral-error metric.

User subsequently requested the specific power-mismatch ratio
`abs(P_model-P_truth)/abs(P_GC-P_truth)`. Generated separately as
`power_mismatch_vs_gc_{historical,current}_all_fields.pdf` and per-field PNGs
with `plot_mgda_relative_power_mismatch_20261106.py`. GC=1, lower is better.
Raw powers are pooled before taking differences and ratios. No denominator
floor or clipping; exact zero denominators are undefined/NaN. A y-axis linear
on0–1 and logarithmic above1 displays peaks when GC power nearly matches truth.
This metric differs from the earlier forecast-error-spectrum ratio; filenames
and provenance retain that distinction.

<a id="e007"></a>
## e007 — Full2023 relative-power MGDA evaluation

Date:2026-11-06. Execution: submitted. Outcome: pending.

User requested extending the exact power-mismatch comparison to the main-text
full2023 set. Default scope stated before submission: relative-power MGDA late
SWA and main-text GC-Mamba M1, with paired vanillaGC forecasts. The optional
broader-model question had no response before proceeding with that stated
scope. No amplitude/pattern model, new training or checkpoint selection.

Frozen campaign:
[metadata](../../../artifacts/evaluations/v24_mgda_power_2023_20261106/campaign.json).
The122 exact origins every third day from Jan1 through Dec30 and40 six-hour
leads match the existing main-text protocol. Cold full-feedback forecasts,
zero initial memory, same2023 store and vanillaGC/normalization identities.
Existing annual M1 records do not include these spectral observations, so both
models need new spectral forecasts. The evaluator also records exact GC loss,
physical RMSE and forecast-error spectra. The requested plot is specifically
`abs(mean(Pmodel)-mean(Ptruth))/abs(mean(PGC)-mean(Ptruth))`: pool powers across
starts first, use M1's GC as the canonical denominator, GC=1, lower is better.
No floor/clipping is imposed on these diagnostic ratios; exact zero denominators
remain undefined. Report full-rollout/day10 MSE and relative-power-mismatch
reductions separately; preserve baseline/truth drift as warnings. Aggregated
power-mismatch scores average relative errors over nine fields and positive
modes, then pool leads before taking ratios, rather than averaging plotted ratios.
This is descriptive evaluation on an already explored year, not independent
confirmation or evidence of training-seed robustness.

Training/evaluator code comes from the proven e004 frozen snapshot, preserving
its historical loader. Only the new evaluation orchestration script and wrapper
are added; all runtime files, checkpoint/config inputs and shard manifests are
hash recorded. Eight shards per model contain16/16/15/15/15/15/15/15 origins.
GPU array **16982943_0–15**: indices0–7 M1,8–15 relative-power MGDA SWA.
CPU report **16982944** waits on successful array completion. No concurrency
throttle. Each GPU task requests1gpu80,8CPUs,12GiB,62min; report2CPUs,4GiB,30min.
Resource basis: completed eight-start spectral work938.9s/7.50GiB and matched
16-start full2023 shards22–24min/7.4–7.8GiB. Slurm resource test passed.

Validation: frozen runtime/input/specification checks passed; formula/zero-
denominator checks passed; a synthetic122-origin/eight-shard report fixture
reproduced known MSE and mismatch reductions, generated plots and a completion
record. The fixture is explicitly not scientific evidence. No GPU qualification
rerun. Submission records and launch validation are in the campaign. Scientific
completion requires `complete.json` and `report/report.json`; planned outputs
include the all-field ratio PDF, rollout plots, CSVs and raw pooled spectra.

## 2026-11-06 — Existing scalar AMSE lambda comparisons retrieved

User requested the existing MSE/power graphs after pointing out that lambda0.5
had already been tested. Replotted saved evaluations only; no training or new
forecast jobs. The October4 campaign contains a completed, matched500-update
continuation from September24 MSE SWA (not principal M1): exact-grid MSE and
spectral lambda0/0.25/0.5, with seed22, B4/BPTT24/K20 and stopped weather
feedback. Positive lambda ramps over100 updates. All five saved checkpoints
per arm have forecast evaluations.

On the same eight exploratory2023 starts, using the parent's GC denominator,
the final exact-GC rollout MSE reductions are13.1374/12.9757/11.9645/9.7863%
for gridMSE/spectral0/lambda0.25/lambda0.5. Corresponding mean relative zonal
power errors are56.4936/56.4634/54.9090/53.4048%, versus49.3765% for GC.
These describe an accuracy–power tradeoff. The score averages nine primitive
fields, positive modes1–180 and days1/3/7/10, pooling raw powers over starts
first. It is distinct from forecast-error spectral power.

Earlier lambda0.1 and1 trajectories share a500-update warm-up parent and
2000-update training budget, with different frozen runtime versions. Their
prescribed SWA checkpoints and both campaign-specific MSE controls are plotted
in a separate group. Lambda1 has4.3291% higher rollout MSE than GC while its
power-error score is35.8009% lower. These cannot be joined to the mature-parent
500-update pilot as a controlled lambda sweep.

The historical scalar loss is spectralMSE+lambda*(AMSE1-spectralMSE), not the
new exact-grid-MSE-plus-excess objective. Avoid proposing another lambda grid
without accounting for this existing evidence. Full-GC fine-tuning is not
represented by these correction-model runs. Finite truth/baseline differences
are retained as warnings, with no numerical acceptance gate.

Artifacts: [overview](../../../plots/analyze_models/images/resolution_eval/amse_lambda_comparison_20261106/mse_and_power_overview.png),
[checkpoint evolution](../../../plots/analyze_models/images/resolution_eval/amse_lambda_comparison_20261106/pilot_checkpoint_tradeoff.png),
[summary](../../../plots/analyze_models/data/resolution_eval/amse_lambda_comparison_20261106/summary.csv),
[source hashes and warnings](../../../plots/analyze_models/data/resolution_eval/amse_lambda_comparison_20261106/provenance.json).
The output directory also contains separate day10 absolute power, forecast/truth
power, forecast-error/GC-error and absolute-power-mismatch/GC-mismatch plots.
Reproduce with `source scripts/graphcast_env.sh && python scripts/analyze_models/plot_amse_lambda_comparison_20261106.py`.

<a id="scalar-plan-20261106"></a>
## 2026-11-06 — Decision: document the fresh scalar λ and smoothing comparison

User requested a detailed shareable Markdown plan, exact formulas and links to
existing checkpoints. Saved the [collaborator plan](../../../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md)
and [verified inventory](../../../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_inventory_2026-11-06.json).
Execution: planned, not submitted. Scientific observation: pending. No training
implementation or job changes were made for this documentation request.

The planned core is ten fresh seed-22 runs: GC–Mamba with frozen public GC and
full-GC fine-tuning, each at λ=0/0.25/0.5/0.75/1. Loss is exact GC grid MSE plus
outer λ times the stable AMSE1-minus-spectral-MSE excess; internal AMSE λ stays
one. The λ1 endpoint remains a hybrid because grid and fitted spectral norms
differ. All arms use full weather gradients, B4/BPTT24/K20/segment96, and2,500
updates with final-checkpoint reporting. Method-specific λ0 LR calibration
retains the winning 500-update trajectories, costing 26,000 total core/calibration
updates. Principal M1 remains a separate historical stopped-feedback SWA reference.

The plan makes a proposed shared seeded segment-order manifest explicit, so
both architectures see identical data within a seed and future full-GC seed
repeats vary training order. It specifies the training/development/evaluation
split, target-timestamp checks, and conversion of the existing 32 development
selection indices into a timestamp manifest. The 2023 cohort has already been
explored and is labeled reused evaluation data.

Smoothing fits output-only Gaussian or four-band spherical attenuation on 2022,
preserves the native unresolved residual and exact grid mean, ties wind-component
parameters, and freezes filters before 2023. Four-band gains minimize pooled
exact normalized grid MSE relative to unfiltered development MSE plus a fixed
0.01 quadratic smoothness/identity penalty. This removes a separate regularizer
search; both filter families remain visible in reporting.

Verified nine existing checkpoint byte hashes and all plan-local links; no
checkpoint was unpickled. The inventory includes source/config/statistics and
data-cohort locators. Full-GC K20 fit/runtime remain unmeasured; the scalar dispatch,
full-GC adapter, sampling replay and smoothing fitter/observer need implementation.
Current checkout HEAD is `fc6a306c32d8ea4a84abd117c5663bd933367638` with pre-existing
uncommitted work, so future jobs need an immutable runtime snapshot. Numerical
variance and finite worsening remain warnings, not release or evaluation gates.

Next action, when implementation is requested: follow plan §7–8; retain the first
useful resource-measurement chunks, then calibrate the learning rates and execute
the bounded sweep. Existing e005/e006 training and r006 extreme-event evaluation
retain their separate scope/status. No new scientific result is asserted here.

<a id="e008"></a>
## e008 — Fresh scalar λ, full-GC fine-tuning and smoothing

Event/record date: 2026-11-07, America/New_York. Execution: submitted; initially
queued. Outcome: pending. User explicitly authorized implementation and start
of all three workstreams after reviewing the detailed plan.

**Question:** compare fresh GC–Mamba versus all-parameter GC fine-tuning under
the exact grid-MSE/AMSE-excess tradeoff, and determine what output attenuation
alone can explain. The [plan](../../../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md)
contains exact mathematical, initialization, optimizer, filter and evaluation
definitions and now records the launch.

**Implementation:** ordinary scalar AdamW, stable excess with inner λ1 and outer
λ=0/0.25/0.5/0.75/1, λ0 transform fast path, genuine full-GC parameter gradients,
native model-family tagging, direct full-field evaluation, step-zero checkpoints,
and shared seeded segment replay. GC–Mamba uses fresh Mamba1/seed22/zero output
heads with public spatial overlays and a frozen public GC. It never loads M1
or another trained correction as initialization. Both methods use full weather
BPTT24/K20/segment96, batch4, and a 2,500-update schedule. Gaussian and four-band
filters preserve the unresolved field and exact weighted mean, fit on2022 and
score output-only on2023. Existing jobs/snapshots are untouched.

**Artifacts:** [training manifest](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/manifest.json),
[smoothing manifest](../../../artifacts/evaluations/v24_smoothing_m1_20261107/manifest.json).
Shared frozen source digest is
`ab39c404b062a55f2ee93883672442973fa4a2351699529ac11492a1a6b88475`.
It contains the actual dirty-workspace implementation, independent of future edits.
The prepared format1 loader was verified against a real finite T2m field and
keeps `WEATHER_PREPARED_BOUNDED_READS=0`; no legacy loader replacement was needed.
Actual manifests contain106 eligible96-position segments and2,500 update mappings;
the last training target is2021-12-19T12. Development32 is resolved from actual
year-sliced store coordinates, from2022-01-02T00 to2022-12-01T12. Annual122 retains
the existing2023 cadence and January2024 verification tail. The year is reused
evaluation data, not an independent confirmatory test.

**Submitted jobs:** training array17022621 has GC λ0 rates1e-5/3e-5 at indices0/1,
Mamba λ0 spatial rates3e-5/1e-4 at2/3, and positive λ0.5 lower-rate profiles at4/5.
Each retains8 actual updates and requests1gpu80,8CPUs,384GiB,180min. Controllers
17022622 and17022623 depend only on their own method's three profiles. A GC
resource failure does not block Mamba. They size later jobs from measured timing
and host RSS, continue the two λ0 candidates to500, evaluate development32,
select a rate, and run the five λ arms in two production parts1250/2500. Winning
λ0 trajectories resume their own state. λ0.5 retains its profile only if its
rate is selected, otherwise it restarts fresh and the extra8 updates are recorded.
Full-GC fit is not yet demonstrated; LoRA has not been substituted.

Standalone GC/M1 smoothing first-start job17022624 requests1gpu80,4CPUs,32GiB,
120min; controller17022625 releases remaining development forecasts, fitting,
first annual forecast, measured annual shards and report. First forecasts remain
in their planned cohorts. All five top-level submissions were verified PENDING;
both resource shapes passed Slurm test-only checks. No concurrency throttle.

**Automatic downstream work:** each finished training method launches its five
models' smoothing/evaluation immediately. Saved intermediate checkpoints provide
development history. Final reports retain raw and filtered exact losses, all83
primitive spherical spectra, zonal diagnostics, negative humidity/precipitation
statistics and finite numerical warnings. The aggregate waits for all ten plus
M1, uses one canonical GC denominator, and writes λ-labeled plots and paired
date-block exact-loss intervals/contrasts. Spectral coordinates are descriptive
in the automatic summary; per-date powers remain available for nonlinear
spectral resampling. Additional seeds/AR horizons are not submitted.

**Validation:** focused scalar loss/gradient, real tiny-GraphCast initialization
and all-block gradients, streamed feedback, both-family native resume, manifest
mutation rejection, smoothing identity/mean/harmonic/GC-weighted scoring and
finite-warning behavior passed. Root data/rollout suite:47 passed,6 device-only
skips; engine agent:94 distinct checks across its focused runs; orchestration10;
smoothing12; aggregate report3 including a complete synthetic ten-arm fixture.
These suite counts are reported separately and may overlap. The synthetic
fixture is implementation evidence, not a scientific result. Cross-review fixed
historical stage proofs to pin immutable metrics snapshots while validating live
history before resume. Numerical drift/worsening remains advisory; nonfinite or
integrity failures remain errors.

**Next:** inspect real first-start logs and retained resource proofs. Successful
full-GC execution will establish whether LoRA is unnecessary for this shape;
resource failure requires diagnosis before changing the protocol. Scientific
comparison waits for final checkpoints and matched evaluation. The development
handoff gives monitoring and safe-resume details.

<a id="e008-fresh-lr3em7-20261107"></a>
### November 7 evening — Preserve pending λ0 job; start fresh at peak3e-7

Execution: pending after activation. Outcome: pending for the new trajectory.
The user explicitly authorized replacing the existing pending λ0 production
resume with a fresh public-GC initialization, while retaining its scheduled
job ID. This is an amendment within e008, not a new research direction.

**Observed motivation:** completed full-GC500-update calibrations gave exact
dev32 rollout losses12.3394641575 at peak1e-5 and16.4926026612 at3e-5. Their
public-GC reference losses were10.833993 and10.832740 respectively, yielding
13.8958% and52.2477% increases. The small finite reference difference remains
measured numerical variation, not an error. Both calibrations completed on
one80GB GPU; jobs17024647_0/1 took8h06m19s/8h04m07s for the resumed492 updates,
after approximately20-minute eight-update profiles. Development jobs17024648_0/1
took50m52s/50m09s. Steady p90 was59–60s/update. Full tuning therefore fits the
tested shape; LoRA is unnecessary as a feasibility fallback.

**Decision:** do not carry the earlier higher-LR500-update state into the new
λ0 run. Use peak3e-7 provisionally, informed by original GraphCast's low-rate
AR fine-tuning and the completed calibration evidence, without claiming a
parameter-count scaling rule or a measured optimum. Keep100-update warmup,
cosine decay to3e-8, seed22, batch4, K20/BPTT24 and the frozen2500-update data
order. Original higher-LR calibrations remain immutable evidence. Finite loss
increases do not automatically abort valid bounded training or its evaluation.

**Concrete change:** active task0 in `selected_arms_full_gc.json` now names
`gcft_s22_l000_lr3em7_fresh_20261107`, with `resume_step=0`. Training jobs
17049162_0→17049163_0 retain their IDs, submission age and existing one80GB GPU,
8CPU,384GiB,30-hour-per-part requests. The fresh first part reaches1250; second
part resumes exactly to2500. Its approximately21-hour nominal first-part cost
fits the existing30-hour request using the measured60s/update evidence.

This narrow authorization changes onlyλ0. The other four GCFT arms remain
peak1e-5; λ0.5 still resumes its original eight-update profile. The resulting
comparison must not be described as fixed-LR λ-only. Additional fresh1e-6/3e-6
pilots were discussed but were not submitted by this action.

**Downstream repair:** new job17062379 evaluates the new run's saved step500
checkpoint on the original32 development starts and40 leads after first-part
completion. It calls the unchanged frozen evaluator and writes a new matching
`calibration_evaluation/<new_id>/score.json`. Existing history job17049164_0
now waits on both17049163_0 and17062379, allowing its original λ0 reuse branch
to consume genuine fresh-run evidence. Final controller17049165 and smoothing
dispatch use the revised active selected-arm mapping.

**Provenance and validation:** [revision manifest](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/revisions/20261107_gcft_l000_fresh_lr3em7/manifest.json),
[activation](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/revisions/20261107_gcft_l000_fresh_lr3em7/activation.json),
[dispatch checks](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/revisions/20261107_gcft_l000_fresh_lr3em7/validation.json).
Original and replacement mappings are archived and sealed. Only the new
config's peak/end LR and output run name differ from the original. Frozen
source digest and original manifest/initial configs remain unchanged; the
small pinned evaluation helper lives outside that snapshot. Real frozen CLI
dispatch was exercised without training, verifying fresh1250 and resumed2500
paths; config loading, unchanged other task mappings, source/input hashes and
wrapper syntax passed. No GPU qualification or repeated profile was added.

<a id="e008-smoothing-spectra-20261109"></a>
## 2026-11-09 — Completed GC/M1 smoothing spectra inspected

Within e008, the user requested power spectra for the completed GC/M1 smoothing
comparison. Reused the saved122-start2023,40-lead statistics; no forecasts or
training were rerun. Both filters were independently fitted on32 development
starts in2022 and applied to output fields without feedback of filtered fields.

The [smoothing report](../../../artifacts/evaluations/v24_smoothing_m1_20261107/report.json)
pins `annual_pooled.npz`; its SHA256 was rechecked. The analysis reproduces the
saved macro power error exactly (maximum absolute difference0). The
[reproducible report and tables](../../../plots/analyze_models/data/resolution_eval/v24_smoothing_m1_spectra_20261109/report.md)
and [figures](../../../plots/analyze_models/images/resolution_eval/v24_smoothing_m1_spectra_20261109)
were generated with
[report_v24_smoothing_spectra.py](../../../scripts/analyze_models/report_v24_smoothing_spectra.py).

| Model/filter | Exact GC rollout loss reduction vs raw GC (%) | Rollout mean relative power error (%) | Day10 relative power error (%) |
| --- | ---: | ---: | ---: |
| GC raw | 0.00 | 59.47 | 65.69 |
| GC Gaussian | 8.87 | 79.17 | 92.45 |
| GC band | 9.39 | 75.70 | 88.37 |
| M1 raw | 14.38 | 67.57 | 84.35 |
| M1 Gaussian | 14.95 | 77.23 | 91.00 |
| M1 band | 15.33 | 73.90 | 86.18 |

Power error is100 times the equal-weight mean of
`abs(pooled_forecast_power / pooled_truth_power - 1)` over83 physical channels,
degrees1–179 and40 leads. Powers are pooled over starts before ratios. Degree0
is excluded, with no climatological-mean subtraction; no zero-truth bins occur.
This diagnostic is distinct from GC-weighted grid loss. The displayed mean
channel power ratio can cancel excesses and deficits, so the report also gives
the absolute-error score and physical spectra for six fields at day10.

**Interpretation:** raw M1 has better loss and better macro power matching than
either filtered-GC baseline, but worse power matching than raw GC. Both filter
families improve rollout loss while worsening the overall power score for each
model. Intermediate-scale deficits coexist with high-degree excesses in some
channels; no uniform-smoothing or physical-mechanism claim follows. The results
are descriptive fixed-checkpoint estimates on the reused2023 cohort, without new
spectral bootstrap intervals. Numerical diagnostics and reference bounds are
preserved in `summary.json`; none are used as finite-variance acceptance gates.

**Next:** incorporate these reference spectra into Ilya's λ tradeoff analysis
when the new models' evaluations complete.

<a id="e009"></a>
## e009 — Full-GC AR horizons, checkpoint spectra and SWA

Event date: 2026-11-09
Recorded: 2026-11-09 (America/New_York)
Execution: prepared; new jobs not submitted
Outcome: pending

**Question and change:** The user requested full-GC K8/K12/K16/K20 training,
2500 updates, evaluations every500, possible SWA and power spectra for each.
The fixed24-step window uses16/12/8/4 teacher-prefix steps respectively. K20
is the existing `3e-7` run; K8/K12/K16 are fresh from public GC with the same
seed, learning-rate schedule and frozen data order. Prefix length therefore
changes with K and limits causal interpretation.

**Existing evidence:** K20 update500 exact development rollout loss10.337334
versus public GC10.840716, a4.6434% reduction on32 2022 starts; day10 loss
reduction5.3090%. Its update1250 checkpoint exists; the continuation was
pending at the queue check. The prior higher-LR500-update degradations remain
recorded in e008. These are intermediate results, not conclusions about the
new K8/K12/K16 models.

**Prepared implementation:** [plan](../../../docs/experiments/v24_full_gc_ar_horizons_2026-11-09.md),
[manifest](../../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/manifest.json),
[submission plan](../../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/submission_plan.json).
The design evaluates native500/1000/1500/2000/2500 and SWA1000–2500 for all
four horizons on the32-start2022 cohort and saves spherical forecast/truth/error
power at all40 leads and83 channels. Finite drift/loss degradation is advisory;
all valid trajectories reach their planned endpoints. No 2023 confirmation or
training-seed generalization is implied.

**Validation:** horizon/configuration and dependency tests, real spherical
observer and report tests, and existing scalar-campaign tests passed. Full-GC
SWA was also built from the real K20 500/1000 checkpoints in a temporary
directory; exact averaging of a parameter leaf and model-family identity were
verified. K20 accounting supplies budget evidence, and `sbatch --test-only`
accepted representative training/evaluation requests without submitting jobs.

### Reviewer-comparison audit — 2026-11-09

The user asked to wait before submission and check fairness relative to M1.
The prepared report contains only full-GC horizons/public GC on32 development
starts; it does not include M1 or a2023 evaluation. The 32 saved M1 development
initialization times match the prepared manifest exactly (four shard metric
hashes verified), and M1 raw power/loss statistics already exist for122 2023
starts. Reuse these rather than duplicate forecasts where the saved statistics
are sufficient. The M1 development calibration files do not contain raw spherical
powers; those require separate extraction or evaluation. The repeated public-GC
rollout loss across the32 shared development dates was10.822104 in the M1
calibration versus10.840716 in the K20 evaluation, an absolute difference
0.018612 or0.1720% relative to the former. Retain this finite difference as a
warning, not a failure or selection gate, and report headline comparisons with
one explicitly identified common baseline.

M1 and the new full-GC recipe differ in gradient handling, optimizer stages,
data ordering and trainable parameter count despite matching backbone, training
years, K20 window, batch and total updates. Different K also changes the teacher
prefix within the fixed24-step window. The draft saves error/cross powers but
only plots forecast power, and a power spectrum alone does not resolve the
reviewer's smoothing concern. The [audit and proposed revisions](../../../docs/experiments/v24_full_gc_ar_horizons_2026-11-09.md#reviewer-comparison-audit--before-submission)
identify a common2023 comparison, paired uncertainty and filter controls.

**Next:** revise the draft for M1/2023 and report the training differences; keep
all valid trajectories and their evaluation regardless of finite metrics. Do
not submit the current draft while this review hold remains in place.

<a id="e010"></a>
## e010 — GC/M1 output-smoothing strength tradeoff

Event date: 2026-11-09
Recorded: 2026-11-09 (America/New_York)
Execution: submitted; initially pending for priority/dependency
Outcome: pending

**Question and authorization:** Following the completed GC/M1 smoothing spectra
(e008), the user asked for a series of different smoothers to explore MSE versus
spectrum and explicitly authorized implementation/submission. This is distinct
from e009's full-GC horizon proposal, which remains on hold.

**Fixed design:** Public GC and principal M1, same122 starts in2023,40 leads,
83 channels. Global Gaussian σ=0,.25,.5,.75,1,1.5,2,3,4,8 degrees; blends
α=0,.1,…,1 toward independently development-fitted Gaussian and four-band
profiles. The existing32-start2022 fits are reused unchanged. Shared identity
gives30 distinct points/model. No training, refitting or selection on2023.
Show all points and retain the disclosed prior exposure of2023.

**Implementation and evidence:** [protocol](../../../docs/experiments/v24_smoothing_strength_sweep_2026-11-09.md),
[manifest](../../../artifacts/evaluations/v24_smoothing_sweep_m1_20261109/manifest.json),
[validation](../../../artifacts/evaluations/v24_smoothing_sweep_m1_20261109/checks/pre_submission_validation.json).
One forecast pass saves exact native-grid E0/q/Q and spectral moments of actual
synthesized corrections for11 bases. This avoids incorrectly substituting a
continuous spherical norm for GC's native-grid loss and supports later α
refinement without another forecast pass. Filters preserve the discrete mean
and unrepresented residual and are never fed back into forecasting.

All15 focused tests passed, including direct-field polynomial equivalence,
original endpoint equivalence, native/spherical norm distinction, pool-before-ratio,
provenance checks, finite-warning behavior and Slurm dependencies. Both plot
layouts rendered and were inspected using synthetic data; previews are not
scientific evidence. Source inherits the parent's forecast implementation plus
five explicit extensions; digest
`22d780d97df6119954c3ca1f4349cf107ecdca26892ae271bd71f9f63780f5e9`.
M1 checkpoint SHA256 is
`b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0`.

**Submission:** evaluation array17213596 (eight15/16-start shards) and dependent
report17213597. Initial queue check: all evaluation tasks pending for priority;
report pending on dependency. Per GPU shard:4CPUs,20GiB host RAM,180min,1×80GB
GPU, no throttle. Completed parent array17028677 used43–46.5min and≤10.3GiB
for17/18 starts; new resources allow11 rather than2 corrections and additional
moments without redundant qualification. Statistics are about41GB uncompressed.

**Interpretation and next action:** No new forecast result yet. Inspect completion
proofs and report before claiming a tradeoff. Main metric is exact GC rollout
loss versus mean absolute relative power error, with date powers pooled before
ratios and equal weighting over channels/positive degrees/leads. Save day10
degree curves, error-power/correlation summaries and per-start exact losses.
Initial plots have no confidence intervals. Repeated endpoint/mean drift is
recorded with reference bounds as warnings, never a finite-metric stop rule.

<a id="e011"></a>
## e011 — Fast raw-M1 and fresh lambda checkpoint MSE/spectrum pilot

**Date:** 2026-11-10 (America/New_York). **Execution:** running (submitted,
initially pending). **Outcome:** pending.

User requested M1 without SWA, corresponding existing lambda checkpoints, and
a fast pilot showing MSE and power spectra. Use the predetermined final M1
main-stage raw2000 checkpoint (2500 total updates including warm-up) and all ten
fresh GC–Mamba/full-GC scalar-λ arms at their latest common saved raw1250
endpoint. Include historical M1 SWA separately. Existing fresh continuation
jobs remain unchanged; the four-new-horizon training hold remains intact.
No answer to the optional historical-versus-fresh clarification arrived before
preparation, so the latest fresh campaign was used as the stated default.

Four starts are the first development32 date in each quarter: Jan2 00Z,
Apr3 06Z, Jul7 18Z, Oct7 06Z in2022. All models forecast40 six-hour leads with
cold zero recurrent state/full weather feedback. Save exact GC grid MSE and
physical T179 truth/forecast/error/cross powers for all83 channels/leads.
Report MSE curves, T2m/Z500/U850 day5/day10 powers/ratios, and pooled MSE/power
tradeoffs. This is interim four-date evidence, not annual skill or a search for
the best checkpoint. GCFT λ0 uses peak LR3e-7 versus1e-5 for positiveλ, so it is
not a fixed-LR λ-only experiment.

Reused32-start2022 evidence: raw M1 rollout MSE9.167232 versus paired public
GC10.818541 (15.2637% reduction); SWA9.064574 versus10.835462 (16.3434%). Retain
the finite repeated-baseline difference as a warning. The new pilot uses one
common GC denominator and records decreases versus measured baseline spread
without any acceptance gate. Missing/corrupt inputs, source/checkpoint mismatch
and nonfinite computation remain errors.

Evidence: [protocol/checkpoint table](../../../docs/experiments/v24_raw_checkpoint_pilot_2026-11-10.md),
[manifest](../../../artifacts/evaluations/v24_raw_checkpoint_pilot_20261110/manifest.json),
[submission records](../../../artifacts/evaluations/v24_raw_checkpoint_pilot_20261110/submissions).
Runtime digest `8203142838c3c60027df2f816a15ab11cb1da0cfb1e57f2e23ec2fe09d1b5324`.
Twelve focused driver tests passed; synthetic report rendering and provenance
validation passed. Evaluation array17249273 has12 unthrottled elements,
1×80GB GPU/4CPUs/16GiB/62min each. Report17249274 is CPU2/8GiB/15min with
afterok dependency. Measured references17022624/17024230_1 justify memory and
roughly10–20min/model runtime;62min avoids≤1h gputest routing limits. Queue
time is unknown. Submission is not scientific completion evidence.

Next: inspect complete records and the report under
`plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110/`;
PNG/PDF figures go in the matching `images/resolution_eval/` directory.

### e011 scope clarification: both lambda sets — 2026-11-10

The delayed clarification explicitly requests **both fresh and historical AMSE
sets**. The original12-task pilot remains intact. Added eight historical raw
evaluations on the identical four2022 starts/40leads: September24 λ0.1 and its
MSE control at2000; September19 λ1/control at2000; October4 spectralλ0,
λ0.25,λ0.5 and exact-grid MSE control at500. October evaluated endpoints are
unaveraged, but their training initializer was September MSE SWA. These are
three histories with an older spectral-interpolation objective, not a single
controlled lambda sweep or the fresh objective. M1 raw/SWA forecasts are reused
from original tasks0/1 after validating parent/cohort/checkpoint/source identity.

Historical extension campaign:
`artifacts/evaluations/v24_legacy_raw_checkpoint_pilot_20261110/`.
Array17250938 runs indices2–9, oneGPU/4CPU/16GiB/62min each, no throttle.
Report17250939 waits for that array and17249273_0/1. Runtime inherits all
computational source from the original frozen pilot, with three orchestration/
report overlays; digest
`5c714cfb5e73c34f5c43a32ce93c3cfcff5892a46b070c4810b01e27cb9061c5`.
15 driver/preparation and6 reused-report tests passed; grouped plots rendered
and inspected with synthetic input. New four-date scientific results pending.

**Completed cached evidence:** [report](../../../plots/analyze_models/data/resolution_eval/amse_lambda_raw_cached_20261110/report.md)
and [overview](../../../plots/analyze_models/images/resolution_eval/amse_lambda_raw_cached_20261110/mse_and_power_overview.png)
reuse all eight raw endpoints on the same eight2023 starts/40leads. Main exact
MSE and zonal powers use identical forecast samples; development32 losses are
separate. Sources/checkpoint hashes and coverage verified (97 pinned inputs);
ten finite reference warnings retained. No new inference for this report.

Within October, spectralλ0→λ0.25→λ0.5 gives exact MSE9.774562→9.888138→10.132798
and aggregate zonal power mismatch56.8131%→55.0899%→53.3911%. September24
λ0.1 gives MSE10.060745 versus control9.958371, power55.3268% versus57.2455%.
September19 λ1 gives MSE12.258391 versus control9.914750/publicGC11.212183,
but power27.5748% versus control57.2108%/GC49.5065%. Thus power matching
improves at an accuracy cost within each history. This is descriptive8-date
evidence without intervals or training-seed variation. Zonal scores pool powers
across dates, then average absolute relative errors across nine primitive fields,
modes1–180, leads and30–60°N/S bands; they are not the spherical83-channel score.
Finite baseline differences/bounds and decreases exceeding observed baseline
spread are saved as advisory diagnostics. Follow both sets through the new
matched spherical evaluation before comparing them to raw M1.

### e011 completion inspected — 2026-11-11

**Execution:** completed. **Outcome:** descriptive/inconclusive beyond the
four-date pilot; removing M1 SWA is unsupported as an improvement here.
All20 unique evaluations and both reports completed with exit0. Sealed reports
and all referenced checkpoint/output hashes verified, with4dates/40leads and
byte-identical M1 reuse. [Full results](../../../docs/experiments/v24_raw_checkpoint_pilot_2026-11-10.md)
link both tables and spectrum figures. No new jobs or training changes.

Canonical public-GC rollout MSE11.133143. M1 raw2000 gives9.424010
(15.3518% reduction) and spherical power error75.9431%; SWA gives9.171727
(17.6178%) and71.8861%. SWA lowers raw M1 MSE2.6770% and power mismatch4.057pp.
Raw/SWA day10 MSE22.180490/21.857055; day10 power error106.5706%/93.6930%.

Fresh GC–Mamba λ0/.25/.5/.75/1 at1250: MSE10.8511/10.9865/11.0667/11.0064/
11.0951, spherical power error64.321/63.440/62.843/63.112/64.801%. All have
higher MSE than M1 but lower aggregate power mismatch. Fresh full-GC λ0
improves MSE4.6934%; positiveλ worsens MSE9.28–17.35%. λ1 full-GC power
error58.539% improves over GC61.571% at12.608% worse MSE. Retain the λ0
peakLR3e-7 versus positiveλ1e-5 caveat; these are interim1250 checkpoints.

Historical Octoberλ0/.25/.5: MSE9.187175/9.260470/9.462377 and power error
73.5976/72.0097/71.0146%. λ0.25 improves both versus raw M1 (MSE1.7354%
lower), while M1 SWA is slightly better on both. λ0.5 improves power further
but has higher MSE. Septemberλ0.1/control: MSE9.580461/9.379569 and power
76.9957/74.8652%; Septemberλ1/control:10.781693/9.436260 and76.7759/76.2668%.
Unlike the cached2023 zonal score, the September positiveλ arms do not improve
the new spherical aggregate. Different dates/domain/channels/operator mean
the earlier blanket spectral interpretation must remain confined to its
eight2023-start zonal setting. Plots show scale dependence: M1 attenuates U850
intermediate/fine scales, while Z500 has excess at the highest degrees.

Observed repeated-GC rollout spread0.0793851 (0.71305% of canonical loss),
day10 spread0.471028 fresh/0.294249 historical. Raw→SWA MSE decrease0.252283
exceeds the baseline spread, whereas fresh GC–Mamba λ0.5/λ1 decreases do not.
These are only baseline-repeat diagnostics, not model repeatability or CIs.
All finite warnings remain advisory (22 fresh/18 historical); zero excluded
truth-power bins. Tiny differences such as rawM1 versus Octoberλ0.5 MSE
(0.03837) are below this observed spread and should not be ranked strongly.

Resource evidence from sacct:20 GPU tasks took14m44s–15m46s each, totaling
5.0644 GPU-hours; maximum batch RSS8.7528GiB. Reports49/52s. Keep16GiB host
RAM for this shape;62min allocation avoids documented short-partition routing.
Scientific next step is broader matched evaluation if requested; this report
does not authorize new seeds, training or removal of the full-GC horizon hold.

### e011 final2500 raw/SWA follow-up — 2026-11-11

**Execution:** submitted, waiting for training. **Outcome:** pending for the
fresh models. The user explicitly requested2500 updates and comparison of MSE
and power spectra both with and without SWA. The1250 points were intermediate
operational boundaries, not the intended final training budget. All ten fresh
scalar-lambda runs retain their existing2500 schedules and queued second halves
(fullGC17049163, GC–Mamba17062358); no training job was replaced or restarted.

[Protocol](../../../docs/experiments/v24_final_checkpoint_comparison_2026-11-11.md)
and [sealed plan](../../../artifacts/evaluations/v24_final_checkpoint_comparison_20261111/plan.json).
CPU materialization17353172 waits for both training arrays, validates each
final2500 completion proof and hashes actual source checkpoints, then builds
ten equal-weight SWAs at1000/1500/2000/2500 with zero recurrent memory.
Evaluation17353173_2–21 supplies20 new four-start/40-lead exact-MSE and spherical
forecast comparisons; report17353174 follows. Principal M1 raw/SWA outputs are
reused after cohort/checkpoint/source checks, yielding22 model rows plus GC.
The SWA window matches M1's total-update1000–2500 window; training recipes still
differ. GCFTλ0 peak3e-7 versus positiveλ1e-5 remains an explicit limitation.

Frozen runtime digest:
`90ccacb212e0ba60c7e9e428f94cda3480c8086e371c58120a79546a6471df52`.
The source inherits the completed pilot computation with only preparation,
wrapper, evaluation orchestration and reporter overlays. Eight focused driver
tests and15 report/reuse tests passed; three resource test-only submissions
passed. Existing pilot timing/memory evidence sizes20 new GPU tasks at16GiB,
4CPUs,62min each (about5GPUh execution total), plus8GiB/30min CPU preparation
and4GiB/15min reporting. No extra numerical qualification or finite-skill gate.
Synthetic full-size plots and real reused M1 figures were inspected.

The [completed M1-only report](../../../plots/analyze_models/data/resolution_eval/v24_m1_raw_swa_20261111/report.md)
reformats the existing evidence without new inference: raw/SWA rollout MSE
9.424010/9.171727 and spherical power mismatch75.9431%/71.8861%; SWA lowers raw
MSE2.6770% and mismatch4.0570pp. Day10 MSE reduction versus raw is1.4582% and
power mismatch reduction12.8777pp. MSE decreases exceed observed baseline
repeat spread, retained as advisory rather than a full model-noise estimate.
Both M1 variants still worsen aggregate power matching versus GC61.5708%.
Finite warnings remain preserved; no annual or training-seed claim is made.

### e011 matched-budget setup audit and comparison — 2026-11-12

**Execution:** setup audit completed; four new forecast evaluations submitted.
**Outcome:** pending at1000/1250. The user questioned full-GC lambda0 raw1250
MSE10.610624 versus historical M1 and requested accessible1000/1500 checkpoints.
M1 main-step counters exclude500 preceding updates. November1500 does not yet
exist; matched total1000 uses M1 main500 and November step1000. Matched total1250
uses M1 main750 and November step1250. This is a saved-checkpoint comparison;
no training schedule was changed or started.

Array17404694_2–5 evaluates M1 total1000/1250 and November GC–Mamba/full-GC
lambda0 total1000. Report17404732 reuses both completed November1250 forecasts
and M1 final raw/SWA. All new tasks were pending for priority; the report was
pending its dependency at submission inspection. The common four2022 starts,
40leads, cold_full/zero-state protocol, public baseline, exact GC loss and T179
spectra are inherited from the completed pilot. Computational-source parity
and serialized checkpoint step/family/config/backbone checks passed.

The audit found no mismatched evaluation dates, metric, state policy or full-GC
weight loading. Full-GC's fine-tuned tree differs from public GC in256/262 leaves
and is used independently of the public reference. Training is not identical:
M1 trains correction parameters with stopped weather gradients and a500+2000
schedule/reset, whereas November full GC trains all backbone parameters with
full weather gradients and one2500 schedule. Peak rates differ; M1's original
normalized residual loss runs inside BF16 casting while November lambda0 uses
full-field FP32 grid MSE. Mathematical weights match but arithmetic and
backward paths do not. M1/NovemberGCM preclip median gradient norms0.333/19.948
also demonstrate a changed optimization regime, not an isolated causal proof.
All1250 NovemberGCMlambda0 logged spectral contributions are zero.

Existing32-start/40lead raw results at500 total updates are M1 MSE9.819859,
NovemberGCM10.314396 and fullGC10.337334. Using common publicGC10.844687,
reductions are9.4500%,4.8899%,4.6784%; original paired reductions remain in
the saved JSONs. Baseline repeat spread0.0251871 is smaller than both gaps to
M1, but is not model noise or a confidence interval. Physical dates match,
although older evaluators used their respective source versions; the new
comparison fixes this. The budget mismatch alone is not a sufficient account
of the available evidence; which training change matters remains unisolated.

[Protocol and sources](../../../docs/experiments/v24_matched_budget_comparison_2026-11-12.md).
Campaign `artifacts/evaluations/v24_matched_budget_comparison_20261112/` has
manifest hash `2ee6bb4c9e351ea6f03c24e2393bdb76b14079d6d29e1523d99177c9f14aed1f`
and source digest `04e82e3820ecb955e7ecd5a21fed1b91c58a252a6b443a36ef58c7d03b595d3d`.
Four focused preparation tests and15 report/reuse tests passed. Four GPU tasks
use16GiB/62min each, expected about1GPUh total from the completed pilot;
CPU report uses4GiB/15min. Finite differences remain warnings. The final2500
raw/SWA dependency chain and separate four-horizon training hold are preserved.

<a id="e012"></a>
## e012 — Fresh GC–Mamba lambda sweep matched to M1's training recipe

Event date: 2026-11-12 (America/New_York).
Execution: running (submitted; initially pending in Slurm).
Outcome: pending.

**Current scope:** after reviewing the proposal, the user authorized restarting
GC–Mamba only and explicitly preserving full GC/DeepMind. The five-arm new
campaign and mixed downstream comparison were submitted on November12; no new
training or evaluation result is available. The original two-family review is
preserved below and is superseded by the narrower activation entry.

### Original review, before activation

**Question and change:** the user rejects the unintended training-recipe
differences between M1 and November GC–Mamba lambda0. They request the same LR,
schedule, loss calculation and other settings for all GC–Mamba lambda arms,
and the corresponding full-GC recipe with its current learning rates retained.
They request two allocations of1,250 updates and a report for checking first.

**Proposed setup:** five lambdas per family (0/.25/.5/.75/1), fresh public-GC
initialization with seed22 and fresh Mamba/zero heads where applicable.
Historical M1 initial500 uses constant1e-4 after50-update ramp and all LR
multipliers1; main2000 uses cosine1e-4→1e-5 with50-update ramp and Mamba3×.
The initial/main boundary transfers parameters only and resets optimizer, RNG,
cursor and recurrent state. Restore stopped weather-feedback gradients, native
BF16 MSE, batch4/BPTT24/K20/segment96 and deterministic contiguous segment groups.
Source and segment-array inspection confirmed all106 historical/current eligible
segments match; the old November order shuffled these segments by epoch.

Each arm applies its lambda during both stages. Positive lambda adds the existing
FP32 stable spectral AMSE excess to the native base loss; lambda0 retains M1's
original native loss path without a spherical transform. Full GC uses its native
GraphCast BF16 loss and complete forecast, with weather gradients stopped.
Its initial/main peak remains3e-7 for lambda0 and1e-5 for positive lambda;
main cosine ends atone-tenth of the peak. The full-GC LR/lambda confound remains
explicitly preserved by the user's instruction to retain current rates.

**Allocation mapping:** first part initial500 + main750 =1,250 total; second part
exactly resumes main750 through2000 =another1,250. No reset occurs at the
allocation boundary, which falls midsegment. Final SWA uses main500/1000/1500/2000,
the same total1000/1500/2000/2500 positions as historical M1.

**Implementation state:** the [review plan](../../../docs/experiments/v24_m1_matched_lambda_restart_2026-11-12.md)
and [machine-readable specification](../../../artifacts/plans/v24_m1_matched_lambda_restart_20261112/review_plan.json)
are prepared, with10 arms/20 stage specifications and pinned reference inputs.
These are not executable training configurations. Current validation requires
full weather gradients for scalar excess and full GC. Native-loss-plus-excess,
full-GC stopped-feedback prediction semantics, step-zero recording and stage
orchestration need implementation and focused checks before activation.
Full GC's state-increment target and M1's correction target have matching
mathematical weights but inherently different BF16 rounding representations.

**Scheduler finding:** current continuation arrays17062358/17049163 have spooled
phase1 arguments that force old resume1250→2500. Unlike the November7 phase0
revision, mutable selected-arm mappings cannot redirect them to a fresh start.
Installed Slurm26.05.5 has no documented user-level spooled-command replacement;
do not rewrite frozen source to bypass it. The clean proposal replaces these
arrays and old downstream history/controllers/final raw-SWA plans with new IDs
and new queue age. Preserve all completed evidence and independent diagnostic
17404694→17404732. Rebuild final plans rather than merely changing dependencies,
because old plans pin old configurations, checkpoint paths, proofs and sources.

**Validation and decision:** ten arms, twenty stages, per-arm rates and2,500 total
updates were checked; pinned active reference inputs remained unchanged. No
training code, active mapping, job or frozen source was changed. Existing
384GiB/one80GB-GPU/8CPU/about30h requests provide conservative provisional
budgets from measured old p90 costs near59–60s/update. Await the requested review,
then implement/check the declared paths before activation. Finite drift, clipping
and worsening remain warnings; no new numerical qualification or descent gate.

### 2026-11-12 — User approval narrowed to GC–Mamba; replacement submitted

The user's authorization was: cancel and submit fresh **only GC–Mamba**, keeping
DeepMind/full GC. Consequently no full-GC configuration, training trajectory,
source snapshot or queued training/history/finish job was changed.

**Implemented recipe:** GC–Mamba lambda0 uses M1's original native loss path;
positive lambda adds the existing FP32 stable spectral excess to the same native
BF16 base loss through the new `native_mse_amse_excess` mode. All five arms use
M1's stopped weather-feedback gradients, retained memory gradients, historical
deterministic segment order, batch4/BPTT24/K20/segment96, seed22 and AdamW settings.
Initial500 uses constant1e-4 after50-update ramp and Mamba multiplier1. Main2000
uses cosine1e-4→1e-5 with50-update ramp and Mamba multiplier3. The initial/main
transition transfers parameters only and resets optimizer, RNG, data cursor and
recurrent state. Lambda applies during both stages. Finite worsening and drift
remain recorded warnings rather than stopping or acceptance conditions.

**Jobs submitted:** first-part array **17406861_0–4** runs initial500+main750.
Second-part array **17406862_0–4** depends with `aftercorr:17406861` and exactly
resumes main750→2000, another1,250 updates and2,500 total. The allocation split
does not reset any state. Development history array **17406863_0–4** follows
each completed second part and evaluates total500/1000/1500/2000. Finish
**17406864** follows the full second-part array and dispatches development
fitting and annual smoothing using the actual main-step2000 endpoint.

The superseded GC–Mamba chain **17062358/17062359/17062360** and obsolete joint
final-comparison chain **17353172/17353173/17353174** were cancelled. Full-GC
**17049163/17049164/17049165** and saved-checkpoint diagnostic
**17404694→17404732** remain unchanged. All completed old checkpoints,
evaluations, configurations and frozen sources are preserved.
The [activation verification](../../../artifacts/plans/v24_m1_matched_lambda_restart_20261112/activation_gcm_only/activation.json)
at14:18 Eastern confirms all seven new job roots pending with their prescribed
dependencies, all six obsolete roots cancelled, and the five protected roots
unchanged.

**Mixed final raw/SWA comparison:** materialization **17406897** depends on
unchanged full-GC array17049163 and new GC–Mamba array17406862. Evaluation
**17406898_2–21** and report **17406899** follow. The twenty new evaluations
cover raw and SWA weights for five arms per family; M1 raw/SWA forecasts are
reused. All use the same four2022 starts and40 leads, exact GC MSE and spherical
spectra. GC–Mamba SWA stored steps500/1000/1500/2000 and full-GC steps
1000/1500/2000/2500 represent the same total-update positions. Source/weight
identities are checked, and future weights receive hashes only after completion.

The new finish hook retains independently development-fitted smoothing and
122-start annual evaluation. Its dependency-based annual join uses actual
submission records from new GC–Mamba and unchanged full GC; it does not wait
for cancelled old GC–Mamba completion. The combined report uses the completed
M1 reference, one common public-GC denominator and explicit training-recipe
limitations. Annual jobs beyond the submitted finish hook have not yet been
assigned IDs or produced results.

**Provenance:** [new training manifest](../../../artifacts/checkpoints/v24_Ilya/res1_m1_matched_gcm_lambda_20261112/manifest.json),
record hash `f75dbbdec1e1af4147e822bfe8a639e4775589d1abebdf26fbb1b13b29ce739c`,
frozen source `a6a1c4e7ae895eb788227924146b3da84e5542bb10b7f5e294ebbe15943abd7f`.
[Mixed final plan](../../../artifacts/evaluations/v24_mixed_final_comparison_20261112/plan.json),
record hash `f6d00da5679d73c3285435ba0d71a6dddd800c956b19e00b35a8da1d05ff4f10`,
frozen evaluator `8a5f9c9a8b02995e333de37a68d626ea5aaf940294b9c8a20ea21e7fa1935150`.
The evaluator inherits the original pilot's forecast/metric computation;
training implementation changes do not alter reused reference forecasts.

**Validation and resources:**116 native-loss/reset/resume tests,8 campaign
tests and18 mixed comparison/SWA/annual-report tests passed. All ten stage
configs validated. Training requests one80GB GPU,8CPUs,384GiB and30hours per
part; history uses20GiB/4hours and finish8GiB/30min. Mixed materialization uses
8GiB/30min, each four-date forecast16GiB/62min, and report4GiB/15min. These
submissions establish execution plans, not scientific improvement.

**Interpretation and next step:** inspect actual completion records and the
new MSE/power reports. GC–Mamba now follows M1's recipe; full GC deliberately
retains full weather gradients, FP32 grid loss, its original data order and
continuous schedule, with lambda0 peak3e-7 and positive-lambda peak1e-5. The
mixed comparison therefore does not isolate architecture under identical
training conditions. The separate four-fresh-horizon full-GC hold remains.

### 2026-11-12 — Portable execution handoff for both studies

The user requested runnable code and a README for Lianghong's cluster and
explicitly selected both the M1-matched GC–Mamba lambda sweep and the separate
four-fresh-horizon full-GC study. This is a preparation milestone, not a new
training result or a remote submission. Existing Della jobs were unchanged.

The [handoff README](../../../docs/experiments/README_LIANGHONG_PORTABLE_RUNS.md)
provides exact setup, two-part training, raw/SWA comparison, development-only
selection, annual evaluation and fitted-filter/strength commands. The
[code archive](../../../artifacts/handoffs/lianghong_weather_20261112/lianghong_weather_20261112_code.tar.gz)
contains the submitted computation unchanged plus portable adapters. Separate
public-GC/M1 reference weights and checksums are in the same handoff directory.
Large prepared arrays and evaluation truth remain external inputs with recorded
locators/inventories; the README distinguishes historical truth from Lianghong's
revised truth and records the runtime difference in her earlier package.

All31 packaged CPU tests passed. Real-input preparation validated all five
GC–Mamba arms, all four fresh full-GC horizons, relocated full-GC sample replay,
and M1 smoothing with fixed development32/annual122 cohorts. Evidence is in
`artifacts/handoffs/lianghong_weather_20261112/validation/`. The portable source
digest is `8b363120bf494948770ecfa90c460391057600401edd4690eed23d3e70d01f1e`.
No remote installation, GPU run or scientific improvement is claimed. Next:
Lianghong supplies local data/scheduler paths and returns job records and
completion artifacts; keep this execution distinct from the active Della runs.

### 2026-11-12 — User excludes full-GC peak1e-5 from comparisons

The user requested deleting the full-GC1e-5 results or marking them bad and
never retrieving them for comparison, and specified3e-7 for full-GC work.
Chose preservation with explicit scientific-comparison exclusion. Exact run
identities and rationale are in the [policy](../../../docs/experiments/v24_full_gc_comparison_policy_2026-11-12.md);
root AGENTS.md records the lasting instruction. This is a user-directed
scope decision, not an execution-failure classification or a finite-loss gate.
Historical numerical values, source integrity and warning evidence remain intact.

Only full-GC lambda0 has3e-7 results: step500 exact rollout loss10.337334 versus
GC10.840716 on32 development2022 starts (4.6434% reduction), and step1250
10.610624 versus11.133143 onfour2022 starts/40leads (4.6934%). These cohorts
cannot be treated as a learning curve. The accepted1250 model has day10 loss
reduction8.0881%; no positive-lambda3e-7 result or final2500 checkpoint exists.

Live Slurm inspection found full-GC17049163_0–4 pending: lambda0 retains3e-7,
positive lambda retains1e-5. Their history/finish and mixed final chain remain
pending, with the old frozen mixed plan still referencing excluded lineages.
The new GC–Mamba17406861/862 remains pending with M1 spatial peak1e-4 and
main Mamba multiplier3. Portable full-GC horizons already use3e-7 atlambda0.

Asked whether to replace the pending high-rate full-GC continuations with
fresh3e-7 runs; no scheduling action is claimed by this eligibility update.
Existing high-rate1250 checkpoints cannot be relabeled or resumed as fresh
low-rate experiments. Preserve the eligible lambda0 trajectory and the
independent GC–Mamba recipe. Current report code applies explicit lineage
exclusions to new presentation outputs while preserving sealed old reports.

Implementation verification:36 report/preparation regressions passed, followed
by8 focused policy/scalar tests after ensuring a known excluded lineage cannot
be made eligible by relabeling its learning rate. Cached pilot reporting and
the combined M1 figure now use separate `_eligible_20261112_v2` directories;
eligible tables contain nine rows and exclude the four high-rate production
arms. Original13-row measurements remain sealed. There was no new inference,
checkpoint rewrite, job submission or cancellation in this marking operation.

<a id="e013"></a>
## e013 — Fresh positive-lambda full-GC sweep at peak3e-7 (superseded)

**Later November12 correction:** superseded by [e014](#e014). All newly
submitted cosine training/downstream jobs below were canceled while pending,
as were oldlambda0 continuation/history17049163_0/17049164_0. Their saved
artifacts remain historical. No e013 positive-lambda training result was produced.

Date:2026-11-12 (America/New_York). Initial execution: submitted/pending;
canceled before starting under e014. Outcome: superseded; no new
positive-lambda cosine result was produced.

**Decision:** the user explicitly chose to replace the four rejected full-GC
positive-lambda continuations with fresh3e-7 runs. Preserve the existing3e-7
lambda0 trajectory, GC–Mamba training/history, saved-checkpoint diagnostic,
and all historical weights/logs/source. The e012 exclusion remains in force.

**Recipe:** lambda0.25/0.5/0.75/1 start independently from public GC-small
with new optimizer state, peak3e-7→3e-8,100-update warm-up and continuous2500
updates. Seed22, batch4, BPTT24/K20, segment96, full weather gradients, exact
grid MSE plus stable AMSE excess, and frozen shuffled order match the accepted
lambda0 recipe. Fresh0→1250 and exact-resume1250→2500 are allocation boundaries,
not an optimizer reset. No profile, LR calibration or GPU qualification added.

**Activation:** new arrays17421838→17421839 each contain four arms; history
17421840 and completion17421841 follow. Retained lambda0 uses17049163_0 and
17049164_0. GC–Mamba17406861/62/63 and diagnostic17404694/17404732 are unchanged.
After verifying pending state, canceled only old positive full-GC train/history
17049163_1–4/17049164_1–4 and obsolete shared controllers17049165,17406864,
17406897,17406898,17406899. No high-rate weights initialize replacements.

**Evaluation:** new mixed final17422018→17422019→17422020 compares raw2500
and SWAs at total1000/1500/2000/2500 on four common2022 starts/40leads, reusing
M1 raw/SWA. External annual controller17422049 validates all ten endpoints,
preserves GCM initial500/main2000 proof evidence, then dispatches development32
filter fitting and annual122 raw/Gaussian/band evaluations and paired reports.
Both start after17049163_0:17421839:17406862 and never invoke canceled finish
hooks. The active family comparison retains different documented training
recipes; within full GC all five arms now use the same low-rate schedule.

**Validation and resources:**18 training/scalar tests and34 focused downstream
tests passed, plus frozen-source/config validation and eight Slurm test-only
requests. All649 parent training files are identical; two new orchestration
files were added. Prior full-GC1250 jobs took19:57:40–20:07:45 and224.534–228.220GiB
RSS, supporting1×80GB GPU/8CPU/384GiB/30h per part. No numerical-descent gate was
introduced. Final queue inspection confirms all eight new job groups pending,
correct dependencies, protected jobs intact and canceled jobs absent.

**Provenance and next action:** [protocol, source hashes and complete job table](../../../docs/experiments/v24_full_gc_positive_lambda_lr3em7_restart_2026-11-12.md),
[active five-arm full-GC selection](../../../artifacts/plans/v24_full_gc_lr3em7_restart_20261112/active_full_gc_selection.json),
and [activation receipt](../../../artifacts/plans/v24_full_gc_lr3em7_restart_20261112/activation/complete.json).
Inspect sealed training/evaluation completion records before claiming new skill.
Finite deterioration and repeat differences remain recorded warnings; let valid
bounded runs reach their planned endpoints and evaluations.

<a id="e014"></a>
## e014 — Five fresh full-GC lambda arms at constant3e-7

Date:2026-11-12 (America/New_York). Execution: submitted; no completed
constant-rate training/evaluation result. Outcome: pending.

**Decision:** after the GraphCast final autoregressive-stage schedule discussion,
the user requested keeping3e-7 only. The full-GC schedule is therefore constant
from the first update, with no warmup and no decay. All five lambdas0/.25/.5/.75/1
start fresh from public GC-small with fresh optimizer state. The earlierlambda0
run's1250-update weights used warmup/cosine and cannot represent this new
schedule. Retain those scores/checkpoints only with their actual historical
recipe. The original user-directed exclusion of full-GC1e-5 lineages remains.

**Recipe:** preserve seed22, batch4, BPTT24/K20,96-step segments, full weather
gradients, exact-grid-MSE-plus-stable-AMSE-excess objective, and the frozen
shuffled sample order. Train2500 updates in fresh0→1250 and exact-resume
1250→2500 allocations. Native config uses `learning_rate_schedule=constant`,
`warmup_steps=0`, `learning_rate=3e-7`, `end_learning_rate=null`; the null value
is the preserved validator's required representation, and effective LR stays
3e-7 at every update. No profile or calibration run was added.

**Activation:** fresh five-arm training arrays **17429091→17429092**, history
**17429093**, and completion **17429094** are submitted. After verifying pending
state, canceled e01317421838/39/40/41, mixed17422018/19/20, annual17422049,
and oldlambda0 train/history17049163_0/17049164_0 before execution. No prior
weights, logs, manifests or frozen source were deleted. GC–Mamba training/history
**17406861/17406862/17406863** is unchanged. The saved-checkpoint diagnostic
17404694_2–5/17404732 completed and remains available under its actual recipes.

**Evaluation:** rebuilt mixed raw/SWA and annual pipelines depend only on
**17429092:17406862**, combining all five fresh constant-rate full-GC arms
with existing M1-recipe GC–Mamba. Mixed raw2500/SWA uses four2022 starts/40leads,
reuses principal M1 raw/SWA forecasts, and averages the prescribed total
updates1000/1500/2000/2500. Annual smoothing fits on32 development2022 starts
and evaluates122 annual2023 starts with raw/Gaussian/band results and paired
reporting. These remain comparisons of different family training recipes.
Submitted mixed jobs are **17429215→17429216_2–21→17429217**; annual/smoothing
controller is **17429218**, with the same two-parent dependency. No unassigned
future jobs or completed evaluations are claimed here.
Final queue verification found all eight new job groups **PENDING** with
correct dependencies, all canceled jobs absent and protected GC–Mamba arrays
present. The [sealed activation receipt](../../../artifacts/plans/v24_full_gc_constant_lr3em7_20261112/activation/complete.json)
and active selection in its plan root preserve that inspected state.

**Validation and resources:** nine focused training tests passed, including
the actual Optax constant LR and serialization/resume at1250; frozen runtime
validated all five configurations and LR atindices0/99/100/1249/1250/2499.
All eight training/downstream Slurm test-only calls passed before submission.
All649 original computational files are unchanged, with two orchestration
additions. Comparable1250 jobs completed in19:57:40–20:07:45 at224.534–228.220GiB
RSS, supporting the retained1×80GB GPU/8CPU/384GiB/30h request per allocation.
Finite loss increases and drift remain recorded warnings, not stopping gates.

**Portable handoff:** the corrected package
`artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7/` and read-only
[portable instructions](../../../docs/experiments/README_LIANGHONG_PORTABLE_RUNS.md)
apply constant3e-7/no-warmup to Lianghong's four fresh full-GC horizons atlambda0.
Validation passed34 CPU tests and four real full-GC configurations;672 original
computational source files remain identical. GC–Mamba templates are unchanged.
Old handoff archives stay immutable; no remote execution or delivery is claimed.

**Provenance:** [current protocol](../../../docs/experiments/v24_full_gc_constant_lr3em7_2026-11-12.md),
[training manifest](../../../artifacts/checkpoints/v24_Ilya/res1_full_gc_constant_lr3em7_20261112/manifest.json),
manifest hash `9b4ac57585cfe9d738b52103bc4fa8095c687d889a711525d4073e46da0b4ef1`,
training source `e8e07f2a7c3cd549e37a6cdc88c967a51fdb549960a2bc0f1df78afc30a70c88`.
New evaluation roots:
`artifacts/evaluations/v24_constant_lr_mixed_final_comparison_20261112/` and
`artifacts/evaluations/v24_constant_lr_gc_lambda_annual_20261112/`.
Inspect sealed training and evaluation completion records before reporting
skill. Existing cosine-lambda0 results do not establish this schedule's outcome.

<a id="e015"></a>
## e015 — GraphCast-style optimal filtering for the v4 GC/M1 comparison

**Date:** 2026-11-12 (America/New_York); updated November 13.
**Execution:** 2022 fit complete; 92/122 annual starts evaluated.
**Outcome:** interim MSE/power tradeoff; full-year result pending.

**Question:** does the principal M1 advantage persist after public GC and M1
each receive the degree-wise spectral filter from GraphCast Supplement §7.4.2?
The user requested reworking smoothing and evaluating on v4's 2023 set, then
explicitly required fitting on 2022.

**Protocol:** fit unrestricted real gains independently by predictor, lead,
physical variable/pressure level, and degree 0–179 on the existing 32 development
starts. Evaluate 122 starts at 00 UTC every third day in 2023, with 40 six-hour
leads on the native 1°/13-level grid. Use public GC-small and the exact principal
M1 SWA from v4; no other model enters. There is no lead/wind tying, Gaussian or
four-band restriction, regularization, gain clamp, denominator floor, residual
restoration, or discrete-mean correction. Zero forecast-power modes receive
identity with diagnostics. Raw rollout feedback is unchanged. Raw and pure
projection outputs accompany the optimal filter to expose truncation effects.

**Scoring:** exact original GC grid loss remains primary, using the copied
canonical v4 GC metrics as common denominator. Preserve raw GC/M1 rerun drift,
separately compare filtered M1 with filtered GC, and report physical spectral
MSE/RMSE, explicitly named weighted spectral loss, projection residuals,
mean/positivity diagnostics, and paired intervals. The 2023 year has prior
analysis exposure. Finite drift, unusual gains, or worse loss remain warnings.

**Implementation:** new core, observer, campaign and reporter preserve the
completed smoothing run's model/evaluator computation. The old dispatcher only
gains optional observer injection. The actual v4 recovery evaluator differs
in hash, so original reference metrics and rerun differences remain explicit.
Old 2022 statistics are pooled too coarsely to refit; old 2023 summaries lack
coefficients/fields needed for arbitrary filtered grid loss. Fresh forecasts
are therefore necessary, but no training is required.

**Execution:** **17430308_0–3 → 17430309 → 17430310_0–7 → 17430311**,
all pending at post-submission verification. Four calibration tasks each have
eight starts; eight annual tasks each have 15–16 starts. Each GPU task requests
one 80 GB GPU, four CPUs, 20 GiB host RAM, and 62/90 minutes for calibration/
evaluation. Prior 30–47 minute runs at 9–10.3 GiB support these resources.

**Validation:** 17 focused tests, real PNG/PDF rendering, four Slurm test-only
requests, and a real-field 181×360 T179 geometry check passed. The latter was
finite with QR condition 2.332 and coefficient roundtrip relative error
4.1e-9–2.7e-8. A local Zarr async probe stalled; direct chunk decoding succeeded,
and the production loader remains unchanged.

**Evidence:** [protocol](../../../docs/experiments/v24_optimal_smoothing_v4_2026-11-12.md),
[manifest](../../../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/manifest.json),
[validation](../../../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/checks/validation.json).
Manifest `b69ebc7e623b2a2bab0645a6d020f8807645f70b5eadc76dced61b3e116449af`;
source `a470b86a6936c744308e1bfabbd08fe3ceabd5411019d76f7830947ce6033fa6`.
Next: inspect actual completion and `report.json` before interpreting improvement.
The report job automatically produces the tables and figures.

### November 13 interim evaluation

All 32 development starts and the fit completed; annual shards 0–5 completed
92 starts from January 1 through October 1 at 40 leads. Shards 6/7 (October 4
through December 30, 30 starts) remain pending on priority; report17430311 waits
on them. Thus execution is still partial, with substantial fourth-quarter
coverage missing. The interim report does not replace annual completion files.

Canonical v4 GC-denominated exact loss reductions (rollout / day10) are:
filtered GC **10.9123% / 18.6650%**, raw M1 **14.7112% / 22.4317%**, filtered
M1 **15.3533% / 25.4324%**. Direct filtered M1 versus filtered GC is **4.9850%**
rollout reduction, paired95% interval **4.1243–5.9559%**, and **8.3204%** at
day10, interval **6.9930–10.0064%**. These intervals resample five-start blocks
on this completed subset, conditional on fixed checkpoints and 2022 filters.
Raw M1 versus filtered GC retains4.2643% rollout reduction. Optimal filtering
therefore substantially narrows the gap on this subset while retaining an M1
advantage; annual conclusions remain pending.

Filtering lowers each model's own raw rollout loss by10.8808% (GC) and0.7528%
(M1). Aggregate raw rerun differences versus canonical v4 are−0.0353% and
+0.0382%; both filtering decreases exceed those measured aggregate differences.
Per-date/lead maximum relative differences are3.9907% and1.7923%, retained as
warnings. The observed repeat difference is not a complete noise distribution.
Observer/evaluator FP64/FP32 differences remain below their reference bounds.

All1,195,200 fitted gains are finite, with18,014 negative and93,153 greater
than1, none above10 in magnitude. Unit-gain projection alone worsens grid loss;
the fitted gains produce the improvement. Preserve this as unrestricted optimal
filtering, not pure attenuation. Completed calibration took22–23min/shard;
evaluation took37–39min/shard, within requested resources.

Evidence: [interim readout](../../../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/interim_20261113/REPORT.md),
[sealed result](../../../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/interim_20261113/report.json),
and paired losses/CSV in the same directory. Frozen source and input checks
passed; all92 statistic files were hash-checked. No jobs or scientific settings
were changed. Next: inspect the remaining shards and automatic annual report.

### November 13 power-spectrum comparison

The user requested power spectra alongside MSE. Saved physical statistics on
the same 92 dates give rollout power errors 59.4141% / 78.3995% for raw /
filtered GC and 67.5267% / 77.6371% for raw / filtered M1. Day-10 values are
65.6292% / 92.1136% and 84.1939% / 90.5157%, respectively. The diagnostic is
`100 × mean |P_forecast/P_truth − 1|`, pooling dates before taking ratios and
equally weighting 83 channels, degrees 1–179, and leads. It uses physical
fields without climatology subtraction, excludes degree zero, and is not
total-energy error or phase accuracy. No undefined bins or denominator floors.

Both filters worsen aggregate power error at every lead while improving MSE.
Raw M1 beats filtered GC on both aggregate metrics at every lead. Versus raw
GC, however, raw M1 improves MSE but worsens power matching. Filtered M1 has
slightly lower power error than filtered GC, without a spectral confidence
interval. These results do not establish universal fine-scale power recovery
or improvement at every variable/degree. This qualifies the MSE-only reading.

Evidence: [joint report](../../../plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113/REPORT.md),
sealed `summary.json`, pooled arrays and degree/lead CSVs in the same directory;
three PNG/PDF pairs show degree ratios at days 1/3/7/10, power errors by lead,
and MSE/power tradeoffs. The reporter verifies input hashes/dates and reproduces
the preceding exact MSE values. No forecasts rerun or filters refit on 2023.

Full-year follow-up **17470053 → afterok:17430311** was submitted and verified
pending on dependency. It requests 8 GiB / two CPUs / 20 minutes, with successful
Slurm test-only validation, and requires all 122 starts. Campaign subdirectory
`power_reporting_20261113/` pins the submission and separate source digest
`e3226c6153d824c1fa583d02689dee85e50b11db1673ec38d189c025620ab9dc`.
It will write `v24_optimal_smoothing_v4_2023_power_20261113` data/images and a
sealed completion record. The original fit/evaluation snapshot is unchanged.

<a id="training-steps-k-20261113"></a>
## November 13 analysis of saved training-step and K experiments

Event date: historical September–November 2026 experiments; synthesis recorded
2026-11-13 (America/New_York). This is completed analysis of existing artifacts,
not a new training attempt. The user requested an experiment summary and useful
plots after asking whether MSE and power dependence on training steps and K
could be inferred.

The [report](../../../plots/analyze_models/data/resolution_eval/v24_training_steps_and_k_20261113/REPORT.md)
contains four PNG/PDF figures, source-hashed CSVs and provenance. The three
reproduction scripts are `plot_v24_training_k_summary_20261113.py`,
`plot_v24_m1_steps_summary_20261113.py` and
`plot_v24_mgda_steps_summary_20261113.py` under `scripts/analyze_models/`.
No model execution, scheduler changes or remote execution occurred.

- The completed GC–Mamba K sweep and historical matching K20 recipe provide
  18 exact-loss records: six AR-tail settings at independent first-stage500,
  raw total2500 and SWA, all on the same32 2022 starts/40leads. SWA rollout
  reductions are−8.871%,−7.706%,+0.087%,+3.335%,+5.574%,+16.343% for
  teacher forcing/tail0,K4,K8,K12,K16,K20. Raw K16 is below raw K12 on rollout
  reduction, so the raw trend is not strictly monotonic. Fixed BPTT24 means
  changing K also changes the teacher prefix; historical K1 means tail0, with
  memory retained. K20 is a separately executed historical reference, not a
  new sweep arm or a seed replicate. No matched K-sweep spectra were found.
- On four identical2022 dates, principal M1 raw total1000/1250/2500 has exact
  rollout MSE9.528498/9.325560/9.424010 and spherical power mismatch
  70.497%/70.072%/75.943%. SWA gives9.171727 and71.886%; it is an average at
  the same budget, not a later update. The1250→2500 MSE increase is1.0557%
  and mismatch rises5.8708pp. The earlier1000→1250 MSE decrease is2.1298%.
  The three raw points do not identify an optimum. Actual day10 T2m/Z500/U850
  spectra and all40-lead score curves accompany the checkpoint summary.
- The older relative-power MGDA and matched MSE continuation have raw
  checkpoints0/500/1000/1500/2000 at fixedK20. On the same eight2023 spectral
  dates, final MGDA changes exact MSE by+1.1437% and zonal mismatch by−14.7502%
  versus the mature parent; MSE control changes them by+3.2256%/−0.9935%.
  This supports a spectral/accuracy tradeoff, not preservation of parent MSE.
  The separate32-date exact evaluations are not used in this joint plot.

All curves describe seed22 only. M1's83-channel spherical mismatch and MGDA's
nine-field zonal metric use different scales, fields and leads; never pool
their scores or compare the four-date and32-date M1 losses directly. Training
K is distinct from forecast lead. The new constant3e-7 full-GC artifacts contain
no completed local result at inspection; excluded peak1e-5 results and their
continuations are absent from every generated comparison.

Numerical diagnostics remain advisory. Across18 K evaluations, baseline
rollout range is0.045172 (0.41745% of mean), exceeding K8 SWA's0.08716% gain.
M1's observed public-GC difference is0.079385; both the0.202937 decrease and
subsequent0.098450 increase exceed it in magnitude, but it is not complete model
noise. MGDA baseline range is0.088768; maximum parent-baseline deviation is
0.066007/0.5906%. No spectral uncertainty interval was added. Finite drift and
unfavorable results remain warnings, never release or training-stop gates.

Validation reproduced saved exact-loss values/ratios and M1 spherical scores,
checked common dates/leads and completion, pinned sources and scripts, and
visually inspected all four figures. Next useful evidence would be matched
spectral evaluation of retained K-sweep checkpoints; this requires inference
but no new historical GC–Mamba training. It was not submitted by this analysis.

<a id="literature-spectral-20261113"></a>
## November 13 — Literature-style spectral diagnostics and presentation plan

Execution: completed offline analysis and figure preparation. Outcome: supports
an error/amplitude tradeoff for the fixed original M1 checkpoint on the existing
92-date subset; no additional training-seed or annual-coverage claim.

The user requested original AMSE graphs, clarification of their MSE results,
new plots using established literature axes, and a presentation plan. Original
AMSE arXiv:2501.19374v2 was downloaded unchanged, with original figures and source
provenance under `artifacts/literature/amse_2501.19374v2/`. Its ensemble-mean RMSE
is not the deterministic native-grid GraphCast loss used here; do not infer a
comparable scalar MSE penalty or preservation from the published plots.

New script `scripts/analyze_models/plot_v24_literature_spectral_comparison_20261113.py`
reads only the pinned interim optimal-filter pooled statistics and fitted filter
metadata. Seven PNG/PDF pairs are in
`plots/analyze_models/images/resolution_eval/v24_literature_spectral_comparison_20261113/`:
five-day and full 6-hour/5-day/10-day amplitude/coherence panels, represented
physical spectral RMSE by lead, physical spectra at the three selected leads,
and the separate exact native-grid GraphCast loss comparison. Numeric curves,
channel weights, hashes, definitions and checks are in the matching data folder.

All plots use the same 92 starts from January 1 through October 1, 2023; filters
were fitted on 32 development starts in 2022. The sample remains incomplete
relative to the planned 122. AMSE-style amplitude and coherence are computed
per channel after pooling dates, then averaged with normalized variable/pressure
weights. Field-standard-deviation normalization is not applied to dimensionless
ratios. This differs from the earlier equal-channel aggregate power-error score.

Raw M1 versus filtered GC has 4.2643% lower exact rollout loss, 2.2379% lower
day-5 loss and 4.6311% lower day-10 loss. Filtering both models gives differences
4.9850%, 1.6333% and 8.3204%. At day 5 and degree 40, weighted amplitude ratios
are 0.7937/0.6545 for raw GC/M1, while coherence is 0.5360/0.5479. This illustrates
stronger amplitude damping with modestly better pattern agreement; no uniform
spectral-fidelity improvement or spectral significance is asserted.

The exact-loss curves retain early-lead native-grid degradation from filtering.
The fitted objective and reconstruction operate on represented T179 coefficients,
so spectral RMSE is not native-grid RMSE. The native-loss plot uses paired newly
evaluated raw GC (14.6811% M1 rollout reduction), not the archived canonical v4
baseline (14.7112% on the same subset); direct M1/filtered-GC contrasts agree.
No seed or date-confidence bands are invented for the spectral plots.

Validation reproduced saved exact and spectral losses, verified source hashes,
channel weights and common dates, and checked the spectral error/power/cross
identity (maximum relative residual 1.82e-15). Figures were visually inspected
and independently reviewed; short-figure footer spacing was corrected. No jobs,
forecast runs, filter refitting, manuscript edits or publication occurred.

The [presentation plan](../../../docs/reports/v24_spectral_presentation_plan_2026-11-13.txt)
prioritizes accuracy before/after filtering and five-day amplitude/coherence,
with all three spectral leads and variable-specific spectra retained in the
appendix. Replace the clearly labeled interim plots when the already planned
complete annual report becomes available. Original AMSE figures remain reference
material and are not overlaid as numerically comparable model results.

<a id="matched-reset-spectra-20261113"></a>
## November 13 — Add matched reset/reset to every literature-style figure

Execution: existing-loss addition completed; raw spectral evaluation submitted
as array **17494525**, with report/plot refresh **17494526** after success.
The first queue inspection found all eight shards pending for priority and
the report waiting on its dependency.
Outcome: spectral comparison pending. The user requested the reset/reset model
in all seven figures to test whether persistent memory offers a better
accuracy/amplitude/coherence tradeoff. That advantage is a hypothesis, not an
assumed result.

Use the original matched `R_matched` SWA, SHA256
`db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`, from
`artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/runs/branch/swa/swa_step00500-02000.pkl`.
The historical unmatched reset and inference-reset M1 are different controls.
The original matched evaluation has 122 dates of exact metrics but no spectral
moments or physical field exports. Searches found no other saved spectra for
this checkpoint.

The loss figure now adds its archived results restricted to the identical
92-date plotting cohort: rollout loss 10.36585848, day-5 loss 8.16351652 and
day-10 loss 25.50644833. Existing rerun M1 losses are respectively 5.7670%,
2.7381% and 9.2730% lower. These comparisons mix the archived reset evaluation
with the already plotted M1 rerun and remain descriptive. On spectral rerun
completion, use its exact loss and preserve the archived/new differences.

All seven plotting functions and numeric exports now accept raw reset spectra,
shown in purple. Raw reset is explicitly labelled as a reference in both
columns of the full amplitude/coherence figure; no optimal-reset curve or
filter is fabricated. The six spectral figures remain unchanged until the
new measurements exist, with pending status recorded beside the figures.

The new `v24_matched_reset_spectra_20261113` campaign freezes computational
source from the original optimal-filter campaign and adds only raw observation,
orchestration and plotting. It explicitly resets SSM and convolution state
each prediction, retains cold/full feedback, and collects the same T179,
83-channel raw moments on exactly 92 dates and forty leads. No training or
filter fitting is required. Eight eleven/twelve-date shards request one
A100-80GB PCIe, four CPUs, 16 GiB and 62 minutes each, based on completed
17430308/17430310 resource measurements. A completion report will regenerate
all seven PNG/PDF pairs in the requested folder.

Frozen source digest is
`b20c455c03e8c7e8f010fae274d74292f3c58dce221e51d3c9e0488e8dd34b19`;
campaign hash is
`cacad87f9b89b4c45a6f6e3b7f42dedc004c4d13853f92ededaf98d1a752057b`.
Frozen-source checks and targeted CPU validation passed (known native weighted
losses 1.1/0.275, spectral closure 9.76e-17, unchanged inputs, all eight reset
configurations and finite drift retained). See campaign
`checks/cpu_validation.json`. The workspace implementation is uncommitted;
the immutable snapshot, not a repository revision alone, identifies this run.

The protocol and evidence are in
[the dated evaluation specification](../../../docs/experiments/v24_matched_reset_spectra_2026-11-13.md).
Finite forecast rerun differences remain measured warnings, never acceptance
gates. Checkpoint/source mismatches, corrupt inputs and nonfinite computation
remain errors. No spectral tradeoff or uncertainty conclusion is available yet.

Later November 13, the user requested faster results through GPU test and
explicitly specified a **59-minute** walltime. Submitted preview job
**17495398**, confirmed in `gputest` with QoS `gpu-test` and limit `59:00`.
The initial queue state was pending. It uses four predeclared quarter-start
dates available in the existing cohort: January 1, April 1, July 3 and October 1.
The GC/M1 moments are repooled from their saved per-date outputs on exactly
these dates; only the matched reset forecasts are new. The existing 92-date
array/report remain unchanged.

Preview campaign: `artifacts/evaluations/v24_matched_reset_preview4_20261113/`.
One allocation collects the reset moments and regenerates all seven figures
under `plots/analyze_models/images/resolution_eval/v24_literature_spectral_preview4_20261113/`.
Explicit `--preview` and a distinct output name prevent overwriting the 92-date
figures. Every preview figure is labelled preliminary. Expected execution is
about 15–20 minutes after starting; queue delay is unknown. The allocation is
59 minutes as requested, not the earlier proposed 30-minute budget.

Frozen source digest:
`91e7137e01519eb1af0e50143493615654afe6d5c42e2ccd8ff7d769c94ff836`.
Campaign hash:
`ee2b37c9aa1880d8882ddae4b6f5d37cfe802a66c0f36e557f362de4f188d529`.
Preparation verified the exact four-date ordering, original per-date source and
filter pins, matching channels/leads and frozen computational code. The raw
observer/evaluator are reused unchanged; no redundant qualification forecast
was launched. Spectral results remain pending.

**Completion update:** preview17495398 completed in14m35s. All seven PDFs and
seven PNGs were generated with raw reset/reset in purple, and all fourteen
recorded hashes plus the plot-refresh/report pins were independently verified.
The amplitude/coherence and power/loss layouts were visually checked. Full
92-date evaluation17494525 and report17494526 remain pending at this inspection.

Four-date exact losses (M1/reset) are10.00279967/10.16497006 averaged over the
rollout,8.69140780/8.66693135 atday5, and22.74875986/23.79152289 atday10.
Thus M1 reduces rollout/day10 loss1.5954%/4.3829%, while day5 is essentially
tied (M1 loss0.2824% higher). Mean absolute power-ratio errors are respectively
67.4916%/65.9942%,69.3650%/68.8544%,84.2451%/76.1326%: reset has better power
matching on these aggregates. The spectral metric pools dates before forming
ratios and equally averages83channels and positive degrees; it is not the
variable/pressure-weighted amplitude/coherence ordinate.

Atday5/degree40 (~1000km), M1/reset amplitude ratios are0.65681/0.67593 and
coherences0.55012/0.55067. Atday10 their coherences are0.33634/0.31542 with
stronger M1 damping. The preview does not demonstrate a uniformly better
M1 tradeoff or statistical significance; it supports distinguishing lower
error from spectral fidelity and awaits the full matched-date comparison.

Finite reset-rerun drift is retained in the report: pooled loss changes by
-0.02931 (about-0.29%) versus archived reset; maximum individual relative
difference1.40%. The small day5 ordering should not be emphasized. This is a
measured warning rather than a gate; nonfinite/source integrity checks passed.

<a id="feature-spectra-20261113"></a>
### November 13: separate day-5 spectra and coherence for every feature

The user requested separate coherence and power-spectrum figures, then a
feature-level breakdown identifying relatively good and poor power matching.
This offline analysis reuses the completed four-date preview above: 2023
January 1, April 1, July 3 and October 1, fixed original M1/matched-reset SWAs,
raw GC and the same development-fitted optimal GC filter. No new forecast,
training, filter fit or checkpoint selection was performed. Code is
`scripts/analyze_models/plot_v24_spectra_by_feature_20261113.py`; output is under
the existing preview's `by_feature/` directories in images and data.

Every one of the 83 physical channels is shown separately: five surface
channels and thirteen levels for each of six atmospheric variables. Physical
power, AMSE-style amplitude ratio and coherence each have their own atlas.
The familiar-field examples are fixed T2m/Z500/T850/Q700/TP6h/W500. A separate
rank-selected figure shows the exact three lowest and three highest M1
equal-degree mismatch scores, avoiding an undocumented favorable selection.
GC/filtered-GC coherence retains complementary dashes; no filtered M1 appears.

Two diagnostics use pooled moments and degrees 1–179, excluding the global
mean: equal-degree mismatch is `100 mean_l |Pf/Pt - 1|`; ERA5-power-weighted
mismatch is `100 sum_l |Pf-Pt| / sum_l Pt`. Lower is closer to the truth power
spectrum. They are not forecast MSE or exact native-grid GraphCast loss.
Equal-degree weighting emphasizes weak spectral tails; power weighting can
hide their deficits. Physical units are kept separate across variables.

| M1 field, day 5 | Equal-degree mismatch | ERA5-power-weighted mismatch |
| --- | ---: | ---: |
| T2m | 16.495% | 2.313% |
| Z500 | 71.290% | 1.373% |
| T850 | 52.988% | 1.887% |
| Q700 | 73.700% | 16.94% |
| Precipitation, 6-hour accumulation | 83.194% | 71.340% |
| Vertical pressure velocity, 500 hPa | 77.163% | 71.788% |

T2m is a relatively favorable example on both metrics. Z500/T850 preserve
most power at the dominant large scales, while small-scale relative errors
remain substantial. Precipitation and W500 retain only about 28.9% and 28.3%
of positive-degree truth power, respectively. The equal-degree ranking is
best for W1000/T2m/MSLP and worst for Z50/Z200/Z250; Z50's 129.34% score
coexists with only 3.36% power-weighted mismatch, illustrating why neither
ranking should be called overall physical quality by itself.

M1 has lower equal-degree mismatch than raw GC in 3/83 channels and than reset
in 30/83; the corresponding power-weighted counts are 11/83 and 29/83. These
are descriptive four-date counts, not independent statistical tests. No broad
spectral superiority or annual significance is supported. Keep the existing
reset-rerun finite-drift warning and distinguish coherence from power fidelity.

Evidence: [rankings](../../../plots/analyze_models/data/resolution_eval/v24_literature_spectral_preview4_20261113/by_feature/feature_ranking_5d.csv),
[readout](../../../plots/analyze_models/data/resolution_eval/v24_literature_spectral_preview4_20261113/by_feature/REPORT_5d.txt),
and [provenance/definitions](../../../plots/analyze_models/data/resolution_eval/v24_literature_spectral_preview4_20261113/by_feature/summary_5d.json).
An independent read-only calculation confirmed the rankings and selected-field
scores. Next, repeat these diagnostics on the completed matched larger cohort;
retain these preliminary artifacts separately.

<a id="smoothing-manuscript-20261114"></a>
### November 14: manuscript revision and paired smoothing-control interval

The user approved the proposed smoothing revision and requested all new
corrections in red. `Overleaf/v4.tex` now adds a main-results subsection,
two main figures, and an appendix with filter definitions, joint loss/power
scores, feature and lead diagnostics, and the separately labelled four-date
matched-reset preview. Earlier revisions remain blue; original arXiv prose
and assets are preserved. Existing appendix figures are copied under their
new v4 numbers, and the reviewer-response figure reference is updated.

The manuscript claim is narrower than a universal architecture/Pareto claim:
on 92 common dates, raw M1 has lower native GraphCast loss and lower declared
relative power mismatch than development-fitted degree-wise filtered GC.
Raw GC remains better on power matching. The established carry/carry versus
reset/reset training comparison supports a memory benefit in forecast loss;
the four-date spectral preview does not establish the same spectrum or an
accuracy gain at matched spectral fidelity. No new evaluation or training
was submitted by this revision.

The raw-M1-versus-filtered-GC contrast now has an uncertainty calculation:
rollout reduction **4.264298% [3.3738,5.3438]%** and day-10 reduction
**4.631064% [3.1632,6.5746]%** (pointwise paired 95% intervals). The computation
uses saved exact native-grid losses on the 92 common dates, the frozen
reporter's `bootstrap_endpoints`, and the same 10,000 non-wrapping five-start
draws and seed as the preceding report. Slot substitution changes only the
compared pair. Reproduction of the published both-filtered interval agrees
within 1e-10 percentage points. Filters and checkpoints are fixed; uncertainty
from training seeds, filter fitting and different years is not estimated.

Evidence: [contrast values/provenance](../../../artifacts/evaluations/v24_smoothing_revision_20261114/contrast_ci.json),
[reproduction](../../../artifacts/evaluations/v24_smoothing_revision_20261114/compute_contrast_ci.py),
[figure manifest](../../../plots/analyze_models/data/resolution_eval/v4_smoothing_paper_20261114/manifest.json),
and [paper archive](../../../Overleaf/archive/Graphs/v4_smoothing_20261114/).
The original `v4.tex` and PDF are retained under
`artifacts/evaluations/v24_smoothing_revision_20261114/before/`.

Six new PDF/PNG pairs use the original validated pooled moments. Day-5 feature
annotations compare M1 and original GC using identical equal-degree power
mismatch. The numerical archive covers all 83 channels and retains
power-weighted mismatch and coherence separately. Source pins, exact-loss
reproduction, spectral identities and plot readability were independently
checked. The manuscript build and synchronization are recorded after final
verification; this entry does not imply completed annual/reset spectra.

**Completion:** the checked ICLR PDF has 12 main-text pages and 33 pages in
total. New text, equations, table, bibliography entry and figure frames/captions
are red; prior blue changes remain intact. References resolve, all six copied
figure PDFs match their validated source hashes, and visual review passed.
The pre-existing 0.92pt overfull box in the unchanged common-grid appendix
remains the only overfull-box warning. The compiled `Overleaf/v4.pdf`, source,
figure/data archive and corrected reviewer-response figure reference were
synchronized to `Reminguch/Non-Markovian-Weather`, branch `main`, commit
`02630e592dca82e761ece71c070f43ec6d5ed88b`, preserving repository history.
The parent Weather_global repository was not committed or pushed.

<a id="appendix-e-four-models-20261114"></a>
### November 14: require reset/reset in every Appendix E comparison

**Table presentation update, November 14:** following the user's request,
Table 7 expresses both rollout loss and power mismatch as percentage reductions
relative to raw GraphCast: `100 * (1 - model_score / raw_GraphCast_score)`.
Raw GC is 0/0; filtered GC is 10.36/-31.66%; carry/carry is 15.60/-13.20%;
reset/reset is 14.23/-10.69%. Positive values indicate improvement and negative
values deterioration. The percentages normalize distinct diagnostics; their
magnitudes cannot be combined as interchangeable costs and benefits. Underlying
scores, cohort, spectra and experiments remain unchanged. The new
`table7_relative_to_graphcast.{csv,json}` exports in the four-model archive
preserve original metrics, converted values, formula and input hash. Validation
and publication are recorded in
`artifacts/evaluations/v4_table7_relative_percent_20261114/validation.json`.
Compiled and visually checked the 35-page PDF; Table 7 remains on page 28.
Published to the linked Overleaf repository at
`996aca283998c295781d30789370f3caa20facf1`.

The user first requested replacing filtered M1 in Table 7, then clarified that
reset/reset must be included systematically in all Appendix E experiments.
The required comparator set is now raw GraphCast, development-filtered
GraphCast, original raw carry/carry (M1), and matched raw reset/reset.
Filtered M1 is omitted from active appendix tables, figures and interpretation.

A fresh filesystem audit found no completed statistics or report in the larger
matched-reset spectral campaign. GC/M1 filtering remains at 92 completed dates,
but only four dates have complete common spectral outputs for all four models:
2023-01-01, 2023-04-01, 2023-07-03 and 2023-10-01 at 00 UTC. The truth powers
match exactly, all moments are finite, and the reset checkpoint is the canonical
matched SWA `db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`.
No scheduler changes, inference or training were launched.

Every active Appendix E table and figure is therefore restricted to that same
four-date cohort. Figures 11–15 are regenerated with all four curves; Figures
16–18 already satisfy the requirement and their PDFs are reused unchanged.
The main-text appendix summary is updated consequentially. The principal
122-date memory-policy experiment, five-seed results and their uncertainty
remain unchanged. Existing 92-date filtering outputs and confidence intervals
are archived; they are not presented as uncertainty for the four-date subset.

| Four-date predictor | Exact rollout loss | Relative power mismatch (%) |
| --- | ---: | ---: |
| GraphCast, raw | 11.851948 | 59.6218 |
| GraphCast, filtered | 10.623956 | 78.4991 |
| Carry/carry (M1), raw | 10.002800 | 67.4916 |
| Reset/reset, raw | 10.164970 | 65.9942 |

Carry/carry has 5.84675% lower rollout loss than filtered GC and 1.59538% lower
than reset/reset on this cohort; the corresponding day-10 contrasts are
8.44402% and 4.38292%. At day 5 it has 0.28241% higher loss than reset/reset.
These are descriptive fixed-checkpoint point estimates. Carry/carry's power
mismatch is lower than filtered GC's but higher than raw GC's and reset/reset's;
no identical-spectrum or general memory-related Pareto advantage is asserted.
The actual raw memory-policy powers remain unchanged.

Independent calculations from the saved NPZ moments reproduced the table and
contrasts. GC unit-gain projection loss is 11.957431 and its filtering gain is
10.36110%. Finite rerun differences remain provenance, not exclusion gates:
against the same-date frozen references, raw GC differs by -0.09708% and M1
by +0.06642%; reset pooled drift is about -0.29%, with maximum individual
relative difference 1.40%. These differ from the archived 92-date diagnostics
because the date cohort has changed.

New reproduction script:
`scripts/analyze_models/plot_v4_appendix_e_four_models_20261114.py`.
Numerical exports/manifest are under
`plots/analyze_models/data/resolution_eval/v4_appendix_e_four_models_20261114/`;
the portable archive is `Overleaf/archive/Graphs/v4_appendix_e_four_models_20261114/`.
Original manuscript/source files are preserved in
`artifacts/evaluations/v4_appendix_e_four_models_20261114/before/`.
Future extensions must include all four required predictors on an identical
completed cohort. Build, figure checks and publication are recorded in the
adjacent `validation.json`.

Completed: all 24 source/output hash pins and 16 portable archive/figure pins
verified. The ICLR PDF compiles to 35 pages (10 main-text pages), with no
errors, unresolved references or overfull boxes. Independent rendered-page
inspection of Appendix E pages 26–31, including Table 7, found no clipping or
overlap. All figure width options are preserved. The compiled PDF is
`Overleaf/v4.pdf`. Published the source, five replacement figures, numerical
archive and PDF to the linked Overleaf repository on `main` at
`70651a298f404ac4acbe0ac6b04b2939c22cfc83`; the sync clone is clean and matches
origin. The parent Weather_global repository was not committed or pushed.

<a id="equal-power-memory-20261114"></a>
### November 14: direct test of the identical-power memory interpretation

The user proposed emphasizing that carry/carry has lower MSE than reset/reset
at the same power spectrum. Inspection confirmed that the larger matched-reset
spectral report remains absent; its 92-date scalar-loss entries are archived
losses, not completed spectral measurements. A new offline diagnostic therefore
uses the existing four-date moments without generating forecasts, fitting an
operational filter, or changing the manuscript.

Primary target: `Pstar=sqrt(Pcarry*Preset)`, independently for every channel,
lead and degree after pooling the four dates. Positive gains
`g_i=sqrt(Pstar/P_i)` give identical date-pooled powers. The target uses only
the two forecast spectra, not ERA5 amplitude. The represented matched error is
`Pstar+Ptruth-2*g_i*C_i`, equivalently
`Pstar+Ptruth-2*sqrt(Pstar*Ptruth)*rho_i`. Original GraphCast variable/pressure
weights divided by squared difference-standard deviations aggregate channels.
Primary degrees are 1--179; including degree zero is a secondary check.

| Endpoint | Carry spectral MSE | Reset spectral MSE | Carry higher by |
| --- | ---: | ---: | ---: |
| Ten-day rollout | 9.807773778 | 9.719026607 | 0.9131% |
| Day 5 | 8.416758727 | 8.386612922 | 0.3595% |
| Day 10 | 22.706819015 | 22.570319195 | 0.6048% |

The same ordering holds when matching both spectra to carry power, reset
power or truth power (the last is explicitly an oracle diagnostic), and when
degree zero is included. An independent calculation through the coherence
identity reproduced the primary numbers; raw weighted spectral losses were
verified against the saved observer values. These results do not support
the proposed equal-spectrum advantage on this preview. They do not establish
significance or a conclusion for the full year, and do not invalidate the
separate matched-training forecast-loss result. Raw exact native-grid rollout
loss remains 1.5954% lower for carry/carry on the same four dates.

This is a retrospective coefficient-domain diagnostic. Equality holds for
pooled power, not necessarily each individual forecast; gains are computed
from the evaluation forecasts rather than fitted on development data. No
native-grid MSE after reconstruction is claimed. Finite reset-rerun drift is
retained, and no noise-derived gate is imposed. A first diagnostic arithmetic
check used an inappropriate unit-dependent tolerance on a difference of large
degree-zero geopotential powers; it was replaced by a recorded dimensionless
identity residual. No moments or scientific scores were changed.

Source: `scripts/analyze_models/diagnose_v24_equal_power_memory_20261114.py`.
Evidence: [summary and definitions](../../../artifacts/evaluations/v24_equal_power_memory_preview4_20261114/summary.json),
[readout](../../../artifacts/evaluations/v24_equal_power_memory_preview4_20261114/REPORT.txt),
and [all leads/targets](../../../artifacts/evaluations/v24_equal_power_memory_preview4_20261114/leadwise.csv).
No manuscript assertion or remote repository state was changed. The next
scientific decision requires the larger matched spectral cohort rather than
strengthening the claim from visually similar aggregate curves.

<a id="smoothing-appendix-layout-20261114"></a>
### November 14: appendix presentation of original carry/carry and reset/reset spectra

The user clarified that the manuscript should compare the original forecast
outputs, show power spectra for both memory policies, and move detailed
smoothing discussion to the appendix with a short main-text paragraph.
`Overleaf/v4.tex` now follows that layout. The artificial equal-power diagnostic
above remains research evidence only and is not used in the manuscript plots.

All smoothing figures and methods are in Appendix E. Original Figures 7–10
retain their numbering; Figures 11–15 cover the separate 92-date filtering
cohort. New Figures 16–18 use exactly the four preview dates for every model:
native-grid loss and reduction across forty leads, amplitude and coherence at
days 5 and 10, and per-field physical power spectra at those leads. Physical
spectra show T2m, Z500, Q700 and TP6h with ERA5 reference, preserving original
carry/carry and reset/reset powers. There is no amplitude rescaling. The
1.5954% rollout and 4.3829% day-10 carry loss reductions remain paired with
their actual scale-dependent power differences; day-5 loss is nearly tied.

The new script and data are pinned in the
[figure manifest](../../../plots/analyze_models/data/resolution_eval/v4_memory_spectra_appendix_20261114/manifest.json).
An independent check reproduced all 5,728 physical-power export rows and all
1,432 amplitude/coherence rows directly from saved moments. Mean weighted
coherence over degrees is carry/reset 0.50937/0.50797 at day 5 and
0.38160/0.37505 at day 10; corresponding mean amplitude is 0.65452/0.65943 and
0.58978/0.63942. These descriptive four-date quantities do not establish
spectral superiority or significance. No new forecast or training was run.

The portable paper archive is
[`Overleaf/archive/Graphs/v4_memory_spectra_appendix_20261114/`](../../../Overleaf/archive/Graphs/v4_memory_spectra_appendix_20261114/).
All new prose, captions and frames remain red, preserving earlier blue edits.
Response references were restored to Figures 5/9. Compilation, final visual
review and linked-repository synchronization are recorded in
`artifacts/evaluations/v24_smoothing_revision_20261114/appendix_move_validation.json`.

**Completion:** final build and independent visual review passed: 10 main-text
pages, 35 pages total, Appendix E starts on page 26, and Figures 16–18 appear
on pages 33–35. References resolve and there are no overfull boxes. Compiled
`Overleaf/v4.pdf` and `replies.pdf` are updated. The linked manuscript repository
was pushed successfully on `main` at commit
`d72425e58b0413ffebcd010443a01c058db75533`; the parent repository was not committed
or pushed. Source/PDF hashes and validation details are recorded above.

<a id="matched-reset-spectra-fast-20261114"></a>
### November 14: accelerate the 92-date matched reset spectra

Execution: submitted and released; pending at final inspection. Outcome:
pending. The user requested more than four dates and specified three GPU-test
tasks with 50-minute limits, with the remaining shards on regular GPU for
61 minutes. Arrays `17553195_0–2` (`gputest` / `gpu-test`, 36 dates) and
`17553196_3–7` (`gpu` / `gpu-short`, 56 dates) now cover all 92 fixed starts.
Report `17553202` uses `afterok:17553195:17553196` to refresh seven literature
figure pairs automatically.

Slurm refused an in-place queue change. The replacement arrays/report were
submitted held, original jobs `17494525/17494526` canceled while all were
pending, then replacements released. No outputs were lost. GPU-test allows
three running and 25 submitted jobs per user; no other user GPU-test tasks
were present at preparation. No explicit array throttle was added. Resources
remain one `gpu80&pcie` GPU, four CPUs and 16 GiB per forecast task. Preview
timing (523 seconds for the first date, approximately 84 thereafter) supports
25–30 minutes per eleven/twelve-date shard after starting. At final inspection
GPU-test tasks awaited available nodes, regular GPU tasks awaited priority,
and the report awaited dependencies; queue delay is unknown.

The sealed manifest, frozen source, original matched reset SWA, reset policy,
ordered January 1–October 1 cohort, forty leads and 83 channels are unchanged.
Scientific pins, including reusable same-date GC/M1 moments, were verified.
No training, filter fitting or checkpoint selection is added. Completion of
these forecasts remains the next step; the four-date preview is still the
completed reset spectral evidence, with no new scientific conclusion.

Evidence: [plan](../../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/fast_20261114_plan.json),
[commands and IDs](../../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/fast_20261114_jobs.json),
[activation](../../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/fast_20261114_activation.json),
and [protocol](../../../docs/experiments/v24_matched_reset_spectra_2026-11-13.md).

<a id="feature-spectra-day10-20261114"></a>
### November 14: add day-10 counterparts to every per-feature spectral plot

The user requested day-10 versions of the existing day-5 plots under
`plots/analyze_models/images/resolution_eval/v24_literature_spectral_preview4_20261113/by_feature/`.
The unchanged script `scripts/analyze_models/plot_v24_spectra_by_feature_20261113.py`
supports `--lead-hours 240` and reads index 39 from the saved four-date moments.
All plots retain raw GraphCast, raw M1, development-filtered GraphCast and raw
matched reset/reset; physical power plots also include ERA5. No forecasts,
training, filter fitting, amplitude rescaling or model selection were performed.

Completed 57 new files: 30 PDFs and 27 PNGs with `_10d` suffixes, including
separate power/amplitude/coherence figures for familiar fields, best/worst
fields and each variable group. Each metric also has a seven-page atlas of
all 83 channels. Best/worst fields are reranked at day 10; the familiar fields
are unchanged. M1/raw-GC panel annotations retain the existing absolute
equal-degree power-mismatch percentages. The plot guide explains lead-specific
rankings and how to compare fixed fields between days 5 and 10.

The first rendering process exited 143 after completing amplitude/coherence
and starting power. Power was completed separately through the same module
with its runtime metric selection restricted to `power`; the final summary
consolidates all 57 figures. No plotting source change was required. All 65
source/data/figure hashes were independently verified; a direct Z500 check
matched the 240-hour source power, amplitude and coherence exactly for every
model. Representative six-field and 13-panel figures passed visual inspection.
All 62 existing day-5 artifacts retain their original hashes.

Data, rankings, provenance and checks are in the corresponding data folder:
`plots/analyze_models/data/resolution_eval/v24_literature_spectral_preview4_20261113/by_feature/{feature_ranking_10d.csv,feature_curves_10d.npz,summary_10d.json,REPORT_10d.txt,validation_10d.json}`.
These remain descriptive four-date fixed-checkpoint diagnostics. This task
changes no manuscript or Overleaf files and performs no git publication.

<a id="matched-spectra-36-20261114"></a>
### November 14: recalculate every plotted comparison on 36 newly completed dates

The user clarified that all compared results must use the newly available
data. Completed checked reset shards 00–02 supply 36 starts, every three days
from January 1 through April 16, 2023. The remaining five shards have no
completion checks at this update. The new immutable cohort snapshot excludes
the independent four-date preview; GC, filtered GC and raw M1 moments are
re-pooled on exactly the same 36 dates as raw reset/reset. No inference,
training, filter fitting or scheduler changes were made for this analysis.

Sources and newly pooled reports:
`artifacts/evaluations/v24_matched_spectra_completed_20261114/` contains
`cohort.json`, `gc_m1_reference/`, `reset/`, and per-initialization exact loss
arrays for both reports. The new offline script is
`scripts/analyze_models/pool_v24_completed_matched_spectra_20261114.py`.
All paired truth moments match exactly; all 87 source/output pins were checked.
Reset rerun drift is retained: +0.0629% pooled versus the archived same-date
control, maximum individual relative difference 2.4649%. These are advisory
numerical diagnostics, not exclusion gates.

| Period | Carry/carry exact loss | Reset/reset exact loss | Carry loss reduction vs reset | Carry/reset power mismatch (%) |
| --- | ---: | ---: | ---: | --- |
| Forty-lead rollout | 9.749073 | 10.372953 | 6.0145% | 67.1364 / 65.4186 |
| Day 5 | 7.832126 | 8.061116 | 2.8407% | 68.6626 / 68.1628 |
| Day 10 | 23.499227 | 26.168777 | 10.2013% | 84.4739 / 75.7435 |

Relative to filtered GC, carry/carry has 4.2468% lower rollout loss,
1.9775% lower day-5 loss, and 5.4747% lower day-10 loss. Raw GC/filter rollout
losses are 11.483224/10.181461. Current rollout percentage reductions relative
to raw GC are: filtered GC 11.3362% MSE and -33.6017% power-mismatch reduction;
carry/carry 15.1016% and -14.5861%; reset/reset 9.6686% and -11.6542%. The two
percentage columns normalize distinct diagnostics. The day-5 comparison now
has modestly better MSE at similar aggregate mismatch, but this is not a
matched-power causal experiment; at day 10 spectral mismatch differs more.
This January–April subset is not annual evidence or a training-seed estimate.

The literature plotter now supports an explicit completed subset, uses only
the four intended comparisons in every figure/CSV, and includes separate
day-10 amplitude/coherence outputs. Its multilead figure separates amplitude
and coherence rather than comparing filtered M1. The per-feature plotter
accepts explicit pooled sources and arbitrary identical completed cohorts,
with per-metric rendering batches that merge only matching provenance. The
legacy default four-date loader remains compatible with other scripts.

New outputs are under
`plots/analyze_models/{images,data}/resolution_eval/v24_literature_spectral_matched36_20261114/`.
The score CSV and `REPORT.txt` cover rollout/day-5/day-10 metrics on all four
models. The `by_feature/` subfolder contains both day-5 and day-10 sets for
all 83 channels. Original four-date and 92-date outputs are preserved, with
update pointers. Source snapshots accompany the new data. Final integrity
and visual checks are in `validation.json` in that data directory. This task
does not modify or publish the manuscript.

Final validation passed: 24 aggregate figure files plus 57 per-feature files
at each lead (138 PDF/PNG files total), with six seven-page atlases. Both
feature summaries report all three metrics complete on the identical 36-date
cohort. All active comparison CSVs contain exactly the four requested models;
all 12 model/window score rows independently reproduce the pooled sources.
All 114 historical four-date feature figures retain their original hashes.
Representative aggregate and feature layouts were visually checked, and the
day-10 exported power, amplitude and coherence match the pooled sources
exactly for all 83 channels. The source and data snapshots support later
updates without mixing cohorts.


November 14 presentation addition: `improvement_summary_rollout_day5_day10.pdf`
and `.png` summarize the same 36-date `comparison_scores.csv`. Separate panels
show exact GraphCast MSE reduction and power-mismatch reduction relative to raw
GraphCast for rollout, day 5 and day 10. GC is the explicit zero baseline;
filtered GC, carry/carry and reset/reset have labelled bars. Both metrics use
`100*(1-model_score/raw_GC_score)` independently; negative values denote
worsening. All 12 input rows/formulas were independently verified and final
labels visually inspected. No forecasts, scores or interpretation changed.
Reproducer: `scripts/analyze_models/plot_v24_improvement_summary_20261114.py`.
The figure manifest and source snapshot accompany the existing 36-date data.

<a id="matched-reset-spectra-wave2-20261114"></a>
### November 14: second GPU-test wave for the remaining matched spectra

At the user's request, three more pending shards moved to GPU-test after the
first three completed. Array `17560836_3–5` covers 34 dates at 50 minutes per
task; retained `17553196_6–7` covers the final 22 dates at 61 minutes on regular
GPU, preserving its queue age. This uses three GPU-test tasks, within the
per-user running limit. The existing 36 completed dates and interim figures
remain intact; no additional scientific result is claimed.

Replacements were submitted held before canceling pending originals
`17553196_3–5` and report `17553202`, then released. New report `17560837`
depends on `afterok:17560836:17553196_6:17553196_7` and validates all eight
shards. Source/campaign and scientific reference pins are unchanged. All
replacement jobs were initially pending without holds. Each forecast task
retains one `gpu80&pcie` GPU, four CPUs and 16 GiB RAM; measured first-wave
execution was 24m33s–24m55s, with future queue delay unknown.

Submission evidence is under
`artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/`:
`gputest_wave2_20261114_plan.json`, `gputest_wave2_20261114_jobs.json`, and
`gputest_wave2_20261114_activation.json`. Final `gputest_wave2_20261114_queue.txt`
confirms all five remaining forecasts pending without holds and the report
waiting on the correct dependency. See the
[updated protocol](../../../docs/experiments/v24_matched_reset_spectra_2026-11-13.md).


<a id="appendix-e-matched36-20261114"></a>
### November 14 — Appendix E adopts the fixed 36-date diagnostic

User authorized rewriting Appendix E around the lead-dependent accuracy versus
spectral-fidelity interpretation and explicitly retaining the 36-date data
while the rest of the spectral campaign finishes. All current Appendix E
numbers, Table 7 and Figures 11–18 now use the same 36 January–April dates.
No new forecasts, training, filter fits or uncertainty estimates were run.

The short red main-text paragraph and red appendix distinguish two findings:
carry/carry improves both aggregate scores relative to the tested filtered-GC
baseline (4.25% lower rollout exact GC MSE, power mismatch 67.14% versus
78.28%); relative to reset/reset its lower MSE accompanies increased power
mismatch. At day 5 the loss advantage is 2.84% and mismatch is 68.66% versus
68.16% (0.50 percentage points). At day 10 the loss advantage is 10.20% and
mismatch is 84.47% versus 75.74%. Over the full rollout, the advantage is
6.01% with mismatch 67.14% versus 65.42%. This does not establish equal
spectra or uniformly better tradeoffs than reset/reset. Five days identifies
the K20 feedback-conditioned training tail, not the entire training horizon.

The revised Table 7 and compact Figure 16 show both metrics as separate
percentage reductions relative to raw GC for rollout/day5/day10. All plots
retain raw GC, filtered GC, raw carry/carry and raw reset/reset. No filtered
M1 or inference-only carry/reset is reintroduced. Raw amplitudes remain
unchanged, and principal 122-date and five-seed results remain unchanged.

Sources and products:
- Frozen source pools: `artifacts/evaluations/v24_matched_spectra_completed_20261114/`.
  The pooling script reuses the saved cohort and refuses changed outputs;
  full-campaign reporting has a separate destination.
- Paper script: `scripts/analyze_models/plot_v4_appendix_e_matched36_20261114.py`.
- New plots/data: `plots/analyze_models/{images,data}/resolution_eval/v4_appendix_e_matched36_20261114/`.
- Overleaf archive: `archive/Graphs/v4_appendix_e_matched36_20261114/`, including
  every plotted curve, all 83 channels at 6/120/240h, cohort IDs, source/output
  hashes and code snapshots. Full 40-lead pooled moments remain pinned in
  the parent workspace. The filter JSON is retained losslessly compressed.

All 12 score rows agree with the earlier frozen36 comparison export within
1e-12. Exact native-grid losses are pooled before ratios; no spectral proxy
replaces MSE. Numerical rerun drift is retained as a warning: reset loss is
0.0629% higher pooled, with maximum individual relative difference 2.4649%.
The appendix has no new date/seed confidence bands. Historical four-date and
92-date-only artifacts remain available and are not mixed into these results.

Final verification: 34-page PDF compiles without unresolved references or
overfull boxes; main page 9 and Appendix E pages 26–34 were visually inspected.
Independent review confirmed all table cells, eight figure hashes and the
unchanged manuscript scope/widths. Published source, figures, fixed36 archive
and `Overleaf/v4.pdf` to manuscript `main` as `710c74a`. Parent
Weather_global repository was not committed or pushed. Validation record:
`artifacts/evaluations/v4_appendix_e_matched36_20261114/validation.json`.


<a id="smoothing-section46-20261114"></a>
### November14 — Dedicated Section4.6 on smoothing and spectral fidelity

User rejected the brief smoothing paragraph within Section4.5 and requested a
separate Section4.6 explaining why spectra matter, citing literature, giving
the GraphCast-style filtering procedure, and showing the four-model comparison.
They authorize a main/appendix split now and defer page-space optimization.

Implemented five red main paragraphs: power as scale-dependent variability;
MSE-induced smoothing with citations to Lam2023 and Subich2025; degree-wise
least-squares filtering fitted on32development2022dates and applied output-only;
the four models on the fixed36dates; day5support and day10degradation. The
existing day5amplitude/coherence and three-window percentage plots are now
mainFigures7/8. Detailed filtering equations, score definitions, Table7,
per-field/multilead diagnostics and physical powers stay in AppendixE.

No data or scientific scores changed. Day5carry has2.84%lower MSE versusreset
with0.50pp greater power mismatch. Atday10the relative advantage is10.20%,
but absolute errors and power mismatch grow. Carry's exactMSE grows7.83→23.50.
The equal-degree mean of the channel-weighted coherence curves decreases
fromday5today10 for every model: GC0.50145→0.33215; GCfilter0.50141→0.33150;
carry0.51263→0.37010; reset0.50925→0.35693. These confirm long-lead degradation
without dismissing the ten-day forecast task. RawGCretains closest aggregate
power matching; identical spectra or universal dominance overreset is notclaimed.

Primary-source check: GraphCast arXiv:2212.12794v2 Supplement7.4.2 Eq35 defines
the isotropic least-squares filter; our grid/resolution/scoring remain distinct.
AMSE arXiv:2501.19374v2 Sections2.1–2.2 Eqs2–4 motivate amplitude/coherence and
MSE smoothing; itsFigure3uses6h/5d/10d. No claim about deterministicMSE is based
on its lagged-ensembleFigure4. Existing bibliography entries suffice.

The fixed36archive `Overleaf/archive/Graphs/v4_appendix_e_matched36_20261114/`
remains unchanged. New `v4_smoothing_section46_20261114/figure_mapping.json`
records byte-identical renamedfigurecopies; originalarXivassets remainintact.
All widthsarepreserved. One existingcommon-gridcaptionuses anunbreakablemodel
name to prevent a new overfullline after renumbering9→11; its wording is intact.
The rest of Sections1–4.5 is unchanged. PDFhas36pages, Section4.6startspage10,
mainfigureson11/12; detailedAppendixEstartspage28. Compilation hasno warnings,
references resolve, and reviewedpagesarelegible. Validation/beforefiles:
`artifacts/evaluations/v4_smoothing_section46_20261114/`.

Published Section 4.6, renamed figure copies and checked `Overleaf/v4.pdf`
to manuscript `main` as `15823a8`. Remote and local manuscript heads
match; the parent Weather_global repository was not committed or pushed.


### November 14 — Replace the smoothing summary chart with a table

The user requested converting main Figure 8 to a table and revising the
Section 4.6 opening to motivate power spectra as an additional measure of
quality, because lower MSE can come at the cost of spatial detail. The existing
Appendix E percentage table is moved to main Table 2, replacing the chart;
the duplicate appendix table is removed and all cross-references are updated.
The old chart and all original assets remain available.

Final metric convention after clarification: retain BOTH MSE reduction and
power-mismatch reduction as percentages relative to raw GraphCast. Do not label
these as raw loss values. Zero means baseline performance, not zero error;
negative power-mismatch reduction means worse spectral fidelity. The user
briefly considered absolute ERA5-relative power mismatch, then superseded that
choice by keeping both reduction percentages with explicit labels. All 24
numeric cells remain unchanged on the fixed 36-date cohort.

All subsequent figure numbers are updated through byte-identical copies;
widths and original arXiv files remain unchanged. Archive and mapping:
`Overleaf/archive/Graphs/v4_smoothing_table_20261114/`. Frozen numerical data
remain in `v4_appendix_e_matched36_20261114/`. No forecasts, filters or uncertainty
calculations were rerun, and this edit adds no extremes claims. Validation and
before-files are in `artifacts/evaluations/v4_smoothing_table_20261114/`.

Final verification: all 24 cells and preserved figure identities/widths checked;
the fixed36 archive is unchanged. Table 2 on page 11 was visually reviewed;
the 35-page PDF has no compilation warnings. Published the revised manuscript,
figure copies, table archive and compiled PDF to manuscript `main` as `35e618b`.
The parent Weather_global repository was not committed or pushed.

<a id="matched-reset-spectra-wave3-20261114"></a>
### November 14 — Final two spectral shards moved to GPU-test

Shards 00–05 have completed and passed their saved checks, providing 70/92
dates. Applying the existing request to accelerate work through GPU-test,
the remaining 22 dates move to `17572313_6–7`, each with a 50-minute limit,
one `gpu80&pcie` GPU, four CPUs and 16 GiB RAM. No other user GPU-test jobs
were queued at preparation; two tasks fit the three-running-job limit.
Recent completed shards took 34–36 minutes, within this allocation.

Replacement jobs were submitted held. Pending `17553196_6–7` and old report
`17560837` were canceled and confirmed absent before replacement release.
New report `17572314` depends on `afterok:17572313` and validates all eight
shards, including the six already completed. Initial states are pending
without holds. Campaign/source, checkpoint and scientific-reference pins
are unchanged; fixed 36-date figures and manuscript artifacts are preserved.
No final 92-date comparison or new scientific conclusion is claimed.

Evidence is under the existing campaign's `submissions/` directory:
`gputest_wave3_20261114_plan.json`, `gputest_wave3_20261114_jobs.json`, and
`gputest_wave3_20261114_activation.json`. See the
[protocol](../../../docs/experiments/v24_matched_reset_spectra_2026-11-13.md).
Final `gputest_wave3_20261114_queue.txt` records task 6 configuring on
`della-l02g13`, task 7 pending, and the report waiting on its correct dependency.

<a id="v5-spectra-matched70-20261114"></a>
### November 14: replot v5 on the completed 70-date intersection

The user requested updated v5 graphs and correctly noted that the standard
2023 protocol contains 122 starts (January 1–December 30, every third day).
The 92-date spectra campaign was an interim match to the six completed GC/M1
optimal-filter rerun shards, ending October 1. That rerun's last 30 starts were queued separately;
the reset spectral campaign never included those additional dates. Completing
its existing jobs therefore gives 92, not 122, matched spectra.

When offered annual completion or an update now, the user explicitly chose
the completed subset. We froze checked reset shards 00–05 before pooling:
70 starts from January 1–July 27, all 40 six-hour leads and 83 channels.
Every predictor uses those exact dates. No partial shard, four-date preview,
new forecast, filter fitting or bootstrap draw is included.

New pooled source: `artifacts/evaluations/v24_matched_spectra_v5_replot_20261114/`.
New replotter: `scripts/analyze_models/plot_v5_matched_spectra.py`.
It generates the five active v5 Figures 6 and 10–13, compact numerical exports
and Table 1 values. Both `Overleaf/v5.tex` and standalone `v5_clean.tex` have
synchronized active numbers; historical review deletions and 36-date data
remain preserved. M1/R1 naming, compact legends and provisional status remain.
Removed physical-power figures are not restored. The actual interim cohort
and measured rerun drift remain in internal comments and manifests, following
the author's existing presentation convention.

| Endpoint | M1 loss reduction vs R1 | M1 power mismatch | R1 power mismatch |
| --- | ---: | ---: | ---: |
| Rollout | 5.9335% | 67.5267% | 66.0062% |
| Day 5 | 2.9176% | 69.5045% | 68.9669% |
| Day 10 | 9.6956% | 84.3515% | 76.3935% |

Loss reductions use the exact saved native-grid GraphCast weighted MSE,
pooled before ratios. Power mismatch uses date-pooled physical powers,
equally averaged over the selected leads, 83 channels and degrees 1–179.
M1 has 4.0637% lower rollout loss than filtered GC, with power mismatch
67.5267% versus 78.4145%; raw GC retains the closer power score, 59.3066%.
These are provisional point estimates, without new spectral uncertainty or
full-year conclusions. Reset rerun drift versus the archived same-date loss
is +0.0456% pooled and at most 2.4649% individually, retained as a warning.

Both PDFs compile with no unresolved references or overfull boxes; the clean
version retains ten main-text pages and 29 total pages. The five plots and
Table 1 were visually checked. All 48 exported metric values match an
independent calculation to 2.85e-14 absolute difference; portable replay
matches all curve/layout signatures. Build, data and verification artifacts
are under `artifacts/manuscripts/v5_spectral_matched70_20261114/`.
The portable archive is
[`Overleaf/archive/Graphs/v5_spectral_matched70_20261114/`](../../../Overleaf/archive/Graphs/v5_spectral_matched70_20261114/).

Published the checked manuscript-folder update to
`Reminguch/Non-Markovian-Weather` on `main` at
`e85def324aac6c9e32f463ab6cb14a1531e74bb4`; the remote head was verified.
The parent Weather_global repository was not committed or pushed. The
replotting source and exact data are pinned in the new archive, independently
of the parent workspace's uncommitted changes. The clean and tracked PDFs
remain outside the Overleaf repository, under the build directory above.

<a id="matched-reset-spectra-tail30-20261114"></a>
### November 14 — Audit existing annual spectra and submit the missing reset dates

Execution: running (submitted; initially pending); outcome: pending.
The user explicitly authorized submitting
the 30 missing matched-reset starts as nine short GPU-test jobs. This extends
the existing spectral evaluation to the standard 122-date 2023 cohort; it does
not change training, the checkpoint, the reset policy or the selected dates.

The read-only inventory corrects an overbroad interpretation of the earlier
92-date explanation. Both
`artifacts/evaluations/v24_smoothing_m1_20261107/evaluate/*/statistics/` and
`artifacts/evaluations/v24_smoothing_sweep_m1_20261109/evaluate/*/statistics/`
contain 122 distinct per-date files, January 1–December 30 at a three-day stride,
including all 30 October 4–December 30 dates. These store spherical moments for
83 channels, degrees 0–179 and forty six-hour leads. Inspected metadata confirms
the original M1 SWA SHA256
`b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0`.
Thus raw GC/M1 spectra need not be recomputed merely because the newer
`v24_optimal_smoothing_v4_20261112` rerun currently has 92 completed dates.
Older filtered variants are different treatments and cannot be substituted
for the current optimal-filter comparison without checking compatibility.

No alternative correct-R1 tail spectral moments or saved forecast fields were
found in the searched local evaluation, plotting and archived-results roots,
the empty `../Weather_ilya` directory, or the inspected saved-output roots under
`della:/scratch/gpfs/DABANIN/lm8598/Weather_Global/`. The historical October 18
matched-reset campaign has annual scalar losses, not the spectral moments
needed here. The LH extreme-event forecast-cache manifest identifies seed 22
carry at raw step 2000 (SHA256 `bbfd7f59…`), not original M1 SWA or matched R1,
and only four surface variables. The November 6 annual MGDA campaign stores
11-variable zonal spectra; those cannot substitute for the spherical diagnostic.
These are scoped inventory findings, not proof that no other external cache exists.

New campaign root: `artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/`.
Use exactly the 30 standard initialization IDs from October 4 through
December 30, split into three four-date and six three-date jobs. Request
`gpu-test` with a 50-minute walltime for each of the nine jobs; the per-user
limits are three running and 25 submitted tasks. Keep the original R1 SWA
SHA256 `db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`,
zero initial memory, reset before every step, cold full forecasts, zero warm-up,
alpha one, forty leads and the established 83-channel spherical observer.
The annual report must validate and combine the existing 92 reset dates with
these 30 without duplicate dates or silent cohort changes. Preserve measured
finite rerun drift as a warning, not a rejection gate.

Before the tail submission, the original 92 campaign completed: final tasks
`17572313_6` and `_7` finished in 23m42s and 23m28s, then report `17572314`
completed in 1m09s. All eight shard checks cover 92 dates, and the inspected
`v24_matched_reset_spectra_20261113/report/summary.json` records
`status=complete`, `count=92`. GPU-test is empty at the reported preparation
check; its verified limits remain three running and 25 submitted tasks. Expected
tail runtime is approximately 15–25 minutes per task after starting, with an
unknown queue delay and the requested 50-minute allocation.

Submitted November 14 at 23:09:01 America/New_York (November 15 04:09:01 UTC):
GPU array `17574447_0–8`, followed by CPU report `17574448` with
`afterok:17574447`. The inspected submission records request nine tasks on
`gpu-test`, each with one `gpu80&pcie` GPU, four CPUs, 16 GiB RAM and 50 minutes;
the scheduler routes them to `gputest`. The report requests two CPUs, 12 GiB and
30 minutes and writes `annual122_report/` inside the tail campaign. The initial
queue snapshot shows all nine forecasts pending with reason `None`, and the
report pending on `Dependency`; no task was held. GPU-test's user limit permits
at most three simultaneous tasks and 25 submitted, so all nine can be queued
without exceeding the submission limit.

Saved preparation checks passed: all nine configurations preserve the original
forecast settings, the 30 tail dates plus original 92 form exactly 122 unique
standard starts, the inherited source is verified, and all seven arrays in the
original 92 pool reproduce bit for bit. These checks validate preparation and
pooling, not completed tail forecasts. Frozen source digest:
`7a41a65f49c6b9c9b38da6757a09d0c45e520f2168328da435038bde6b887ba3`;
campaign hash:
`d8fee6b03eab86b34a4ec88553f65365f957b2e432dd2651247e5d494ba8aeaa`.

Evidence: [preparation checks](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/checks/preparation_validation.json),
[forecast submission](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/evaluate.json),
[report submission](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/report.json),
and [initial queue](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/initial_queue.txt).
No tail completion or annual scientific result is claimed. The frozen 70-date
v5 plots and their source archive remain intact, with no manuscript change.
See the [updated protocol](../../../docs/experiments/v24_matched_reset_spectra_2026-11-13.md).

A subsequent verified queue snapshot shows all nine forecasts still pending for
`Priority`, with report `17574448` pending on its dependency. Evidence:
[`submissions/verified_queue.txt`](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/verified_queue.txt).

<a id="v5-spectra-matched122-20261115"></a>
### November 15 — Validate the completed annual spectra and update v5

Evaluation completion: 2026-11-14, 23:56:24 America/New_York.
Recorded and manuscript update authorized: 2026-11-15 (America/New_York).
Evaluation execution: completed.
Scientific outcome: fixed-checkpoint annual spectral comparison; no new seed
replication or spectral uncertainty estimates.

The nine tail tasks `17574447_0–8` and annual CPU report `17574448` all finished
`COMPLETED`, exit code `0:0`. Their nine saved shard checks cover
4+4+4+3+3+3+3+3+3=30 dates. The completed annual report joins these with the
original 92 reset dates, exactly matching the 122 unique standard starts from
January 1 through December 30, 2023, every third day. Each date has all 40
six-hour leads through 240 h, 83 channels and degrees 0–179. The reset model is
the original matched SWA `db0e8032…`, with reset before every step and the
unchanged cold/full forecast protocol.

Evidence: [annual summary](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/annual122_report/summary.json),
[report completion check](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/checks/report.json),
and [pooled spectra](../../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/annual122_report/power_pooled.npz).
The summary records `status=complete`, `scope=annual122`, `count=122`; its
pooled SHA256 is
`501a33216228a94e88b2f9739494fa1a661296d9c851ebdad1ed7137096e7a57`.
The November 15 audit independently verified that digest, the exact ordered
annual cohort, all nine complete checks and finite values in all seven numeric
pooled arrays. The newer optimal-filter campaign
`artifacts/evaluations/v24_optimal_smoothing_v4_20261112/` now also contains
122 unique date statistics. Earlier 92-date/pending statements above are
historical snapshots, superseded by this inspected evidence.

Preserved numerical diagnostics: reset rerun pooled exact loss is
10.359790727055467 versus archived 10.359537610436073, difference
+0.000253116619395 (about +0.00244%). Maximum local relative drift is
0.0329456857334976 (3.29457%), exceeding the 0.001 (0.1%) advisory reference.
The report retains its warning and all measured values; finite drift is not
an acceptance gate. These observed differences do not establish a noise-based
significance threshold. Nonfinite and source-integrity checks still apply.

The automatic annual report deliberately records `plot_refresh=false` and
preserves the original 92 pool. The user subsequently authorized replacing
the active v5 70-date spectral presentation with the completed 122-date
comparison and removing its provisional wording. Use separate replot root
`artifacts/evaluations/v24_matched_spectra_v5_replot_20261115`, paper archive
`Overleaf/archive/Graphs/v5_spectral_matched122_20261115`, and manuscript build
root `artifacts/manuscripts/v5_spectral_matched122_20261115/build`.
All earlier 36/70-date exports remain historical evidence. The original
memory-policy bootstrap intervals and seed results remain a separate analysis;
no intervals from an earlier spectral subset are transferred to the annual
spectral figures. This update launches no new forecasts or training.

The completed [annual replot summary](../../../artifacts/evaluations/v24_matched_spectra_v5_replot_20261115/summary.json)
records `complete_matched_2023` and all four comparators: raw GraphCast, raw M1,
development-filtered GraphCast and raw R1. Figures 6 and 10–13 are regenerated
under `_v5_matched122` filenames; their numerical curves, Table 1 rows, code
and source pins are in the
[portable archive](../../../Overleaf/archive/Graphs/v5_spectral_matched122_20261115/summary.json).
The [independent arithmetic check](../../../artifacts/evaluations/v24_matched_spectra_v5_replot_20261115/independent_numeric_validation.json)
recomputes 48 metrics directly from pinned pooled NPZ files, with maximum
absolute difference `1.4210854715202004e-14`. The plot summary checks matching
dates/truth, spectral closure, finite values and unchanged raw M1/R1 powers
and exact losses. No forecasts, filter fits or bootstrap draws were added.

| Annual comparison | Day 5 | Rollout, 6–240 h | Day 10 |
| --- | ---: | ---: | ---: |
| M1 exact-GC-loss reduction versus R1 (%) | 2.8507 | 5.7560 | 9.2566 |
| M1 minus R1 power mismatch (percentage points) | +0.5030 | +1.5053 | +7.9609 |
| M1 exact-GC-loss reduction versus filtered GC (%) | 2.3561 | 4.0183 | 3.8324 |

At day 5, absolute power mismatch is 69.5984% for M1 and 69.0954% for R1;
at day 10 it is 84.3419% and 76.3811%. M1 lowers both exact loss and power
mismatch relative to filtered GC, but raw GC retains the smallest power
mismatch. The conclusion remains a tradeoff: lower M1 loss does not establish
an advantage at identical spectral fidelity. Exact loss is pooled before
forming reductions; power mismatch is the equal-channel/positive-degree mean
absolute error of date-pooled prediction/truth power ratios. These remain
fixed-checkpoint point estimates without spectral confidence intervals.

Both `Overleaf/v5.tex` and `Overleaf/v5_clean.tex` now reference the five new
annual PDFs, with revised Table 1, cohort descriptions, captions and numerical
discussion. The active spectral text removes provisional wording while
retaining the fixed-checkpoint and no-new-interval qualifications. Before-edit
source copies and the exact numeric replacement map are preserved under
`artifacts/manuscripts/v5_spectral_matched122_20261115/`; compilation evidence
belongs in its separate `build/` directory.

The [final manuscript validation](../../../artifacts/manuscripts/v5_spectral_matched122_20261115/validation.json)
records successful marked and clean v5 PDFs (37 and 29 pages) with zero LaTeX
warnings and no provisional-word occurrences. The clean main text remains
ten pages. Annual numerical claims are checked, original collaborator comments
are retained, and visual inspection of clean pages 9–10 and 22–25 found readable
figures, captions, equations and table without clipping or overlap. The v96w
response is synchronized and standalone/combined response builds pass. This
completes the requested local manuscript update and its separately archived
build evidence; original seed/bootstrap results and earlier cohort archives
remain intact.

<a id="v5-variable-spectral-audit-20261115"></a>
### November 15 — Audit per-variable accuracy and spectra; select balanced illustrations

Date: 2026-11-15 (America/New_York).
Execution: completed analysis of existing evaluation outputs.
Outcome: descriptive support for variable- and lead-dependent accuracy/spectral
tradeoffs; selected illustrations are exploratory, not a confirmatory ranking.

The user requested per-variable metrics and examples showing better and worse
behavior after discussing a per-scale 25% power-retention criterion. We
examined the same four fixed checkpoints (raw GC, M1, output-filtered GC and
matched R1), all 122 standard 2023 starts, 40 leads and 83 variable/pressure
channels. No new training, forecasts, Slurm jobs or manuscript edits were made.

The existing optimal-filter `annual_pooled.npz` contains physical native-grid
`grid_mse[lead,branch,variant,channel]`; the reset annual pool contains
`normalized_grid_mse[lead,branch,channel]`, converted using the shared
normalization scales squared. Channel RMSE is the square root after pooling
MSE across the selected leads. Summing native channel MSE with GraphCast's
original weights/scales reconstructs saved exact loss to `1.42e-14` maximum
absolute difference. Spectral error power is not substituted for grid error.

Retained the existing equal-degree mismatch and added a separately named
truth-power-weighted mismatch within each physical channel:
`100 * sum_{lead,degree=1..179} abs(Pf-Py) / sum_{lead,degree=1..179} Py`.
Per-degree coherence is averaged with the same truth-power weights; the
equal-degree coherence is also retained. No physical powers with unlike units
are combined. The new weighting reduces weak-tail influence but can emphasize
large-scale static structure because the saved spectra do not subtract a
climatology. It is supplemental, not a replacement definition of the original
manuscript's spectral score.

All source files were checked against the completed v5 archive's pinned
digests. Dates, channel layouts and truth spectra match. The reconstructed
aggregate loss and equal-degree scores reproduce the existing annual table.
The original finite reset-rerun drift warnings remain in the new summary;
they are advisory and do not establish significance thresholds. An independent
agent recomputed 72 metrics and 36 contrasts for T2m/T400/Z500/TP6h from the
original arrays using scalar summation, with maximum discrepancy `2.84e-14`,
and checked 144 temporal threshold-crossing rows.

M1 versus R1 categories use only the observed signs of RMSE and
truth-power-weighted mismatch differences. They are not acceptance tests:

| Window | Lower RMSE / lower mismatch | Lower RMSE / higher mismatch | Higher RMSE / lower mismatch | Higher RMSE / higher mismatch |
| --- | ---: | ---: | ---: | ---: |
| Day 5 | 31 | 50 | 1 | 1 |
| Day 10 | 8 | 73 | 0 | 2 |
| Rollout | 7 | 75 | 0 | 1 |

At day 10, the original equal-degree mismatch worsens in all 83 channels.
This distinction must remain visible when discussing the favorable cases
under power weighting. The selected illustrations are:

| Field and lead | M1 RMSE reduction vs R1 (%) | Power-weighted mismatch difference (pp, M1 minus R1) | Interpretation |
| --- | ---: | ---: | --- |
| T2m, day 10 | 4.9817 | -0.3770 | Lower RMSE and power-weighted mismatch; equal-degree mismatch still increases |
| T400, day 5 | 1.9411 | -0.0844 | Lower RMSE and both mismatch measures with higher coherence; later leads show a tradeoff |
| U850, day 10 | 5.5588 | +16.1644 | Lower RMSE and higher coherence with markedly stronger attenuation |
| Q100, day 10 | -1.0132 | +1.1805 | Modest deterioration in RMSE, both mismatch measures and coherence |

T2m illustrates why a variable should not receive a permanent good/bad label:
its day-5 power-weighted mismatch increases, while its day-10 value decreases.
Q100 is an adverse example relative to R1, not a claim of worse skill against
every comparator. All four predictors appear in the selected plots, and all
83 channels remain in the tables and overview scatter plot.

Per-channel temporal crossings use `Pf/Py < 1-deficit` at fixed degree, with
deficits 20%, 25% and 30%, and degrees 4/10/20/40/80/160. The exports flag
crossing by the first six-hour evaluation and later recovery across the
threshold; an absent crossing is explicitly limited to the ten-day window.
The per-variable figures show the full degree range 1–179, with the 25%
amplitude threshold and 20–30% band. These quantify attenuation, not a general
useful-forecast horizon or a release gate.

Evidence and reproducibility:

- [Analysis script](../../../scripts/analyze_models/analyze_v5_variable_spectra_20261115.py)
- [Source pins, methods, warnings and validation](../../../plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115/summary.json)
- [All channel metrics](../../../plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115/channel_metrics.csv)
- [All contrasts](../../../plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115/channel_contrasts.csv)
- [Per-scale crossings](../../../plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115/scale_crossings.csv)
- [Selected examples and rationale](../../../plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115/selection.json)
- [Independent numeric audit](../../../plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115/independent_validation.json)
- [Accuracy and weighted spectral curves](../../../plots/analyze_models/images/resolution_eval/v5_variable_spectral_audit_20261115/selected_accuracy_and_spectral_metrics.pdf)
- [Per-variable amplitude/coherence spectra](../../../plots/analyze_models/images/resolution_eval/v5_variable_spectral_audit_20261115/selected_scale_diagnostics.pdf)
- [All-channel comparison](../../../plots/analyze_models/images/resolution_eval/v5_variable_spectral_audit_20261115/all_channels_accuracy_power.pdf)

Run from the project root:
`source scripts/graphcast_env.sh && MPLCONFIGDIR=/tmp/v5-variable-spectra-mpl python scripts/analyze_models/analyze_v5_variable_spectra_20261115.py --fields T2m,T400,U850,Q100`.
The data export includes a script snapshot and digest; current repository
changes are uncommitted. `--plot-only` reuses the compact audited exports.
The selected figures were visually inspected. These findings support a
balanced per-variable illustration set; manuscript insertion remains a
separate editorial step.

<a id="v5-variable-spectral-presentation-20261115"></a>
## Authorized v5 variable-wise spectral presentation

Date: 2026-11-15 (America/New_York). Category: analysis/presentation.
User authorization: "let's do that", adopting the proposed Section 4.6 and
Appendix C revision. This uses the existing fixed-checkpoint 122-date,
40-lead evaluation; no new training, forecasts or Slurm jobs were launched.

**Change and rationale.** Section 4.6 retains the GraphCast-weighted combined
83-channel amplitude/coherence comparison. Appendix C now pairs amplitude
and coherence in four-column figures at day 5 and day 10, with T2m, Z500,
U850 and Q100. Z500 replaces the exploratory T400 illustration as a standard
circulation diagnostic with scale-dependent attenuation and excess tail
power. T2m illustrates relatively retained structure (with the terrain
caveat), U850 an accuracy/attenuation tradeoff, and Q100 an adverse day-10
result. Selection is explicitly post hoc and illustrative, not a ranking.
The prior T400 exploration remains intact.

**Numerical result.** Native-grid RMSE reductions of M1 relative to R1 are:

| Field | Day 5 (%) | Day 10 (%) |
|---|---:|---:|
| T2m | 1.372487 | 4.981699 |
| Z500 | 2.020252 | 4.597727 |
| U850 | 1.756267 | 5.558820 |
| Q100 | 0.843011 | -1.013207 |

Physical squared errors are pooled over dates and area-weighted grid points
before taking square roots; these are not aggregate GraphCast MSE reductions.
M1 improves RMSE in 81/83 channels at each selected lead, while equal-degree
power mismatch worsens for all 83 at day 10. The paper retains this adverse
aggregate evidence and the complete channel exports. The scalar comparison
moves to Appendix C Table 6; native-grid field reductions form Table 7.
The original equal-degree score remains distinct from the supplementary
ERA5-power-weighted mismatch, which emphasizes energetic scales.

**Reference and limits.** The sqrt(0.75) line denotes a 25% power deficit
only for a single channel at a given scale. In combined curves it is a
weighted-amplitude reference, not a threshold on mean power loss or skill.
Supporting 20/25/30% scale-crossing diagnostics preserve censoring and
recrossings; no universal horizon or formal spectral equivalence is claimed.
Day 5 retains the wording "similar aggregate power mismatch". The scores
remain descriptive fixed-checkpoint point estimates without new uncertainty
intervals or channel-wise significance tests. Existing finite rerun warnings
are retained in source manifests, not used as failure gates.

**Artifacts and validation.** Both `Overleaf/v5_clean.tex` and `v5.tex` are
synchronized, along with the v96w response in TXT and TeX. Figure numbers
6 and 10--15 are unchanged. All four new figure assets retain full plotted
coordinates: Z500 amplitudes above one and Q100's negative coherence are
visible; no power-mismatch annotations appear on coherence panels. The
renderer agrees with independent channel exports at tolerance 1e-13 and
detached replay reproduces every plotted-coordinate signature. PDFs were
compiled and revised pages visually checked; obsolete historical figure
references in the tracked copy were corrected without deleting the captions.

- [Portable renderer, figures, complete channel data and provenance](../../../Overleaf/archive/Graphs/v5_variable_spectral_presentation_20261115/README.txt)
- [Selection rationale](../../../Overleaf/archive/Graphs/v5_variable_spectral_presentation_20261115/selection.json)
- [Exact selected RMSE values](../../../Overleaf/archive/Graphs/v5_variable_spectral_presentation_20261115/selected_native_grid_rmse.csv)
- [Figure coordinate checks](../../../Overleaf/archive/Graphs/v5_variable_spectral_presentation_20261115/figure_checks.json)
- [Detached replay check](../../../Overleaf/archive/Graphs/v5_variable_spectral_presentation_20261115/portable_replay_validation.json)
- [Clean manuscript PDF](../../../artifacts/manuscripts/v5_variable_spectral_revision_20261115/build/v5_clean.pdf)
- [Build/source validation](../../../artifacts/manuscripts/v5_variable_spectral_revision_20261115/validation.json)

The build directory also retains the tracked manuscript and standalone/combined
reviewer responses. Before-edit snapshots and page previews accompany it.

<a id="v5-spectral-reference-explanation-20261115"></a>
## Explain the attenuation reference and its M1/R1 interpretation

Date: 2026-11-15. Category: analysis/presentation.
User feedback: the sqrt(0.75) caption was cryptic and the main text did not
explain why the line was chosen, cite the methodology, or say what to assess.

Verified Subich et al. (2025), Section 3.1, at
https://arxiv.org/html/2501.19374v2#S3.SS1. Their 25% per-scale power-deficit
choice is illustrative. For combined curves, the weighted mean of amplitude
ratios is not the square root of the weighted mean power ratio. Main text,
caption, Appendix C and v96w replies now explain the line's purpose: compare
where appreciable attenuation is reached; a lower-wavenumber first crossing
means coarser scales are affected. Crossings use sampled integer degrees
without interpolation. Spatial crossings at a fixed lead are distinguished
from the separate per-channel crossing times at fixed wavenumbers.

The unchanged combined-curve exports give day-5 M1/R1 first crossings of
16/16 at sqrt(0.75), versus 7/12 at day 10. At sqrt(0.8) and sqrt(0.7), the
pairs are 14/14 and 18/19 at day 5, versus 7/10 and 9/14 at day 10. Thus the
qualitative contrast is stable across these nearby illustrative choices.
The maximum absolute M1/R1 amplitude difference is 0.01759 at day 5 and
0.14085 at day 10; maximum coherence differences are 0.01043 and 0.03373.
The full curves support similar combined spectral behavior at day 5, while
day 10 has stronger M1 attenuation despite generally slightly higher
coherence. No formal equivalence or universal effective resolution is claimed.

- [Derived crossings and interpretation](../../../Overleaf/archive/Graphs/v5_spectral_reference_explanation_20261115/README.txt)
- [Crossing table](../../../Overleaf/archive/Graphs/v5_spectral_reference_explanation_20261115/combined_reference_crossings.csv)
- [Source digest and curve differences](../../../Overleaf/archive/Graphs/v5_spectral_reference_explanation_20261115/interpretation.json)
- [Updated clean PDF](../../../artifacts/manuscripts/v5_spectral_reference_explanation_20261115/build/v5_clean.pdf)
- [Build checks](../../../artifacts/manuscripts/v5_spectral_reference_explanation_20261115/validation.json)

Both manuscript versions and both response formats were synchronized. Existing
figure arrays were reused; no forecasts or training were rerun. The earlier
illustration revision was published as manuscript commit 5737cce; this follow-up
retains its figures and adds interpretation and derived crossing evidence.

### Appendix interpretation by variable (authorized follow-up, November 15)

User accepted the proposed discussion beside Figures 11--12. Appendix C now
interprets ranges below sqrt(0.75), alongside coherence and native-grid RMSE:
day-5 T2m remains above the line for M1/R1 and U850 remains below from degree15;
day-10 U850 remains below from5 for M1 versus9 for R1 despite lower M1 RMSE.
T2m has only isolated M1 dips at day10; Z500's fine-scale excess power retains
low coherence, while Q100 has a sustained deficit from11 for both models and
slightly worse M1 RMSE. First crossings are not treated as sharp skill limits.
The existing per-variable RMSE values and terrain caveat are preserved.

Both manuscript sources are synchronized. The same 16 field/lead/model curves
were inspected without rerunning forecasts. The retained range evidence and
source digest are in
[per_variable_evidence.json](../../../artifacts/manuscripts/v5_appendix_reference_interpretation_20261115/per_variable_evidence.json).
Builds, before-edit snapshots, diffs and page previews are in
`artifacts/manuscripts/v5_appendix_reference_interpretation_20261115/`.
