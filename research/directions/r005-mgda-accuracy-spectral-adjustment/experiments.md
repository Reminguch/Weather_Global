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
