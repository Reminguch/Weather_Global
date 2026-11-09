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
