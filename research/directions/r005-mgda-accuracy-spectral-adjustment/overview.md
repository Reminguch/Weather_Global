# r005 — Can MGDA improve spectral fidelity while preserving exact GC accuracy?

Category: training
Status: active
Updated: 2026-11-02 (America/New_York)

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

Review the measured tradeoff before choosing another experiment; do not promote
MGDA as a joint win or automatically extend training. Exact GC loss remains
primary; power/coherence are secondary. Finite warnings did not block completion
and are not evidence of scientific failure or improvement. No new submission,
2,000-step extension, di32 arm or SWA is authorized by this completed pilot.

## Records

- [Original attempt](experiments.md#e001) and [warning-only recovery](experiments.md#e002)
- [Advisory-decrease continuation](experiments.md#e003)
- [Shared full-BPTT campaign](experiments.md#e004)
- [Implementation and resumption](devel.md)
