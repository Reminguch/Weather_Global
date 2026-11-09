# r005 — Can MGDA improve spectral fidelity while preserving exact GC accuracy?

Category: training
Status: active
Updated: 2026-11-09 (America/New_York)

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
