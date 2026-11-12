# GC–Mamba lambda restart with M1's training recipe

Date: **2026-11-12, America/New_York**. Status: **GC–Mamba-only restart submitted;
all new jobs pending at verification**. The user approved cancellation and fresh submission
for GC–Mamba only, explicitly keeping DeepMind/full-GC unchanged. This supersedes
the full-GC revision proposed below; that portion remains an unexecuted proposal.

Old GC–Mamba continuation/history/controller jobs `17062358/17062359/17062360` and
the obsolete shared final-comparison chain `17353172/17353173/17353174` were cancelled
after verifying they were pending. Full-GC `17049163/17049164/17049165` and diagnostic
`17404694/17404732` were preserved. [Cancellation and protected-job records](../../artifacts/plans/v24_m1_matched_lambda_restart_20261112/activation_gcm_only/cancellation.json)
and [protected-job verification](../../artifacts/plans/v24_m1_matched_lambda_restart_20261112/activation_gcm_only/protected_jobs_verification.json)
record the scope. All completed old checkpoints and evaluations remain available.

## Executed scope and submitted jobs

Five fresh GC–Mamba arms use lambda **0, 0.25, 0.5, 0.75, 1**. Each follows M1's
initial 500 updates and main 2,000 updates, including the parameters-only transfer
and optimizer/RNG/data/recurrent-state reset between those stages. The allocation
boundary at total 1,250 restores the complete state without another reset.
GC–Mamba uses the learning rates, schedule, stopped weather gradients, native BF16
base loss and deterministic segment order specified in the GC–Mamba column below.
Lambda applies from the first update of both stages. Lambda 0 takes the historical
`mse` path with no spectral transform; positive lambda adds the existing stable
AMSE excess to that same native base loss.

Full-GC keeps its existing continuous schedule, full weather gradients, FP32 base
loss, shuffled order and per-arm rates (`3e-7` at lambda 0; `1e-5` otherwise).
Its frozen source, configurations, training jobs and independent downstream jobs
are unchanged. Thus the new GC–Mamba arms match M1's recipe; full-GC remains a
comparison with a different training recipe, including its existing LR confound.

| Job | Work | Dependency |
| --- | --- | --- |
| `17406861_0–4` | Fresh initial 500 + main 750 = total 1,250 | None |
| `17406862_0–4` | Exact resume main 750 → 2,000 = total 2,500 | Corresponding `17406861` arm |
| `17406863_0–4` | Development MSE histories at total 500/1,000/1,500/2,000 | Corresponding `17406862` arm |
| `17406864` | Validate endpoints and launch development/annual smoothing | All `17406862` arms |
| `17406897` | Materialize final raw/SWA comparison | Full-GC `17049163` and new GC–Mamba `17406862` |
| `17406898_2–21` | Twenty final raw/SWA MSE and spherical-spectrum evaluations | `17406897` |
| `17406899` | Combined final report, including historical M1 and public GC | All `17406898` arms |

The final diagnostic keeps the same four 2022 starts and 40 leads as the pilot.
GC–Mamba's final main checkpoint 2,000 represents 2,500 total updates; its SWA
averages main 500/1,000/1,500/2,000. Full-GC uses its existing total-step counters
and SWA 1,000/1,500/2,000/2,500. Historical M1 raw/SWA forecasts are reused with
the same frozen forecast/metric source. Development-32 fitting and annual-122
raw/filtered comparisons are launched after valid endpoints by the new controller;
the annual join uses the unchanged full-GC smoothing outputs.

Training requests one 80GB GPU, 8 CPUs, 384 GiB RAM and 30 hours per allocation.
The new stopped-feedback implementation's production runtime remains unmeasured.
Finite loss increases and numerical drift remain recorded warnings, not gates.

Implemented and checked: native-loss/gradient semantics, lambda-0 bypass,
parameters-only reset, exact midsegment resume, checkpoint counters, final SWA
mapping and downstream joins. The native/config/endpoint/checkpoint regression
suite passed 116 tests, campaign tests passed 8, and mixed-final/SWA/annual-report
tests passed 18. All ten actual stage configurations loaded. Parsed lambda-0
configurations match M1's scientific settings after equivalent path/default
normalization; positive-lambda configurations differ only in objective
([parity record](../../artifacts/checkpoints/v24_Ilya/res1_m1_matched_gcm_lambda_20261112/checks/recipe_parity.json)).
Submission is not a training result; new metrics are pending.
[Final activation verification](../../artifacts/plans/v24_m1_matched_lambda_restart_20261112/activation_gcm_only/activation.json)
at 19:18 UTC (14:18 EST) confirmed every replacement job pending with its intended
dependency and all six replaced job roots cancelled in Slurm accounting.

Frozen artifacts and exact submission commands:

- [Training manifest](../../artifacts/checkpoints/v24_Ilya/res1_m1_matched_gcm_lambda_20261112/manifest.json):
  record `f75dbbdec1e1af4147e822bfe8a639e4775589d1abebdf26fbb1b13b29ce739c`;
  source `a6a1c4e7ae895eb788227924146b3da84e5542bb10b7f5e294ebbe15943abd7f`.
- [Mixed final comparison plan](../../artifacts/evaluations/v24_mixed_final_comparison_20261112/plan.json):
  record `f6d00da5679d73c3285435ba0d71a6dddd800c956b19e00b35a8da1d05ff4f10`;
  source `8a5f9c9a8b02995e333de37a68d626ea5aaf940294b9c8a20ea21e7fa1935150`.
- Exact job commands are in each campaign's `submissions/` directory.

## Original review proposal — retained as history

The remainder preserves the original two-family proposal. Its full-GC changes
and replacement instructions were **not executed**. The executed scope and job
table above supersede those portions; GC–Mamba implementation is now complete.

The user originally requested a fresh GC–Mamba lambda sweep matching historical M1's
learning rate, schedule, loss calculation and other training settings, and an
analogous full-GC sweep retaining its current learning rates. They explicitly
request checking this plan before job changes. The existing mismatched
1,250-update checkpoints remain historical evidence, not initializers.

The [machine-readable review specification](../../artifacts/plans/v24_m1_matched_lambda_restart_20261112/review_plan.json)
contains all ten arms and twenty stage specifications, pinned reference inputs,
the two-allocation mapping and the jobs to replace. It is deliberately labeled
**non-executable**: the native-loss/stopped-feedback extension described below
must be implemented and checked before activation. It is not a validated
training configuration or a submitted experiment.

## Exact training recipe

Five lambda values per family: **0, 0.25, 0.5, 0.75, 1**. Each arm starts fresh
from the same public GraphCast-small 1-degree checkpoint; GC–Mamba uses the
historical spatial overlay, fresh seed-22 Mamba parameters and zero output head.
No trained M1, old November model, profile or calibration weights are reused.
All arms within a family share the same initial parameter values.

| Setting | Revised GC–Mamba | Revised full GC |
| --- | --- | --- |
| Initial stage | 500 updates, constant LR after 50-update ramp | Same stage shape |
| Main stage | 2,000 updates, 50-update ramp and cosine decay | Same stage shape |
| Initial-stage peak LR | `1e-4`, all parameter groups 1× | Retain current per-arm rate below |
| Main-stage spatial LR | `1e-4 → 1e-5` | Retain current peak, cosine to one tenth |
| Main-stage Mamba LR | 3× spatial: `3e-4 → 3e-5` | Not applicable |
| Weather feedback | Predicted weather is fed forward; its gradients are stopped (`closed_loop_sg`) | Same detached-weather protocol |
| Memory gradients | Retained through the 24-step window | No Mamba memory |
| Base loss | Historical native BF16 residual-MSE path | Corresponding native GraphCast BF16 loss path |
| Training order | Historical deterministic contiguous segment groups | Identical order |

All other shared settings are copied from M1: batch4, BPTT24, K20, segment96,
seed22, AdamW betas0.9/0.98, weight decay1e-4 on all parameters, clip1, BF16 model
computation and FP32 weather tape. The initial500 stage already uses K20; it
does not shorten the autoregressive horizon.

At the initial500/main boundary, transfer **parameters only**. Reset the
optimizer, RNG, data cursor and recurrent state, exactly as M1 did. The main
stage starts again at the beginning of the historical training order. Source
inspection and actual segment arrays confirm this order is deterministic
contiguous grouping, whereas the current November manifest uses shuffled epoch
permutations. All106 underlying eligible segments match.

“Keep full-GC LR as it is now” is interpreted literally per existing arm:

| Full-GC lambda | Initial-stage peak/constant LR | Main cosine LR |
| ---: | ---: | ---: |
| 0 | `3e-7` | `3e-7 → 3e-8` |
| 0.25, 0.5, 0.75, 1 | `1e-5` | `1e-5 → 1e-6` |

Thus full-GC lambda0 versus positive lambda retains the existing learning-rate
confound. GC–Mamba's five arms use one identical schedule and differ only in
lambda. Full GC shares the recipe except for the retained LR and the intrinsic
model/target representation differences.

## Loss and required implementation

Use `L = G_native + lambda * E`, where `G_native` is the historical native
GraphCast weighted-MSE calculation and `E` is the existing stable nonnegative
spherical AMSE excess, calculated in FP32/complex64. Apply each arm's lambda
during **both** stages, starting from its first update. The spherical operator,
degrees0–179 and excess definition are unchanged.

At lambda0, GC–Mamba must take the original `loss_kind="mse"` path unchanged;
do not construct or evaluate a spherical transform. For positive lambda, call
native `loss_and_predictions` once, retain that base loss and use its returned
physical forecast for the excess. Do not silently replace native loss with the
current FP32 full-field `grid_mse_fp32` calculation.

This needs a deliberate backend extension, not just changed JSON flags:

1. Add a native-MSE-plus-excess objective and explicit configuration validation.
   Log native MSE and spectral excess separately, with lambda applied once.
2. Extend stopped-feedback training and validation to a full-GC predictor that
   returns the complete forecast. Never add public GC to that output. The
   current residual-only stopped-feedback backend would otherwise double-add it.
3. Preserve step-zero checkpoints and unambiguous stored/main/total-update
   metadata. Implement the two-stage reset and exact allocation-boundary resume.
4. Freeze a new campaign source snapshot and rebuild downstream plans with the
   new checkpoint identities and counters.

Full GC's native normalizer uses the state-increment target, while M1's
normalizer uses the correction target `truth - frozen_GC`. Their mathematical
forecast-error weighting is the same, but BF16 rounding cannot be bitwise
identical across these different predictor representations. This does not
justify retaining the current FP32 full-field base loss.

Before activation, check native lambda0 loss/gradient parity, absence of a
spectral computation at zero, correct positive-lambda addition, full-GC
detached-weather semantics without double addition, training/validation
consistency, fresh initialization, identical ordering and both state-transfer
boundaries. Reuse relevant existing checks; no extra GPU qualification job or
measured-descent gate is proposed. Finite numerical differences remain warnings.

## Two 1,250-update jobs per arm

| Allocation | Work | End state |
| --- | --- | --- |
| First 1,250 updates | Fresh initial500; parameters-only reset; main1–750 | Total1,250, main checkpoint750 |
| Second 1,250 updates | Exact resume main750; continue through main2000 | Total2,500, main checkpoint2000 |

There is **no** optimizer, RNG, data-order or recurrent-state reset at total
1,250. That point is an execution split, not a new training stage. M1's
historical main750 falls midsegment, so the complete state must be restored.
Use a separate output directory for each arm's initial and main stages and
record total counts explicitly. The main cosine schedule always spans2,000
updates; never reset it to a new1,250-step schedule for the second allocation.

Four new training arrays cover two families and two parts, each with five
unthrottled elements. Part2 uses `aftercorr` on the corresponding part1 array.
Provisional requests retain the proven current shape: one80GB GPU,8CPUs,
384GiB host RAM and30hours per part. Old full-feedback production measured
p90 58.76/59.71seconds per update for GC–Mamba/fullGC, approximately20.4/20.7hours
per1,250 updates. The first allocation adds a second initialization/compile;
the revised stopped-feedback runtime is not yet measured. These are conservative
budgets, not claims of measured cost for the new implementation.

## Replacement of queued jobs

The existing continuation arrays **cannot cleanly be changed to fresh starts
through the present dispatcher**. Their spooled batch arguments fix the old
campaign and phase1, which unconditionally resumes1250 and stops2500. The
installed Slurm26.05.5 exposes no documented user update of the batch script or
arguments. Changing working directory or selected-arm configurations does not
change that dispatch. Do not rewrite a frozen source tree to bypass it.

Therefore the proposed activation uses **replacement jobs with new IDs and new
queue age**, rather than promising preservation of the old IDs. This differs
from the November7 revision, which redirected a still-pending phase0 job whose
existing dispatcher supported fresh initialization.

After user review and implementation checks, recheck job state and replace
only the following pending old-production chain:

| Old jobs | Replacement role |
| --- | --- |
| `17062358_0–4`, `17049163_0–4` | New fresh first-part arrays; new dependent second-part arrays |
| `17062359_0–4`, `17049164_0–4` | New history evaluations using total-update-aware checkpoint paths |
| `17062360`, `17049165` | New finish/smoothing controllers |
| `17353172 → 17353173_2–21 → 17353174` | New final raw/SWA materialization, forecasts and report |

The old final plan pins old configurations, source, midpoint proofs and paths;
changing dependency IDs alone would be incorrect. Rebuild it explicitly.
Cancel dependent old jobs before replacing their parent training arrays, and
retain exact command/submission records. If the observed pending state has
changed by activation, reconcile that state before acting.

Keep **`17404694 → 17404732`** unchanged: this diagnostic evaluates saved old
checkpoints and remains useful independently. Preserve all completed checkpoints,
metrics, evaluations and frozen sources from the old campaign. The separate
four-fresh-horizon full-GC study remains on hold.

## Evaluation and review boundary

Retain the matched development32 and annual122 cohorts,40 six-hour leads,
DeepMind GC and historical M1 raw/SWA references. Report exact original GC MSE
and actual spherical spectra/power ratios. The raw main2000 checkpoint has
2,500 total updates. Equal-weight SWA uses main500/1000/1500/2000, corresponding
to total1000/1500/2000/2500. Use the existing smoothing scope and independently
development-fitted filters with the new model identities.

The original review decision covered this recipe and the clean replacement of
the queued continuation chain. The subsequent authorization narrows execution
to GC–Mamba; full-GC training, rates, source and independent downstream jobs stay
unchanged. The original review specification remains preserved as proposed history.
