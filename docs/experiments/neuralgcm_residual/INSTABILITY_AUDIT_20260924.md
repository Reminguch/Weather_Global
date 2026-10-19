# K=1 residual NeuralGCM instability audit

Follow-up: the user authorized a fresh run with numerical constraints and a
rebalanced loss. See the [versioned restart](FEEDBACK_V2_RESTART_20260924.md).
The findings below describe the original checkpoints and diagnostic interventions.

The observed failures have reproducible causes in the residual-to-solver
interface. Unconstrained pressure increments cause the immediate second-step
error increase. Injected degree-zero divergence drives a separate, sustained
pressure drift and eventual nonfinite states. The frozen NGCM's stochastic
forcing is disabled. This is not evidence that NGCM randomness corrupted the
cache or the comparison.

Removing direct pressure increments and preserving the solver's pre-correction
degree-zero divergence and vorticity coefficients eliminates the observed
catastrophic drift in the tested winter/summer, two-checkpoint interventions.
This does **not** establish forecast improvement, nor stability for every origin
or an arbitrarily long integration. Existing weights still produce worse
five-day errors than the baseline after that intervention.

## Reproducibility and scope

- Experiment: `res2p8_w128_256_di16_32_train2015_2021_20260920_scatter_v3`.
- Frozen source identity:
  `02afdad970c37dec9cc67b34a780be0621aba1f24f8773e7917a7ad2aeee2ec1`.
- Checkpoints: `checkpoint_pass_05.pkl`, epoch 5, optimizer update 2,120.
- All diagnostics load the frozen experiment source and pinned libraries.
  Production checkpoints were verified unchanged. None of these interventions
  changes the running production experiment or its objective.
- Eight independent diagnostic jobs completed with exit code `0:0`. All used
  Slurm `gputest` / `gpu-test`, one GPU and a one-hour limit.
- Full K=1 replay: jobs **14349822 / 14349823**, w128/di16 and w256/di32,
  all 1,455 validation records per checkpoint.
- Pressure interventions: jobs **14350308 / 14350310 / 14350311 / 14350309**,
  w128/di16, w128/di32, w256/di16, w256/di32 respectively. Each covers four
  origins, January 30, May 18, July 3 and October 16, 2022, at 00 UTC.
- Degree-zero interventions: jobs **14350828 / 14350835**, w128/di16 and
  w256/di32, January 30 and July 3, 2022, at 00 UTC, nine variants each.

[Machine-readable evidence](instability_audit_20260924.json) includes checkpoint,
source, diagnostic-script and original-report hashes, per-field metrics,
intervention trajectories and objective derivatives. Original full reports and
logs are retained under `logs/neuralgcm_instability_20260924/` locally.

## 1. Immediate second-step error: excessive direct pressure corrections

`NativeAdapter.apply_increment` in
[`native_state.py`](../../../src/models/neuralgcm_residual/native_state.py)
converts each nodal correction to modal coordinates, applies the generic
spectral mask, and adds it directly to the seven native fields. There is no
pressure-specific constraint or consistency check with the other state fields.
The generic mask does not impose the constraints described in section 2.

At the winter origin, w128/di16 predicts normalized log-pressure corrections
ranging from about -72 to +204. After scaling and spectral projection, the
actual log-pressure increment ranges from about -0.026 to +0.075, approximately
-2.57% to +7.82% in surface pressure in a single correction.

The following experiment applies a correction **once at 6 hours**, then advances
NGCM to 12 hours **without a second residual correction**. Thus the second-step
failure cannot be explained by a second recurrent-memory update.

| First correction, winter w128/di16 | Normalized MSE at 12 h |
| --- | ---: |
| No correction, frozen NGCM | 17,204.66 |
| Full residual | 139,391.72 |
| Pressure channel only | 139,796.19 |
| All channels except pressure | 19,404.57 |

Across four seasonal origins, pressure-only impulses also cause the large
second-step increase in w256/di32. The four-model pressure sweep and objective
derivatives are retained in the evidence JSON. Small pressure perturbations
along the learned correction direction reduce the current aggregate loss at
all 16 model/origin pairs. Their cloud-loss contribution dominates the
derivative. This shows that the objective rewards this direction locally; it
does not prove that every learned correction is globally optimal.

Surface pressure participates in the primitive equations and in the decoder's
sigma-to-pressure interpolation. Changing it while leaving the other native
fields fixed can improve decoded cloud scores while damaging the subsequent
physical evolution. The frozen NGCM physics parameterization, by comparison,
has `prediction_mask['log_surface_pressure'] = False`.

## 2. Separate runaway mechanism: injected degree-zero divergence

Deleting the pressure channel alone does **not** resolve the later NaNs.
The residual adapter also permits arbitrary degree-zero (`l=0`) divergence and
vorticity increments. These are spatially constant modes at each sigma level.
The solver's log-surface-pressure tendency contains the negative vertical
integral of divergence. Repeated extra mean divergence can therefore drive
large pressure drift.

The original NGCM learned physics predicts velocity tendencies and transforms
them to divergence/curl through `DivCurlNeuralParameterization`. Our residual
instead adds divergence/vorticity directly, bypassing that construction.
In Dinosaur, inverse-Laplacian reconstruction also sets the `l=0` inverse
eigenvalue to zero. This mode consequently does not appear in reconstructed
wind in the same way as ordinary divergence modes, although it can still enter
the pressure tendency.

The decisive counterfactual keeps the same checkpoint and residual computation,
changing only the injected degree-zero coefficient. **It preserves the existing
solver coefficient, rather than forcing the whole state to zero.** The learned
encoder's native state can already contain nonzero coefficients.

| Winter w128/di16, 120 h rollout | Outcome / endpoint loss | Area-mean surface pressure |
| --- | ---: | ---: |
| Frozen NGCM | 21,197.03 | 98,560.64 Pa |
| Full residual | NaN at 120 h | NaN |
| Divergence corrections only | NaN at 114 h | NaN |
| Same divergence corrections, remove injected `l=0` | 23,215.79 | 98,560.77 Pa |
| Full residual, remove injected divergence `l=0` | 1,445,733.38 | 100,482.51 Pa |
| No pressure increment, remove injected divergence and vorticity `l=0` | 38,808.25 | 98,560.45 Pa |

The same divergence-only experiment fails at 114 h for the summer w128/di16
origin. Removing its injected `l=0` component gives finite 120 h loss 22,785.54
and area-mean pressure 98,596.47 Pa, versus baseline 98,596.43 Pa.
For w256/di32, removing this component likewise prevents the divergence-only
catastrophe at both origins. With both mode constraints and no direct pressure
increment, all four model/origin combinations finish 120 h without the observed
catastrophic drift. Their endpoint losses still exceed baseline:

| Checkpoint / origin | Frozen NGCM 120 h loss | Constrained-intervention 120 h loss |
| --- | ---: | ---: |
| w128/di16, winter | 21,197.03 | 38,808.25 |
| w128/di16, summer | 20,609.26 | 60,253.77 |
| w256/di32, winter | 21,197.03 | 54,026.00 |
| w256/di32, summer | 20,609.26 | 94,955.80 |

These are individual endpoint losses, not five-day averaged selection scores.
Vorticity-mode removal further improves these interventions but does not, by
itself, prevent the original NaNs. Divergence-mode removal is the decisive
intervention for the demonstrated pressure runaway.

## 3. Cloud loss is legitimate, but this objective is not the original loss

Cloud ice and cloud liquid are original NGCM variables and participate in the
original training objective. The authors' supplementary information, section
G.3, uses 24-hour difference statistics, generally pooled over levels, with
specific humidity as the per-level exception. It additionally scales cloud
variables by **0.05** before the loss, specifically to balance variable
contributions. That is an amplitude scaling, not a statement that cloud terms
have a 0.05 MSE weight. Deterministic training also includes spectral and bias
losses and losses on the internal model representation.
[Original paper](https://www.nature.com/articles/s41586-024-07744-y),
[supplementary information, G.3–G.4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07744-y/MediaObjects/41586_2024_7744_MOESM1_ESM.pdf#page=25).

Our custom `neuralgcm_field_normalized_mse_v1` uses equal field coefficients,
six-hour change scales separately at each pressure level, and a cloud standard
deviation floor of `1e-9`. Upper-level cloud scales hit that floor. Cloud terms
contribute more than 99.999% of the baseline aggregate loss in the complete
validation audit. This can hide worse temperature, geopotential, wind and
humidity. The cloud contribution alone does not establish a benefit from memory.

The plotted aggregate improvements are reproducible, but are not evidence of
general weather-skill improvement. Full validation replay reproduced the saved
w128/di16 and w256/di32 scores to FP32 tolerance. Live and cached states and
forcing matched exactly on eight checked dates per model. An independent NumPy
float64 loss calculation also agreed. See the
[plot caption and per-field audit](../../../plot/README.md).

## 4. Randomness and the training-process failure

The actual loaded backbone is `deterministic_2_8_deg.pkl`. Its configuration
contains `StochasticPhysicsParameterizationStep.randomness_module =
@ZerosRandomField`. The class name includes "Stochastic", but the selected
random field is zero. The residual branch also has dropout zero.

The termination of three training jobs is a separate control-flow issue.
[`worker.py`](../../../src/models/neuralgcm_residual/worker.py) saves the one-step
validation result before autoregressive validation. When the cold rollout has
nonfinite origins, `evaluate_origins` returns an ineligible report and
`select_checkpoint` raises `ValueError`. The callback does not handle this
expected selection rejection, so training exits despite an already-saved
checkpoint. This exception does not fabricate the prior K=1 scores and is not
the cause of the physical divergence.

## Required implementation changes before a new training run

1. Version the correction contract. Prevent additional `l=0` divergence and
   vorticity injection, preserving baseline state coefficients. Either disable
   direct pressure correction initially, as tested here, or implement and test
   a physically consistent pressure/mass correction. Do not silently apply a
   different adapter to old checkpoint results and call it the same experiment.
2. Version and rebalance the loss after auditing per-variable and per-level
   contributions. Original NGCM includes cloud losses, but does not use this
   experiment's cloud normalization. Re-establish physical-unit skill metrics.
3. Treat nonfinite validation as a recorded, ineligible checkpoint, rather than
   an unhandled checkpoint-selection exception. Do not select failed origins
   away or silently report only successful trajectories.
4. Train under the revised contract and repeat all 32 cold validation origins,
   followed by the declared warm and held-out checks. A successful 120 h
   intervention on existing weights is not a completed retraining experiment.

This audit implements and verifies the interventions in diagnostic scripts.
Production adapter, loss, frozen source, checkpoints and running jobs remain
unchanged. It identifies the demonstrated causes and a tested correction
direction; it does not claim the old K=1 models are repaired forecasts.

## Reproduce the diagnostic evidence

The three `scripts/diagnostics/run_neuralgcm_*.sbatch` files run the corresponding
frozen-checkpoint analyses under `gpu-test`. They require the local experiment
artifacts. After all eight recorded jobs complete, export the portable evidence:

```bash
python scripts/diagnostics/summarize_neuralgcm_instability.py
```
