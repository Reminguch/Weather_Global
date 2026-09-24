# Proposal for review: equal-variable physical MSE

**Later user direction, September 24:** the primary experiment now uses a
public five-term NeuralGCM loss reconstruction with a frozen backbone and
memory-only cross-step gradients, comparing K=1 / 6 h and K=2 / 12 h. See the
[implemented specification](../docs/experiments/neuralgcm_residual/NGCM_ALIGNMENT_K2_20260924.md).
The proposal below is retained as an alternative and was never activated.

**For Ilya's review. Status: proposed, not implemented in production training.**
The running jobs and all checkpoints currently plotted were trained with the
6 h v2 loss. This document proposes a replacement normalization for a matched
pilot. It does not claim to reproduce the original NeuralGCM training loss.

## Proposed decision

Give all seven variables equal weight by expressing each variable's physical
MSE relative to the frozen baseline's training-set physical MSE. **Do not use
the six-hour or 24-hour temporal-change sigma.** Freeze the seven reference
errors before training. For the first pilot, retain the existing spatial and
pressure weights to isolate the effect of this normalization change.

## Motivation

Equal nominal coefficients did not produce a balanced objective. Under the
original README formula, baseline cloud liquid water contributes **97.01%**
of validation loss and cloud ice contributes **2.99%**. Some cloud-liquid
normalization scales are at the `1e-9 kg/kg` floor. At 175 hPa, baseline RMSE
is `1.66994e-6 kg/kg`, approximately 1,670 times that scale. After squaring,
pressure weighting and the common `1/7` coefficient, this one level contributes
approximately **4,484 of the total 20,832 loss units**.

The later v2 normalization instead gives geopotential **95.06%** of baseline
validation loss. These are empirical contributions under two custom scoring
rules, not physical-energy shares or measured gradient shares. They motivate
balancing relative forecast errors directly. See the
[measured definitions and diagnosis](LOSS_DEFINITION_AND_DESIGN.md#3-measured-problems-with-both-recipes).

## Exact proposed K=1 objective

For sample n, variable v, pressure level p, and grid cell x, define the physical
error as prediction minus ERA5 at the **same six-hour forecast valid time**.
Let A denote Gaussian-weighted latitude integration and uniform longitude mean.
Retain all existing 37 pressure levels, with `b[p] = p / sum_p(p)`.

```text
E_model[n,v] = sum_p b[p] * A((prediction_model[n,v,p] - ERA5[n,v,p])**2)

B[v] = mean over fixed training calibration samples n of E_frozen_baseline[n,v]

L[n] = (1/7) * sum_v E_residual[n,v] / B[v]
L_batch = mean_n L[n]
```

The variables are temperature, geopotential, eastward wind, northward wind,
specific humidity, cloud ice, and cloud liquid water. Each coefficient outside
the ratio is **exactly `1/7`**, including both cloud variables. There are no
additional factors such as geopotential amplitude 2, cloud amplitude 0.05,
or a separate cloud penalty.

E and B have matching physical units squared. Each ratio is dimensionless.
The equivalent error scale is the **baseline RMSE `sqrt(B[v])`**, not a
temporal-change standard deviation. Converting a variable's units consistently
in E and B leaves the objective unchanged. Averaging raw physical MSEs without
this rescaling would make variable importance depend on arbitrary unit choices.

## What equal weighting means

On the calibration set, every baseline component has mean `E_baseline[v]/B[v] = 1`,
and baseline total loss is one. Reducing any one variable's mean MSE from its
baseline reference to `0.9 * B[v]` lowers total loss by `0.1/7`, provided the
other terms stay fixed. Thus a 10% temperature MSE reduction and a 10% cloud
MSE reduction have the same effect. This statement concerns **MSE**, not RMSE.

This balances the reference contributions. It does **not** force all seven
contributions to remain equal as the model learns. A variable whose relative
error grows contributes more. Individual batches and validation sets need not
give baseline score one because B is fixed from the training calibration set.
Always evaluate baseline and residual with the same fixed B and target samples.
Equal reference loss contributions also do not imply equal gradient norms.

## Calibration and implementation contract

1. Select a fixed calibration set from **2015–2021 training data only**, covering
   the training years and seasons. Declare the sample selection before comparing
   residual results. Do not fit B from the 2022 validation results already plotted
   or the held-out 2023 test set.
2. Run the frozen baseline with the intended forecast initialization and lead,
   and compute E using exactly the same grid, truth, levels and reductions as
   the residual objective. For K=1, these are six-hour predictions initialized
   from ERA5 at each origin, following the existing residual-memory policy.
3. Save the seven B values, physical units, calibration sample identities,
   checkpoint identity, reductions and source hashes as a versioned artifact.
   Restore those same constants on checkpoint resume.
4. Require B to be finite and positive. Audit near-zero references before using
   their inverse. Any necessary floor must have a documented physical meaning
   for that variable's squared units; a universal numerical epsilon is unsuitable.
5. Keep B fixed. Do not divide by current residual batch errors or continuously
   rescale from validation scores. Mean squared errors must be aggregated before
   forming the fixed calibration constants.

No new B artifact has been fitted or activated by this documentation change.

## Remaining vertical and rollout choices

Equal-variable scaling solves the imbalance **between variables at calibration**.
It preserves the relative contributions of pressure levels **within each variable**.
Large upper-atmosphere geopotential errors can therefore still dominate the
geopotential component, and their representation error still needs the proposed
0 h encode/decode audit.

For the first comparison, keep `b[p]` unchanged. Inspect per-level contributions
and physical errors. If pressure-band balancing is needed, define physical bands
and their weights before evaluating candidate improvements, fit B[v,g] on the
training split, and use `mean_v sum_g w[g|v] * E[v,g]/B[v,g]`. This is an
additional design change; equal-variable normalization alone does not implement it.

For K=2, use actual 6 h then 12 h rollouts with the predicted state passed between
steps. One explicit extension is to define E[v] as the mean of its 6 h and 12 h
physical MSEs, and recalibrate B[v] on that exact baseline protocol. This keeps
equal variable references across the combined objective but does not guarantee
equal contributions from each lead. Separate B[v,lead] with an equal-lead average
is a different option. Do not reuse K=1 calibration while claiming equalized
K=2 or K=20 contributions. The physical-state gradient policy is a separate choice.

## Requested review decisions and validation

- Is equal relative physical MSE across **all seven variables** the right primary
  objective for this residual experiment?
- Is retaining pressure-proportional weights for the first pilot acceptable,
  with pressure-band balancing evaluated separately?
- For multi-step training, should calibration use combined-lead error or separate
  reference errors per lead?

After fixing these choices, compare matched residual initializations and training
budgets. Report each variable's physical RMSE and bias, per-level contributions,
the unchanged common evaluation scores, and closed-loop stability. A better
aggregate loss alone is insufficient. Independent representative-data GPU smoke
tests must use `gpu-test`, no explicit partition, and at most one hour per job.

## Figures to review alongside this proposal

- [Improving upper-air geopotential and humidity, individual time series and all four configurations](upper_air_physical_eval_20260924/README.md)
- [Physical variables versus physical time, displayed in the plot README](README.md#physical-variables-vs-physical-time)
- [Seven larger individual physical-time panels](README.md#physical-unit-forecasts)
- [Full-year physical RMSE and checkpoint identities](k1_physical_eval_20260924/README.md#physical-forecast-errors)
- [Original README, current v2, and simplified 24 h scoring comparisons](loss_definitions_20260924/README.md)

The physical-time curves are successive **K=1, six-hour forecasts**, with elapsed
valid-time axes. They are not free-running 72-hour trajectories. Loss-curve axes
remain **training step (optimizer updates)**.
