# Frozen NeuralGCM with Residual Mamba: matched K=1 and K=2

**Latest user direction, September 24, 2026:** freeze NeuralGCM, train only our
residual head, and compare K=1 (6 h) with K=2 (12 h). Cross-step differentiation
through Mamba memory is sufficient. The physical solver remains outside the
backward pass. We do not require the unpublished original training gin bindings.

**Status:** implemented in a separate training entry point. The independent
real-data GPU pilot and the K=1/K=2 production-CLI resume tests passed.
The detailed numerical/training tests across all four architectures are still
in progress before the new training matrix. See the
[smoke-test report](PAPER_LOSS_SMOKE_20260924.md) for measured results and job IDs.
Historical v2 checkpoints and plots retain their original loss definition.

This document supersedes its earlier full-solver-gradient proposal. It also
supersedes the equal-variable proposal as the primary next experiment. The
implementation is a **documented reconstruction of the public five-term loss**,
not a claim to reproduce unavailable original training bindings.

## What K and the training update mean

An episode starts from one ERA5 origin and zero residual memory. For each of K
steps, the frozen model advances six physical hours, the residual branch adds a
native-state increment, and the corrected state becomes the next forecast's
input. ERA5 does not replace the intermediate prediction. Parameters remain
fixed during the episode. One optimizer update aggregates two independent
origins and all their forecast leads.

| Setting | K=1 | K=2 |
| --- | --- | --- |
| Forecast outputs scored | +6 h | +6 h and +12 h |
| Residual injections per episode | 1 | 2 |
| Physical forecast horizon | 6 h | 12 h |
| Independent origins per optimizer update | 2 | 2 |
| Normalization difference interval | 24 h | 24 h |

The selected checkpoint's internal timestep is one hour. Six hours is our
residual injection interval. Twelve hours in the paper is an initial training
trajectory length. Neither is the optimizer's wall-clock update interval.
The paper starts its 2.8°/1.4° curriculum at 12 h and extends to 24 h at optimizer
step 2,000. Our matched experiment compares two fixed short horizons for the
initial 2,000 updates. It does not implement the paper's entire curriculum.

## Frozen model and gradient contract

The encoder, dynamics, learned physics and decoder parameters are frozen.
Only Residual Mamba parameters appear in the optimizer state.

```text
baseline[k+1] = stop_gradient(NGCM_6h(stop_gradient(state[k])))
(delta[k+1], memory[k+1]) = ResidualMamba(
    theta, memory[k], stop_gradient(features(state[k])), known_inputs[k])
state[k+1] = apply_increment(baseline[k+1], scale_increment(delta[k+1]))
prediction[k+1] = frozen_decoder(state[k+1])
```

The loss differentiates through the current residual increment and the frozen
decoder's **input**, and through the residual memory recurrence. It does not
propagate through a subsequent physical integration or the next step's physical
features. The next step still consumes the corrected state in its forward
calculation. Frozen parameters and stopped state derivatives are separate choices;
both are intentional here, following the user's explicit correction.

The batch/time objective is differentiated jointly, then its cotangents are
passed backwards through the residual recurrence. This preserves the coupled
batch-bias derivative. Summing independently computed per-example losses would
not implement that derivative.

## Implemented five-term loss

```text
L = 20 M_data + M_model
    + 0.1 M_data_spectrum + 0.1 M_model_spectrum + 2 M_bias
```

- **Data accuracy:** decoded pressure-level prediction versus ERA5, using the
  squared spherical norm of the scaled and filtered modal error.
- **Model accuracy:** corrected native state versus the frozen encoder's ERA5
  representation, with the corresponding modal norm.
- **Two spectrum losses:** compare square roots of summed squared modal
  coefficients over zonal wavenumber, retaining total wavenumbers 0–42.
- **Data bias:** average modal-amplitude errors across batch and forecast time
  **before squaring**. This follows the public `BatchMeanSquaredBias` default
  `abs(modal)`. The displayed paper equation can be read as signed coefficients;
  the amplitude choice is explicitly recorded rather than left implicit.

Modal norms are divided by the sphere area. Fields are summed, levels are
uniformly averaged, and batch/time accuracy and spectrum terms are averaged.
No pressure-proportional level weights or seven-variable average are retained.
The initial analysis is excluded from scoring.

Data fields are temperature, geopotential, eastward/northward wind, specific
humidity, cloud ice and cloud liquid water, on all 37 pressure levels. Native
fields are temperature variation, vorticity, divergence, specific humidity,
cloud ice, cloud liquid water and log surface pressure, on 32 sigma levels
(surface pressure has one level).

### Scales and cloud weights

Each loss field is divided by its training-only **24 h difference standard
deviation**, estimated from 60 selected training snapshots spanning 2015–2021.
Population moments pool samples, horizontal points and levels, except specific
humidity, whose scales remain per level. Pooling includes between-sample mean
shifts. Degenerate scales fail validation; no arbitrary scale floor is added.
Statistics use uniform gridpoint moments. Spherical loss norms and physical
verification use their own proper area weighting.

Multiplicative amplitude factors applied **before squaring** are:

| Variable | Amplitude | Corresponding squared factor |
| --- | ---: | ---: |
| Geopotential | 2 | 4 |
| Specific humidity | 0.66 | 0.4356 |
| Each cloud species | 0.05 | 0.0025 |
| Native log surface pressure | 5 | 25 |
| Other fields | 1 | 1 |

Thus this is not an equal-variable objective. In particular, cloud amplitude
0.05 means an MSE multiplier of 0.0025, not 0.05. Input-feature and native-increment
normalization remain separate from these new loss scales.

Accuracy and bias amplitudes have time factor `(1 + lead_hours/24)^(-1/2)`.
Spectrum amplitudes use `(1 + (lead_hours/40)^4)^(-1/2)`.

### Explicit reconstruction choices

The public paper describes order-12 filtering fitted to relative HRES error.
Those original bindings are unavailable. For these short 6/12 h forecasts we
record `exp(-log(2) * (l/120)^24)`, with half amplitude at wavenumber 120,
for both spaces and leads. On the selected grid (maximum wavenumber 64), this
is almost identity. This is an approximation, not the fitted original filter.

Native **scale statistics** use physical pressure-to-sigma interpolation and
wind-to-vorticity/divergence conversion. Surface pressure uses the checkpoint's
auxiliary orography, omitting its learned orography perturbation. Native
**training targets** use the frozen learned encoder. These different operations
serve different purposes and are recorded in the artifacts.

The all-field/all-level selection, uniform level reduction, initial-time
exclusion, batch size two, and fixed K comparison are explicit experiment
choices. The inference checkpoint is not evidence of original training bindings.

## Matched training settings

The matrix is 2.8° only, with width 128/256 × Mamba `d_inner` 16/32 × K 1/2,
for eight runs. Each starts with fresh zero output weights and the same seed,
training-origin order, loss-statistics artifact, input/increment statistics,
validation origins and pretrained checkpoint. Each episode resets memory.

Training uses 2015–2021, validation 2022, and reserves 2023 for test. Adam uses
beta1 0.9, beta2 0.95, epsilon 1e-6, zero weight decay, peak LR 0.002 and a
2,000-update warmup. The retained longer-run schedule stays constant to update
15,000 then halves every 10,000 updates. The present budget ends at 2,000.
Paper settings are a starting point, not evidence of an optimal residual-head LR.

The existing causal 24 h-lag/persistent forcing policy and the residual adapter's
`no_pressure_zero_mean_v2` correction restrictions remain explicit interface
choices. They are not additional NeuralGCM paper loss terms.

Fresh-process replay also pins the Python hash seed, cuBLAS workspace policy and
XLA autotuning level. The exact execution environment is part of the run identity.
The initial smoke exposed a cross-process baseline discrepancy; the detailed
report records its correction and the status of the required rerun.

Training logs all five weighted terms and field contributions. Validation uses
identical origins for frozen baseline and residual forecasts and saves
Gaussian-area physical RMSE for every field, pressure level and forecast lead.
Validation loss averages the fixed batches' objectives, including their
batch-dependent bias terms. It does not claim a single bias average over the
entire validation set. Training plots use optimizer update count on the x-axis;
physical forecast comparisons use lead/valid time.

## Code and references

- [Loss implementation](../../../src/models/neuralgcm_residual/paper_loss.py)
- [Training-only 24 h statistics](../../../src/models/neuralgcm_residual/paper_statistics.py)
- [Memory-gradient trajectory trainer](../../../src/models/neuralgcm_residual/trajectory_training.py)
- [Training entry point](../../../scripts/training/train_neuralgcm_paper_residual.py)
- [Detailed real-model tests](../../../scripts/training/smoke_neuralgcm_paper_detailed.py)
- [Production CLI and resume tests](../../../scripts/training/smoke_neuralgcm_paper_cli.py)
- [Paper training and curriculum](https://arxiv.org/html/2311.07222v3#S7.SS1)
- [Paper rescaling and losses](https://arxiv.org/html/2311.07222v3#S7.SS3)
- [Public reference metrics](../../../third_party/neuralgcm/neuralgcm/reference_code/metrics.py)
