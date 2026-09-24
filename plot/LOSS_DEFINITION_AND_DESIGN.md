# Current loss, README differences, and proposed redesign

**Latest design direction:** [frozen NeuralGCM, public five-term loss
reconstruction, K=1 / K=2](../docs/experiments/neuralgcm_residual/NGCM_ALIGNMENT_K2_20260924.md).
This uses memory-only cross-step gradients and supersedes the equal-variable
proposal as the primary next experiment. The historical audit below is unchanged.

**Status:** the running experiment still optimizes
`neuralgcm_pooled_change_mse_v2`, using **6 h** change statistics. The 24 h
calculations and three-definition curves are offline rescoring of the same
checkpoints. No new objective has been activated or retrained in this audit.

## 1. Exact definition of the current training loss

The seven decoded pressure-level variables are temperature, geopotential,
eastward wind, northward wind, specific humidity, cloud ice, and cloud liquid
water. All 37 available pressure levels from 1 to 1000 hPa are included.

For a single forecast, let e[v,p,x] be prediction minus ERA5 in physical units.
The objective is

```text
L_v2 = (1/7) * sum_v sum_p b[p] * area_mean((a[v] * e[v,p] / c[v,p])**2)

b[p] = p / sum_p(p)
area_mean = uniform longitude mean of the Gaussian-weighted latitude sum
```

The training-only per-level scales originally fitted to six-hour ERA5 changes
are denoted s6[v,p]. The fitted manifest records 10,223 training origins.
The scales include the documented positive floors. v2 constructs

```text
c[v,p] = s6[v,p]                       for specific humidity
c[v,p] = sqrt(mean_p(s6[v,p]**2))       for every other variable
```

The pooled scale is an **RMS of per-level standard deviations**. It does not
include the variance of differences between the per-level mean changes, and
therefore is not generally the same as computing one pooled standard deviation
directly from all observations and levels.

Notation matters: the original README's `a_v=1/7` is a coefficient **outside**
the square. The `a[v]` in the v2 formula above is a separate amplitude **inside**
the square. Geopotential amplitude 2 therefore multiplies its MSE by 4, not 2.

| Variable | Amplitude a[v], before squaring | Squared multiplier |
| --- | ---: | ---: |
| Temperature | 1 | 1 |
| Geopotential | 2 | 4 |
| Eastward and northward wind | 1 | 1 |
| Specific humidity | 0.66 | 0.4356 |
| Cloud ice and cloud liquid water | 0.05 | 0.0025 |

The original scale floors are 0.1 K for temperature, 1 m²/s² for geopotential,
0.1 m/s for either wind component, 1e-7 kg/kg for specific humidity, and
1e-9 kg/kg for either cloud species. These are applied before the v2 pooling.
Units disappear after dividing by the corresponding physical-unit scales.

During K=1 pretraining, each optimizer update averages 24 chronological
six-hour examples. Each example starts its physical forecast from ERA5;
recurrent memory is carried within segments. A block of 24 examples does not
make it a 144 h physical rollout. The configured later fine-tuning stage averages
loss over 20 live six-hour predictions. The current loss uses neither the
paper's lead-time scaling nor filtered, spectral, model-space, or bias terms.

Implementation: [loss.py](../src/models/neuralgcm_residual/loss.py),
[normalization.py](../src/models/neuralgcm_residual/normalization.py),
[pretrain.py](../src/models/neuralgcm_residual/pretrain.py), and
[finetune.py](../src/models/neuralgcm_residual/finetune.py).

## 2. What the original README required, and what changed

The [original README, section 9](../docs/experiments/neuralgcm_residual/README.md#9-objective-validation-and-final-report)
specifies

```text
L_readme = (1/7) * sum_v sum_p b[p] * area_mean((e[v,p] / s6[v,p])**2)
```

It explicitly asks for per-level six-hour change statistics, pressure-proportional
weights, all seven fields, and positive scale floors. It explicitly states that
this is not the original NeuralGCM objective. The first implementation followed
that specification. The [v2 revision](../docs/experiments/neuralgcm_residual/FEEDBACK_V2_RESTART_20260924.md)
subsequently introduced pooling and amplitudes.

| Detail | Original README | Current v2 |
| --- | --- | --- |
| Temporal difference for scales | 6 h | 6 h, unchanged |
| Per-level scales | Every variable | Humidity only; other scales pooled by RMS |
| Extra variable amplitudes | None | Z=2, Q=0.66, clouds=0.05 |
| Pressure-proportional level weights | Yes | Yes, unchanged |
| Seven-field average | Yes | Yes, unchanged |
| Decoded grid MSE | Yes | Yes, unchanged |
| Complete NeuralGCM paper loss | No | No |

The current loss is a **revision of the README proposal**, not an exact copy
of the initial recipe. The fact that the initial recipe was followed does not
establish that it was well balanced for this checkpoint and dataset. The README
itself asked for a contribution audit before locking weights. Correcting cloud
dominance without checking which term replaced it was insufficient.

## 3. Measured problems with both recipes

These numbers use identical baseline forecasts and the same w128/di16 residual
checkpoint at epoch 7, update 2,968, on all 1,455 six-hour 2022 validation origins.

| Quantity | Original README | Current v2 |
| --- | ---: | ---: |
| Baseline total loss | 20,832.35 | 1.138953 |
| Residual total loss | 54,454.91 | 0.177667 |
| Baseline geopotential share | 0.00048% | 95.06% |
| Baseline cloud-liquid share | 97.01% | 0.0229% |
| Baseline cloud-ice share | 2.99% | 0.0079% |

Absolute losses in different columns are on different scales. The change in
loss reduction is a change in the scoring rule, not a change in predictions.

### Original README: tiny cloud scales create huge normalized errors

At 175 hPa, baseline cloud-liquid RMSE is about **1.67e-6 kg/kg** and the
per-level change scale is floored at **1e-9 kg/kg**. Its normalized RMSE is
about **1,670** before squaring. That single level contributes about **4,484**
to the total README baseline loss of 20,832. Similar floor-dominated levels
accumulate into nearly the entire objective. Equal field coefficients of 1/7
do not create equal field contributions.

More explicitly, that level contributes

```text
(1/7) * (175 / 15548) * (1.66994e-6 / 1e-9)**2 = approximately 4,484
```

This is **21.52% of the total loss from a single cloud-liquid level**. The
150, 125 and 100 hPa cloud-liquid scales also sit at the same floor and
contribute approximately 3,131, 2,275 and 2,119, respectively. The floor
prevents division by zero, but is still much smaller than these forecast
errors. The effective coefficient on physical squared error is
`b[p] / (7 * s6[v,p]**2)`, so equal nominal field weights cannot ensure equal
importance after normalization. These measured validation contributions do
not by themselves identify the source of the cloud forecast errors or their
training-gradient contributions.

The problem is not simply that geopotential has large physical units. Consistently
converting both errors and scales from geopotential to geopotential height leaves
their ratio unchanged. The relevant quantity is forecast error **relative to the
chosen scale**, followed by its explicit variable and level weights.

### Current v2: suppressing cloud terms exposes upper-atmosphere geopotential

v2 pools cloud scales and applies a 0.05 amplitude, removing their former
dominance. Geopotential receives amplitude 2. At 2 hPa its baseline RMSE is
**20,535 m²/s²**, while its pooled six-hour scale is **339.37 m²/s²**.
After the amplitude factor the effective denominator is **169.68 m²/s²**,
giving normalized RMSE about **121**. Its squared error can overwhelm smaller
normalized errors even after the low-pressure level weight is applied.

Geopotential at 1–7 hPa alone contributes **88.81%** of total baseline v2 loss.
The reported 84.40% aggregate reduction mostly measures reduction of this
upper-atmosphere error. Standard-level T850, Z500, winds, humidity and clouds
can worsen at the same time. See the
[physical error audit](k1_physical_eval_20260924/README.md).

These shares were measured on validation forecasts with the training loss
formula. They are not a measurement of training-set shares or gradient shares.

### Neither temporal standardization nor pressure weighting guarantees balance

Temporal change variance measures natural variation, not the frozen model's
forecast error or its representation error. A variable can vary little in ERA5
yet have a substantial model bias. Squaring their ratio can dominate loss.
Pressure-proportional weights are also a custom discrete weighting, not proof
of equal importance, pressure-layer thickness integration, or paper equivalence.

The [24 h sensitivity audit](loss_alignment_audit_20260924/README.md) confirms
that a new normalization interval alone is insufficient. With uniform statistical
pooling and current pressure weights, baseline geopotential still contributes
91.30%. Equal-level averaging increases that share to 97.94% in the simplified
candidate. The apparently larger improvement under that candidate does not
mean the forecasts became more accurate.

### Representation coverage needs an independent check

The official decoder code uses linear vertical extrapolation for temperature
and geopotential and constant extrapolation for winds and tracers when output
pressure levels fall outside the model-level centers. The 2.8° reference
configuration has 32 equally spaced sigma layers. At surface pressure 1000 hPa,
the top center corresponds to about 15.625 hPa; several of the dominant
geopotential levels lie above it.

This is a **plausible source of representation error, not a measured attribution
of the full forecast error**. The necessary diagnostic is to compare ERA5 with
the frozen encode/decode round trip at 0 h, then compare 6 h and 12 h forecast
errors using matched masks. A pressure-level output being available does not
by itself establish that every output level is equally well resolved.

Source: [official decoder](../third_party/neuralgcm/neuralgcm/legacy/decoders.py)
and [2.8° reference configuration](../third_party/neuralgcm/neuralgcm/reference_code/paper_configs/deterministic_2_8_deg.gin).
This audit has not yet measured that 0 h round-trip decomposition.

## 4. Proposed loss design

The immediate objective is an interpretable residual-correction experiment in
which an improvement in one problematic field cannot hide broad degradation.
Two separate candidates answer different scientific questions. Neither should
be silently presented as the full original paper loss.

### A. Minimal NeuralGCM-style simplified control

Use actual training-only 24 h change standard deviations, pool directly over
the intended dimensions except humidity, and use the paper's variable and
lead-time amplitudes. Use Gaussian area integration and explicitly document
the pressure-level selection and vertical reduction. Omit the additional
loss types only as the user has authorized. Keep this candidate as the
**alignment control**, including when its aggregate result is unfavorable.

Before training, verify the pressure-level coverage and masks against the
checkpoint and available official loss configuration. Do not infer the complete
paper configuration from the default metric reducer alone. Do not select a
mask to maximize validation improvement. Continue reporting every level,
including any levels excluded from the primary training score.

The current 24 h plotted candidate implements a disclosed version of this
simplification. Its severe geopotential imbalance is a reason to complete the
coverage and contribution audit before treating it as a suitable production loss.

Its exact retained term, before the paper's common data-term multiplier 20, is

```text
sigma24[v,p] = std_train(ERA5[v,p,t+24h] - ERA5[v,p,t])
               pooled directly across levels except for humidity

L_A(tau) = 1 / (1 + tau/24h)
           * sum_v mean_p area_mean((a[v] * e[v,p,tau] / sigma24[v,p])**2)
```

The plotted K=1 score uses `tau=6h`, giving a squared-error time factor of 0.8.
For a K=2 pilot scored at both forecast leads, the proposed reduction is
`(L_A(6h) + L_A(12h)) / 2`; the respective time factors are 0.8 and 2/3.
Both leads use the same **24 h training statistics**. Neither a 24 h rollout
nor changing the six-hour sampling interval is required.

Relative to v2, this plotted candidate changes the statistical sample and
pooling estimator, the difference interval, the vertical reduction, the field
reduction, and the common lead-time factor. It is not a controlled ablation of
the interval alone. The separate matched-snapshot 6 h / 24 h audit holds the
other choices fixed to isolate that interval.

### B. Equal-variable relative MSE without temporal sigma

[Standalone proposal and review questions for Ilya](EQUAL_VARIABLE_LOSS_PROPOSAL.md)

**Recommended next pilot:** use the frozen baseline's training-set physical
MSE as each variable's fixed reference. Do not use six-hour or 24-hour change
standard deviations, the v2 variable amplitudes, or an extra cloud multiplier.
All seven variables receive exactly the same coefficient, `1/7`.

For each variable v, define physical MSE E[v] with the existing Gaussian area
weights and pressure-proportional level weights. For the initial K=1 comparison,
use the existing six-hour forecast protocol:

```text
E[v] = sum_p b[p] * area_mean((prediction[v,p] - ERA5[v,p])**2)
b[p] = p / sum_p(p)

B[v] = mean over fixed training calibration origins of E_baseline[v]

L_equal = (1/7) * sum_v E_residual[v] / B[v]
```

The constants B[v] have the physical units of their variable squared, making
each ratio dimensionless. Equivalently, errors are divided by the fixed
baseline RMSE `sqrt(B[v])` before squaring. This is rescaling by the baseline's
forecast error, **not by the temporal-change sigma**. Directly averaging the
seven raw physical MSEs would still make the choice of physical units determine
their importance.

On the training calibration set, each baseline component averages exactly
one and baseline total loss is one. A 10% reduction in temperature MSE and a
10% reduction in cloud-liquid MSE each lower the objective by `0.1 / 7`, when
the other terms are fixed. These percentages are **MSE**, not RMSE, reductions.
This gives equal importance to equal changes relative to each variable's
fixed baseline reference. Individual samples and validation averages need not
have baseline score one; always score the baseline on the same evaluation set.

Keep B fixed throughout training. Do not normalize by the current model's
batch errors or recompute B from validation/test errors. Using the same error
as both numerator and denominator would make the ratio constant, or change
the intended objective if that denominator were detached. Equal scalar loss
weights also do not guarantee equal parameter-gradient norms.

Compute B on a declared training-only calibration set covering the training
years and seasons. Use the same level masks, spatial weights, forecast times,
initialization and recurrent-memory protocol for the compared models. Audit
near-zero B values before training; zero or nonfinite values must fail, and any
floor must be separately justified in that variable's squared physical units.
Do not silently introduce a common numerical floor across incompatible units.

For K=2, calibrate the declared combined score
`(E[v,6h] + E[v,12h]) / 2` on actual baseline rollouts from the training split.
Use that same reduction for the residual model, advancing from the predicted
six-hour state. An alternative is separate fixed constants for each lead with
an explicit equal-lead average. A K=1 calibration must not be presented as an
equalized K=2 or K=20 objective without recalibration for that protocol.

### Optional extension: equal pressure bands within each variable

Equal-variable scaling prevents geopotential from dominating **other variables
at calibration**, but does not prevent its upper-atmosphere levels from
dominating **its own component**. First inspect the per-level contributions
under L_equal. If pressure-band balancing is needed, predeclare physically
motivated bands and calibrate each band directly in physical units:

```text
B[v,g] = training mean of baseline physical MSE in band g
L_equal_bands = (1/7) * sum_v sum_g w[g|v] * E_residual[v,g] / B[v,g]
sum_g w[g|v] = 1
```

Band weights, within-band level reductions, and coverage must be stated before
comparing candidate improvements. Do not automatically divide by an independent
baseline error at every pressure level; nearly perfect levels can create
excessive weights. Start with variable-only rescaling while retaining the
existing level weights, so the first pilot isolates the normalization change.

This is **a custom residual-learning objective**, not a reproduction of the
NeuralGCM paper loss. Physical RMSE, bias, per-variable degradation and multi-step
stability remain necessary evaluation criteria. None of these proposed
constants has been substituted into the currently running training jobs.

### Training and selection checks

1. Audit train-only loss contributions and gradient contributions by field and
   pressure band. Balanced scalar loss does not guarantee balanced gradients.
2. Check zero-residual equivalence, normalization units, forecast-time alignment,
   and exact use of the same metric for baseline and residual predictions.
3. Evaluate per-variable physical RMSE and bias at 6 h and 12 h on a fixed
   validation set. For K=2, advance from the predicted 6 h state, with no ERA5
   reset between steps. Report the physical gradient policy separately.
4. Treat improvement only in an aggregate score as insufficient. Use declared
   per-variable degradation tolerances or a Pareto comparison, then extend the
   rollout to test stability. Do not select weights by the 2023 test results.
5. Keep numerical conservation constraints and the frozen backbone separate
   from loss changes. Compare fresh, matched residual initializations if the
   goal is to attribute a difference to the training objective.

**Recommendation:** test B's seven equally weighted relative physical MSEs
as the next residual-learning pilot, keeping the existing pressure weights for
the first comparison. Retain A as a disclosed simplified alignment control.
Complete the 0 h representation and pressure-coverage checks before attributing
upper-atmosphere errors to forecasting alone. GPU pilot tests must use
`gpu-test`, no explicit partition, and at most one hour per job.

## 5. Evidence

- [NeuralGCM paper, normalization and deterministic training objective, G.3–G.4](https://arxiv.org/pdf/2311.07222#page=41)
- [Pinned official reference metric reductions](../third_party/neuralgcm/neuralgcm/reference_code/metrics_base.py)
- [Three definitions, total and per-variable training-step curves](loss_definitions_20260924/README.md)
- [Train-only normalization and weighting audit](loss_alignment_audit_20260924/README.md)
- [Full-year physical errors and three-line forecasts](k1_physical_eval_20260924/README.md)
- [Paper training definitions, G.3–G.4](https://arxiv.org/pdf/2311.07222#page=41)
- [Official default metric reductions](https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/reference_code/metrics_base.py)
