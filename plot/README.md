# NeuralGCM residual: loss comparisons and physical forecasts

> **The original README loss, the running v2 loss, and the 24 h simplified
> loss are different definitions.** Every curve below rescores the same
> v2-trained checkpoints and the same frozen NGCM baseline. The 24 h panels
> do **not** represent a model retrained with 24 h normalization.

## Requested comparisons

All training curves use **training step (optimizer update count)** on the
horizontal axis. Every definition provides total loss, seven separate variable
contributions, and improvement over the baseline. The black dashed line is
frozen NGCM; colored lines identify the four residual configurations.

| Scoring definition | Total + seven variables | Improvement | Interpretation |
| --- | --- | --- | --- |
| **1. Original README loss** | [Loss curves](loss_definitions_20260924/readme_v1/loss_by_training_step.png) | [Improvement curves](loss_definitions_20260924/readme_v1/improvement_by_training_step.png) | Per-level 6 h scales, pressure weights, seven-field mean |
| Current training loss v2 | [Loss curves](loss_definitions_20260924/current_v2/loss_by_training_step.png) | [Improvement curves](loss_definitions_20260924/current_v2/improvement_by_training_step.png) | Pooled 6 h scales and additional variable amplitudes; this is the actual training objective |
| **2. 24 h simplified MSE** | [Loss curves](loss_definitions_20260924/normalized24_mse/loss_by_training_step.png) | [Improvement curves](loss_definitions_20260924/normalized24_mse/improvement_by_training_step.png) | NeuralGCM-style 24 h normalization with reference default reductions; still differs from the complete paper loss |

![Three definitions compared](loss_definitions_20260924/total_improvement_comparison.png)

[Complete figures, individual variable plots, checkpoint steps and validation details](loss_definitions_20260924/README.md) ·
[Loss CSV](loss_definitions_20260924/loss_curves.csv) ·
[Per-level physical MSE CSV](loss_definitions_20260924/physical_mse_by_step.csv) ·
[Source hashes and scoring coefficients](loss_definitions_20260924/provenance.json)

Loss magnitudes across definitions have different scales and are not directly
comparable. Within each definition both models use exactly the same statistics,
weights, forecast times, targets and grid. **Negative improvement means worse
than baseline.** Each variable panel includes all 37 pressure levels and shows
its contribution to total loss, not a single-level physical RMSE.

## Loss mismatch and geopotential dominance

[Exact current loss, differences from the original README, diagnosed problems,
and proposed redesign](LOSS_DEFINITION_AND_DESIGN.md)

**The 95.06% geopotential share belongs to the modified v2 loss, not the
original README formula.** Using identical baseline predictions from the full
1,455-origin K=1 audit:

| Baseline loss contribution | Original README | Current v2 |
| --- | ---: | ---: |
| Geopotential | **0.00048%** | **95.06%** |
| Cloud liquid water | **97.01%** | **0.0229%** |
| Cloud ice | **2.99%** | **0.0079%** |

The original recipe is dominated by cloud terms. The v2 revision pools
per-level scales and applies amplitudes before squaring: geopotential 2,
humidity 0.66, and cloud species 0.05. Together with large normalized
upper-atmosphere geopotential errors, these changes shift the dominant term
to geopotential. The 1–7 hPa geopotential levels alone contribute **88.81%** of
baseline v2 loss. These are empirical loss contributions, not NeuralGCM's
prescribed percentages or physical-energy shares.

The previously reported roughly **84% improvement is a reduction in this
custom v2 loss**. It does not mean that all weather variables improve by 84%.
The physical audit shows worse standard-level temperature, wind, humidity,
cloud and Z500 RMSE for those evaluated checkpoints.

**24 h normalization alone does not fix the imbalance.** A train-only
60-snapshot audit still gives geopotential **91.30%** with uniform statistical
pooling and current pressure weights. Using equal-level averaging, the default
in the official reference reducer, increases its share to **97.94%**. Neither
calculation is a claim to reproduce the complete paper objective. See the
[controlled normalization and weighting audit](loss_alignment_audit_20260924/README.md).

The percentages in this section use the earlier w128/di16 checkpoint at epoch
7, update 2,968, over all 1,455 six-hour 2022 validation forecasts. The new
training-step figures include later completed epoch checkpoints and retain
all checkpoint identities. Rescoring does not change any physical prediction.

## What the README and paper actually specify

The initial project specification is
[section 9: Objective, validation and final report](../docs/experiments/neuralgcm_residual/README.md#9-objective-validation-and-final-report).
It explicitly requests six-hour change standard deviations, pressure-proportional
level weights and an average over seven decoded fields. It explicitly calls
this a custom loss rather than the original NeuralGCM objective. The
[v2 restart](../docs/experiments/neuralgcm_residual/FEEDBACK_V2_RESTART_20260924.md)
subsequently changed pooling and variable amplitudes.

The [paper, G.3–G.4](https://arxiv.org/pdf/2311.07222#page=41) instead uses
24-hour difference scales and combines filtered MSE, model-space, spectral and
bias terms. The user has accepted temporarily omitting these additional terms
while checking the remaining normalization and reductions as closely as possible.
The 24 h candidate remains explicitly labeled **simplified data MSE**.
Exact paper statistical samples and full training-loss bindings, including
level masks, have not been reproduced. Uniform and Gaussian choices for fitting
statistics are both documented in the sensitivity audit. This is why the
24 h panel cannot be labeled the paper's complete training loss.

The 24 h interval is only the interval used to fit normalization scales from
training ERA5. It does not require a 24 h forecast; all forecasts plotted here
are K=1, six-hour predictions. Training has not been restarted with a new loss.

## Physical-unit forecasts

[ERA5 / baseline / residual three-line comparisons](k1_physical_eval_20260924/README.md)
cover all seven available pressure-level variables, including the improved
upper-atmosphere geopotential levels. Physical-time axes start from a fixed
origin and use 6 h increments. Individual PNG, PDF, SVG and CSV files are
available alongside whole-year RMSE for all 37 pressure levels.

## Earlier snapshot and reproduction

The [earlier v2 training snapshot](k1_training_snapshot_20260924.md), its
[figure](k1_improvement_vs_training_step.png),
[CSV](k1_improvement_vs_training_step.csv), and
[provenance](k1_improvement_vs_training_step.provenance.json) are retained as
history. Use the three-definition comparison above for the current analysis.

Redraw the new comparisons from the committed portable data:

```bash
python plot/plot_loss_definitions.py
```
