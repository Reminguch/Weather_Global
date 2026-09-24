# K=1 validation loss versus training step

![K=1 validation loss reduction versus training step](k1_improvement_vs_training_step.png)

**Caption.** Relative reduction in the existing seven-field normalized MSE of
residual NGCM compared with frozen NGCM during K=1 pretraining at 2.8° resolution.
Improvement is `100 × (1 − residual_NGCM_loss / NGCM_loss)`. Training step counts
optimizer updates. Markers show initialization and completed epochs 1–5
(0, 424, 848, 1,272, 1,696, and 2,120 updates). All configurations use the same
1,455 six-hour validation forecasts from 2022 and seed 22. The baseline loss is
20,832.350542310996. **This aggregate is overwhelmingly dominated by cloud-water
terms. Full validation replay of w128/di16 and w256/di32 confirms its reduction
despite worse temperature, geopotential, wind, and humidity errors. These curves
do not establish overall forecast-skill improvement or closed-loop stability.**

[PDF](k1_improvement_vs_training_step.pdf) ·
[SVG](k1_improvement_vs_training_step.svg) ·
[CSV](k1_improvement_vs_training_step.csv) ·
[Provenance](k1_improvement_vs_training_step.provenance.json) ·
[Independent metric audit](k1_metric_audit.json)

## Metric audit

On September 24, 2026 UTC, diagnostic jobs 14349822 and 14349823 independently
reloaded the epoch-5 checkpoints, repeated all 1,455 validation records in their
original order, and reproduced the saved aggregate scores within FP32 tolerance.
Eight dates per model matched live and cached native states and forcing exactly.
An independent NumPy float64 loss reduction also agreed with the JAX scores.

The current custom objective divides squared errors by training-data six-hour
change variances per field and pressure level. Upper-level cloud-water channels
with near-zero variance use a standard-deviation floor of `1e-9`. The resulting
cloud terms dominate the aggregate. The plotted numbers are valid reductions of
this particular objective, but do not demonstrate a better weather forecast.

| Epoch-5 normalized MSE / NGCM normalized MSE | w128 / di16 | w256 / di32 |
| --- | ---: | ---: |
| Temperature | 6.06× | 8.00× |
| Geopotential | 103.53× | 186.85× |
| U wind | 1.41× | 1.56× |
| V wind | 1.31× | 1.41× |
| Specific humidity | 1.59× | 1.70× |

Ratios above one indicate degradation. These are normalized **MSE** ratios
across all pressure levels, not RMSE ratios or single-level weather scores.
The closed-loop root causes and controlled interventions are documented in the
[instability audit](../docs/experiments/neuralgcm_residual/INSTABILITY_AUDIT_20260924.md).

## Reproduce

Python and Matplotlib are sufficient. From the repository root:

```bash
python plot/plot_k1_improvement.py
```

To refresh the snapshot from the experiment's existing files:

```bash
python plot/plot_k1_improvement.py --experiment-root /path/to/experiment
```

The experiment is
`res2p8_w128_256_di16_32_train2015_2021_20260920_scatter_v3`.
The provenance file records checkpoint and validation hashes and the metrics
prefix used in the original snapshot. CSV timing columns are retained as source
metadata; the horizontal coordinate uses only `update`.
The plot has no embedded title, subtitle, caption, or endpoint annotations.
The old training-time figure has been removed.
