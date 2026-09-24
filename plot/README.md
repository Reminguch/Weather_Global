# K=1 improvement versus training step

> **Important: the current training loss differs from the NeuralGCM paper.**
> The custom objective is dominated by geopotential: it contributes **95.06%**
> of baseline loss, and geopotential at **1–7 hPa alone contributes 88.81%**
> of the total. The approximately **84% improvement below is a reduction in
> this custom loss**, not an 84% improvement across weather variables.
> Full-year physical evaluation shows that T850, Z500, wind, humidity, and cloud
> errors at the reported standard levels worsen for the evaluated checkpoints.

![K=1 improvement over NGCM versus training step](k1_improvement_vs_training_step.png)

**Caption.** Relative reduction in the `neuralgcm_pooled_change_mse_v2`
validation objective compared with frozen NGCM during K=1 pretraining at 2.8°.
Improvement is `100 × (1 − residual_NGCM_loss / NGCM_loss)`.
**Training step counts optimizer updates**, as requested for training plots.
Markers show initialization and completed epochs, with 424 updates per epoch.
The inset enlarges the post-initialization points using the same units on both
axes. All configurations use the same 1,455 six-hour validation forecasts from
2022 and seed 22. The baseline loss is **1.1389527632198793**.
Only checkpoints with matching completed-validation receipts enter the snapshot.
Training is ongoing, so each curve ends at its latest completed validation.

[PDF](k1_improvement_vs_training_step.pdf) ·
[SVG](k1_improvement_vs_training_step.svg) ·
[CSV data](k1_improvement_vs_training_step.csv) ·
[Provenance and rollout context](k1_improvement_vs_training_step.provenance.json)

**Physical-variable follow-up:** [six-hour predictions versus ERA5 and NGCM](k1_physical_eval_20260924/README.md)
now provides all seven variables, fixed-start time axes in 6 h increments,
and full-year physical-unit errors. The aggregate improvement is concentrated
in upper-atmosphere geopotential; standard-level forecast errors can worsen.

## Loss mismatch and geopotential dominance

The current objective, `neuralgcm_pooled_change_mse_v2`, is **not the original
NeuralGCM training loss**. Its main differences are:

| Component | Current residual training | NeuralGCM paper |
| --- | --- | --- |
| Normalization interval | Standard deviations of **6 h** ERA5 changes | Standard deviations of **24 h** ERA5 changes |
| Pooling | RMS of per-level standard deviations, except humidity | Standard deviation over longitude, latitude, level, and sample, except humidity, which is normalized per level |
| Loss terms | Decoded pressure-level weighted MSE only | Filtered MSE and spectral loss in both data and model representations, plus bias loss |
| Variable factors | Borrows geopotential 2, humidity 0.66, and cloud species 0.05 as amplitude factors before squaring | These factors accompany the paper's normalization and full objective; internal log surface pressure also has factor 5 |

See [NeuralGCM G.3–G.4](https://arxiv.org/pdf/2311.07222#page=41).
The current implementation additionally uses pressure-proportional vertical
weights. These choices have not been established as a reproduction of the
paper's complete weighting and aggregation.

The **95.06% / 88.81%** shares come from the paired physical audit of
`w128/di16`, epoch 7, training step 2,968, over all **1,455** six-hour forecasts
in the 2022 validation set. This audited checkpoint is later than the w128
snapshot in the training plot. They are measured contributions to our
custom objective, **not weights prescribed by NeuralGCM** and not physical
energy fractions. The strong aggregate improvement is concentrated in
upper-atmosphere geopotential and masks worse errors in other variables.

The loss mismatch and imbalance are established; **changing 6 h normalization
to 24 h has not yet been shown to remove the imbalance**. The relative effect
of normalization, vertical weighting, and the baseline's pressure-level errors
requires separate measurement. The existing plots and checkpoints still use
the current custom objective.

For the next alignment, the user has accepted a **simplified decoded MSE**,
temporarily omitting filtering, model-space, spectral, and bias terms. Its
retained normalization, variable factors, spatial/vertical reductions, and
lead-time scaling must be checked against the paper and official reference
code. Any revised metric must score baseline and residual predictions with
the **same training-only statistics and evaluation protocol**. It must remain
labeled a simplified objective, rather than the paper's full training loss.

[Measured contributions by variable and pressure](k1_physical_eval_20260924/objective_contributions.csv) ·
[Physical RMSE for all seven fields and 37 levels](k1_physical_eval_20260924/physical_metrics_all_levels.csv) ·
[Geopotential loss breakdown](k1_physical_eval_20260924/r2p8_w128_di16/geopotential_loss_breakdown.png)

## Current snapshot

Captured on **September 24, 2026 at 12:44:00 UTC / 08:44:00 EDT** from
`res2p8_w128_256_di16_32_train2015_2021_20260924_feedback_v2`.

| Configuration | Completed epoch | Training step | K=1 validation loss | Improvement |
| --- | ---: | ---: | ---: | ---: |
| w128 / di16 | 6 | 2,544 | 0.179594 | 84.23% |
| w128 / di32 | 6 | 2,544 | 0.180016 | 84.19% |
| w256 / di16 | 5 | 2,120 | 0.178896 | 84.29% |
| w256 / di32 | 5 | 2,120 | 0.178983 | 84.29% |

## Interpretation and multi-step validation

These are **single-step reductions of the new custom objective**, not evidence
of improved five-day forecast skill. At epoch 5, all four models completed all
32 cold and warm validation origins, but their five-day rollout losses were
substantially worse than frozen NGCM.

| Configuration | Cold five-day mean loss | Ratio to NGCM |
| --- | ---: | ---: |
| w128 / di16 | 123.950 | 85.69× |
| w128 / di32 | 123.083 | 85.09× |
| w256 / di16 | 100.314 | 69.35× |
| w256 / di32 | 102.711 | 71.01× |

The common five-day baseline loss is 1.446498353779316. The provenance file
includes the corresponding cold/warm summaries and their source hashes.
Formal K=20 fine-tuning had not started at snapshot time.

This snapshot replaces the previous `20260920_scatter_v3` data. The restart
uses `no_pressure_zero_mean_v2` corrections and a revised loss with pooled change
scales and field weights. Its percentages must not be compared directly with
those of the previous objective. The old metric-audit file was removed from this
directory because it evaluates the superseded loss and checkpoints. Historical
data remain in Git history. See the
[restart description](../docs/experiments/neuralgcm_residual/FEEDBACK_V2_RESTART_20260924.md)
for the current contracts and the
[previous instability audit](../docs/experiments/neuralgcm_residual/INSTABILITY_AUDIT_20260924.md)
for the earlier diagnosis.

## Reproduce

Python and Matplotlib are sufficient. From the repository root, redraw all three
figure formats from the committed CSV without access to the experiment files.

```bash
python plot/plot_k1_improvement.py
```

Refresh CSV, provenance, and figures from the current experiment with

```bash
python plot/plot_k1_improvement.py --experiment-root /path/to/20260924_feedback_v2_experiment
```

When publishing a later snapshot, update this README's timestamp and tables too.
The provenance records the source identity, loss/correction contracts, validation
and checkpoint hashes, completion receipts, and the exact metrics prefix.
CSV timing columns are retained as source metadata; the horizontal coordinate
uses only `update`. The main plot has no embedded title, subtitle, caption, or
endpoint annotations.
