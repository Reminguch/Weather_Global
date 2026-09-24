> **Historical, superseded as the current overview.** These results use older
> checkpoints/scoring definitions, not the current five-term training.
> See the [current results and evaluation limits](../README.md).

# Offline audit of NeuralGCM-style normalization

**Changing to 24 h normalization does not resolve geopotential dominance in
these checkpoints.** This audit uses the same four existing checkpoints and
all 1,455 six-hour 2022 validation forecasts from the
[physical evaluation](../k1_physical_eval_20260924/README.md). It changes only
the scoring formula. It does not retrain models or alter running jobs.

For the representative w128/di16 checkpoint at update 2,968:

| Scoring recipe | Baseline geopotential share | Baseline 1–7 hPa geopotential share | Aggregate MSE reduction |
| --- | ---: | ---: | ---: |
| Current v2 training objective | 95.06% | 88.81% | 84.40% |
| Refit 6 h pooled scales on 60 snapshots, Gaussian statistics, current pressure weights | 93.60% | 87.45% | 80.46% |
| Refit 24 h pooled scales on the same snapshots, Gaussian statistics, current pressure weights | 93.13% | 87.00% | 79.44% |
| 24 h scales, uniform statistics, current pressure weights | 91.30% | 85.29% | 76.93% |
| 24 h scales, uniform statistics, equal-level reference reduction | 97.94% | 97.22% | 93.68% |
| 24 h scales, Gaussian statistics, equal-level reference reduction | 98.22% | 97.50% | 94.16% |

The matched 6 h / 24 h rows isolate the change interval while using the same
snapshots, estimator and reductions. Comparing either with current v2 also
changes the sample set, pooling estimator and scale floors. The two equal-level
rows additionally sum fields and apply the paper's six-hour lead-time amplitude
factor, whose square is `1 / (1 + 6/24) = 0.8`. These common scalar changes affect
absolute scores, but not percentage reductions or field shares.

For comparison, applying the **unmodified original README recipe** to the same
baseline physical errors gives geopotential **0.00048%**, cloud liquid water
**97.01%**, and cloud ice **2.99%**. Thus the original recipe is cloud-dominated,
whereas the later v2 recipe is geopotential-dominated. The 95.06% figure must
not be attributed to the original README formula. The
[three-definition training-step curves](../loss_definitions_20260924/README.md)
show this distinction for all evaluated checkpoints.

The high v2 and 24-hour percentages remain dominated by upper-atmosphere geopotential.
**Physical RMSE values are unchanged by rescoring.** Better aggregate scores
under a different formula do not mean that any forecast has improved.

## Alignment decisions and remaining differences

The user accepted temporarily omitting spatial filtering, internal model-space
loss, spectral loss and bias loss. The retained candidate is **simplified
decoded MSE using NeuralGCM-style normalization**, not the complete paper loss.

| Component | Audit implementation | Evidence or limitation |
| --- | --- | --- |
| Difference interval | Actual ERA5 differences at t+24 h minus t | Paper G.3; 6 h sampled data supplies these pairs |
| Fitted data | 60 fixed, evenly spread origins in 2015–2021 only | Paper uses 10–60 snapshots; our years and dates differ |
| Pooling | Population standard deviation over samples, levels and grid points; humidity retains its level axis | Includes between-level and between-sample mean shifts, unlike RMS of per-level standard deviations |
| Statistics latitude weights | Both uniform and Gaussian versions | G.3 does not specify quadrature; both are disclosed rather than claiming exact identity |
| Scale floors | None; nonpositive scales fail explicitly | Avoids silently retaining custom numerical floors |
| Variable amplitudes before squaring | Geopotential 2, humidity 0.66, cloud species 0.05, temperature and winds 1 | Paper G.3; internal log surface pressure is outside the retained decoded fields |
| MSE spatial reduction | Gaussian area weights, uniform longitude weights | Same physical MSEs for baseline and residual models |
| Vertical and field reduction | Test current pressure weights and the official reference default of equal-level mean and field sum | Exact paper training-loss bindings and level masks are not supplied by this audit |
| Lead-time amplitude scaling | `(1 + lead_hours / 24)^(-1/2)` in equal-level candidates | Paper G.3; all forecasts in this audit have lead 6 h |
| Overall data-loss multiplier | Report M_data without the paper's lambda_data=20 | A common multiplier cannot alter ratios or field shares when this is the only term |

Primary references: [paper G.3–G.4](https://arxiv.org/pdf/2311.07222#page=41)
and [official metric reducers](https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/reference_code/metrics_base.py).
The default reference reducer is not evidence that every training configuration
uses all 37 pressure levels with identical transforms. The complete original
training-loss bindings, statistical samples and filtering configuration still
need verification before any exact-reproduction claim.

## Origin of the current loss

The [initial project README, section 9](../../docs/experiments/neuralgcm_residual/README.md#9-objective-validation-and-final-report)
explicitly specified six-hour change standard deviations, seven-field averaging,
pressure-proportional level weights and decoded MSE. It also explicitly stated
that this was not the original NeuralGCM objective. The initial implementation
followed that recipe. The
[September 24 v2 restart](../../docs/experiments/neuralgcm_residual/FEEDBACK_V2_RESTART_20260924.md)
subsequently added RMS pooling of per-level scales and the variable amplitudes.
The running v2 loss is therefore a revision of that README recipe, not an exact
copy of the initial formula or the paper's full loss.

## Reproduction and checks

- [Training-only statistics and all 180 source-frame hashes](statistics.json)
- [All four configurations, scoring recipes and source hashes](summary.json)
- [Contributions for every recipe, field and level](contributions.csv)

Both lag statistics use the same 60 origins, spanning 2015-01-01 00 UTC through
2021-12-30 18 UTC. Every accessed source frame remains in the training split and
was verified against its prepared-data hash. No validation or test snapshot
was used to fit scales. Both models always use identical scales.

Reweighting is valid here because each retained term is a linear combination
of per-field, per-level mean squared errors. It cannot reconstruct spectral,
bias or filtered losses from these aggregates. The historical recipe reproduces
all four recorded validation objectives within floating-point tolerance.
CPU tests compare streaming moments against independent direct weighted
reductions and aggregate rescoring against direct normalized field errors.

From the repository root, use a new output directory:

```bash
python scripts/diagnostics/audit_neuralgcm_loss_alignment.py \
  --plan logs/neuralgcm_physical_eval_20260924/evaluation_plan.json \
  --output /tmp/neuralgcm_loss_alignment_new
python -m pytest -q tests/neuralgcm_residual/test_loss_alignment_audit.py
```

This is an offline CPU calculation, not a GPU smoke test, K=2 evaluation, or
production training-statistics replacement. It establishes that interval
alignment alone is insufficient; it does not certify a new training objective.
