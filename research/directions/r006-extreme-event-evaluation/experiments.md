# Attempts and evidence

<a id="e001"></a>
## e001 — Full matched 2023 extreme occurrence and intensity evaluation

Event date: 2026-11-06
Recorded: 2026-11-06 (America/New_York)
Execution: completed (published report inspected 2026-11-07)
Outcome: pending scientific interpretation

**Authorization:** The user selected the proposed full extreme-event evaluation
and instructed “submit this.” No new training was submitted.

**Comparison:** Principal M1 `di16_fast_mamba` SWA500–2000, checkpoint hash
`b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0`, and existing
relative-power MGDA SWA1100–2000, hash
`1398d7faab0881983764a77f1a3db2b2ee9341deed9bf34a2451f46413135284`, each paired with
the same public GraphCast Small checkpoint. These are not the proposed future
fixed-lambda models. The MGDA comparison is descriptive after earlier pilot inspection.

**Setup:** All 122 stride-three-day initializations from January1 through
December30 2023, cold full-feedback forecasts with zero initial memory and alpha1,
40 six-hour leads through day10. Late-December forecasts verify into January2024.
The common dataset and rollout implementation are inherited from the frozen
November6 spectral campaign; its moments cannot recover extreme-event statistics,
so new inference is required.

**Thresholds:** Local T2m (K), derived WS10 (m/s), and TP24 (mm), using existing
1991–2020 historical data. Circular ±30-day windows at each valid UTC hour,
linear q2/q5/q95/q98, excluding February29 reference targets. Each window contains
1830 historical values. TP24 sums four preceding six-hour amounts, retaining
antecedent leap-day intervals before excluding targets. Rain scoring begins at
forecast24h and uses only forecast amounts. Negative precipitation is retained
without clipping and counted in diagnostics.

**Scores:** q98 hot/wind/rain and q2 cold, with q95/q5 sensitivity. Global and
land>=0.5 domains, exact spherical cell-area weights including finite polar caps.
Rain eligibility is historical local TP24q95>1mm for both thresholds. Save weighted
hits/misses/false alarms/correct negatives, raw counts and valid area; report CSI,
recall, precision, false-alarm ratio, frequency bias and event rates. Truth-tail
MAE/bias/RMSE and observed/predicted tail intensity remain separate diagnostics.
Undefined ratios remain undefined. Pool additive statistics before taking ratios.

Paired circular five-start blocks (nominal15days),2000 replicates,seed20261106,
provide pointwise date-sampling intervals at days1/5/10 and pooled rollout.
Exact original-GC MSE is also reported using pooled raw losses and a common M1
baseline, with each model's paired baseline and finite repeatability differences retained.

**Provenance:** [Frozen campaign](../../../artifacts/evaluations/v24_extremes_2023_20261106/campaign.json),
[source hashes](../../../artifacts/evaluations/v24_extremes_2023_20261106/source_hashes.json),
[submission commands and IDs](../../../artifacts/evaluations/v24_extremes_2023_20261106/submissions.json).
The snapshot reuses the prior spectral inference source and overlays only the new
climatology, observer, report and orchestration modules. Existing unrelated dirty
workspace changes were preserved. A first preparation attempt encountered a missing
package initializer before submission; its partial snapshot was retained under
`v24_extremes_2023_20261106_incomplete_prepare`, then preparation completed successfully.

**Submitted chain:**

| Stage | Job | Resources |
| --- | --- | --- |
| Historical thresholds | 17016908, array0–11 | Each2CPUs,8GiB,2h; no GPU |
| Threshold verification and manifest binding | 17016909 | 2CPUs,4GiB,15min; afterok thresholds |
| Paired forecast evaluation | 17016910, array0–15 | Each1gpu80,8CPUs,16GiB,62min; afterok binding |
| Automatic report | 17016911 | 2CPUs,8GiB,30min; afterok inference |

No array concurrency throttle was imposed. Initial `squeue` confirmed all four
submissions; dependent stages wait for their prerequisites.

**Validation and budgeting:**17 focused tests passed, covering calendar/leap and
precipitation boundaries, quantile computation, occurrence/false alarms, polar
weights, observer factory/input hashes and synthetic final reporting. Comparable
GPU tasks16982943_0/1/4 completed in22m48–24m22 with7.44–7.73GiB peak host RAM.
48 real source chunks loaded in0.202s. Synthetic1830x1024 quantiles took0.05144s
per tile, projecting about20min per threshold worker before I/O; its reference
and output maps total approximately3GiB. Local Zarr reads required execution
outside the tool sandbox because its asynchronous event-loop socket is blocked
there; production Slurm jobs do not use that sandbox.

**Initial observation (November6):** Pending. Submission is not evidence of completed evaluation or
scientific improvement. The report will preserve finite drift as warnings; no
variance, per-update descent or metric-based release gate is applied.

**Next action:** Lianghong reviews the completed report before making any
extreme-event preservation claim; see the update below.

### 2026-11-07 — Published report found during Lianghong handoff

The [completion record](../../../artifacts/evaluations/v24_extremes_2023_20261106/complete.json)
and [report metadata](../../../plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/report.json)
both record completion with122 paired starts and40 leads. The report SHA256
matches the completion record:
`e37a12c5e53543f65b2db76a471ad261b3ba0e508ba489c10fd44ded5b483b24`.
The [report](../../../plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/report.md),
`endpoints.csv`, `per_lead.csv`, `accuracy.csv` and figure directory are present.
This is a read-only inspection of existing outputs, not a new evaluation run or
a complete scientific review of the event metrics. It supersedes the initial
pending execution status; no blanket extreme-skill conclusion is inferred.

Lianghong is assigned occurrence/intensity review and preparation of figures and
interpretation in the [handoff README](../../../docs/experiments/README_LH.md).
