# Attempts and evidence

<a id="e001"></a>
## e001 — Full matched 2023 extreme occurrence and intensity evaluation

Event date: 2026-11-06
Recorded: 2026-11-06 (America/New_York)
Execution: completed (published report inspected 2026-11-07)
Outcome: unsupported for preservation of day-5 primary/global CSI and recall
(November14 readout below; no blanket claim across metrics or operating points)

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

### 2026-11-14 — Canonical M1 fixed-threshold readout and scope correction

Inspected the saved [endpoints](../../../plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/endpoints.csv)
for M1, day5, primary thresholds and global area weighting. Values below are
dimensionless CSI; paired intervals refer to M1 minus raw GraphCast.

| Event | GraphCast CSI | M1 CSI | Difference | Paired pointwise 95% interval |
| --- | ---: | ---: | ---: | --- |
| Hot T2m, q98 | 0.28020 | 0.23315 | -0.04706 | [-0.05235, -0.04191] |
| Cold T2m, q2 | 0.21794 | 0.19918 | -0.01876 | [-0.02702, -0.01138] |
| WS10, q98 | 0.18132 | 0.11326 | -0.06806 | [-0.07406, -0.06202] |
| TP24, q98 | 0.19333 | 0.15178 | -0.04154 | [-0.04552, -0.03768] |

Recall differences at the same operating points are also negative, with all four
paired 95% intervals below zero. This does not support preservation of M1's
day-5 fixed-threshold occurrence skill. Conditional intensity metrics and
aggregate MSE answer different questions; neither establishes event-detection
preservation. These scores do not compare full precision–recall curves. In the
GraphCast paper ([v2, Supplement8.3](https://arxiv.org/html/2212.12794v2))
sweeps a scalar anomaly gain around climatological
medians; that procedure is distinct from Supplement7.4's optimal spatial filter.
The current experiment performs neither operation.

The user explicitly clarified that the earlier 2023 pilot was not used for
model selection. This supersedes active prior-exposure/selection framing in
earlier notes and the archived report's limitation text. The immutable e001
report and its checksum remain preserved. Date-sampling versus training-seed
uncertainty remains a valid distinction.

<a id="e002"></a>
## e002 — Matched reset/reset extension of the fixed-threshold extremes comparison

Event date: 2026-11-14
Recorded: 2026-11-14 (America/New_York)
Execution: running (regular-GPU replacement submitted; not completed)
Outcome: pending

**Authorization and question:** The user requested submission of the matched
reset/reset extreme-event comparison. Compare raw GraphCast, canonical
carry/carry M1 and separately trained reset/reset to assess memory-policy
differences under the existing fixed-threshold protocol. No spatially filtered
GraphCast, gain sweep or new training is included.

**Protocol:** Reuse e001's completed 1991–2020 thresholds, land mask and canonical
M1/GC sufficient statistics. Evaluate the original matched reset/reset
SWA500–2000 checkpoint, SHA256
`db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`,
on the identical 122 2023 starts and 40 six-hour leads. Reset both SSM and
convolution state at every autoregressive step. Preserve primary/sensitivity
thresholds, global/land domains, rain eligibility and forecast-only TP24,
additive pooling, and the same paired circular date-block intervals. The
canonical M1 branch supplies the common raw-GC reference; the reset run's GC
repeat is retained as finite-repeatability evidence, not a release gate.

**Implementation and provenance:** Campaign
[`v24_matched_reset_extremes_20261114`](../../../artifacts/evaluations/v24_matched_reset_extremes_20261114/);
[driver](../../../scripts/experiments/run_v24_matched_reset_extremes.py) and
[reporter](../../../scripts/analyze_models/report_v24_matched_reset_extremes.py).
The campaign freezes e001's computational source with orchestration/report
overlays and pins checkpoint, threshold, source and input identities. The
report will include GC, M1 and reset/reset, including direct paired M1-minus-reset
event differences. All eight shards must complete before the final report.

**Resources:** Eight shards of 15/16 starts, each one80GB GPU,8CPUs,16GiB;
dependent report2CPUs,4GiB,10min. The initial59min GPU-test request was canceled
on user instruction; the replacement uses `gpu-short` QoS and61min, which the
site routes to the regular `gpu` partition. Canonical extremes
`17016910_0–15` completed in23:04–24:18 with at most9.50GiB host RAM;
reset spectral shards `17553195_0–2` took24:33–24:55 and at most8.04GiB.
These support a roughly30–35min forecast estimate per reset-extremes shard.
The canonical report used49s/0.27GiB. No new qualification forecast is planned.

**Submission/status:** Initial forecast array `17564440_0–7` and dependent
report `17564441` were pending, then canceled after the user's instruction not
to occupy GPU-test. Active replacement forecast array is `17564547_0–7`, with
dependent report `17564548`; see the
[submission receipt](../../../artifacts/evaluations/v24_matched_reset_extremes_20261114/submissions.json)
and [scheduling override](../../../artifacts/evaluations/v24_matched_reset_extremes_20261114/submissions/scheduling_override.json).
An intermediate request with explicit `--partition=gpu` was rejected by site
policy and created no job; successful submission uses automatic partition
selection with `gpu-short`/61min. Existing reset-spectra tasks are unchanged.
The replacement queue check confirmed `17564547_0–7` pending with QoS
`gpu-short`, partition `gpu`,61min, and the CPU report pending on dependency;
the canceled GPU-test IDs were absent. The workspace driver now defaults to
this regular-GPU schedule and honors active scheduling-override receipts.
The numerical manifest and source remain pinned; the scheduling override
records this queue-only change. Submission does not establish completed
evaluation or a scientific result.

**Validation:** Eight focused tests passed (three driver, five reporter). Frozen
configurations validated all eight shard counts `[16,16,15,15,15,15,15,15]`;
canonical M1/GC loading verified 142 source pins. See
[validation receipt](../../../artifacts/evaluations/v24_matched_reset_extremes_20261114/validation.json).
Manifest SHA256 is
`3632bed5d6c71231d2aa670e82b521d62c84f07fb2ca7715b521717ab76a55c8`;
source digest is
`43e1be8b4619fb057008d2ef804020f37fc58fcb64dda2f083cec2c94978b8a5`.
The first unsubmitted preparation failed because the inherited snapshot manifest
included itself; its directory is preserved as a sibling ending `_prepare_failed`.
Excluding that legacy manifest from the new snapshot fixed preparation without
changing forecast computation. Nonfinite computation and integrity failures remain
errors; finite drift and weak skill remain recorded warnings. Next: inspect the
completed report before drawing a reset/reset extreme-event conclusion.
