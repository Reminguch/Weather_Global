# GraphCast-style optimal filtering on the v4 evaluation cohort

Date: 2026-11-12 (America/New_York).

The user requested replacement of the restricted smoothing comparison with the
GraphCast Supplement §7.4.2 method, fitting on 2022 and evaluating on the 2023
set used in `Overleaf/v4.tex`. This experiment compares public GraphCast_small
and the principal M1 SWA checkpoint. It does not modify model training, existing
campaigns, portable handoff archives, or the manuscript.

## Protocol

- Fit on the established 32 development initializations in 2022, at all 40
  six-hour leads. The existing manifest is copied and pinned, not regenerated
  from a new random draw.
- Evaluate the exact ordered v4 cohort: 122 initializations at 00 UTC every
  third day from January 1 through December 30, 2023; 6–240 h leads, with truth
  extending through January 9, 2024. Native grid: 181×360, 1°, 13 pressure levels,
  83 physical channels. The 2023 year has prior analysis exposure; this is an
  additional analysis, not a newly untouched test set.
- Use the same principal M1 SWA, SHA-256
  `b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0`,
  and public GraphCast_small baseline as v4. Both models follow their own raw
  trajectories. Filtering is output postprocessing and never changes feedback.
- Independently fit one unrestricted real gain for every predictor, lead,
  variable/level, and degree 0–179. Sum physical forecast power and real
  forecast/truth cross-power over fitting dates and longitudinal modes, including
  negative-order multiplicity, then divide pooled cross-power by pooled forecast
  power. There is no wind tying, lead pooling, Gaussian restriction, band
  interpolation, attenuation clamp, regularizer, or denominator floor.
- Exactly zero forecast power makes a gain unidentified; select identity and
  record its count. Negative, amplifying, or large finite gains are retained and
  diagnosed. Nonfinite values or corrupt/inconsistent provenance are errors.
- Freeze the gains before scoring any 2023 forecast. The fitted constant mode
  is allowed to change. The filtered field is `S[b A f]`, with neither unresolved
  residual restoration nor a discrete-mean correction.

The reference is [GraphCast Supplement §7.4.2](https://arxiv.org/html/2212.12794v2).
This reproduces its degree-wise least-squares filtering procedure at our
resolution, using the established normalized QR spherical transform. It does
not claim identical numerical results to their 0.25°/T719 discretization. Their
stated objective specifies no gain bounds; observed fitted gain diagnostics are
reported rather than presuming every fitted coefficient is an attenuation.

## Scores and controls

Every annual forecast produces three variants: raw native field `f`, unit-gain
projection `S[A f]`, and fitted filter `S[b A f]`. The projection control separates
truncation/reconstruction effects from learned gains. Native-grid scores compare
against the original, unfiltered truth; spectral scores compare represented
forecast coefficients with represented truth coefficients.

The primary aggregate remains the exact original GraphCast loss, with difference
normalization, exact latitude/pole-cell weights, pressure weights, and variable
weights. Canonical v4 raw GraphCast values from the eight original M1 metric
shards supply the common reduction denominator. New raw GC/M1 scores are retained
and compared with those canonical values; finite repeated-run differences and
reference bounds are warnings, not gates. Also report the direct comparison of
independently filtered M1 and independently filtered GC.

Secondary outputs include physical channel spectral MSE/RMSE, explicitly named
GC-weighted spectral loss, power diagnostics, actual mean changes, projection
residuals, and negative humidity/precipitation diagnostics. No hidden clipping
is applied. Pool squared errors across starts/leads before ratios or roots.
The report retains per-initialization values and paired five-start block-bootstrap
intervals using the existing v4 convention; these concern this fixed cohort and
checkpoints, not training-seed or between-year variability.

## Implementation and provenance

- [Core](../../src/analysis/optimal_spherical_filter.py)
- [Streaming observer](../../src/analysis/optimal_filter_observer.py)
- [Campaign driver](../../scripts/experiments/run_v24_optimal_smoothing.py)
- [Reporter](../../scripts/analyze_models/report_v24_optimal_smoothing.py)
- Campaign: `artifacts/evaluations/v24_optimal_smoothing_v4_20261112/`

The source snapshot extends the completed GC/M1 smoothing source. All model,
normalization, data loading, and forecast evaluator code is byte-identical to
that parent; explicit overlays add the new filter/observer/report/driver and an
optional observer argument to the old dispatcher. The actual v4 recovery
evaluator differs from that later observer-capable evaluator, so canonical v4
metrics are preserved separately and repeated-run differences remain visible.
The source digest, checkpoint/data/manifests, a copy of v4, and copies of its
eight M1 metric shards are pinned in the campaign manifest.

Existing 2022 smoothing statistics pooled four lead groups and four basis
functions, so they cannot determine the new degree-wise optimum. Existing 2023
power/cross-power summaries could give spectral-only results, but lack the full
coefficients or grid fields needed to score arbitrary filtered native fields.
Fresh forecast passes are therefore required for the requested exact grid
evaluation. No new training is needed.

## Resources and execution

The authorized chain is four development shards (eight starts each), one CPU
fit, eight evaluation shards (15–16 starts each), then the CPU report. Arrays
have no concurrency throttle. Each GPU task requests one 80 GB GPU, four CPUs,
20 GiB host RAM, and 62 minutes for development or 90 minutes for evaluation.
Previous development shards took 30–33 minutes for 10–11 starts. Completed
annual tasks `17028677_1–7` took 43–47 minutes for 17–18 starts and approximately
9.9–10.3 GiB RSS. These support the new requests without another qualification
job. Finite loss increases or unfavorable filter results do not cancel scoring.

Prepare from the workspace, then submit using the prepared immutable source:

```bash
source scripts/graphcast_env.sh
python scripts/experiments/run_v24_optimal_smoothing.py prepare \
  --root artifacts/evaluations/v24_optimal_smoothing_v4_20261112
python artifacts/evaluations/v24_optimal_smoothing_v4_20261112/source/scripts/experiments/run_v24_optimal_smoothing.py \
  submit --root artifacts/evaluations/v24_optimal_smoothing_v4_20261112
```

Submission receipts live under `submissions/`. A successful submission is not
an evaluation result. `report.json` with verified complete annual inputs is the
completion evidence; `REPORT.md`, CSV tables, and summary figures provide the
readable evaluation. Execution IDs and validation are recorded below after
actual submission.

## Submitted execution

Submitted chain: **17430308_0–3 → 17430309 → 17430310_0–7 → 17430311**.
All jobs were pending at the November 12 post-submission inspection. No fitted
filter or 2023 scientific result exists yet. Dependencies wait for every shard.

Manifest hash:
`b69ebc7e623b2a2bab0645a6d020f8807645f70b5eadc76dced61b3e116449af`.
Source digest:
`a470b86a6936c744308e1bfabbd08fe3ceabd5411019d76f7830947ce6033fa6`.

Validation: 17 focused tests passed (core 8, observer 3, campaign 3, report 3),
actual PNG/PDF rendering passed, and four Slurm test-only requests passed.
The production 181×360 T179 geometry check used real T2m/MSLP fields from both
datasets: all outputs finite, maximum QR condition 2.332, coefficient roundtrip
relative error 4.1e-9–2.7e-8. Measured projection residual and mean changes are
retained in the [validation evidence](../../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/checks/validation.json).
Local xarray/Zarr async opening stalled during a read-only probe; direct chunk
decoding succeeded, and the proven production loader was left unchanged.

Monitor with `squeue -j 17430308,17430309,17430310,17430311`.
The final job writes `REPORT.md`, `report.json`, `summary.csv`, `perlead.csv`,
`physical_channel_spectral.csv`, `per_initialization.csv`, `annual_pooled.npz`,
and `summary.png`/`summary.pdf` in the campaign directory after successful
evaluation. Compare each observed loss decrease with the retained raw rerun
drift descriptively; finite drift never blocks completion or filter scoring.

## Interim result — November 13, 2026

All 32 development starts and the 2022 filter fit completed. Annual shards
0–5 completed: 92 matched starts, January 1–October 1, at all 40 leads.
Shards 6/7 remain queued for October 4–December 30 (30 starts); the automatic
annual report remains dependent. No completed worker failed or emitted an
observer/evaluator precision warning.

The [interim report](../../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/interim_20261113/REPORT.md)
and its sealed JSON preserve this incomplete subset separately from the annual
completion artifacts. Relative to canonical v4 raw GC on these same dates:

| Model/output | Rollout exact GC loss reduction | Day-10 reduction |
| --- | ---: | ---: |
| GC, optimal filter | 10.9123% | 18.6650% |
| M1, raw | 14.7112% | 22.4317% |
| M1, optimal filter | 15.3533% | 25.4324% |

Directly comparing the independently filtered models, M1 has 4.9850% lower
rollout loss (paired 95% interval 4.1243–5.9559%) and 8.3204% lower day-10 loss
(6.9930–10.0064%). Filtering lowers each model's own raw rollout loss by
10.8808% for GC and 0.7528% for M1. Raw M1 still improves on filtered GC by
4.2643% over the rollout. These are interim results, not the full 2023 sample.

Aggregate repeated raw losses differ from canonical v4 by −0.0353% for GC and
+0.0382% for M1; both filtering decreases exceed those measured aggregate
differences. Per-initialization/lead maxima are 3.9907% and 1.7923%, respectively,
and are retained as finite warnings. A single rerun difference is not a full
noise distribution or an acceptance threshold. Unit-gain projection alone
worsens native-grid loss; the gains arise from fitted spectral weights.

All 1,195,200 fitted gains are finite: 18,014 are negative, 93,153 exceed one,
and none has magnitude above ten. This is unrestricted optimal spherical
filtering, not attenuation-only smoothing. Fit ranges and complete provenance
are in the interim JSON.

## Joint MSE and power comparison — November 13, 2026

The user explicitly requested power spectra alongside MSE. The saved physical
statistics support this without rerunning forecasts or refitting filters.
The same 92 dates and 40 leads give:

| Forecast | Exact GC rollout loss ↓ | Rollout power error (%) ↓ | Day-10 power error (%) ↓ |
| --- | ---: | ---: | ---: |
| GC raw | 11.448889 | 59.4141 | 65.6292 |
| GC optimal filter | 10.203157 | 78.3995 | 92.1136 |
| M1 raw | 9.768064 | 67.5267 | 84.1939 |
| M1 optimal filter | 9.694526 | 77.6371 | 90.5157 |

Power error is `100 × mean |P_forecast/P_truth − 1|`, pooling power over dates
before taking ratios, then equally weighting 83 physical channels, positive
degrees 1–179, and leads. Degree zero is excluded; no climatological mean is
removed. This measures amplitude fidelity under the stated weighting, not
phase accuracy or the fraction of total atmospheric energy in error.

Filtering improves MSE but worsens aggregate power matching for both models
at every lead. Raw M1 beats filtered GC on both aggregate measures at every
lead. Raw M1 versus raw GC is a tradeoff: better MSE but worse power matching.
Filtered M1 has lower MSE and slightly lower power error than filtered GC;
the spectral difference is a point estimate without a confidence interval.
These comparisons do not imply an advantage at every channel or degree.

The [joint report and metric definitions](../../plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113/REPORT.md)
link degree curves at days 1/3/7/10, leadwise power errors, and an MSE/power
scatter plot. The new reporter verifies source/filter/date provenance and
reproduces the preceding exact MSE values. Raw and pure projection controls
have identical represented power, so the plots show four distinct curves.

Full-year power reporting is queued as **17470053**, dependent on successful
annual report **17430311**. This CPU task requests 8 GiB, two CPUs and 20 minutes
for archive pooling and plotting; Slurm test-only passed. Its independently
frozen source digest is
`e3226c6153d824c1fa583d02689dee85e50b11db1673ec38d189c025620ab9dc`.
Provenance and submission receipts are under campaign
`power_reporting_20261113/`; full-cohort plots/tables will use the name
`v24_optimal_smoothing_v4_2023_power_20261113`. This follow-up requires a sealed
122-start annual report. The existing forecast/fit snapshot is unchanged.
