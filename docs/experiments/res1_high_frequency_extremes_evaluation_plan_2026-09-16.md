# Detailed plan: res1 spatial structure and extreme-weather evaluation

Updated 2026-09-17. **Status: the canonical exact-loss subset and a zonal-spectrum pilot are submitted; global spherical-harmonic and extreme-event diagnostics remain proposed.** The frozen three-model, 365-daily-start evaluation is recorded in [its protocol](../../artifacts/evaluations/v24_res1_2023_daily_20260917/protocol.json) and [submission record](../../artifacts/evaluations/v24_res1_2023_daily_20260917/submissions.json): preflight `14077246`, gate `14077255`, production `14077262`, report `14077269`. Existing finalized ARCO data passed the bounded data preflight, so no new download was needed for this run. Numerical settings for the broader diagnostic study below remain proposed defaults to finalize using 2022 development data before examining those 2023 comparisons.

The [implemented Fourier pilot](v24_res1_spectral_pilot_2026-09-17.md) now evaluates the same three checkpoints on eight fixed **2023** starts, as requested by the user. It exports nine physical fields and records zonal power, error and correlation for eleven observables through ten days. Replacement GPU array `14080456` and automatic report `14080470` are submitted; superseded 2022 jobs `14078754` and `14078766` were cancelled with partial files preserved. This implements an exploratory diagnostic and exporter; it does not complete the full-year confirmation study below.

The objective is to determine whether our strongest res1 corrections reduce the original GraphCast loss while preserving useful spatial detail and extreme-event skill. The deliverable is a reproducible comparison against the same frozen GraphCast baseline, with uncertainty estimates and an explicit pass, degradation, or inconclusive decision for each tested property. Spatial structure, event occurrence, and event intensity are evaluated separately.

**1. Fixed experimental design**

| Item | Proposed specification |
| --- | --- |
| Models | Three completed V24 checkpoints listed below; no new checkpoint selection on the test period |
| Baseline | The matched, frozen GraphCast Small checkpoint |
| Development period | Existing 32 matched starts from 2022; extra 2022 starts only if needed for calibration or engineering |
| Primary test | All 365 daily initializations at 00 UTC, 2023-01-01 through 2023-12-31 |
| Optional replication | Daily 00 UTC starts in 2024, reserved if changes are made after inspecting 2023 |
| Forecast | Cold start, zero recurrent state, full feedback, residual alpha 1, no state reset between forecast steps |
| Inputs and leads | Two input frames, at t0−6 h and t0; 40 scored predictions at 6, 12, …, 240 h |
| Grid and targets | Native res1 grid: 181 latitudes × 360 longitudes, 13 pressure levels |
| Figures | Days 1, 3, 5, 7, 10; full six-hour curves retained |
| Primary decision leads | Days 1, 5, 10 |
| Uncertainty unit | Paired blocks of initialization dates, retaining all leads and spatial fields together |
| Claim scope | Forecast structure and extremes represented on this 1° sampled ERA5 grid |

The training manifest records 2015–2021 residual training and 2022 validation. The baseline checkpoint was trained on 1979–2015. A bounded review found no 2023/2024 use in the shortlisted configurations and experiment records; complete the provenance audit when freezing the protocol. [Current manifest](../../data/graphcast/graphcast/dataset/anchor_manifests/v24_Ilya_res1/metadata.json)

The daily-start design is chosen before seeing test results. A cheaper, month-balanced 96-start set can be used instead if the full-year design is too costly, but that decision must precede inspection of its scores. If 96 starts are merely the first batch of the 365-start experiment, retain the commitment to the full set; do not stop when results look favorable.

**2. Freeze the model and baseline identities**

| ID | Candidate | SWA window | Saved rollout-loss reduction | Saved day-10 reduction |
| --- | --- | --- | ---: | ---: |
| M1 | Batch-4 `di16_fast_mamba` | 500–2000 | 16.343% | 23.683% |
| M2 | Batch-4 `di32_joint` | 500–2000 | 16.294% | 23.321% |
| M3 | `mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k` | 2000–8000 | 16.283% | 23.211% |

These are **exact original GraphCast loss reductions on the development set**, not results of the proposed tests. They come from the [LR/di CSV](../reports/v24_mamba_lr_di_10day_2026-09-14.csv) and [optimizer CSV](../reports/v24_optimizer_search_2026-09-10.csv), with adjacent source manifests.

Checkpoint paths relative to the repository:

```text
M1 artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/swa/swa_step00500-02000.pkl
M2 artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di32_joint/swa/swa_step00500-02000.pkl
M3 artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904/mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k/swa/swa_step02000-08000.pkl

Baseline:
data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz
```

Saved baseline rollout losses vary slightly between runs (approximately 10.8355, 10.8385, and 10.8223). Treat the candidates as a near-tied shortlist. Do not infer that M1 is significantly better than M2 or M3.

Create a protocol manifest containing checkpoint/config/statistics SHA-256 hashes, code snapshot and dirty-tree patch, dependency versions, GPU type, numerical settings, evaluation modes, date manifest, masks, climatology hashes, metric definitions, and inference/analysis seeds. Include vendored `third_party/graphcast` in the snapshot. Record the exact `mean_by_level`, `stddev_by_level`, and `diffs_stddev_by_level` files; do not recompute model normalization from test data.

**3. Stage a complete and compatible test dataset**

The inspected local res1 store ends in 2022. Its source WB2 archive ends at **2023-01-10 18 UTC**, despite a description that mentions 2023. Check actual coordinates, not the year embedded in a filename. [WB2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5)

Use final ERA5 from a source covering the complete requested interval. The default acquisition route is Copernicus pressure-level and single-level data, with an alternative cloud archive allowed only after verifying dates, variables, accumulation conventions, and parity. [Pressure-level archive](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels), [single-level archive](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

| Component | Required contents |
| --- | --- |
| Six-hour store interval | 2022-12-30 00 UTC through 2024-01-10 18 UTC, inclusive |
| Raw precipitation history | Start at least one day before the first stored timestamp |
| Surface targets | T2m, U10, V10, mean sea-level pressure, six-hour total precipitation |
| Pressure targets | Temperature, geopotential, U, V, specific humidity, vertical velocity |
| Pressure levels, hPa | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |
| Static fields | Land–sea mask and surface geopotential required by the checkpoint |
| Forcings | Year/day progress and top-of-atmosphere incident solar radiation, following the existing GraphCast pipeline |
| Diagnostic auxiliary | Historical surface-pressure climatology for fixed above-ground masks, if not already available |

Implementation steps:

1. Add a fixed-date staging entry point, provisionally `scripts/stage_era5_res1_evaluation.py`, backed by a module in `src/data_operations/staging/`. Reuse validated request/conversion helpers from [fetch_era5_rolling_window.py](../../src/data_operations/download/fetch_era5_rolling_window.py), but do not use its relative-date interface unchanged.
2. Partition requests by actual calendar month for every variable family. Record request hashes and actual date ranges in filenames/manifests. Avoid constructing a cross-product of years, months, and days that downloads unintended periods.
3. Retrieve state fields at 00/06/12/18 UTC and the hourly precipitation intervals needed for six-hour totals. Validate final-ERA5 version handling rather than silently selecting the first `expver` entry.
4. Reproduce the current spatial preprocessing: the maintained res1 path selects every fourth 0.25° node. Use exact coordinate selection onto the existing 1° nodes. Direct 1° downloads or conservative regridding are different preprocessing choices and require an explicit sensitivity experiment; do not substitute them silently.
5. Form TP6(t) by summing the six hourly accumulations ending at t−5 h, …, t. Validate this convention against the source metadata and existing WB2 totals. Do not multiply one hourly accumulation by six or difference already hourly totals.
6. Keep model precipitation in metres; convert to millimetres only for diagnostic reporting. Preserve geopotential in m²/s², temperature in K, wind in m/s, humidity in kg/kg, pressure in Pa, and vertical velocity in the checkpoint's pressure-velocity convention.
7. Preserve the existing forcing calculation. When TISR is synthesized, the vendored implementation uses a one-hour integration ending at the valid timestamp, even for six-hourly forecasts. Do not replace it with a six-hour radiation total.
8. Write monthly chunks atomically into a separate evaluation store, with resumable download/conversion status. Do not append the test year into training stores or training anchor manifests.

Before accepting the store, require exact grid/level/time coverage, no duplicate or missing timestamps, finite required fields, valid units and dimensions, compatible static fields, and correct precipitation boundaries across days/months/years. Compare the new source against existing WB2 data on preselected winter/spring/summer/autumn 2022 windows and all four valid hours. Report field differences, spectra and tail-quantile differences; resolve material source discrepancies before test inference. Store tolerances and the parity report in the protocol.

The padded store contains 1,508 six-hour timestamps and 83 target planes per time: 5 surface + 6 × 13 pressure-level fields. These dimensions provide an inexpensive completeness check.

**4. Separate initialization selection from data coverage**

The current [data loader](../../src/models/mamba/v24_Ilya/data.py) first slices all data to `val_year`. This would discard necessary December input history and January verification targets.

Add an explicit initialization-manifest mode:

- Retain the padded dataset, while accepting only the listed 2023 initialization timestamps.
- Store UTC initialization time, input timestamps, all 40 valid times, ordinal, season, and a stable ID. Resolve array indices from timestamps when opening the store.
- Validate t0−6 h and every target through t0+240 h. No missing starts may be silently dropped.
- Keep the historical index-selection mode unchanged so matched32 comparisons can be reproduced.
- Do not reserve the legacy unused 24-step warmup in the new cold-start manifest. Use only the actual two-frame input and ten-day forecast requirements.
- Make per-initialization RNG streams independent of shard membership or the other starts selected, using stable UTC IDs and explicit stream labels. Do not use Python's process-dependent built-in hash.
- Record initialization-year and valid-year separately: late-December 2023 forecasts legitimately verify in January 2024.

The existing implementation already preserves RNG keys across shards of a fixed global selection. The new scheme additionally preserves them when the selection is extended. RNG behavior is not an established explanation for the historical baseline differences.

**5. Build independent climatologies and masks**

Use **1991–2020** for the main diagnostic climatology. This gives a longer reference for q98 events than seven years while remaining independent of 2022 development and 2023 testing. A 2015–2021 climatology is acceptable for code debugging only; switching the final reference requires a protocol revision before test inspection.

For each grid point and valid UTC hour, use a circular ±30-day calendar window and a frozen empirical quantile estimator (`linear` interpolation). Produce day-of-year/hour means and q2/q5/q50/q95/q98 for the required observables. Define a 365-day calendar, exclude February 29 from reference sampling, and interpolate its thresholds for an optional 2024 evaluation. Retain raw sample counts. Approximately 1,830 observations contribute per window before missing values; they are not all statistically independent.

Derive WS10 and WS850 from U/V before calculating climatology. Derive rolling TP24 from four TP6 values with the same ending-time convention as forecasts. Include boundary history for the first reference accumulation. Produce means for T850 and Q700 anomaly spectra as well.

Stage only selected levels/variables, or compute climatology in spatial tiles directly from the archived data. Avoid downloading all 83 planes for 30 years. Historical surface pressure, if used for masking, is a diagnostic auxiliary and does not alter model inputs.

Freeze:

- Exact spherical cell-area weights, including finite polar-cap areas; normalize within each valid diagnostic domain. Keep GraphCast's own latitude weighting unchanged for its canonical loss.
- Land: land–sea mask ≥0.5; ocean: <0.5; fractional-mask sensitivity optional.
- Derivative domain: |latitude| ≤85°, with all finite-difference stencil points satisfying a common climatological terrain mask. Look up historical monthly mean surface pressure for the valid month and retain points above the evaluated pressure level. This mask does not guarantee above-ground values at every valid time; add a separately labeled sensitivity using verifying surface pressure if strict instantaneous exclusion is needed.
- Rain-event eligibility: local climatological q95 of TP24 must exceed 1 mm/day. Report the excluded area and the implications for dry climates. Both q95 and q98 use this same fixed mask.
- Geographic summaries: global; NH/SH extratropics (20–85°); tropics (20°S–20°N); land/ocean. Regional/seasonal combinations are descriptive unless explicitly promoted to primary endpoints before the test.

Primary global spectral transforms use complete pressure-level fields as supplied by ERA5, including extrapolated values below terrain. Do not zero-fill terrain masks before a global transform: that creates artificial high-frequency edges. Masked physical derivatives and regional errors are reported separately.

**6. Compute the diagnostic package**

| Family | Primary observables | Required saved information |
| --- | --- | --- |
| Canonical accuracy | All GraphCast variables and levels | Per-initialization exact loss at every lead, plus rollout reduction |
| Spatial structure | T850, Q700, WS850, WS10 | Power, cross-power, error power by degree; band amplitudes, correlation and RMSE |
| Derivative observables | Relative vorticity850, horizontal temperature-gradient850 | Area-weighted squared error, signed bias where meaningful, and field second moments |
| Extreme occurrence | Hot T2m, cold T2m, strong WS10, heavy TP24 | Area-weighted hits/misses/false alarms/correct negatives, valid area, raw counts |
| Extreme intensity | Same four observables | Weighted distributions, quantile bias and explicitly conditional tail errors |
| Spatial event extent | WS10 and TP24 exceedances | FSS numerator/denominator by threshold and neighborhood radius |
| Temporal evolution | T2m, WS10, MSLP | Six-hour and 24-hour increment errors, variances and large-change counts |

For canonical accuracy, aggregate raw exact GraphCast losses over samples and leads before computing `100 × (1 − model_loss / baseline_loss)`. Never average per-lead percentage reductions to obtain the rollout percentage. Secondary metrics retain their own names and units.

Spatial implementation:

- Prefer the already installed `dinosaur.spherical_harmonic` transform, with `equiangular_with_poles` geometry, after a native-grid validation experiment. Pin the tested dependency version.
- The installed latitude quadrature has a conservative orthogonality limit around degree 90 on 181 latitudes. Therefore set the initial **confirmatory fine-scale band to 450–800 km**, degrees 50–88 using wavelength 2πR/√[ell(ell+1)]. Use degrees 20–49 for approximately 800–2000 km and 1–19 for larger scales; report degree zero separately.
- This refines the earlier 400–800 km suggestion. Treat 400–450 km and 250–400 km as exploratory until a transform supporting them passes the same reconstruction/leakage tests. Test leakage into the primary band from degrees 100–179 as well: accurate recovery of bandlimited synthetic fields alone does not establish accurate analysis of unfiltered weather fields. If this contaminates the proposed band materially, validate an alternative transform or change the band before freezing the protocol; do not assume the quadrature limit alone solves aliasing.
- Report spectra of raw fields and anomalies relative to the fixed climatology. Scalar wind-speed spectra are scalar variance spectra, not kinetic-energy spectra.
- A pole-safe vector transform and kinetic-energy spectra are optional extensions. The installed vector helper divides by cos(latitude); do not apply it naively at the poles.
- Use the transform's quadrature for spectral calculations, and report Parseval error and retained degree range. A truncated spectral sum is not the full original GraphCast loss.

For coefficients f_hat and y_hat, use consistent real-field mode multiplicities and sum over the declared modes before forming ratios:

```text
P_f = mean_cases sum_B |f_hat|²
P_y = mean_cases sum_B |y_hat|²
C_fy = mean_cases Re sum_B f_hat * conjugate(y_hat)
A = sqrt(P_f / P_y)                         # amplitude ratio
rho = C_fy / sqrt(P_f * P_y)                 # signed spectral correlation
E = P_f + P_y - 2*C_fy                       # error power
band_RMSE = sqrt(E)                         # under the frozen area normalization
```

Mark negligible truth-power cases undefined; do not stabilize them into apparently skillful values. With positive truth power and zero forecast power, amplitude is zero, log-amplitude error is unbounded, and correlation is undefined: this is not a preservation pass. To explain the effect of correction, additionally decompose error change with d = full − baseline:

```text
E_full − E_baseline
  = P_d + 2*Re mean_cases sum_B (baseline_hat − truth_hat)*conjugate(d_hat).
```

Here d is the difference between fully evolved trajectories, not the instantaneous neural residual.

Derivative implementation uses spherical distances and periodic longitude. Score both vector components of the temperature-gradient error; a gradient-magnitude plot alone loses directional information. Validate vorticity sign and units with analytic winds. Report derivative RMS ratios alongside errors to distinguish damping from improved placement.

Extreme implementation:

- Hot/cold events are truth above q98/below q2, with q95/q5 sensitivity. Wind and precipitation use q98, with q95 sensitivity.
- Apply the same historical threshold to the raw forecast and truth. Evaluate all verification pairs, including non-events, so false alarms are counted.
- Report precision, recall, CSI, false-alarm ratio, and forecast/observed event frequency, with their numerators/denominators. Undefined denominators stay undefined.
- PR curves use continuous anomaly scores with the truth-event definition fixed; they are secondary discrimination diagnostics. Any calibration is learned using development data and shown separately from raw forecasts.
- Show weighted quantile curves and tail-conditioned bias/MAE, clearly identified as intensity/distribution diagnostics rather than complete event-skill scores.
- Exact pooled quantiles require retained fields or a valid distribution representation. Do not average per-initialization quantiles and call the result the pooled quantile.
- Preserve raw negative precipitation in diagnostic exports and report its frequency/magnitude. Any nonnegative-clipped sensitivity must be labeled and cannot replace the raw result.

TP24 is available at leads 24, 30, …, 240 h. At each lead it is the sum of the preceding four forecast TP6 values, independently for baseline, model and truth. A 24-hour total is not available at lead 6 h.

For FSS, create area-weighted exceedance fractions within **geodesic circular radii 250, 500, 1000 km**; these numbers are radii, not box widths. Use the same valid cells and neighborhood normalization for all streams. Then compute:

```text
FSS = 1 − sum[w*(forecast_fraction − truth_fraction)²]
          / sum[w*(forecast_fraction² + truth_fraction²)].
```

Pool numerator and denominator before taking the ratio. Exclude undefined no-event denominators from the aggregate with counts recorded. Precompute a spherical neighborhood operator and validate dateline/polar behavior; ordinary rectangular longitude windows do not have fixed physical size.

Add SEEPS for TP24 as a secondary categorical precipitation metric, following WB2's dry/light/heavy definition. Its heavy category is not the same as a rare q98 event. [WeatherBench 2](https://arxiv.org/html/2308.15560v2)

For temporal increments, retain a four-step history, plus the t0 field. Six-hour differences begin at lead 6 h and 24-hour differences at lead 24 h. Each forecast uses its own preceding predicted state; only the initial field is shared truth. Large-change thresholds come from historical increments. This is a test of resolved temporal changes, not sub-six-hour variability.

**7. Include an explicit smoothing control**

On the 2022 development baseline, fit a non-amplifying isotropic low-pass filter by variable and lead. Define the filter family and bounded candidate strengths before fitting, include the unfiltered baseline, and select by the corresponding development error. Archive fitted parameters.

The purpose is to show how much accuracy can be obtained by damping the baseline alone. Apply the frozen filter to 2023 baseline forecasts without retuning. Report the same spatial/event diagnostics for applicable fields.

If the filter is applied only to exported diagnostic channels, report only those channels' errors; do not advertise an all-variable GraphCast-loss reduction. A full canonical-loss comparison would require filtering/scoring all target fields. No filtered forecast is fed back into the model in this control.

**8. Statistical analysis and decision rules**

Declare the following primary family for each of M1/M2/M3, at days 1/5/10:

| Property | Endpoints | Proposed practical degradation margin |
| --- | --- | --- |
| Fine-scale skill | 450–800 km RMSE and spectral correlation for T850/Q700/WS850/WS10 | +2% relative RMSE; −0.02 correlation |
| Fine-scale amplitude | Difference in absolute log amplitude-ratio error: abs(log A_model) − abs(log A_baseline) | +log(1.05) |
| Sensitive observables | Vorticity850 and gradient850 RMSE | +2% relative RMSE |
| Extreme detection | q98 hot/wind/rain and q2 cold CSI and recall | −0.02 absolute score |
| Extreme intensity | Truth-tail conditional MAE for the same four events, always accompanied by the occurrence endpoints | +2% relative MAE |
| Event footprint | q98 wind/rain FSS at 500 km radius | −0.02 absolute score |

Primary spectral endpoints use anomalies relative to the fixed climatology; raw-field spectra are secondary. This expands to 84 preservation endpoints per model, 252 for the three candidates. Include the three headline accuracy contrasts in the simultaneous inference family as well. The breadth of this family makes a blanket preservation claim demanding; report narrower supported findings when the full set is inconclusive. Conditional tail MAE tests intensity error on observed extremes, while the mandatory occurrence endpoints retain the false-alarm check.

These margins are **proposed study choices**, not established standards. Finalize their scientific meaning before test inspection; also show zero-margin results for the stricter “no worse” interpretation. Near-zero baseline errors or undefined event metrics cannot support a relative-margin decision; label the endpoint unsupported rather than choosing a favorable denominator.

Keep exact original GraphCast rollout-loss reduction as a separate headline endpoint with a paired confidence interval. To support the combined claim, its lower confidence bound must exceed zero and the prespecified preservation endpoints must satisfy their margins.

Analysis procedure:

1. Retain sufficient statistics for each initialization and lead. Resample the same dates for baseline and every model.
2. Use a paired moving-block bootstrap with a proposed 14-day block length and 10,000 replicates, seed 20260916. Preserve seasonal coverage by resampling within the five contiguous initialization-date segments Jan–Feb, Mar–May, Jun–Aug, Sep–Nov and December, retaining each segment's original number of starts. Draw contiguous blocks without wrapping December to January, concatenate and trim to the segment length. Daily ten-day forecast windows overlap; spatial cells are not independent samples. Check 10- and 21-day block lengths and an unstratified annual-block sensitivity, since segmentation breaks some dependence at seasonal boundaries.
3. Recompute ratios, square roots, correlations and FSS after resampling numerator/denominator statistics. Never bootstrap an average of already normalized per-start ratios as a substitute.
4. Orient all primary contrasts so positive means degradation. Construct simultaneous 95% lower and upper confidence bounds over the declared model × endpoint family, using a validated maximum-absolute-standardized-deviation bootstrap. Use the upper bounds for noninferiority and lower bounds for degradation. Apply consistent sign conventions to headline accuracy contrasts. Also show ordinary paired intervals for interpretation. Validate constant/degenerate endpoints explicitly.
5. Label each endpoint: supported noninferiority if its upper bound is below the margin; clear degradation if its lower bound exceeds the margin; otherwise inconclusive. Use simultaneous bounds when making a combined cross-model or cross-endpoint claim.
6. Publish the full primary scorecard. Secondary regional/seasonal/threshold analyses are descriptive unless preregistered; do not promote favorable ones after inspection.
7. Report independent event episodes as well as area-weighted exceedance counts. If too few episodes support an endpoint or most bootstrap replicates have undefined denominators, mark it inconclusive and specify what additional period would help.

Choose the final uncertainty implementation and block-length rule on 2022 data. The 14-day value is a starting protocol choice, not an assertion that dependence ends after 14 days. Do not stop sampling based on a favorable p-value.

Outcome-conditioned RMSE or MAE alone cannot establish extreme-event skill, because it excludes false alarms. This is the rationale for scoring occurrence over the whole test set. [Forecaster's Dilemma](https://arxiv.org/abs/1512.09244)

**9. Implementation changes and output contracts**

All filenames marked “new” below are proposed deliverables, not existing commands.

| Work item | Existing integration point | Planned change |
| --- | --- | --- |
| Explicit dates and protocol | [config.py](../../src/models/mamba/v24_Ilya/config.py), [data.py](../../src/models/mamba/v24_Ilya/data.py) | Optional UTC initialization manifest and padded-context loading; legacy behavior preserved |
| Streaming export | [evaluation.py](../../src/models/mamba/v24_Ilya/evaluation.py), [rollout.py](../../src/models/mamba/v24_Ilya/rollout.py) | Diagnostic consumer hooks at start/update/finish for each initialization; bounded field histories |
| Exact loss by case | [metrics.py](../../src/models/mamba/v24_Ilya/metrics.py) | Export exact per-initialization loss contributions without changing the existing objective |
| Diagnostic calculations | New `src/analysis/res1_extremes/` package | Separate modules for climatology, spectra, derivatives, events, neighborhoods, statistics and schemas |
| Data preparation | Existing data-operation helpers | New fixed-date staging/validation driver and climatology builder |
| CPU analysis | New `scripts/analyze_models/analyze_res1_extremes.py` | Read exported chunks, produce per-initialization sufficient statistics |
| Merge and reporting | Existing [V24 shard merger](../../scripts/analyze_models/merge_v24_Ilya_eval_shards.py) as reference | New diagnostic merger/report driver with strict completeness and provenance checks |
| Orchestration | New experiment driver and Slurm templates | Prepare, validate, pilot, infer, analyze, merge, report stages with explicit manifests |

The current `consume_prediction` callback already receives physical-FP32 truth, independent baseline and corrected fields together before deleting them. Rollout uses `retain_predictions=False`. Extend this path rather than introducing another forecasting implementation. Existing aggregate JSONs cannot reconstruct the new diagnostics.

Export **nine primitive diagnostic planes**: T2m, U10, V10, MSLP, TP6, T850, Q700, U850, V850. Compute WS10/WS850, TP24 and derivatives downstream. Z500 maps or more pressure levels are optional additions with their storage costs declared. Keep exact all-variable loss calculations online before unexported fields are discarded.

Use a chunked, initialization-addressable store with model ID, init_time, lead_hours, valid_time, lat, lon and variable metadata. Truth can be referenced from the immutable input store by valid time rather than duplicated for every forecast. Preserve the raw physical-FP32 exports.

For the first implementation, retain the existing paired baseline/model inference. Archive one canonical pure-baseline forecast per initialization, from a designated fixed execution path, and use it for every model's final comparison. Other paired baseline outputs are repeatability checks. Final exact loss ratios must use that same canonical baseline loss, not three different denominators.

Measure baseline repeatability on 2022 under the pinned environment and identical inputs. Set a tolerance materially smaller than the scientific margins; investigate discrepancies rather than relaxing the threshold after seeing test results. Bitwise identity is preferred where attainable. If numerical variation could change a preservation decision, repeat affected cases or report the decision as inconclusive.

A later optimization may skip the duplicated standalone baseline branch after parity tests. It cannot skip GraphCast evaluations on each model's corrected state. The existing training baseline cache is not an interchangeable forecast cache.

Per-initialization sufficient statistics should include:

```text
identity: model_id, init_time, lead_hours, protocol_hash, checkpoint_hash
exact_loss: canonical baseline and model contribution, sample weight
spectra: degree, power_forecast, power_truth, cross_power, error_power
derivatives/increments: weighted SSE, sums, second moments, valid_weight
events: threshold_id, hit/miss/false_alarm/correct_negative weights and counts,
        truth-tail absolute/signed-error sums and valid-weight denominators
FSS: radius_km, numerator, denominator, valid_weight
distributions: retained fields or declared weighted histogram representation
```

Partial output is currently progress reporting, not resumable evaluation. Implement completion records per initialization: write temporary chunks/statistics, validate them, atomically publish completion, then update a shard manifest. Restart only missing or invalid initializations. Merge must reject duplicate IDs, missing leads, mixed protocol hashes, inconsistent masks, or a changed baseline.

**10. Work sequence and completion gates**

| Phase | Work | Completion gate |
| --- | --- | --- |
| A. Freeze scope | Record M1/M2/M3 and baseline, intended date split, candidate metric family and source provenance | Reviewable protocol draft and existing-path/hash checks |
| B. Prepare data | Build fixed-date staging and reference-climatology path; source parity checks | Complete padded 2023 store plus independent climatology/masks |
| C. Build diagnostics | Date-manifest support, streaming export, CPU diagnostic modules, merge/resume | Scientific and pipeline checks below pass |
| D. Technical pilot | One 8-start 2022 pilot for each distinct architecture/configuration | Resource measurements; stable baseline; no memory growth; correct outputs |
| E. Development report | Reproduce historical matched32 and compute all new diagnostics for three models | Existing loss consistent within measured repeatability; smoothing control fitted; final bands/margins frozen |
| F. Lock protocol | Hash final config, primary endpoint expansion, thresholds, source versions and implementation | 2023 scores remain unexamined; every intended start listed |
| G. Test inference | Three candidates × 365 starts; optional staged batches with unchanged protocol | All expected model/start/lead outputs complete |
| H. CPU analysis | Spectra, events, FSS, derivatives, bootstrap, case maps | Audited sufficient-statistics merge and complete primary scorecard |
| I. Report | Tables, figures, provenance and explicit claim decisions | Results distinguish preservation, degradation, and insufficient evidence |

Data preparation and diagnostic implementation can proceed in parallel. Do the transform and exporter pilots on existing 2022 fields before waiting for the whole 2023 download. Resource planning below is for review; submission is a later execution step.

Meaningful validation checks:

- A single supported spherical harmonic has power in the expected degree; known attenuation changes amplitude predictably; phase-shifted and noisy fields demonstrate that matching power does not imply matching correlation/error.
- Native-grid transform reconstruction, quadrature normalization, Parseval consistency and spectral leakage are measured through the maximum claimed degree, including contamination from injected higher-degree modes and representative unfiltered weather fields.
- Analytic temperature and rotational wind fields verify gradient/vorticity signs, units, longitude wrapping and mask handling.
- Tiny hand-calculated contingency tables and FSS examples cover perfect forecasts, misses, false alarms, displaced events and zero denominators.
- A deliberately damped/noisy forecast is caught by the intended preservation metrics; an identity correction reproduces baseline statistics.
- Hourly-to-six-hour and six-hour-to-daily precipitation tests cover midnight, month-end and year-end; no target precipitation enters inputs prematurely.
- A January 1 initialization retains December input history; December 31 retains January targets; sharding/subsetting does not change a start's RNG.
- Sharded versus unsharded analysis gives matching pooled results; duplicate/missing shards are rejected; interruption/restart does not double-count.
- Bootstrap uses paired date blocks and recomputes nonlinear metrics; a constructed worsening case fails noninferiority and a wide-uncertainty case remains inconclusive.

Run Python and tests through `source scripts/graphcast_env.sh`. Tests should exercise these scientific and persistence properties, not merely repeat formula implementations.

**11. Compute and storage budget**

Read-only accounting/log evidence from comparable completed jobs:

| Job | Actual workload | Elapsed | Peak CPU RAM |
| --- | --- | ---: | ---: |
| `13740086_6` | Three 32-start/40-lead evaluations, including M1 SWA | 3:06:27 | 9.48 GiB |
| `13617441_3` | Three 32-start/40-lead evaluations, including M2 SWA | 3:08:01 | 9.79 GiB |
| `13435335_7` | Eight 8-start screens plus two 32-start evaluations | 4:53:45 | 9.49 GiB |

The first two totals are approximately one hour per checkpoint, not three hours per checkpoint. [M1 log](../../logs/v24_r1_b4lr_13740086_6.out), [optimizer log](../../logs/v24_r1_opt_eval_13435335_7.out)

Observed first starts took about 523 seconds and subsequent starts about 100–104 seconds. A planning model is:

```text
Existing paired evaluator minutes(N) ≈ 7 + 1.72*N
Three-model hours(N, shard_size=16)
  ≈ 3 * [7*ceil(N/16) + 1.72*N] / 60
```

The seven minutes combine compilation, reading and startup; they are not a separately measured compile time. This estimate already includes an independent baseline in each pair, plus GraphCast on corrected states. New export and diagnostic overhead are not measured yet.

| Scope, three candidates | Historical core GPU-hour reference |
| --- | ---: |
| 32 development starts, 16-start shards | ~3.5 |
| 96 starts, if selected in advance | ~10.4 |
| 365 daily test starts | ~39.5 |

Pilot request: **one GPU, eight CPUs, 16 GiB host RAM, 01:02:00**, eight starts per distinct configuration. Existing sampled GPU use was approximately 2.6–2.7 GiB on A100-80GB cards, separate from the ~10 GiB host peaks; new code needs its own measurement. The 62-minute limit also avoids the locally observed routing issue for exactly one-hour GPU jobs.

Prefer 16-start production shards if the pilot supports them: 23 shards per candidate, 69 model/shard tasks for 365 starts. The old timing predicts ~35 minutes per full shard. Choose walltime at 1.25–1.5 times the measured slow-shard estimate and RAM at approximately 1.3–1.6 times observed peak. Do not invent a fixed overhead multiplier for the new diagnostics or assume queue waiting time.

Use CPU/network stages for staging, CPU workers for transforms/FSS/bootstrap, and GPUs for inference/export. Measure one representative monthly download and one diagnostic shard before allocating those stages. Use `afterok` dependencies plus manifest completeness checks for merges. Do not add an array concurrency throttle; pack work into fewer sequential tasks if submission-count limits require it.

Uncompressed FP32 capacity estimates:

- Padded all-target input: approximately **30.4 GiB**, excluding auxiliary fields/metadata.
- Nine exported planes × 40 leads: approximately **0.0874 GiB per forecast/start**.
- Three models + one baseline + repeated truth for 365 starts: approximately **160 GiB**. Referencing truth by valid time reduces duplication, but capacity planning should not assume compression.
- Thirty years of nine selected climatology planes: approximately **103 GB** if fully materialized; tiled computation can avoid retaining this whole intermediate.
- Raw 0.25° staging is roughly 16 times the 1° spatial size for a comparable variable/time set. Process monthly batches and remove only validated, regenerable temporary downloads.

Reserve roughly **300–400 GiB** for the minimal campaign and temporary files, then revise from actual monthly/chunk sizes before bulk staging. This is a capacity estimate, not a promise about compressed size. Exporting all 83 target planes for every rollout would be much larger and is outside the default plan.

**12. Artifacts and report deliverables**

Use the existing layout, with a fresh run ID and immutable protocol hash:

```text
plots/analyze_models/data/resolution_eval/res1_high_frequency_extremes_<run_id>/
  protocol.json
  checkpoint_manifest.json
  init_manifest_2022.json
  init_manifest_2023.json
  source_snapshot/
  data_validation/
  climatology/
  forecasts/
  diagnostics/per_init/
  diagnostics/merged/
  bootstrap/
  resource_pilot/
  completion_manifest.json

plots/analyze_models/images/resolution_eval/res1_high_frequency_extremes_<run_id>/
docs/reports/res1_high_frequency_extremes_<run_id>.{md,pdf,csv,sources.json}
```

The report should include:

1. Exact original GraphCast rollout loss and lead curves, using the common baseline.
2. Fine-scale amplitude, correlation and error panels, with tested wavelength bounds and uncertainty.
3. Vorticity/gradient and temporal-increment error/amplitude panels.
4. Hot/cold/wind/rain event scorecards, tail-quantile plots and event counts.
5. FSS versus radius at each display lead, with raw-grid event scores alongside it.
6. Raw baseline, smoothed baseline and corrected-model comparisons.
7. Maps for preselected heat, cold, wind and rain episodes, using identical color scales and retaining cases where corrections hurt.
8. The full primary decision table, including inconclusive endpoints, limitations and machine-readable provenance.

Select case-study episodes from truth by a fixed, declared severity/area rule within predeclared regions, with temporal separation to avoid repeatedly showing one storm. They illustrate behavior and do not replace the annual aggregate. No case may be chosen because our model looks especially good.

Acceptance of the implementation requires complete artifacts and scientific checks. Acceptance of the scientific claim is separate: an unfavorable or inconclusive result is still a completed evaluation.

**13. Literature basis and wording of conclusions**

| Primary source | Relevance |
| --- | --- |
| [GraphCast, Lam et al. (2023), supplement §§7.4–7.5 and 8](https://arxiv.org/html/2212.12794v2) | Spectral error/power, blurring controls, severe-event diagnostics |
| [WeatherBench 2, Rasp et al. (2024)](https://arxiv.org/html/2308.15560v2) | Spectra and categorical precipitation checks complement average errors |
| [Husain et al. (2025)](https://arxiv.org/html/2407.06100v3) | Joint spectral amplitude/correlation and separation of transient structure |
| [Subich et al. (2025)](https://arxiv.org/html/2501.19374v2) | Smoothing under MSE and the distinction between amplitude and forecast skill |
| [Roberts and Lean (2008)](https://doi.org/10.1175/2007MWR2123.1) | Neighborhood fractions for spatial event verification |
| [Lerch et al. (2017)](https://arxiv.org/abs/1512.09244) | Pitfalls of scoring only observed extremes |

Useful local precedents are the older [extreme-record plotter](../../results/2026-05-23-v22/plot_extreme_records_v22.py) and [field exporter](../../scripts/training/full_mamba_v23/save_extreme_records_data_v22cl.py). Their older model paths and record-conditioned summaries are not evidence for V24 and should not be reused unchanged.

The [April design note](../../results/2026-04-04_residual_memory_design.md) says a residual connection cannot be worse than baseline. It makes zero correction representable; it does not guarantee the learned solution or every observable will improve.

If the relevant confirmation criteria pass, a bounded conclusion is: “On the prespecified 2023 evaluation, the reduction in original GraphCast loss was accompanied by preserved structure at the tested spatial scales and no material degradation in the specified extreme-event diagnostics relative to matched GraphCast forecasts.” Name the model, scales, leads, thresholds and margins. If any criterion fails, state the affected variable/lead and qualify the claim.

This design does not establish skill for gusts, individual convective cells, subgrid extremes, operational initialization, or cyclone-core intensity. Independent stations, IVT and cyclone tracking can be added as separately specified follow-up studies.
