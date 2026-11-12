# 2023 extreme-event evaluation

## A2 percentile thresholds

Public GraphCast Small and the local seed-22 GraphCast–Mamba carry/carry replay each ran their own closed-loop forecasts. The Mamba checkpoint is the raw main-stage step 2,000 after 500 warm-up updates. Both models were evaluated on the same 122 starts at 00 UTC every three days from January 1 to December 30, 2023, with 40 six-hour leads on the native 1° grid and the quality-controlled ERA5 truth. The manifests preserve checkpoint, input and frozen evaluator identities with SHA-256 digests.

- [Endpoint table](endpoints.csv) covers 24, 120 and 240 hours. [Per-lead table](per_lead.csv) covers all eligible leads. Both give public GC and raw M1 scores, M1-minus-GC differences and pointwise 95% intervals.
- [Summary](summary.json) documents the scoring protocol and main endpoint rows. [A2 figure PDF](figE1_m1_raw2k_vs_gc_per_lead.pdf) and [PNG](figE1_m1_raw2k_vs_gc_per_lead.png) show both models under the primary thresholds.
- [Per-start sufficient statistics](sufficient_statistics.npz) contain additive areas, counts and error sums. The array axes are initialization, event, threshold set, region, model, lead and statistic. The accompanying vectors name all axes, and the model labels `baseline` and `full` mean public GC and raw M1 respectively. Rain statistics begin at 24 hours.

The ERA5 1991–2020 climatology pools ±30 calendar days separately by UTC time at each grid cell and omits February 29. Primary thresholds are q98 for heat, wind and rainfall and q2 for cold. Sensitivity uses q95/q5. TP24 sums four forecast increments and applies a climatological q95 > 1 mm eligibility threshold. Scores use spherical grid-cell areas for global and land samples. Paired pointwise intervals use 2,000 circular moving-block resamples of five starts, with seed 20261106.

At 240 hours under the global primary thresholds, GC-to-raw CSI changed from 0.160 to 0.108 for heat, 0.045 to 0.010 for wind and 0.046 to 0.021 for rain. Their paired difference intervals lie below zero. Cold CSI changed from 0.068 to 0.058, with a difference interval that includes zero. The declines for heat, wind and rain occur in the raw checkpoint before any weight averaging.

## Record exceedance figures

The [24-hour PDF](record_exceedance_24h.pdf) and [240-hour PDF](record_exceedance_240h.pdf) use ERA5 2015–2021 monthly records at each grid cell. Heat, cold and wind RMSE and signed bias are evaluated on the same truth-selected record-breaking grid-cell and start pairs, with equal weight per pair. The [record methods](README_record_exceedance_24h.md) describe the records, bins, dates and models. The [bin data](record_exceedance_bins.csv) give the plotted counts, RMSE and bias, plus additive signed and squared error sums. Empty bins have blank score fields.

## Provenance and outstanding comparisons

The [campaign manifest](provenance/manifest.json), [forecast identity](provenance/forecast_cache_manifest.json), [climatology metadata](provenance/climatology_metadata.json), [ERA5 truth audit](provenance/native_truth_audit.json) and [initialization manifest](provenance/initializations_2023_stride3.json) identify inputs. The provenance directory preserves the local scorer and methods source as references. It contains metadata and compact outputs for review, while the linked campaign paths refer to source-machine inputs and caches.

The historical Della raw and SWA checkpoints, frozen scoring inputs and full A output tables were unavailable locally. A2 uses a local recipe replay, reconstructed climatology and local scorer. The historical SWA E1/E1b tables and combined plot, the paired SWA-versus-raw extremes comparison, and the eight-model S128 K scan with E2/E3 remain outstanding.
