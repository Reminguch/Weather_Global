# r006 — Extreme-event skill of the selected corrections

Category: evaluation
Status: active
Updated: 2026-11-07 (America/New_York)
Related: [matched memory](../r001-matched-memory-training/overview.md),
[accuracy/spectral tradeoff](../r005-mgda-accuracy-spectral-adjustment/overview.md)

## Question and hypothesis

Do the principal M1 correction and the existing relative-power MGDA checkpoint
retain useful occurrence and intensity skill for hot/cold surface temperature,
strong surface wind and heavy 24-hour rain relative to frozen vanilla GraphCast?
Lower aggregate forecast MSE alone does not establish this claim.

## Current understanding

The earlier user-reported extreme results were preliminary and do not establish
preservation for principal M1. On November 6 the user explicitly authorized
submission of the full extreme evaluation. All stages were pending at the initial
queue check. During the November7 handoff inspection, [e001](experiments.md#e001)
had published a complete report for122 starts and40 leads; the report checksum
matches the completion record. Occurrence/intensity outputs are available, but
their scientific interpretation has not been completed in these notes.

The campaign uses all 122 dates in the principal 2023 evaluation, 40 six-hour
leads, and independent 1991–2020 local-calendar thresholds. It measures occurrence
over all cases, including false alarms, and reports truth-tail intensity errors
separately. It evaluates M1 and the already available relative-power MGDA SWA;
the new GC/GC–Mamba lambda-sweep checkpoints are outside this campaign.

This implements the extreme-occurrence/intensity portion of the
[earlier broad proposal](../../../docs/experiments/res1_high_frequency_extremes_evaluation_plan_2026-09-16.md),
with the now-established 122-start matched cohort. It does not implement that
proposal's 365-start campaign, FSS, derivative diagnostics, station comparisons,
or cyclone tracking. It makes no blanket noninferiority/preservation decision.
Finite numerical drift remains advisory, with its measured values retained.

## Next decision

Lianghong owns the results review and manuscript-ready figures/interpretation;
see the [handoff README](../../../docs/experiments/README_LH.md). Inspect the saved
automatic report and paired intervals before interpreting improvement,
degradation or uncertainty. Reuse the completed forecasts and thresholds. The
year has already been explored; date-bootstrap intervals do not measure
training-seed variability.

## Records

- [Attempts and evidence](experiments.md)
- [Implementation and resume](devel.md)
