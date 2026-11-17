# r006 — Extreme-event skill of the selected corrections

Category: evaluation
Status: active
Updated: 2026-11-14 (America/New_York)
Related: [matched memory](../r001-matched-memory-training/overview.md),
[accuracy/spectral tradeoff](../r005-mgda-accuracy-spectral-adjustment/overview.md)

## Question and hypothesis

Do the principal M1 correction and the existing relative-power MGDA checkpoint
retain useful occurrence and intensity skill for hot/cold surface temperature,
strong surface wind and heavy 24-hour rain relative to frozen vanilla GraphCast?
Lower aggregate forecast MSE alone does not establish this claim.
The current extension asks how the separately trained reset/reset control
compares with carry/carry M1 on the identical event protocol.

## Current understanding

The completed [e001](experiments.md#e001) has 122 starts and 40 leads. Inspection
of its canonical M1 endpoints on November14 finds worse day-5 primary-threshold
global CSI and recall than raw GraphCast for all four events, with paired
pointwise 95% difference intervals entirely below zero. CSI for GC/M1 is
hot 0.28020/0.23315, cold 0.21794/0.19918, wind 0.18132/0.11326 and
rain 0.19333/0.15178. This is negative evidence for preservation of occurrence
skill at those operating points, despite M1's lower aggregate forecast loss.
It does not establish dominance of either full precision–recall curve.

The campaign uses all 122 dates in the principal 2023 evaluation, 40 six-hour
leads, and independent 1991–2020 local-calendar thresholds. It measures occurrence
over all cases, including false alarms, and reports truth-tail intensity errors
separately. It evaluates M1 and the already available relative-power MGDA SWA;
the new GC/GC–Mamba lambda-sweep checkpoints are outside this campaign.

The user authorized [e002](experiments.md#e002) to add the original matched
reset/reset SWA500–2000 checkpoint. The initial GPU-test array `17564440` and
report `17564441` were canceled while pending when the user instructed that
this campaign must not occupy GPU-test. Replacement forecast array `17564547_0–7`
and report `17564548` are submitted with regular-GPU routing (`gpu-short`,61min).
The replacement queue check confirmed all eight forecasts pending on partition
`gpu` and the CPU report waiting on dependency; the old GPU-test jobs are absent.
It reuses the completed thresholds, land mask and canonical M1/GC sufficient
statistics, and evaluates reset/reset on the same 122 starts and 40 leads.
Both SSM and convolution state reset every autoregressive step. This comparison
uses raw forecasts and fixed climatological thresholds; spatial filtering and
gain-swept precision–recall curves are outside its scope. Eight focused tests
passed. Numerical source, thresholds and cohort are unchanged by the scheduling
correction; no reset extreme-event result exists yet.

This implements the extreme-occurrence/intensity portion of the
[earlier broad proposal](../../../docs/experiments/res1_high_frequency_extremes_evaluation_plan_2026-09-16.md),
with the now-established 122-start matched cohort. It does not implement that
proposal's 365-start campaign, FSS, derivative diagnostics, station comparisons,
or cyclone tracking. It makes no blanket noninferiority/preservation decision.
Finite numerical drift remains advisory, with its measured values retained.

## Next decision

Inspect e002's automatic three-model report after the queued evaluation and
paired M1-minus-reset intervals before attributing an extreme-event difference
to memory policy. Reuse the completed canonical evidence. Date-bootstrap
intervals describe forecast-sample uncertainty for fixed checkpoints, not
training-seed variability. The user clarified that the earlier 2023 pilot did
not inform model selection; do not treat prior pilot evaluation as evidence of
a selection problem. The [handoff README](../../../docs/experiments/README_LH.md)
retains the broader ownership record; no manuscript edit is part of this run.

## Records

- [Attempts and evidence](experiments.md)
- [Implementation and resume](devel.md)
