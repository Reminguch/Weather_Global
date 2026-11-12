# Final 2,500-update MSE and power-spectrum comparison, with and without SWA

Date: 2026-11-11 (America/New_York). Status: **superseded on November 12**.

The user authorized a fresh GC–Mamba-only restart. Pending jobs
`17353172/17353173/17353174` were cancelled because they pinned the old GC–Mamba
continuations. The [replacement protocol and submitted jobs](v24_m1_matched_lambda_restart_2026-11-12.md)
combine the new M1-recipe GC–Mamba runs with unchanged full-GC `17049163`.
The new final chain is `17406897 → 17406898 → 17406899`, using
`artifacts/evaluations/v24_mixed_final_comparison_20261112/`.
The completed M1 reference comparison below remains valid. The remaining
November 11 submission details are preserved as history.

The user requested the final 2,500-update comparison after seeing the
1,250-update pilot, including both unaveraged and SWA weights. The ten fresh
scalar-lambda runs were already scheduled for 2,500 updates in two execution
parts. The first part ends at 1,250 to fit the training allocation; it is not
the scientific endpoint. At inspection, all ten second halves remain queued
for priority: full GC `17049163_0–4`, GC–Mamba `17062358_0–4`.

## Fixed comparison

- Five lambda values per family: 0, 0.25, 0.5, 0.75, 1.
- Each run contributes its raw update-2,500 checkpoint and equal-weight SWA
  of updates 1,000, 1,500, 2,000 and 2,500. The average is for inference only;
  recurrent memory starts at zero, and no optimizer state is averaged.
- Principal M1 raw and historical SWA are reused from the completed
  [November 10 pilot](v24_raw_checkpoint_pilot_2026-11-10.md). M1 raw main
  step 2,000 follows 500 warm-up updates. Its SWA main steps 500/1,000/1,500/2,000
  therefore correspond to total updates 1,000/1,500/2,000/2,500, matching the
  new average's update counts. Training recipes still differ.
- Use the same four quarterly 2022 starts and 40 six-hour leads as the pilot,
  cold full-feedback forecasts, no output smoothing, and one common public-GC
  denominator. This is a matched development diagnostic, not annual skill.
- Preserve the existing scalar-lambda training and downstream development/
  annual evaluation. The separate four-fresh-horizon full-GC study remains held.

Full GC lambda 0 uses peak LR `3e-7`, while positive lambda uses `1e-5`.
GC–Mamba uses `3e-5` throughout. The full-GC lambda comparison therefore changes
both loss weight and learning rate; SWA-versus-raw comparisons stay within each
training trajectory. Historical September/October lambda runs retain their
separate training histories and are not relabeled as fresh 2,500-update runs.

## Measurements and figures

Report the exact original GraphCast weighted grid MSE across the rollout and
at day 10, with lead curves, GC-relative reductions and within-run raw/SWA
changes. Show actual physical spherical power spectra and forecast/ERA5 power
ratios for T2m, Z500 and U850 at days 5 and 10, plus MSE-versus-power-mismatch
plots. Lambda colors are shared between raw and SWA, with distinct line styles.

The aggregate spherical power mismatch pools powers over dates first, then
averages absolute relative power errors equally across 83 channels, degrees
1–179 and leads. It describes amplitude matching, not phase alignment or the
exact GC objective. Raw truth, prediction, error and cross-power statistics
remain available for all channels and leads.

Finite loss worsening, weak improvement, power deterioration, repeatability
differences and numerical drift remain warnings. Save measurements and
reference bounds, and separately report whether decreases exceed observed
baseline repeat spread. That spread is not a full model-noise estimate and
does not gate completion. Nonfinite values and source/input/checkpoint identity
failures remain errors.

## Execution and provenance

Campaign: `artifacts/evaluations/v24_final_checkpoint_comparison_20261111/`.
The immutable plan pins the selected runs, configurations, existing midpoint
proofs, reference evaluations and frozen computational source. Future weights
are not assigned speculative hashes. After both training arrays finish, a CPU
stage validates the actual final completion records, hashes the source
checkpoints, builds ten SWAs and seals the evaluation manifest.

The dependent GPU array evaluates 20 new models. M1 raw/SWA forecasts are
reused only after checking their cohort, checkpoint and computational-source
identity. A dependent CPU report writes the combined comparison under
`plots/analyze_models/data/resolution_eval/v24_final_checkpoint_comparison_20261111/`
and the matching `images/resolution_eval/` directory. Submission is not
completion; final results require evaluation completion records and the report.

Resource evidence: the completed four-start pilots took 14m44s–15m46s per model,
maximum 8.75 GiB host RSS. Each new evaluation requests one 80GB GPU, four CPUs,
16 GiB RAM and 62 minutes; the latter avoids the documented one-hour gputest
routing boundary. Expected total execution is about five GPU-hours, excluding
queue time. Sequential CPU SWA preparation requests 8 GiB/30 minutes; reporting
requests 4 GiB/15 minutes. No extra GPU qualification or training restart.

Implementation: [preparation/submission driver](../../scripts/experiments/prepare_v24_final_checkpoint_comparison.py),
[Slurm wrapper](../../scripts/experiments/v24_final_checkpoint_comparison.slurm),
and [reporter](../../scripts/analyze_models/report_v24_raw_checkpoint_pilot.py).
Exact commands and job IDs are retained in the campaign's `submissions/`.

Submitted chain: CPU materialization **17353172**, evaluation array
**17353173_2–21**, CPU report **17353174**. Materialization depends on both
existing training arrays `17049163` and `17062358`; evaluation depends on
materialization, and reporting depends on the complete evaluation array.
The runtime digest is
`90ccacb212e0ba60c7e9e428f94cda3480c8086e371c58120a79546a6471df52`.
Eight driver/SWA tests and fifteen paired-report/reuse tests passed; all three
Slurm test-only requests passed. Synthetic full-comparison figures and real
M1-only figures were inspected. No final fresh-model result is available yet.

## Completed M1 reference comparison

A separate [M1 raw/SWA report](../../plots/analyze_models/data/resolution_eval/v24_m1_raw_swa_20261111/report.md)
reuses the two completed forecasts and provides uncluttered MSE curves, actual
power spectra, ERA5 ratios and a paired tradeoff figure. Its sealed manifest is
`artifacts/evaluations/v24_final_checkpoint_comparison_20261111/reference_only/manifest.json`.
M1 raw/SWA rollout MSE is 9.424010/9.171727; spherical power mismatch is
75.9431%/71.8861%. SWA reduces raw MSE by 2.6770% and power mismatch by
4.0570 percentage points. The loss decrease exceeds the observed baseline
repeat spread, which remains an advisory reference rather than a model-noise
estimate. Both versions still have worse aggregate power matching than public
GC (61.5708%), despite lower exact MSE. The effects vary by channel and scale.
