# Development — 2023 extreme-event evaluation

Updated: 2026-11-14 (America/New_York)

## Current extension: matched reset/reset (e002)

The November14 campaign
[`v24_matched_reset_extremes_20261114`](../../../artifacts/evaluations/v24_matched_reset_extremes_20261114/)
initially used forecast array `17564440_0–7` and report `17564441`; both were
canceled while pending after the user instructed that this campaign must not
occupy GPU-test. Active replacement array is `17564547_0–7`, with report
`17564548`. Existing GPU-test spectral work is unchanged.
Queue verification found all eight forecasts pending on partition `gpu` with
QoS `gpu-short` and61min; the CPU report awaited dependency. Old GPU-test IDs
were absent.

- [Driver](../../../scripts/experiments/run_v24_matched_reset_extremes.py):
  freezes canonical November6 computation with orchestration/report overlays,
  binds original reset SWA500–2000 and reuses the identical 122 starts, 40 leads,
  historical thresholds and land mask. Both SSM and convolution state reset at
  every autoregressive step. Checkpoint SHA256:
  `db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`.
- [Reporter](../../../scripts/analyze_models/report_v24_matched_reset_extremes.py):
  reuses canonical M1 and its GC branch, adds reset/reset and direct paired
  M1-minus-reset scores, checks common truth/exposure, and preserves finite GC
  repeatability differences as warnings. Scope is fixed thresholds without
  spatial filters or gain-swept precision–recall curves.
- Eight15/16-date replacement tasks use `gpu-short` QoS and
  1gpu80/8CPUs/16GiB/61min each, routed automatically to regular `gpu`; the report
  uses2CPUs/4GiB/10min. Eight focused tests passed, all frozen shard configurations
  validated and142 canonical source pins checked; see
  [validation.json](../../../artifacts/evaluations/v24_matched_reset_extremes_20261114/validation.json).

Resume from `manifest.json`, `source/snapshot.json`, `submissions.json` and
`submissions/*.json` in the campaign. Inspect receipts and Slurm before recovery;
do not edit frozen files or duplicate forecasts. The failed unsubmitted first
snapshot remains in the sibling `_prepare_failed` directory. Its legacy snapshot
manifest was excluded from the final source list, with computation unchanged.

The original numerical manifest remains unchanged; inspect the
[scheduling override](../../../artifacts/evaluations/v24_matched_reset_extremes_20261114/submissions/scheduling_override.json)
when resuming, rather than rerunning the frozen default GPU-test submit command.
The site rejects explicit partition flags; use `gpu-short`/61min and automatic
regular-GPU routing. The rejected explicit-partition request created no job.
The current workspace driver defaults to `gpu-short`/61min and recognizes the
active scheduling-override receipts. That orchestration change does not alter
the frozen numerical source used by worker/report jobs.

```bash
squeue -j 17564547,17564548
sacct -j 17564547,17564548 --format=JobID,State,Elapsed,MaxRSS -P
```

Outputs publish under matching
`plots/analyze_models/{data,images}/resolution_eval/v24_matched_reset_extremes_20261114/`.
Inspect completion and report evidence before claiming results. The existing M1
day-5 primary/global CSI and recall worsen versus GC; [e001](experiments.md#e001)
records that fixed-threshold result. The user clarified that the earlier 2023
pilot did not inform model selection; new reports omit the archived report's
stale prior-exposure limitation.

## Original campaign implementation (e001)

- `src/analysis/extremes_climatology.py`: twelve CPU workers for three observables
  by four hours. Historical reference maps are reusable after checksum validation;
  completed threshold arrays are published atomically. Finalization verifies all
  outputs and produces a hashed metadata manifest.
- `src/analysis/extremes_observer.py`: streaming observer using the maintained
  evaluator hook. Saves per-initialization occurrence/intensity sufficient
  statistics with all model, threshold and mask provenance.
- `scripts/analyze_models/report_v24_extremes_2023.py`: complete-coverage checks,
  paired block intervals, CSV/JSON/Markdown and PNG/PDF figures.
- `scripts/experiments/run_v24_extremes_2023.py` and adjacent Slurm wrapper:
  preparation, threshold jobs, immutable manifest binding, GPU evaluation and report.

## Original campaign validation

Seventeen tests passed:

```bash
source scripts/graphcast_env.sh
JAX_PLATFORMS=cpu python -m pytest -q tests/v24_Ilya/test_extremes_climatology.py tests/test_extremes_observer.py tests/v24_Ilya/test_extremes_report.py
```

Real metadata/grid/time coverage and bounded historical-field reads passed.
November7 handoff inspection found the completed production report for122 starts
and40 leads, with its checksum matching `complete.json`. The November14 readout
reviews canonical M1 occurrence endpoints; see [e001](experiments.md#e001) and the
[handoff README](../../../docs/experiments/README_LH.md).
No training code or live prior jobs were modified.

## Original campaign archive

Campaign on Della:
`/scratch/gpfs/DABANIN/iv9432/Weather_global/artifacts/evaluations/v24_extremes_2023_20261106`.

```bash
squeue -j 17016908,17016909,17016910,17016911
sacct -j 17016908,17016909,17016910,17016911 --format=JobID,State,Elapsed,MaxRSS -P
```

Thresholds precede binding; inference uses bound manifests under `shards/00`–`07`.
Do not edit frozen source or bound manifests in place. Check `submissions.json`
before any recovery to avoid duplicate forecasts. The submit command intentionally
refuses an existing submission record. Preserve partial outputs for diagnosis.

The report publishes to
`plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/` and
`plots/analyze_models/images/resolution_eval/v24_extremes_2023_20261106/`,
then writes the campaign's `complete.json`. These outputs are now present and
the report checksum was verified during the November7 handoff. The November14
readout supersedes the pending-interpretation status for canonical M1's day-5
primary/global occurrence scores. These old queue commands are for diagnostics,
not instructions to repeat the completed campaign.
