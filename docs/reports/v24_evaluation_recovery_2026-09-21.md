# V24 evaluation recovery, 2026-09-21

Both training campaigns completed. This recovery reuses their saved checkpoints
and reruns evaluation and reporting only. At submission verification, all eight
GPU tasks were pending and both new reports were pending on their dependencies.
Submission does not imply completed evaluation or successful GPU smoke checks.

Future-run prevention and the new preparation/launch contract are documented in
[frozen experiment workflows](../experiments/frozen_experiment_workflows.md).

| Campaign | Failed evaluations | Recovery array | New report |
|---|---|---|---|
| FP32-MSE / AMSE | 14147423, 14147425 | 14246320_0–1 | 14246359 |
| Spatial-LR 8k | 13963693_0–5 | 14246322_0–5 | 14246363 |

## Causes and fixes

The AMSE evaluator required `.zmetadata` while the existing 2022 dataset is
Zarr v3 with inline consolidated metadata in `zarr.json`. The dataset path was
valid. The evaluator now validates and fingerprints either consolidated format.
Future AMSE preflights validate metadata for both the 2022 exact-loss dataset
and the 2023 spectral dataset; the old smoke covered only 2023.

The spatial launcher checked live workspace files against submission-time
hashes and executed evaluator subprocesses from that workspace. Seven files
had changed. All 33 files in its original checksum manifest were recovered
from saved copies and verified. The recovery executes those original model
files from an isolated runtime, with explicit frozen script/import paths and
absolute paths to the existing data and checkpoints.

The spatial saved snapshot is partial. Remaining supporting modules come from
the complete, checksum-verified AMSE source snapshot of September 19. This
dependency source is explicitly recorded in the recovery manifest. The original
33 spatial files take precedence. Import validation confirms that the spatial
configuration and evaluator resolve inside the recovery runtime.

Training manifests and source snapshots are preserved. Recovery source and
checkpoint hashes are verified at execution. Each GPU task runs a two-step
forecast from its completed training checkpoint in a separate process before
full evaluation. AMSE retains the prescribed final/SWA pair, matched32 2022
origins and eight 2023 spectral starts. Spatial retains the original shared
start/final8000/SWA2000–8000 evaluations and matched32 comparison.

## Validation

- 23 focused tests passed, including Zarr v2/v3 fingerprinting, metadata-change
  rejection, missing-exact-data preflight failure, source/checkpoint mutation
  rejection and frozen subprocess path selection.
- Shell syntax and `git diff --check` passed.
- All eight AMSE specifications (two arms × two checkpoints × two datasets)
  were built and verified against actual files.
- Both datasets opened successfully, exposed all 1,460 six-hour timestamps in
  their evaluation year, and returned a finite physical T2m sample.
- Real-data checks ran outside the filesystem/network sandbox because Zarr's
  asynchronous reader stalled inside it. No dataset changes were needed.

## Resources and execution records

Each evaluation requests one 80GB GPU, eight CPUs and 16GiB CPU memory;
AMSE gets four hours and spatial gets four and a half hours, including smoke.
Arrays are unthrottled. Each report requests two CPUs, 8GiB and 30 minutes,
with `afterok` on its replacement evaluation array.

Budget evidence: job 13740086_6 completed three matched32 evaluations in
3:06:27 with 9,706.50 MiB peak RSS; job 13900169 took 3:05:57 with
10,098,656 KiB RSS. Spectral jobs 14080456_0–2 took 16–17 minutes with
approximately 7.3–7.8 GiB RSS.

Recovery artifacts, validation records, exact submission commands and logs:

- [AMSE recovery](../../artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919/evaluation_recovery/20260921/manifest.json)
- [AMSE submission](../../artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919/evaluation_recovery/20260921/submission_evaluation.json)
- [Spatial recovery](../../artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916_8k/evaluation_recovery/20260921/manifest.json)
- [Spatial submission](../../artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916_8k/evaluation_recovery/20260921/submission_evaluation.json)

The older report jobs 14147426 and 13963694 are superseded by the new reports;
their failed-dependency submissions were not modified.

```bash
squeue -j 14246320,14246322,14246359,14246363
```
