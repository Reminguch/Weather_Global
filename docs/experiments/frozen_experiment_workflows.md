# Preventing evaluation and cache failures after queued training

Updated 2026-09-21. This addresses recurring workflow failures: a later stage
must execute the source saved for its experiment, and preparation must check
the datasets that evaluation will actually use.

## Evidence of recurrence

| Run | Failure | Cause |
|---|---|---|
| AMSE / FP32-MSE, evaluations 14147423 and 14147425 | Missing `.zmetadata` | The existing 2022 dataset is Zarr v3, with consolidated metadata in `zarr.json`. The dataset directory was valid. |
| Spatial-LR 8k, array 13963693 | Source checksum mismatch | Seven live workspace files changed after preparation; the launcher checked and executed the live workspace despite having saved source. |
| September 16 post-training array 13968877 | Unexpected `is_training` argument | The snapshot omitted vendored GraphCast and imported an incompatible installed version. |
| Baseline cache release 14143123 | Pilot compatibility mismatch | `endpoint_step.py` changed between the pilot and release. The compatibility check correctly rejected a changed producer. |

Evidence: [evaluation recovery](../reports/v24_evaluation_recovery_2026-09-21.md),
[spatial failure](../../logs/v24_spatial_8k_13963693_0.err),
[post-training failure](../../logs/v24_r025_posteval_13968877_0.err), and
[baseline release failure](../../logs/v24_baseline_release_14143123.err).

## Source and configuration contract

The maintained batch4, spatial-LR, AMSE and baseline-cache preparation paths use
[`frozen_runtime.py`](../../scripts/experiments/frozen_runtime.py). It saves all
Python, shell and Slurm source under `src`, `scripts` and `third_party/graphcast`
before submission. A missing vendored GraphCast fails preparation. The manifest
records the original workspace and source hashes; verification rejects changed,
missing or extra executable files. An existing snapshot is verified and reused,
never refreshed from the live workspace.

Stages run saved entry points with the snapshot and vendored GraphCast on
`PYTHONPATH`. `PYTHONSAFEPATH=1` prevents nested Python commands from importing
the workspace through their current directory. Data and output paths continue
to resolve in the original workspace. Prepared configurations are pinned by
hash, and the baseline workflow saves its own normalized configuration.

The baseline pilot, release, production shards and final verification share one
snapshot created **before the pilot**. Existing parity, compatibility, checksum
and accounting checks still apply. The cache manifest format and generation
fingerprints are unchanged.

Old partial snapshots and pilots without frozen workflow provenance require an
explicit recovery or a new experiment directory. Do not replace their recorded
checksums with current workspace hashes. The September 21 recovery snapshots
and already submitted jobs remain separate from these future-workflow changes.

## Preparing future runs

For a new spatial 8k experiment, use a new directory:

```bash
source scripts/graphcast_env.sh
experiment_root="$PWD/artifacts/checkpoints/v24_Ilya/NEW_SPATIAL_8K"
python scripts/experiments/run_v24_res1_slow_spatial_8k.py prepare \
  --experiment "$experiment_root"
python scripts/experiments/frozen_runtime.py verify \
  --source "$experiment_root/source"
```

The maintained 8k wrapper prepares all six arms directly; no driver imported
from a historical artifact is needed. It records an 8,000-update budget and SWA
steps 2,000/4,000/6,000/8,000. The batch4 and 2k spatial wrappers support the same
`prepare --experiment` interface.

Submit the **saved** Slurm script, passing stage, absolute experiment path and
absolute workspace path. Set `--chdir` to that workspace. Stages are `smoke`
(tasks 0–1), `seed` (0–1), `train` (0–5), `eval` (0–5), then `summarize`.
Keep successful dependencies between stages; choose resources for each stage
from its pilot/accounting evidence. Training requires the shared seed checkpoint
to exist and verifies its configuration before branching. Evaluation uses the
budget and SWA steps from the prepared manifest. GPU telemetry is retained by
the shared launcher.

For AMSE, use `run_v24_amse_pair.py prepare --experiment NEW_DIRECTORY` followed
by its existing `submit` stage. Both 2022 exact-loss and 2023 spectral metadata
are checked before freezing, and GPU preflight checks them again.

For baseline caches, use
[`submit_v24_baseline.py prepare`](../../scripts/preprocessing/BASELINE_CACHE.md#commands)
with new pilot and production roots. Preparation alone launches no jobs;
`--submit` submits the pilot and dependent release using the saved workflow.
Use this entry point instead of the old direct Slurm argument lists.

## Dataset checks and limits

[`frozen_zarr_metadata`](../../src/data_operations/loaders/zarr_metadata.py)
validates and fingerprints consolidated Zarr v2 `.zmetadata` or Zarr v3
`zarr.json`. It hashes the original bytes, preserving existing v2 identities.
Invalid v3 metadata cannot silently fall back to stale v2 metadata.
Stage-time checks reselect metadata from the dataset root, so adding a v3 file
while leaving a pinned v2 file unchanged also fails verification.

AMSE, batch4/spatial and spectral preparation use this shared check. Daily 2023
preparation checks production and pilot metadata, but its production chunk audit
still requires Zarr v2 and rejects v3 explicitly before freezing. Metadata
validation does not replace coordinate coverage, chunk integrity or GPU forecast
checks. The Python environment and external data remain dependencies outside the
source snapshot.

## Regression coverage

Tests execute saved parent and child processes after changing live workspace
code and contaminating `PYTHONPATH`. They cover missing vendored dependencies,
snapshot/configuration/data mutation, Zarr v2/v3 metadata, preparation failure
before freezing, and the real saved 8k wrapper/Slurm launch path. Baseline tests
exercise pilot, generation and verification after workspace changes and retain
the release compatibility guard.

Validation: **123 tests passed** across the 11 targeted workflow test files
(one existing JAX deprecation warning). Shell syntax checks and
`git diff --check` also passed. These are CPU workflow regressions; no new GPU
training campaign is submitted by this fix.
