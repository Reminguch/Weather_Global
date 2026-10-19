# v24 res1 open-loop baseline cache

The producer runs only frozen GraphCast Small. Each existing training chunk has
24 predictions: observed input windows at indices 0–3, followed by baseline-only
feedback. Prediction 3 first enters the input at index 4. Physical inputs restart
at each chunk; Mamba state is neither constructed nor saved.

The existing manifest supplies 424 training chunks (2015–2021) and 60 validation
chunks (2022). Its incomplete segment tails, 50 and 18 anchors respectively, are
recorded as exclusions. The full FP32 prediction payload is 234.03 GiB.

## Commands

Run from the repository root in the GraphCast environment:

```bash
source scripts/graphcast_env.sh
python scripts/preprocessing/precompute_v24_baseline.py \
  --config configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json \
  --output-root data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20 \
  --dry-run
```

For a new cache, prepare the workflow **before submitting the pilot**:

```bash
python scripts/preprocessing/submit_v24_baseline.py prepare \
  --config configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json \
  --pilot-root data/graphcast/graphcast/dataset/precomputed_baselines/NEW_CACHE_pilot \
  --output-root data/graphcast/graphcast/dataset/precomputed_baselines/NEW_CACHE \
  --submit
```

Use new, empty cache roots. Omit `--submit` to prepare and inspect the frozen
workflow without launching jobs. Repeating the same command reuses its original
snapshot and recorded job IDs; it never refreshes source from the workspace.
`PILOT_ROOT/workflow.json` binds the complete source snapshot (including vendored
GraphCast), pinned configuration, workspace and output paths. Configuration data
paths are made absolute. `workflow_submission.json` records pilot/release IDs.

The four-chunk pilot uses one gpu80 GPU, 8 CPUs, 32 GiB host RAM, and 1:02
walltime. It selects the first and last
training and validation chunks. The pilot checks all 96 predictions against a
separately initialized maintained training baseline transform, with independent
observed-window loading and AR concatenation; no residual model is initialized.
It also checks reconstructed inputs, forcing times, residual targets, and exact
FP32 disk round trips. Errors must be at most 1e-5 in normalized units.
Generation and the online reference use `--xla_gpu_deterministic_ops=true`,
matching the repository's GPU parity-test convention. This reduction policy is
part of cache compatibility. The original online training job is unchanged;
its default GPU reductions need not be bitwise reproducible across executions.

The wrapper submits release with an `afterok:PILOT_ID` dependency. Release
verifies the original frozen workflow and pilot, checks accounting, then submits
eight unthrottled GPU shards sized from measured memory and throughput. It also
submits CPU verification dependent on the whole array. Pilot, release, every
production shard and final verification execute the same checked snapshot;
subprocess imports use that snapshot and data/output paths use the workspace.
All stages reject changed snapshot files or pinned configurations before work.
The production submission IDs and sizing evidence are recorded in
`PRODUCTION_ROOT/submission.json` and appended to `current_experiments.md`.

Do not launch the Slurm scripts directly with the former short argument lists:
they now require the original workflow's explicit workspace/source/pilot paths,
which the wrapper supplies. Historical pilots without `workflow.json` cannot
be rebound to today's source. They need an explicitly audited recovery or a new
pilot in new cache roots; existing manifests and compatibility checks remain
unchanged. In particular, a passed pilot does not authorize ignoring a later
source mismatch.

## Layout, integrity, and reuse

- `manifest.json`: immutable source/configuration/code fingerprints, coordinates,
  variable shapes, chunk indices/timestamps, exclusions, and shard assignment.
- `shard_NNN/chunk_NNNNNN/VARIABLE.npy`: arrays shaped `[24, *spatial_dimensions]`,
  always FP32. Variable and coordinate ordering follows the prepared store.
- `shard_NNN/COMPLETE.json`: complete coverage and checksums for that shard.
- `READY.json`: written only after successful global verification; distinguishes
  the pilot subset from the complete production cache.
- The pilot additionally writes `pilot_report.json` with parity results, timing,
  and peak process memory. Slurm GPU telemetry is written under `logs/`.

Workers use separate temporary shard directories and an exclusive writer lock.
`--resume` verifies and skips completed shards; interrupted temporary shards are
rebuilt. A corrupted completed shard fails verification rather than being silently
accepted. To deliberately regenerate one, move its completed shard directory
outside the cache root and remove `READY.json`, then rerun that shard with
`--resume` and run global verification. Source/configuration changes require a
new cache root. Production generation requires an allocated GPU.

`--max-chunks N` creates a separately identified representative subset, balanced
between splits where possible; it must not share a root with the full dataset.
The default shard count is eight. The pilot uses `--num-shards 1 --max-chunks 4`.
A full verification for a prepared workflow can also be invoked with the
snapshot runtime (replace `PILOT_ROOT`, `WORKSPACE` and `PRODUCTION_ROOT` with
the absolute paths from `workflow.json`):

```bash
python PILOT_ROOT/source/scripts/experiments/frozen_runtime.py exec \
  --source PILOT_ROOT/source --workspace WORKSPACE \
  --script scripts/preprocessing/submit_v24_baseline.py -- run \
  --phase verify --pilot-root PILOT_ROOT --config PILOT_ROOT/frozen_config.json \
  --output-root PRODUCTION_ROOT
```

Prepared source arrays are identified by metadata/coordinate hashes and per-file
size/mtime, rather than rereading hundreds of GiB to hash weather fields.
Checkpoint and normalization files are SHA256 hashed. Treat prepared arrays as
immutable. Mamba architecture, optimizer, loss selection, and learned parameters
are excluded from compatibility; baseline architecture, numerical precision,
source selection, and forecast schedule are included.

## Reader

```python
from pathlib import Path
from src.models.mamba.v24_Ilya.baseline_cache import (
    BaselineCacheReader, build_manifest, load_chunk, load_context,
)

context = load_context(Path("configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json"))
cache_root = Path("data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20")
expected = build_manifest(context, num_shards=8)
reader = BaselineCacheReader(cache_root, expected["compatibility_sha256"])
item = expected["chunks"][0]
chunk = load_chunk(context, item)
for inputs, baseline, residual_target, forcings in reader.iter_chunk(
    item["id"], chunk, context[4].time_step
):
    # Train the correction branch here. No GraphCast call is needed.
    pass
```

The reader memory-maps prediction files and reconstructs windows with the same
helpers used by training. Consume each chunk before loading another: the prepared
loader reuses owned truth buffers. The reader requires global completion by
default and validates shard coverage/schema on first use; full checksums are
verified by the generation/resume/final-verification commands.

The submitted online training job is unchanged. Building the offline Mamba
optimization loop is subsequent work.
