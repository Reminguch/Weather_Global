# Conditional 40-year cached open-loop training

The implementation prepares one shared K=20 frozen-GraphCast cache, then trains
the requested correction models. The original default is fresh width128 and
width512; `prepare --widths 512 --init-from CHECKPOINT` instead preserves the
checkpoint's native architecture and trained residual weights. Warm starts reset
optimizer, RNG, cursor and recurrent memory, and pin the parent checkpoint hash.
Qualification verifies exact parent-weight transfer instead of requiring zero
output heads. GPU batch feasibility and production runtime remain pilot-measured.

The October14 campaign `res1_cached_40yr_w512_di64_20261014_v5` requests only
width512 di64/BCG2, initialized from the previous cached update4,000. It uses
5,000 additional updates, batch4, a50-update warmup to constant1e-4, and the
1982–2021/2022 split below. Its download record is
`artifacts/data_staging/res1_1982_2022_recovery_20261014/download_status.json`.
Recovery writes eight-timestep blocks with two workers, verifies finite values,
records block progress, and preserves the prior failed attempts. The old
precompute-only campaign remains immutable and watches its original status path.
The new gate allows143hours (`--download-wait-hours 143`), based on the first32
recovered timestamps taking235seconds; this is a waiting budget, not a completion
promise. The CPU allocation includes five additional minutes for exit/reporting
and requests `vlong`. Slurm's test submission accepts this budget and rejects
seven days, despite the CPU partition advertising a longer maximum.

## Commands

For **precomputation only**, prepare a new root with `--stop-after cache_verify`.
This endpoint is frozen into the campaign identity and blocks training and
evaluation submissions, including direct submission attempts. The ordinary
`submit` command then releases only download verification, annual preparation,
GPU cache qualification, cache generation and full cache verification. Its final
marker is `reports/precompute_complete.json`. An existing full-training campaign
cannot be converted by passing this flag at submission time.

The October 13 precompute campaign uses
`artifacts/checkpoints/v24_Ilya/res1_cached_40yr_precompute_20261013/` and the
recovered download record under
`artifacts/data_staging/res1_1982_2022_recovery_20261013/`. The original downloader
stopped during 1988 despite publishing that year's coordinate metadata. Recovery
rewrites and verifies the complete 1988 weather payload before appending later
years; completed prior years are retained. Its scripts, prior status/metadata,
repair checksums and log are preserved in the recovery directory.

Cache verification requests 16 GiB: the earlier 234 GiB cache verification
job 13745393 completed in 18 minutes but reached its entire 8 GiB allocation.
Production generation continues to use fresh pilot measurements.

Run from the repository root in the GraphCast environment. `prepare` writes
configs and freezes source, checkpoint/statistics hashes, and campaign identity.
It permits the already-running download to remain incomplete. `submit --dry-run`
prints the initial Slurm command and downstream stages without enabling automation
or submitting any jobs.

```bash
source scripts/graphcast_env.sh
CAMPAIGN_ROOT=artifacts/checkpoints/v24_Ilya/res1_cached_40yr_20261012
python scripts/experiments/run_v24_cached_40yr.py prepare --campaign-root "$CAMPAIGN_ROOT"
python scripts/experiments/run_v24_cached_40yr.py submit --campaign-root "$CAMPAIGN_ROOT" --dry-run
python scripts/experiments/run_v24_cached_40yr.py status --campaign-root "$CAMPAIGN_ROOT"
```

The following command is the execution boundary. It submits the CPU download
gate and enables automatic downstream submissions once their prerequisites pass:

```bash
python scripts/experiments/run_v24_cached_40yr.py submit --campaign-root "$CAMPAIGN_ROOT"
```

The default download record is
`artifacts/data_staging/res1_1982_2022_20261012/download_status.json`; override it
with `prepare --download-status PATH` only when preparing a new campaign.
`status` reports the pinned download record, saved jobs, and Slurm accounting.
`execute` is the internal Slurm entry point, not an interactive launch command.

## Dependencies and qualification

```text
download_gate (CPU, waits for the configured budget; default 72 hours)
  → prepare_pilot (1982 and leap-year 1984)
  → prepare_years (41-task CPU array)
  → prepare_verify + anchor manifest + target-year checks
  → cache_full / cache_mig → collectors → cache_release
  → cache_generate (at least 40 GPU shards) → cache_verify / READY.json
  → training_w128_full / training_w128_mig → collectors → width-128 release
  → training_w512_full / training_w512_mig → collectors → width-512 release
  → independent checkpointed training chains → evaluations → report
```

Scientific stages use `afterok` dependencies and check saved evidence before
releasing successors. Qualification collectors alone use `afterany`, allowing an
OOM or timeout on one GPU type to be recorded without treating it as successful
qualification. An integrity failure blocks release. Arrays have no concurrency
throttle.

- The download gate requires a completed producer record, its matching source
  identity, a final non-dry-run report covering all years, and the exact six-hour
  time grid. Zarr coordinates alone never signal completion.
- Preparation writes bounded monthly windows into annual FP32 shards, verifies
  data and checksums, and resumes only completed verified months. The two pilot
  years are reused. The full root becomes trainable only after all 41 years
  verify. Storage checks retain 256 GiB beyond estimated outstanding payloads.
- Cache pilots generate and replay representative first/last training and
  validation chunks. They check FP32 disk round trips and online parity before
  production generation. CPU verification checks every completed cache shard
  before publishing a full `READY.json`.
- Both full A100 40 GB (`gpu40&nomig`) and 40 GB MIG (`intel&gpu40`) allocations
  are qualified. If both pass, production can use `gpu40`; if only one meets
  resource limits, production uses that type. Requests use the slower qualified
  measurements. No production throughput or queue-time promise is implied.
- Each width probes batches 1, 2, and 4. Batch 4 must complete 70 updates, stay
  below 32 GiB measured GPU memory, preserve independent sequence lanes, pass
  checkpoint and initialization checks, traverse representative years without
  accumulating host memory, and complete full held-out validation without
  changing training state. Failure does not silently reduce batch size or change
  the learning setup.
- Production allocations are sized from measured host RAM and end-to-end time.
  Training chains stop at checkpoint boundaries and fit within 24-hour requests.
  The number of cache shards increases beyond 40 if necessary for that limit.

The training numerical policy is `advisory`: finite GPU checkpoint-replay
tolerance drift is reported separately. Raw data/cache identity, finite values,
fresh initialization, state/cursor structure, independent batch-lane agreement,
and validation coverage remain required. The baseline-cache pilot retains its
own normalized parity check.

## Scientific settings

| Setting | Both runs |
|---|---|
| Data | 1° grid, 13 pressure levels; 1982–2021 training, 2022 validation |
| Frozen baseline | GraphCast Small, width 512, mesh 5, 16 message-passing steps |
| Correction spatial model | Fresh initialization; width 128 or 512; 2 message-passing steps |
| Temporal model | Two Mamba layers, inner/state dimension 16, BCG1, convolution width 4, dt rank 32 |
| Initial correction | Zero output head |
| Open-loop chunks | 24 predictions: 4 observed-input windows followed by 20 baseline-feedback steps |
| Sequence memory | Carry within each 96-step segment; reset between segments; BPTT 24 |
| Batch | 4 independent sequence lanes on one GPU |
| Exposure | 5,000 optimizer updates = 20,000 sequence chunks per width |
| Optimizer | AdamW with decoupled weight decay, LR 1e-4 after 50-update warmup, constant schedule; seed 22 |
| Other optimizer settings | Betas 0.9/0.98, weight decay 1e-4, gradient clipping 1 |
| Precision | BF16 compute; FP32 cached physical weather fields |
| Objective | All-step original GraphCast loss |
| Checkpoint/validation cadence | Save every 250 updates; full held-out validation every 500 |

The correction never feeds its prediction into the cached training trajectory.
Both widths share one immutable baseline cache. Bounded prepared reads return
owned arrays and release source mappings, enabled by
`WEATHER_PREPARED_BOUNDED_READS=1` in the frozen Slurm runtime.

The existing anchor convention excludes incomplete 96-step segment tails.
Expected full-data coverage is 2,432 training chunks and 60 validation chunks
(1,440 validation prediction steps);
the generated manifest and its excluded-tail report are authoritative. Target
timestamps are explicitly checked to remain in their intended year split.
Validation runs at batch 1 over every complete validation segment. The 608
training segments divide evenly into groups of four, so this dataset loses no
additional training segment to an incomplete batch group.

For each width, select the checkpoint with the lowest exact aggregate original
GraphCast loss across complete 2022 validation segments. Evaluate it on the same
32 fixed starts for 40 six-hour leads (10 days), reporting baseline-feedback
`cold_bp` and corrected-feedback `cold_full` separately. Memory starts at zero
for each initialization. This is a 2022 validation/evaluation comparison; the
workflow does not claim an independent 2023 test result. Loss improvement is
`100 * (1 - aggregate_model_loss / aggregate_baseline_loss)`.

## Artifacts and recovery

Default campaign root:
`artifacts/checkpoints/v24_Ilya/res1_cached_40yr_20261012/`.

| Location under campaign root | Contents |
|---|---|
| `campaign.json`, `configs/`, `source/` | Pinned experiment definition and frozen runtime |
| `state.json`, `intents/` | Job IDs, attempts, automation state, submission-recovery evidence |
| `reports/`, `logs/` | Stage results, integrity hashes, Slurm output |
| `qualification/` | GPU/batch measurements, initialization, parity, and resume evidence |
| `training/w128/runs/cached/`, `training/w512/runs/cached/` | Checkpoints and training/validation metrics |
| `evaluation/`, `results.json`, `REPORT.md` | Selected-checkpoint rollout results and comparison |

Data paths under `data/graphcast/graphcast/dataset/`:

- Source: `wb2_res1_levels13_1982_2022.zarr`.
- Prepared data: `prepared_stream_1982_2022/res1/`.
- Anchors: `anchor_manifests/v24_Ilya_res1_1982_2022/`.
- Shared cache: `precomputed_baselines/v24_res1_gcsmall_1982_2022_bptt24_k20/`.

After inspecting `status`, retry a recorded failed stage with its exact key:

```bash
python scripts/experiments/run_v24_cached_40yr.py resume --campaign-root "$CAMPAIGN_ROOT" --stage prepare_years
python scripts/experiments/run_v24_cached_40yr.py resume --campaign-root "$CAMPAIGN_ROOT" --stage cache_generate
```

These are alternative recovery examples, not commands to run consecutively.
For a training continuation use the recorded `train_w128_XXXXX` or
`train_w512_XXXXX` key shown in `state.json`; the measured allocation determines
its endpoint. A `release_failed` stage can be resumed to complete downstream
submission without repeating successful scientific work.

Retries refuse to duplicate queued/running work. They preserve independent
branches, retire only blocked descendants of the failed attempt, and refuse to
replace descendants that already produced scientific output. Annual preparation
keeps verified months; cache generation keeps complete verified shards and
regenerates an interrupted shard. Training resumes the latest compatible complete
checkpoint, including optimizer, RNG, per-lane cursor, and recurrent state.

An unresolved submission intent means the scheduler may have accepted a job
before its ID was recorded. Reconcile its stable Slurm comment and job ID before
retrying; do not delete the intent blindly. Source/config/provenance mismatches
fail closed. Scientific changes require a new campaign identity rather than
editing the frozen run.
