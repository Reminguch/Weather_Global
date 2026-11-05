# Parallel ERA5 restoration, 2026-11-04

Restoration now uses **16 concurrent monthly workers** on the internet-enabled
`della-vis1` staging node: the original four workers plus twelve additional
workers. All data, logs and runtime files remain on scratch. The data producer,
official pretrained NeuralGCM checkpoint, original 128×64 Gaussian grid,
37 pressure levels, six-hour sampling and conservative regridding method are
unchanged.

## Why this runs on the staging node

The user requested parallel downloads using multiple GPUs. H200 network probe
**16873636** on `della-i22g1` failed in three seconds because
`storage.googleapis.com` could not be resolved. The
[actual probe log](validation/parallel_20261104/network-h200.txt) is retained.
Additional GPU allocations would not make that node able to fetch the source.
The successful implementation therefore adds concurrent CPU download processes
on the existing internet-enabled node. GPU acceleration is not claimed.

Recent completed-month profiles showed median frame times of 12.18 seconds for
source reading/decompression and 14.69 seconds overall. The remaining work
includes conservative regridding, compression, writing and hashing. Increasing
independent monthly requests addresses the observed source-read bottleneck.

The staging node exposed 80 CPU affinity slots and about 559 GiB available RAM
at the resource check. The original workers use the first 16 affinity slots;
the new workers use slots 16–63, four per worker. These are affinity limits on a
shared node, not exclusive Slurm reservations.

## Coordination and recovery

`parallel_ngcm_restore.py` selects unfinished months in reverse order. The
original supervisor continues in its existing forward order. Both launch
`restore_ngcm_staging.sh`, which now holds a per-month advisory lock throughout
the producer process. A second additional worker receives exit code 75 for a
busy month and tries other work. Original workers wait on the same lock.
Already completed months with both READY and profile receipts return without
redownloading or overwriting their profile.

The original four processes were already running when locking was introduced.
Their then-active indices **24, 29, 30 and 31** were excluded from the extra pool
to avoid touching those initially unlocked writers. All later launches use the
new wrapper. Subsequent inspection confirmed all sixteen live workers holding
distinct monthly locks.

Run the cooperating pools on **the same staging node**. Cross-node advisory-lock
behavior was not established and must not be assumed. Check locks on the owner
node, not on a different login node.

Each additional month is retried at most three times after an error. Existing
SHA256 frame receipts permit continuation. The original supervisor alone owns
the final merge, which verifies all frame hashes and requires all 13,148 frames
before publishing COMPLETE. Increasing concurrency does not mark partial data
as complete.

To restart the additional pool after its previous process has stopped, run on
`della-vis1` from the repository root:

```bash
NGCM_PYTHON=../Weather_Global/.venv_ngcm_data/bin/python
NGCM_PARALLEL_LOGS="$PWD/logs/restore_parallel_20261104"
nohup "$NGCM_PYTHON" -u scripts/preprocessing/parallel_ngcm_restore.py \
  --workers 12 --cpu-offset 16 --exclude 24 29 30 31 \
  --log-root "$NGCM_PARALLEL_LOGS" \
  > "$NGCM_PARALLEL_LOGS/supervisor.log" 2>&1 < /dev/null &
```

The same log-root lock prevents a duplicate pool on that node. If attaching to
a different legacy run, first identify and exclude its active unlocked months.
Do not copy the above exclusions without checking that context.

Status and throughput, measured from completed frame-receipt timestamps:

```bash
"$NGCM_PYTHON" scripts/preprocessing/ngcm_download_status.py --check-local-locks
```

Omit `--check-local-locks` when checking file progress from another node.

## Verification and measured speed

- A completed month returned successfully without changing its profile hash.
- A held month lock caused a second nonblocking wrapper to return 75.
- An out-of-range month index was rejected before starting a producer.
- The runner and data-producer SHA256 values still matched the immutable
  restoration contract. No existing data were invalidated.
- A check on `della-vis1` found **16 live download processes and 16 held locks**.
  The [status receipt](validation/parallel_20261104/verified-status.json) records
  the actual month assignments, observation time and completion counts.

| Measurement | Completed frames per 10 minutes | Equivalent hourly rate |
|---|---:|---:|
| Immediately before adding the pool | 162 | 972 |
| Recent window after about 58 minutes with the pool | 466 | 2,796 |

The observed gain is **2.88×**, not a claimed fourfold gain from the worker
count. This is a before/after observation rather than a controlled bandwidth
benchmark. Network contention makes each individual worker slower. At this
observation, **6,202 / 13,148 frames and 44 / 108 months** were complete.
Future throughput and the final validation/merge duration can differ.
