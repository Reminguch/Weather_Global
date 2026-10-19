# Development — Fresh cached data scaling

Updated: 2026-10-15 (America/New_York)

## Implementation

The [controller](../../../scripts/experiments/run_v24_fresh_data_scale.py)
registers the completed fresh width128/7-year control and schedules three new
lineages. Its frozen manifest, configuration hashes, initial checkpoint hashes,
stage proofs and submission ledger identify the executable experiment. See
[e001](experiments.md#e001) for the active campaign root and job IDs.

Each new lineage runs a separate 70-update GPU qualification with full common
validation, native checkpoint serialization, independent restart/replay and
first/middle/last-year bounded-read checks. Qualification does not become
production training. Production starts again from saved step zero, advances
through five exact-resume 4,000-update jobs, selects on the fixed ten-point
validation grid, then runs 2,000 closed-loop updates and parent/final 2022
evaluation. Only after all four final checkpoints are frozen are the four
8-task 2023 evaluation arrays released, followed by strict merges and comparison.

Open training keeps the historical control's training arithmetic. The source
provenance file records restored historical files and retained bounded/year-
sharded I/O. Training and cache paths have explicit identity checks; finite
restart floating-point drift is advisory, while nonfinite values, changed
identities, RNG/cursor mismatch and serialization changes stop execution.

The aligned cache links 2,008 historical chunks from the existing 40-year
producer and 484 original seven-year chunks. All original shared weather data
must match bitwise; timestamp, baseline, normalization and prediction checksum
checks must pass. This avoids approximately one TiB of duplicate predictions.
The cache is not considered ready until verification publishes its READY marker.

The producer compatibility bridge repairs only a definite scheduler rejection
of an explicit GPU partition. It reuses the producer's immutable code and
existing successful stage reports, verifies scheduler completion, and resumes
the idempotent submission tail. It does not rerun scientific stages, cancel
jobs or change training settings. Scientific failures and ambiguous submission
intents require inspection.

## Resources and validation

Measured comparison jobs: width128 cached20k **15295330**, 37h12m and312.64GiB;
width512 cached4k **15295411**, 12h34m and400GiB; width128 closed2k **15517958**,
5h09m and247.57GiB; width512 closed2k **15575009**, 6h40m and247.96GiB.
Historical cached peaks include accumulated mapped pages. All new workers use
`WEATHER_PREPARED_BOUNDED_READS=1`; qualification measures the new peak.

Qualification requests one full A10080,4CPUs,128GiB,4h. Subsequent 4k cached
stages request at least96GiB and1.5× measured peak, rounded up to16GiB; time
uses1.4× measured p90 update cost plus validation and restart overhead.
Closed2k requests384GiB/8h for width128 and416GiB/10h for width512,8CPUs.
2022 evaluations request20GiB/2h; 2023 shards20GiB/90min,4CPUs,one A10080.
Arrays have no concurrency throttle. GPU partition selection is left to Della;
QoS is chosen from its verified24h/72h/144h GPU limits.

CPU cache alignment requests64GiB/8CPUs/4h,40 import tasks each8GiB/2CPUs/4h,
and final verification16GiB/4CPUs/4h. Wait and scheduler bridge jobs each request
2GiB/1CPU/143h5min. Their143h deadline includes queue-dependent upstream work
after they start; expiration stops the downstream release.

Tests cover fresh-state transfer, cache provenance and corruption, exact-resume
boundaries, selection, evaluation coverage and merging, submission identity,
final-model gates and scheduler repair. Real CPU startup has also exercised
the historical runtime, actual saved width128 initial state and existing cache:
424 training chunks and15 validation segments. GPU measurements and scientific
results remain pending until the queued jobs execute.

## Resume

Work from the repository root in the GraphCast environment:

```bash
source scripts/graphcast_env.sh
```

Inspect the campaign `state.json`, `reports/` and `logs/`, plus Slurm accounting.
Run its frozen controller's `status --campaign-root <root>` to verify identities
and display recorded jobs. A successful stage with `release_failed` can resume
its submission tail with `advance --campaign-root <root> --stage <key>` after
checking that its Slurm job is completed and reconciling any submission intent.
Existing recorded job IDs are reused; do not blindly resubmit jobs.

For scientific failure or a timeout, preserve the logs and checkpoints first;
prepare an explicit recovery with the same initial state and data/protocol.
Never modify an existing source snapshot. A source edit in the workspace does
not change already submitted work. Later job IDs are added automatically to
the campaign ledger; refresh the research record when results arrive.
