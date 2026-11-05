# README v2 implementation

This implements the exact README published on `residualmamba_NGCM` at
`52da7e75`. The normative README is copied verbatim. This file records the
implementation and operating instructions separately.

## Objective and output contract

Only temperature, geopotential, zonal wind, meridional wind and specific
humidity receive corrections and enter the loss, in that order. The head has
145 channels, reshaped to `[5, 29, lon, lat]`. Each field has weight one.
Gaussian latitude weights have mean one; pressure weights are `p / mean(P)`
over the 29 levels from 50 to 1000 hPa. The reduction is the sum over fields,
mean over weighted levels and grid points, then mean over all 24 steps.

Cloud ice and cloud liquid remain in all three weather input blocks. They are
copied bitwise from the official pretrained NeuralGCM at every level. All
fields at 1–30 hPa are also copied bitwise. The input remains 796 channels.
No dynamic weights, scale changes, feedback, optimizer changes or curriculum
are introduced. The main protocol remains BPTT 24, segment 96, ar_tail_k 20.

## Code changes

- `config.py` and `configs/default.json` use schema 2 and 145 outputs, and
  remove `cloud_alpha`.
- `loss.py` accepts five-field targets, implements the Gaussian/pressure
  reduction, and scatters corrections only into scored cells.
- `model.py` creates the 145-channel zero head.
- `train.py` uses a `[24, 5, 29]` census and reports five field contributions
  and shares in validation, alongside totals and per-step values.
- `evaluate.py` restricts the C1 bias table, physical metrics and normalized J
  to the five fields. Saved summaries include `J_per_field` and `J_field_share`.
  `report.py` prints the five contributions and shares.
- Dense validation and checkpoint plotting recognize five-field reports.
  Plot roots must not mix v1 and v2 objectives.
- Statistics, input features, time indexing, native rollout and raw baseline
  cache generation are unchanged.

## Reuse data without resuming a v1 model

The completed v1 tail20 input artifacts can be imported into an **empty** v2
run root. Run from the repository root, using the existing scratch Python
environment and deterministic execution policy:

```bash
export PYTHONPATH=experimental/residualmamba_NGCM
python -m residualmamba_ngcm reuse-v1 \
  --config experimental/residualmamba_NGCM/configs/default.json \
  --donor-root /scratch/path/to/diagnostic_2k_20261105 \
  --run-root /scratch/path/to/new_v2_run
```

`reuse.py` pins the qualified v1 producer, validates its configuration, prepared
store, checkpoint hashes, statistics, geometry, cache READY and 484 chunk
receipts. It creates links to unchanged inputs and records their original
provenance in a new v2 preflight manifest. Chunk payload hashes remain checked
when loaded. No forecast arrays are copied or regenerated. Donor roots and
checkpoints are never overwritten. `stats` and `cache` build commands reject
an imported root so they cannot accidentally modify shared artifacts.

This convenience importer recognizes the existing tail20 diagnostic producer.
The separate tail1 loader also consumes its existing raw cache with v2 targets;
automated tail1 run migration is not included here. Its cache does not need to
be regenerated. The v1 tail1 and tail20 jobs continue on their frozen source
snapshots.

A fresh model and optimizer are required. A v1 run/checkpoint identity cannot
resume under schema 2. Recompute the v2 census and qualification. The normal
production CLI still requires the annual native baseline audit. Importing the
bounded diagnostic's inputs does not waive that requirement or claim production
qualification. All paths, logs, caches and outputs remain on scratch.

## Validation

The full 2022 fixed validation set was recomputed directly from the existing
stats and all 60 cached chunks. The zero-head v2 objective is
**3.5251650667**, matching README §7. The maximum absolute difference between
the new step/field/level contributions and the five corresponding v1 census
contributions is **8.08e-9**.

| Field | Zero-head loss | Share |
|---|---:|---:|
| Temperature | 0.5466317494 | 15.51% |
| Geopotential | 0.5522368643 | 15.67% |
| Zonal wind | 0.7131393809 | 20.23% |
| Meridional wind | 0.5387226912 | 15.28% |
| Specific humidity | 1.1744343809 | 33.32% |

No individual field/level exceeds 10% of the objective averaged over steps.
Checking the one-step prefix and each tail lead separately, the largest
field/level share is 3.291%. No new humidity scale floor is indicated by this
census. This does not establish forecast improvement. The earlier seven-field total
must not be compared directly with the v2 total.

CPU tests cover the independent loss formula and gradient, excluded fields,
signed-zero preservation, full BPTT and continuation, fieldwise evaluation
summaries, protocol rejection, and reused artifact tampering. The H200 smoke
script `tests/v2_smoke.sbatch` runs the full mesh-4 width-512 model, checks real
cached inputs, runs full 24-step BPTT updates, and compares uninterrupted
training with a fresh-process resume across parameters, memory, optimizer,
RNG and cursor. It also exercises the native rollout smoke.

Executed evidence is recorded in `validation/20261105_v2/`. All 46 CPU tests,
the full-cache loss regression, actual input import, H200 model smoke, native
rollout and fresh-process restart checks passed. All five resumed state groups
are bitwise identical, with maximum absolute difference zero. H200 job 16928171
used the frozen implementation commit `db93a3a`. No v2 2k forecast-skill claim is
made by this implementation change.
