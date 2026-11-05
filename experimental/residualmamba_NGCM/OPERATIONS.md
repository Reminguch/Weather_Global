# Running the open-loop NeuralGCM residual implementation

This implementation follows [README.md](README.md). The frozen backbone is the
**official pretrained NeuralGCM**, not a recovered residual checkpoint. The
residual starts from fresh parameters. No previously trained residual weights,
native increments, adaptive weights or closed-loop training code are loaded.

## What changed

The branch operates on two decoded weather frames and the next native baseline
forecast. Its 796 inputs include the specified deterministic forcings and the
persisted SST/sea-ice fields. A fresh mesh-4, width-512 GraphCast with the v24
interleaved Mamba blocks produces 203 normalized corrections, covering seven
variables and 29 levels. Upper levels are copied unchanged, including their bit
patterns. The corrected output never enters the native trajectory or the next
residual input window.

Training uses three independent one-step baseline forecasts followed by a
single native 21-step tail. The residual differentiates through all 24 Mamba
steps. Memory is detached between chunks, carried across four chunks and reset
at the next 96-anchor segment. `attach_temporal`, `build_segments`,
`advance_cursor` and `build_optimizer` are imported from v24 without editing
those implementations. The optimizer is AdamW with beta2=0.98, 200-step warmup,
1e-4 peak LR and a cosine decay to 1e-5 at update 10,000.

Normalization is `(weather - mean) / std`; residual targets and physical
corrections use the adjacent **6-hour difference** standard deviation without
subtracting a mean. Loss sums seven field scores with Gaussian area weights,
pressure weights normalized over the 29 corrected levels, and 0.05 weights for
cloud ice and liquid. There is no sliding-window weighting in this protocol.

The user's subsequent bounded 2,000-update H200 request has a separate
[automatic diagnostic procedure](DIAGNOSTIC_2K.md), including its explicit
MENGDIW cache location, deferred full production audit, one-hour job continuation
and fixed validation at updates 1,000 and 2,000.

## Deterministic GPU execution and restart

The default `deterministic_gpu_v1` execution policy configures XLA before JAX
initializes a backend. Import `residualmamba_ngcm` before calling `jax.devices()`,
creating device arrays or otherwise initializing JAX in an embedding script.
The CLI and supplied Slurm scripts do this automatically. The required options
are:

```text
--xla_gpu_enable_scatter_determinism_expander=true
--xla_gpu_exclude_nondeterministic_ops=true
--xla_gpu_deterministic_ops=true
--xla_gpu_autotune_level=0
```

Conflicting options and imports that are too late to set missing options fail
explicitly. Restored parameters, memory, optimizer state and RNG are placed on
the device before the next update. The execution policy is included in artifact
identity, so previous artifacts cannot silently qualify a new execution setup.
`NGCM_GPU_EXECUTION_POLICY=diagnostic_external` is only for controlled probes;
the production pipeline rejects it.

The original GPU discrepancy reproduced without saving or loading a checkpoint.
The combined execution policy eliminated it in repeated-execution, recompilation
and fresh-process probes on H200. The optimized deterministic scatter path took
about 4.3 seconds per synthetic BPTT24 update, versus 46–47 seconds with the slower
deterministic fallback and 0.35 seconds with the original nondeterministic setup.
These exclude compilation and are diagnostic timings, not production throughput.
See [GPU_RESTART_DIAGNOSTIC_20261104.md](GPU_RESTART_DIAGNOSTIC_20261104.md) for
the comparisons, their scope and the actual trainer regression result.

## Environment and recovered prerequisites

Everything in this session is under
`/scratch/gpfs/MENGDIW/sh4809/weatherforecast/`.

| Item | Location relative to that directory |
|---|---|
| New code worktree | `Weather_Global_rebuild/` |
| Python 3.11 environment | `Weather_Global/.venv_ngcm_data/` |
| Official NGCM checkpoint | `Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl` |
| GC-small checkpoint (configuration only) | `Weather_Global_rebuild/data/prerequisites/graphcast_small.npz` |
| ERA5 recovery root | `Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104/` |
| Download logs | `Weather_Global_rebuild/logs/restore_20261104/` |

The exact environment snapshot is [requirements-tested.txt](requirements-tested.txt).
The large prerequisite files, environments and downloaded weather are excluded
from Git. SHA256 identities:

```text
NGCM: bdec1b4612c7385fc492aa031db252c66fb74b788a7efbb118b8b60b06644d3e
GC-small: e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29
```

The model package is load-only. Download helpers live separately under
`scripts/preprocessing/restore_ngcm_*` and `supervise_ngcm_restore.py`. They
reuse the original producer's conservative skip-NaN regridding and nearest
fill on the original **128 longitude × 64 Gaussian latitude, 37-level, 6-hour**
grid. They stream 0.25-degree raw frames and retain only the prepared frames.
Each prepared file has a SHA256 receipt. The merge verifies every file and
requires all **13,148** timestamps before publishing the complete store.

The two-frame download pilot passed shape, dtype, finiteness, checksum and
no-redownload resume checks. Compute nodes could not resolve Google Cloud, so
restoration runs on the previously used internet-enabled `della-vis1` node,
with four original workers and four CPU threads per worker. On November 4,
twelve cooperative workers were added on the same node, bringing the total to
sixteen after a successful concurrency smoke test. Observed aggregate throughput
increased about 2.9 times. Per-month locks and completed-month skipping prevent
duplicate writes. See the [parallel download record](../../scripts/preprocessing/DOWNLOAD_PARALLEL_20261104.md)
for the H200 network probe, launch commands and measured evidence. The supervisor retries each
month at most three times and can be restarted to resume outstanding work.
This node switch does not change the data producer or its output contract.

## Commands

Run from the new repository root on an allocated compute node. Keep runtime
caches on scratch:

```bash
export PYTHONPATH="$PWD/experimental/residualmamba_NGCM"
export TMPDIR="$PWD/.tmp"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NGCM_PYTHON=../Weather_Global/.venv_ngcm_data/bin/python
NGCM_CONFIG=experimental/residualmamba_NGCM/configs/default.json
NGCM_RUN="$PWD/artifacts/residualmamba_ngcm/main"
```

The full cache must use a writable **DABANIN scratch** allocation, as required
by the specification. The current session has not established access or quota
for a suitable DABANIN directory. Set `NGCM_CACHE` to a real authorized path;
do not substitute the example text below or invent a quota result. The quota
JSON has `cache_path` (absolute path) and `available_bytes`, from a real quota
audit, plus any supporting notes. Filesystem free space alone is not a quota
audit. At least 110 GiB is required for the cache.

```bash
# Set these to an authorized DABANIN path and its recorded quota audit:
# NGCM_CACHE=/scratch/gpfs/DABANIN/<allocation>/<run>/cache
# NGCM_QUOTA_AUDIT=/scratch/.../quota-audit.json

"$NGCM_PYTHON" -m residualmamba_ngcm preflight \
  --config "$NGCM_CONFIG" --run-root "$NGCM_RUN" \
  --store ../Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104/res2p8 \
  --checkpoint ../Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl \
  --checkpoint-sha256 bdec1b4612c7385fc492aa031db252c66fb74b788a7efbb118b8b60b06644d3e \
  --gc-checkpoint data/prerequisites/graphcast_small.npz \
  --gc-sha256 e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29 \
  --cache "$NGCM_CACHE" --quota-audit "$NGCM_QUOTA_AUDIT"

"$NGCM_PYTHON" -m residualmamba_ngcm stats --config "$NGCM_CONFIG" --run-root "$NGCM_RUN"
"$NGCM_PYTHON" -m residualmamba_ngcm cache --config "$NGCM_CONFIG" --run-root "$NGCM_RUN"
"$NGCM_PYTHON" -m residualmamba_ngcm verify-cache --config "$NGCM_CONFIG" --run-root "$NGCM_RUN"
"$NGCM_PYTHON" -m residualmamba_ngcm census --config "$NGCM_CONFIG" --run-root "$NGCM_RUN"
```

`cache --indices 4 28 ...` builds a selected subset. It cannot publish READY
until all 484 train/validation chunks exist and validate. Do not run overlapping
cache producers for the same chunk. A valid existing cache can be consumed by
another control run after matching its producer/data/checkpoint identities.

If a field/level contributes more than 10% globally or within a step class,
inspect `census.json` before training. Retaining the specified scales requires
`census-decision.json` containing the census SHA256, `action` equal to
`retain_v1_scales`, and a concrete `rationale`. Changing floors instead requires
a versioned scientific specification and recomputed statistics; the code never
silently tunes a floor from validation data.

```bash
# These are expensive scientific stages, not smoke tests.
# C1 computes a train-only bias table and reuses its native trajectories for
# the 2015–2021 part of the year-by-year baseline audit.
"$NGCM_PYTHON" -m residualmamba_ngcm bias --config "$NGCM_CONFIG" --run-root "$NGCM_RUN"
"$NGCM_PYTHON" -m residualmamba_ngcm baseline-audit --config "$NGCM_CONFIG" --run-root "$NGCM_RUN"
```

The final deterministic 2.8-degree model was trained on 1979–2017, with 2018
used during development, according to the authors' [training methods,
section 7.2](https://arxiv.org/html/2311.07222v3#S7.SS2). Thus 2015–2017 overlap
our residual training period. This concerns the **official pretrained
backbone**, not deleted residual experiments. The year-by-year audit is the
check requested by specification section 14.3. It does not automatically change
the fixed 2015–2021 residual training split.

## Tests and qualification

The unit suite covers time indexing, future-truth isolation, the **actual v24
iterator**, native continuation against the existing native evaluator using a
test backbone, input/target normalization, Gaussian weights, upper-level
scatter, cloud scale floors, full BPTT versus an explicit reverse VJP, zero
chunk-boundary state gradients, cursor advancement, checkpoint integrity,
exact split-versus-uninterrupted continuation, E1 timestamp alignment and loss
concentration. These are necessary checks, not proof of forecast improvement.

```bash
JAX_PLATFORMS=cpu "$NGCM_PYTHON" -m pytest -q \
  experimental/residualmamba_NGCM/tests -p no:cacheprovider \
  --basetemp="$PWD/.tmp/pytest-final"

sbatch experimental/residualmamba_NGCM/tests/deterministic_resume.sbatch
```

The H200 script requests **ailab**, `gpu-test`, one GPU and one hour. It uses
the actual 8192-grid-node, 2562-mesh-node, width-512, four-Mamba-layer model,
checks the zero head and unchanged upper levels bitwise, checks nonzero memory,
and executes three optimizer updates with full 24-step BPTT plus a checkpoint
continuation comparison. It then invokes the production trainer in independent
Python processes, comparing three uninterrupted updates with two updates followed
by a fresh-process resume to update three, including parameters, memory, Adam
state, RNG and cursor. Its inputs are
synthetic and its numerical losses are not weather skill measurements.

The final stage uses 29 already restored real ERA5 frames and the
official checkpoint. It builds one complete 24-step chunk, serializes it,
and compares the 21-step tail to the existing native evaluator. Its pilot
normalization is diagnostic only. It cannot substitute for production T1,
which requires three randomly selected training chunks and one validation
chunk with the full train-only statistics.

`qualify` verifies all three smoke source identities, checks the GPU execution
policy and bitwise restart comparisons, and reruns the unit tests. Supply the
actual successful receipts:

```bash
"$NGCM_PYTHON" -m residualmamba_ngcm qualify \
  --config "$NGCM_CONFIG" --run-root "$NGCM_RUN" \
  --gpu-smoke logs/deterministic-resume-<job>/gpu-smoke.json \
  --native-smoke logs/deterministic-resume-<job>/native/result.json \
  --resume-smoke logs/deterministic-resume-<job>/trainer/comparison.json
```

Per the repository's `AGENTS.md`, finite numerical repeatability differences
are recorded with their reference bounds and warnings. They do not silently
cancel a bounded experiment. The user's specific November 4 request to fix GPU
restart drift adds a regression check for that fix; it does not impose a
monotonic-loss or forecast-improvement condition. Nonfinite computation, corrupt inputs, mismatched
identities and unusable execution remain errors. Test losses need not decrease
at every update. Validation monitors all 15 segments and does not select a
checkpoint or an SWA window.

## Training, continuation and evaluation

```bash
"$NGCM_PYTHON" -m residualmamba_ngcm train \
  --config "$NGCM_CONFIG" --run-root "$NGCM_RUN" --steps 500

"$NGCM_PYTHON" -m residualmamba_ngcm train \
  --config "$NGCM_CONFIG" --run-root "$NGCM_RUN" --steps 1000 \
  --resume "$NGCM_RUN/checkpoints/step_00500.pkl"

# Repeat bounded invocations until update 10000. The LR schedule always uses
# max_steps=10000; --steps only bounds this invocation.
"$NGCM_PYTHON" -m residualmamba_ngcm swa --config "$NGCM_CONFIG" --run-root "$NGCM_RUN"

"$NGCM_PYTHON" -m residualmamba_ngcm evaluate \
  --config "$NGCM_CONFIG" --run-root "$NGCM_RUN" \
  --model-checkpoint "$NGCM_RUN/swa_02000_08000.pkl" \
  --eval-root "$NGCM_RUN/eval/e2_swa_cold"

"$NGCM_PYTHON" -m residualmamba_ngcm report \
  --config "$NGCM_CONFIG" --run-root "$NGCM_RUN" \
  --eval-root "$NGCM_RUN/eval/e2_swa_cold"
```

Checkpoints include parameters, Adam state, Mamba memory, RNG, update and the
next chronological cursor. Resume requires an identical run identity, including
the deterministic execution policy. See [VALIDATION.md](VALIDATION.md) for the
verified restart comparisons. Bitwise equivalence across different GPU types,
JAX/CUDA versions or CPU versus GPU is not claimed. SWA
averages parameters from exactly 13 checkpoints, updates 2000–8000 inclusive;
it never averages memory or optimizer state. Evaluate update 10000 separately.

E2 uses every eligible 00/12 UTC origin in 2023, with 40 forecast steps. Warm-3
uses a separate `--warm` output directory and reports skipped boundary origins.
E1 additionally requires `--e1-manifest`, `--gc-eval-times` (the actual GC
validation dataset's timestamp NPY) and `--gc-logged-origins` (JSON `origins`).
The 32 indexed timestamps must match the recorded GC origins exactly. Those
original GC artifacts are not present in this recovered checkout; no starts
have been guessed. 2022 has historical tuning exposure and is not described as
an untouched test set.

Controls C2/C3/C4 use separate config files that change only `arm`, separate
run roots and the same fixed optimizer, training duration, seed and SWA window.
C2 resets memory every step, C3 also appends lead/240, and C4 removes the next
baseline input. C0 uses `evaluate --native`; C1 uses
`evaluate --bias-table "$NGCM_RUN/bias.npz"`.

Evaluation saves Gaussian-weighted MSE/RMSE and ACC for **every field × level ×
lead**, headline quantities, the training-weighted J curve, paired weekly
bootstrap intervals, and Z500/T850/U850 spherical power spectra at
24/120/240 hours. Percent improvements mean **MSE** reduction. Leads 132–240 h
extend beyond the training tail. Neither relative improvement nor long-lead
stability has yet been established by a production run of this new code.

## Validation status

See [VALIDATION.md](VALIDATION.md) for the dated results and any outstanding
checks. Download progress is separate from model qualification. The package
will not treat a partial data store or a smoke test as production readiness.
