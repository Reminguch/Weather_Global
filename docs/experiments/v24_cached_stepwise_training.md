# Matched res1 cached stepwise training

This experiment replaces repeated frozen GraphCast inference with the verified
GC trajectory cache. Residual spatial processing, Mamba calls, the 24-step host
state tape and reverse-time VJPs remain stepwise. No sequence batching or custom
Mamba kernel is used.

## Scientific configuration

Both fresh runs use residual width512/MP2, Mamba1 di64/BCG2/state16/conv4/two
layers per insertion, batch1, BPTT24, segment96, AR tail20 and all-step loss.
Spatial weights receive the official GC Small overlay; residual and temporal
output heads start at zero. AdamW uses constant LR1e-4, no warmup/decay,
betas(0.9,0.98), weight decay1e-4, clipping1 and seed22. Both runs execute4,000
updates with checkpoints and all15-segment validation every500 updates.

Inputs restart at each cached chunk. SSM and convolution state carry across
the four chunks in a segment, with gradients stopped at update boundaries.
Both components reset at the next segment. Training RNG and cursor are saved
in native exact-resume checkpoints and are independent of validation.

## Entry points

Always activate the repository environment first:

```bash
source scripts/graphcast_env.sh
python scripts/experiments/run_v24_cached_pair.py prepare \
  --experiment-root artifacts/checkpoints/v24_Ilya/res1_cached_stepwise_pair_20260914 \
  --cache-root data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20
```

Preparation writes paired configurations and freezes source into the experiment
directory. Pending jobs execute that snapshot; changes to the working checkout
cannot silently change an experiment. Producer-hashed cache files are unchanged.

The launcher stages are `submit-smoke`, `submit-gates`, `submit-resource-check`,
and `submit-production`, each with `--experiment-root`. Add `--dry-run` to inspect
the resolved resource request without submitting. Production submission creates
the independent online/cached jobs and an `afterok` reporting job.

Use `submit-smoke --auto-advance` to run the complete authorized pipeline. Each
successful stage submits the next with an `afterok` dependency and its measured
resource budget. A failed check stops advancement. `pipeline_status.json` and
`submission_*.json` record the stage and job IDs. Existing production runs require
an explicit `execute --resume PATH`; uncommitted metric tails are archived before
the checkpoint is replayed.

The initial smoke runs canonical initialization in its own process, followed by
three updates per backend. The full gate allocation is sized from these times:
historical deterministic GC cache generation was considerably slower than the
older online training log, so that older log is not a production runtime promise.

## Correctness gates

- Hash every shared and run-local initial numerical state tree.
- Compare data at the first complete segment, next reset, final training chunk,
  and first/final validation chunks.
- Compare full-chunk predictions, losses, terminal states, gradients and one
  update, including nonzero output projections and incoming memory.
- Obtain online gradients through an ordinary optimizer wrapper around the
  unchanged maintained update. Compare each nonzero parameter group.
- Run20 paired updates and require exact same-backend checkpoint/resume replay.
- Prevent cached workers from constructing frozen-GC predictors.

Loss relative tolerance is1e-4; prediction/state/gradient relative L2 tolerance
is1e-3; nonzero gradient cosine must be at least0.99999. Near-zero values use
absolute tolerance1e-5. Failed or stale provenance gates block production.

## Evaluation and resources

A common streaming evaluator computes both the original BF16 training objective
and exact FP32 original GraphCast loss on physical predictions. It retains one
prediction at a time. The primary improvement is100*(1-corrected_loss/baseline_loss),
using aggregate losses rather than averaged improvement percentages.

Start with one full A10040GB (`gpu40&nomig`), eight CPUs and128GiB for the
two-hour preflight. Measure50 warm updates and all60 validation chunks during
the full gates. Repeat cached execution with four-core affinity established
before Python imports. CPU thread limits are recorded with the benchmark.

Cached production RAM is1.4 times measured peak, rounded up to16GiB increments,
minimum32GiB. A separate job must complete70 updates plus full validation under
that actual smaller allocation before production can start. Prefetch defaults
to zero; the reader supports at most one independently owned prefetched chunk.

Final reporting replays saved500/2,000/4,000 checkpoints through both backends,
plots paired curves and reports measured speed and memory. Acceptance requires
final exact validation losses within1% and loss reductions within0.5 percentage
points. Cache-generation cost is separate from reusable-cache training cost.

## Output layout

- `shared/`: canonical native step-zero checkpoint and initialization identity.
- `source/`: frozen executed source and file hashes.
- `reports/`: preflight, full parity, paired continuation, benchmarks, resource
  verification and final comparison.
- `runs/online/` and `runs/cached/`: native checkpoints and training/validation
  metrics; numerical state is never reconstructed from a parameter-only file.
- `logs/`: Slurm output and sampled GPU memory.

CPU tests verify the update arithmetic, cache ownership, streaming exact loss,
checkpoint identity and submission gates. Full-resolution claims require the
GPU gate reports; merely having this implementation does not establish parity
or a speedup.
