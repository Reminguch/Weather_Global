# Fresh four-arm restart after the numerical audit

User authorization on September 24, 2026: fix the numerical constraints and
loss, start all four configurations from scratch, and submit formal jobs and
smokes together. The user subsequently clarified that **formal training must
not wait for the full smoke jobs**. The current production jobs have no smoke
dependencies and run independently of those checks.

## Frozen experiment

```text
/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/artifacts/checkpoints/neuralgcm_residual/res2p8_w128_256_di16_32_train2015_2021_20260924_feedback_v2
```

Frozen source identity:
`8ff196d75294a14eab757a9f5d3df8e9cea78e35356861691cc2a02fd4f984b1`.

The four configurations remain width 128/256 × d_inner 16/32, seed 22,
2015–2021 training, 2022 validation, 2023 test. They start from fresh zero
residual heads and fresh optimizers, memory and RNG. No old residual checkpoint
is used to initialize these runs. The frozen NGCM backbone is unchanged.

The K=1 budget remains 20 epochs, 8,480 optimizer updates. The previously
configured formal fine-tuning stage remains 20 forecast steps and 2,000
updates. **K=2 is an additional short transition/resume smoke**, not a silent
change of the formal 20-step budget. Formal jobs use 24-hour `gpu-short` slices,
with checkpointed continuation if necessary. Each smoke uses `gpu-test`, one
GPU and a one-hour limit, without an explicit partition request.

## Changes being tested

- `no_pressure_zero_mean_v2` disables additional direct surface-pressure
  corrections and preserves the solver's existing degree-zero divergence and
  vorticity coefficients. It only constrains the increment. Baseline coefficients,
  state layout and auxiliary carry remain unchanged.
- `neuralgcm_pooled_change_mse_v2` pools the existing training-only six-hour
  change variances across levels for every field except specific humidity.
  Concretely, it uses the root mean square of existing per-level standard
  deviations. This does not include a between-level mean-difference term.
  Amplitude factors before squaring are cloud ice/liquid 0.05, geopotential 2,
  humidity 0.66, and temperature/winds 1. Pressure and Gaussian area weights
  retain their existing definitions. This is a newly named objective, **not an
  exact reproduction of the original NGCM loss**, which also uses 24-hour
  statistics and additional internal-state/spectral/bias terms.
- The correction and loss contracts enter configuration/checkpoint identities.
  Stage transfer now checks those identities, source and configuration too.
- Cold validation with failed origins is recorded as ineligible. The callback
  proceeds to warm validation and records completion without calling checkpoint
  selection on that invalid result. It neither drops failed origins nor selects
  the failed checkpoint.
- The common-grid evaluation uses the configured version of the objective.

The old baseline cache and training statistics are reused read-only. Preparation
checks the cache producer's source hashes, the parent verification receipt,
dataset and manifest identities, READY hashes, and statistics/climatology
hashes. Runtime checks the full training-origin count and hash. `CacheReader`
continues to verify every shard against READY on first access. Numerical smokes
also compare live and cached gradients. This avoids recomputing unchanged
baseline forecasts while keeping the revised residual contract explicit.

## Submitted jobs

| Configuration | Full GPU smoke | Fresh formal pretraining |
| --- | --- | --- |
| w128 / di16 | 14352533 | 14353383 |
| w128 / di32 | 14352534 | 14353384 |
| w256 / di16 | 14352535 | 14353385 |
| w256 / di32 | 14352536 | 14353386 |

The original pending formal jobs 14352581–14352584 were cancelled and replaced
with the jobs above after the user's clarification. Both the scheduler smoke
dependency and the runtime requirement for full smoke reports were removed.
Slurm confirmed `Dependency=(null)` for all four replacements. These jobs use
the separate, hashed operational runner
`pipeline/early_start_v1/run_neuralgcm_feedback_early_start.py`; its adjacent
`policy.json` records the user's instruction, source identity, runner hash, and
the successful independent-smoke report hash. The frozen numerical source,
configuration, cache/statistics checks, and checkpoint identities are unchanged.
Startup audits record this authorization explicitly, without creating or
claiming missing full-smoke pass evidence. Pretraining completion still submits
fine-tuning with an `afterok` dependency on its own pretraining slice, and stage
transfer still checks the selected parent's hash.

At replacement submission, the first three full smoke reports were present
and passed; the w256/di32 smoke had started. The production jobs were pending
cluster scheduling, with no estimated start time available yet.

Independent representative-data GPU smoke **14351743** passed first. It fit its
own isolated pilot statistics. Cached/live state, features, loss, gradients and
updates matched exactly. Saving, reloading and continuing also matched exactly,
and the frozen backbone parameters were unchanged. Its verified report remains
a startup requirement. Full transition checks continue independently and retain
their own pass/fail reports.

The initial attempt to attach that already-completed pilot job ID as a Slurm
dependency was rejected with `Job dependency problem`, after Slurm had removed
it from the active dependency table. No formal job was created on that attempt.
The completed report and core-code hashes were verified instead; its report
hash is stored in the submission graph. That first submission retained all four
full-smoke dependencies, which were subsequently removed as described above.
The generic submitter handles completed pilot evidence with a CPU regression
test. The independent-start wrapper has three passing CPU checks covering
dependency-free initial submission, continuation dependencies, and authorization
identity/evidence rejection. These operational changes do not modify the frozen
numerical source.

The remaining old-configuration job **14322230** was cancelled after the new
jobs were confirmed in Slurm. Its latest saved checkpoint was update **2,600**.
All old experiment artifacts remain intact.

## Smoke coverage and evidence locations

Each full smoke first runs the existing 15 production-shape checks, including
40-step zero-residual identity, live/cache loss and gradients, recurrent gradient
reference, memory carry, physical stop-gradient, real 96-record pretraining,
checkpoint resume, fixed-sample learning, trained-parameter stage transfer,
20-step cold/warm gradients, fine-tuning resume and validation.

It then executes the actual new production CLI in an isolated child experiment
for K=1 and K=20. This tests output-directory creation, gate/audit ordering,
validation/selection receipts, and selected-parent handoff. Finally, it runs
K=1→K=2 with trained parameters, two updates, a saved midpoint, and a separate
process that resumes from that midpoint. Final parameters, optimizer, memory,
RNG, cursor and timing-independent metrics must match exactly.

A real-grid stress check verifies that constrained coefficients and pressure
remain exactly unchanged by an increment. An eight-date training-only loss
audit rejects cloud dominance of 90% or more. The smoke report is written only
after all children succeed and all comparisons pass.

Per-arm evidence is under `checks/restart_smoke/<run_id>/`, with final
`PASSED.json`. The durable submission graph is `pipeline/restart_chain.json`.
The superseded dependency graph is retained as
`pipeline/restart_chain_with_smoke_dependencies.json`; the current independent
startup graph is also recorded in `pipeline/early_start_chain.json`.
Operational and training logs are under `logs/` and `runs/<run_id>/seed22/`.

Submission is not a pass result. Check the live reports and scheduler states
before claiming the full smoke suite or formal training has completed.
