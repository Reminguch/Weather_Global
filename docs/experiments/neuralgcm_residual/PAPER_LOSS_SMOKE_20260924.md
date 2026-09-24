# Detailed smoke tests for frozen-NGCM K=1 / K=2 training

**Latest update, September 24, 2026:** width128/d16 completed its full detailed
**K=1 and K=2** tests successfully in **23 min 10 s** (job 14374769, exit 0).
The production CLI test also passed for both horizons in **17 min 43 s**
(job 14374768, exit 0), including exact independent-process checkpoint/metric
resume and fresh-process baseline equality.

**User-authorized startup change:** all eight production jobs now have their
smoke dependencies removed. The completed width128/d16 and CLI tests provide
the representative startup evidence. Other architecture tests continue in
parallel. At release, all eight jobs were pending GPU scheduling, with Slurm
`DEPENDENCY=(null)`. Passing the representative architecture does not certify
width256 memory use or architecture-specific numerical checks; those checks
continue without blocking startup, as requested.

The pinned execution configuration resolves the earlier cross-process discrepancy
in the tested cases. Its individual cause has not been isolated, because three
execution settings were pinned together. Seven CPU tests and the independent
real-data pilot also passed.

[Production CLI report](evidence_20260924/cli_smoke_v2.json) ·
[width128/d16 complete K=1/K=2 report](evidence_20260924/w128_d16_full.json) ·
[User-authorized release record](evidence_20260924/startup_release.json)

The [experiment definition](NGCM_ALIGNMENT_K2_20260924.md) describes the exact
loss, scales, frozen model and memory-only gradient contract. Every GPU smoke
uses Slurm `gpu-test`, an A100 80 GB request, no explicit partition and a one-hour
limit. The scheduler selects `gputest`. Production training uses `gpu-short`.

## Completed independent pilot

Job **14373556** completed successfully on real ERA5 data in **245.26 seconds**.
It used a separate ten-snapshot training-only calibration, before depending on
the production scale artifact. Its outputs are never used as production
checkpoints or loss statistics.

| Check | Measured result |
| --- | --- |
| Zero-head complete native state at +6 h | Exact match, maximum absolute difference 0 |
| Zero-head complete native state at +12 h | Exact match, maximum absolute difference 0 |
| K=1, three real optimizer updates | Finite loss/gradient; maximum parameter change 3.0015e-6 |
| K=2, three real optimizer updates | Finite loss/gradient; maximum parameter change 3.0015e-6 |
| Frozen NGCM parameter hash | Unchanged |

The first warmup update has zero learning rate. Later updates change the head.
K=1 pilot losses were 4294.5024, 4294.5024 and 4294.3145. K=2 losses were
3943.6177, 3943.6177 and 3943.4265. These few updates are execution checks, not
forecast-skill evidence. They use pilot scales and must not be compared directly
with production-scale losses.

[Machine-readable pilot report](evidence_20260924/independent_pilot.json).

The shared production loss scales were then fitted successfully by CPU job
**14374045**, using 60 training snapshots and 24 h differences. Fitting took
7.44 seconds (16 seconds for the Slurm job). The same
[scale artifact](evidence_20260924/loss_statistics.json) is used for every K and
architecture. The numerical tests do not depend on a missing original gin file.

## Completed CPU checks

Command:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv_neuralgcm/bin/python -m pytest -q \
  tests/neuralgcm_residual/test_paper_trajectory.py \
  tests/neuralgcm_residual/test_gradients.py
```

**7 passed** in 10.17 seconds. Checks include averaging batch/time bias before
squaring, squared cloud amplitude factors, finite derivatives at zero spectrum,
correct pooled population moments including mean shifts, and agreement between
the host-tape memory gradient and an independent differentiable toy unroll.
These CPU checks support, but do not replace, the real-model tests below.

## Detailed real-model matrix

| Width | `d_inner` | GPU test job | Horizons | Status |
| ---: | ---: | ---: | --- | --- |
| 128 | 16 | 14374769 | K=1 and K=2 | Passed both; completed |
| 128 | 32 | 14374770 | K=1 and K=2 | K=1 passed; K=2 running |
| 256 | 16 | 14374778 | K=1 and K=2 | Running |
| 256 | 32 | 14374779 | K=1 and K=2 | Running |

The completed width-128 K=1 gradient comparisons have relative L2 errors of
1.052e-6 and 9.125e-7. Both have exact zero-head identity, zero physical-feedback
gradient, nonzero hidden-network gradients and exact update-10 to update-20
process replay. The width128/d16 K=2 gradient check also passed with relative
L2 error **5.884e-7** and gradient cosine **0.9999999999998734**. Its second step
consumes the corrected first state, verified exactly. See the
[K=2 numerical-check report](evidence_20260924/w128_d16_K2_checks.json);
the [complete K=2 training/resume report](evidence_20260924/w128_d16_K2.json)
also passed, including 20 actual updates and exact replay from update 10.

Each job performs:

1. Exact zero-head state identity and forecast-clock checks at the scored leads.
2. A nonzero head-weight/bias and nonzero-memory probe. A bias-only perturbation
   would leave hidden-layer gradients zero, so it is insufficient for this test.
3. An independent NumPy float64 evaluation of all five loss terms, checked
   against the JAX result (`rtol=3e-5`, `atol=2e-5`). This certifies the documented
   formula, not parity with unpublished paper bindings.
4. Direct automatic differentiation of the entire recorded residual recurrence,
   compared with the custom reverse tape. Required relative L2 error is at most
   2e-5 and gradient cosine at least 0.99999, with nonzero hidden-layer gradients.
5. A perturbation check requiring exactly zero gradient through physical inputs
   and the frozen physical baseline.
6. For K=2, verification that step two's features come from the corrected first
   prediction, rather than ERA5 or an uncorrected forecast.
7. Twenty actual optimizer updates on two fixed representative origins, with
   finite gradients and nonzero residual parameter movement.
8. A new Python process loading the update-10 checkpoint and replaying updates
   11–20 on the same allocated GPU. Parameters, complete Adam state, RNG and all
   recorded loss/gradient metrics must match exactly.
9. A final frozen-backbone hash check and measured device-memory report.

Keeping reference and resume in the same allocation isolates restart correctness
from differences between GPU hardware. It does not promise bitwise identity
across GPU models or software versions.

Earlier submissions 14374090/96/97 and 14374121 were canceled when review found
that the nonzero-gradient probe only changed the zero head's bias. The revised
jobs above also perturb its weights. Production initialization is unchanged.
Those canceled tests are not counted as passes.

## Production entry-point and physical validation test

Job **14374768 passed**, exercising the actual production command for both K values.
For each K it compares a continuous 12-update run with a run stopped at update
10 and resumed in a fresh process. It also injects uncommitted metric tails to
verify recovery. It checks:

- Exact residual parameters, Adam state, RNG, metric history and best-loss value.
- No `DONE.json` for an intentionally interrupted run.
- Preservation of discarded log tails in audit files.
- Baseline and best-checkpoint creation, plus successful final completion.
- Finite, nonnegative physical RMSE for all seven fields, all 37 pressure levels
  and each lead, with shapes `[1, 37]` and `[2, 37]`.
- Unchanged pretrained NGCM weights.

The initial CLI job 14374360 completed its continuous K=1 run and wrote physical
validation outputs, but a fresh process produced a slightly different frozen
baseline (loss 5850.3452 versus 5850.5356). It was canceled before declaring any
resume success. The current submission pins `PYTHONHASHSEED=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `--xla_gpu_autotune_level=0`, in addition to
the existing deterministic-operation flags. These settings are recorded in the
run identity. The replacement test passed complete cross-process K=1/K=2 parity.

The compiler choice is motivated by [OpenXLA's determinism guidance](https://openxla.org/xla/determinism).
Autotuning can select different floating-point reduction kernels in separate
compilations. The successful rerun confirms equality under the pinned environment for the
tested cases; it does not identify which individual setting caused the change.

## Training submissions and reproducibility

The first eight dependent production submissions (14374583–14374590) were
canceled together with the superseded tests after the baseline mismatch. They
never began training. Their [submission record](evidence_20260924/training_jobs.json)
is retained as canceled, not as evidence of active production runs.

All new local outputs are under `logs/ngcm_aligned_20260924/`, separate from
existing v2 jobs and artifacts. Revised tests execute immutable
`source_train_v4`, source ID
`f4346f67bf51ca59dcd0d85d67dbaaebd802f2fab77c6c5ce330c059d9c44423`.
Eight replacement 2,000-update jobs now use this same pinned execution policy.
Their original per-architecture dependencies have now been removed under the
[user-authorized representative-test policy](evidence_20260924/startup_release.json).
Only scheduler resource/priority availability remains before startup.

| Width | `d_inner` | K=1 job | K=2 job |
| ---: | ---: | ---: | ---: |
| 128 | 16 | 14374862 | 14374863 |
| 128 | 32 | 14374864 | 14374865 |
| 256 | 16 | 14374866 | 14374867 |
| 256 | 32 | 14374868 | 14374870 |

[Original submission commands and current dependency override](evidence_20260924/training_jobs_v2.json).

These tests assess loss arithmetic, the chosen gradient contract and operational
correctness. Long-run stability and forecast skill require training and held-out
validation. Existing published physical-time plots describe older v2 checkpoints.
