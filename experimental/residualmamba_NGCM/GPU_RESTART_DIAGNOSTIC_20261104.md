# GPU restart diagnosis and correction, 2026-11-04

The initial H200 smoke test resumed a saved update-2 checkpoint and found a
maximum absolute difference of **0.00161409378** after update 3. The checkpoint
round trip was lossless. Repeating the same compiled GPU update with identical
device inputs also produced differences, isolating an execution-time component.
The largest discrepancy was in recurrent Mamba memory, rather than parameters.
The default execution policy now requests deterministic GPU operations and
compilation choices. The model, loss, optimizer schedule and official pretrained
NeuralGCM weights are unchanged.

## Controlled diagnosis

All probes used the actual mesh-4, width-512 GraphCast-Mamba model, 796 input
channels, 203 output channels and a full 24-step BPTT update on an H200. They
started from the same saved full-model checkpoint. Inputs were synthetic, so
these are numerical execution tests, not forecast-skill experiments.

| Execution profile | Same-executable repeat, maximum absolute difference | Save/load/place and update | Fresh-process comparison | Steady update time |
|---|---:|---:|---|---:|
| [Original GPU defaults](validation/20261104_deterministic/fast.json) | 0.00157005 | 0.00161409 | Not measured | 0.35 s |
| [Autotuning disabled only](validation/20261104_deterministic/no_autotune.json) | 0.001527 | 0.001756 | Not measured | About 2.38 s |
| [Deterministic operations, default scatter lowering](validation/20261104_deterministic/deterministic_fresh.json) | 0 | 0 | Bitwise equal | 46–47 s |
| [Deterministic operations, optimized scatter lowering](validation/20261104_deterministic/optimized_fresh.json) | 0 | 0 | Bitwise equal | 4.33–4.34 s |

Slurm job **16862674** ran the first three profiles, including a separate process
for the deterministic profile. Job **16864076** ran the optimized profile twice
in separate processes. Serialization, device placement, host-input execution,
recompilation and the fresh-process optimized comparison were all bitwise equal.
These probe receipts retain the initial implementation's source identity because
they preceded policy integration. They diagnose the cause and do not replace
qualification against the integrated source.
Timings exclude the first compilation and refer only to this synthetic probe.
The optimized deterministic profile remains about 12 times slower than the
original nondeterministic update in this test; production throughput is pending.

In the original repeated-execution comparison, parameter differences reached
only 7.45e-9 and optimizer-state differences 4.66e-9. Memory differences reached
0.00157005. Thus the earlier 0.00161 number must not be described as a parameter
error of that size. Disabling autotuning alone did not remove the discrepancy.

GraphCast message aggregation uses `jraph.segment_sum`, which involves scatter
reduction. OpenXLA describes both compilation-time algorithm selection and
execution-time atomic accumulation as possible sources of nondeterminism in its
[GPU determinism documentation](https://openxla.org/xla/determinism). The evidence
here establishes execution nondeterminism and verifies a combined correction;
it does not isolate one CUDA kernel as the sole cause. The optimized scatter
option is documented in the [XLA option definitions](https://github.com/openxla/xla/blob/main/xla/xla.proto).

## Implementation

`runtime.py` applies the versioned `deterministic_gpu_v1` policy before JAX
backend initialization:

```text
--xla_gpu_enable_scatter_determinism_expander=true
--xla_gpu_exclude_nondeterministic_ops=true
--xla_gpu_deterministic_ops=true
--xla_gpu_autotune_level=0
```

Explicit conflicting values and late initialization fail with a clear error.
The execution policy is recorded alongside dependency versions and participates
in production artifact identity. Diagnostic external flags are allowed only for
the controlled probe scripts and cannot start production training.

The trainer places all resumed parameter, memory, optimizer and RNG arrays on
the device before its next compiled update. The probe showed that host inputs
could select an additional JIT variant despite identical values. Device placement
avoids that difference in the trainer, but was not sufficient to resolve the
original nondeterministic execution by itself.

## Regression verification

The CPU suite passed **23 tests**, including four new tests for flag conflicts,
early and idempotent configuration, late backend initialization and preventing
diagnostic settings from starting a production run.

The integrated H200 regression **16864762** passed the full-model smoke and the
actual production `train.run` comparison. Three uninterrupted updates and two
updates followed by a fresh-process resume to update three gave the following
results, recorded in [trainer-resume.json](validation/20261104_deterministic/trainer-resume.json):

| Checkpoint component | Changed leaves | Maximum absolute difference | Bitwise equal |
|---|---:|---:|---|
| Parameters | 0 | 0 | Yes |
| Mamba memory | 0 | 0 | Yes |
| Adam optimizer state | 0 | 0 | Yes |
| RNG | 0 | 0 | Yes |
| Chronological cursor | 0 | 0 | Yes |

The [full-model smoke](validation/20261104_deterministic/gpu-smoke.json) also
reported zero restart difference and preserved the baseline and upper-level bit
patterns. The actual trainer's second and third uninterrupted updates took
4.437 and 4.443 seconds after its first compilation. Both
[uninterrupted](validation/20261104_deterministic/trainer-full.jsonl) and
[split](validation/20261104_deterministic/trainer-split.jsonl) traces are archived.
The [official pretrained NeuralGCM pilot](validation/20261104_deterministic/native-smoke.json)
under the same policy also passed: 29 restored ERA5 frames, one full 24-step
baseline chunk, a lossless cache round trip and maximum native-evaluator
difference **0.0**. The complete job exited successfully after **10 min 32 s**.

Integrated scientific source identity:
`85c78befa884098f8f37874705ff2bda2307b0f2314dfcb1f42d3eb553135b45`.

The comparison checks raw bytes, dtype, shape and tree structure for
parameters, recurrent memory, optimizer state, RNG and the chronological cursor.
It does not substitute numerical tolerances for bitwise equality. The verified
scope is a fixed H200 environment; equivalence across GPU types, JAX/CUDA
versions or CPU versus GPU is not claimed. Long-run forecast improvement still
requires the planned production training and evaluation.
