# Validation record — 2026-11-04

The new open-loop implementation has passed the historical checks below. This is an
implementation and smoke-test result, **not a demonstration of improved weather
forecast skill**. The official pretrained NeuralGCM weights are unchanged.

The initial GPU restart discrepancy has since been isolated and a deterministic
execution policy has passed independent probes and the actual trainer's
fresh-process continuation. See
[GPU_RESTART_DIAGNOSTIC_20261104.md](GPU_RESTART_DIAGNOSTIC_20261104.md) for the
cause, correction and fresh-process verification. Historical receipts are kept
so the original warning remains auditable.

## Current evidence, after the runtime correction

H200 job **16864762** completed with exit code **0** in **10 min 32 s**, on
ailab. All three real-model receipts below refer to the integrated source
`85c78befa884098f8f37874705ff2bda2307b0f2314dfcb1f42d3eb553135b45`.

| Check | Result | Evidence |
|---|---|---|
| CPU suite after the correction | 23 passed, including execution-policy regression checks | [unit-tests.txt](validation/20261104_deterministic/unit-tests.txt) |
| Full GraphCast-Mamba GPU smoke | Passed; checkpoint continuation bitwise equal, max difference 0 | [gpu-smoke.json](validation/20261104_deterministic/gpu-smoke.json) |
| Actual production trainer across fresh Python processes | Uninterrupted 3 versus split 2+1; parameters, memory, Adam, RNG and cursor all bitwise equal | [trainer-resume.json](validation/20261104_deterministic/trainer-resume.json) |
| Official pretrained NGCM with deterministic GPU policy | Passed on 29 real ERA5 frames; native evaluator difference 0; cache round trip bitwise equal | [native-smoke.json](validation/20261104_deterministic/native-smoke.json) |
| Source and execution-policy qualification | Three smoke receipts checked against this source and policy, with a fresh CPU suite | [qualification.json](validation/20261104_deterministic/qualification.json) |

The correction addresses repeatability in the tested H200 environment. It does
not establish equality across hardware or dependency versions, production data
readiness, or improved forecast skill. The optimized deterministic update took
about 4.3 seconds in the controlled probe, compared with 0.35 seconds using the
original nondeterministic configuration. See the diagnostic report for the full
timing comparison and evidence distinguishing execution from serialization.

## Initial implementation evidence, before the runtime correction

| Check | Result | Evidence |
|---|---|---|
| CPU unit suite | 19 passed, 8 dependency deprecation warnings, 10.20 s | [unit-tests.txt](validation/20261104/unit-tests.txt) |
| Actual GraphCast-Mamba model on ailab/H200 | Passed, Slurm 16860678, exit 0, job elapsed 1 min 44 s | [gpu-smoke.json](validation/20261104/gpu-smoke.json) |
| Actual ERA5 + official pretrained NGCM | Passed, Slurm 16860946, exit 0, job elapsed 14 min 47 s | [native-smoke.json](validation/20261104/native-smoke.json) |
| Matching source qualification | Passed, both smoke receipts and rerun unit suite checked | [qualification.json](validation/20261104/qualification.json) |
| Two-frame ERA5 download and resume | Passed, no second download of completed frames | [download-pilot.json](validation/20261104/download-pilot.json) |

Tested scientific source identity:
`db4d8f39fb7aca18f919f3320d4e3e57d971d39d569799f3f11b353bd748f70c`.

## What the real-model tests exercised

The GPU test initialized the actual 796-input/203-output, mesh-4, width-512
GraphCast-Mamba model. It has **11,814,031 parameters** and **99,692,544 bytes**
of recurrent state. The zero head reproduced the supplied baseline bitwise,
including signed zeros. The unchanged upper levels also passed bitwise checks.
Memory became nonzero after a step. Three full 24-step BPTT optimizer updates
completed with finite loss, gradients, parameters and optimizer/memory state.

The synthetic losses were 0.0510000102, 0.0510000102 and 0.0509608388. The first
update uses LR zero, as specified by the warmup schedule. These numbers test the
training machinery; they are not ERA5 evaluation results.

The initial GPU test also serialized a checkpoint after update 2 and replayed update 3.
**GPU continuation was not bitwise identical in that original configuration:** the largest absolute difference
across parameter, memory and optimizer arrays was **0.00161409378**. This was
finite and was initially recorded as a warning under `AGENTS.md`. The user then
explicitly requested a fix. Subsequent controlled probes isolated execution
nondeterminism, with the largest differences in Mamba memory and a lossless
checkpoint round trip. The independent CPU split-versus-uninterrupted training test was
bitwise identical, including the optimizer, memory, RNG and cursor. Complete
resume state was already implemented; the added runtime policy addresses GPU
numerical execution. The diagnosis does not attribute the entire discrepancy to
one isolated CUDA kernel.

The native test used 29 restored ERA5 frames, all seven weather fields and the
official checkpoint. It generated the three independent one-step forecasts and
the 21-step native tail, for a `[24,7,37,128,64]` chunk. Cache serialization was
bitwise lossless. Its tail matched the existing native evaluation routine with
**maximum difference 0.0**. Checkpoint geography and actual ancillary features
also passed shape/finiteness checks.

An earlier native-test attempt (16860766) stopped because the test fixture
stored relative frame paths. That fixture was corrected to use verified absolute
paths, and the successful run above exercised the fix. The data files were not
corrupt. The first Slurm download probe (16858654) could not resolve Google
Cloud; downloads were moved to the established network-enabled staging node.

## Still required before production training and scientific claims

- Completion and verification of all 13,148 ERA5 frames for 2015–2023. The
  detached four-worker monthly download is still running on `della-vis1`.
- An authorized DABANIN cache directory and recorded quota audit, as requested
  by the specification. The current session has not established that access.
- Full train-only statistics, including recorded cloud scale floors, and all
  424 training / 60 validation cache chunks.
- Production T1 on three randomly selected train chunks and one validation
  chunk, plus the full-cache native-tail checks. The real pilot above is one
  training chunk and does not replace this requirement.
- The zero-head census on all validation segments. Any field/level dominance
  above 10% globally or within a step class needs the specified recorded
  decision before training.
- The year-by-year native baseline audit required by README section 14.3.
- Actual 10,000-update training, fixed-window SWA and full E2 forecasts. No
  production training or forecast-improvement claim is included in this commit.
- Original E1 GC anchor manifest, actual GC dataset timestamps and recorded GC
  origin timestamps. They are absent from the recovered checkout, so E1 cannot
  yet be run and no replacement origins have been guessed.

The load-only model package enforces the production data, provenance, cache,
census, qualification and baseline-audit prerequisites. The download and
partial-data smoke helpers are separate and cannot mark a partial store as a
complete production dataset.
