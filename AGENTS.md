# Experiment preferences

User terminology instruction on 2026-11-04: call the downloaded frozen model
"official pretrained NeuralGCM" or "original NeuralGCM baseline", never
"old NGCM". Earlier/deleted experiments refer to residual code and residual
training results, not to the official pretrained backbone. Keep these distinct
in Chinese and English reports. All weather data, environments, caches, logs,
temporary files and experiment outputs must remain on scratch, not in home.
GitHub publishing for this project uses the user's `hsz0403` (huangsuozhi)
account. On this host, `ssh -T git@github.com` verified that identity on
2026-11-04; use `git@github.com:Reminguch/Weather_Global.git` for pushes.
The HTTPS credential cache belongs to another account and must not be used
as a substitute for the requested identity.

User instruction on 2026-11-04 specifically requests fixing the residual-Mamba
GPU restart drift, rather than merely reporting it. Use its versioned
deterministic GPU execution policy and verify fresh-process continuation.
This does not introduce a monotonic-loss or forecast-improvement gate.

User follow-up on 2026-11-04 requests automatically running a bounded 2,000-update
loss diagnostic after the restored data are complete, using ailab H200 gpu-test
jobs and checkpoint continuation. The separate DIAGNOSTIC_2K.md documents the
MENGDIW scratch diagnostic cache and deferral of the full production year-by-year
audit. Preserve the qualified architecture, loss and original LR schedule.

User instruction reaffirmed on 2026-10-19: finite numerical variance,
repeatability differences and finite restart drift are warnings, not failure or
release-blocking conditions. Preserve their measured values, reference bounds
and warnings in experiment artifacts and research notes. Do not reintroduce a
hard variance gate or make noise-derived acceptance margins an indirect gate.
Report separately whether a decrease exceeds the measured noise. Nonfinite
values, corrupt inputs and source/checkpoint integrity failures remain errors.

User recommendation added on 2026-10-28: let bounded experiments reach their
planned endpoint and evaluation whenever execution remains valid. Finite loss
increases, lack of measured descent, finite stalls, numerical drift and weak
intermediate metrics should normally be logged warnings, not reasons to abort,
block release or cancel evaluation. In stochastic or approximate-gradient
training, do not impose per-update monotonic loss improvement unless the user
explicitly requests that experiment. Preserve the scheduled optimizer update
and clearly distinguish completing training from demonstrating scientific
improvement. This supersedes the earlier permission to require a measured
decrease at every update.

Keep checks proportionate: reuse valid existing evidence, avoid redundant
qualification jobs, and summarize routine verification briefly. Do not burden
the user with repeated approval requests or long checklists for authorized work.
Retain hard stops for nonfinite computation, corrupted/mismatched inputs or
checkpoints, unusable execution, resource limits, and explicit user stop rules.

Current MGDA implementation and rerun history:
[research/directions/r005-mgda-accuracy-spectral-adjustment/devel.md](research/directions/r005-mgda-accuracy-spectral-adjustment/devel.md).

User clarification on 2026-11-05: keep the running ar_tail_k=20 diagnostic and
add an independent ar_tail_k=1 2k run. This is not a sequential curriculum.
TAIL1_COMPARE.md documents the separately versioned extension, its derived
truth prefix of 23, shared full training statistics, and common tail20
validation. Preserve the original running source snapshot and its jobs.

User update on 2026-11-05: cancel v1 residual training and evaluation and stop
its resubmission controllers. Preserve v1 checkpoints/results and shared raw
inputs. Continue with the qualified loss v2 2k H200 diagnostic described in
experimental/residualmamba_NGCM/V2_DIAGNOSTIC_2K.md. This supersedes the earlier
instruction to keep the v1 tail20 and tail1 jobs running.

User follow-up on 2026-11-05: submit an additional independent loss v2
ar_tail_k=1 2k H200 run. Keep the running v2 ar_tail_k=20 arm unchanged and
compare both on the same tail20 validation inputs. Reuse the authenticated
raw tail1 cache and shared train-only statistics, with fresh learned state.
