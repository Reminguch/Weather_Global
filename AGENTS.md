# Experiment preferences

User schedule correction on 2026-11-12: full-GC fine-tuning uses **constant LR
`3e-7` from the first update**, with no warmup or cosine decay. This applies to
the five-lambda Della sweep and the portable full-GC horizon handoff for LH.
Start all five lambda arms fresh, including lambda0; preserve earlier cosine
checkpoints as historical evidence and never label them constant-rate runs.
GC–Mamba's separate M1-matched schedule remains unchanged. See the
[constant-rate protocol](docs/experiments/v24_full_gc_constant_lr3em7_2026-11-12.md).

User instruction added on 2026-11-12: exclude the full-GraphCast fine-tuning
results trained at peak LR `1e-5` in
`res1_fresh_gc_mamba_gc_lambda_20261107` from scientific comparisons, tables,
plots, rankings and conclusions. They are rejected training settings, retained
only for provenance or an explicitly requested failure diagnosis; do not pull
them up as comparison results. This includes their continuations and SWAs.
Use constant LR `3e-7` as specified above for full-GC fine-tuning going forward. This does not change
GC–Mamba's separate M1-matched learning-rate schedule. Do not relabel high-LR
weights or their continuations as fresh `3e-7` experiments. See
[comparison policy](docs/experiments/v24_full_gc_comparison_policy_2026-11-12.md).
This explicit user exclusion is not an automatic finite-loss acceptance gate.

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
