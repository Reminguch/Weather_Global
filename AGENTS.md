# Experiment preferences

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
