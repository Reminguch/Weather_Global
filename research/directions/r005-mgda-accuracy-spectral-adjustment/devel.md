# Development — MSE / AMSE-adjustment MGDA

Updated: 2026-10-18 (America/New_York)

## Current implementation

New objective pair `mse_amse_excess` routes exact grid MSE and the stable,
nonnegative spectral adjustment through the existing explicit two-gradient
tape. Existing `amse_split` and GC/power modes retain their original behavior.
The optimizer direction modes share the pair second-moment estimator.

The new measured-loss acceptance helper is host-side. Each candidate reruns
the physical trajectory; candidate trials never become starting points for
later trials. Accepted candidate end memory is committed, and RMS count advances
once. `backtracking_trials.jsonl` records all candidates and numerical floors;
`failed_update.json` records an exhausted search without accepting the batch.
Exact checkpoint resume trims these logs to the checkpoint boundary.

`qualify_mgda_training` initializes the parent, tests four unchanged batches
with fixed inputs and closed-loop feedback, and discards all proposed updates.
Qualification measures actual step time and peak memory for production budgets.

## Validation and blockers

CPU objective, optimizer, acceptance, transaction, campaign and report checks
pass:189 targeted regressions and32 campaign checks (overlapping counts).
Frozen-runtime startup, finite parent checkpoint, historical prepared-loader
compatibility, coordinate coverage and scheduler test-only validation pass.
GPU qualification15787947 is queued, followed by release15787948. There is no evidence yet that the
stopped-feedback gradients yield acceptable regenerated-rollout steps on real
weather batches; an inability to do so is a diagnostic result, not permission
to force steps or silently alter objectives.

## Resume

Use the new campaign driver, its manifest and frozen source, rather than
editing an already submitted snapshot. Inspect `submissions/`, `checks/`,
`failures/`, and `runs/*/failed_update.json`. Do not automatically retry failed
stages. The terminal report is `evaluation/report/pilot_report.json`.

No larger continuation is authorized by this pilot. Review exact GC loss
against the original parent and both controls, and power/coherence diagnostics,
before proposing the next scientific step.

Source digest: `c9fc590940a97931ebdc22122966071cb8800f82ee7ee1e4e1cd66611ffa61c4`.
The historical evaluation store is Zarr v3 with `zarr.json`; restricted sandbox
access stalled on its external filesystem, while approved access worked. Future
submission-host verification needs the same external-filesystem/Slurm access.
The pilot snapshot and scientific data were not altered to work around this.
