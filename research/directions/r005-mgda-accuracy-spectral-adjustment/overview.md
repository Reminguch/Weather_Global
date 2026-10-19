# r005 — Can MGDA improve spectral fidelity while preserving exact GC accuracy?

Category: training
Status: active
Updated: 2026-10-18 (America/New_York)

## Question and hypothesis

Can a continuation from a good MSE-trained residual model improve spectral
fidelity without losing exact GraphCast grid accuracy? Test MGDA on exact GC
grid MSE and the nonnegative spectral adjustment
`AMSE_spectral - MSE_spectral`, with measured finite-step acceptance.

Both terms use the existing full-field difference-standard-deviation, variable
and pressure weights. The adjustment uses matching spectral MSE, never grid MSE,
and is computed directly to avoid cancellation. Their sum is a grid-MSE-based
adjusted objective; it is not exactly the existing spherical AMSE.

## Motivation and comparison

The earlier GC/power MGDA continuation used a different power-error surrogate.
The October 12/13 amplitude/pattern split optimized A and C, with GC loss only
diagnostic. That split substantially degraded forecast skill in both arms.
Its first-order checks did not enforce decreases after finite parameter updates.
See the preserved [diagnosis](../../../current_experiments.md#split-amse-failure-diagnosis--2026-10-17)
and [original split protocol](../../../docs/experiments/v24_amse_split_mgda_submission_plan_2026-10-12.md).

The new pilot compares MGDA, MSE-only, and a fixed sum of the new pair. All use
the same pair-gradient RMS preconditioning rule, so the control does not change
to a single-gradient second-moment estimator.

## Current understanding

The user authorized implementation and submission on October 18. Qualification
job15787947 and dependent release15787948 were submitted at12:28EDT and are
pending. Numerical qualification and scientific results are pending. The pilot uses one original
width512/di16 MSE checkpoint, not either degraded amplitude/pattern endpoint.
The qualification is discarded and does not consume production updates.

Each MGDA candidate must decrease both measured objectives on the same
regenerated K20 trajectory, starting from identical memory and random keys.
The controls accept decreases in their own objective. This is a per-batch
acceptance condition, not a guarantee for future batches or validation skill.
Zero directions or repeated acceptance failure stop the pilot and retain the
failure evidence; no objective weight is forced to keep training moving.

## Next decision

Review qualification and the three 200-update endpoints on eight matched
seasonal 2022 starts and 40 six-hour leads. Exact GC loss is primary; power and
coherence are secondary. Compare against both the original parent and matched
controls. No 2,000-update extension, di32 arm, SWA, or automatic retry is enabled.

## Records

- [Attempt and evidence](experiments.md#e001)
- [Implementation and resumption](devel.md)
