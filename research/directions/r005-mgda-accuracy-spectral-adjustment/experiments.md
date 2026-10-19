# Attempts and evidence

<a id="e001"></a>
## e001 — Exact GC MSE and spectral AMSE-adjustment pilot

Event date: 2026-10-18
Recorded: 2026-10-18 (America/New_York)
Execution: running
Outcome: pending

**Question and change:** Test whether MGDA can improve spectral fidelity without
sacrificing exact GC accuracy. Replace the failed A/C pair with exact GC grid
MSE and the nonnegative AMSE spectral adjustment. Add actual post-update
rollout checks and backtracking rather than relying on gradient dot products.

**Setup:** Original width512/di16, residual MP2, Mamba1/BCG1/two layers,
state16/conv4/automatic rank. Initialize every residual parameter from
`res1_batch4_temporal_lr_20260908/runs/di16_joint/swa/swa_step00500-02000.pkl`,
SHA256 `25a0c980a3ae2f64d45bdfab95c876738dd9e8817edccfef3fc810a2dd47545b`.
The baseline remains frozen. All three arms have fresh optimizer, RNG, cursor
and zero initial memory; seed22, batch4, K20/BPTT24/segment96, all-step
supervision and within-segment memory carry. Retain original training inputs
and order (2015–2021 training, 2022 development evaluation).

**Objectives and controls:** `L1=exact_GC_grid_MSE`,
`L2=AMSE_spectral-MSE_spectral>=0`; full83-channel GC normalization and weights,
T179, lambda1, epsilon1e-12. No separate objective scaling or truth-power
normalization. Directions: MGDA; MSE-only; fixed sum `L1+L2`.
All arms update the second moment with `0.5*(g1**2+g2**2)`, decay0.98,
epsilon1e-8, spatial multiplier1/Mamba3 inside the metric, no first moment or
weight decay, global clip1.

**Acceptance:** Initial trial LR1e-6, at most eight halvings (nine candidates).
Measure three unchanged forward repeats each batch. Require decreases above
the maximum of absolute floor1e-8, relative floor1e-7, and four times the
observed repeatability span. Reject excessive repeatability noise above
`max(1e-8,1e-3*abs(loss))`. MGDA must lower both raw objectives; MSE control
must lower L1; fixed-sum control must lower L1+L2. All candidate evaluations
regenerate the complete trajectory from the identical original state and keys.
RMS moments are proposed once, and only the accepted parameters and regenerated
end memory are committed. Exhausted search aborts and records the failed batch.

**Stages:** Four fixed qualification batches test fixed-input baseline-feedback
and closed-loop candidates without applying updates. Passing qualification
releases three independent 200-update arms, checkpoint every50. Evaluate only
parent and final endpoints on Jan/Feb/Apr/May/Jul/Aug/Oct/Nov15, 2022,
00UTC, each40 six-hour leads, cold/full feedback and zero initial memory.
Both exact and spectral stages use those same eight origins. Final200 is
primary; no checkpoint selection or 2023 evaluation is included.

**Provenance:** Campaign root:
`artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/`.
Full current executable source and vendored GraphCast are frozen, with the
historical parent prepared-array loader restored and explicitly fingerprinted.
The old source archive omitted dependencies and environment, so exact
bitwise historical reproduction is not claimed. The campaign manifest pins
parent/config/data metadata/stats/evaluation origins and frozen-source digest.
Frozen source digest:
`c9fc590940a97931ebdc22122966071cb8800f82ee7ee1e4e1cd66611ffa61c4`.
Qualification15787947 and afterok release15787948 submitted at12:28EDT,
October18. Both are pending; training/evaluation IDs are created conditionally
by the release stage after successful qualification. The execution field
includes this queued stage and does not assert GPU work has started.

**Evidence:** [Driver](../../../scripts/experiments/run_v24_mse_amse_excess_pilot.py),
[losses](../../../src/models/mamba/v24_Ilya/training/mse_amse_excess_loss.py),
[acceptance](../../../src/models/mamba/v24_Ilya/training/mgda_acceptance.py).

**Observation:** 189 targeted regression checks passed, with32 campaign checks
also passing (counts overlap). Frozen-source/input verification, finite parent
parameter load, real historical prepared-loader startup, all three configuration
parses, eight-origin coverage, shell syntax and Slurm test-only preflight pass.
GPU qualification and scientific evaluation are pending.

**Decision / next action:** Qualification and its conditional release pipeline
are submitted. Review the pilot before authorizing any larger campaign.
Existing experiments remain preserved.

### 2026-10-18, 12:28 EDT — Submission

- Qualification: one A10080,8CPUs,320GiB host RAM,4h. Historical same-shape
  di16 job15515799 used234.71GiB and22h53m for1,900 updates.
- Release:2CPUs,8GiB,15min, afterok15787947. It requests unthrottled three-arm
  training only after passing the numerical proof. Training memory is rounded
  from1.35×max(historical, measured peak); time from200×measured p90×1.4 plus
  15min startup, bounded to512GiB/48h. Evaluation arrays are unthrottled,
 16GiB/62min each; CPU report12GiB/30min.
- [Qualification submission](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/submissions/qualify.json),
  [release submission](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/submissions/release.json),
  [CPU/runtime preflight](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/checks/pre_submit.json),
  [scheduler preflight](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018/checks/scheduler_preflight.json).
- An interrupted preparation under restricted external-filesystem access is
  retained in the sibling `_preparation_sandbox_attempt` directory. It has no
  submitted jobs and is not a scientific run. The final campaign was prepared
  afresh with verified access to the unchanged historical Zarr v3 store.
