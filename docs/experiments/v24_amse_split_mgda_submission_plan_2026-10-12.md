# AMSE split MGDA submission plan for di16 and di32

Prepared 2026-10-12. Scope: two 2,000-update development runs that apply MGDA
to the amplitude and pattern terms of the existing spherical AMSE loss.
Implementation and submission were authorized on 2026-10-12. The implementation
passed CPU checks, source was frozen, and the initial six Slurm stages were
submitted. Following the revised budget, each width proceeds to
2,000 updates after technical qualification. Step 500 is an intermediate
evaluation checkpoint and does not gate continued training.

Proposed campaign root:
`artifacts/checkpoints/v24_Ilya/res1_amse_split_mgda_di16_di32_20261012`.
Use a new root if that path has acquired different contents before preparation.

## Objectives and optimizer

Split the existing full AMSE implementation at `amse_lambda=1`:

\[
A=\left\langle\sum_\ell(\sqrt{P_f(\ell)}-\sqrt{P_y(\ell)})^2\right\rangle_w,
\qquad
C=\left\langle\sum_\ell2\max(P_f(\ell),P_y(\ell))(1-\rho_\ell)\right\rangle_w.
\]

Here the weighted reduction preserves the existing GraphCast variable,
pressure-level and difference-standard-deviation conventions, over all 83
channels, examples and supervised steps. Use the existing global spherical
T179 transform, degree zero included, and its stabilized implementation with
`power_epsilon=1e-12`. The displayed equations omit that numerical regularizer.
The executable definition is the `amplitude` and `pattern` outputs of
[spectral_loss.py](../../src/models/mamba/v24_Ilya/training/spectral_loss.py).
No variable masks, degree tapers or division by truth power are introduced.

Calibrate on the same 32 training batches at each initialization, with no
parameter updates. Freeze one scalar shared by both objectives and both widths:

\[
s=\frac{1}{64}\sum_{d\in\{16,32\}}\sum_{b=1}^{32}(A_{d,b}+C_{d,b}),
\qquad L_1=A/s,\quad L_2=C/s.
\]

Require finite positive `s`, and retain every batch identity and raw value.
This common scaling preserves the relative units of A and C. Separately
normalizing their means would define a different optimization problem.

Use the existing shared RMS metric and two-gradient MGDA solver. For
`g1=grad(L1)`, `g2=grad(L2)`, update the RMS moment with
`0.5 * (g1**2 + g2**2)`. Include the Mamba-to-spatial multiplier of 3 in the
positive diagonal metric D before solving. With `hij = gi.T @ D @ gj`, choose

\[
\alpha=\operatorname{clip}_{[0,1]}
\frac{h_{22}-h_{12}}{h_{11}+h_{22}-2h_{12}},
\]

using the existing deterministic alpha=0.5 convention when the denominator is
zero. Compute that denominator directly as `(g1-g2).T @ D @ (g1-g2)`, as the
existing solver does, to avoid cancellation in the displayed equivalent sum.
Apply the metric to the combined gradient after its global-norm clip.
Keep RMS decay 0.98, epsilon 1e-8, clip 1, no momentum and no weight decay.
Retain the checks on proposed directions and actual rounded parameter updates
from [mgda_optimizer.py](../../src/models/mamba/v24_Ilya/training/mgda_optimizer.py).
These check first-order descent, not guaranteed loss reduction on later batches.

Allow the full alpha interval, with no minimum weight or stall fallback in this
experiment. Record A, C, A+C, their normalized values, alpha, both gradient norms,
gradient cosine, metric products, actual update norm, clipping and zero-update
counts. Record exact original GC loss separately. Plot small update norms as
well as exact stalls: MGDA can stall when an objective gradient vanishes or
the gradients oppose each other.

## Matched starting models and training settings

Use the matched batch4 joint pair from
`artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/`:

| Arm | Initialization relative to that directory | SHA256 |
| --- | --- | --- |
| di16 | `di16_joint/swa/swa_step00500-02000.pkl` | `25a0c980a3ae2f64d45bdfab95c876738dd9e8817edccfef3fc810a2dd47545b` |
| di32 | `di32_joint/swa/swa_step00500-02000.pkl` | `46220eda60f80b22fccc4a205a345ada4d9afda399c410bb1d9354c0392659f9` |

Both files exist and their hashes were checked for this plan. They share the
original joint pretraining recipe. The later mature di16 checkpoint used in
the GC/power MGDA experiment has a different training history; it is a
historical reference, not the starting point for this width comparison.

| Setting | Both arms |
| --- | --- |
| Frozen baseline | GraphCast Small, 1 degree, mesh5, width512, 16 message-passing steps |
| Residual network | Width512, 2 message-passing steps; Mamba1, BCG1, state dimension 16, convolution width 4, 2 layers, automatic rank |
| Width difference | `temporal_d_inner=16` or `32` only |
| Training sequence | Batch4, BPTT24, autoregressive tail20, segment96, all 24 steps supervised |
| State and feedback | Carry within each segment; reset at segment boundaries; stopped physical feedback (`closed_loop_sg`) |
| Precision | BF16 neural execution; FP32 objective and existing physical/state boundaries |
| Randomness | Seed22 and identical training batch order |
| Spatial learning rate | 1e-6 to 1e-5 over updates 1–100; cosine to 3e-6 at 1000; constant afterward |
| Mamba multiplier | 3, included in the shared metric |
| Endpoint | Native update 2,000; save every 50 so qualification and continuation use one immutable configuration |

The schedule keeps its original 2,000-update horizon. Qualification and resume
must preserve the schedule and completed-update count. Initialize optimizer, RNG, cursor
and recurrent state once at update 0. Calibration state is discarded.
Both arms undergo the same transition from their historical loss evaluation
precision to the new FP32 objective.

## Implemented submission requirements

1. Expose A and C through the training step with explicit objective identities
   and a new common-scale calibration schema. The current
   [mgda_step.py](../../src/models/mamba/v24_Ilya/training/mgda_step.py)
   assumes GC/power and must not label amplitude as GC loss. Rename optimizer
   diagnostics consistently and keep exact GC loss as a separate diagnostic.
2. Add a dedicated driver and Slurm wrapper, provisionally
   `scripts/experiments/run_v24_amse_split_mgda.py` and
   `scripts/experiments/v24_amse_split_mgda.slurm`. These paths are proposed,
   not executable entry points today. The existing `run_v24_mgda.py` launches
   different objectives, arms and endpoints.
3. Reuse the runner's existing operational `stop_after` mechanism so that
   qualification uses the production configuration with `max_steps=2000`,
   stopping at 100 without changing the schedule or resume identity. Support
   a separate shadow restart so the canonical trajectory cannot be overwritten.
4. Check A+C against the existing lambda=1 loss and its gradients; exercise
   zero-power/equal-field cases, solver endpoints, a vanishing gradient,
   conflicting gradients, frozen baseline parameters and checkpoint state
   restoration. Validate frozen child imports and dependency generation.
5. Prepare and dry-run the complete bounded workflow. Freeze source under
   `src`, `scripts` and `third_party/graphcast`; pin configurations, checkpoint
   hashes, statistics, training manifests and both evaluation cohorts. Verify
   coordinate coverage and consolidated Zarr v2 or v3 metadata. Follow
   [the frozen workflow contract](frozen_experiment_workflows.md).

Run Python through `scripts/graphcast_env.sh`. Jobs must execute their frozen
entry points and vendored GraphCast with `PYTHONSAFEPATH=1`. GPU stages must
unset an inherited `JAX_PLATFORMS=cpu`; preserve the working XLA setting
`--xla_gpu_enable_triton_gemm=false`. Record environment versions and GPU type.

## Submission stages and dependencies

| Stage | Jobs or tasks | Successful prerequisites | Work and output |
| --- | --- | --- | --- |
| Calibration | Separate di16 and di32 GPU jobs | Prepared frozen experiment | 32 forward batches each; raw A/C and batch provenance |
| Combine calibration | 1 CPU job | Both calibration jobs | Freeze common s and resolved objective configuration |
| Qualification | Separate di16 and di32 GPU jobs | Combined calibration | Canonical updates 0–100; save 50 and 100; shadow restart 50–100; serialization and resource evidence |
| Release | 1 CPU job | Both qualifications | Verify technical checks and size each continuation from measurements |
| Training continuation | Separate di16 and di32 GPU jobs | Release | Resume each canonical 100 checkpoint for 1,900 more updates, ending at 2,000 |
| Exact evaluation | Array 0–5 | Both training runs complete | di16/di32 at 0/500/2000; 32 matched 2022 starts, 40 leads |
| Spectral evaluation | Array 0–5 | Both training runs complete | The same six checkpoints; 8 matched 2023 starts, 40 leads; spectra and accuracy together |
| Report | 1 CPU job | Both evaluation arrays complete | Final and intermediate metrics, absolute spectra, RMSE rollouts and training diagnostics |

Each continuation is one uninterrupted job after qualification, with checkpoint
writes every 50 updates. Evaluation of step 500 runs afterward alongside the
other declared checkpoints, so evaluation does not interrupt either trajectory.
The native 2,000-update checkpoint is primary; there is no selection of an earlier
checkpoint based on development scores. SWA is outside this initial submission;
the retained checkpoints allow a separately requested secondary analysis.

Use `afterok` dependencies, `--kill-on-invalid-dep=yes`, recorded job IDs and
duplicate-submission guards. Arrays are unthrottled. Failed stages produce
diagnostic status and leave dependent work blocked; there are no automatic
retries or additional training arms. A status command must identify missing
outputs even when the successful-completion report cannot run.

Qualification restores parameters, RMS moments, RNG, data cursor, recurrent
state and update index. Save/load integrity and discrete state must match
exactly. Compare subsequent losses, updates and the step 100 endpoint between
canonical and shadow paths; retain numerical drift measurements. The prior
BF16 MGDA restart showed parameter relative-L2 drift of 3.38e-5 and recurrent
state drift of 2.09e-2, so bitwise equality after recomputation is inappropriate.
Record numerical thresholds before GPU qualification. Under the user-authorized
2026-10-13 policy revision, finite replay drift above these thresholds emits a
warning and is retained in the qualification report without blocking promotion.
Missing state, changed provenance, nonfinite values, and non-exact serialization
remain hard failures. Continue the canonical 100 checkpoint, never the shadow one.

No forecast-skill threshold is required to release updates 101–2000. This first
gate checks implementation, restart and resources. A configuration or source
change requires a new frozen experiment. The explicit 2026-10-13 recovery imports
the saved canonical and shadow trajectories, verifies state-preserving relocation,
and reassesses qualification under the advisory policy without repeating updates.

## Resource requests and batch accounting

All GPU jobs request one A100 80GB device, 8 CPUs and account `dabanin`, using
the established `--gres=gpu:1 --constraint=gpu80` convention. Host memory below
is separate from GPU memory. Record GPU telemetry during qualification.

| Stage | Proposed host memory per job | Walltime per job | Basis |
| --- | --- | --- | --- |
| Calibration di16 / di32 | 128 / 160 GiB | 90 min | Forward-only; nearby 32-update AMSE checks peaked around 92 GiB; di32 allowance is unmeasured |
| Qualification di16 / di32 | 320 / 384 GiB | 210 / 240 min | Prior 100-update plus restart MGDA qualification used 223.16 GiB and 136 min; extra allowance for the new loss and width |
| Combine / release | 8 GiB, 2 CPUs | 20 min | Metadata, checks and resource calculation |
| Continue 100 to 2000 | Measured per arm; around 320 GiB if historical peak holds | Measured per arm; around 31–32.5 h with old timings | Formula below |
| Exact evaluation | 16 GiB | 90 min | Previous 32-start tasks took 59–60 min and 7.5–7.7 GiB |
| Spectral evaluation | 16 GiB | 60 min | Previous 8-start tasks took 15.5–15.7 min and 7.5–7.7 GiB; allow spherical diagnostics expanded from 11 fields to all 83 channels |
| Report | 12 GiB, 2 CPUs | 30 min | CPU aggregation and plotting |

For each continuation, request host memory
`ceil(1.35 * max(234.225, measured_peak_GiB) / 16) * 16` GiB and time
`ceil(1.35 * (1900 * p90_update_seconds + 1200) / 1800) * 30` minutes.
Use native-grid qualification measurements including data waits and checkpoint
I/O, with compilation separated into the allowance. Account for slower restart
execution. If the measured shape exceeds these assumptions, revise the request
from evidence before release. Use the existing QOS rule: `gpu-short` through
480 minutes, `gpu-medium` above that within its established limit.

Historical sizing evidence: GC/power MGDA job 15190770 completed 2,000 updates in
23h50 with approximately 231 GiB peak host memory; MSE job 15190771 took 15h36.
Comparable AMSE/MSE jobs 14147422/24 and 14397797/99 reached 234.225 GiB. Old MGDA
logged median end-to-end updates of 42.37 seconds, versus 27.56 for MSE. These
are proxies: spherical A/C gradient computation has not been timed at either
new width. Resource evidence lives with the prior
[MGDA qualification](../../artifacts/checkpoints/v24_Ilya/res1_mgda_gc_power_2k_recovery_20261007/checks/mgda_smoke.json)
and [training check](../../artifacts/checkpoints/v24_Ilya/res1_mgda_gc_power_2k_recovery_20261007/checks/mgda_train.json).

Per width, the model receives 2,000 updates, comprising 100 qualification updates
and 1,900 continuation updates. An additional 50 updates are executed only for
the shadow restart proof: **2,050 optimizer batches plus 32 forward calibration
batches per width**, or 4,100 optimizer batches and 64 calibration batches total.
Each optimizer batch contains 4 sequences of 24 supervised steps: 192,000
supervised step instances contribute to each final model. Overlap means these
are not 192,000 independent forecasts. Promoting qualification saves 100 repeated
updates per width compared with discarding it and restarting from zero.

Further speed work may share the spherical transform and forward tape between
A and C, or vectorize the two VJPs. Benchmark against a sequential reference
and retain it only if gradients agree and memory fits. This is optional;
reducing the horizon, variables or spectral cutoff would change the experiment.

With the fixed batch and BPTT window, 2,000 total updates primarily increase
runtime and stored checkpoints, rather than activation memory per update.
The separate di32 qualification is still necessary: doubling the temporal
inner width enlarges its state and activations, while the two spherical
objective gradients introduce a workload not measured by the older power-loss
run. Check host-cache growth against the completed 2,000-update history as well
as the qualification peak. A successful first 100 updates does not rule out
later nonfinite values, resource problems or MGDA near-stalling.

There is positive native-width evidence: the historical di32 joint MSE job
13617440_3 completed its 500-update warm-up and 2,000-update branch. Its sampled
GPU peak was 24.50 GiB on an A10080 and branch median update time was 28.24 seconds.
The di16 joint continuation 13807466_0 used 23.14 GiB and 26.35 seconds per update.
The saved recurrent-state tape allocation grows from 4.45 to 8.91 GiB between
these widths. These single-objective runs support feasibility but do not measure
split-MGDA costs. See the historical campaign's
[training logs](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/logs).

## Evaluation at updates 500 and 2000

Evaluate each width's own step 0 and native 500 and 2,000 checkpoints against the common
frozen GraphCast baseline. Preserve the evaluation state initialization policy
from the existing matched protocol. Use one designated baseline result per
cohort and audit repeated baseline computations for drift rather than averaging
forecasts. Historical MSE, fixed-AMSE and old MGDA results are context only
unless their initialization, cohort and scoring provenance match.

The 32-start 2022 cohort reports exact original GC loss, day 10 loss and per-variable
RMSE over all 40 six-hour leads. The 8-start 2023 cohort reports those accuracy
metrics on the same cases as A/C, spherical power and coherence, and the legacy
zonal diagnostics. A joint accuracy/spectrum claim must use the same cohort.
Both cohorts have already been inspected and are development evidence.

Show absolute forecast, truth and GC spectra first, with ratios second. Include
surface temperature, both surface wind components, MSLP, precipitation, and
upper-air temperature, humidity, geopotential and both wind components at
declared levels. Save spherical diagnostics for all 83 primitive channels with
the opt-in full-field observer; retain the legacy 11-field zonal outputs. Include
degree 0 in saved outputs, label anomaly-only summaries, and retain the existing
distinction between fitted spectral error, projected grid error and unresolved
grid error. Pool squared errors before square-rooting RMSE. For an unqualified
improvement percentage, aggregate exact GC losses before taking their ratio.

Save per-initialization results for the 2023 cohort, and aggregate metrics with
matched origin indices for the 2022 cohort. Save training traces, native checkpoints, source
and input hashes, job accounting, and plotted tables under the campaign;
images go under the existing `plots/analyze_models/images/resolution_eval/`
convention. The report should show both width outcomes, including regressions,
and relate alpha and update norms to changes in A, C, RMSE and spectra.

The report compares each final native checkpoint with its initialization and
the common GC baseline, and uses step 500 to describe the trajectory. Any later
SWA over steps 1100–2000 is secondary. This pair estimates continuation performance
and the effect of width under this recipe. It cannot isolate the benefit of
adaptive coefficients from optimizer changes. A later fixed-alpha 0.5 control
must use the same two-gradient RMS metric to answer that question; the existing
single-loss optimizer squares the summed gradient and is not that control.

## Submission status

Slurm accepted the following initial stages on 2026-10-12. They were pending at
the first accounting check; GPU qualification remains outstanding.

| Stage | Job ID | Initial dependency |
| --- | --- | --- |
| di16 calibration | 15443267 | None |
| di32 calibration | 15443268 | None |
| Combine common scale | 15443269 | Both calibrations |
| di16 qualification | 15443270 | Combined calibration |
| di32 qualification | 15443271 | Combined calibration |
| Release 1,900 remaining updates per width | 15443272 | Both qualifications |

The release job will submit the two continuation jobs with measured resources,
the two six-task evaluation arrays, and the terminal report. Those downstream
job IDs do not exist until qualification passes.

Frozen source digest:
`c2341d1ec0772a87985b7b2e4d63c7eed958ea930b7d23d8e511b47db549fbaa`.

CPU evidence includes 99 existing AMSE/MGDA regression tests, 41 new integration
tests before the final restart-metadata addition, 19 final campaign tests,
and 29 width/spherical/evaluation tests. These suites overlap and should not be
summed. Tests include direct gradient comparisons, real runner state restoration
with a synthetic model, numerical/provenance rejection paths and report rendering.
Both evaluation cohorts passed actual coordinate coverage checks.

The campaign's `checks/cpu_validation.json`, `checks/submission_plan.json`,
`submissions/` and `logs/` retain validation, exact commands and subsequent
execution evidence. To inspect the current stage, run:

```bash
source scripts/graphcast_env.sh
python scripts/experiments/run_v24_amse_split_mgda.py status \
  --experiment artifacts/checkpoints/v24_Ilya/res1_amse_split_mgda_di16_di32_20261012
```
