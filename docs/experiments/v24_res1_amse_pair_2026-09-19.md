# Res1 residual Mamba: matched FP32 MSE and full-forecast AMSE

Status as of 2026-09-21: both arms completed 2,000 training updates on September 20.
The first evaluations failed because their metadata check assumed Zarr v2 for
the existing Zarr v3 dataset. The fix passed local validation; replacement
evaluation array **14246320_0–1** and report **14246359** are submitted.
See the [recovery record](../reports/v24_evaluation_recovery_2026-09-21.md).
Completed evaluation results remain pending.

Pre-submit validation: **49 focused tests passed**, alongside **89 legacy
endpoint/config tests** and a **28-pass integration/DP/cached regression suite**
(6 four-device checks skipped; one-replica pmap tested). Source/config/input
checks and shell syntax passed. Frozen source digest:
`49e2b09d8ec0278bdeb97b8aa5a7070733f2c3c7471f3ca1c96d4d371fa63ef7`.

Experiment: `artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919/`.

## Question and matched setup

Can full-forecast adjusted MSE reduce the extra spectral smoothing of the
trained residual branch while retaining useful deterministic forecast skill?
This applies equation 6 of [Subich et al.](https://arxiv.org/html/2501.19374v2)
to the physical corrected forecast `f = stop_gradient(GC) + residual` and truth.
Only residual spatial and Mamba parameters are optimized. GC remains frozen.

Both arms load the identical native 500-update warm-up checkpoint from
`res1_batch4_temporal_lr_20260908/seeds/di16_fast_mamba_warmup/`.
The existing di16-fast SWA scored 16.343447% exact GraphCast rollout-loss reduction
on the historical matched32 development set. It is a reference, not an untouched
test winner or proof of superiority over the nearby di32 configuration.

Settings inherited from that run:

- Residual width512/MP2, Mamba1 di16/BCG1/state16/conv4/two layers;
  original GraphCast Small width512/MP16/mesh5 baseline at 1 degree.
- Segment96, four 24-step BPTT chunks, K20 autoregressive tail, all24-step loss.
  State carry and segment resets unchanged. `closed_loop_sg` stops gradients
  through physical feedback; temporal credit flows through recurrent state.
- Batch4, seed22, 2000 updates; AdamW(0.9,0.98), weight decay1e-4, clip1;
  spatial LR1e-4, Mamba multiplier3; warmup50, cosine endpoint1e-5.
- Checkpoints every250; primary SWA averages branch updates500/1000/1500/2000.
- Same prepared2015–2021 training data, data order and anchor manifests.
  Native `--init-from` resets optimizer/RNG/cursor and uses the established
  batch4 initialization/state policy identically in both arms.

The MSE control computes original GraphCast weighted grid MSE in FP32 on full
fields. Historical training computed the loss inside its BF16 predictor wrapper;
therefore a contemporary FP32 control is required. Both arms keep BF16 neural
execution and FP32 physical/state boundaries.

## Spectral objective and numerical scope

At each total spherical wavenumber, AMSE is

`(sqrt(Pf) - sqrt(Py))^2 + 2 max(Pf, Py) (1 - coherence(f,y))`.

Compute power/coherence for each batch member, variable, pressure level and
supervised step before aggregation. Preserve difference-standard-deviation
scaling, original variable weights and pressure-proportional level weights.
Do not average the batch powers before applying the training loss.

The self-contained JAX spherical transform uses normalized associated Legendre
functions with longitude FFT and area-weighted QR latitude analysis. The regular
181x360 grid uses triangular T179. The global mean is retained; longitude Nyquist
and unresolved components are excluded. Coefficient power is the continuous-area
norm of the fitted spherical field; it is not exactly the discrete GraphCast
grid norm for arbitrary unresolved data. This numerical distinction from the
grid-MSE control is explicit. A shared power floor1e-12 stabilizes coherence at
zero without an auxiliary loss-mixing weight. The spectral transform/scoring
uses FP32/complex64, with geometry prepared in FP64.

## Gates, resources and execution

CPU tests cover analytic spherical harmonics, the AMSE amplitude/coherence
identity, finite perfect/zero-field gradients, weighting, full-field supervision,
recurrent reverse gradients, config compatibility and workflow gates.

An unthrottled two-task GPU preflight runs64 independent measurement updates
per arm from the common warm-up, checks finite losses/gradients and changed
parameters, validates checkpoint identity, and reloads it for a short cold
forecast. Preflight uses the same model/batch/sequence and warmup50 with its
cosine shortened to64; it never supplies production starting weights.

Initial resources per preflight: one A10080,8 CPUs,320GiB CPU RAM,2h.
Accounting: reference13740084_6 used22:40 for warmup+training; recovery
13807466_0–2 needed~16h for2000 updates, with peak CPU RSS up to228.86GiB.
The dependent CPU release job submits production only after both preflights
succeed. It derives each arm's walltime from measured p90 update duration,
startup/checkpoint allowance and35% margin; memory uses the larger of preflight
peak and historical long-run peak with40% margin, rounded to16GiB.
Jobs exceeding the six-day queue limit are rejected for explicit staging.

Each arm receives dependent GPU evaluation, followed by a CPU report after both
evaluations succeed. Source/config/checkpoint hashes bind all gates. Queued jobs
execute an immutable source snapshot. No existing run is overwritten.

## Evaluation

Evaluate native step2000 and the prespecified SWA for both arms:

- Exact original GraphCast loss: same32 historical2022 cold starts, zero state,
  full corrected feedback,40 six-hour leads.
- Saved2023 spectral pilot: same8 starts and40 leads, with power-spectrum
  relative error and spectral coherence alongside exact loss. These cases have
  already been inspected and are development diagnostics, not an untouched test.
- Retain each evaluation's paired original-GC baseline and report repeatability
  drift explicitly; do not silently assert a shared identical baseline.

The comparison must expose power recovery, coherence changes and exact-GC-loss
changes separately. Sharper spectra alone do not establish a better forecast.

Entry points: `scripts/experiments/run_v24_amse_pair.py`,
`scripts/experiments/v24_amse_pair.slurm`, and
`scripts/analyze_models/evaluate_v24_amse_pair.py`.
