# GC/M1 smoothing strength versus MSE and power spectra

Updated: 2026-11-09 (America/New_York). Owner: **Ilya**, reviewer-response item 1.
The user authorized implementation and submission. This evaluation is separate
from the full-GC horizon proposal, whose submission remains on hold.

## Question and fixed comparison

How much of the exact GraphCast loss improvement can output smoothing recover,
and what power-spectrum distortion accompanies each strength? Compare public
GC-small with principal M1 (`di16_fast_mamba/swa/swa_step00500-02000.pkl`) on the
same established 122 starts in 2023, 40 six-hour forecast leads, and all 83
channels on the native 1° grid. There is no new training.

Three prespecified series are evaluated independently for both models:

| Series | Settings | Meaning |
| --- | --- | --- |
| Global Gaussian | σ = 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8 degrees | One angular standard deviation for every channel and lead; degree gain `exp(-σ_rad² ℓ(ℓ+1)/2)`. |
| Fitted Gaussian blend | α = 0, 0.1, …, 1 | Blend toward each model's existing development-fitted Gaussian profile. |
| Fitted four-band blend | α = 0, 0.1, …, 1 | Blend toward each model's existing development-fitted four-band profile. |

Both blends use `fα = f + α(F(f) − f)`. A Gaussian-profile blend is a mixture
with the identity; it is not another literal Gaussian width. All three series
share exactly the same raw α/σ=0 reference: 30 distinct points per model, 60
in total. α=1 recovers the previous fitted-filter endpoints.

Reuse the independent GC and M1 profiles in the
[completed parent campaign](../../artifacts/evaluations/v24_smoothing_m1_20261107/manifest.json),
fitted on 32 development starts in 2022. No fitting or selection uses the 2023
curve. Show all prespecified strengths and describe the curves as exploratory;
prior 2023 analysis exposure is already disclosed. A visually favorable point
does not establish an independently selected optimum.

## Exact loss and spectra

Filters are output-only. The raw forecast remains the autoregressive input;
truth is unchanged. Retain the discrete area-weighted mean and the residual
outside the represented spherical harmonics, using the existing T179 operator.
Do not clip humidity, precipitation, powers or reconstructed losses.

One forecast pass scores all settings. For each of 11 nonidentity filter bases,
save the actual synthesized physical correction `d = F(f) − f` through its
moments. Native-grid loss is reconstructed as
`E(α) = E0 + 2α⟨(f−y)d⟩ + α²⟨d²⟩`, with exact pole-cell weights. Apply the
original GC difference-standard-deviation, variable and pressure weights after
reconstruction. Continuous spherical error power is a separate diagnostic and
cannot replace this grid loss.

Save physical forecast/truth/cross powers and correction power/cross moments
per start, lead, model, channel and degree. This permits further α refinement
without forecasting again. Spherical powers use analyzed synthesized
corrections; separate FP32 reanalysis of combined fields can differ slightly.
Finite drift and differences remain measured warnings, not release gates.
Nonfinite values or corrupt/mismatched source, checkpoint and data identities
remain errors.

The primary plot has exact GC rollout loss on the vertical axis and mean
absolute relative power error on the horizontal axis. Pool powers across
initializations before computing `|Pforecast/Ptruth − 1|`; average over all 83
channels, degrees 1–179 and all 40 leads. This spectral score uses equal channel
weights, unlike the GC loss. Also show day-10 power curves over degree for each
series and save leadwise numeric results, error power and correlation summaries.
Zero-truth bins are undefined and counted, without a denominator floor. Initial
curves are point estimates; no spectral confidence interval or training-seed
claim is implied.

## Execution and artifacts

Campaign: `artifacts/evaluations/v24_smoothing_sweep_m1_20261109/`.
The new immutable source snapshot inherits the parent's forecasting code and
adds only the sweep observer, driver, wrapper, report and an optional observer
argument in the reusable worker. The parent filters, checkpoints and campaign
are pinned by SHA256 and left unchanged. The fresh λ-model campaign is unchanged.

Eight balanced shards contain 15–16 starts each. Request **one 80 GB GPU,
4 CPUs, 20 GiB host RAM and 180 minutes per shard**, without an array throttle.
Completed parent array `17028677` used 43–46.5 minutes and up to 10.3 GiB host
RAM for 17–18 starts. Eleven corrections replace two, while repeated spectral
work and zonal diagnostics are reduced. The new allowance covers this extra
analysis and larger saved moments. Reuse these measurements without a separate
qualification run. Raw moments are approximately 41 GB before compression for
the whole campaign. The final dependent report requests 2 CPUs, 8 GiB and
45 minutes.

```bash
source scripts/graphcast_env.sh
python scripts/experiments/run_v24_smoothing_sweep.py prepare \
  --root artifacts/evaluations/v24_smoothing_sweep_m1_20261109 \
  --parent-root artifacts/evaluations/v24_smoothing_m1_20261107
python scripts/experiments/run_v24_smoothing_sweep.py submit \
  --root artifacts/evaluations/v24_smoothing_sweep_m1_20261109
```

Preparation is immutable and refuses an existing directory. `plan` prints the
commands for an already prepared campaign. `submit` records job IDs and reuses
them on repeat calls. A report runs only after successful completion of the
whole evaluation array; no automatic retries or training jobs are released.
Source/checkpoint integrity is verified at execution, and raw loss is compared
with the original evaluator as a numerical diagnostic.

Final figures and tables go under
`plots/analyze_models/{images,data}/resolution_eval/v24_smoothing_sweep_m1_20261109/`.
`report.json` in the campaign directory records completion and provenance.
Submitted November9: evaluation array **17213596**, dependent report
**17213597**. The first queue check showed all eight evaluation tasks pending
for priority and the report pending on its dependency. No new sweep results
are available yet. All15 focused tests passed, and both plot layouts rendered
successfully using synthetic data, which is not scientific evidence. See the
[validation record](../../artifacts/evaluations/v24_smoothing_sweep_m1_20261109/checks/pre_submission_validation.json).

Frozen source digest:
`22d780d97df6119954c3ca1f4349cf107ecdca26892ae271bd71f9f63780f5e9`.

```bash
squeue -j 17213596,17213597
```
