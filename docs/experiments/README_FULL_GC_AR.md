# Full GraphCast fine-tuning: AR horizons, checkpoint spectra and SWA

Updated: **2026-11-12, America/New_York**.

**Latest schedule correction:** use **constant `3e-7` from the first update,
without warmup or decay**, for all four fresh full-GC horizons. The revised
portable handoff has suffix `constant_lr3em7`; the previous package is archival.
The Della five-lambda full-GC sweep is also being replaced with five fresh
constant-rate arms; see the [current protocol](v24_full_gc_constant_lr3em7_2026-11-12.md).
Earlier job-status paragraphs below describe historical activations and are
superseded by that protocol. GC–Mamba's M1 recipe remains unchanged.

**Runnable handoff for Lianghong:** [portable code, setup and exact commands](README_LIANGHONG_PORTABLE_RUNS.md)
now cover both the M1-matched GC–Mamba lambda sweep and this four-fresh-horizon
full-GC study, including raw/SWA MSE and spectra, 2022 selection, 2023 evaluation
and development-fitted smoothing. Use those portable launchers on her cluster;
they require local data paths and scheduler settings. The old Della horizon
launcher below still reuses K20 and must not be used for her fresh study.

**Owner:** Lianghong owns the four new training runs, their spectral analysis
and the smoothing pipeline specified below. Reuse the existing public-GC/M1
reference artifacts and the same filter definitions.

**Assignment:** Lianghong should train **four fresh full-GC runs: K8, K12, K16
and K20**, each starting from the public pretrained GraphCast checkpoint.
Here, starting from scratch means a new fine-tuning run with a fresh optimizer;
it does not mean random initialization of GraphCast. Do not resume or reuse
Ilya's K20 fine-tuning checkpoints as one of these four runs.
Compare the **power-spectrum/MSE tradeoff** across horizons, training checkpoints
and SWAs, with public GC and principal M1 as references.

**Status:** portable implementation prepared, **not submitted on Lianghong's
cluster**. The earlier Della campaign reused Ilya's K20 run and remains held.
The portable handoff creates all four horizons fresh and includes M1 references,
2023 comparison and spectral reporting. No new Della horizon jobs were submitted.

**November 12 GC–Mamba-only restart submitted:** after reviewing the proposal,
the user approved replacing only GC–Mamba and explicitly keeping DeepMind/full-GC
unchanged. The [review plan and activation record](v24_m1_matched_lambda_restart_2026-11-12.md)
specifies initial500 plus main2000, stopped weather gradients, native BF16 loss,
historical deterministic data order and two allocations of1250 updates.
Old GC–Mamba `17062358/17062359/17062360` and the obsolete shared final report chain
were cancelled. Five fresh lambda arms are submitted as `17406861_0–4` for the
first 1,250 updates, followed by `17406862_0–4` to reach 2,500 total updates.
History/controller jobs are `17406863/17406864`; the replacement final raw/SWA
MSE and spectrum chain is `17406897 → 17406898 → 17406899`. All were pending at
verification. Full-GC training, history and controller `17049163/17049164/17049165`
remain unchanged, including their existing training recipe. Completed artifacts
and the saved-checkpoint diagnostic below are retained. Implementation and
reset/resume checks passed; new training results are pending.

**November 12 matched-budget audit:** the earlier full-GC raw1250 versus M1
comparison matched forecast conditions, but M1 had2500 total training updates.
At the user's request, [a saved-checkpoint comparison](v24_matched_budget_comparison_2026-11-12.md)
now compares M1, November GC–Mamba lambda0 and full-GC lambda0 at1000 and1250
total updates, with exact MSE and spherical spectra. M1 main steps500/750
include500 preceding updates; November1500 checkpoints are not yet available.
Four missing forecasts are submitted as17404694_2–5, with report17404732;
existing1250 and final M1 raw/SWA forecasts are reused. Results are pending.
The audit verified matching evaluation conditions and correct full-GC weight
loading, but training feedback gradients, optimizer schedules, loss arithmetic
and sample ordering differ. Existing matched32-start results already show M1
ahead at500 total updates (MSE9.8199 versus full-GC10.3373); these older
evaluations used their respective runtimes. The new comparison fixes one
frozen runtime. This does not release the four-fresh-horizon training hold.

**November11 pilot result:** both [non-SWA M1/lambda pilots](v24_raw_checkpoint_pilot_2026-11-10.md)
completed on four2022 starts ×40 leads. M1 raw step2000 has exact rollout
MSE9.4240 and spherical power error75.94%; SWA has9.1717 and71.89%, so removing
SWA worsens both aggregate metrics here. Public GC is11.1331/61.57%.
Fresh raw1250 models remain worse than M1 on MSE. Historical Octoberλ0.25
improves both metrics versus raw M1, while M1 SWA is slightly better on both.
[Fresh results/figures](../../plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110/report.md)
and [historical results/figures](../../plots/analyze_models/data/resolution_eval/v24_legacy_raw_checkpoint_pilot_20261110/report.md)
are complete. This small pilot does not release the new horizon training hold.

**November10 protocol:** the separate [non-SWA M1/lambda pilot](v24_raw_checkpoint_pilot_2026-11-10.md)
uses M1's final raw main-stage step2000 (2500 total updates) and the existing
ten fresh λ arms' common raw step1250. Four2022 starts ×40 leads provide exact
MSE curves, power spectra/ratios and an MSE–power tradeoff, with public GC and
M1 SWA references. This saved-checkpoint evaluation is authorized separately;
the new horizon training campaign's submission hold remains in place.
The user's follow-up includes **both** fresh and historical λ experiments:
the extension adds raw step2000 for the old λ0.1/λ1 pairs and raw step500 for
the October λ0/0.25/0.5 continuations and their controls, reusing M1 results.
The [cached non-SWA MSE/power figures are already available](../../plots/analyze_models/data/resolution_eval/amse_lambda_raw_cached_20261110/report.md)
on eight2023 starts; those zonal spectra remain separate from the new spherical
pilot. Training histories are labeled separately throughout.

## Purpose and existing evidence

Test whether further full-model fine-tuning of public GraphCast on our small
training dataset improves forecasts, and how the autoregressive training
horizon, training duration and checkpoint averaging affect accuracy and spectra.
This addresses R1's request for direct multi-step fine-tuning; spectral and
smoothing comparisons also contribute to R1/R3's forecast-structure concern.
Equal training budgets with M1 are not a requirement for this experiment.

**Preliminary result from Ilya:** after just **500 fine-tuning updates**, the K20
run at peak learning rate `3e-7` **with cosine decay and warmup** reduced exact GraphCast rollout loss by
**4.6434%** on **32 development starts in 2022**; the day-10 reduction was
**5.3090%**. This motivates the new four-run study. Ilya's run is background
evidence only, not a constant-rate result or Lianghong's K20 arm.
Consequently, a general claim that further fine-tuning cannot help is unsupported.
These are intermediate development results. See the [saved K20 development result](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/calibration_evaluation/gcft_s22_l000_lr3em7_fresh_20261107/evaluation.json).

## Models and checkpoints

All linked artifacts are on Della under
`/scratch/gpfs/DABANIN/iv9432/Weather_global`.

| Model | Checkpoint or run | Role |
| --- | --- | --- |
| Public GC | [GraphCast-small public NPZ](<../../data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz>) | Initializer for each full-GC run and unmodified forecast baseline; 1° grid, 13 pressure levels. |
| Principal M1 | [di16_fast_mamba / swa_step00500-02000.pkl](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/swa/swa_step00500-02000.pkl) | Historical K20 GC–Mamba reference; retain this exact checkpoint. |
| Ilya's preliminary K20 | [gcft_s22_l000_lr3em7_fresh_20261107](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/runs/gcft_s22_l000_lr3em7_fresh_20261107) | Evidence of improvement at 500 updates; not an initializer or one of Lianghong's four runs. |
| Lianghong's K8/K12/K16/K20 | Four new run directories, to be prepared | All four start independently from public GC with fresh optimizer state. |

Save each new run's checkpoints, optimizer state, configuration and evaluations
in its own directory. Each horizon should produce its own raw
`checkpoint_step00002500.pkl` and its own SWA. Keep Ilya's existing run intact.

## Training design

Use the **standard GraphCast loss (λ=0)**, 2015–2021 training data, seed 22,
batch four, 96-step segments and the same frozen training-window order. Update
all GC parameters with full weather-feedback gradients (`closed_loop`). Use
AdamW with constant LR `3e-7`, no warmup or decay, and
**2,500 optimizer updates**. The positive-λ sweep is a separate experiment.

| Horizon | Total training window | Teacher-forced prefix | Feedback-conditioned tail | Action |
| --- | ---: | ---: | ---: | --- |
| K8 | 24 steps | 16 steps | 8 steps | Fresh run from public GC |
| K12 | 24 steps | 12 steps | 12 steps | Fresh run from public GC |
| K16 | 24 steps | 8 steps | 16 steps | Fresh run from public GC |
| K20 | 24 steps | 4 steps | 20 steps | Fresh run from public GC |

Each step is six hours. Changing K within a fixed 24-step window also changes
teacher forcing. Interpret this as a comparison of training recipes; **K20 is
the closest horizon match to principal M1**.

Evaluate raw checkpoints at **500, 1,000, 1,500, 2,000 and 2,500 updates**, with
a power-spectrum comparison for every checkpoint. Also evaluate an equal-weight
**SWA of total updates 1,000, 1,500, 2,000 and 2,500** for each horizon. Report
raw and averaged results separately. These averaging positions correspond to
M1's main-stage updates 500, 1,000, 1,500 and 2,000 after its 500-update warm-up.
The four horizons therefore provide 20 raw checkpoints and four SWAs.

## Fair comparison with M1

Use the same public backbone initializer, grid, training years, evaluation dates,
truth, normalization, forecast leads and exact GC loss. Report the actual
training cost and checkpoint-selection rule for each model.

Historical M1 and full GC have meaningful training differences:

| Setting | Principal M1 | Full-GC horizon campaign |
| --- | --- | --- |
| Trainable parameters | 10,348,879 correction parameters; frozen GC | 35,979,347 backbone parameters |
| Weather-feedback gradients | Stopped (`closed_loop_sg`) | Full (`closed_loop`) |
| Optimizer schedule | 500-update warm-up, then 2,000-update main stage with optimizer reset | One 2,500-update schedule with 100-update warm-up |
| Training-window order | Historical sampler | Frozen shuffled order shared across horizons |
| Checkpoint | Historical SWA | Raw checkpoints and proposed SWA |

This is a comparison of forecasting methods with documented training recipes.
It does not isolate memory or architecture; the matched reset-trained control
addresses the memory claim. One learning rate and one seed cannot establish
optimal GC tuning or robustness across training realizations.

## Evaluation and deliverables

1. **Development:** compare all 24 GC checkpoints/SWAs, public GC and principal
   M1 on the same **32 starts in 2022**, forecasting **40 six-hour leads**. Reuse
   existing M1 exact-loss records. Its saved development calibration statistics
   lack raw spectra, so add a spectral evaluation for this cohort.
2. **Checkpoint selection:** for each horizon, include the update-2,500 raw
   checkpoint and SWA. Also include the raw checkpoint with the lowest pooled
   exact GC loss on 2022 among updates 500/1,000/1,500/2,000/2,500, if distinct;
   an exact tie chooses the earlier update. This gives eight to twelve models
   for smoothing and 2023 evaluation. Use this rule before inspecting their
   2023 results. Report all 24 raw development checkpoints/SWAs, including
   unfavorable results.
3. **Reviewer evaluation:** compare the chosen GC models with principal M1 and
   public GC on the same **122 starts in 2023**, with 40 leads. Reuse completed
   M1/public-GC reference statistics, and evaluate Lianghong's new checkpoints
   for all four horizons, including the new K20.
   Report rollout/day-10 exact loss and paired block-bootstrap intervals for
   model differences. Disclose prior 2023 exposure; these intervals quantify
   forecast-sample uncertainty for fixed checkpoints, not training variability.
4. **Spectra:** compare forecast and error power, power ratios and spectral
   correlation alongside exact GC loss. Preserve all channels/leads; use T2m,
   Z500 and U850 at days 5 and 10 for the compact figure set. Include M1 and
   public GC in the comparisons. The draft saves error/cross powers but
   currently plots forecast power only.
5. **Power-spectrum/MSE tradeoff:** for all four horizons, plot each 500-update
   checkpoint and SWA on a common tradeoff plot: **forecast-power mismatch
   relative to ERA5 on the horizontal axis and exact GC weighted MSE on the
   vertical axis** (lower is better on both axes). Pool powers over dates first,
   then average `100 * abs(Pforecast / Ptruth - 1)` equally over 83 channels,
   degrees 1–179 and the reported leads. Exclude zero-truth-power bins with
   recorded counts and use the same mask for all models. This matches the
   existing GC/M1 report. Mark public GC and principal M1 as references;
   distinguish horizons and label checkpoint updates/SWA. Use identical dates
   and lead ranges for both metrics. Show the full comparison on 2022 and the
   selected models on 2023, with rollout and day-10 summaries. Identify where
   lower MSE accompanies improved, unchanged or worsened power matching, and
   whether SWA changes that tradeoff. Use the degree-resolved spectra, error
   power and correlation to interpret which spatial scales drive the changes.
6. **Smoothing:** apply the fixed pipeline below to every selected checkpoint,
   with its own 2022-fitted filters. Compare the resulting curves with public GC
   and M1. Power amplitude alone cannot establish whether improved MSE reflects
   better forecast structure.

Use exact original GraphCast loss for unqualified improvement percentages:
`100 * (1 - pooled_model_loss / pooled_public_gc_loss)`. Pool losses over the
requested dates/leads before taking the ratio and use a common public-GC
denominator. Retain finite repeated-baseline differences and their reference
bounds as numerical warnings; report separately whether decreases exceed the
measured noise. Let valid runs reach their scheduled endpoints and evaluation.

The final handoff should contain a loss/checkpoint table with paired intervals,
training curves, **power-spectrum/MSE tradeoff plots and a short interpretation**,
degree-resolved spectral comparisons, smoothing comparisons, checkpoint/config
identifiers and measured GPU-hours/memory. Conclusions should describe the
benefit or lack of benefit under the tested settings.

## Fixed smoothing pipeline

**Decide the protocol now; the new GC strength-curve results are not a
prerequisite.** The [fitted-filter GC/M1 comparison is already complete](../../plots/analyze_models/data/resolution_eval/v24_smoothing_m1_spectra_20261109/report.md)
on 122 starts in 2023. GC's Gaussian and four-band filters reduced rollout loss
by 8.87% and 9.39%, while mean relative power error increased from 59.47% raw
to 79.17% and 75.70%, respectively. The [broader strength sweep](v24_smoothing_strength_sweep_2026-11-09.md)
has been submitted, with no saved evaluation results at this review. Reuse its
reference curves once available; do not choose the new models' protocol from
favorable points on those 2023 curves.

1. **Generate development forecasts.** For each of the eight to twelve selected
   checkpoints above, use the same 32 starts in 2022 and 40 leads. Collect
   exact-grid smoothing calibration statistics from the physical forecast fields.
   Existing raw spectra alone cannot recover exact filtered grid MSE.
2. **Fit two profiles independently for each checkpoint.** Use the existing
   fitter and fixed settings in the table below. Fit per variable/pressure level
   and within lead groups **6–24, 30–72, 78–168 and 174–240 hours**. Tie U/V
   parameters at each level and U10/V10. Reuse the [frozen public-GC/M1 profiles](../../artifacts/evaluations/v24_smoothing_m1_20261107/filters.json)
   for those reference models; fit new profiles for every new checkpoint/SWA.
3. **Freeze the filters.** Save checkpoint and data-manifest hashes, fitted
   parameters, grouping, objective and numerical diagnostics before 2023 scoring.
   Use both filter families in the report; no family or strength is selected on
   2023.
4. **Evaluate the fixed strength grid on 2023.** For each selected model, report
   the three series below on the same 122 starts. Generate each forecast once
   and stream exact-grid error/correction moments and spectral cross-moments
   for all strengths. Reconstruct the curves offline; repeated forecasts for
   individual smoothing strengths are unnecessary.
5. **Compare curves and endpoints.** Include raw, fully fitted Gaussian and
   fully fitted band endpoints, plus every prescribed intermediate strength.
   Show whether the new GC models improve the MSE/power tradeoff relative to
   both public GC and M1, and whether differences remain after each model has
   its own fitted smoother. Report the curves as descriptive comparisons; do
   not present a visually selected 2023 point as an independently chosen optimum.

| Series | Fixed settings |
| --- | --- |
| Global Gaussian | Angular standard deviation σ = **0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8°**, shared across all channels/leads. No fitted parameters. |
| Fitted Gaussian blend | Fit each group's σ from **0, 0.25, 0.5, 1, 2, 4, 8°** by minimum exact 2022 GC loss; ties choose smaller σ. Evaluate `f + α(F(f) − f)` for **α = 0, 0.1, …, 1**. |
| Fitted four-band blend | Fit gains in **[0, 1]** at degrees **1, 10, 40, 179**, interpolated in log degree, using the existing regularized exact-grid objective with **η = 0.01**. Evaluate the same **α = 0, 0.1, …, 1** blends. |

These series share a single raw point, giving **30 distinct points per model**.
Use the existing T179 spherical operator, preserving the discrete area-weighted
mean and unrepresented residual. Filtering is **output-only**: feed the raw
forecast into the next step and leave ERA5 truth unchanged. Retain the existing
no-clipping convention and report any negative humidity/precipitation values as
diagnostics. Compute headline MSE on the native grid with exact GC weights;
spherical coefficient error is a separate diagnostic.

The reusable components are [the filter fitter/operator](../../src/analysis/spherical_smoothing.py),
[calibration/evaluation observer](../../src/analysis/smoothing_observer.py) and
[strength-curve observer](../../src/analysis/smoothing_sweep_observer.py).
The original strength-sweep launcher is restricted to the completed GC/M1
campaign. The new portable smoothing adapter applies the same computation to
each finalist with local inputs and separately fitted filters; its commands are
included in the [portable handoff](README_LIANGHONG_PORTABLE_RUNS.md).

## Implementation status and supporting records

The earlier Della implementation remains a three-new-run/reused-K20 historical
design, with its manifest and source preserved. The separate portable drivers
now implement four fresh full-GC runs, M1 raw/SWA references, development
checkpoint selection, annual evaluation, expanded spectral plots and fitted
filter/strength comparisons. Training, evaluation and smoothing reuse the
existing computation; the new adapters supply local paths and orchestration.
The [portable README](README_LIANGHONG_PORTABLE_RUNS.md) is the executable handoff.
No new jobs were submitted while preparing it; no cross-cluster result is claimed.

- [Earlier design, comparison audit and resource estimates](v24_full_gc_ar_horizons_2026-11-09.md): its K20 reuse and three-new-run training estimate are superseded by this four-fresh-run assignment.
- [Earlier campaign manifest](../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/manifest.json) and [submission plan](../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/submission_plan.json): these still encode K20 reuse and do not implement the revised assignment.
- [Campaign driver](../../scripts/experiments/run_v24_full_gc_ar.py) and [Slurm wrapper](../../scripts/experiments/v24_full_gc_ar.slurm).
- [Existing M1/GC evaluation and smoothing references](../../artifacts/evaluations/v24_smoothing_m1_20261107/manifest.json).
- [Lianghong's broader reviewer-response handoff](README_LH.md).
