# Full GraphCast fine-tuning: AR horizons, checkpoint spectra and SWA

Updated: **2026-11-09, America/New_York**.

**Owner:** Lianghong, coordinating spectral and smoothing comparisons with Ilya.

**Status:** the K8/K12/K16 extension is prepared but **not submitted**. It is on
hold while the comparison with principal M1 is reviewed. The existing K20 run
is already submitted and remains separate from this hold. M1 integration,
the 2023 comparison and expanded spectral reporting are required revisions,
not completed features of the prepared campaign.

## Purpose and existing evidence

Test whether further full-model fine-tuning of public GraphCast on our small
training dataset improves forecasts, and how the autoregressive training
horizon, training duration and checkpoint averaging affect accuracy and spectra.
This addresses R1's request for direct multi-step fine-tuning; spectral and
smoothing comparisons also contribute to R1/R3's forecast-structure concern.
Equal training budgets with M1 are not a requirement for this experiment.

The existing K20 run at peak learning rate `3e-7` reduced exact GraphCast rollout
loss by **4.6434%** at update 500 on **32 development starts in 2022**; the day-10
reduction was **5.3090%**. Higher-rate 500-update calibrations worsened loss.
Consequently, a general claim that further fine-tuning cannot help is unsupported.
At the November 9 review, K20 had reached 1,250 updates; its final checkpoint
and 2023 evaluation were pending. See the [saved K20 development result](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/calibration_evaluation/gcft_s22_l000_lr3em7_fresh_20261107/evaluation.json).

## Models and checkpoints

All linked artifacts are on Della under
`/scratch/gpfs/DABANIN/iv9432/Weather_global`.

| Model | Checkpoint or run | Role |
| --- | --- | --- |
| Public GC | [GraphCast-small public NPZ](<../../data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz>) | Initializer for each full-GC run and unmodified forecast baseline; 1° grid, 13 pressure levels. |
| Principal M1 | [di16_fast_mamba / swa_step00500-02000.pkl](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/swa/swa_step00500-02000.pkl) | Historical K20 GC–Mamba reference; retain this exact checkpoint. |
| Existing full-GC K20 | [gcft_s22_l000_lr3em7_fresh_20261107](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/runs/gcft_s22_l000_lr3em7_fresh_20261107) | Reuse its checkpoints and existing continuation. |
| New full-GC K8/K12/K16 | [Prepared campaign manifest](../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/manifest.json) | Three independent starts from public GC; no training submitted. |

The expected K20 final raw checkpoint, once training completes, is:

```text
artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/
  runs/gcft_s22_l000_lr3em7_fresh_20261107/checkpoints/checkpoint_step00002500.pkl
```

## Training design

Use the **standard GraphCast loss (λ=0)**, 2015–2021 training data, seed 22,
batch four, 96-step segments and the same frozen training-window order. Update
all GC parameters with full weather-feedback gradients (`closed_loop`). Use
AdamW with a 100-update warm-up, peak LR `3e-7`, cosine decay to `3e-8`, and
**2,500 optimizer updates**. The positive-λ sweep is a separate experiment.

| Horizon | Total training window | Teacher-forced prefix | Feedback-conditioned tail | Action |
| --- | ---: | ---: | ---: | --- |
| K8 | 24 steps | 16 steps | 8 steps | Fresh run from public GC |
| K12 | 24 steps | 12 steps | 12 steps | Fresh run from public GC |
| K16 | 24 steps | 8 steps | 16 steps | Fresh run from public GC |
| K20 | 24 steps | 4 steps | 20 steps | Reuse existing run |

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
2. **Checkpoint selection:** define and freeze the selection rule using 2022
   only. Proposed 2023 coverage is each horizon's final raw checkpoint and SWA,
   plus any distinct intermediate checkpoint selected on development data.
   Report the complete development curves, including unfavorable results.
3. **Reviewer evaluation:** compare the chosen GC models with principal M1 and
   public GC on the same **122 starts in 2023**, with 40 leads. Reuse completed
   reference statistics and the existing K20 final evaluation when available.
   Report rollout/day-10 exact loss and paired block-bootstrap intervals for
   model differences. Disclose prior 2023 exposure; these intervals quantify
   forecast-sample uncertainty for fixed checkpoints, not training variability.
4. **Spectra:** compare forecast and error power, power ratios and spectral
   correlation alongside exact GC loss. Preserve all channels/leads; use T2m,
   Z500 and U850 at days 5 and 10 for the compact figure set. Include M1 and
   public GC in the comparisons. The draft saves error/cross powers but
   currently plots forecast power only.
5. **Smoothing:** coordinate with Ilya to fit smoothing independently on 2022
   for each new finalist, reusing the existing GC/M1 filters. Apply the fixed
   filters on 2023. Power amplitude alone cannot establish whether improved
   MSE reflects better forecast structure.

Use exact original GraphCast loss for unqualified improvement percentages:
`100 * (1 - pooled_model_loss / pooled_public_gc_loss)`. Pool losses over the
requested dates/leads before taking the ratio and use a common public-GC
denominator. Retain finite repeated-baseline differences and their reference
bounds as numerical warnings; report separately whether decreases exceed the
measured noise. Let valid runs reach their scheduled endpoints and evaluation.

The final handoff should contain a loss/checkpoint table with paired intervals,
training curves, spectral comparisons, smoothing comparisons, checkpoint/config
identifiers and measured GPU-hours/memory. Conclusions should describe the
benefit or lack of benefit under the tested settings.

## Implementation status and supporting records

The prepared training, checkpoint evaluation and SWA code passed the recorded
technical checks. **M1 report integration, 2023 evaluation/selection logic,
expanded spectral plots and filtering for new finalists remain outstanding.**
No new jobs were submitted for this README; the submission hold remains in place.

- [Detailed design, comparison audit and resource estimates](v24_full_gc_ar_horizons_2026-11-09.md).
- [Prepared submission plan](../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/submission_plan.json).
- [Campaign driver](../../scripts/experiments/run_v24_full_gc_ar.py) and [Slurm wrapper](../../scripts/experiments/v24_full_gc_ar.slurm).
- [Existing M1/GC evaluation and smoothing references](../../artifacts/evaluations/v24_smoothing_m1_20261107/manifest.json).
- [Lianghong's broader reviewer-response handoff](README_LH.md).
