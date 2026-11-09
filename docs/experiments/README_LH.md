# Lianghong (LH): reviewer-response experiment handoff

Updated: **2026-11-09, America/New_York** (GC fine-tuning campaign review).

Lianghong owns **original GraphCast fine-tuning, extreme-event analysis, and
training-seed replication**. Ilya owns the MSE/spectral tradeoff and smoothing
study. This README records the assignments and starting points; no new jobs were
submitted while preparing it.

## Ownership and priorities

Importance is rated out of five. Difficulty describes implementation and analysis;
seed repeats are straightforward but require additional GPU time.

| Priority | Reviewer concern | Owner | Importance | Difficulty | Deliverable |
| --- | --- | --- | --- | --- | --- |
| 1 | Lower MSE may reflect smoothing; power spectra and λ tradeoffs | **Ilya** | 4/5 | Moderate | MSE, power/error spectra, correlation and development-fitted smoothing comparisons for M1, public GC and both methods at λ = 0, 0.25, 0.5, 0.75 and 1. |
| 2 | Further fine-tuning of the already fine-tuned GC baseline | **Lianghong** | 4/5 | Relatively easy | Assess standard-loss full-GC fine-tuning at K8/K12/K16/K20 through 2,500 updates, with checkpoint spectra and optional SWA. Compare with public GC and principal M1 on common evaluation dates; equal training budgets are not required. New submissions are on hold; see the [dedicated experiment README](README_FULL_GC_AR.md). |
| 3 | Extreme-event performance | **Lianghong** | 4/5 | Easy | Review the completed occurrence/intensity evaluation and prepare tables, figures and a short interpretation. |
| 4 | Multiple training seeds | **Lianghong** | 4/5 | Easy; additional compute | Proposed first scope: principal M1 recipe and matched reset-trained control at seeds 23 and 24, alongside existing seed 22. |
| 6 | Predicted-field maps and gradient diagnostics | **Unassigned** | 3/5 | Easy–moderate | Nobody is working on this so far. Whoever finishes their assigned work first could take it on, using the same forecast models without another training campaign. |
| 7 | R1: Equal-budget LoRA / parameter-efficient fine-tuning comparison | **Unassigned** | 2/5 | Moderate | Compare LoRA with GC–Mamba under matched data, rollout protocol and optimizer-update budgets; report actual compute cost separately. Still open and lower priority than the main comparison. |
| 8 | R2: Matched GRU/LSTM or fixed-history comparison | **Unassigned** | 2/5 | Moderate–hard | Useful for testing architecture choice; lower priority while we make no claim that Mamba is uniquely necessary or superior. |

## Workspace and checkpoint references

The artifacts below are on **Della** under
`/scratch/gpfs/DABANIN/iv9432/Weather_global`. Paths in code blocks are relative to
that workspace unless stated otherwise. Sharing this README does not copy the
checkpoints or datasets; Lianghong needs access to the shared filesystem or a copy.

Run Python/analysis commands from the repository root with the existing environment:

```bash
cd /scratch/gpfs/DABANIN/iv9432/Weather_global
source scripts/graphcast_env.sh
```

| Reference | Exact checkpoint | Use |
| --- | --- | --- |
| **Public GC** | [GraphCast-small public NPZ](<../../data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz>) | Unmodified baseline and pretrained initializer: 1° grid, 13 pressure levels. |
| **M1** | [di16_fast_mamba / swa_step00500-02000.pkl](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/swa/swa_step00500-02000.pkl) | Principal historical K20 GC–Mamba model; existing seed-22 result. |
| **Matched reset** | [res1_selected_matched_reset_20261015 / swa_step00500-02000.pkl](../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/runs/branch/swa/swa_step00500-02000.pkl) | Same configured architecture/training protocol as M1, with memory reset during training and forecasting. |
| **Relative-power MGDA** | [res1_mgda_gc_power_2k_recovery_20261007 / swa_step01100-02000.pkl](../../artifacts/checkpoints/v24_Ilya/res1_mgda_gc_power_2k_recovery_20261007/runs/mgda/swa/swa_step01100-02000.pkl) | Secondary comparison already included in the extremes campaign; distinct from the new scalar-λ models. |

## 1. Spectral tradeoff, λ sweep and smoothing — Ilya

Ilya owns the five-value λ sweep for both fresh GC–Mamba and full GC, including
power/error spectra, correlation and development-fitted smoothing. Follow the
[spectral campaign plan](v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md)
and active mappings for [full GC](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/selected_arms_full_gc.json)
and [GC–Mamba](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/selected_arms_gc_mamba.json).
Include principal M1 and public GC as references.

The standard-loss GC result from item 2 also supplies the GC λ=0 point here.
The current GC λ=0 run uses peak LR `3e-7`, while the four positive-λ GC runs use
`1e-5`; disclose this difference when interpreting the sweep. The positive-λ
runs and smoothing analysis belong to this workstream.

**November9 reference result:** the [completed GC/M1 power-spectrum report](../../plots/analyze_models/data/resolution_eval/v24_smoothing_m1_spectra_20261109/report.md)
now includes raw/Gaussian/band-filtered spectra, figures and numeric tables.
Raw M1 has better loss and macro power matching than filtered GC, while raw GC
has the lowest overall power error. This supplies the historical references for
item1; the new λ-model comparison remains pending.

## 2. Further GraphCast fine-tuning with the standard loss — Lianghong

**Question:** The public GraphCast baseline has already undergone pretraining
and multi-step fine-tuning. Does further full-model fine-tuning on our small
2015–2021 training dataset improve its forecast skill? Use the **standard GC
loss**, corresponding to **λ=0** in the shared campaign. The five-value spectral
sweep is Ilya's item 1.

**Framing revised November7:** equal training budgets with GC–Mamba are not a
requirement for this item. Document the full-GC settings actually tested, their
training cost and the observed benefit or lack of benefit. The equal-budget
adaptation comparison belongs to the **unassigned LoRA item 7** below.

The implementation and training submissions already exist. Full-GC profiles and
500-update learning-rate calibrations completed; the final comparison remains
outstanding in the inspected records. Start by checking the existing campaign's
submission records and outputs before deciding whether recovery is needed.

The completed 500-update λ=0 calibrations at peak learning rates `1e-5` and
`3e-5` increased exact development rollout loss by **13.8958%** and **52.2477%**
relative to their public-GC references. These runs support the bounded statement
that **no further improvement was observed in those tested configurations**.
The lower-rate `3e-7` run has since **improved** the 32-start 2022 development
rollout loss by **4.6434%** at update 500 (day 10: **5.3090%**). It reached
update 1,250; its final checkpoint and 2023 evaluation remain pending. A general
claim that further fine-tuning cannot help is therefore not supported.

Suggested response based on the completed evidence:

> The GraphCast baseline has already been fine-tuned for multi-step prediction.
> Additional full-model fine-tuning at two higher learning rates worsened
> development loss, while a lower learning rate improved it by 4.64% at update
> 500. The final evaluation remains pending.

Update that statement to reflect all completed runs when writing the final
response, including any improvements found by the remaining experiments.

- [Existing campaign protocol and standard-loss GC revision](v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md).
- [Training campaign](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/manifest.json), [submission records](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/submissions), and [execution handoff](../../research/directions/r005-mgda-accuracy-spectral-adjustment/devel.md).
- [Active full-GC run mapping](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/selected_arms_full_gc.json): use the task with `lambda: 0.0` and its actual `id`.

Start from **public GC** and update all backbone parameters with the standard
loss. The existing submitted standard-loss run uses training years
2015–2021, batch four, 24-step BPTT/K20, 96-step segments, full weather-feedback
gradients and 2,500 optimizer updates. These describe the existing run; item 2
does not require matching GC–Mamba's training budget. Retain the submitted run's
planned endpoint and evaluation.

The current run's primary output, once completed, is its final native checkpoint:

```text
artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/
  runs/gcft_s22_l000_lr3em7_fresh_20261107/checkpoints/checkpoint_step00002500.pkl
```

This is the **raw, unaveraged endpoint**. The SWA proposed in the
[dedicated experiment README](README_FULL_GC_AR.md) is a separate checkpoint and
result. Evaluate the fine-tuned GC against the unmodified public GC baseline.
Include M1 as a separate correction reference; its historical training used
stopped weather-feedback gradients and SWA.

**Current GC λ=0 revision:** the active run is
`gcft_s22_l000_lr3em7_fresh_20261107`, starting fresh at peak LR `3e-7`.
Read the active mapping; the old 500-update calibration winner is not this run's
production parent.

**Deliverable:** exact GC loss and reduction versus public GC, rollout/day-10
results and paired intervals on the common 122-start 2023 evaluation, plus GPU
hours and memory. Summarize whether further fine-tuning helped under the tested
small-data settings, without claiming that the baseline cannot be improved by
other settings or more data. Pass the final standard-loss checkpoint,
configuration and results to Ilya for reuse as the GC λ=0 reference in item 1.
Use the 2022 development cohort for tuning decisions and the common 2023 cohort
for evaluation, as specified in the protocol.

The K8/K12/K16/K20 extension, checkpoint/SWA schedule, comparison with M1 and
remaining evaluation work are documented in the separate
[full-GC experiment README](README_FULL_GC_AR.md).

## 3. Extreme-event analysis — Lianghong

**Status checked for this handoff:** evaluation and report generation are complete
for **122 paired starts and 40 six-hour leads**. The completion record's report
checksum matches the saved report. The scientific interpretation remains to be
written; completed execution alone does not establish improved extreme skill.

Start from these existing outputs:

- [Campaign and checkpoint definitions](../../artifacts/evaluations/v24_extremes_2023_20261106/campaign.json), [completion record](../../artifacts/evaluations/v24_extremes_2023_20261106/complete.json), and [protocol notes](../../research/directions/r006-extreme-event-evaluation/experiments.md#e001).
- [Report](../../plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/report.md) and [full report metadata](../../plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/report.json).
- [Endpoint scores and intervals](../../plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/endpoints.csv), [per-lead scores](../../plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/per_lead.csv), and [figures](../../plots/analyze_models/images/resolution_eval/v24_extremes_2023_20261106).

The primary comparison is **M1 versus public GC**. The existing relative-power
MGDA SWA is a secondary comparison. The campaign uses local-calendar thresholds
from **1991–2020**, with q98 hot/wind/rain and q2 cold events, plus q95/q5
sensitivity, on the same 2023 dates.

**What to do:** summarize occurrence skill (CSI, recall, precision, false alarms
and frequency bias) separately from truth-tail intensity MAE/bias/RMSE. Cover hot
and cold 2-m temperature, 10-m wind speed and 24-hour rainfall, with global/land
results and paired intervals at days 1, 5 and 10. Rain scores begin at 24 hours.
Inspect both improvements and degradations; better conditional intensity errors
do not by themselves establish better event detection. These are extremes on the
sampled 1° ERA5 grid.

**Deliverable:** a compact table, selected figures and a paragraph explaining
which event types/lead times improve, worsen or remain uncertain. Reuse the
completed forecasts and thresholds.

## 4. Training-seed replication — Lianghong

**Recommended first scope for the paper:** reproduce the principal **M1
carry/carry versus matched reset/reset** comparison at seeds **23 and 24**.
Together with existing seed 22, this gives three training realizations per
policy. It adds four independently trained models, each with its own warm-up and
main stage. These repeats are proposed work; no new seed jobs were launched for
this handoff.

Configuration references:

| Policy | 500-update warm-up recipe | 2,000-update main-stage recipe |
| --- | --- | --- |
| M1 carry | [warmup_di16_fast_mamba.json](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/configs/warmup_di16_fast_mamba.json) | [di16_fast_mamba.json](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/configs/di16_fast_mamba.json) |
| Matched reset | [warmup.json](../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/configs/warmup.json) | [branch.json](../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/configs/branch.json) |

1. Initialize each realization from **public GC**, with fresh Mamba parameters
   and zero output heads under the original overlay recipe. Set the seed in both
   stages, use new output directories, and point each main stage to its own
   warm-up result. The trained M1/reset SWAs are comparison references, not
   initializers for independent training repeats.
2. Preserve the original M1 protocol, including `closed_loop_sg`, K20, batch
   four, optimizer schedules and Mamba learning-rate multiplier. Carry/carry
   retains memory; reset/reset clears both SSM state and convolution cache before
   every step in **both** stages. Retain the original sampling procedure rather
   than silently adopting the fresh λ campaign's shuffled training order.
3. Produce SWA from **main-stage updates 500, 1,000, 1,500 and 2,000**, after the
   independent 500-update warm-up. Evaluate each resulting SWA on the same 122
   starts and 40 leads with the common exact GC metric and matching memory policy.
4. Report each seed's rollout/day-10 reduction and carry-minus-reset contrast,
   with a descriptive mean and spread. Keep date-bootstrap uncertainty separate
   from variation across trained models; three seeds give limited precision.

Use the [matched-control source provenance](../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/source_provenance.json)
and [evaluation protocol](../../artifacts/evaluations/v24_matched_reset_2023_20261018/protocol.json)
to prepare the repeats. Freeze and record the runtime used. The historical
environment was not fully archived, so matching the configured recipe does not
establish bitwise reproduction of the original M1 run.

The broader [fresh λ plan](v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md)
also describes repeats of a development-selected **fresh GC/GC–Mamba pair**.
Those test robustness of the new comparison; keep them distinct from the
principal M1/control replication above. Repeating the entire λ grid is not needed
for this first seed study.

## 7. Equal-budget LoRA comparison — unassigned

This is the **equal-budget adaptation comparison**, rated **2/5 importance** with
moderate implementation difficulty. Nobody is assigned to it yet. Compare a
LoRA adaptation of the same public GC checkpoint with GC–Mamba using the same
available training data, rollout protocol and optimizer-update budget. Specify
LoRA's target layers and rank, and report trainable parameter counts, GPU hours,
memory and inference cost. Equal update counts do not imply equal compute cost.

Select configurations on development data and use the common forecast evaluation
protocol. This remains an open, lower-priority task; full-GC fine-tuning and the
matched reset control do not replace it.

## Reporting and coordination

Use exact original GraphCast loss for unqualified improvement percentages:
`100 * (1 - pooled_model_loss / pooled_public_gc_loss)`. Pool losses over the
requested dates/leads before taking the ratio. Label physical RMSE, spectral
diagnostics and event metrics separately.

Let valid bounded runs reach their planned endpoints and evaluation. Finite
loss increases, stalls, repeatability differences and restart drift are warnings;
retain their measurements and reference bounds and report separately whether a
decrease exceeds measured noise. Nonfinite computation, corrupt/mismatched
inputs or checkpoints, unusable execution and resource limits remain errors.

Keep submitted source snapshots and completed artifacts intact, and use the
existing campaign records to avoid duplicate work. Send Ilya the final tables,
figures, checkpoint/configuration identifiers and a short interpretation for
integration into the spectral study and reviewer response.
