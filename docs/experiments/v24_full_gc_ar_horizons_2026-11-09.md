# Full GraphCast: AR horizons, checkpoint spectra and SWA

**Date:** 2026-11-09, America/New_York.  
**Status:** on hold for reviewer-comparison revisions; new jobs have not been submitted. The prepared code and resources passed technical checks, but the current submission plan does not yet include the M1 or 2023 comparisons described in the audit below.  
**Question:** How do the number of feedback-conditioned training steps and
training duration affect forecast accuracy and power spectra when fine-tuning
the public full GraphCast model with the standard GC loss?

The user requested K8, K12, K16 and K20, 2,500 updates, evaluation at every 500
updates, possible SWA, and a spectrum comparison for every checkpoint. K20 is the
existing fresh `3e-7` run; its first 1,250 updates have completed and continuation
job `17049163_0` was pending at the 2026-11-09 queue check. The K20 500-update
checkpoint reduced exact rollout loss by **4.6434%** versus public GC on 32
development starts in 2022; its day-10 reduction was **5.3090%**. Those are
fixed-checkpoint development results, not final or 2023 results. See the
[K20 evaluation](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/calibration_evaluation/gcft_s22_l000_lr3em7_fresh_20261107/evaluation.json).

## Reviewer-comparison audit — before submission

**Verdict:** the prepared four-horizon experiment is useful for understanding
full-GC training, but its current output is not yet sufficient for the intended
reviewer-facing comparison with M1. Do not submit this draft as the complete
reviewer campaign. Keep the existing K20 continuation and its downstream jobs.

1. **Include principal M1 on exactly the same dates.** The prepared report
   currently plots only full-GC horizons, public GC and ERA5. The existing
   [M1 smoothing campaign](../../artifacts/evaluations/v24_smoothing_m1_20261107/manifest.json)
   already contains per-initialization exact-loss records for the same 32 2022
   development starts, and raw M1/GC/truth spherical statistics for all 122
   2023 starts. The four development-shard metric hashes and the 32-date set were
   checked against the prepared manifest. Reuse those records. M1's 2022
   calibration files do not save raw spherical powers, so a same-cohort M1
   spectral forecast would be needed to overlay it on each development spectrum.
   Repeated public-GC rollout losses on these dates were 10.822104 in the M1
   calibration and 10.840716 in the new K20 evaluation: an absolute difference
   of 0.018612 (0.1720% relative to the former). Preserve this finite difference
   as a numerical warning and use one common baseline for headline comparisons;
   it is not a release gate.
2. **Add the 2023 reviewer-facing comparison.** The prepared AR sweep evaluates
   only the 32 development starts used to select M1 and assess GC settings.
   After development, compare full-GC finalists with principal M1 and public GC
   on the existing 122-start, 40-lead 2023 cohort. Reuse M1's saved forecasts
   and the existing K20 final 2023 evaluation when available. A practical
   proposal is the final raw checkpoint and SWA for each horizon, plus any
   intermediate checkpoint selected on 2022 if it is a distinct finalist. Freeze
   the selection rule before examining 2023. Reuse the established paired
   block-bootstrap for fixed-checkpoint differences and disclose prior 2023
   exposure; the intervals do not measure training-seed variability.
3. **Keep the training differences visible.** Full GC and M1 share the public
   backbone, training years, batch size, K20 window and total update count, but
   the historical M1 uses stopped weather-feedback gradients, a two-stage
   schedule with optimizer reset, and its original sampler; the prepared GC
   arms use full feedback gradients, a single cosine schedule and a frozen
   shuffled order. M1 trains 10,348,879 correction parameters; full GC trains
   35,979,347 parameters. These are valid model-specific training choices, but
   the result must not be described as isolating architecture or memory, nor as
   a perfectly identical training protocol. Report schedules, checkpoint rule,
   GPU-hours and memory alongside skill. The existing fresh GC–Mamba/full-GC
   λ=0 comparison supplies a separate, more closely matched training protocol.
4. **Answer the spectral question with more than power amplitude.** Preserve
   the prepared raw spherical power for every development checkpoint, and
   compare M1 and the 2023 GC finalists on the same dates using forecast and
   error spectra, power ratios and spectral correlation. The driver saves
   error/cross powers but currently plots forecast power only. For smoothing,
   reuse the existing development-fitted GC/M1 filters and the separate λ
   campaign's independently fitted filters. A new horizon finalist needs the
   same filtering opportunity before drawing a filtering-controlled conclusion.
   Power matching alone does not establish spatial or phase skill.
5. **Keep conclusions bounded.** The `3e-7` GC learning rate is promising at
   update500 on 2022, but one learning rate across four horizons cannot establish
   that GC is fully optimized. All horizons and fixed checkpoints should be
   reported, including unfavorable results; no finite-metric acceptance gate is
   introduced. K changes both the feedback tail and teacher prefix in the
   fixed24-step design. The campaign does not replace seed replication, a LoRA
   comparison, or matched GRU/LSTM/fixed-history experiments.

The added reviewer-facing evaluations and their selection/reporting logic are
**recommendations, not yet implemented** in the frozen campaign. The task was
paused for this review; no new training or evaluation jobs were submitted.

## Design

| Arm | BPTT window | Teacher-forced prefix | Feedback-conditioned tail | Source |
| --- | ---: | ---: | ---: | --- |
| K8 | 24 | 16 | 8 | New full-GC run |
| K12 | 24 | 12 | 12 | New full-GC run |
| K16 | 24 | 8 | 16 | New full-GC run |
| K20 | 24 | 4 | 20 | Existing full-GC run; reuse its checkpoints |

Each step is six hours. Keeping the 24-step window, objective reduction,
96-step segments and frozen update-to-segment mapping identical avoids changing
the sampled training windows. It means that varying K also varies the number
of teacher-forced steps. This is a fixed-window teacher-forcing comparison,
not a comparison with a fixed four-step teacher prefix and varying window length.
That interpretation should accompany every result.

The three new arms start independently from the same public GraphCast-small
checkpoint as K20, not from a trained K20 checkpoint. Full GC parameters are
trainable. All four use seed 22, the same 2015–2021 training data and frozen data
order, batch 4, `closed_loop` full weather-feedback gradients, exact GC grid MSE
(λ=0), AdamW, 100-update warm-up, peak LR `3e-7`, cosine decay to `3e-8`, and a
fixed 2,500-update schedule. The only intended new-arm differences are K and
the output location; actual costs and any finite drift remain recorded.

Evaluate native checkpoints at **500, 1,000, 1,500, 2,000 and 2,500** updates.
Build an inference-only, equal-weight **SWA of updates 1,000, 1,500, 2,000 and
2,500**, independently for each arm, and evaluate it too. Report the final raw
checkpoint and SWA separately; the SWA is not a resumable optimizer state.
The first two checkpoint evaluations can run after the 1,250-update training
part while the second part is queued or running. Later checkpoints and SWA are
evaluated after the final training part. Valid runs reach the scheduled endpoint
and evaluation regardless of finite loss increases, stalls or weak metrics.

## Evaluation and outputs

Use the exact same **32 development initializations from 2022**, 40 six-hour
forecast leads, cold full-feedback protocol, public-GC checkpoint, truth, and
normalization statistics used for the K20 calibration. These are development
comparisons; 2023 confirmation is not part of the prepared run.

Each of the 24 model/checkpoint combinations (4 horizons × 5 raw checkpoints
plus 4 SWAs) saves exact original GraphCast loss and spherical T179 physical
forecast, truth, cross and error power for all 83 primitive channels and all
40 leads. The report pools powers over initializations before deriving comparisons.
Plots compare K8/K12/K16/K20, public GC and ERA5 at days 5 and 10 for T2m,
Z500 and U850; all other channels and leads remain available in the pooled NPZs.
Degree zero is retained in data and omitted from logarithmic plots. Physical
spherical power is distinct from exact discrete-grid GC loss.

For the main loss reductions, use the public-GC rollout from the K20 update-500
evaluation as one common denominator. Preserve repeated-baseline differences,
their reference bounds and warnings; separately report whether each loss change
exceeds the observed baseline-only spread. That spread is a limited numerical
reference, not full model repeatability, and never a gate. The evaluation uses
the same single training seed for all runs; no seed robustness claim follows.

## Provenance and execution

- [Campaign manifest](../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/manifest.json)
  and [submission plan](../../artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109/submission_plan.json).
- [Driver](../../scripts/experiments/run_v24_full_gc_ar.py),
  [Slurm wrapper](../../scripts/experiments/v24_full_gc_ar.slurm), and
  [focused tests](../../tests/v24_Ilya/test_full_gc_ar_campaign.py).
- Frozen source digest:
  `77f264914960a3bb471a8588b3594ad9dbe836d6267da8b6f76432bfb27903d3`.
  Parent K20 source digest:
  `ab39c404b062a55f2ee93883672442973fa4a2351699529ac11492a1a6b88475`.
  Every source file in the parent snapshot matched the workspace before the
  new campaign snapshot was frozen; the new driver and wrapper were added.
- Existing K20 training and the λ/smoothing campaign remain unchanged. Its
  update-500 loss was already evaluated, but new forecasts are needed to save
  the requested spherical spectra for each checkpoint.

The launcher uses two 1,250-update GPU parts for each new horizon, eight early
GPU evaluation tasks, four CPU SWA tasks, sixteen later GPU evaluation tasks,
and a CPU report. Arrays are unthrottled. The SWA stage waits for both the three
new continuations and the existing K20 continuation. If K20 has already
completed when submitted, the launcher verifies its final checkpoint instead
of relying on an expired dependency.

Measured K20 job `17049162_0` used one 80 GB GPU, 8 CPUs and 384 GiB host RAM;
1,250 updates took 20h07m26s, with `.batch` MaxRSS 235,440,960 KiB and steady
p90 57.799 s/update. Each new training part requests the same GPU/CPU/memory
and 30h walltime. A similar spherical evaluation used 9.8 GiB and 45.8 minutes
for 18 starts; the prepared 32-start evaluation requests one 80 GB GPU, 4 CPUs,
20 GiB and 150 minutes. Approximate actual compute is **120 GPU-hours** for the
three new training trajectories plus roughly **32 GPU-hours** for 24 evaluations,
before queue variation; K20's already-submitted continuation is additional
existing work. Training/evaluation Slurm requests passed `sbatch --test-only`.

Focused tests cover all horizon configurations and checkpoint/task mappings,
real spherical observer transform and reporting, scheduler dependencies, finite
diagnostics, and full-GC SWA. The real existing K20 update-500/1000 checkpoints
were also averaged on CPU in a temporary directory; the averaged first parameter
leaf and `full_gc` model-family identity were verified. No GPU qualification run
was added. `sbatch --test-only` is validation, not a job submission or reservation.

The prepared launcher submits the dependency chain with:

```bash
source scripts/graphcast_env.sh
python scripts/experiments/run_v24_full_gc_ar.py submit \
  --experiment artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109
```

Submission records and Slurm job IDs appear under the campaign `submissions/`
directory. Do not refresh its source snapshot or overwrite existing runs.
