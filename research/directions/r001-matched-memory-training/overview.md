# r001 — Does training with memory improve the selected configuration?

Category: training
Status: active
Updated: 2026-10-18 (America/New_York)

## Question and hypothesis

Does retaining memory during training and forecasting improve exact GraphCast
loss relative to resetting memory at every prediction, with the selected model's
architecture and optimization protocol held fixed?

## Current understanding

The selected di16/Mamba-LR×3 checkpoint has complete evaluations on 122 starts
throughout 2023. Its ten-day exact GraphCast loss reduction is 14.4240% with
memory carry and 4.5725% with inference-time reset. This establishes dependence
on carried state; the inference intervention introduces training/forecasting
mismatch and does not isolate the benefit of training with memory.

The older V22 reset-trained model belongs to a different training configuration.
Its effect cannot be assumed equal to a reset-trained version of the selected
model. The user authorized a new matched reset-training run on 2026-10-15.
[Attempt e001](experiments.md#e001) changes the per-step training memory policy
in both original optimization stages. Its completed matched evaluation is
recorded below; the earlier configuration is retained only as historical evidence.

Training job15622842 completed its 500-update warm-up, 2,000-update main stage,
and prescribed four-checkpoint SWA. The SWA SHA256 is
`db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`.
CPU preflight verifies all10,348,879 parameters are finite FP32 and that their
tree matches selected M1. Configuration differences are the intended memory
policy and paths/independent reset-trained warm-up. Historical training source
recovery was incomplete; the exact new runtime is frozen, without claiming
bitwise historical reproduction.

The user authorized completing evaluation on October18. [Attempt e002](experiments.md#e002)
uses the original frozen evaluator, unchanged122-start/40-lead2023 manifest,
cold full-feedback forecasts, zero initial memory and per-step reset. Array
15788731 evaluates eight shards; report15788744 depends on all shards succeeding.
Existing M1 and M1_reset forecasts are reused. This is a new planned training-policy
contrast with paired intervals and sensitivity analyses; it does not inherit
the original confirmatory endpoint family.

All eight shards and report15788744 completed successfully. Complete24-shard
coverage (three policies ×122 starts ×40 leads) was independently rechecked,
and pooled raw losses reproduce the report. Exact GC loss reduction is14.4240%
for carry/carry,4.5725% for carry/reset, and9.1627% for matched reset/reset.
Carry/carry exceeds matched reset/reset by5.2614 percentage points over the
rollout (paired pointwise95%CI4.698–5.862; see report for full precision)
and8.0225 points at day10 (CI7.202–9.047). All prescribed block-length,
quarterly and pilot-exclusion sensitivities retain a positive rollout interval.
Own-baseline denominator sensitivity gives5.341 points, preserving the result.
This supports a benefit from training/forecasting with persistent memory for
this configuration and training realization, beyond the inference-reset result.
It does not measure training-seed variation. The repeated baseline has0.0872%
pooled drift and4.3923% worst start/lead drift; these remain disclosed numerical
warnings under the frozen advisory policy.

## Next decision

The user authorized manuscript replacement on October 18. The matched result
now replaces the provisional historical reset/reset result in
[`Overleaf/arxiv_versoin.tex`](../../../Overleaf/arxiv_versoin.tex) and its rebuilt
PDF, including Figure 6 and its reproducible source/data exports. Carry/reset
remains the distinct inference-only intervention. The contribution, conclusion,
training description, and new uncertainty appendix state the fixed-checkpoint,
single-training-realization scope, separate planned analysis, historical source
limitation, and numerical denominator sensitivity.

No additional run is required for this replacement. Replication across training
seeds would address the remaining unmeasured training variability if broader
generalization is sought.

## Records

- [Attempts and evidence](experiments.md)
- [Prepared experiment](../../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/manifest.json)
- [Frozen evaluation protocol](../../../artifacts/evaluations/v24_matched_reset_2023_20261018/protocol.json)
- [Evaluation submission](../../../artifacts/evaluations/v24_matched_reset_2023_20261018/submissions/jobs.json)
