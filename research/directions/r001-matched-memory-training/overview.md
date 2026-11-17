# r001 — Does training with memory improve the selected configuration?

Category: training
Status: active
Updated: 2026-11-14 (America/New_York)

## Question and hypothesis

Does retaining memory during training and forecasting improve exact GraphCast
loss relative to resetting memory at every prediction, with the selected model's
architecture and optimization protocol held fixed?

## Current understanding

**November 14 presentation decision:** The active manuscript and reviewer
response now focus on carry/carry versus independently trained reset/reset.
The user removed the carry/reset inference-only diagnostic as peripheral to
this comparison. Figure 6 retains its original matched SWA/date-interval panel
and independent raw-checkpoint panel with five carry seeds and one reset seed;
the carry/reset curves and associated prose/table row are omitted. All retained
scores and uncertainty bands are unchanged. Historical carry/reset evidence is
preserved below and in archived artifacts, not promoted as the manuscript's
memory test. See the [revision record](experiments.md#two-memory-policies-20261114).

**November 14 user clarification, reaffirming November 13:** The earlier
eight-start 2023 spectral pilot did not influence principal model selection or
similar decisions. Do not frame it as test-set tuning or raise its results in
the manuscript or reviewer response. The user explicitly rejects the sentence
about earlier pilot exposure and its associated discussion. `v4.tex` removes
that sentence, the Appendix A.3 pilot-exclusion sensitivity/result, and the
Appendix C pilot-exposure sentence. Historical runs and analysis artifacts stay
available as internal provenance. This supersedes the earlier note to retain
manuscript exposure disclosures; it does not change the actual evaluation data,
scores or uncertainty calculations. See the
[updated clarification](experiments.md#pilot-wording-cleanup-20261114).

**November 13 existing-data replication update ([e003](experiments.md#e003)):**
Five fresh local M1 carry/carry retrainings, seeds 22–26, now measure training-seed
variation for this reconstructed recipe. Their SWAs reduce exact GraphCast loss
by **14.1682% ± 0.4332 percentage points** over the ten-day rollout and
**21.0283% ± 0.7502 points** at day 10 (mean ± sample SD). Every seed has all
122 starts × 40 leads. The independent local seed 22 is not the published seed-22
checkpoint; the historical configs/environment were unavailable locally, and
local verification includes NCAR repairs at three valid times. These five runs
therefore remain a separate replication cohort, not six pooled seeds with the
original selected model.

The new v4 Figure 4 shows five-seed SWA curves and ±1 seed SD. Figure 6 preserves
the original matched-SWA result and date-bootstrap intervals in one panel and
adds local raw main-step-2,000 carry/control results in another. The local
reset/reset control still has **one training seed**; its raw checkpoint cannot
be mixed with the five SWAs to claim a multi-seed matched memory contrast.
Figure preparation reused existing evidence without training, forecast generation
or job submission. The original arXiv source/assets remain unchanged by this
November update. Checked v4 (10 main pages, 24 total), figures, data and
response draft were pushed to Non-Markovian-Weather/main as `a5df67a`; no
OpenReview or arXiv upload was performed. The inherited AI-use statement
needs author correction for the new analytical assistance.

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

The November 13 update is prepared for [`Overleaf/v4.tex`](../../../Overleaf/v4.tex)
using the separate figures and provenance in [e003](experiments.md#e003).
Finish the v4 build/review and record publication separately. Carry-model seed
variation is now measured; matched reset-trained seed variation remains
unmeasured. Further paired carry/reset training replications would address
broader training-policy claims, but are not required to present this existing
evidence and have not been launched.

## Records

- [Attempts and evidence](experiments.md)
- [Prepared experiment](../../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/manifest.json)
- [Frozen evaluation protocol](../../../artifacts/evaluations/v24_matched_reset_2023_20261018/protocol.json)
- [Evaluation submission](../../../artifacts/evaluations/v24_matched_reset_2023_20261018/submissions/jobs.json)
