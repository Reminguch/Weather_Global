# r004 — Does more cached training data improve each residual width?

Category: data
Status: active
Updated: 2026-10-17 (America/New_York)
Related: [superseded width-only proposal](../r003-fresh-residual-width/overview.md)

## Hypothesis and four-arm design

Does adding 1982–2014 to cached open-loop K20 training improve subsequent
closed-loop forecast skill? Does width512 benefit more than width128?

| Residual width | Open-loop training years | Closed-loop training years | Training needed |
| --- | --- | --- | --- |
| 128 | 2015–2021: 7 years | 2015–2021 | Reuse completed fresh di64 control |
| 128 | 1982–2021: 40 years | 2015–2021 | New |
| 512 | 2015–2021: 7 years | 2015–2021 | New fresh control |
| 512 | 1982–2021: 40 years | 2015–2021 | New |

This requires three new training lineages, not four. The existing width128
di64 arm completed20k cached updates, selected its14k checkpoint and completed
2k closed-loop updates. Its2022 evaluation is complete; the common2023
evaluation must be checked/run as needed. Preserve its protocol for the new
arms rather than introducing a sampler change that forces a rerun.

Common downstream fine-tuning isolates the benefit of larger open-loop
pretraining. Expanding closed-loop data too would be a separate follow-up
estimating the benefit of expanding both stages.

All residual grid-to-mesh, mesh processor, mesh-to-grid and Mamba parameters
are freshly initialized; residual/Mamba output projections retain their zero
initialization. Both data arms within a width share the exact saved initial
weights and optimizer state. The separate pretrained GraphCast baseline stays
frozen. No new arm initializes from trained residual weights.

## Fixed training protocol

- Same di64, B/C groups1, explicit dt-rank32, state16, conv4, two temporal
  layers, two spatial processor steps, K20, BPTT24, segment96, batch1 and seed22.
  Set baseline architecture.width=512 and residual_width=128 or512.
- Open loop: 20,000 updates, 200-update warmup to constant LR1e-4, AdamW betas(0.9,0.98),
  weight decay1e-4, clip1, all-step MSE, maintained BF16 compute/FP32 weather
  tapes. Save checkpoints every1,000 updates. Validate all15 common 2022 segments every2,000 updates. Select the
  lowest exact original GraphCast validation loss on this fixed grid; ties
  go to the earlier update. Also report the fixed20k validation endpoint.
- Closed loop: transfer selected weights, reset optimizer/RNG/cursor/memory,
  then train2,000 updates on the identical seven-year split in all arms, with
  corrected-weather feedback and stop-gradient, LR1e-5 and200-update warmup.
  The final2k checkpoint is the prespecified primary forecast endpoint.
- Preserve the existing control's deterministic chronological segment traversal
  in every arm. Keep all four chunks within each segment ordered; carry memory
  within segments, detach at optimizer boundaries and reset between segments.
  Save epoch/segment/offset for exact resume. Report actual per-year exposure:
  partially completed epochs overweight the earlier dates. Use the control's
  frozen training implementation, checking any necessary compatibility changes.

The seven-year cache has424 training chunks; the aligned forty-year construction
expects2,432 training chunks (502 added historical segments plus106 original
seven-year segments). It omits18 transition anchors to preserve the original
restart boundaries. At batch1,20k updates is
about47 versus8 passes. Equal updates give equal sampled-window budgets:
the larger arm trades repetition for diversity. Selected-parent updates may
differ under the common checkpoint-search rule. A null result means no gain
demonstrated at this budget, not that larger data can never help. Matching
passes would need about115k large-data updates and adds a compute difference;
any such convergence study must be reported separately.

## Data controls and readiness

Use2022 for validation/selection and the existing122-start2023 benchmark for
evaluation. Preserve the original2015–2021 training chunks and2022 validation
chunks exactly inside the large-data construction. Add earlier complete
segments without shifting the shared weather restart boundaries. If the
current forty-year builder does not preserve those boundaries, implement an
explicit compatible cache/index construction. Verify shared timestamps,
inputs, stats and baseline predictions before treating it as a matched superset.

The [seven-year cache](../../../data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20/manifest.json)
is complete. At submission on2026-10-15 at21:15EDT, the forty-year cache is
pending; source downloading has completed1982–2002 and is processing2003.
Cache readiness is a prerequisite
for the actual hypothesis, not a reason to replace it with a width-only test.
Keep normalization and the frozen baseline common. Adding older years also
changes the historical distribution and overlap with baseline pretraining;
the interpretation is specifically the benefit of adding these years.

## Evaluation and decision

Evaluate each selected parent and final checkpoint on common32-start2022
cold rollouts for development diagnostics. Freeze the four final checkpoints
before evaluating the same122 starts in2023,40 six-hour leads, zero initial
memory and corrected feedback.2023 is outside residual training but has prior
project exposure; it is not an untouched test set.

For each width, the primary quantity is the forty-year minus seven-year gain
in exact pooled GraphCast loss reduction against one common baseline. Positive
is better. The difference of those gains across widths is a secondary
interaction; also report day10 and compute/storage cost. Use paired start-block
bootstrap uncertainty following the
[existing2023 protocol](../../../docs/experiments/v24_2023_stride3_memory_2026-10-04.md),
with simultaneous intervals for the two within-width primary comparisons.
This initial probe uses one training seed. Repeat promising effects with
additional seeds before claiming generality; forecast-start intervals do not
measure seed or year variability.

## Submitted campaign and next action

The already authorized [forty-year width512 campaign](../../../artifacts/checkpoints/v24_Ilya/res1_cached_40yr_w512_di64_20261014_v5/campaign.json)
uses trained parent weights, B/C groups2, batch4 and5k updates. It is a separate
warm-start adaptation experiment and cannot replace a fresh arm here. It has
not been cancelled. Its frozen scientific source and settings are unchanged;
a separate CPU compatibility bridge handles Della's rejection of an explicit
GPU partition when releasing its next jobs.

The user authorized all three missing lineages on2026-10-15. Qualification
jobs **15656209** (width512/7y), **15656210** (width128/40y), and **15656211**
(width512/40y) are submitted. As of2026-10-17 at17:25EDT, the seven-year
width512 arm has passed qualification and completed12,000/20,000 open-loop
updates. Its next4k stage15659895 is pending priority. Closed-loop training
and forecast evaluation have not started. Both forty-year arms still await
the prepared data and cache. The source download reports complete at04:16EDT
today, but producer gate15575578 and wait/bridge jobs remain pending a
maintenance reservation. The prepared-data and cache roots do not yet exist.

Best width512 cached validation so far is step8,000 (exact full-rollout loss3.693784,
1.390% reduction); step12,000 has worsened to0.607% reduction. These are
15-segment open-loop validation scores, not the final closed-loop forecast
benchmark. No data-size comparison is available. The existing automatic
pipeline continues through20k, selection, closed2k and evaluation.

The [campaign manifest](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/manifest.json)
pins all three configurations, the reused control and immutable runtime.
The [state ledger](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2/state.json)
will accumulate later job IDs. Scientific evidence for the data-size hypothesis
is still pending. Next inspect qualification and cache proofs, then compare the
prespecified within-width gains after all four finals complete evaluation.
See [e001](experiments.md#e001) for job IDs, dated corrections and validation,
and [development handoff](devel.md) for resources and recovery commands.
