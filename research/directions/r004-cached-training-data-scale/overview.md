# r004 — Does more cached training data improve each residual width?

Category: data
Status: active
Updated: 2026-10-28 (America/New_York)
Related: [superseded width-only proposal](../r003-fresh-residual-width/overview.md)

## Hypothesis and four-arm design

Does adding 2002–2014 to cached open-loop K20 training improve subsequent
closed-loop forecast skill? Does width512 benefit more than width128?

| Residual width | Open-loop training years | Closed-loop training years | Training needed |
| --- | --- | --- | --- |
| 128 | 2015–2021: 7 years | 2015–2021 | Reuse completed fresh di64 control |
| 128 | 2002–2021: 20 years | 2015–2021 | New |
| 512 | 2015–2021: 7 years | 2015–2021 | Reuse completed fresh control |
| 512 | 2002–2021: 20 years | 2015–2021 | New |

The user changed the larger-data arm from forty to twenty years on2026-10-28
after the forty-year preparation failed for insufficient storage. The original
[e001](experiments.md#e001) history and snapshots remain preserved; follow
[e002](experiments.md#e002) for the replacement. Only two new lineages are needed.
Reuse width128/7y (20k open search, selected14k, closed2k) and width512/7y
(20k open search, selected8k, closed2k), including their exact saved fresh
initial weights. Reuse valid resource and qualification evidence and bind new
cache/data identities during startup instead of repeating scientific pilots.

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

The seven-year cache has424 training chunks. The twenty-year aligned cache
will add complete2002–2014 historical segments while retaining all original
seven-year windows. Record exact counts and excluded boundary anchors from
its generated manifest. Equal20k update budgets trade repetition for data
diversity; selected parents may have different updates under the same fixed
selection rule. Report actual year exposure from the chronological sampler.
A null result applies to this budget, rather than all possible larger-data
training. A study that also matches epoch counts would change compute budget.

## Data controls and readiness

Use2022 for validation/selection and the existing122-start2023 benchmark for
evaluation. Preserve the original2015–2021 training chunks and2022 validation
chunks exactly inside the large-data construction. Add earlier complete
segments without shifting the shared weather restart boundaries. If the
twenty-year builder does not preserve those boundaries, implement an
explicit compatible cache/index construction. Verify shared timestamps,
inputs, stats and baseline predictions before treating it as a matched superset.

The seven-year cache is complete. Reuse the completed1982–2022 download,
preparing only2002–2022 in new dedicated roots. The conservative payload
estimate is1.494614TiB including the unchanged256GiB reserve; the design check
found2.212723TiB available. Cache manifest construction refines the estimate.
No source redownload or data deletion is needed. Both widths share one aligned
cache. Preserve source weather, normalization, baseline and shared prediction
identities before releasing training.

## Evaluation and decision

Evaluate each selected parent and final checkpoint on common32-start2022
cold rollouts for development diagnostics. Freeze the four final checkpoints
before evaluating the same122 starts in2023,40 six-hour leads, zero initial
memory and corrected feedback.2023 is outside residual training but has prior
project exposure; it is not an untouched test set.

For each width, the primary quantity is the twenty-year minus seven-year gain
in exact pooled GraphCast loss reduction against one common baseline. Positive
is better. The difference of those gains across widths is a secondary
interaction; also report day10 and compute/storage cost. Use paired start-block
bootstrap uncertainty following the
[existing2023 protocol](../../../docs/experiments/v24_2023_stride3_memory_2026-10-04.md),
with simultaneous intervals for the two within-width primary comparisons.
This initial probe uses one training seed. Repeat promising effects with
additional seeds before claiming generality; forecast-start intervals do not
measure seed or year variability.

## Current evidence and next action

The seven-year finals remain the only completed data-size controls: on32 matched
2022 starts and40 six-hour leads, exact loss reduction against one common saved
baseline is10.784610% for width128 and6.443408% for width512; day10 is16.848425%
and7.875807%. Width512's own saved baseline gives6.469032%/7.955506%. Finite
baseline and restart drift remain advisory. This is one seed and a development
cohort; no larger-data result or common four-arm2023 result exists.

The old forty-year producer failed before preparing arrays onOctober20 at
2.314TiB free versus2.680TiB required. Its fresh-consumer alignment and cache/
qualification jobs were left pending behind that failure. The October28
replacement preserves completed controls and old immutable snapshots and uses
new producer/consumer roots; old unstarted forty-year descendants are superseded.
The separate warm-start forty-year adaptation experiment remains historical
context and is outside the two fresh twenty-year lineages.

The replacement is submitted: source verification16392462 completed and
automatically released preparation16392486 (queued). Cache chain16392463–66
and CPU startup16392467/68 are active or pending their required data. The five obsolete unstarted40y
submissions were cancelled and their old release disabled. Complete the
twenty-year preparation, verified cache and two fresh lineages,
then evaluate the prescribed within-width data gains after freezing all four
finals. See [e002](experiments.md#e002) for current submission evidence and
[development handoff](devel.md) for resource and recovery details.
