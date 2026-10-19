# r003 — Does residual width improve fresh open-loop to closed-loop training?

Category: training
Status: closed — superseded before execution
Updated: 2026-10-15 (America/New_York)

**Correction:** The user clarified that the intended hypothesis concerns larger
training data. The seven-year-only proposal below is archived, not the current
recommendation. No jobs were submitted. Follow the corrected
[7-versus-40-year design at both widths](../r004-cached-training-data-scale/overview.md).

## Question and hypothesis

Does increasing the spatial residual width from128 to512 improve cold,
corrected-feedback forecast skill after fully fresh cached K20 training and
closed-loop fine-tuning, at a common training and checkpoint-search budget?
The hypothesis is that the larger spatial branch improves forecast skill;
its additional training and inference cost must also be reported.

## Recommended data span

Use **seven training years, 2015–2021, for both widths**. Use2022 for validation
and checkpoint selection, then evaluate frozen checkpoints on the existing
122-start2023 benchmark. This is a temporal out-of-training comparison;2023
has already been examined in this project and is not an untouched test year.

The [existing K20 cache](../../../data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20/manifest.json)
has424 training chunks in106 complete96-step segments, plus60 validation
chunks in15 segments. Its READY marker covers all8 shards and484 chunks.
It can serve either residual width and either spatial initialization policy.
The larger1982–2022 cache is absent as of this design; a40-year training-data
comparison should be a separate, paired follow-up once that cache is ready.

## Matched architecture and training

| Setting | Both arms |
| --- | --- |
| Residual spatial width |128 or512: the experimental factor |
| Residual initialization | `fresh`; zero spatial tensors copied from GraphCast |
| Spatial depth |2 processor steps |
| Mamba |Mamba1, di64, BCgroups1, explicit dt_rank32, state16, conv4, two layers per insertion |
| Baseline |Same frozen pretrained GC Small, width512,16 processor steps |
| Sequence |K20, BPTT24,96-step segments, batch1 |
| State |Carry SSM/conv values within a segment; detach at optimizer boundaries; reset at segment boundaries |
| Numerics and loss |Maintained stepwise BF16 computation/FP32 weather tape, all-step GraphCast-weighted MSE; no auxiliary loss |
| Optimizer |AdamW betas(0.9,0.98), weight decay1e-4, gradient clip1 |
| Initial training seed |22 for both; this first comparison does not estimate training-seed variability |

Fresh means no pretrained residual encoder, processor or decoder weights.
The new residual output head and Mamba output projections retain their zero
initialization. Set `architecture.width=512` for the baseline and
`residual_width=128` or512 for the trainable branch. Both branches train all
residual spatial and temporal parameters; the separate baseline stays frozen.

1. **Cached open-loop K20:**20,000 updates, constant LR1e-4, **no warmup**,
   matching the existing fresh width128 protocol. Checkpoint and validate every
  2,000 updates on all15 complete2022 segments. Select the lowest exact FP32
   original GraphCast validation loss from this ten-checkpoint grid; ties go
   to the earlier update. This is a common training/search budget, not a claim
   that the selected parents have identical update counts.20k is about47.2
   passes over the424 training chunks. Weather restarts follow the cache's
  24-step chunk schedule;96 steps are not one uninterrupted weather forecast.
2. **Closed-loop K20:** transfer all weights from each selected parent, create
   a fresh Adam state/RNG/cursor, and zero recurrent state. Train2,000 additional
   updates with corrected-weather feedback and stop-gradient through weather,
   LR1e-5 and200-update warmup. Save every500 updates and track the same fixed
   eight2022 validation segments. The final2,000-update checkpoint is the
   prespecified primary endpoint; intermediate checkpoints are descriptive.

Use the same immutable runtime for both widths. The existing fresh width128
di64 run already matches the intended architecture and open-loop schedule;
its selected14k parent and2k continuation may be reused after checking the
full config, runtime, data and state semantics. Otherwise rerun it. The prior
width512 overlay result is historical context, not a matched control: it also
used BCgroups2 and dt_rank=auto. No new runs are submitted by this design.

## Evaluation and interpretation

Evaluate each selected open-loop parent and final closed-loop checkpoint on
the same32 starts in2022, then the two final checkpoints on the existing122
starts in2023,40 six-hour leads, cold zero memory and corrected feedback.
Freeze checkpoints before the2023 evaluation. Reuse a single compatible
GraphCast baseline and disclose repeated-baseline numerical discrepancies.

Primary comparison: the difference between widths in pooled exact original
GraphCast loss reduction,100*(1-sum(model loss)/sum(baseline loss)), across
all2023 starts and leads. Report day10 as secondary, and show parent-to-final
changes on2022 without treating2022 as independent confirmation. Use paired
start-block bootstrap uncertainty following the
[existing2023 protocol](../../../docs/experiments/v24_2023_stride3_memory_2026-10-04.md).
These intervals condition on the trained seed and year; they do not establish
robustness over training seeds. Repeat both arms with additional seeds before
claiming a general width advantage. A negative or inconclusive width result
is a valid outcome. Two thousand closed-loop updates tests a fixed budget,
not convergence.

## Compute planning

The existing fresh width128 di64 cached20k run took37h12m (job15295330).
The previous width512 cached4k run took12h34m (job15295411); a linear projection
is about63 GPU-hours for20k, with initialization and validation overhead and
different Mamba settings making this approximate. The width512 closed-loop2k
run took6h40m (job15575009). These estimates exclude queue time and final
evaluation. Historical reuse could avoid repeating width128 training.

CPU-RAM observations were312.64GiB,400.00GiB and247.96GiB, respectively.
The cached width512 run reached its400GiB allocation in accounting; that is
not evidence of spare headroom. Measure the actual fresh configuration before
fixing submission resources. Keep long training resumable in immutable stages.

## Next decision

Implement the paired protocol when execution is requested: audit reuse of the
existing width128 control, freeze the matching fresh width512 config/runtime,
and qualify its initialization, cache compatibility and resource usage.
See [attempt e001](experiments.md#e001) for inspected evidence and limits.
