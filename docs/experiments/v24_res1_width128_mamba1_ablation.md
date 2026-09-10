# Res1 width-128 ablation against the best completed v24 Mamba1 reference

Status: revised at user request on 2026-09-09 to reuse the existing best Mamba1
run as the comparator. Only width-128 training task **13656412_1** and its
evaluation **13656423_1** remain scheduled. The fresh width-512 task and its
evaluation were cancelled before starting. The obsolete three-arm summary job
13656432 was cancelled; replacement summary **13657089** depends only on
13656423_1 and is recorded in `jobs.json`.
Concrete configs, source snapshot and job records:
`artifacts/checkpoints/v24_Ilya/res1_width128_mamba1_20260909/`.
Evidence inspected: 2026-09-09.

## Question and reference

Can a width-128 residual GC-Mamba branch retain the correction quality of the
width-512 branch under the same training budget? The frozen forecasting baseline
remains GraphCast_small, resolution 1 degree, mesh 5, width 512, 16 processor steps.

The strongest completed matched-32 result found among local v24 res1 Mamba1
evaluations is:

- Run: `artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904/mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k`.
- Configuration: the run's `run_config.json` (use the saved configuration as the source of truth).
- Checkpoint: `swa/swa_step02000-08000.pkl`.
- Evaluation: `eval/cold_full_zero_exact_gc/matched_v22_reference32/swa_step02000-08000.json`.
- Exact original GraphCast rollout loss reduction: **16.282891%**.
- Residual branch: **10,348,879** parameters; frozen baseline: **35,979,347**.

This reference uses Mamba inner dimension **16**, BC groups **1**, rather than
the di64/bcg2 configuration discussed initially. Match the reference for this ablation.

## Initialization is a separate experimental factor

The reference copies 94 parameter leaves from pretrained GC into its residual
spatial branch, leaving 46 fresh. This is recorded in the saved configuration
and implemented by `overlay_matching_params` in the training runner.
Width-128 spatial matrices cannot use that same overlay. Comparing a freshly
initialized narrow branch only against the existing pretrained wide branch
would mix width and initialization effects.

The user selected a direct comparison with the existing best run, requiring
only one new training run:

| Arm | Residual width | Residual spatial initialization | Role |
| --- | ---: | --- | --- |
| A | 512 | Existing pretrained GC overlay | Reuse the best reference as the practical target |
| C | 128 | Fresh; disable all residual GC overlay | Narrow candidate |

Compare C versus A to assess whether the cheaper model is competitive with our
best run. Document that this includes both width and spatial initialization
differences; it does not isolate width alone. The initially proposed fresh
width-512 initialization control was removed at user request. Fresh means the normal spatial
initializers, Mamba1 initialization, and the existing zero-output initialization
settings; do not zero all weights. Do not transfer a trained residual checkpoint,
slice pretrained weights, or leave a partial overlay of shape-compatible biases.

## Settings held fixed

Clone the saved reference configuration, changing only output paths, the explicit
residual initialization policy, and residual width for C. The new candidate uses:

- Residual processor steps: 2; temporal location: `mesh_processor_interleaved`.
- Mamba: `d_inner=16`, `bc_groups=1`, `d_state=16`, `d_conv=4`, `temporal_layers=2`.
- Mamba1 initialization, convolution bias enabled, other temporal bias disabled,
  dropout 0, temporal zero-output initialization enabled; preserve all dt initialization settings.
- Explicit `temporal_dt_rank="32"` in the candidate. At width 512 this matches the
  reference's `auto`; leaving auto at width 128 would reduce the rank to 8.
- Training years 2015–2021 and validation year 2022; same prepared store, statistics,
  anchor manifest and ordered segment schedule. Record and compare input fingerprints.
- Segment length 96, BPTT length 24, autoregressive tail 20, `closed_loop_sg`
  feedback, temporal state carry, and `all_steps` loss.
- Global batch size 1, one device; preserve batch size even if width 128 frees memory.
- 10,000 optimizer updates, seed 22; Adam betas (0.9, 0.98), LR 1e-4,
  cosine decay to 1e-5, warmup 200, weight decay 1e-4 on all parameters,
  gradient clipping 1.0. Train the whole residual branch; freeze the baseline.
- bf16 model precision, fp32 weather tape, `explicit_reverse_vjp` backend.
- Checkpoints every 1,000 updates; inline validation disabled as in the reference.

Equal seeds specify the same sampling/initialization policy, not identical random
weights across different tensor shapes. Do not retune learning rate, batch size,
loss, or rollout horizon in the primary comparison.

## Evaluation and decision rule

Prespecify the primary artifact as the equal-weight SWA of checkpoints at
2k, 3k, 4k, 5k, 6k, 7k, and 8k updates, exactly matching A. Complete 10k training
updates and report the 10k checkpoint separately as a convergence diagnostic.
Avoid choosing a different SWA window for the narrow model using these results.

Use the existing v24 evaluator, the reference evaluation data and identical
32 `chosen_idx` values from A's JSON. Match `cold_full`, zero residual state,
residual alpha 1, 40 six-hour forecast steps (10 days), no truth-history warm start,
and a pure baseline self-rollout for the denominator. Preserve the existing
wrapper's cold-mode argument semantics, including its `--warmup-steps 24` argument;
validate effective cold initialization against the saved reference metadata.

Require all 32 samples and all 40 leads, identical anchor semantics, and matching
baseline loss arrays within numerical tolerance before comparing results.

Primary metric: `100 * (1 - full_rollout_loss / baseline_rollout_loss)`, using
`original_graphcast_loss` from the evaluator. Aggregate losses before the ratio.
Report the complete per-lead loss curve as well, especially late-rollout behavior.

Proposed screening criterion: C retains at least 95% of A's loss reduction,
i.e. **at least 15.469%** reduction, with no material late-rollout deterioration.
This is an engineering target, not a statistical equivalence claim. Report C–A
in percentage points regardless of whether that target is met.

If C is promising, consider additional training seeds, and
confirm on additional prespecified held-out anchors. The historical 32-anchor
set was used to select A, so it is a development comparison, not an untouched test.

Record parameter counts by encoder, processor, decoder, temporal modules and
output head; peak GPU memory; compilation time; and median steady-state seconds
per update on the same hardware. Compare timing over matching schedule positions,
excluding checkpoint/SWA/evaluation time, and also report total training time.
The frozen baseline remains a fixed compute and parameter cost.

## Implementation prerequisites

1. Add an optional `residual_width` with backward-compatible fallback to `width`;
   apply it only to the residual model config. Carry it through evaluation CLI,
   checkpoint architecture validation, SWA, resume and run metadata.
2. Add an explicit residual spatial initialization policy (`baseline_overlay` or
   `fresh`), defaulting to current behavior. Fresh skips the complete residual
   overlay, while the baseline still loads all pretrained parameters strictly.
3. Preflight the new candidate: baseline parameter count/hash matches the reference,
   residual overlay count is zero, tensors use the intended spatial width,
   effective dt rank is 32, initial residual output is zero, and training updates
   affect only the residual branch. Check finite loss/gradients and checkpoint
   round-trip evaluation before a full run.
4. Preserve the historical reference artifact. Record the new code revision and
   dirty diff because current training files have local changes. If those changes
   alter training semantics, reproduce arm A under the same code before making a
   causal claim about initialization; the original A remains the practical target.
5. Budget training/evaluation jobs from measured preflight timings and existing
   comparable job history using the slurm-budgeter skill before submission.

Width 128 reduces encoder and decoder hidden dimensions as well as the processor.
Most internal dense weights scale with width squared; input/output projections,
fixed-size output head, and fixed-inner-dimension temporal terms do not all follow
that scaling. Measure exact counts during preflight instead of claiming a uniform
16-fold reduction or a corresponding whole-model speedup.
