# M1 versus November lambda-zero models at matched training budgets

Date: **2026-11-12 (America/New_York)**. Status: submitted; results pending.

Evaluation array **17404694_2–5** supplies the four missing forecasts; CPU
report **17404732** depends on the array. Four completed forecasts are reused.
The sealed manifest hash is
`2ee6bb4c9e351ea6f03c24e2393bdb76b14079d6d29e1523d99177c9f14aed1f`;
the frozen source digest is
`04e82e3820ecb955e7ecd5a21fed1b91c58a252a6b443a36ef58c7d03b595d3d`.
Four preparation tests and fifteen existing report/reuse tests passed.
Serialized checkpoint progress, family, baseline hash and resolved configuration
were checked before submission; inherited forecast/metric source parity passed.
Commands and job IDs are retained in the campaign's `submissions/` directory.

The user questioned the November full-GC raw update-1,250 result
(exact rollout MSE 10.610624; 4.6934% reduction against public GC), requested
comparison with accessible 1,000- or 1,500-update M1 checkpoints, and asked
whether the setup is the same. The earlier table used M1 after 2,500 total
updates and November models after 1,250. It was a matched forecast evaluation,
but neither a matched training budget nor an identical training recipe.

## Checkpoint mapping and evaluation

Historical M1's main-stage checkpoint counter excludes its initial 500 updates.
These counts are optimizer updates, not forecast leads.

| Total training updates | M1 main-stage checkpoint | November GC–Mamba lambda 0 | November full-GC lambda 0 |
| ---: | ---: | ---: | ---: |
| 1,000 | 500 | 1,000 | 1,000 |
| 1,250 | 750 | 1,250 | 1,250 |
| 1,500 | 1,000 | Not yet available | Not yet available |

Evaluate the three raw checkpoints at total update 1,000 and M1 at total update
1,250. Reuse the two completed November update-1,250 forecasts and the final
M1 raw/SWA references. This requires four new model evaluations. Intermediate
weights are unaveraged; the final M1 SWA is a separately labeled reference.
There is no checkpoint selection based on the resulting scores.

Use the completed [pilot](v24_raw_checkpoint_pilot_2026-11-10.md)'s frozen
computational source, baseline, normalization and exact same four initialization
dates: 2022-01-02 00Z, 2022-04-03 06Z, 2022-07-07 18Z and 2022-10-07 06Z.
Each forecast has 40 six-hour leads, cold full feedback, zero recurrent state,
no forecast warm-up and no smoothing. Report exact original GraphCast weighted
grid MSE, lead curves, spherical power spectra and forecast/ERA5 ratios, and
the same 83-channel, degrees 1–179 aggregate power mismatch as the parent pilot.
All reductions use one common public-GC loss denominator.

Campaign: `artifacts/evaluations/v24_matched_budget_comparison_20261112/`.
Outputs will use the corresponding campaign directory under
`plots/analyze_models/data/resolution_eval/` and `images/resolution_eval/`.
The four-date comparison is a development diagnostic, not annual skill or
training-seed robustness. Matching updates does not match GPU cost or isolate
architecture from training choices.

## What matches, and what differs

The public GraphCast-small 1-degree baseline checkpoint and normalization match.
The saved evaluation records confirm the common forecast protocol. Inspection
of the frozen full-GC loader confirms that its forecast uses the fine-tuned
parameter tree, with the original public weights loaded separately for the
reference. The checkpoint changed 256 of 262 parameter leaves; it is not a
duplicate of public GC. Embedded architecture metadata matches the pinned
configuration. No loading or rollout-mode mismatch was found in this audit.
The base training-anchor manifest matches; November binds an explicit frozen
order manifest, whereas M1 used the historical sampler. Both use training years
2015–2021, batch four, 24-step windows, K20 and 96-step segments. M1 and November
GC–Mamba have the same correction architecture and frozen backbone. Full GC
instead trains all backbone parameters and has no Mamba correction.

| Training choice | Historical M1 | November GC–Mamba lambda 0 | November full-GC lambda 0 |
| --- | --- | --- | --- |
| Trainable parameters | 10,348,879 correction parameters | Same correction parameters | 35,979,347 GC parameters |
| Weather-feedback gradients | Stopped (`closed_loop_sg`) | Full (`closed_loop`) | Full (`closed_loop`) |
| Schedule | 500-update initial stage, then 2,000 updates with optimizer reset | One 2,500-update schedule | One 2,500-update schedule |
| Main spatial/backbone peak LR | `1e-4` | `3e-5` | `3e-7` |
| Mamba LR multiplier | 1 during initial stage, 3 during main stage | 3 | Not applicable |
| Loss arithmetic | Normalized residual GraphCast MSE in BF16 model wrapper | Full-field FP32 grid MSE | Full-field FP32 grid MSE |

The loss weights represent the same GraphCast MSE mathematically, but arithmetic
and differentiated paths differ. Lambda zero has no spectral term: all 1,250
November GC–Mamba records have zero weighted spectral excess, no computed
spectral diagnostics, and total loss equal to FP32 grid MSE. M1's initial stage
already used K20; it was not a shorter-horizon curriculum.

M1 main-stage pre-clipping gradient norms had median 0.333 and maximum 0.924;
November GC–Mamba lambda zero had median 19.948 and range 11.794–52.770. With
clip threshold 1, all 1,250 November updates were clipped versus none in M1's
main stage. This is evidence of a different optimization regime, not an isolated
causal explanation or an execution failure. Full-GC learning-rate retuning also
changed the comparator relative to earlier high-rate calibration attempts.

Configuration sources: [M1](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/run_config.json),
[November GC–Mamba](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/runs/gcm_s22_l000_lr0/run_config.json),
[November full GC](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/runs/gcft_s22_l000_lr3em7_fresh_20261107/run_config.json).

## Existing evidence at 500 total updates

Saved exact-loss evaluations already compare all three raw models at 500 total
updates on the same 32 physical 2022 starts and 40 leads, cold full feedback and
zero state. The November development manifest derives its dates from M1's
original sample indices. Numeric indices differ by three because the evaluation
data window starts at a different offset; the physical dates match.

| Raw model, 500 total updates | Exact rollout MSE | Paired public-GC MSE | Reduction using common GC denominator 10.844687 |
| --- | ---: | ---: | ---: |
| M1 initial-stage endpoint | 9.819859 | 10.844687 | 9.4500% |
| November GC–Mamba lambda 0 | 10.314396 | 10.819500 | 4.8899% |
| November full-GC lambda 0 | 10.337334 | 10.840716 | 4.6784% |

The common denominator is M1's saved public-GC evaluation. The maximum baseline
repeat difference here is 0.0251871; both model gaps to M1 exceed that measured
baseline difference. This does not estimate model repeatability or forecast
sampling uncertainty. Original paired reductions remain in the JSONs (9.4500%,
4.6685%, 4.6434%). These older runs used their respective evaluator versions;
the new comparison fixes one frozen runtime. Intermediate spherical powers do
not exist for this 32-start evidence. Do not combine its absolute MSE values
with the four-start pilot's table.

The earlier gap therefore cannot be attributed solely to unequal update counts.
No existing comparison isolates which training change caused it.

Sources: [M1 initial-stage evaluation](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/shared_start.json),
[GC–Mamba update 500](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/calibration_evaluation/gcm_s22_l000_lr0/evaluation.json),
[full-GC update 500](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/calibration_evaluation/gcft_s22_l000_lr3em7_fresh_20261107/evaluation.json),
[date mapping](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/manifests/development_2022_32.json).

## Resources and existing work

Completed pilot arrays 17249273 and 17250938 took 14m44s–15m46s per model, with
maximum host RSS 8.75 GiB. Reuse one 80GB GPU, four CPUs, 16 GiB RAM and 62 minutes
per new evaluation; 62 minutes avoids the documented one-hour routing boundary.
Expected execution is approximately one GPU-hour across the four tasks,
excluding queue time. The CPU report needs two CPUs, 4 GiB and 15 minutes.

The existing final 2,500-update raw/SWA chain remains pending its training
dependencies. This comparison evaluates available weights only. The separate
four-fresh-horizon training hold remains in place. Finite worsening, numerical
drift and clipping remain recorded warnings; nonfinite computation and
source/input/checkpoint integrity failures remain errors.
