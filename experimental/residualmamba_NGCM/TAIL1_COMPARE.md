# Independent ar_tail_k=1 versus ar_tail_k=20 diagnostic

The user requested an additional independent 2,000-update ar_tail_k=1 run on
2026-11-05. The existing ar_tail_k=20 run continues from its original immutable
snapshot. This experiment is not a K=1-to-K=20 curriculum.

## Scientific contrast

| Setting | Existing run | Additional run |
| --- | --- | --- |
| ar_tail_k | 20 | 1 |
| BPTT | 24 | 24 |
| Derived truth prefix | 4 | 23 |
| Independent one-step prefix forecasts | 3 | 22 |
| Contiguous native tail | 21 forecasts, 6–126 h | 2 forecasts, 6–12 h |
| Updates | 2,000 | 2,000 |
| Segment memory | Carry within 96 anchors; detach every 24; reset each segment | Same |

Here ar_tail_k counts continuation steps after the first forecast. It does not
equal the total number of forecast leads. For tail1, native one-step forecasts
start at a0 through a0+21. One native trajectory starts at a0+22 and produces
the last two predictions. Its forcing remains the lagged forcing at a0+22.
At the final step, the residual inputs are truth at a0+22 and the native
forecast valid at a0+23. No future truth enters that window.

All other settings match: seed 22, fresh residual initialization, frozen official
pretrained NeuralGCM, GraphCast mesh 4 / width 512 / two message-passing steps,
the same Mamba architecture, 128 by 64 Gaussian grid, seven fields, correction
on 29 levels above 30 hPa in pressure value (50–1000 hPa), FP32, fixed field
weights 1 for the five main fields and 0.05 for each cloud field. The same
verified all-training-time mean, std and adjacent-6h diff_std are reused.
AdamW uses beta=(0.9,0.98), weight decay 1e-4 and gradient clipping at 1.
The 200-update warmup reaches 1e-4, followed by the original 10k-step cosine
schedule toward 1e-5. The 2k bound does not compress that schedule.

Both arms retain Mamba memory. Their comparison measures sensitivity to the
training forecast horizon, not the causal contribution of memory. A memory
reset ablation would be a separate experiment. Training losses across these
two horizons are not directly comparable because their prediction difficulty
differs.

## Reuse and isolation

The extension does not edit any qualified v1 core module. `Tail1Config` changes
only ar_tail_k and its derived truth prefix, and delegates model and optimizer
construction to that core. Its additional code hashes are pinned separately.
The new plan records hashes of the parent plan, setup, geometry, train-only
statistics, complete baseline cache manifest and qualification evidence.

Tail1 needs a separate baseline cache: replacing most long-lead predictions
with independent one-step predictions changes the origins and inputs. Reading
two entries from a tail20 chunk would not implement this experiment. Two
disjoint H200 cache lanes generate and verify all 484 train/validation chunks.
No data download or full statistics recomputation is required. All new files
and caches are under MENGDIW scratch; the same bounded diagnostic exception
and deferred full production audit documented in DIAGNOSTIC_2K.md apply.

## Smoke tests and execution

CPU tests compare the input windows with an independently implemented
GraphCast iteration, test future-truth exclusion, check native origin and
forcing continuation, assert identical architecture and all other settings,
and exercise the scheduler's stage transitions and fresh-process QA script.
The complete CPU suite passed 38 tests in 14.45 seconds. Its output is saved
in `validation/20261105_tail1/cpu-tests.txt`. GPU evidence is written by the
submitted run under `qa/COMPLETE.json`; CPU success alone is not GPU evidence.

The H200 smoke uses real tail1 data and the full model. It compares continuous
three-update training with a callback-triggered checkpoint at update two and
fresh-process continuation to update three. It records differences separately
for parameters, Mamba memory, Adam state, RNG and cursor. Independent native
evaluator calls verify a prefix origin and the complete 6h/12h tail. Finite
differences remain measured warnings; nonfinite or corrupt artifacts fail.
The completed cache receives additional native checks at three training
anchors and one validation anchor before the main training is released.

Each GPU job uses ailab / gpu-test / one H200 and a one-hour walltime. Cache
lanes skip verified completed chunks after restart. Training saves a complete
checkpoint before its soft walltime and resumes until update 1,000 and then
2,000. The original tail20 job is not cancelled or modified. Slurm enforces
the shared three-job gpu-test limit if cache and QA jobs overlap that run.

## Evaluation

At both 1,000 and 2,000 updates, evaluate all 15 fixed validation segments
twice: once on tail1 inputs and once on the exact existing tail20 validation
cache. The second evaluation is the common comparison with the original arm,
including its native zero-head baseline. Validation resets memory at segment
boundaries and never trains on validation data. Each segment result is saved
to allow one-hour jobs to resume validation without repeating completed work.

The final report includes raw and rolling training losses, a matched-anchor
comparison across training epochs, and both validation objectives. These are
bounded diagnostics, not a claim of full forecast-skill improvement.
