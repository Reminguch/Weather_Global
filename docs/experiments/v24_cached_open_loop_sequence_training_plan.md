# Efficient cached open-loop residual GC + Mamba training

Date: 2026-09-14. Status: deferred sequence-batching proposal. The approved first
experiment is [cached stepwise training](v24_cached_stepwise_training.md), which
retains the existing residual/Mamba execution and numerical policy. The broader
sequence-batching changes below are not part of that implementation.

## 1. Objective and first target

Train a small residual GraphCast + Mamba model on fixed, precomputed GraphCast trajectories. At each forecast step the model predicts a correction to the cached GC field. Only the baseline GC prediction supplies subsequent weather inputs. Residual predictions never feed back into this training trajectory.

The first complete implementation should:

- Consume the existing res1 cache without regenerating it.
- Train complete 24-step sequences with a loss at every step and gradients through Mamba throughout the chunk.
- Carry recurrent values across the four chunks of a 96-step segment, detaching gradients at optimizer boundaries; reset memory at a new segment.
- Batch spatial work across time, process Mamba sequences at each graph depth, and use the fastest validated scan backend for the actual tensor shapes.
- Keep data movement, activation storage, and compilation bounded.
- Support exact resume, cached validation, and streaming open-loop evaluation with the same numerical semantics.

Recommended first learning architecture: residual width 128, two spatial message-passing steps, Mamba1 inner width 16 / B-C groups 1 / state size 16 / convolution width 4 / dt rank 32 / two Mamba layers per temporal insertion. Use fresh spatial initialization, the existing zero residual head and zero temporal output projections, and dropout 0, matching the compact architecture already represented in the repository. The frozen baseline retains its original width 512 and 16 message-passing steps.

Keep a width-512 / di64 / BCG2 configuration for comparison with the completed online reference. Backend comparisons must hold architecture fixed. Changing width is a separate experiment.

## 2. What exists and what is missing

| Component | Present state | Required work |
|---|---|---|
| Cache producer and verification | Complete production cache, pilot parity report, READY marker | Reuse and verify provenance |
| Cache reader | Memory-mapped GC fields; reconstructs windows and residual targets | Owned, batched sequence tensors; bounded prefetch |
| Residual model | Maintained v24 architecture and normalization | Sequence entry point with stable parameter names |
| Interleaved graph/Mamba execution | One weather step per GraphCast call | Execute each graph depth across a complete sequence |
| Mamba sequence block | Accepts a time axis | Efficient causal convolution, explicit FP32 state, scan backend selection |
| Temporal recurrence | Sequential `jax.lax.scan` | Reference plus associative-scan implementation; profile before fused-kernel work |
| Optimizer/checkpoint/validation utilities | Online BPTT runner | Cached runner and versioned execution metadata |
| Evaluation | Baseline- and corrected-feedback rollouts exist | Cached metrics and a streaming predictor matching the new state/loss contract |

Relevant existing files:

- `src/models/mamba/v24_Ilya/baseline_cache.py`
- `src/models/mamba/v24_Ilya/model.py`
- `src/models/mamba/modules/temporal_mesh_mamba_Ilya.py`
- `third_party/graphcast/graphcast/graphcast.py`
- `src/models/mamba/v24_Ilya/training/{config,data,frame_data,endpoint_step,runner,validation}.py`
- `src/models/mamba/v24_Ilya/{checkpoint,metrics,evaluation,rollout}.py`
- `scripts/preprocessing/BASELINE_CACHE.md`

## 3. Data and objective contract

### Existing cache

Root: `data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20`.

| Property | Value |
|---|---|
| Training coverage | 2015–2021; 106 segments; 424 chunks |
| Validation coverage | 2022; 15 segments; 60 chunks |
| Total | 484 chunks; 11,616 predictions; eight shards |
| Prediction storage | FP32; 234.03 GiB total payload |
| Baseline neural compute | BF16, deterministic GPU reductions recorded in provenance |
| Chunk schedule | 24 predictions; observed windows at indices 0–3; baseline feedback at indices 4–23 |
| Longest uninterrupted forecast | 21 six-hour predictions from the last observed window: 126 hours |
| Physical restart | At every cache chunk |
| Saved Mamba features/state | None |

The full 96-step segment is not one 24-day GC forecast. It consists of four physically restarted 24-step chunks. The recurrent carry policy across those restarts must be explicit.

At cached step t, let X_t be its reconstructed two-frame weather window, g_t the cached GC prediction, y_t the ERA5 target, and f_t the known forcing. Train:

    (r_t, state_t) = ResidualModel(parameters, X_t, f_t, state_{t-1})
    target_t       = y_t - g_t
    corrected_t    = g_t + r_t

Use the exact original GC weighted squared-error objective, including difference-standard-deviation scaling, original variable weights, pressure weights, and exact latitude weights. Start with equal weights on all 24 steps. Reduce over valid examples and steps using sums and counts; padded lanes must have zero weight.

Separate three settings in the new configuration:

1. Cache trajectory schedule: read from immutable cache metadata.
2. Loss-step selection: initially all 24 steps.
3. Temporal gradient window: initially 24 steps.

Spatial microbatch size and Mamba node-tile size are execution settings. Neither changes the loss or detaches temporal gradients. Later loss masks or shorter gradient windows must not masquerade as a change to the cache's `ar_tail_k`.

## 4. Phase A — cache contract, loader, and serial reference

### A1. Preserve cache provenance

The current cache fingerprint includes whole source files, including `model.py`, `endpoint_step.py`, `training/data.py`, `frame_data.py`, and the vendored `graphcast.py`. Regenerating a manifest after residual-only edits to these files would reject the existing cache.

Implement cached functionality primarily in new modules, keeping hashed producer files intact initially. Separate:

- Producer identity: saved manifest hash, generation source hashes, baseline/checkpoint/statistics identity, physical dtype and baseline precision, source selection, coordinates, and rollout schedule.
- Consumer identity: cached training config, residual architecture, state/loss precision, source revision, scan backend, sampler and optimizer settings.

Validate the saved producer manifest and READY marker, and compare actual source-data/checkpoint/statistics identities with the recorded identities. Consumer code changes must not require rebuilding GC trajectories. If a shared hashed file must eventually change, use a documented compatibility adapter tied to the old producer fingerprint and a baseline-equivalence check. Never overwrite the old manifest or unconditionally disable compatibility validation.

### A2. Materialize safe sequence batches

Add a cached data adapter with explicit named fields:

    CachedSequenceBatch:
      dynamic_inputs     [B, T, 2, ...]
      static_inputs      [spatial dimensions, ...]
      forcings           [B, T, ...]
      residual_targets   [B, T, ...]
      valid_step_mask    [B, T]
      chunk_ids, segment_ids, timestamps, lane_reset_mask

Use typed variable arrays plus stable channel/coordinate metadata. Preserve the original feature ordering and forcing timestamps. Keep the forecast sequence axis separate from the two-frame input axis. Provide optional baseline/truth access through a validation-only streaming view for exact physical-field metrics; training does not need duplicate truth and target tapes.

- Prototype with the existing reader as the reconstruction oracle.
- Replace hot-path per-frame xarray work with contiguous array packing once reconstruction tests pass.
- Allocate owned buffers or independent workspace slots for every lane and prefetch slot. Existing truth buffers are reused: collecting several generators and then loading another chunk can overwrite earlier data.
- Keep cache memmaps read-only. Compute targets into owned FP32 buffers; do not retain duplicate full truth and residual-target tapes during training.
- Bound prefetch to one next batch initially. Release a slot only after its device transfer completes.
- Store static fields/coordinates once where possible; avoid needless host copies across time.
- For later B>1, assign separate segments to lanes, preserving chunk order within each lane. Pad final incomplete groups with masks rather than silently losing training coverage.

Start with deterministic manifest order. Optional later shuffling permutes whole segments, with seed/permutation and lane assignment recorded for resume. Never independently shuffle steps in the stateful experiment.

### A3. Cached serial reference

Build a residual-only per-step trainer consuming the new adapter. It runs no baseline forward pass and does not keep baseline parameters on the GPU. Loading baseline weights on the host for compatible residual initialization is allowed.

First reproduce the maintained online numerical policy on the representative first/last training and validation chunks. Compare reconstructed inputs, residual targets, per-step predictions, loss, gradients, final SSM/conv state, and one optimizer update. Keep this reference as a diagnostic backend after sequence training is implemented.

Deliverable: an executable cached trainer that is correct before sequence optimization, plus a source/manifest compatibility report.

## 5. Phase B — explicit precision and state semantics

There are two existing numerical differences that must be handled deliberately:

- A BF16 one-step Mamba call rounds its recurrent state after every weather step. The current full-sequence SSM scan keeps FP32 state internally until sequence end. A parallel affine scan cannot exactly reproduce intermediate BF16 rounding.
- The existing BF16 loss wrapper also casts normalized targets and loss computation. Computing the exact weighted loss in FP32 is a separate numerical change.

Define a versioned efficient-execution contract:

| Quantity | Proposed policy |
|---|---|
| Cached weather, input normalization, forcings | FP32 boundaries and normalization |
| Spatial and Mamba projections | BF16 neural activations |
| SSM recurrence and persistent SSM state | FP32 at every read, update, return, and checkpoint |
| Convolution history | Same projected-activation values/dtype in serial and sequence paths; record this policy |
| Physical residual prediction and GC addition | FP32 |
| Residual targets, loss weights, loss reductions | FP32 |
| Parameters, optimizer, accumulation | FP32 |

Implement a serial reference under this new contract. Sequence, associative, and any fused backend must match this reference within declared floating-point tolerances. Compare the legacy serial contract separately; do not claim bitwise equivalence between old BF16 state rounding and the new FP32 recurrence.

Explicit state tensors are preferable around the temporal kernels, because the existing BF16 Haiku variable view can silently downcast state reads. Ensure the terminal SSM state is never converted back to BF16 before the next call.

The streaming evaluation predictor must implement the same contract. A new cached checkpoint must not be accepted silently by an evaluator that reinstates legacy per-step rounding. Introduce a distinct cached checkpoint format/execution version and a dedicated loader; preserve residual parameter names so weight initialization and comparisons remain possible.

## 6. Phase C — sequence execution through the residual model

The main reordering is across graph depth:

```text
Reconstructed cached weather windows [B, T, 2, ...]
                     |
      normalize and encode spatially over B*T
                     |
      graph processor stage 0 over B*T
                     |
      Mamba sequence 0 over [T, mesh, B, width]
                     |
      graph processor stage 1 over B*T
                     |
      Mamba sequence 1 over [T, mesh, B, width]
                     |
         decode spatially over B*T
                     |
       FP32 per-step residual loss, averaged
```

Each temporal insertion retains the configured internal Mamba layers. Preserve node and edge activations for every time step through the graph stages. Mamba updates node features; it must not reset edge features or rebuild later stages from the original embedded edges.

Create a sequence-specific residual model adapter using the maintained graph primitives. Do not treat T as an enlarged batch while leaving the existing Mamba wrapper unchanged: that would create independent streams instead of temporal memory.

Preserve existing parameter keys and shapes, especially the graph scopes, `mesh_interleaved_temporal_r{r}_s{s}`, `layer_norm_{i}`, `mamba_block_{i}`, and `temporal_residual_head`. Verify direct loading of a known residual parameter tree. Additional Haiku wrapper scopes must not silently rename parameters.

Keep a single JIT-compiled forward/loss/backward update for each bounded shape. Avoid the existing host-driven per-step forward and reverse loop in the efficient backend. Return compact loss statistics and the final state; do not copy every intermediate prediction/state back to the host.

### Memory controls that preserve the objective

- Spatial microbatches: process q=1,2,4,8 independent flattened B*T frames at a time, with rematerialization around expensive graph stages. Retain complete mesh sequences between temporal insertions.
- Temporal node tiles: each tile covers a subset of independent mesh-node/sample streams and contains all T steps. Try 256, 512, and 1024 streams after measuring.
- Execute memory tiles with a bounded loop/map and rematerialization; applying `vmap` to every tile simultaneously can restore the original peak working set.
- Apply node tiling only to Mamba; graph spatial updates require the full graph or an explicitly correct distributed graph algorithm.
- Recompute encoder grid skip features for decoder microbatches if storing every grid skip becomes expensive. Accumulate decoder losses as weighted sums/counts.
- Preserve full 24-step gradients through spatial microbatches, scan tiles, and recomputation. Microbatch boundaries are not BPTT boundaries.

For scale: one BF16 grid-latent tensor at res1 (181×360), width 512, B=1,T=24 is about 1.49 GiB, before other activations. The small parameter count alone does not determine activation memory. Use [JAX rematerialization](https://docs.jax.dev/en/latest/gradient-checkpointing.html) to trade recomputation for retained activations.

## 7. Phase D — optimize Mamba for the measured shapes

### D1. Batched causal convolution

Replace the Python loop over time with a grouped causal convolution over each projected sequence. Prepend the incoming convolution history, reproduce the existing kernel orientation, and return the final d_conv-1 projected inputs as the next history. Test d_conv=1 and d_conv=4, including nonzero incoming history and chunk boundaries.

### D2. Scan backends

Expose a small backend interface with identical inputs, outputs, and derivatives:

1. `serial_scan`: FP32 `lax.scan`, serving as the semantic reference.
2. `associative_scan`: parallel affine recurrence using `jax.lax.associative_scan`.
3. `fused_scan`: optional later implementation after profiling and compatibility checks.

For h_t = a_t * h_{t-1} + b_t, compose a left interval followed by a right interval as:

    compose((a_left, b_left), (a_right, b_right)) =
      (a_right * a_left, a_right * b_left + b_right)

Use the prefix transforms to apply the nonzero incoming state. Preserve input-dependent delta/B/C, grouped B/C broadcasting, D skip, gating, normalization, and final-state gradients. The operation is associative in real arithmetic; FP32 operation ordering still requires tolerance-based checks. [JAX documents this parallel primitive](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html).

Do not assume the associative version is fastest: T=24 is short and there are many independent mesh streams. Benchmark both complete forward/backward kernels and complete optimizer updates. Chunk scan work by independent streams to avoid materializing every expanded [B,T,mesh,d_inner,d_state] coefficient/state array at once.

Mamba's efficient algorithm combines scan, fusion, and recomputation; an associative scan alone does not reproduce its memory efficiency. Use the [original selective-scan design](https://arxiv.org/html/2312.00752v2#S3.SS3) as the basis for a fused backend only if profiling shows that scan/expansion traffic is a material remaining bottleneck.

Environment inspected for this plan: JAX/JAXLIB 0.9.0.1, Haiku 0.0.16, Optax 0.2.7; no standalone Triton, jax-triton, or mamba-ssm package was observed by the audit. Reconfirm versions and an A100-compatible JAX GPU integration before choosing a custom kernel. Keep the first functioning implementation in the existing environment; plan any kernel dependency work separately.

Deliverable: a measured backend choice for T=24, with larger lengths treated as later benchmarks rather than assumed production requirements.

## 8. Phase E — runner, batching, checkpoint, validation

### Runner

Use a dedicated cached entry point and config to avoid overloading the existing `ar_tail_k`/BPTT contract. Reuse optimizer schedules, parameter-group LR logic, zero-head initialization, compatible overlay helpers, atomic writes, and exact metric code.

Each update:

1. Read the next manifest chunk for each active lane.
2. Reset both SSM and convolution state for new segments as required.
3. Transfer the bounded owned batch; execute one sequence loss/gradient/update.
4. Carry final state values into the next chunk, with no gradient through the previous optimizer update.
5. Advance the cursor to the next chunk and record true examples/steps consumed.
6. Checkpoint or validate at configured update boundaries.

Start with B=1 on one GPU. After correctness and memory checks, tune B=2 and B=4. Later multi-GPU data parallelism partitions independent segments, keeps per-lane state, and weights gradient sums by valid counts. It is not necessary for the first efficient implementation.

### Checkpoint and resume

Save residual parameters, optimizer/schedule state, RNG, per-lane SSM/conv state, completed update count, next-chunk cursor, segment order/permutation, lane assignment, sampler seed, masks, cache identity, numerical contract, execution backend, and resolved configuration.

Checkpoint at completed optimizer boundaries only. Asynchronous prefetch is disposable on restart and must not advance the committed cursor. Test continuation across both a chunk boundary and a segment reset. Validation must use separate state and RNG and must never modify training state or cursor.

Use a cached-specific checkpoint format and loader. Provide explicit parameter export/init-from support where parameter trees match; do not call an optimizer restart from a different numerical contract an exact resume. Later SWA support must validate matching parameter/config contracts and evaluate from the requested reset/warmup policy.

### Validation

Provide two complementary evaluations:

- Cached validation: all 15 held-out 2022 segments, following the same four-chunk carry/reset schedule. Periodic development checks may use a fixed declared subset; final metrics use all 60 chunks. Report all-step and AR-lead metrics separately; the first three observed-window predictions are not consecutive leads of the final rollout.
- Streaming baseline-feedback validation: run GC sequentially from declared observed starts and apply the new residual predictor under its FP32-state contract. Reuse `run_v24_Ilya_rollout` with `full_feedback=False` and the exact metric accumulator. This checks deployment behavior and sequence/streaming agreement.

Use baseline-feedback semantics (`cold_bp`/`warm_bp`) with explicit state initialization; existing `cold_full` results use a different feedback policy. For a strictly matched reconstruction check, reproduce the cache's observed prefix and segment-state policy; a cold forecast from zero memory is a distinct evaluation protocol.

Compute improvement as 100*(1 - aggregated_corrected_GC_loss / aggregated_baseline_GC_loss). Do not average per-lead improvement percentages or substitute equal-variable RMSE. Accumulate metrics without retaining spatial-bias maps unless requested.

A 10-day evaluation is useful but extends past the 126-hour training horizon. Treat it as extrapolation. If training at 10-day leads is desired, generate a new, independently versioned longer-horizon GC cache later.

## 9. Correctness gates

| Gate | Required evidence |
|---|---|
| Data | All selected windows, targets, forcings, coordinates and timestamps match existing reader; no split leakage or source-buffer mutation |
| Cache contract | Missing/partial/mismatched cache fails clearly; consumer changes do not rewrite producer provenance |
| Parameter compatibility | Known parameter tree loads with identical keys/shapes; zero head gives baseline-only output |
| Legacy serial audit | Cached per-step execution matches live-baseline execution under the same legacy numerical policy |
| Baseline elimination | Cached initialization and updates succeed when baseline model `init` and `apply` are replaced by failure sentinels |
| Efficient numerical contract | Serial, sequence, tiled, and parallel implementations agree under FP32 recurrence/loss policy |
| Gradient validity | Compare every parameter-group gradient and one optimizer update; late-only loss reaches earlier temporal inputs/state |
| Causality | Changing a future input cannot change earlier outputs; changing target-only tensors with reconstructed observed inputs held fixed cannot change predictions |
| Stream independence | Batched lanes and Mamba node tiles do not mix state; preserve graph spatial coupling |
| State boundaries | Nonzero SSM/conv carry, physical chunk restarts, new-segment resets, and masked lanes behave as specified |
| Resume | Interrupted/restarted and uninterrupted runs consume the same data and match params, state, optimizer and metrics on the same backend |
| Evaluation | Cached and matched streaming outputs/metrics agree; numerical contract is honored by checkpoint loading |
| Memory/runtime | Stable repeated-update memory, bounded prefetch, no shape-driven recompilation in steady state |

Tests must include nonzero residual heads and nonzero temporal output projections, or a suitable trained checkpoint. The default zero heads can make upstream gradients zero and hide temporal implementation errors. Include nonzero initial memory, grouped B/C, convolution widths 1/4, multiple temporal layers, B=1/2, T=1/24, and a non-power-of-two scan case.

Use FP32 reference comparisons with initial atol=1e-5/rtol=1e-4 on suitably scaled toy/normalized outputs, loss, state, and gradients; record max absolute and relative errors. These are proposed engineering tolerances, not measured promises for full weather arrays. Inspect full-resolution reduction and BF16 errors before setting the production tolerance. Keep exact checks for identities, indices, masks, parameter keys, source immutability, and same-backend deterministic resume. Never relax a failing gate merely to make an optimization pass.

## 10. Benchmark and first learning experiment

### Backend benchmark matrix

| Backend | Purpose |
|---|---|
| Online legacy serial | Historical reference, including live baseline computation |
| Cached legacy serial | Isolate removal of live GC inference |
| Cached FP32-state/loss serial | Reference for the proposed numerical contract |
| Cached sequence, serial scan | Isolate time batching and fewer host launches |
| Cached sequence, associative scan | Isolate parallel recurrence |
| Cached sequence, fused scan | Only if justified by the profile |

Match parameters, architecture, batch, T, incoming state, data, objective and precision within each execution-speed comparison. Explicitly separate the legacy-to-new precision change. Run width-128 and width-512 comparisons separately.

Compile separately, then measure three blocks of about 20 steady updates per viable backend. Synchronize device work before timing boundaries. Collect:

- End-to-end seconds/update and valid weather steps/second, including loading and transfer.
- Compute-only median/p90 and startup/compile time.
- Cache read/packing, transfer, spatial forward/backward, Mamba forward/backward, optimizer and validation times.
- Host RSS and GPU allocation/peak, separately.
- First traversal and repeated traversal results, labelled accurately rather than claiming a cold filesystem cache.

Existing online logs start the update timer after `build_chunk`, so their `step_seconds` are not end-to-end throughput.

For the first learning test, use a fixed compact architecture and identical initialization/data order/optimizer settings for:

1. Stateful Mamba with full 24-step gradients and segment carry.
2. The same model with both SSM and convolution history reset at every step.

Do not implement the second condition by calling today's `apply_stateless` once on the full sequence; that still creates temporal context inside the sequence. The reset version must be genuinely per-step and keep architecture/parameter count unchanged.

Run an engineering smoke covering at least one full segment plus resume, followed by a 500-update learning pilot. A provisional optimizer recipe is AdamW peak LR 1e-4, betas (0.9,0.98), weight decay 1e-4, clipping 1, 200-update warmup, cosine decay to 1e-5 over the intended 10k-update schedule, seed 22, equal initial spatial/Mamba LR multipliers. Treat 500 updates as a checkpoint of that schedule rather than compressing the whole schedule into the pilot. Continue to the longer comparison after correctness, stable memory and learning behavior are established.

Compare quality at equal training examples and also at equal GPU-hours. The reset condition's lower cost is an architectural ablation effect, not a scan-backend speedup. If temporal memory adds no held-out benefit, retain the useful cached trainer and investigate that result before expanding Mamba sweeps.

## 11. Resource plan and measured reference

Completed online reference: job 13560724, width512 / di64 / BCG2 / B=1 / T=24, 10,000 updates.

- Elapsed: 30h42m29s.
- First logged update: 696.84 seconds, including first compilation/update work.
- Last 100 logged updates: median 10.322 seconds, p90 10.378 seconds, excluding data loading.
- Slurm batch MaxRSS: approximately 82.20 GiB; original request 288 GiB.
- Sampled GPU peak: 8,285 MiB on A100 80GB PCIe; sampling may miss short peaks.
- Source: `logs/v24_r1_open_di64_bcg2_13560724.out`, its `train_metrics.jsonl`, and read-only Slurm accounting inspected for this plan.

The existing width-128 run reports 730,447 residual parameters and about 6.428 seconds/update late in training, but uses corrected feedback. It is architecture/sizing context, not a matched open-loop speed baseline.

Provisional first full-resolution smoke allocation: one A100 80GB, eight CPUs, 128 GiB host RAM, two hours. This is based on the online reference plus memory/compile headroom; it is not evidence that arbitrary full-time batching fits. Keep each smoke to a bounded set of backends/shapes and split larger profiling work into later jobs.

After the smoke, set CPU memory to approximately 1.25–1.6 times measured peak. Estimate walltime from startup + p90 update time*updates + validation/checkpoint overhead, with 1.2–1.5 times margin. Use measured GPU memory to select spatial and Mamba tile sizes; `--mem` only controls host RAM. Bound prefetch and never load the complete 234 GiB cache into host memory.

All Python/tests/analysis run through `source scripts/graphcast_env.sh`. GPU work belongs in a Slurm allocation. The plan does not submit or modify jobs. Future experiment launchers should record configs, code snapshots, dependencies, accounting and GPU telemetry, using unthrottled arrays unless a concurrency limit is requested.

## 12. Proposed files and implementation order

The following are proposed additions, not files claimed to exist:

| File/module | Responsibility |
|---|---|
| `v24_Ilya/training/cached_config.py` | Explicit cache, objective, gradient, precision, batching and backend contract |
| `v24_Ilya/training/cached_data.py` | Provenance adapter, owned batches, lane sampler and prefetch |
| `v24_Ilya/cached_model.py` | Sequence/streaming residual adapters, stable parameter scopes |
| `mamba/modules/temporal_mesh_mamba_sequence.py` | Explicit FP32-state block, batched conv, serial/associative scan |
| `v24_Ilya/training/cached_step.py` | Serial reference and JIT sequence loss/gradient/update |
| `v24_Ilya/training/cached_checkpoint.py` | Versioned checkpoint, numerical contract, exact cursor/state resume |
| `v24_Ilya/training/cached_runner.py` | Training loop, logging, optimizer integration and periodic validation |
| `v24_Ilya/training/cached_validation.py` | Cached metric reduction and matched streaming checks |
| `scripts/training/train_v24_cached_open_loop.py` | Dedicated CLI, dry run, resume and benchmark modes |
| `scripts/analyze_models/eval_v24_cached_open_loop.py` | Contract-aware streaming/cached evaluation |
| `scripts/experiments/benchmark_v24_cached_open_loop.slurm` | Bounded parity and throughput jobs |
| `configs/experiments/v24_Ilya/cached_open_loop/` | Compact learning pair and matched reference configs |
| `tests/v24_Ilya/test_cached_{data,sequence,training,checkpoint,evaluation}.py` | Behavioral acceptance tests above |

Paths abbreviated with `v24_Ilya/` and `mamba/modules/` are under `src/models/mamba/`. Shared helpers should be reused where safe; additions to shared APIs should be minimal and accompanied by compatibility checks.

Suggested reviewable milestones:

1. Data/config plus cached serial reference and provenance tests.
2. Explicit numerical contract plus serial/streaming temporal implementation and checkpoint contract.
3. Sequence graph execution, bounded spatial work, full-gradient tests.
4. Batched convolution, associative scan, tiling and benchmark harness.
5. Complete runner, exact resume, cached/streaming evaluation, documentation.
6. Measured compact-model pilot and matched memory ablation; production resource sizing.
7. Fused kernel or multi-GPU work only if the completed profile justifies it.

The first five milestones produce an end-to-end usable pipeline. The sixth establishes whether it is faster and scientifically useful. The seventh is optional further optimization.
