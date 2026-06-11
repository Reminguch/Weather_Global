# Res1 BPTT Memory Optimization Plan

This note summarizes the expected memory pressure for res1 official GraphCast
Mamba runs and the optimization path for fitting the useful variants on a
single GPU.

The main conclusion is that res1 BPTT16 is not primarily limited by model
parameters or by the raw grid latent tensor. It is mostly limited by retained
GNN activations: mesh processor edge/message activations, mesh2grid
edge/message activations, grid2mesh activations, MLP hidden states, layernorm
intermediates, and BPTT boundary tensors. Mamba state can also become large
when the temporal dimensions are high.

## Baseline Assumptions

The estimates below use the current res1 official setup:

| Setting | Value |
| --- | --- |
| Resolution | `1.0` |
| Mesh size | `5` |
| Latent width | `512` |
| Batch size | `1` |
| Precision | `bf16` |
| BPTT steps | `16` |
| Official GraphCast processor | `mp=16` |
| Residual Mamba target processor | `mp=2` |
| Temporal insertion count | `2` |

Approximate graph sizes at res1, mesh5:

| Graph item | Count |
| --- | ---: |
| Grid nodes | `65,160` |
| Mesh nodes | `10,242` |
| Grid2mesh edges | `101,892` |
| Mesh2grid edges | `195,480` |
| Mesh edges | `81,900` |

These are lower-bound tensor sizes. They count a single bf16 tensor of the
given shape, not every hidden activation that XLA may retain for backward.
Actual live memory is larger because each MLP and graph update can keep
multiple intermediates.

| Tensor class | Per step | Across BPTT16 |
| --- | ---: | ---: |
| Grid latent / `latent_grid_nodes` | about `64 MiB` | about `1.0 GiB` |
| Mesh latent | about `10 MiB` | about `160 MiB` |
| Grid2mesh edge latent/message | about `100 MiB` | about `1.55 GiB` per retained edge tensor |
| Mesh2grid edge latent/message | about `191 MiB` | about `3.0 GiB` per retained edge tensor |
| Mesh processor edge tensor | about `80 MiB/layer` | about `1.25 GiB` per retained tensor per layer |
| Output grid tensor | about `10 MiB` | about `165 MiB` |

Mamba recurrent state lower bounds for two temporal insertions:

| Temporal dims | Across BPTT16 |
| --- | ---: |
| `d_inner=128`, `d_state=64` | about `5-6 GiB` |
| `d_inner=256`, `d_state=128` | about `20-21 GiB` |

The measured res2 BPTT16 run reserved about `40 GiB` on GPU even though the
simple tensor categories add to much less. This is expected: `nvidia-smi`
observes allocator reservation, and XLA may retain several hidden tensors per
GNN layer, per BPTT step, plus workspace and compiled buffers.

## Expected Memory By Model Family

### `gc_mamba`

`gc_mamba` keeps the official GraphCast path in the training rollout and adds
Mamba modules inside that path. With the official res1 model, the processor has
`mp=16`, so mesh edge activations are multiplied by sixteen processor steps.

The raw lower bound already shows the danger:

- Mesh processor edge tensor: about `1.25 GiB` per retained tensor class per
  layer across BPTT16.
- With `mp=16`, one retained mesh-edge activation class is already about
  `20 GiB`.
- Real processor layers retain more than one class of activation, so the
  practical cost can be much larger.
- Mesh2grid adds about `3 GiB` per retained edge tensor class.
- Grid2mesh adds about `1.55 GiB` per retained edge tensor class.
- Large Mamba state can add another `5-21 GiB`.

Practical estimate:

- `gc_mamba` at res1, official `mp=16`, BPTT16 is unlikely to fit on one GPU
  without rematerialization and autograd-path pruning.
- The `d_inner=256`, `d_state=128` variant is especially risky because Mamba
  state alone is around `20-21 GiB` before GNN activations.
- The `d_inner=128`, `d_state=64` variant is more plausible, but still depends
  on aggressive rematerialization of the processor and mesh2grid path.

### `res_mamba` With Precomputed Baselines

The target residual setup is different. The official GraphCast baseline should
be run outside the differentiated training step, and the training batch should
provide precomputed baseline predictions or residual targets. Those loaded
baseline fields should be treated as constants, with gradients stopped.

In that form, `res_mamba` does not need to retain official GraphCast `mp=16`
activations in the loss graph. The trainable residual model can use a smaller
processor, for example `mp=2`.

Practical estimate:

- The `mp=2` residual model removes most of the official processor activation
  burden.
- Grid2mesh and mesh2grid remain important because the residual model still
  maps between grid and mesh.
- Mamba state becomes a major remaining cost, especially for
  `d_inner=256`, `d_state=128`.
- `res_mamba mp=2`, BPTT16, `d_inner=128`, `d_state=64` should be the first
  single-GPU target.
- `res_mamba mp=2`, BPTT16, `d_inner=256`, `d_state=128` may need Mamba remat
  or a shorter BPTT/chunking fallback.

## Why Grid Inputs Are Not The First Bottleneck

The grid latent tensor is large, but not the dominant term:

- Grid latent across BPTT16 is about `1 GiB`.
- Mesh2grid edge/message tensors are about `3 GiB` per retained tensor class.
- Official `mp=16` mesh processor edge tensors are about `20 GiB` per retained
  tensor class across BPTT16.

Therefore, optimizing away only `input_grid_nodes`, `latent_grid_nodes`, or
`output_grid_nodes` cannot solve the official res1 memory problem by itself.
The better first target is retained GNN hidden activations, especially mesh
processor steps and mesh2grid.

The grid-side skip still matters functionally. `mesh2grid` consumes both the
updated mesh nodes and the grid-side skip latent produced by `grid2mesh`. Exact
training should keep that value, or recompute it during backward, rather than
dropping it.

## Build 1: Conservative, No Deliberate FLOPs Increase

Goal: reduce unnecessary gradient memory without adding activation
recomputation.

Implementation direction:

1. Split params before differentiation:
   - `trainable_mamba_params`
   - `frozen_graphcast_params`
2. Define the training loss as a function of both trees, but differentiate only
   the trainable tree:

   ```python
   loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
       trainable_mamba_params,
       frozen_graphcast_params,
       batch,
       ...
   )
   ```

3. Merge trainable and frozen params only where a full tree is required by the
   model apply function or checkpoint saving.
4. Keep optimizer state only for Mamba params.
5. For residual Mamba, precompute official baseline predictions or residual
   targets outside the jitted training loss.
6. Apply `jax.lax.stop_gradient` to any loaded baseline fields used inside the
   loss.

Expected memory/FLOPs tradeoff:

- No intentional recomputation.
- Lower backward memory because frozen GraphCast params are not differentiated.
- Potentially lower compile/backward work because the gradient output tree is
  smaller.
- This is necessary but probably not enough for official res1 `gc_mamba mp=16`.
- This may be enough to make the smaller `res_mamba mp=2` variants practical,
  especially with `d_inner=128`, `d_state=64`.

Current issue to fix:

- The current frozen-Mamba setup masks optimizer updates for non-Mamba params,
  but still calls `jax.value_and_grad(loss_fn)(full_params, ...)`.
- That computes a full parameter gradient tree and can keep frozen GraphCast
  modules on the gradient path even though their updates are discarded.

Checkpoint handling:

- Keep frozen params unchanged.
- Save full checkpoints by merging the updated Mamba leaves back into the
  original full parameter tree.
- The saved format should stay compatible with existing evaluation/loading
  code.

## Build 2: Optimal, Moderate Rematerialization

Goal: target the largest retained activations while keeping the recomputation
cost reasonable.

Start from the Conservative build, then add rematerialization in this order:

1. Rematerialize each mesh processor step.
   - This is the highest-value change for official `gc_mamba mp=16`.
   - It prevents storing the heavy per-step mesh edge/message MLP activations
     for every processor layer across BPTT.
2. Rematerialize `mesh2grid`.
   - Mesh2grid has about `3 GiB` per retained edge tensor class across BPTT16.
   - It is close to the output side and is a good checkpoint boundary.
3. Rematerialize interleaved Mamba blocks only if profiling shows Mamba state or
   temporal hidden activations are the next limiting factor.
   - This is most relevant for `d_inner=256`, `d_state=128`.
4. Initially leave `grid2mesh` stored.
   - Grid2mesh is large, but usually smaller than mesh2grid and the full
     processor stack.
   - Rematerialize it only after profiling the Optimal build.

Expected memory/FLOPs tradeoff:

- Backward recomputes processor-step and mesh2grid forward work.
- Wall-clock time increases, but memory drops substantially.
- For `res_mamba mp=2`, this should be a strong single-GPU target.
- For official `gc_mamba mp=16`, this is the first build that has a realistic
  chance on a large single GPU, especially with smaller Mamba dimensions.

Suggested fit order:

1. Validate `res_mamba mp=2`, `d_inner=128`, `d_state=64`, BPTT16.
2. Try `res_mamba mp=2`, `d_inner=256`, `d_state=128`, BPTT16 with Mamba remat
   enabled if needed.
3. Try `gc_mamba mp=16`, `d_inner=128`, `d_state=64`, BPTT16.
4. Treat `gc_mamba mp=16`, `d_inner=256`, `d_state=128`, BPTT16 as high risk
   unless profiling shows enough headroom.

## Build 3: Extreme, Maximum Memory Savings

Goal: keep exact training as long as possible while maximizing memory savings.

Start from the Optimal build, then add:

1. Rematerialize `grid2mesh`.
   - This removes another large grid-edge activation source.
   - It recomputes the grid-side skip during backward instead of storing its
     producer graph.
2. Rematerialize all temporal/Mamba blocks.
   - This targets the large Mamba state cases.
3. Consider BPTT chunking if exact BPTT16 still does not fit.
   - This reduces peak memory but changes the gradient horizon if gradients are
     stopped between chunks.
   - Use this only as an explicit approximation or as a separate experiment.
4. Consider a custom VJP or fused frozen GraphCast roundtrip only if standard
   checkpointing is insufficient.
   - The custom VJP would choose exactly which values to save and which values
     to recompute.
   - This is higher engineering risk and should come after profiling simpler
     remat boundaries.

Expected memory/FLOPs tradeoff:

- Largest memory reduction.
- Highest recomputation cost.
- Most likely to fit difficult single-GPU cases.
- Slower training should be expected.

## Autograd Path Requirements

The Mamba-only training path should not compute and then discard full
GraphCast gradients.

Required behavior:

- Only Mamba leaves are passed as differentiable arguments to
  `jax.value_and_grad`.
- Frozen GraphCast leaves are closed over or passed as nondifferentiated
  arguments.
- The optimizer sees only Mamba leaves.
- Frozen leaves are copied unchanged into the saved full checkpoint.
- Any baseline fields used by residual training are constants from the point of
  view of the training loss.

Important distinction:

- `stop_gradient` on a baseline prediction or loaded residual target can remove
  that baseline from the gradient path.
- `stop_gradient` on the grid-side skip value does not remove the need to have
  the value itself for `mesh2grid`; it only prevents gradients flowing into the
  skip producer. Exact training still needs either to store the skip value or
  recompute it.

## Validation Plan

Unit and small-shape tests:

1. Param partition/merge:
   - Partition a full tree.
   - Modify only Mamba leaves.
   - Merge back.
   - Assert all frozen leaves are byte-for-byte or numerically unchanged.
2. Gradient target:
   - Run a tiny loss.
   - Assert gradients exist for Mamba leaves.
   - Assert frozen GraphCast leaves are not part of the gradient tree.
3. Remat equivalence:
   - Run tiny non-remat and remat models.
   - Compare loss and Mamba gradients within bf16 tolerance.
4. Residual baseline equivalence:
   - Compare one tiny online-baseline residual target against a precomputed
     residual target.
   - Assert the training loss matches within tolerance.
5. Checkpoint compatibility:
   - Save a merged full checkpoint.
   - Load it through the existing evaluation path.

Profiling acceptance:

1. Record peak GPU memory and step time before the change.
2. Record the same metrics for Conservative, Optimal, and Extreme builds.
3. Use `res_mamba mp=2`, `d_inner=128`, `d_state=64`, BPTT16 as the first
   single-GPU acceptance target.
4. Report whether `gc_mamba mp=16` fits only after the Optimal build has been
   profiled.

## Recommended Order Of Work

1. Implement Conservative autograd partitioning.
2. Implement precomputed or stopped residual baselines for `res_mamba`.
3. Profile `res_mamba mp=2`, BPTT16.
4. Add remat around each mesh processor step.
5. Add remat around `mesh2grid`.
6. Profile again and only then decide whether to remat Mamba and `grid2mesh`.

This ordering gives the cleanest separation between:

- memory saved by removing unnecessary gradient paths,
- memory saved by activation recomputation,
- and time lost to extra FLOPs.
