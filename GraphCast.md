# Minimal GraphCast Toy-Training Pipeline

This document summarizes the **minimal set of GraphCast repository modules** you should import/copy, and the full training pipeline for a toy run.

## 1) Minimal modules to import from `google-deepmind/graphcast`

### Required core model/graph modules
- `graphcast/graphcast.py`  
  Main one-step GraphCast model (`GraphCast`, `ModelConfig`, `TaskConfig`).
- `graphcast/deep_typed_graph_net.py`  
  Deep typed GNN block used by GraphCast encoder/processor/decoder.
- `graphcast/typed_graph.py`  
  Data structures: `TypedGraph`, `NodeSet`, `EdgeSet`, `EdgesIndices`.
- `graphcast/typed_graph_net.py`  
  Typed graph interaction/update primitives.
- `graphcast/model_utils.py`  
  Grid/node/edge spatial feature engineering and dataset stacking/unstacking.
- `graphcast/grid_mesh_connectivity.py`  
  Grid↔mesh connectivity (`radius_query_indices`, `in_mesh_triangle_indices`).
- `graphcast/icosahedral_mesh.py`  
  Multi-resolution icosahedral mesh hierarchy.
- `graphcast/losses.py`  
  Latitude/pressure-weighted losses.
- `graphcast/predictor_base.py`  
  Predictor API used by wrappers and training/inference entrypoints.
- `graphcast/xarray_jax.py` and `graphcast/xarray_tree.py`  
  Make xarray containers compatible with JAX transformations.

### Required training wrappers/utilities
- `graphcast/normalization.py`  
  `InputsAndResiduals` wrapper: normalize inputs/forcings and train residual targets.
- `graphcast/autoregressive.py`  
  Wrap one-step predictor into multi-step autoregressive training.
- `graphcast/data_utils.py`  
  Extract `inputs`, `targets`, `forcings`; derive progress features and optional TISR.

### Optional but useful
- `graphcast/casting.py` (`Bfloat16Cast`) for memory/perf.
- `graphcast/solar_radiation.py` if computing `toa_incident_solar_radiation`.
- `graphcast/checkpoint.py` for checkpoint IO.
- `graphcast/rollout.py` for long inference rollout (non-differentiable).

## 2) Full toy-training pipeline (end-to-end)

## Step A: Define small toy configs
1. Build `ModelConfig` with small values (example):  
   `mesh_size=2..3`, `latent_size=32`, `gnn_msg_steps=2..4`, `hidden_layers=1`, `radius_query_fraction_edge_length=0.6`.
2. Build `TaskConfig` with low-res levels (e.g. 13 pressure levels) and 6h input window.

## Step B: Prepare dataset and split into model tensors
1. Start from one xarray dataset containing variables, coords (`time`, `lat`, `lon`, `level`, and typically `datetime`).
2. Use `data_utils.extract_inputs_targets_forcings(...)` with:
   - `input_variables`
   - `target_variables`
   - `forcing_variables`
   - `pressure_levels`
   - `input_duration`
   - `target_lead_times` (for toy training, begin with one step: `6h`)
3. This utility:
   - Subselects pressure levels.
   - Adds derived forcing vars if requested (day/year progress sin/cos).
   - Adds TISR forcing if requested.
   - Returns `(inputs, targets, forcings)` aligned for autoregressive setup.

## Step C: Build predictor stack in the same order as upstream demo
1. Base one-step model:  
   `predictor = graphcast.GraphCast(model_config, task_config)`
2. Optional precision wrapper:  
   `predictor = casting.Bfloat16Cast(predictor)`
3. Normalize + residual target wrapper:  
   `predictor = normalization.InputsAndResiduals(predictor, stddev_by_level, mean_by_level, diffs_stddev_by_level)`
4. Autoregressive wrapper:  
   `predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)`

The wrapper order above is what the official notebook uses.

## Step D: What happens inside `GraphCast` (graph construction + model)

### D1. Generate mesh and grid node definitions
1. Mesh hierarchy: `icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_size)`.
2. Mesh node lat/lon are derived from mesh vertices (cartesian→spherical→lat/lon).
3. Grid nodes are flattened from lat-lon meshgrid (`num_lat * num_lon`).

### D2. Define graph edges (connectivity)
1. **Grid2Mesh edges**: `grid_mesh_connectivity.radius_query_indices(...)`  
   Radius based on `max_edge_length(finest_mesh) * radius_query_fraction_edge_length`.
2. **Mesh graph edges**: constructed from merged multimesh connectivity.
3. **Mesh2Grid edges**: `grid_mesh_connectivity.in_mesh_triangle_indices(...)`  
   Each grid point linked to the 3 mesh vertices of containing triangle.

### D3. Define node/edge features
1. `model_utils.get_bipartite_graph_spatial_features(...)` generates structural node and edge features:
   - node lat/lon features
   - relative positional edge features
   - normalized edge lengths/offsets
2. Inputs/forcings are stacked into per-grid-node channel vectors (`model_utils.dataset_to_stacked` path).

### D4. Define MLPs and GNN layers
GraphCast instantiates three `DeepTypedGraphNet` blocks:
1. **Encoder (`grid2mesh_gnn`)**  
   - embeds raw grid/mesh node features and grid2mesh edge features  
   - 1 message-passing step
2. **Processor (`mesh_gnn`)**  
   - message passing on mesh-only graph  
   - `gnn_msg_steps` steps
3. **Decoder (`mesh2grid_gnn`)**  
   - mesh→grid message passing  
   - outputs `grid_nodes` channels matching target variable count

Internally, `DeepTypedGraphNet` builds per-node-type/per-edge-type MLP update functions with optional layer norm.

### D5. Forward path summary
1. xarray inputs + forcings -> flattened grid node features.
2. Run encoder (grid→mesh latents).
3. Run mesh processor.
4. Run decoder (mesh→grid outputs).
5. Convert flat grid outputs back to xarray target template.

## Step E: Normalization, residual targeting, and loss
1. `normalization.InputsAndResiduals`:
   - normalizes inputs and forcings using dataset statistics.
   - for variables present in both inputs and targets, trains on normalized residuals (`target - last_input`).
   - unnormalizes predictions back to physical scale.
2. One-step loss in `graphcast.GraphCast.loss_and_predictions` uses `losses.weighted_mse_per_level(...)`:
   - latitude weighting (area-aware)
   - level weighting (pressure-level aware)
   - per-variable scalar weights

## Step F: Training functions and update step
1. Define Haiku transformed functions:
   - `run_forward(...)`
   - `loss_fn(...)`
2. Wrap with `hk.transform_with_state`.
3. Initialize params/state with sample `(inputs, targets, forcings)`.
4. Compute gradients via `jax.value_and_grad`.
5. Apply optimizer update (Optax or your own JAX optimizer).
6. Repeat over minibatches.

For a strict toy run, train only 1-step lead time first (`targets.time = [6h]`) and disable long rollouts.

## Step G: Evaluation and rollout
1. For differentiable training-time multi-step: keep `autoregressive.Predictor`.
2. For long inference rollout without backprop: use `rollout.py`.
3. Save/load checkpoints with `checkpoint.py` if needed.

## 3) Minimal practical recipe for your first toy run
1. Keep only 13 levels, 1-degree data, mesh size 2-3, latent 32.
2. Train on a single lead time (`6h`) first.
3. Include only core target vars you can reliably load.
4. Add forcings (`day/year progress` and optionally `toa_incident_solar_radiation`) only after baseline works.
5. Once stable, increase `gnn_msg_steps`, mesh size, and forecast horizon.

## Sources examined
- https://github.com/google-deepmind/graphcast
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/README.md
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast_demo.ipynb
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/graphcast.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/grid_mesh_connectivity.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/model_utils.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/deep_typed_graph_net.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/typed_graph.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/typed_graph_net.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/normalization.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/autoregressive.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/losses.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/data_utils.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/casting.py
- https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/solar_radiation.py
