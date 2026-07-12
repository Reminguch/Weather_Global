# res=0.25 residual-Mamba spatial-sharding (4x80GB, 1 data x 4 spatial)

Validation probes for training a residual Mamba head on frozen
GraphCast_operational (0.25deg, mesh 2to6). Longitude-shard grid nodes + g2m/m2g
edges across 4 GPUs; replicate mesh (40,962 nodes) + all params. Only cross-GPU
comm = ONE psum at the grid->mesh aggregation.

## Status (all validated on CPU 4-sim-devices unless noted)
- partition_probe.py          Phase 1/1.5 PASS: partition algebra on EXACT
                              0.25/mesh-6 graph; grad-equivalent to unsharded;
                              m2g 778,680x4 balanced; g2m E_max pad ~0.01%.
- probe_gc025_forward.py      Phase 3A (GPU): real GC-operational fwd = 16.2 GiB,
                              C_G=0.36s.
- probe_residual025_bwd.py    Phase 3B (GPU): residual head fwd+bwd = 65.5 GiB,
                              C_R=0.56s -> sharding is a MEMORY necessity.
- phase2_sharded_interaction.py  Phase 2 unit PASS: REAL grid2mesh
                              InteractionNetwork, 4-way sharded + psum, rel 5e-7,
                              HLO = 1 all-reduce, 0 grid all-gather.
- phase2_full_forward.py      Phase 2 e2e PASS: encode->mesh(3)->decode, 4-way
                              sharded, final grid rel 3.7e-7, 1 all-reduce total.

## Key mechanisms proven
1. psum via custom aggregate_edges_for_nodes_fn (callable): out=segment_sum;
   if num_segments==N_MESH(+1 sentinel): psum(out,"spatial"). Discriminator is
   the static num_segments. grid2mesh include_sent_messages=False so it fires
   exactly once, into mesh only.
2. Padded g2m edges MUST route to a SENTINEL mesh row (receiver=N_MESH), agg
   over N_MESH+1, slice [:N_MESH]. Masking padded edge INPUT features is
   insufficient (edge MLP uses sender/receiver node feats -> nonzero messages).
3. m2g partition by grid RECEIVER is exactly balanced (3x local grid); no psum
   (num_segments = local grid, not N_MESH+1).

## REMAINING = plumbing into graphcast.py (needs GPU iteration; do in a dedicated
## session with GPU-in-the-loop to validate the 65.5->~16-20 GiB memory drop):
Subclass GraphCast (and thread through GCResidualWithZeroHead):
- override _maybe_init to install HOST-prebuilt LOCAL partitioned TypedGraphs
  for _grid2mesh_graph_structure and _mesh2grid_graph_structure (+ sentinel mesh
  row); _mesh_graph_structure stays global (replicated).
- set _num_grid_nodes = local slab size; _num_mesh_nodes += 1 (sentinel).
- override _inputs_to_grid_node_features to emit the LOCAL grid slab;
  _grid_node_outputs_to_prediction to reshape the LOCAL slab.
- monkeypatch self._grid2mesh_gnn._aggregate_edges_for_nodes_fn = psum-agg
  (it is dereferenced at CALL time, so post-__init__ patch takes effect).
- wrap frozen-GC-fwd (stop_gradient) + residual head + optimizer in shard_map;
  GPU-probe sharded peak memory + HLO audit (allowed: 1 g2m all-reduce + grad
  reductions; forbidden: grid-sized all-gather).
See memory project_res025_sharding_plan.md for full budgets/decisions.
