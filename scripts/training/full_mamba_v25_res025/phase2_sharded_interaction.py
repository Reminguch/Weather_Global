#!/usr/bin/env python3
"""Phase 2 milestone: the REAL GraphCast grid2mesh InteractionNetwork, run
4-way longitude-sharded with a psum-injected aggregation, must match the
unsharded reference AND compile with NO grid-sized all-gather.

This exercises graphcast's actual typed_graph_net.InteractionNetwork +
_node_update + segment_sum scatter — not a toy — so a PASS de-risks the
graphcast.py surgery. include_sent_messages_in_node_update=False (grid2mesh
default) => the only aggregation is into mesh nodes (num_segments == N_MESH),
which is exactly where the cross-shard psum belongs. The discriminator is
num_segments == N_MESH (static python int at trace time).
"""
import os
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_force_host_platform_device_count=4")
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
from functools import partial
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
from jax.sharding import Mesh, PartitionSpec as P
try:
    from jax import shard_map
    SM_KW = {"check_vma": True}
except ImportError:
    from jax.experimental.shard_map import shard_map
    SM_KW = {"check_rep": True}

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))
from graphcast import icosahedral_mesh, grid_mesh_connectivity, typed_graph, typed_graph_net  # noqa: E402

N_SHARD, LAT = 4, 512  # latent size (small for CPU)
RES = 1.0
grid_lat = np.arange(-90, 90.001, RES, dtype=np.float32)
grid_lon = np.arange(0, 360, RES, dtype=np.float32)
NLAT, NLON = len(grid_lat), len(grid_lon)
N_GRID, COLS = NLAT * NLON, NLON // N_SHARD
meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=5)
finest = meshes[-1]
N_MESH = icosahedral_mesh.merge_meshes(meshes).vertices.shape[0]
frac = 0.5999912857713345
rq = np.linalg.norm(
    finest.vertices[icosahedral_mesh.faces_to_edges(finest.faces)[0]]
    - finest.vertices[icosahedral_mesh.faces_to_edges(finest.faces)[1]], axis=-1).max() * frac
g_grid, g_mesh = grid_mesh_connectivity.radius_query_indices(
    grid_latitude=grid_lat, grid_longitude=grid_lon, mesh=finest, radius=rq)
g_grid = g_grid.astype(np.int32); g_mesh = g_mesh.astype(np.int32)
E = len(g_grid)
print(f"grid={N_GRID} mesh={N_MESH} g2m_edges={E}")

CH = 4  # feature width for the probe
rng = np.random.default_rng(0)
grid_feat = rng.normal(size=(N_GRID, CH)).astype(np.float32)
mesh_feat = rng.normal(size=(N_MESH, CH)).astype(np.float32)
edge_feat = rng.normal(size=(E, CH)).astype(np.float32)

EK = typed_graph.EdgeSetKey("g2m", ("grid_nodes", "mesh_nodes"))
def make_graph(gfeat, gsend, mrecv, efeat, n_grid_local, mfeat):
    nm = mfeat.shape[0]
    return typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=jnp.array([1]), features=()),
        nodes={"grid_nodes": typed_graph.NodeSet(n_node=jnp.array([n_grid_local]), features=gfeat),
               "mesh_nodes": typed_graph.NodeSet(n_node=jnp.array([nm]), features=mfeat)},
        edges={EK: typed_graph.EdgeSet(n_edge=jnp.array([efeat.shape[0]]),
                                       indices=typed_graph.EdgesIndices(senders=gsend, receivers=mrecv),
                                       features=efeat)})

def mlp(name, out):
    return hk.nets.MLP([LAT, out], name=name, activation=jax.nn.gelu)

def build_net(agg):
    def upd_edge(e, sn, rn):
        return mlp("edge", CH)(jnp.concatenate([e, sn, rn], -1))
    def upd_grid(n, recv):
        return mlp("gnode", CH)(n)                      # grid receives nothing
    def upd_mesh(n, recv):
        return mlp("mnode", CH)(jnp.concatenate([n, recv["g2m"]], -1))
    return typed_graph_net.InteractionNetwork(
        update_edge_fn={"g2m": upd_edge},
        update_node_fn={"grid_nodes": upd_grid, "mesh_nodes": upd_mesh},
        aggregate_edges_for_nodes_fn=agg,
        include_sent_messages_in_node_update=False)

mesh_feat_j = jnp.asarray(mesh_feat)

# reference: unsharded, plain segment_sum
def ref_fn(gfeat, gsend, mrecv, efeat):
    g = make_graph(gfeat, gsend, mrecv, efeat, N_GRID, mesh_feat_j)
    out = build_net(jraph.segment_sum)(g)
    return out.nodes["mesh_nodes"].features
ref_t = hk.without_apply_rng(hk.transform(ref_fn))
prng = jax.random.PRNGKey(0)
params = ref_t.init(prng, jnp.asarray(grid_feat), jnp.asarray(g_grid),
                    jnp.asarray(g_mesh), jnp.asarray(edge_feat))
ref_mesh = ref_t.apply(params, jnp.asarray(grid_feat), jnp.asarray(g_grid),
                       jnp.asarray(g_mesh), jnp.asarray(edge_feat))

# ---- partition (reuse Phase 1.5 scheme) ----
def shard_of(ids): return (ids % NLON) // COLS
owner = shard_of(g_grid)
emax = int(np.bincount(owner, minlength=N_SHARD).max())
gl, ml, va, ef = [], [], [], []
for r in range(N_SHARD):
    eid = np.flatnonzero(owner == r)
    loc = (g_grid[eid] // NLON) * COLS + (g_grid[eid] % NLON) - r * COLS
    pad = emax - eid.size
    gl.append(np.pad(loc, (0, pad)).astype(np.int32))
    ml.append(np.pad(g_mesh[eid], (0, pad), constant_values=N_MESH).astype(np.int32))  # padded->sentinel
    va.append(np.pad(np.ones(eid.size, bool), (0, pad)))
    ef.append(np.pad(edge_feat[eid], ((0, pad), (0, 0))))
gl, ml, va, ef = map(np.stack, (gl, ml, va, ef))
grid_blocks = np.stack([grid_feat.reshape(NLAT, NLON, CH)[:, s*COLS:(s+1)*COLS].reshape(NLAT*COLS, CH)
                        for s in range(N_SHARD)])

def sharded_agg(data, seg, num_segments):
    out = jraph.segment_sum(data, seg, num_segments)
    if num_segments == N_MESH + 1:        # grid->mesh (incl sentinel row) aggregation
        out = jax.lax.psum(out, "spatial")
    return out

def sharded_core(gblk, gl_, ml_, va_, ef_):
    # mesh feats + one sentinel row (index N_MESH) that soaks up padded edges
    mfeat = jnp.concatenate([mesh_feat_j, jnp.zeros((1, CH), mesh_feat_j.dtype)], 0)
    g = make_graph(gblk, gl_, ml_, ef_, NLAT * COLS, mfeat)
    out = build_net(sharded_agg)(g).nodes["mesh_nodes"].features
    return out[:N_MESH]                   # drop sentinel
sharded_t = hk.without_apply_rng(hk.transform(sharded_core))
mesh_dev = Mesh(np.array(jax.devices()[:N_SHARD]), ("spatial",))

@partial(shard_map, mesh=mesh_dev,
         in_specs=(P(), P("spatial"), P("spatial"), P("spatial"), P("spatial"), P("spatial")),
         out_specs=P(), **SM_KW)
def run_sharded(p, gblk, gl_, ml_, va_, ef_):
    return sharded_t.apply(p, gblk[0], gl_[0], ml_[0], va_[0], ef_[0])

args = tuple(jnp.asarray(a) for a in (grid_blocks, gl, ml, va, ef))
sh_mesh = run_sharded(params, *args)
d = float(jnp.abs(ref_mesh - sh_mesh).max()); r = float(jnp.abs(ref_mesh).max())
print(f"mesh-node output  max|diff|={d:.2e}  max|ref|={r:.2e}  rel={d/max(r,1e-9):.2e}")

# HLO scan: forbid grid-sized all-gather
lowered = jax.jit(run_sharded).lower(params, *args)
hlo = lowered.compile().as_text()
import re
bad = [ln.strip() for ln in hlo.splitlines()
       if "all-gather" in ln and (str(N_GRID) in ln or str(E) in ln)]
allred = [ln for ln in hlo.splitlines() if "all-reduce" in ln]
print(f"all-reduce ops: {len(allred)}   forbidden grid-sized all-gather: {len(bad)}")
for b in bad[:5]: print("  BAD:", b[:120])
ok = d < 1e-4 * max(r, 1.0) and len(bad) == 0
print("\nPHASE2 SHARDED INTERACTION-NETWORK:", "PASS" if ok else "FAIL")
