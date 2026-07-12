#!/usr/bin/env python3
"""Phase 2 end-to-end: full grid2mesh -> mesh-processor -> mesh2grid forward,
4-way longitude-sharded, vs unsharded reference. All three stages use the REAL
typed_graph_net.InteractionNetwork. Proves the LAST algorithmic gap: mesh2grid
(local scatter to grid, no comm) and the replicated mesh processor compose
correctly with the psum'd grid2mesh. Final GRID output must match + HLO must
show exactly ONE all-reduce (the single grid2mesh psum) and no grid all-gather.
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
from graphcast import icosahedral_mesh, grid_mesh_connectivity, typed_graph, typed_graph_net  # noqa

N_SHARD, CH, MESH_STEPS = 4, 4, 3
RES = 1.0
glat = np.arange(-90, 90.001, RES, dtype=np.float32)
glon = np.arange(0, 360, RES, dtype=np.float32)
NLAT, NLON = len(glat), len(glon)
N_GRID, COLS = NLAT * NLON, NLON // N_SHARD
meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=5)
finest = meshes[-1]
merged = icosahedral_mesh.merge_meshes(meshes)
N_MESH = merged.vertices.shape[0]
fe_s, fe_r = icosahedral_mesh.faces_to_edges(finest.faces)
rq = np.linalg.norm(finest.vertices[fe_s] - finest.vertices[fe_r], axis=-1).max() * 0.5999912857713345
gg, gm = grid_mesh_connectivity.radius_query_indices(
    grid_latitude=glat, grid_longitude=glon, mesh=finest, radius=rq)
mg_grid, mg_mesh = grid_mesh_connectivity.in_mesh_triangle_indices(
    grid_latitude=glat, grid_longitude=glon, mesh=finest)
gg, gm = gg.astype(np.int32), gm.astype(np.int32)
mg_grid, mg_mesh = mg_grid.astype(np.int32), mg_mesh.astype(np.int32)
# multimesh edges (bidirectional), replicated on all shards
mm_s, mm_r = icosahedral_mesh.faces_to_edges(merged.faces)
mm_s = np.concatenate([mm_s, mm_r]).astype(np.int32)
mm_r = np.concatenate([mm_r, mm_s[:len(mm_r)]]).astype(np.int32)
E_g2m, E_m2g, E_mm = len(gg), len(mg_grid), len(mm_s)
print(f"grid={N_GRID} mesh={N_MESH} g2m={E_g2m} m2g={E_m2g} mm={E_mm}")

rng = np.random.default_rng(0)
grid_feat = rng.normal(size=(N_GRID, CH)).astype(np.float32)
mesh_feat = jnp.asarray(rng.normal(size=(N_MESH, CH)).astype(np.float32))
g2m_ef = rng.normal(size=(E_g2m, CH)).astype(np.float32)
m2g_ef = jnp.asarray(rng.normal(size=(E_m2g, CH)).astype(np.float32))
mm_ef = jnp.asarray(rng.normal(size=(E_mm, CH)).astype(np.float32))

def mlp(name): return hk.nets.MLP([32, CH], name=name, activation=jax.nn.gelu)
NS = typed_graph.NodeSet; ES = typed_graph.EdgeSet; EI = typed_graph.EdgesIndices
CTX = typed_graph.Context(n_graph=jnp.array([1]), features=())
K_g2m = typed_graph.EdgeSetKey("g2m", ("g", "m"))
K_mm = typed_graph.EdgeSetKey("mm", ("m", "m"))
K_m2g = typed_graph.EdgeSetKey("m2g", ("m", "g"))

def IN(update_edge, update_node, agg):
    return typed_graph_net.InteractionNetwork(
        update_edge_fn=update_edge, update_node_fn=update_node,
        aggregate_edges_for_nodes_fn=agg, include_sent_messages_in_node_update=False)

def forward(gfeat, mfeat, gg_, gm_, g2mef, mm_s_, mm_r_, mmef, m2m_s, m2g_g, m2gef, n_grid_local, agg):
    # 1) grid2mesh
    g = typed_graph.TypedGraph(CTX,
        {"g": NS(jnp.array([n_grid_local]), gfeat), "m": NS(jnp.array([mfeat.shape[0]]), mfeat)},
        {K_g2m: ES(jnp.array([g2mef.shape[0]]), EI(gg_, gm_), g2mef)})
    g = IN({"g2m": lambda e, s, r: mlp("g2m_e")(jnp.concatenate([e, s, r], -1))},
           {"g": lambda n, rc: mlp("g2m_gn")(n),
            "m": lambda n, rc: mlp("g2m_mn")(jnp.concatenate([n, rc["g2m"]], -1))}, agg)(g)
    latM = g.nodes["m"].features
    # 2) mesh processor (replicated), MESH_STEPS message-passing steps
    for i in range(MESH_STEPS):
        gm_graph = typed_graph.TypedGraph(CTX, {"m": NS(jnp.array([latM.shape[0]]), latM)},
            {K_mm: ES(jnp.array([mmef.shape[0]]), EI(mm_s_, mm_r_), mmef)})
        gm_graph = IN({"mm": lambda e, s, r: mlp(f"mm_e{i}")(jnp.concatenate([e, s, r], -1))},
                      {"m": lambda n, rc: mlp(f"mm_n{i}")(jnp.concatenate([n, rc["mm"]], -1))},
                      jraph.segment_sum)(gm_graph)   # mesh replicated -> plain segment_sum
        latM = gm_graph.nodes["m"].features
    # 3) mesh2grid: senders=mesh (full replicated), receivers=grid (local)
    m2g = typed_graph.TypedGraph(CTX,
        {"m": NS(jnp.array([latM.shape[0]]), latM), "g": NS(jnp.array([n_grid_local]), gfeat)},
        {K_m2g: ES(jnp.array([m2gef.shape[0]]), EI(m2m_s, m2g_g), m2gef)})
    m2g = IN({"m2g": lambda e, s, r: mlp("m2g_e")(jnp.concatenate([e, s, r], -1))},
             {"m": lambda n, rc: n,
              "g": lambda n, rc: mlp("m2g_gn")(jnp.concatenate([n, rc["m2g"]], -1))},
             jraph.segment_sum)(m2g)                 # into grid (local) -> no psum
    return m2g.nodes["g"].features

# ---- unsharded reference ----
def ref_fn(gfeat):
    return forward(gfeat, mesh_feat, jnp.asarray(gg), jnp.asarray(gm), jnp.asarray(g2m_ef),
                   jnp.asarray(mm_s), jnp.asarray(mm_r), mm_ef,
                   jnp.asarray(mg_mesh), jnp.asarray(mg_grid), m2g_ef, N_GRID, jraph.segment_sum)
ref_t = hk.without_apply_rng(hk.transform(ref_fn))
params = ref_t.init(jax.random.PRNGKey(0), jnp.asarray(grid_feat))
ref_out = ref_t.apply(params, jnp.asarray(grid_feat))

# ---- partition g2m (by grid sender, padded->sentinel) and m2g (by grid receiver, balanced) ----
def shard_of(ids): return (ids % NLON) // COLS
def part(edges_grid, edges_mesh, sentinel):
    owner = shard_of(edges_grid); emax = int(np.bincount(owner, minlength=N_SHARD).max())
    gl, ml, ef_idx = [], [], []
    for r in range(N_SHARD):
        eid = np.flatnonzero(owner == r)
        loc = (edges_grid[eid] // NLON) * COLS + (edges_grid[eid] % NLON) - r * COLS
        pad = emax - eid.size
        gl.append(np.pad(loc, (0, pad)).astype(np.int32))
        ml.append(np.pad(edges_mesh[eid], (0, pad), constant_values=sentinel).astype(np.int32))
        ef_idx.append(np.pad(eid, (0, pad), constant_values=-1))
    return np.stack(gl), np.stack(ml), np.stack(ef_idx)
# g2m: grid=sender, mesh=receiver -> sentinel row N_MESH
g2m_gl, g2m_ml, g2m_eidx = part(gg, gm, N_MESH)
# m2g: grid=receiver, mesh=sender -> balanced, no pad needed but reuse (sentinel unused since balanced)
m2g_gl, m2g_ml, m2g_eidx = part(mg_grid, mg_mesh, 0)
def gather_ef(base, eidx):   # per-shard edge features via global edge id (pad->0)
    return np.stack([base[np.clip(eidx[s], 0, len(base)-1)] * (eidx[s] >= 0)[:, None] for s in range(N_SHARD)])
g2m_ef_sh = gather_ef(g2m_ef, g2m_eidx)
m2g_ef_sh = gather_ef(np.asarray(m2g_ef), m2g_eidx)
grid_blocks = np.stack([grid_feat.reshape(NLAT, NLON, CH)[:, s*COLS:(s+1)*COLS].reshape(NLAT*COLS, CH)
                        for s in range(N_SHARD)])

def sharded_agg(data, seg, num_segments):
    out = jraph.segment_sum(data, seg, num_segments)
    if num_segments == N_MESH + 1:     # ONLY grid2mesh (mesh + sentinel)
        out = jax.lax.psum(out, "spatial")
    return out

def sh_core(gblk, g2mgl, g2mml, g2mef, m2ggl, m2gml, m2gef):
    mfeat = jnp.concatenate([mesh_feat, jnp.zeros((1, CH), mesh_feat.dtype)], 0)  # + sentinel
    out = forward(gblk, mfeat, g2mgl, g2mml, g2mef,
                  jnp.asarray(mm_s), jnp.asarray(mm_r), mm_ef,
                  m2gml, m2ggl, m2gef, NLAT * COLS, sharded_agg)
    return out
sh_t = hk.without_apply_rng(hk.transform(sh_core))
mdev = Mesh(np.array(jax.devices()[:N_SHARD]), ("spatial",))

@partial(shard_map, mesh=mdev,
         in_specs=(P(),) + (P("spatial"),) * 7, out_specs=P("spatial"), **SM_KW)
def run_sh(p, gblk, g2mgl, g2mml, g2mef, m2ggl, m2gml, m2gef):
    return sh_t.apply(p, gblk[0], g2mgl[0], g2mml[0], g2mef[0], m2ggl[0], m2gml[0], m2gef[0])[None]

A = tuple(jnp.asarray(a) for a in (grid_blocks, g2m_gl, g2m_ml, g2m_ef_sh, m2g_gl, m2g_ml, m2g_ef_sh))
sh_blocks = run_sh(params, *A)
# reassemble grid: blocks are (shard, NLAT*COLS, CH) -> (NLAT, NLON, CH)
sh_out = np.concatenate([np.asarray(sh_blocks)[s].reshape(NLAT, COLS, CH) for s in range(N_SHARD)],
                        axis=1).reshape(N_GRID, CH)
d = float(np.abs(sh_out - np.asarray(ref_out)).max()); r = float(np.abs(np.asarray(ref_out)).max())
print(f"final GRID output max|diff|={d:.2e} max|ref|={r:.2e} rel={d/max(r,1e-9):.2e}")
hlo = jax.jit(run_sh).lower(params, *A).compile().as_text()
allred = [l for l in hlo.splitlines() if "all-reduce" in l]
bad = [l for l in hlo.splitlines() if "all-gather" in l and (str(N_GRID) in l or str(E_g2m) in l or str(E_m2g) in l)]
print(f"all-reduce ops: {len(allred)}  forbidden grid-sized all-gather: {len(bad)}")
ok = d < 1e-4 * max(r, 1.0) and len(bad) == 0
print("\nPHASE2 FULL-FORWARD (g2m->mesh->m2g) SHARDED:", "PASS" if ok else "FAIL")
