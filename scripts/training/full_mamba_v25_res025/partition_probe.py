#!/usr/bin/env python3
"""Phase 1.5 of the res=0.25 sharding plan: graph-partition correctness probe.

Status target: "partition algebra + exact-graph bookkeeping PASS" (still a toy
network, NOT the real GraphCast forward — that is Phase 2).

v2 changes after review:
  - query radius derived from max finest-mesh edge length x fraction (0.6,
    GraphCast default; TODO read from operational ckpt model_config), float32
    coords like the official code. No hardcoded 0.018.
  - check_rep=True (no silent replication assumptions).
  - per-point forward output regression + mesh-aggregate intermediate check.
  - EdgeShard carries global_edge_id; toy net consumes edge FEATURES gathered
    by global_edge_id, so feature/index misalignment fails the test.
  - bool masks + jnp.where AFTER a biased edge MLP (catches mask-before-bias).
  - int32 indices, explicit f32 params.
  - partition_g2m / partition_m2g wrappers (no dead `by` arg); m2g unpadded.
  - per-edge exclusivity + index-range asserts; g2m degree histogram.
  - jit(value_and_grad) regression vs eager.
  - flatten/block round-trip test with value = lat*10000 + lon.
"""
import os
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_force_host_platform_device_count=4")
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
try:
    from jax import shard_map
    SM_KW = {"check_vma": True}
except ImportError:
    from jax.experimental.shard_map import shard_map
    SM_KW = {"check_rep": True}

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))
from graphcast import icosahedral_mesh, grid_mesh_connectivity  # noqa: E402

N_SHARD, D = 4, 8
RES = float(os.environ.get("PROBE_RES", "1.0"))
MESH_SPLITS = int(os.environ.get("PROBE_SPLITS", "5"))
# exact value from GraphCast_operational ckpt model_config (NOT 0.6 exactly)
RADIUS_FRACTION = float(os.environ.get("PROBE_FRACTION", "0.5999912857713345"))
grid_lat = np.arange(-90, 90.001, RES, dtype=np.float32)
grid_lon = np.arange(0, 360, RES, dtype=np.float32)
NLAT, NLON = len(grid_lat), len(grid_lon)
N_GRID, COLS = NLAT * NLON, NLON // N_SHARD
print(f"RES={RES} splits={MESH_SPLITS} fraction={RADIUS_FRACTION!r} grid={NLAT}x{NLON}")

meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=MESH_SPLITS)
finest = meshes[-1]
N_MESH = icosahedral_mesh.merge_meshes(meshes).vertices.shape[0]

def max_mesh_edge_distance(mesh):
    s, r = icosahedral_mesh.faces_to_edges(mesh.faces)
    return np.linalg.norm(mesh.vertices[s] - mesh.vertices[r], axis=-1).max()

query_radius = max_mesh_edge_distance(finest) * RADIUS_FRACTION
print(f"max finest edge={max_mesh_edge_distance(finest):.9f}  query_radius={query_radius:.9f}")

# BOTH functions return (grid_indices, mesh_indices) — do NOT swap.
g2m_grid, g2m_mesh = grid_mesh_connectivity.radius_query_indices(
    grid_latitude=grid_lat, grid_longitude=grid_lon, mesh=finest, radius=query_radius)
m2g_grid, m2g_mesh = grid_mesh_connectivity.in_mesh_triangle_indices(
    grid_latitude=grid_lat, grid_longitude=grid_lon, mesh=finest)
g2m_grid = g2m_grid.astype(np.int32); g2m_mesh = g2m_mesh.astype(np.int32)
m2g_grid = m2g_grid.astype(np.int32); m2g_mesh = m2g_mesh.astype(np.int32)
E_G2M, E_M2G = len(g2m_grid), len(m2g_grid)

for name, g, m in [("g2m", g2m_grid, g2m_mesh), ("m2g", m2g_grid, m2g_mesh)]:
    assert 0 <= g.min() and g.max() < N_GRID, name
    assert 0 <= m.min() and m.max() < N_MESH, name
deg = np.bincount(g2m_grid, minlength=N_GRID)
print(f"grid={N_GRID} mesh={N_MESH} g2m={E_G2M} m2g={E_M2G}  "
      f"g2m degree min={deg.min()} max={deg.max()} zero={np.count_nonzero(deg==0)}")
assert np.count_nonzero(deg == 0) == 0, "grid nodes with no g2m edge — radius wrong"

# flatten order: node_id = lat_i * NLON + lon_j (validated by round-trip below)
def shard_of(ids):
    return (ids % NLON) // COLS

def audit_m2g(recv):
    assert recv.shape == (3 * N_GRID,)
    assert np.all(np.bincount(recv, minlength=N_GRID) == 3)
    owner = shard_of(recv)
    for r in range(N_SHARD):
        mask = owner == r
        assert int(mask.sum()) == 3 * NLAT * COLS, (r, int(mask.sum()))
        lr = (recv[mask] // NLON) * COLS + (recv[mask] % NLON) - r * COLS
        assert np.all(np.bincount(lr, minlength=NLAT * COLS) == 3)
    print(f"M2G PARTITION AUDIT: PASS (each shard exactly {3*NLAT*COLS} edges)")

audit_m2g(m2g_grid)
assign = shard_of(g2m_grid)[:, None] == np.arange(N_SHARD)[None, :]
np.testing.assert_array_equal(assign.sum(1), np.ones(E_G2M, np.int64))

@dataclass(frozen=True)
class EdgeShard:
    grid_local: np.ndarray
    mesh: np.ndarray
    valid: np.ndarray
    global_edge_id: np.ndarray

def partition_grid_owned(edges_grid, edges_mesh):
    owner = shard_of(edges_grid)
    emax = int(np.bincount(owner, minlength=N_SHARD).max())
    gl, ms, va, ei = [], [], [], []
    for r in range(N_SHARD):
        eid = np.flatnonzero(owner == r).astype(np.int32)
        g, m = edges_grid[eid], edges_mesh[eid]
        loc = ((g // NLON) * COLS + (g % NLON) - r * COLS).astype(np.int32)
        pad = emax - eid.size
        gl.append(np.pad(loc, (0, pad))); ms.append(np.pad(m, (0, pad)))
        va.append(np.pad(np.ones(eid.size, np.bool_), (0, pad)))
        ei.append(np.pad(eid, (0, pad)))
    return EdgeShard(np.stack(gl), np.stack(ms), np.stack(va), np.stack(ei))

def partition_g2m(grid_senders, mesh_receivers):
    return partition_grid_owned(grid_senders, mesh_receivers)

def partition_m2g(grid_receivers, mesh_senders):
    sh = partition_grid_owned(grid_receivers, mesh_senders)
    assert sh.grid_local.shape[1] == 3 * NLAT * COLS and sh.valid.all()
    return sh

G2M = partition_g2m(g2m_grid, g2m_mesh)
M2G = partition_m2g(m2g_grid, m2g_mesh)
print(f"g2m per-shard={np.bincount(shard_of(g2m_grid), minlength=N_SHARD)} E_max={G2M.grid_local.shape[1]}")

# ---- flatten/block round-trip with identifiable encoding ----
code = (np.arange(NLAT)[:, None] * 10000 + np.arange(NLON)[None, :]).astype(np.float32)
def to_blocks(a):
    return np.stack([a.reshape(NLAT, NLON, -1)[:, s*COLS:(s+1)*COLS].reshape(NLAT*COLS, -1)
                     for s in range(N_SHARD)])
def from_blocks(b):
    return np.concatenate([b[s].reshape(NLAT, COLS, -1) for s in range(N_SHARD)],
                          axis=1).reshape(N_GRID, -1)
blk = to_blocks(code.reshape(N_GRID, 1))
for s in range(N_SHARD):
    assert blk[s][0, 0] == 0 * 10000 + s * COLS
    assert blk[s][-1, 0] == (NLAT - 1) * 10000 + (s + 1) * COLS - 1
np.testing.assert_array_equal(from_blocks(blk).ravel(), code.ravel())
print("FLATTEN/BLOCK ROUND-TRIP: PASS")

# ---- toy network: edge features gathered by global_edge_id + biased MLP + where-mask ----
rng = np.random.default_rng(0)
f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
params = {"w_enc": f32(rng.normal(size=(2 * D, D)) * 0.1), "b_enc": f32(rng.normal(size=(D,))),
          "w_mesh": f32(rng.normal(size=(D, D)) * 0.1),
          "w_dec": f32(rng.normal(size=(2 * D, D)) * 0.1), "b_dec": f32(rng.normal(size=(D,)))}
x_grid = f32(rng.normal(size=(N_GRID, D)))
tgt = f32(rng.normal(size=(N_GRID, D)))
g2m_feat = f32(rng.normal(size=(E_G2M, D)))      # structural edge features
m2g_feat = f32(rng.normal(size=(E_M2G, D)))

def seg_sum(d, i, n): return jax.ops.segment_sum(d, i, num_segments=n)

def ref_mesh_agg(p, xg):
    msg = jnp.tanh(jnp.concatenate([xg[g2m_grid], g2m_feat], -1) @ p["w_enc"] + p["b_enc"])
    return seg_sum(msg, g2m_mesh, N_MESH)

def ref_forward(p, xg):
    mesh_h = jnp.tanh(ref_mesh_agg(p, xg) @ p["w_mesh"])
    dec = jnp.concatenate([mesh_h[m2g_mesh], m2g_feat], -1) @ p["w_dec"] + p["b_dec"]
    return seg_sum(dec, m2g_grid, N_GRID)

def ref_loss(p, xg, t): return jnp.mean((ref_forward(p, xg) - t) ** 2)

xb, tb = jnp.asarray(to_blocks(np.asarray(x_grid))), jnp.asarray(to_blocks(np.asarray(tgt)))
g2m_feat_sh = jnp.asarray(np.asarray(g2m_feat)[G2M.global_edge_id])   # (shard,E,D) by edge id
m2g_feat_sh = jnp.asarray(np.asarray(m2g_feat)[M2G.global_edge_id])
Gj = tuple(map(jnp.asarray, (G2M.grid_local, G2M.mesh, G2M.valid, g2m_feat_sh)))
Mj = tuple(map(jnp.asarray, (M2G.grid_local, M2G.mesh, m2g_feat_sh)))
mesh_dev = Mesh(np.array(jax.devices()[:N_SHARD]), ("spatial",))
SPEC_E = (P("spatial"),) * 4
SPEC_M = (P("spatial"),) * 3

def _sharded_core(p, xg, g2m, m2g):
    ggl, gml, gval, gft = (a[0] for a in g2m)
    mgl, mml, mft = (a[0] for a in m2g)
    msg = jnp.tanh(jnp.concatenate([xg[0][ggl], gft], -1) @ p["w_enc"] + p["b_enc"])
    msg = jnp.where(gval[:, None], msg, jnp.zeros((), msg.dtype))   # mask AFTER bias
    mesh_in = jax.lax.psum(seg_sum(msg, gml, N_MESH), "spatial")    # THE collective
    mesh_h = jnp.tanh(mesh_in @ p["w_mesh"])
    dec = jnp.concatenate([mesh_h[mml], mft], -1) @ p["w_dec"] + p["b_dec"]
    return seg_sum(dec, mgl, NLAT * COLS), mesh_in

@partial(shard_map, mesh=mesh_dev,
         in_specs=(P(), P("spatial"), SPEC_E, SPEC_M),
         out_specs=(P("spatial"), P()), **SM_KW)
def sharded_forward(p, xg, g2m, m2g):
    out, mesh_in = _sharded_core(p, xg, g2m, m2g)
    return out[None], mesh_in

@partial(shard_map, mesh=mesh_dev,
         in_specs=(P(), P("spatial"), P("spatial"), SPEC_E, SPEC_M),
         out_specs=P(), **SM_KW)
def sharded_loss(p, xg, tg, g2m, m2g):
    out, _ = _sharded_core(p, xg, g2m, m2g)
    num = jnp.sum((out - tg[0]) ** 2)
    return jax.lax.psum(num, "spatial") / jnp.float32(N_GRID * D)

# 1) per-point forward + intermediate
ref_out = np.asarray(ref_forward(params, x_grid))
sh_blocks, sh_meshin = sharded_forward(params, xb, Gj, Mj)
np.testing.assert_allclose(np.asarray(sh_meshin), np.asarray(ref_mesh_agg(params, x_grid)),
                           rtol=2e-5, atol=2e-5)
np.testing.assert_allclose(from_blocks(np.asarray(sh_blocks)), ref_out, rtol=2e-5, atol=2e-5)
print("PER-POINT FORWARD + MESH-AGG INTERMEDIATE: PASS")

# 2) eager + jit grads
ref_l, ref_g = jax.value_and_grad(ref_loss)(params, x_grid, tgt)
loss_fn = lambda p: sharded_loss(p, xb, tb, Gj, Mj)
sh_l, sh_g = jax.value_and_grad(loss_fn)(params)
jit_l, jit_g = jax.jit(jax.value_and_grad(loss_fn))(params)
print(f"loss ref={ref_l:.8f} sharded={sh_l:.8f} jit={jit_l:.8f}")
ok = abs(ref_l - sh_l) < 1e-6 and abs(sh_l - jit_l) < 1e-6  # few f32 ULPs at loss~3
for k in params:
    d1 = float(jnp.abs(ref_g[k] - sh_g[k]).max()); d2 = float(jnp.abs(sh_g[k] - jit_g[k]).max())
    r = float(jnp.abs(ref_g[k]).max())
    print(f"grad[{k}]: ref-vs-sh={d1:.2e}  sh-vs-jit={d2:.2e}  (max|ref|={r:.2e})")
    ok = ok and d1 < 1e-5 * max(r, 1.0) and d2 < 1e-6
print("\nPHASE1.5 PARTITION PROBE:", "PASS" if ok else "FAIL")
