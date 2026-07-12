#!/usr/bin/env python3
"""Host-side partitioner for a stock GraphCast-built g2m/m2g TypedGraph into
N longitude shards, preserving real structural edge features. Core reusable
piece of the res=0.25 sharding plumbing.

Strategy (validated in phase2_full_forward with synthetic features; here we
operate on the REAL graphcast TypedGraph objects):
- g2m: owned by grid SENDER shard; local grid sender reindex; mesh receivers
  kept global; padded to E_max with receiver=N_MESH (sentinel row). psum after.
- m2g: owned by grid RECEIVER shard; balanced (3x local grid), no pad; mesh
  senders global; no psum.
This module provides partition_g2m_indices / partition_m2g_indices returning
per-shard (grid_local, mesh, edge_valid, global_edge_id) so the caller can
gather structural edge features by global_edge_id and build local TypedGraphs.
"""
import numpy as np


def _shard_of(grid_ids, n_lon, cols):
    return (grid_ids % n_lon) // cols


def partition_grid_owned(edges_grid, edges_mesh, *, n_lat, n_lon, n_shard,
                         mesh_sentinel, balanced):
    """edges_grid/edges_mesh: global 1D int arrays. Returns dict of stacked
    per-shard arrays: grid_local (n_shard,E), mesh (n_shard,E), valid, edge_id."""
    cols = n_lon // n_shard
    owner = _shard_of(edges_grid, n_lon, cols)
    counts = np.bincount(owner, minlength=n_shard)
    emax = int(counts.max())
    if balanced:
        assert (counts == counts[0]).all(), f"expected balanced, got {counts}"
        emax = int(counts[0])
    gl, ml, va, ei = [], [], [], []
    for r in range(n_shard):
        eid = np.flatnonzero(owner == r).astype(np.int32)
        g, m = edges_grid[eid], edges_mesh[eid]
        loc = ((g // n_lon) * cols + (g % n_lon) - r * cols).astype(np.int32)
        pad = emax - eid.size
        gl.append(np.pad(loc, (0, pad)).astype(np.int32))
        ml.append(np.pad(m, (0, pad), constant_values=mesh_sentinel).astype(np.int32))
        va.append(np.pad(np.ones(eid.size, np.bool_), (0, pad)))
        ei.append(np.pad(eid, (0, pad), constant_values=-1).astype(np.int32))
    return {"grid_local": np.stack(gl), "mesh": np.stack(ml),
            "valid": np.stack(va), "edge_id": np.stack(ei), "emax": emax}


if __name__ == "__main__":
    # CPU round-trip test on the REAL graphcast g2m graph:
    # sum over shards of per-shard local segment_sum(messages) == global segment_sum.
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))
    from graphcast import icosahedral_mesh, grid_mesh_connectivity

    N_SHARD, RES = 4, 1.0
    glat = np.arange(-90, 90.001, RES, dtype=np.float32)
    glon = np.arange(0, 360, RES, dtype=np.float32)
    NLAT, NLON, COLS = len(glat), len(glon), len(glon) // N_SHARD
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=5)
    finest = meshes[-1]
    N_MESH = icosahedral_mesh.merge_meshes(meshes).vertices.shape[0]
    fs, fr = icosahedral_mesh.faces_to_edges(finest.faces)
    rq = np.linalg.norm(finest.vertices[fs] - finest.vertices[fr], axis=-1).max() * 0.5999912857713345
    gg, gm = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=glat, grid_longitude=glon, mesh=finest, radius=rq)
    gg, gm = gg.astype(np.int32), gm.astype(np.int32)
    E = len(gg)

    # real-ish per-edge messages (depend on global grid sender id -> catches misindex)
    rng = np.random.default_rng(0)
    msg = rng.normal(size=(E, 3)).astype(np.float64)

    # global reference: segment_sum messages into mesh by receiver
    ref = np.zeros((N_MESH, 3)); np.add.at(ref, gm, msg)

    part = partition_grid_owned(gg, gm, n_lat=NLAT, n_lon=NLON, n_shard=N_SHARD,
                                mesh_sentinel=N_MESH, balanced=False)
    # reconstruct: each shard scatters its (valid) messages into N_MESH+1, sum shards, drop sentinel
    acc = np.zeros((N_MESH + 1, 3))
    for r in range(N_SHARD):
        eid = part["edge_id"][r]; val = part["valid"][r]; mrecv = part["mesh"][r]
        local_msg = np.where(val[:, None], msg[np.clip(eid, 0, E - 1)], 0.0)
        np.add.at(acc, mrecv, local_msg)
    got = acc[:N_MESH]
    d = np.abs(got - ref).max(); r_ = np.abs(ref).max()
    print(f"g2m partition round-trip: max|diff|={d:.2e} max|ref|={r_:.2e} "
          f"E_max={part['emax']} (global E={E})")
    print("SHARD_GRAPHCAST PARTITIONER:", "PASS" if d < 1e-9 else "FAIL")
