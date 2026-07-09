"""Geometric grid <-> mesh projection utilities.

These are pure-numpy precomputations (done once at module construction time).
The projection uses K-nearest-neighbor aggregation with Gaussian weights on the
unit sphere. It is NOT a trained GNN — it is a fixed geometric linear operator,
intentionally cheap so that the MZ-Mamba on the mesh side can absorb more
capacity (larger hidden size, more layers).

A more faithful message-passing GNN variant can be added later; the current
form is enough to give grid points cross-communication via the (lower-res)
mesh bottleneck.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) degrees to unit 3D vectors on the sphere."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)  # [..., 3]


def _icosphere_nodes(mesh_size: int) -> np.ndarray:
    """Return mesh node xyz coordinates (unit sphere) for given icosphere level."""
    # Lazy import so users that only need the config classes don't pay the cost.
    from third_party.graphcast.graphcast import icosahedral_mesh as ico
    meshes = ico.get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_size)
    return np.asarray(meshes[-1].vertices, dtype=np.float64)  # [M, 3]


def build_grid_mesh_projections(
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    mesh_size: int,
    n_grid_neighbors: int = 6,
    n_mesh_neighbors: int = 3,
    sigma_scale: float = 1.0,
) -> Tuple[dict, int]:
    """Precompute fixed grid <-> mesh projection tensors.

    Parameters
    ----------
    lat_deg, lon_deg : 1D arrays
        Latitude (degrees, any order) and longitude (degrees) grid vectors.
        Grid is treated as the outer product ``lat x lon`` in that order.
    mesh_size : int
        Icosphere splits (3 -> 642 nodes, 4 -> 2562, 5 -> 10242).
    n_grid_neighbors : int
        Each mesh node aggregates from its K nearest grid points.
    n_mesh_neighbors : int
        Each grid point aggregates from its K nearest mesh nodes.
    sigma_scale : float
        Multiplier on the automatically-inferred Gaussian bandwidth.

    Returns
    -------
    (arrays, n_mesh) : ``(dict, int)``
        arrays contains 4 np.ndarrays suitable for passing as jax constants:
          * ``g2m_indices`` [M, K_g2m]   (int32) grid-point idx in [0, P_grid)
          * ``g2m_weights`` [M, K_g2m]   (float32) normalized so sum=1 per row
          * ``m2g_indices`` [P_grid, K_m2g] (int32) mesh-node idx in [0, M)
          * ``m2g_weights`` [P_grid, K_m2g] (float32) normalized so sum=1 per row
        ``n_mesh`` = M (number of mesh nodes).
    """
    lat_grid, lon_grid = np.meshgrid(lat_deg, lon_deg, indexing="ij")
    grid_xyz = _latlon_to_xyz(lat_grid.ravel(), lon_grid.ravel())  # [P_grid, 3]
    mesh_xyz = _icosphere_nodes(mesh_size)                          # [M, 3]

    P_grid = grid_xyz.shape[0]
    M = mesh_xyz.shape[0]

    # Cosine distance on the unit sphere: d_ij = 1 - x_i . x_j (range [0, 2]).
    # The dot product matrix is (N_i, N_j); compute once for each direction.

    # ---- Grid -> Mesh: for each mesh node find K nearest grid points --------
    dot_m_g = mesh_xyz @ grid_xyz.T      # [M, P_grid]
    cosdist_m_g = 1.0 - dot_m_g          # smaller = closer

    # argpartition then sort the top-K for each row
    g2m_idx = np.argpartition(cosdist_m_g, n_grid_neighbors - 1, axis=1)[:, :n_grid_neighbors]
    row_ids = np.arange(M)[:, None]
    g2m_dist = cosdist_m_g[row_ids, g2m_idx]
    # Sort within each row so column 0 is the closest (nicer for debugging)
    sort_order = np.argsort(g2m_dist, axis=1)
    g2m_idx = g2m_idx[row_ids, sort_order]
    g2m_dist = g2m_dist[row_ids, sort_order]

    sigma_g = np.median(g2m_dist[:, -1]) * sigma_scale
    sigma_g = max(sigma_g, 1e-6)
    g2m_w = np.exp(-g2m_dist ** 2 / (2 * sigma_g ** 2))
    g2m_w = g2m_w / g2m_w.sum(axis=1, keepdims=True)  # rows sum to 1

    # ---- Mesh -> Grid: for each grid point find K nearest mesh nodes -------
    dot_g_m = grid_xyz @ mesh_xyz.T
    cosdist_g_m = 1.0 - dot_g_m

    m2g_idx = np.argpartition(cosdist_g_m, n_mesh_neighbors - 1, axis=1)[:, :n_mesh_neighbors]
    row_ids = np.arange(P_grid)[:, None]
    m2g_dist = cosdist_g_m[row_ids, m2g_idx]
    sort_order = np.argsort(m2g_dist, axis=1)
    m2g_idx = m2g_idx[row_ids, sort_order]
    m2g_dist = m2g_dist[row_ids, sort_order]

    sigma_m = np.median(m2g_dist[:, -1]) * sigma_scale
    sigma_m = max(sigma_m, 1e-6)
    m2g_w = np.exp(-m2g_dist ** 2 / (2 * sigma_m ** 2))
    m2g_w = m2g_w / m2g_w.sum(axis=1, keepdims=True)

    arrays = dict(
        g2m_indices=g2m_idx.astype(np.int32),
        g2m_weights=g2m_w.astype(np.float32),
        m2g_indices=m2g_idx.astype(np.int32),
        m2g_weights=m2g_w.astype(np.float32),
    )
    return arrays, int(M)


def build_mesh_edges(mesh_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return bidirectional edge sender/receiver indices for the icosphere mesh.

    Edges are derived from the triangular faces of the icosphere at the given
    refinement level via ``faces_to_edges`` (each undirected edge appears twice,
    once in each direction, so message passing is symmetric).

    Parameters
    ----------
    mesh_size : int
        Icosphere refinement level. mesh_size=5 -> 10242 vertices, ~61440 edges.

    Returns
    -------
    senders, receivers : np.ndarray (int32, shape [E])
        Edge sender / receiver node indices in [0, M).
    """
    from third_party.graphcast.graphcast import icosahedral_mesh as ico
    meshes = ico.get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_size)
    final_faces = meshes[-1].faces  # [num_faces, 3]
    senders, receivers = ico.faces_to_edges(final_faces)
    return senders.astype(np.int32), receivers.astype(np.int32)
