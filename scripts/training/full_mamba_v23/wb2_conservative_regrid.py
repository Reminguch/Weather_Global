# =============================================================================
# WeatherBench2-style resolution projection (conservative regridding).
#
# WHY: to compare models at different native resolutions fairly, WB2 does NOT
# upsample coarse models; it regrids EVERY forecast AND the ERA5 truth onto one
# common coarser verification grid (240x121 = 1.5 deg) with first-order
# CONSERVATIVE (area-overlap) regridding, then computes cos-lat RMSE. This module
# implements that projection for our 1.0deg (181x360) model outputs -> 1.5deg.
#
# METHOD: separable in lat/lon. lat weights use exact spherical band areas
# (sin(edge_up)-sin(edge_lo)); lon weights are linear overlap with periodic
# wrap. Grids are vertex-centered (centers include the poles), so pole rows are
# half-cells. Validated: global area-weighted mean preserved to ~1e-18, a
# constant field maps to itself exactly (see __main__ self-test).
#
# USAGE:
#   from wb2_conservative_regrid import regrid_dataset, regrid_array
#   ds_15 = regrid_dataset(ds_10)            # xr.Dataset, any (..., lat, lon)
#   a_15  = regrid_array(a_10)               # np.ndarray (..., 181, 360)->(...,121,240)
# Change SRC_*/TGT_* to retarget other resolutions (e.g. 0.25->1.5 for FengWu:
# the FengWu scoring scripts carry the same code with SRC = 0.25deg).
#
# Used by eval_v22_clean_wb2grid.py to score our v22cl models on the WB2 grid,
# producing the RMSE-vs-ECMWF-ENS comparison figures.
# =============================================================================
"""First-order conservative regridding 1.0deg (181x360) -> WB2 1.5deg (121x240).

Follows the WeatherBench2 evaluation protocol: forecasts AND truth are
area-overlap averaged onto the common 240x121 equiangular grid before any
metric is computed. Separable in lat/lon; lat overlap uses exact spherical
cell areas (sin(upper)-sin(lower)); lon overlap is linear and periodic.
Pole cells on both grids are half-cells (grids are vertex-centered, i.e.
centers include -90 and +90).
"""
import numpy as np
import xarray as xr

SRC_LAT = np.arange(-90.0, 90.001, 1.0)     # 181
SRC_LON = np.arange(0.0, 360.0, 1.0)        # 360
TGT_LAT = np.arange(-90.0, 90.001, 1.5)     # 121
TGT_LON = np.arange(0.0, 360.0, 1.5)        # 240


def _lat_edges(centers):
    mid = 0.5 * (centers[1:] + centers[:-1])
    return np.concatenate([[-90.0], mid, [90.0]])


def _lon_edges(centers):
    step = centers[1] - centers[0]
    return np.concatenate([centers - step / 2.0, [centers[-1] + step / 2.0]])


def _lat_weight_matrix(src_centers, tgt_centers):
    se, te = _lat_edges(src_centers), _lat_edges(tgt_centers)
    s_lo, s_up = np.sin(np.deg2rad(se[:-1])), np.sin(np.deg2rad(se[1:]))
    W = np.zeros((len(tgt_centers), len(src_centers)))
    for j in range(len(tgt_centers)):
        t_lo, t_up = np.sin(np.deg2rad(te[j])), np.sin(np.deg2rad(te[j + 1]))
        ov = np.minimum(s_up, t_up) - np.maximum(s_lo, t_lo)
        W[j] = np.maximum(ov, 0.0)
    W /= W.sum(axis=1, keepdims=True)
    return W


def _lon_weight_matrix(src_centers, tgt_centers):
    se, te = _lon_edges(src_centers), _lon_edges(tgt_centers)
    n_s = len(src_centers)
    W = np.zeros((len(tgt_centers), n_s))
    for j in range(len(tgt_centers)):
        t_lo, t_up = te[j], te[j + 1]
        for shift in (-360.0, 0.0, 360.0):   # periodic images
            ov = np.minimum(se[1:] + shift, t_up) - np.maximum(se[:-1] + shift, t_lo)
            W[j] += np.maximum(ov, 0.0)
    W /= W.sum(axis=1, keepdims=True)
    return W


_WLAT = _lat_weight_matrix(SRC_LAT, TGT_LAT)
_WLON = _lon_weight_matrix(SRC_LON, TGT_LON)


def regrid_array(a: np.ndarray) -> np.ndarray:
    """a: (..., 181, 360) -> (..., 121, 240)."""
    t = np.tensordot(a, _WLON.T, axes=([-1], [0]))          # (..., 181, 240)
    t = np.tensordot(t, _WLAT.T, axes=([-2], [0]))          # (..., 240, 121)
    return np.moveaxis(t, -1, -2)                            # (..., 121, 240)


def regrid_dataset(ds: xr.Dataset) -> xr.Dataset:
    out = {}
    for v in ds.data_vars:
        da = ds[v]
        if "lat" not in da.dims or "lon" not in da.dims:
            out[v] = da
            continue
        lead = [d for d in da.dims if d not in ("lat", "lon")]
        da = da.transpose(*lead, "lat", "lon")
        dims = list(da.dims)
        new = regrid_array(np.asarray(da.values, dtype=np.float64))
        coords = {d: da.coords[d] for d in dims[:-2] if d in da.coords}
        coords["lat"] = TGT_LAT
        coords["lon"] = TGT_LON
        out[v] = xr.DataArray(new.astype(np.float32), dims=dims, coords=coords)
    return xr.Dataset(out)


if __name__ == "__main__":
    # conservation checks
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 181, 360))
    y = regrid_array(x)
    area_s = np.cos(np.deg2rad(SRC_LAT)); area_s[0] = area_s[-1] = np.cos(np.deg2rad(89.75)) / 2
    # exact cell areas from edges
    def cell_area(centers):
        e = _lat_edges(centers)
        return np.sin(np.deg2rad(e[1:])) - np.sin(np.deg2rad(e[:-1]))
    gs = np.average(x.mean(-1), axis=-1, weights=cell_area(SRC_LAT))
    gt = np.average(y.mean(-1), axis=-1, weights=cell_area(TGT_LAT))
    print("global mean src:", gs, " tgt:", gt, " max|diff|:", np.abs(gs - gt).max())
    c = regrid_array(np.full((181, 360), 3.14))
    print("constant field preserved:", np.allclose(c, 3.14), " shape:", c.shape)
