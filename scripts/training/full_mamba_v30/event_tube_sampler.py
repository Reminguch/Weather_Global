"""v30 Event tube sampler — SKELETON.

Returns metadata + truth tube patch ONLY. The GraphCast model still runs on the
FULL global grid; the event tube is consumed only by the target_encoder inside
the contrastive loss.

Sampler responsibilities:
  1. Choose anchor times where SOME event is happening (anomaly z / wind / grad
     above threshold) — "event-enriched sampling", not loss reward.
  2. Choose a patch center (lat_idx, lon_idx) within that anchor's grid.
  3. Build a truth tube: [T_win, P, P, C_vars] of ERA5 truth values around that
     anchor / patch.
  4. Build climate-matched negative tubes: same month, same lat band, different
     year or different anchor.

What this sampler does NOT do:
  - Doesn't replace the model's global input.
  - Doesn't write any loss / reward.
  - Doesn't gate-mask anything in the model.

Mining criteria (Phase-0 default):
  Anchor is "event-rich" if ANY of:
    a) |z(2m_T, last 6h, anchor patch)| > 2.5
    b) wind_speed_z(10m, last 6h, anchor patch) > p95  (p95 ≈ z>1.64)
    c) spatial grad norm of 2m_T > p95 in patch
  These are SAMPLING priors only.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EventTubeBatch:
    """One batch of event-anchored truth tubes.

    Fields are numpy arrays — pin to device at train_step boundary.

    anchor_time_idx : [B]       integer index into the dataset's time axis
    center_lat_idx  : [B]       grid lat index of patch center
    center_lon_idx  : [B]       grid lon index of patch center
    month           : [B]       1..12 (used for climate-matched negs)
    lat_band        : [B]       binned latitude band id (0..5)
    truth_tube      : [B, T_win, P, P, C_vars] float32  ERA5 truth window
    variables       : list[str] variable names (length C_vars)
    patch_size      : int
    time_window     : int
    """
    anchor_time_idx: np.ndarray
    center_lat_idx: np.ndarray
    center_lon_idx: np.ndarray
    month: np.ndarray
    lat_band: np.ndarray
    truth_tube: np.ndarray
    variables: list
    patch_size: int
    time_window: int


# ---------------------------------------------------------------------------
# Public API (skeleton — Phase-0 wiring TBD)
# ---------------------------------------------------------------------------

DEFAULT_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)


def sample_event_anchors(
    ds: xr.Dataset,
    n_anchors: int,
    *,
    rng: np.random.Generator,
    criterion: str = "anomaly_z",
    threshold: float = 2.5,
    patch_size: int = 16,
    variables: Sequence[str] = DEFAULT_VARIABLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample anchor (time, lat, lon) triples from event-rich regions.

    Returns:
        anchor_time_idx : [n_anchors]
        center_lat_idx  : [n_anchors]
        center_lon_idx  : [n_anchors]

    NOTE (skeleton): Phase-0 implementation just picks uniformly at random
    among all (t, lat, lon) cells. The event-criterion path is stubbed and
    will be implemented once the head + loss wiring is verified.
    """
    n_time = ds.sizes.get("time", ds.sizes.get("anchor_time", 0))
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    half = patch_size // 2
    lat_range_lo = half
    lat_range_hi = n_lat - half

    t_idx = rng.integers(0, n_time, size=n_anchors)
    lat_idx = rng.integers(lat_range_lo, lat_range_hi, size=n_anchors)
    lon_idx = rng.integers(0, n_lon, size=n_anchors)   # lon wraps circularly
    return t_idx.astype(np.int64), lat_idx.astype(np.int64), lon_idx.astype(np.int64)


def build_truth_tube(
    ds: xr.Dataset,
    anchor_time_idx: np.ndarray,
    center_lat_idx: np.ndarray,
    center_lon_idx: np.ndarray,
    *,
    time_window: int = 4,        # 4 × 6h = 24h
    patch_size: int = 16,
    variables: Sequence[str] = DEFAULT_VARIABLES,
) -> np.ndarray:
    """Slice [B, T_win, P, P, C_vars] ERA5 truth tubes from the zarr.

    Tube window for anchor t = ds.isel(time=t..t+T_win-1).
    Patch = (center_lat_idx ± P/2, center_lon_idx ± P/2 with lon wrap).

    Returns float32 numpy array; caller is responsible for device transfer.

    SKELETON: this version walks anchors one at a time and uses .isel + numpy.
    A vectorized batched gather is a tiny optimization for later.
    """
    n = len(anchor_time_idx)
    half = patch_size // 2
    n_lon = ds.sizes["lon"]
    tubes = []
    for i in range(n):
        t0 = int(anchor_time_idx[i])
        lat_c = int(center_lat_idx[i])
        lon_c = int(center_lon_idx[i])
        lat_inds = np.arange(lat_c - half, lat_c - half + patch_size)
        lat_inds = np.clip(lat_inds, 0, ds.sizes["lat"] - 1)
        lon_inds = (np.arange(lon_c - half, lon_c - half + patch_size)) % n_lon

        # [T_win, P, P, C]
        per_var = []
        for vname in variables:
            da = ds[vname].isel(time=slice(t0, t0 + time_window))
            if "level" in da.dims:
                # Surface vars only for v0 — TODO: handle pressure-level fields.
                continue
            arr = da.isel(lat=xr.DataArray(lat_inds, dims="lat"),
                          lon=xr.DataArray(lon_inds, dims="lon")).values
            # arr shape: [T_win, P, P]
            per_var.append(arr.astype(np.float32))
        # Stack along channels axis
        tube = np.stack(per_var, axis=-1)   # [T_win, P, P, C_used]
        tubes.append(tube)
    return np.stack(tubes, axis=0)          # [B, T_win, P, P, C]


def sample_climate_matched_negatives(
    ds: xr.Dataset,
    pos_anchor_time_idx: np.ndarray,
    *,
    rng: np.random.Generator,
    n_neg_per_pos: int = 0,
    patch_size: int = 16,
) -> Optional[np.ndarray]:
    """Sample climate-matched negative truth tubes.

    SKELETON: not used by the v0 CLIP-style InfoNCE which uses in-batch other-i
    diagonal negatives. Wired here so Phase-1+ can plug it in without refactor.

    If n_neg_per_pos == 0, returns None.
    """
    if n_neg_per_pos == 0:
        return None
    raise NotImplementedError(
        "climate-matched negative sampling: implement in Phase 1+")


# ---------------------------------------------------------------------------
# B0: per-step climate-matched sampler (used inside train_mz_v20.py main loop)
# ---------------------------------------------------------------------------

class PerStepEventSampler:
    """Per-training-step climate-matched positive+negatives sampler.

    Wraps a PreparedArrayStore. For each training step, given the positive
    anchor's store-time index, returns:
        event_batch = {
            "truth_tube_pos": [1, T_win, P, P, C],
            "truth_tube_neg": [1, Nneg, T_win, P, P, C],
            "center_lat_idx": [1],
            "center_lon_idx": [1],
        }

    Negatives are drawn from `train_split_store_indices` with:
      - same month as pos anchor
      - |anchor_time - pos_time| >= min_time_gap_days
      - same lat band (approximation: same center_lat_idx as pos for B0)
      - lon >= patch_size away from pos_lon (or wraps to be far)

    T_win = 1 in B0 (just the target frame). Extensible later.
    """

    LAT_BAND_EDGES = np.array([-60, -30, 0, 30, 60], dtype=np.float32)

    def __init__(self, store, anchor_indices, split_indices, input_steps, dt,
                 task_cfg, *, patch_size=16, n_neg=4, time_window=1,
                 min_time_gap_days=7,
                 surface_vars=DEFAULT_VARIABLES,
                 mean_by_level=None, stddev_by_level=None):
        self.store = store
        self.anchor_indices = np.asarray(anchor_indices)
        self.split_indices = np.asarray(split_indices)   # into anchor_indices
        self.input_steps = input_steps
        self.dt = dt
        self.task_cfg = task_cfg
        self.patch_size = int(patch_size)
        self.n_neg = int(n_neg)
        self.time_window = int(time_window)
        self.min_time_gap_ns = int(min_time_gap_days) * 24 * 3600 * 1_000_000_000
        self.surface_vars = [v for v in surface_vars]
        # Per-var z-score normalization stats. If missing, tubes are not
        # normalized (raw physical units → target_encoder collapses because
        # 2m_T ~ 280 K and mslp ~ 1e5 Pa dominate). ALWAYS pass these.
        self._mean = mean_by_level
        self._std = stddev_by_level
        # Pre-extract scalar (mean, std) per surface var so _fetch_tube can be
        # fast (no xarray lookup per call).
        self._var_mean = {}
        self._var_std = {}
        if mean_by_level is not None and stddev_by_level is not None:
            for vname in self.surface_vars:
                if vname in mean_by_level.data_vars:
                    self._var_mean[vname] = float(np.asarray(mean_by_level[vname].values))
                    self._var_std[vname]  = float(np.asarray(stddev_by_level[vname].values))
                    # Guard against std == 0.
                    if self._var_std[vname] < 1e-8:
                        self._var_std[vname] = 1.0

        # Store dims (PreparedArrayStore exposes coords via .coords and .sizes;
        # there is no store.lat / store.lon attribute).
        self.n_lat = int(store.sizes["lat"])
        self.n_lon = int(store.sizes["lon"])
        self.lat_vals = np.asarray(store.coords["lat"])

        # Precompute month + lat_band for candidate pool.
        import pandas as pd
        store_times = np.asarray(store.time.values, dtype="datetime64[ns]")
        # pool_store_idx: for each split entry, the absolute store-time index.
        pool_store_idx = self.anchor_indices[self.split_indices]
        self.pool_store_idx = pool_store_idx.astype(np.int64)
        self.pool_time_ns = store_times[pool_store_idx].astype("i8")
        self.pool_month = pd.DatetimeIndex(store_times[pool_store_idx]).month.values.astype(np.int32)

    def _fetch_tube(self, store_idx, center_lat_idx, center_lon_idx):
        """Return [T_win, P, P, C] tube for one anchor + one patch."""
        # NB: input_steps/target_steps must match task_cfg — reuse the
        # standard batch builder to guarantee correct offsets.
        _, tgt, _ = self.store.build_batch_from_indices(
            indices=[int(store_idx)],
            input_steps=self.input_steps,
            target_steps=1,
            task_cfg=self.task_cfg,
            dt=self.dt,
        )
        P = self.patch_size
        half = P // 2
        lat_inds = np.arange(center_lat_idx - half, center_lat_idx - half + P)
        lat_inds = np.clip(lat_inds, 0, self.n_lat - 1)
        lon_inds = (np.arange(center_lon_idx - half, center_lon_idx - half + P)) % self.n_lon

        per_var = []
        for vname in self.surface_vars:
            if vname not in tgt.data_vars: continue
            da = tgt[vname]
            if "level" in da.dims: continue     # skip pressure-level vars
            arr = da.isel(lat=xr.DataArray(lat_inds, dims="lat"),
                          lon=xr.DataArray(lon_inds, dims="lon")).values
            # arr shape typically [batch=1, time=1, P, P] — squeeze batch
            arr = np.squeeze(arr, axis=0).astype(np.float32)
            # Per-var z-score normalization: (x - mean) / std.
            # CRITICAL for target_encoder: raw physical units span 5+ orders of
            # magnitude across vars (mslp ~1e5 Pa vs wind ~10 m/s). Without
            # normalization, encoder collapses.
            if vname in self._var_mean:
                arr = (arr - self._var_mean[vname]) / self._var_std[vname]
            per_var.append(arr)
        if len(per_var) == 0:
            raise RuntimeError(f"No surface vars found for anchor {store_idx}")
        return np.stack(per_var, axis=-1)   # [T_win, P, P, C]

    def _lat_band(self, lat_idx):
        return int(np.digitize(self.lat_vals[lat_idx], self.LAT_BAND_EDGES))

    def _sample_neg_indices(self, pos_store_idx, pos_month, pos_lat_band, rng):
        """Sample Nneg entries from self.pool_store_idx satisfying:
             same month  &&  |dt| >= min_time_gap.
           Falls back progressively if the filtered pool is too small.
        """
        month_mask = (self.pool_month == pos_month)
        pos_t_ns = int(self.pool_time_ns[np.where(self.pool_store_idx == pos_store_idx)[0][0]]) \
            if (pos_store_idx == self.pool_store_idx).any() \
            else None
        if pos_t_ns is None:
            # pos anchor may not be in split (e.g., last anchor of BPTT chunk);
            # look up in store times directly.
            import pandas as pd
            pos_t_ns = int(
                pd.Timestamp(self.store.time.values[pos_store_idx]).value)
        time_ok = np.abs(self.pool_time_ns - pos_t_ns) >= self.min_time_gap_ns
        mask = month_mask & time_ok
        cand = np.where(mask)[0]
        if len(cand) < self.n_neg:
            # relax time gap
            cand = np.where(month_mask)[0]
        if len(cand) < self.n_neg:
            # relax month
            cand = np.arange(len(self.pool_store_idx))
        return rng.choice(cand, size=self.n_neg, replace=False)

    def sample(self, pos_store_idx, rng):
        """Build one event_batch (B=1) for a given pos anchor's store index."""
        import pandas as pd
        P = self.patch_size
        # Positive patch center — sample random within valid range.
        pos_lat = int(rng.integers(P, self.n_lat - P))
        pos_lon = int(rng.integers(0, self.n_lon))

        pos_tube = self._fetch_tube(pos_store_idx, pos_lat, pos_lon)  # [T,P,P,C]

        pos_month = int(pd.Timestamp(self.store.time.values[pos_store_idx]).month)
        pos_lat_band = self._lat_band(pos_lat)
        neg_pool_indices = self._sample_neg_indices(
            pos_store_idx, pos_month, pos_lat_band, rng)
        neg_store_indices = self.pool_store_idx[neg_pool_indices]

        neg_tubes = []
        for neg_t_idx in neg_store_indices:
            # Same-lat-band negatives: pick a lat_idx whose lat_band matches pos.
            # For B0 simplicity, keep the same center_lat_idx as pos (strictest
            # match — same lat exactly, so we ONLY test event structure, not
            # zonal climatology). Lon random but ≥P away from pos_lon (circular).
            neg_lat = pos_lat
            for _ in range(10):
                cand_lon = int(rng.integers(0, self.n_lon))
                d = abs(cand_lon - pos_lon)
                if d >= P and (self.n_lon - d) >= P:
                    neg_lon = cand_lon
                    break
            else:
                neg_lon = (pos_lon + self.n_lon // 2) % self.n_lon
            neg_tubes.append(self._fetch_tube(int(neg_t_idx), neg_lat, neg_lon))

        # Stack + add batch dim.
        pos = pos_tube[None, ...]                    # [1, T, P, P, C]
        neg = np.stack(neg_tubes, axis=0)[None, ...]  # [1, Nneg, T, P, P, C]

        return dict(
            truth_tube_pos=pos.astype(np.float32),
            truth_tube_neg=neg.astype(np.float32),
            center_lat_idx=np.asarray([pos_lat], dtype=np.int32),
            center_lon_idx=np.asarray([pos_lon], dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# Convenience: build a full EventTubeBatch
# ---------------------------------------------------------------------------

def build_event_tube_batch(
    ds: xr.Dataset,
    n_anchors: int,
    *,
    rng: np.random.Generator,
    time_window: int = 4,
    patch_size: int = 16,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    criterion: str = "uniform",
) -> EventTubeBatch:
    """One-shot: sample + build, returns EventTubeBatch."""
    t_idx, lat_idx, lon_idx = sample_event_anchors(
        ds, n_anchors, rng=rng,
        criterion=criterion, patch_size=patch_size,
        variables=variables)
    truth_tube = build_truth_tube(
        ds, t_idx, lat_idx, lon_idx,
        time_window=time_window, patch_size=patch_size,
        variables=variables)
    # Derive metadata
    times = ds["time"].isel(time=t_idx).values if "time" in ds.coords else None
    if times is not None:
        import pandas as pd
        months = pd.DatetimeIndex(times).month.values
    else:
        months = np.zeros_like(t_idx)
    lat_vals = ds["lat"].isel(lat=lat_idx).values
    lat_band = np.digitize(lat_vals, np.array([-60, -30, 0, 30, 60])).astype(np.int64)

    # variables list reflects actually-included surface vars
    used_vars = [v for v in variables if "level" not in ds[v].dims]
    return EventTubeBatch(
        anchor_time_idx=t_idx,
        center_lat_idx=lat_idx,
        center_lon_idx=lon_idx,
        month=months.astype(np.int64),
        lat_band=lat_band,
        truth_tube=truth_tube,
        variables=used_vars,
        patch_size=patch_size,
        time_window=time_window,
    )
