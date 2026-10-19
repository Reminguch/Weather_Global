"""Known geography, lagged surface forcing and deterministic time features."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dinosaur import horizontal_interpolation
from .native_state import state_time


class KnownFeatures:
    names = ("checkpoint_surface_geopotential_scaled", "checkpoint_land_sea_mask",
             "sin_longitude", "cos_longitude", "sin_latitude", "cos_latitude",
             "lagged_sst_scaled", "lagged_sea_ice", "sin_year", "cos_year", "sin_day", "cos_day")

    def __init__(self, model):
        self.model = model
        grid = model.model_coords.horizontal
        self.regrid = horizontal_interpolation.ConservativeRegridder(model.data_coords.horizontal, grid)
        aux = model._structure.specs.aux_features["xarray_dataset"]
        lon, lat = np.meshgrid(grid.longitudes, grid.latitudes, indexing="ij")
        geopotential = np.asarray(aux.geopotential_at_surface.transpose("longitude", "latitude"))
        mask = np.asarray(aux.land_sea_mask.transpose("longitude", "latitude"))
        self.static = jnp.stack([self.regrid(geopotential) / 1e5, self.regrid(mask),
                                jnp.sin(lon), jnp.cos(lon), jnp.sin(lat), jnp.cos(lat)], axis=-1).astype(jnp.float32)
        self.shape = tuple(grid.nodal_shape)
        self.time_unit_seconds = float(model.from_nondim_units(1.0, "s"))
        ref = np.datetime64(model._structure.specs.aux_features["reference_datetime"], "s")
        self.ref_days = float((ref - np.datetime64("1970-01-01", "s")) / np.timedelta64(1, "D"))

    def __call__(self, state, origin_forcing):
        days = state_time(state) * (self.time_unit_seconds / 86400.0) + self.ref_days
        annual, daily = 2 * jnp.pi * (days / 365.2425 % 1), 2 * jnp.pi * (days % 1)
        surface = [self.regrid(origin_forcing["sea_surface_temperature"][0]) / 300.0,
                   self.regrid(origin_forcing["sea_ice_cover"][0])]
        time = [jnp.broadcast_to(f(x), self.shape) for x in (annual, daily) for f in (jnp.sin, jnp.cos)]
        return jnp.concatenate((self.static, jnp.stack(surface + time, axis=-1)), axis=-1).astype(jnp.float32)
