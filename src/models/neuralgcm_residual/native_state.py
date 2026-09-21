"""Pinned v1 native schema, physical nodal features and masked modal increments."""
from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from .io import digest

NATIVE_FIELDS = (("vorticity", "1/s"), ("divergence", "1/s"),
                 ("temperature_variation", "K"), ("specific_humidity", "dimensionless"),
                 ("specific_cloud_ice_water_content", "dimensionless"),
                 ("specific_cloud_liquid_water_content", "dimensionless"),
                 ("log_surface_pressure", "dimensionless"))


def replace(value, **updates):
    if hasattr(value, "replace"):
        return value.replace(**updates)
    if hasattr(value, "_replace"):
        return value._replace(**updates)
    return dataclasses.replace(value, **updates)


def physical_state(state):
    return state.state if hasattr(state, "state") else state


def state_time(state):
    return physical_state(state).sim_time


def stop(tree):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def field_value(state, name):
    core = physical_state(state)
    return core.tracers[name] if name.startswith("specific_") else getattr(core, name)


def tree_schema(tree):
    leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    return {"structure": str(structure), "leaves": [
        {"path": jax.tree_util.keystr(path), "shape": list(np.shape(x)), "dtype": str(np.asarray(x).dtype)}
        for path, x in leaves]}


class NativeAdapter:
    def __init__(self, model, example_state):
        self.model = model
        self.grid = model.model_coords.horizontal
        self.channels = []
        for name, units in NATIVE_FIELDS:
            value = field_value(example_state, name)
            if value.ndim != 3 or tuple(value.shape[-2:]) != tuple(self.grid.modal_shape):
                raise ValueError(f"Unsupported native shape for {name}: {value.shape}")
            self.channels.append((name, units, int(value.shape[0])))
        self.output_size = sum(n for _, _, n in self.channels)
        levels = len(model.model_coords.vertical.centers)
        if self.output_size != 6 * levels + 1:
            raise ValueError("Expected six sigma fields and log surface pressure")
        self.schema = {"version": "native_modal_v1", "channels": self.channels,
                       "nodal_shape": list(self.grid.nodal_shape),
                       "modal_shape": list(self.grid.modal_shape),
                       "sigma_levels": np.asarray(model.model_coords.vertical.centers).tolist(),
                       "state_tree": tree_schema(example_state),
                       "mask_sha256": digest(np.asarray(self.grid.mask).astype(int).tolist()),
                       "units": "physical_SI_nodal_branch__nondimensional_modal_solver"}
        self.identity = digest(self.schema)

    def features(self, state):
        # Channels last, longitude first. Graph adapter transposes to latitude first.
        values = [self.model.from_nondim_units(self.grid.to_nodal(field_value(state, name)), units)
                  for name, units, _ in self.channels]
        return jnp.moveaxis(jnp.concatenate(values, axis=0), 0, -1).astype(jnp.float32)

    def apply_increment(self, state, increment):
        if increment.shape != tuple(self.grid.nodal_shape) + (self.output_size,):
            raise ValueError(f"Wrong native increment shape: {increment.shape}")
        core = physical_state(state)
        updates, tracers, offset = {}, dict(core.tracers), 0
        for name, units, count in self.channels:
            nodal = jnp.moveaxis(increment[..., offset:offset + count], -1, 0)
            modal = self.grid.to_modal(self.model.to_nondim_units(nodal, units))
            modal = jnp.where(self.grid.mask, modal, 0).astype(field_value(state, name).dtype)
            value = field_value(state, name) + modal
            if name.startswith("specific_"):
                tracers[name] = value
            else:
                updates[name] = value
            offset += count
        updates["tracers"] = tracers
        updated = replace(core, **updates)
        # No clipping/filtering/re-encoding of the existing state, including carry.
        return replace(state, state=updated) if hasattr(state, "state") else updated

    def assert_zero_identity(self, state):
        corrected = self.apply_increment(state, jnp.zeros(tuple(self.grid.nodal_shape) + (self.output_size,)))
        for a, b in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(corrected), strict=True):
            np.testing.assert_array_equal(a, b)
