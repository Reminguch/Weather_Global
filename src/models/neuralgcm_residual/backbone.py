"""Frozen complete NeuralGCM v1.2.2, including learned encoder/physics/decoder."""
from __future__ import annotations

from pathlib import Path
import pickle
import urllib.request
import jax
import jax.numpy as jnp
import numpy as np
from .io import sha256, versions, write_json, digest
from .native_state import state_time, tree_schema

UPSTREAM_COMMIT = "c91d2007ca37a4c0842f3ff83256d32e1ec30815"


def download_checkpoint(resolution, root):
    if resolution not in (2.8, 1.4):
        raise ValueError("Only the two initial deterministic backbones are supported")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"deterministic_{str(resolution).replace('.', '_')}_deg.pkl"
    url = "https://storage.googleapis.com/neuralgcm/models/v1/" + name
    temporary = root / (name + ".partial")
    urllib.request.urlretrieve(url, temporary)
    checksum = sha256(temporary)
    target = root / checksum
    target.mkdir(exist_ok=True)
    final = target / name
    if final.exists():
        if sha256(final) != checksum:
            raise ValueError("Existing checkpoint is corrupt")
        temporary.unlink()
    else:
        temporary.replace(final)
    write_json(target / "manifest.json", {"sha256": checksum, "file": name, "url": url,
               "resolution": resolution, "license": "CC-BY-SA-4.0",
               "attribution": "Google LLC, NeuralGCM",
               "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
               "catalog": "https://github.com/neuralgcm/neuralgcm/blob/" + UPSTREAM_COMMIT + "/docs/checkpoints.md"},
               immutable=True)
    return final


class FrozenBackbone:
    def __init__(self, model, checkpoint_sha256):
        self.model = model
        self.checkpoint_sha256 = checkpoint_sha256
        ratio = np.timedelta64(6, "h") / model.timestep
        if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
            raise ValueError("Six hours must be an integer number of internal timesteps")
        self.internal_steps = int(round(ratio))
        self.sim_step = float(model.to_nondim_units(21600.0, "s"))
        # Parameters are closed-over constants. Only branch parameters are optimized.
        self.encode = model.encode
        self.decode = model.decode  # Do NOT stop_gradient the decoder input.
        self._advance = jax.jit(lambda s, f: jax.lax.fori_loop(
            0, self.internal_steps, lambda _, carry: model.advance(carry, f), s))

    @classmethod
    def load(cls, path, expected_sha256=None):
        import neuralgcm
        if neuralgcm.__version__ != "1.2.2":
            raise ValueError("This adapter pins neuralgcm==1.2.2")
        actual = sha256(path)
        if expected_sha256 and expected_sha256 != actual:
            raise ValueError("Backbone checkpoint identity differs")
        # Only locally downloaded, hashed official checkpoints are supported.
        with Path(path).open("rb") as f:
            model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(f))
        return cls(model, actual)

    def advance_6h(self, state, forcing):
        return self._advance(state, forcing)

    def assert_time(self, before, after, steps=1):
        expected = float(state_time(before)) + steps * self.sim_step
        np.testing.assert_allclose(float(state_time(after)), expected, rtol=2e-6, atol=2e-5)

    def manifest(self, example_state=None):
        def coords(c):
            return {"nodal_shape": list(c.horizontal.nodal_shape),
                    "modal_shape": list(c.horizontal.modal_shape),
                    "longitude": np.rad2deg(c.horizontal.longitudes).tolist(),
                    "latitude": np.rad2deg(c.horizontal.latitudes).tolist(),
                    "vertical_levels": np.asarray(c.vertical.centers).tolist()}
        value = {"checkpoint_sha256": self.checkpoint_sha256, "upstream_commit": UPSTREAM_COMMIT,
                 "libraries": versions(), "data_coords": coords(self.model.data_coords),
                 "model_coords": coords(self.model.model_coords),
                 "internal_timestep_seconds": float(self.model.timestep / np.timedelta64(1, "s")),
                 "internal_steps_per_6h": self.internal_steps,
                 "input_variables": self.model.input_variables,
                 "forcing_variables": self.model.forcing_variables,
                 "spectral_mask": np.asarray(self.model.model_coords.horizontal.mask).astype(int).tolist()}
        if example_state is not None:
            value["state_tree"] = tree_schema(example_state)
        value["identity"] = digest(value)
        return value
