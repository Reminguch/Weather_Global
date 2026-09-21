"""Train-only, area-weighted streaming moments with explicit positive floors."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from .config import FIELDS
from .io import atomic_bytes, digest, read_json, sha256, write_json
from .loss import gaussian_weights

LOSS_FLOORS = {"temperature": 0.1, "geopotential": 1.0,
               "u_component_of_wind": 0.1, "v_component_of_wind": 0.1,
               "specific_humidity": 1e-7, "specific_cloud_ice_water_content": 1e-9,
               "specific_cloud_liquid_water_content": 1e-9}
NATIVE_FLOORS = {"vorticity": 1e-8, "divergence": 1e-8, "temperature_variation": 0.1,
                 "specific_humidity": 1e-7, "specific_cloud_ice_water_content": 1e-9,
                 "specific_cloud_liquid_water_content": 1e-9, "log_surface_pressure": 1e-4}


class Moments:
    def __init__(self, channels, latitude):
        self.weight = gaussian_weights(latitude).astype(np.float64)[None, :, None]
        self.n = 0
        self.mean = np.zeros(channels, np.float64)
        self.m2 = np.zeros(channels, np.float64)

    def add(self, value):
        x = np.asarray(value, dtype=np.float64)
        if x.ndim != 3 or not np.all(np.isfinite(x)):
            raise ValueError("Statistics require finite [longitude,latitude,channel] values")
        mean = (x * self.weight).sum(axis=1).mean(axis=0)
        var = ((x - mean) ** 2 * self.weight).sum(axis=1).mean(axis=0)
        self.n += 1
        delta = mean - self.mean
        self.mean += delta / self.n
        self.m2 += var + delta * (mean - self.mean)

    def std(self, floor):
        if self.n == 0:
            raise ValueError("No training observations for normalization")
        return np.maximum(np.sqrt(self.m2 / self.n), floor).astype(np.float32)


class Normalization:
    def __init__(self, manifest):
        self.manifest = manifest
        if manifest["split"] != "train" or manifest["version"] != "normalization_v1":
            raise ValueError("Statistics must be fitted exclusively on training data")
        self.mean = jnp.asarray(manifest["input_mean"], jnp.float32)
        self.scale = jnp.asarray(manifest["input_scale"], jnp.float32)
        self.correction_scale = jnp.asarray(manifest["correction_scale"], jnp.float32)
        self.known_mean = jnp.asarray(manifest.get("known_mean", []), jnp.float32)
        self.known_scale = jnp.asarray(manifest.get("known_scale", []), jnp.float32)
        if not np.all(np.asarray(self.scale) > 0) or not np.all(np.asarray(self.correction_scale) > 0):
            raise ValueError("Nonpositive normalization scales")
        self.identity = digest(manifest)

    def inputs(self, native):
        return (native.astype(jnp.float32) - self.mean) / self.scale

    def increment(self, normalized):
        return normalized.astype(jnp.float32) * self.correction_scale

    def known(self, values):
        return (values.astype(jnp.float32) - self.known_mean) / self.known_scale

    @classmethod
    def load(cls, path):
        return cls(read_json(path))


def fit_statistics(records, adapter, data_latitude, *, dataset_id, output):
    """Records provide independent encoded truth, baseline and next encoded truth.

    Native change standard deviations set increment units, not a regression loss.
    Targets/current truth set the decoded six-hour change loss scales.
    """
    native_lat = adapter.grid.latitudes
    inputs = Moments(adapter.output_size, native_lat)
    corrections = Moments(adapter.output_size, native_lat)
    loss = {k: Moments(len(adapter.model.data_coords.vertical.centers), data_latitude) for k in FIELDS}
    from .features import KnownFeatures
    known_features = KnownFeatures(adapter.model)
    known = Moments(len(known_features.names), native_lat)
    origins, climatology = [], {}
    for record in records:
        if record["split"] != "train":
            raise ValueError("Validation/test data cannot fit normalization")
        x = adapter.features(record["origin_state"])
        inputs.add(x)
        known.add(known_features(record["origin_state"], record["forcing"]))
        corrections.add(adapter.features(record["next_truth_state"]) - adapter.features(record["baseline_state"]))
        for k in FIELDS:
            loss[k].add(np.moveaxis(record["target"][k] - record["truth"][k], 0, -1))
            target = np.asarray(record["target"][k], dtype=np.float64)
            if k not in climatology:
                climatology[k] = np.zeros_like(target)
            climatology[k] += (target - climatology[k]) / (len(origins) + 1)
        origins.append(record["origin"])
    floors = np.concatenate([np.full(n, NATIVE_FLOORS[k]) for k, _, n in adapter.channels])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{k: v.astype(np.float32) for k, v in climatology.items()})
    climate_path = Path(output).with_suffix(".climatology.npz")
    atomic_bytes(climate_path, buffer.getvalue())
    manifest = {"version": "normalization_v1", "split": "train", "dataset_id": dataset_id,
                "native_schema_id": adapter.identity, "origin_count": len(origins),
                "origins_sha256": digest(origins), "input_mean": inputs.mean.tolist(),
                "input_scale": inputs.std(floors).tolist(),
                "known_mean": known.mean.tolist(), "known_scale": known.std(1e-4).tolist(),
                "known_feature_names": list(known_features.names), "known_scale_floor": 1e-4,
                "correction_scale": corrections.std(floors).tolist(),
                "loss_scales": {k: loss[k].std(LOSS_FLOORS[k]).tolist() for k in FIELDS},
                "native_floors": NATIVE_FLOORS, "loss_floors": LOSS_FLOORS,
                "climatology_file": climate_path.name, "climatology_sha256": sha256(climate_path),
                "correction_scale_definition": "std(E(truth_next)-B6h(E(truth)))_physical_nodal"}
    write_json(output, manifest, immutable=True)
    return Normalization(manifest)
