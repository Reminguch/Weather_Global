"""Construct a resolution-specific adapter and an independently initialized branch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import jax
import numpy as np
from .backbone import FrozenBackbone
from .data import PreparedStore, require_complete_experiment_data
from .native_state import NativeAdapter
from .features import KnownFeatures
from .normalization import Normalization
from .loss import WeatherLoss
from .model import make_branch, zero_memory
from .kernels import DecodedTrainer, make_optimizer
from .io import digest, read_json, sha256


@dataclass
class Runtime:
    backbone: object
    adapter: object
    store: object
    normalization: object
    known: object
    branch: object
    trainer: object
    params: object
    memory: object
    climatology: dict



def validate_data_grid(prepared, coordinates):
    """Accept float64 coordinate roundoff, preserving axes, order and units.

    Gaussian latitude calculations differ by a few ulps across CPU platforms
    (observed 1.39e-17 radians on A100 hosts). Coordinates are host float64
    metadata even though model arithmetic is FP32. Never use FP32 tolerances.
    """
    expected = {"longitude": coordinates.horizontal.longitudes,
                "latitude": coordinates.horizontal.latitudes,
                "levels": coordinates.vertical.centers}
    if set(prepared) != set(expected):
        raise ValueError("Prepared data grid keys do not match checkpoint")
    for axis, target in expected.items():
        actual, target = np.asarray(prepared[axis], dtype=np.float64), np.asarray(target, dtype=np.float64)
        if actual.ndim != 1 or actual.shape != target.shape:
            raise ValueError(f"Prepared data grid {axis} shape does not match checkpoint: "
                             f"{actual.shape} != {target.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(target).all():
            raise ValueError(f"Prepared data grid {axis} contains nonfinite coordinates")
        # Pressure levels are exact catalog values. Only angular coordinates
        # tolerate machine roundoff; 1e-14 radians is below 0.1 micrometer.
        atol = 0.0 if axis == "levels" else 1e-14
        if not np.allclose(actual, target, rtol=0.0, atol=atol):
            error = float(np.max(np.abs(actual - target)))
            raise ValueError(f"Prepared data grid {axis} does not match checkpoint: "
                             f"max_abs_difference={error:.17g}, tolerance={atol}")


def build_runtime(config, resources, *, stage):
    from .numerics import verify_runtime
    verify_runtime()
    backbone = FrozenBackbone.load(resources["checkpoint"], resources["checkpoint_sha256"])
    store = PreparedStore(resources["prepared"])
    require_complete_experiment_data(store)
    validate_data_grid(store.manifest["data_grid"], backbone.model.data_coords)
    valid = next(t for t in store.times if (t - np.timedelta64(24, "h")) in set(store.times)
                 and str(t)[:4] == str(t - np.timedelta64(24, "h"))[:4])
    inputs, forcing = store.inputs_and_forcing(backbone.model, valid)
    state = backbone.encode(inputs, forcing)
    adapter = NativeAdapter(backbone.model, state)
    normalization = Normalization.load(resources["statistics"])
    if (normalization.manifest["native_schema_id"] != adapter.identity or
            normalization.manifest["dataset_id"] != store.identity):
        raise ValueError("Normalization schema/data mismatch")
    raw_known = KnownFeatures(backbone.model)
    class NormalizedKnown:
        names = raw_known.names
        def __call__(self, state, forcing):
            return normalization.known(raw_known(state, forcing))
    known = NormalizedKnown()
    grid = adapter.grid
    branch = make_branch(config.architecture, grid.latitudes, grid.longitudes, adapter.output_size)
    params, memory = branch.init(jax.random.PRNGKey(config.seed),
                                normalization.inputs(adapter.features(state)), known(state, forcing))
    params = jax.tree_util.tree_map(lambda x: x.astype(np.float32), params)
    memory = zero_memory(memory)
    loss = WeatherLoss(backbone.model.data_coords.horizontal.latitudes,
                       backbone.model.data_coords.vertical.centers, normalization.manifest["loss_scales"])
    optimizer = make_optimizer(config.pretrain_optimizer if stage == "pretrain" else config.finetune_optimizer)
    trainer = DecodedTrainer(branch, adapter, normalization, backbone, loss, optimizer)
    climate_path = Path(resources["statistics"]).parent / normalization.manifest["climatology_file"]
    if sha256(climate_path) != normalization.manifest["climatology_sha256"]:
        raise ValueError("Training climatology has changed")
    with np.load(climate_path) as f:
        climatology = {k: f[k].copy() for k in f.files}
    return Runtime(backbone, adapter, store, normalization, known, branch, trainer, params, memory, climatology)


def runtime_identities(runtime, config, source_id, cache_id=None):
    from .numerics import POLICY
    result = {"config": config.identity, "architecture": digest(config.to_dict()["architecture"]),
              "backbone": runtime.backbone.checkpoint_sha256, "native_schema": runtime.adapter.identity,
              "normalization": runtime.normalization.identity, "dataset": runtime.store.identity,
              "source": source_id, "known_features": list(runtime.known.names), "numerical_policy": POLICY}
    if cache_id:
        result["cache"] = cache_id
    return result
