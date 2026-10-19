#!/usr/bin/env python3
"""Real deterministic checkpoint smoke on official demo data, not a skill evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import resource

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resolution", choices=("2.8", "1.4"), required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()
    from src.models.neuralgcm_residual.numerics import configure_environment, POLICY
    configure_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import neuralgcm
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.native_state import NativeAdapter
    from src.models.neuralgcm_residual.features import KnownFeatures
    from src.models.neuralgcm_residual.model import make_branch, zero_memory, parameter_counts, geometry_manifest
    from src.models.neuralgcm_residual.config import Architecture
    from src.models.neuralgcm_residual.io import write_json
    from src.models.neuralgcm_residual.checks import tree_comparison

    print('devices', jax.devices(), flush=True)
    backbone = FrozenBackbone.load(args.checkpoint)
    data = neuralgcm.demo.load_data(backbone.model.data_coords).isel(time=0)
    inputs, forcing = backbone.model.data_from_xarray(data)
    started = time.perf_counter()
    state = backbone.encode(inputs, forcing)
    jax.block_until_ready(state)
    encode_seconds = time.perf_counter() - started
    adapter = NativeAdapter(backbone.model, state)
    adapter.assert_zero_identity(state)
    known = KnownFeatures(backbone.model)
    architecture = Architecture(width=128 if args.resolution == "2.8" else 256,
                                d_inner=16 if args.resolution == "2.8" else 32)
    branch = make_branch(architecture, adapter.grid.latitudes, adapter.grid.longitudes, adapter.output_size)
    print('initializing', architecture, flush=True)
    x, f = adapter.features(state), known(state, forcing)
    started = time.perf_counter()
    params, memory = branch.init(jax.random.PRNGKey(22), x, f)
    memory = zero_memory(memory)
    delta, memory = branch.apply(params, memory, jax.random.PRNGKey(23), x, f)
    jax.block_until_ready(delta)
    branch_first_seconds = time.perf_counter() - started
    np.testing.assert_array_equal(delta, 0)
    base, corrected, initial = state, state, state
    comparisons, timings, time_errors = [], [], []
    memory = zero_memory(memory)
    from src.models.neuralgcm_residual.native_state import state_time
    for lead in range(1, args.steps + 1):
        started = time.perf_counter()
        base = backbone.advance_6h(base, forcing)
        delta, memory = branch.apply(params, memory, jax.random.PRNGKey(lead),
                                     adapter.features(corrected), known(corrected, forcing))
        corrected = adapter.apply_increment(backbone.advance_6h(corrected, forcing), delta)
        jax.block_until_ready(corrected)
        tree_comparison(base, corrected, exact=True)
        backbone.assert_time(initial, corrected, lead)
        time_errors.append(float(state_time(corrected) - state_time(initial)) - lead * backbone.sim_step)
        if lead in (1, 20, 40):
            comparisons.append({"lead": lead, **tree_comparison(backbone.decode(base, forcing),
                                                                backbone.decode(corrected, forcing), exact=True)})
        timings.append(time.perf_counter() - started)
        if lead in (1, 20, 40): print('zero residual step',lead,'passed',flush=True)
    print('checking real decoder gradient',flush=True)
    def loss(increment):
        decoded = backbone.decode(adapter.apply_increment(base, increment), forcing)
        return jnp.mean(decoded['temperature'] ** 2)
    started = time.perf_counter()
    gradient = jax.grad(loss)(jnp.zeros_like(x))
    jax.block_until_ready(gradient)
    gradient_seconds = time.perf_counter() - started
    if not bool(jnp.all(jnp.isfinite(gradient))) or float(jnp.linalg.norm(gradient)) == 0:
        raise AssertionError('Frozen decoder input gradient is missing/nonfinite')
    report = {"kind": "official_demo_smoke_only", "not_a_production_gate": True,
              "resolution": args.resolution, "backbone": backbone.manifest(initial),
              "native_schema": adapter.schema, "architecture": architecture.__dict__,
              "numerical_policy": POLICY,
              "geometry": geometry_manifest(architecture, adapter.grid.latitudes, adapter.grid.longitudes, adapter.output_size),
              "devices": [str(d) for d in jax.devices()], "encode_first_seconds": encode_seconds,
              "branch_first_seconds": branch_first_seconds, "zero_residual_comparisons": comparisons,
              "paired_step_seconds": timings, "native_time_errors": time_errors,
              "decoder_gradient_first_seconds": gradient_seconds,
              "decoder_input_gradient_norm": float(jnp.linalg.norm(gradient)),
              "peak_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "gpu_allocator_stats": [d.memory_stats() for d in jax.devices()],
              **parameter_counts(params, memory), "passed": True}
    write_json(args.output, report, immutable=True)
    print('saved',args.output,flush=True)


if __name__ == '__main__':
    main()
