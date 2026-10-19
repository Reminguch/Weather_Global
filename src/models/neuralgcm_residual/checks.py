"""Measured real-model correctness gates. Toy tests cannot write production gates."""
from __future__ import annotations

import copy
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from .cache import live_record
from .model import zero_memory, parameter_counts
from .native_state import stop
from .io import digest, write_json

TOLERANCES = {"loss_rtol": 1e-5, "tree_rtol": 1e-5, "tree_atol": 1e-6,
              "gradient_cosine_min": .99999}


def tree_comparison(a, b, *, exact=False):
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        raise AssertionError("State tree structure differs")
    errors, dots, aa, bb = [], 0., 0., 0.
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if exact:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=TOLERANCES["tree_rtol"], atol=TOLERANCES["tree_atol"])
        errors.append(float(np.max(np.abs(x - y), initial=0)))
        dots += float(np.vdot(x.astype(np.float64), y.astype(np.float64)))
        aa += float(np.vdot(x.astype(np.float64), x.astype(np.float64)))
        bb += float(np.vdot(y.astype(np.float64), y.astype(np.float64)))
    return {"max_abs": max(errors, default=0), "cosine": dots / np.sqrt(aa * bb) if aa * bb else None}


def backbone_parameter_digest(model):
    import hashlib
    h = hashlib.sha256()
    for value in jax.tree_util.tree_leaves(model.params):
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def numerical_preflight(runtime, reader, origins, output, identities):
    """Run before production at both resolutions using complete prepared inputs."""
    b, adapter, trainer = runtime.backbone, runtime.adapter, runtime.trainer
    params, memory = runtime.params, zero_memory(runtime.memory)
    before_hash = backbone_parameter_digest(b.model)
    report = {"identities": identities, "tolerances": TOLERANCES, "zero_residual": [], "passed": False}
    for origin in origins:
        inputs, forcing = runtime.store.inputs_and_forcing(b.model, origin)
        base = b.encode(inputs, forcing)
        corrected = base
        initial = base
        h = zero_memory(memory)
        for lead in range(1, 41):
            delta, h = runtime.branch.apply(params, h, jax.random.PRNGKey(lead),
                runtime.normalization.inputs(adapter.features(corrected)), runtime.known(corrected, forcing))
            np.testing.assert_array_equal(delta, 0)
            corrected = adapter.apply_increment(b.advance_6h(corrected, forcing), runtime.normalization.increment(delta))
            base = b.advance_6h(base, forcing)
            if any(not np.isfinite(x).all() for x in jax.device_get(jax.tree_util.tree_leaves((base, corrected)))):
                raise FloatingPointError(f"Nonfinite zero-residual preflight at {origin}, lead {lead}")
            tree_comparison(corrected, base, exact=True)
            b.assert_time(initial, corrected, lead)
            if lead in (1, 20, 40):
                report["zero_residual"].append({"origin": origin, "lead": lead,
                    **tree_comparison(b.decode(corrected, forcing), b.decode(base, forcing), exact=True)})
    # Nonzero output head AND nonzero incoming SSM and conv state.
    nonzero = jax.tree_util.tree_map(lambda x: x, params)
    found = False
    for name in nonzero:
        if "native_zero_head" in name:
            nonzero[name] = dict(nonzero[name], b=jnp.ones_like(nonzero[name]["b"]) * 1e-3)
            found = True
    if not found:
        raise AssertionError("Residual head not found")
    nonzero_h = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-3), memory)
    cached, live = [], []
    for i in range(24):
        record = reader[i]
        fresh = live_record(b, runtime.store, record["origin"])
        fresh["target"] = record["target"]
        for source, destination in ((record, cached), (fresh, live)):
            destination.append(trainer.record(source["origin_state"], source["baseline_state"], source["forcing"],
                source["target"], runtime.known(source["origin_state"], source["forcing"])))
    key = jax.random.PRNGKey(22)
    cold = trainer.cached_gradients(nonzero, nonzero_h, key, cached)
    hot = trainer.cached_gradients(nonzero, nonzero_h, key, live)
    report["live_cache_loss"] = tree_comparison(cold[0], hot[0])
    report["live_cache_memory"] = tree_comparison(cold[1], hot[1])
    report["live_cache_gradients"] = tree_comparison(cold[3], hot[3])
    if report["live_cache_gradients"]["cosine"] is None or report["live_cache_gradients"]["cosine"] < TOLERANCES["gradient_cosine_min"]:
        raise AssertionError("Live/cached gradients failed cosine tolerance")
    opt = trainer.optimizer.init(nonzero)
    report["live_cache_update"] = tree_comparison(trainer.checked_update(nonzero, opt, cold[3])[:2],
                                                  trainer.checked_update(nonzero, opt, hot[3])[:2])
    # Fixed weights and the same keys, split only at the BPTT boundary.
    def forward(h, records, key):
        scores = []
        for record in records:
            key, sub = jax.random.split(key)
            loss, h, _ = trainer.forward(nonzero, h, sub, *record)
            scores.append(loss)
        return scores, h, key
    whole = forward(nonzero_h, cached, key)
    first = forward(nonzero_h, cached[:12], key)
    second = forward(stop(first[1]), cached[12:], first[2])
    report["chunk_carry"] = tree_comparison((whole[0], whole[1]), (first[0] + second[0], second[1]), exact=True)
    if backbone_parameter_digest(b.model) != before_hash:
        raise AssertionError("Frozen backbone parameters changed")
    # A zero output projection initially blocks upstream recurrent gradients.
    # Setting a small nonzero head weight verifies those paths as well.
    for name in nonzero:
        if "native_zero_head" in name:
            nonzero[name] = dict(nonzero[name], w=jnp.ones_like(nonzero[name]["w"]) * 1e-4)
    _, _, _, gradients = trainer.cached_gradients(nonzero, nonzero_h, key, cached[:2])
    recurrent_norm = sum(float(jnp.sum(x * x)) for name, leaves in gradients.items()
                         if "temporal" in name for x in jax.tree_util.tree_leaves(leaves))
    if recurrent_norm <= 0:
        raise AssertionError("Decoder input / recurrent gradient path is absent")
    report.update(frozen_parameter_hash=before_hash, recurrent_gradient_squared_norm=recurrent_norm, passed=True)
    write_json(output, report, immutable=True)
    return report


def profile_runtime(runtime, reader, origin, output, identities, *, repeats=3):
    from .finetune import episode_gradients
    trainer = runtime.trainer
    def chunk():
        records = []
        started = time.perf_counter()
        for i in range(24):
            r = reader[i]
            records.append(trainer.record(r["origin_state"], r["baseline_state"], r["forcing"], r["target"],
                                          runtime.known(r["origin_state"], r["forcing"])))
        wait = time.perf_counter() - started
        result = trainer.cached_gradients(runtime.params, runtime.memory, jax.random.PRNGKey(22), records)
        jax.block_until_ready(result)
        return wait
    start = time.perf_counter()
    chunk()
    compilation_and_first = time.perf_counter() - start
    times, waits = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        waits.append(chunk())
        times.append(time.perf_counter() - start)
    start = time.perf_counter()
    episode_gradients(trainer, runtime.params, runtime.memory, jax.random.PRNGKey(22),
                      backbone=runtime.backbone, store=runtime.store, known=runtime.known, origin=origin)
    first_live = time.perf_counter() - start
    live_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        episode_gradients(trainer, runtime.params, runtime.memory, jax.random.PRNGKey(22),
                          backbone=runtime.backbone, store=runtime.store, known=runtime.known, origin=origin)
        live_times.append(time.perf_counter() - start)
    # Include the actual validation forward paths when sizing production jobs.
    from .evaluate import forecast_origin
    validation_times = {}
    for warm in (False, True):
        start = time.perf_counter()
        forecast_origin(runtime, runtime.params, origin, steps=20, warm=warm)
        validation_times["warm" if warm else "cold"] = time.perf_counter() - start
    start = time.perf_counter()
    record = reader[0]
    item = trainer.record(record["origin_state"], record["baseline_state"], record["forcing"],
                          record["target"], runtime.known(record["origin_state"], record["forcing"]))
    jax.block_until_ready(trainer.forward(runtime.params, runtime.memory, jax.random.PRNGKey(22), *item))
    one_step_seconds = time.perf_counter() - start
    import resource
    stats = [device.memory_stats() for device in jax.devices()]
    report = {"identities": identities, "devices": [str(d) for d in jax.devices()],
              "device_kinds": [d.device_kind for d in jax.devices()],
              "cached_compile_and_first_seconds": compilation_and_first, "cached_update_seconds": times,
              "cache_reader_seconds": waits, "live20_compile_and_first_seconds": first_live,
              "live20_seconds": live_times, "validation20_seconds": validation_times,
              "one_step_validation_seconds": one_step_seconds,
              "peak_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "gpu_allocator_stats": stats, **parameter_counts(runtime.params, runtime.memory),
              "measured": True, "gpu_measured": any(d.platform == "gpu" for d in jax.devices())}
    write_json(output, report, immutable=True)
    return report
