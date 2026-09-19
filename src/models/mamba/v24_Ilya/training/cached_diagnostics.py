"""Numerical comparisons and process measurements for the cached experiment."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import resource

import numpy as np


def require_matched_gpu_environment():
    import jax
    import jaxlib
    import haiku
    import optax
    flags = os.environ.get("XLA_FLAGS", "").split()
    required = ("--xla_gpu_enable_triton_gemm=false", "--xla_gpu_deterministic_ops=true")
    for flag in required:
        name = flag.split("=", 1)[0]
        assigned = [value for value in flags if value.startswith(name + "=")]
        if not assigned or assigned[-1] != flag:
            raise RuntimeError(f"Required matched/cache numerical setting is missing: {flag}")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("The matched res1 experiment requires exactly one allocated GPU")
    return {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "haiku": haiku.__version__,
            "optax": optax.__version__, "device": devices[0].device_kind,
            "required_xla_flags": list(required)}


def tree_digest(tree):
    import jax
    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def tree_comparison(actual, expected, *, relative=1e-3, absolute=1e-5, cosine=None):
    """Accumulate in FP64 without flattening a large state into another tape."""
    import jax
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    if actual_structure != expected_structure:
        return {"passed": False, "reason": "tree structure differs"}
    aa = bb = difference = dot = maximum = reference_maximum = 0.0
    count = 0
    for left, right in zip(actual_leaves, expected_leaves, strict=True):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape:
            return {"passed": False, "reason": "leaf shape differs"}
        # Iterate bounded pieces; a full-resolution state can exceed 100 MiB.
        left, right = left.reshape(-1), right.reshape(-1)
        for offset in range(0, left.size, 262144):
            a = left[offset:offset + 262144].astype(np.float64)
            b = right[offset:offset + 262144].astype(np.float64)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                return {"passed": False, "reason": "nonfinite value"}
            delta = a - b
            aa += float(np.sum(a * a))
            bb += float(np.sum(b * b))
            difference += float(np.sum(delta * delta))
            dot += float(np.sum(a * b))
            maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
            reference_maximum = max(reference_maximum, float(np.max(np.abs(b), initial=0)))
            count += a.size
    near_zero = reference_maximum <= absolute
    relative_l2 = math.sqrt(difference / bb) if bb else (0.0 if difference == 0 else None)
    similarity = min(1.0, max(-1.0, dot / math.sqrt(aa * bb))) if aa and bb else None
    passed = maximum <= absolute if near_zero else relative_l2 is not None and relative_l2 <= relative
    if cosine is not None and not near_zero:
        passed = passed and similarity is not None and similarity >= cosine
    return {"passed": bool(passed), "relative_l2": relative_l2, "max_abs": maximum,
            "reference_max_abs": reference_maximum, "cosine": similarity,
            "near_zero": bool(near_zero), "elements": int(count)}


def gradient_groups(tree):
    result = {}
    for module, values in tree.items():
        if "temporal_residual_head" in module:
            group = "residual_head"
        elif "temporal" in module:
            group = "temporal_stage_1" if "_s1" in module else "temporal_stage_0"
        else:
            group = module.split("/")[0]
        result.setdefault(group, {})[module] = values
    return result


def gradient_comparison(actual, expected):
    groups_a, groups_b = gradient_groups(actual), gradient_groups(expected)
    if groups_a.keys() != groups_b.keys():
        return {"passed": False, "reason": "gradient groups differ"}
    results = {name: tree_comparison(groups_a[name], values, cosine=0.99999)
               for name, values in groups_b.items()}
    return {"passed": all(value["passed"] for value in results.values()), "groups": results}


def capture_gradients(optimizer):
    """Wrap optimizer state to expose the exact gradients received by online code.

    The maintained update calls this ordinary Optax transformation unchanged.
    The first state component and returned parameter updates are unchanged;
    the second state component is only a diagnostic copy of the input gradient.
    """
    import jax
    import jax.numpy as jnp
    import optax

    def init(params):
        return optimizer.init(params), jax.tree_util.tree_map(jnp.zeros_like, params)

    def update(grads, state, params=None):
        updates, next_state = optimizer.update(grads, state[0], params)
        return updates, (next_state, grads)

    return optax.GradientTransformation(init, update)


def memory_snapshot():
    result = {"host_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
                result[key + "_gib"] = int(value.split()[0]) / 1024**2
    try:
        import jax
        stats = jax.devices()[0].memory_stats() or {}
        result["gpu_memory"] = {k: int(v) for k, v in stats.items() if isinstance(v, (int, np.integer))}
    except RuntimeError:
        pass
    return result


def timing_summary(records, *, weather_steps=24):
    timed = records[20:70]
    if len(timed) != 50:
        raise ValueError("A production benchmark requires fifty updates after twenty warmup/paired updates")
    compute = np.array([row["compute_seconds"] for row in timed])
    elapsed = np.array([row["end_to_end_seconds"] for row in timed])
    return {"timed_updates": 50, "compute_median_seconds": float(np.median(compute)),
            "compute_p90_seconds": float(np.percentile(compute, 90)),
            "end_to_end_median_seconds": float(np.median(elapsed)),
            "end_to_end_p90_seconds": float(np.percentile(elapsed, 90)),
            "valid_weather_steps_per_second": float(weather_steps * len(elapsed) / np.sum(elapsed)),
            "compute_definition": "Synchronized step routine, including required host orchestration and transfers; excludes chunk loading"}
