"""Recorded FP32 and deterministic GPU policy for this experiment only."""
from __future__ import annotations

import os
import shlex

POLICY = {"version": "fp32_deterministic_v1", "matmul_precision": "highest",
          "x64": False, "xla_gpu_exclude_nondeterministic_ops": True}

# Both lowerings implement the same deterministic FP32 contract. The expander
# avoids serial scatter loops (303104 launches per interpolation VJP in the
# pinned CUDA plugin). Executed source snapshots pin this compiler choice.
REQUIRED_FLAGS = ("--xla_gpu_exclude_nondeterministic_ops",
                  "--xla_gpu_enable_scatter_determinism_expander")


def configure_environment():
    """Call before importing JAX or creating any device arrays."""
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    for option in REQUIRED_FLAGS:
        matches = [value for value in flags if value.split("=", 1)[0] == option]
        if any(value not in (option, option + "=true", option + "=1") for value in matches):
            raise ValueError(f"This experiment requires {option}=true")
        if not matches:
            flags.append(option + "=true")
    os.environ["XLA_FLAGS"] = " ".join(flags)
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ["JAX_ENABLE_X64"] = "false"


def verify_runtime():
    import jax
    if jax.config.jax_default_matmul_precision != "highest" or jax.config.jax_enable_x64:
        raise ValueError("Configure the pinned numerical policy before importing JAX")
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    for option in REQUIRED_FLAGS:
        matches = [value for value in flags if value.split("=", 1)[0] == option]
        if not matches or any(value not in (option, option + "=true", option + "=1") for value in matches):
            raise ValueError(f"Pinned deterministic GPU lowering requires {option}=true")
