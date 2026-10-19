"""Independent sequence-lane operations and cached batching qualification."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def state_lane(state, lane: int, batch_size: int):
    """Extract one lane without copying or changing shared scalar state."""
    if batch_size < 1 or not 0 <= lane < batch_size:
        raise ValueError("Invalid recurrent-state lane")
    return jax.tree_util.tree_map(
        lambda value: value[lane:lane + 1] if getattr(value, "ndim", 0) > 0
        and value.shape[0] == batch_size else value, state,
    )


def _dataset_lane(dataset, lane):
    if "batch" not in dataset.dims:
        raise ValueError("Cached sequence fields require a batch dimension")
    return dataset.isel(batch=slice(lane, lane + 1)).assign_coords(batch=[0])


def _combine_states(states):
    def combine(*values):
        if getattr(values[0], "ndim", 0) > 0 and values[0].shape[0] == 1:
            return jnp.concatenate(values, axis=0)
        if not all(np.array_equal(np.asarray(values[0]), np.asarray(value)) for value in values[1:]):
            raise ValueError("Non-lane recurrent state differs between independent samples")
        return values[0]
    return jax.tree_util.tree_map(combine, *states)


def compare_batch_lanes(*, trainer, params, state, keys, batch, batch_size: int) -> dict:
    """Compare batched BPTT against independent lanes at identical weights.

    No optimizer update is performed. Dropout must be disabled so using the
    same step keys defines the same deterministic per-lane calculation. Report
    numerical tolerance drift separately from structural/nonfinite failures,
    as in the existing cached recovery policy.
    """
    from .cached_diagnostics import comparison_integrity, tree_comparison
    if getattr(trainer.config.architecture, "temporal_dropout", 0.0):
        raise ValueError("Independent-lane qualification requires dropout disabled")
    if batch_size < 1 or any(value.sizes.get("batch") != batch_size
                             for group in (batch.inputs, batch.targets, batch.forcings)
                             for value in group):
        raise ValueError("Qualification batch fields disagree with batch_size")
    batched = trainer.loss_and_grad(params, state, keys, batch.inputs, batch.targets,
                                   batch.forcings, retain_predictions=False)
    loss, next_state, _, _, gradient = jax.device_get(batched)
    lane_losses, lane_states, lane_gradients = [], [], []
    for lane in range(batch_size):
        result = trainer.loss_and_grad(
            params, state_lane(state, lane, batch_size), keys,
            tuple(_dataset_lane(value, lane) for value in batch.inputs),
            tuple(_dataset_lane(value, lane) for value in batch.targets),
            tuple(_dataset_lane(value, lane) for value in batch.forcings),
            retain_predictions=False,
        )
        lane_loss, lane_state, _, _, lane_gradient = jax.device_get(result)
        lane_losses.append(float(lane_loss))
        lane_states.append(lane_state)
        lane_gradients.append(lane_gradient)
    mean_gradient = jax.tree_util.tree_map(
        lambda *values: np.mean(np.stack(values), axis=0, dtype=np.float32),
        *lane_gradients,
    )
    checks = {
        "mean_loss": tree_comparison(np.asarray(loss), np.asarray(np.mean(lane_losses), dtype=np.float32)),
        "mean_gradient": tree_comparison(gradient, mean_gradient),
        "independent_lane_state": tree_comparison(next_state, jax.device_get(_combine_states(lane_states))),
    }
    return {"kind": "cached_independent_lanes", "batch_size": batch_size,
            "passed": all(value["passed"] for value in checks.values()),
            "integrity_passed": all(comparison_integrity(value) for value in checks.values()),
            "loss": float(loss), "lane_losses": lane_losses, "checks": checks}
