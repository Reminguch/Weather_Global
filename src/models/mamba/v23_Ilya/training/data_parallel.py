"""Single-host data-parallel tree and device helpers for v23_Ilya."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np


def validate_data_parallel_runtime(
    num_devices: int,
    *,
    require_gpu: bool = True,
) -> tuple[jax.Device, ...]:
    if jax.process_count() != 1:
        raise RuntimeError(
            "v23_Ilya data parallelism is single-host only; "
            f"jax.process_count()={jax.process_count()}"
        )
    devices = tuple(jax.local_devices())
    if len(devices) != num_devices:
        raise RuntimeError(
            f"Expected exactly {num_devices} local devices, found {len(devices)}: "
            f"{[str(device) for device in devices]}"
        )
    if require_gpu:
        non_gpu = [str(device) for device in devices if device.platform != "gpu"]
        if non_gpu:
            raise RuntimeError(
                f"Data-parallel training requires GPU devices, got {non_gpu}"
            )
    return devices


def derive_replica_step_keys(
    master_key,
    *,
    num_replicas: int,
    bptt_steps: int,
):
    """Advance one checkpointed master key and deterministically key every lane."""

    if num_replicas <= 0 or bptt_steps <= 0:
        raise ValueError("num_replicas and bptt_steps must be positive")
    next_master_key, update_key = jax.random.split(master_key)
    keys = jax.numpy.stack(
        [
            jax.numpy.stack(
                [
                    jax.random.fold_in(
                        jax.random.fold_in(update_key, replica_index),
                        bptt_index,
                    )
                    for bptt_index in range(bptt_steps)
                ]
            )
            for replica_index in range(num_replicas)
        ]
    )
    return next_master_key, keys


def _host_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        tree,
    )


def replicate_tree(tree, devices: Sequence[jax.Device]):
    """Create one identical pmap lane on every device."""

    host_tree = _host_tree(tree)
    return jax.tree_util.tree_map(
        lambda value: jax.device_put_sharded(
            [value for _ in devices],
            list(devices),
        ),
        host_tree,
    )


def shard_replica_tree(tree, devices: Sequence[jax.Device]):
    """Place an existing leading-replica host tree onto matching devices."""

    host_tree = _host_tree(tree)

    def shard(value):
        if value.ndim == 0 or value.shape[0] != len(devices):
            raise ValueError(
                f"Replica checkpoint leaf must start with {len(devices)}, "
                f"got shape={value.shape}"
            )
        return jax.device_put_sharded(
            [value[index] for index in range(len(devices))],
            list(devices),
        )

    return jax.tree_util.tree_map(shard, host_tree)


def unreplicate_tree(tree):
    """Return the canonical first replica without transferring all replicas."""

    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value[0])),
        tree,
    )


def replica_tree_to_host(tree):
    return _host_tree(tree)


def replica_max_abs_difference(tree) -> float:
    maximum = 0.0
    for leaf in jax.tree_util.tree_leaves(replica_tree_to_host(tree)):
        if leaf.ndim == 0 or leaf.shape[0] <= 1:
            continue
        if not np.issubdtype(leaf.dtype, np.inexact):
            if not np.all(leaf == leaf[0]):
                return float("inf")
            continue
        difference = np.max(np.abs(leaf - leaf[0]))
        maximum = max(maximum, float(difference))
    return maximum


def pack_replica_trees(
    trees: Sequence[Any],
    devices: Sequence[jax.Device],
) -> tuple[jax.tree_util.PyTreeDef, tuple[Any, ...]]:
    """Pack matching xarray PyTrees into pmap-sharded array leaves."""

    if len(trees) != len(devices):
        raise ValueError(f"Expected {len(devices)} replica trees, got {len(trees)}")
    treedef = jax.tree_util.tree_structure(trees[0])
    replica_leaves = []
    for replica, tree in enumerate(trees):
        candidate = jax.tree_util.tree_structure(tree)
        if candidate != treedef:
            raise ValueError(f"Replica {replica} PyTree structure differs")
        replica_leaves.append(jax.tree_util.tree_leaves(tree))
    packed = tuple(
        jax.device_put_sharded(
            [
                np.asarray(jax.device_get(replica_leaves[replica][leaf_index]))
                for replica in range(len(devices))
            ],
            list(devices),
        )
        for leaf_index in range(len(replica_leaves[0]))
    )
    return treedef, packed


def unpack_replica_trees(
    treedef: jax.tree_util.PyTreeDef,
    packed_leaves: Sequence[Any],
    num_replicas: int,
) -> tuple[Any, ...]:
    host_leaves = [np.asarray(jax.device_get(value)) for value in packed_leaves]
    return tuple(
        jax.tree_util.tree_unflatten(
            treedef,
            [value[replica] for value in host_leaves],
        )
        for replica in range(num_replicas)
    )
