"""Single-host data-parallel tree and device helpers for v23_Ilya."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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


def _numpy_tree(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def _host_tree(tree):
    """Materialize a complete PyTree with one container-level device_get.

    Passing the whole container to JAX lets independent leaf and device copies be
    scheduled together.  Mapping ``device_get`` leaf by leaf serializes those
    synchronization points and is especially expensive for pmap-sharded weather
    trees.
    """

    return _numpy_tree(jax.device_get(tree))


def _start_host_copy(tree) -> None:
    """Start copies for all addressable shards without materializing the tree."""

    for leaf in jax.tree_util.tree_leaves(tree):
        if not isinstance(leaf, jax.Array):
            continue
        shards = leaf.addressable_shards
        if shards:
            for shard in shards:
                shard.data.copy_to_host_async()
        else:
            leaf.copy_to_host_async()


@dataclass
class PendingHostTree:
    """A bounded asynchronous device-to-host PyTree copy.

    The object deliberately retains its device tree until ``materialize`` is
    called.  DP training keeps at most one pending tape boundary so this cannot
    grow into a device-resident BPTT tape.
    """

    _tree: Any
    _materialized: Any | None = None

    def materialize(self):
        if self._materialized is None:
            self._materialized = _host_tree(self._tree)
            self._tree = None
        return self._materialized


def start_tree_to_host(tree) -> PendingHostTree:
    """Start an asynchronous copy and return a later materialization handle."""

    _start_host_copy(tree)
    return PendingHostTree(tree)


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

    first_replica = jax.tree_util.tree_map(lambda value: value[0], tree)
    return _host_tree(first_replica)


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
    leaf_count = len(replica_leaves[0])
    flat_host_leaves = _numpy_tree(
        jax.device_get(
            tuple(
                replica_leaves[replica][leaf_index]
                for replica in range(len(devices))
                for leaf_index in range(leaf_count)
            )
        )
    )
    host_replica_leaves = tuple(
        flat_host_leaves[
            replica * leaf_count : (replica + 1) * leaf_count
        ]
        for replica in range(len(devices))
    )
    packed = tuple(
        jax.device_put_sharded(
            [
                host_replica_leaves[replica][leaf_index]
                for replica in range(len(devices))
            ],
            list(devices),
        )
        for leaf_index in range(leaf_count)
    )
    return treedef, packed


def pack_device_replica_trees(
    trees: Sequence[Any],
    devices: Sequence[jax.Device],
) -> tuple[jax.tree_util.PyTreeDef, tuple[Any, ...]]:
    """Pack matching trees whose leaves already live on their lane devices."""

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
            [replica_leaves[replica][leaf_index] for replica in range(len(devices))],
            list(devices),
        )
        for leaf_index in range(len(replica_leaves[0]))
    )
    return treedef, packed


def unpack_device_replica_trees(
    treedef: jax.tree_util.PyTreeDef,
    packed_leaves: Sequence[Any],
    num_replicas: int,
) -> tuple[Any, ...]:
    """Expose local pmap shards as per-device trees without a host transfer."""

    per_leaf_shards = []
    for value in packed_leaves:
        shards = value.addressable_shards
        if len(shards) != num_replicas:
            raise ValueError(
                f"Expected {num_replicas} addressable shards, got {len(shards)}"
            )
        per_leaf_shards.append(tuple(shard.data[0] for shard in shards))
    return tuple(
        jax.tree_util.tree_unflatten(
            treedef,
            [value[replica] for value in per_leaf_shards],
        )
        for replica in range(num_replicas)
    )


def unpack_replica_trees(
    treedef: jax.tree_util.PyTreeDef,
    packed_leaves: Sequence[Any],
    num_replicas: int,
) -> tuple[Any, ...]:
    host_leaves = _numpy_tree(jax.device_get(tuple(packed_leaves)))
    return tuple(
        jax.tree_util.tree_unflatten(
            treedef,
            [value[replica] for value in host_leaves],
        )
        for replica in range(num_replicas)
    )
