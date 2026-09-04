from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models.mamba.v24_Ilya.training.frame_data import (
    EndpointFrameWorkspace,
    load_endpoint_frame_batch,
)


class FakeStore:
    def __init__(self) -> None:
        time = np.arange(40, dtype=np.float32)
        self._time_indices = np.arange(time.size, dtype=np.int64)
        self.sizes = {"time": time.size}
        self.coords = {"time": self._time_indices, "lat": np.array([-1.0, 1.0])}
        self.data_vars = {
            "x": SimpleNamespace(
                data=np.stack([time, time + 100.0], axis=1),
                dims=("time", "lat"),
                coords={"lat": self.coords["lat"]},
            ),
            "forcing": SimpleNamespace(
                data=time[:, None],
                dims=("time", "channel"),
                coords={"channel": np.array([0])},
            ),
            "static": SimpleNamespace(
                data=np.array([7.0, 8.0], np.float32),
                dims=("lat",),
                coords={"lat": self.coords["lat"]},
            ),
        }


class FakeLazyArray:
    """Array-like source that must be read through its bounded take method."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.take_calls: list[tuple[np.ndarray, int]] = []

    def __array__(self, *args, **kwargs) -> np.ndarray:
        raise AssertionError("lazy source must not be converted with np.asarray")

    def take(self, indices: np.ndarray, *, axis: int = 0) -> np.ndarray:
        self.take_calls.append((np.array(indices, copy=True), axis))
        return np.take(self._values, indices, axis=axis)


TASK = SimpleNamespace(
    input_variables=("x", "forcing", "static"),
    target_variables=("x",),
    forcing_variables=("forcing",),
)


def test_last_step_reads_five_teacher_frames_and_one_truth() -> None:
    batch = load_endpoint_frame_batch(
        store=FakeStore(),
        raw_anchor_indices=np.arange(2, 26),
        input_steps=2,
        truth_prefix_steps=4,
        loss_mode="last_step",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
    )
    assert len(batch.input_frames) == 5
    assert len(batch.forcings) == 24
    assert len(batch.truths) == 1
    assert batch.report.truth_target_steps == 1
    assert batch.report.teacher_input_windows == 4
    assert batch.report.unique_teacher_frames == 5
    np.testing.assert_array_equal(batch.truths[0]["x"].values[0, 0], [26, 126])
    assert set(batch.static_inputs.data_vars) == {"static"}


def test_all_steps_materializes_every_truth() -> None:
    batch = load_endpoint_frame_batch(
        store=FakeStore(),
        raw_anchor_indices=np.arange(2, 6),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
    )
    assert len(batch.input_frames) == 3
    assert len(batch.truths) == 4
    assert [float(value["x"].values[0, 0, 0]) for value in batch.truths] == [
        3.0,
        4.0,
        5.0,
        6.0,
    ]


def test_sparse_steps_materializes_only_exact_horizon_targets() -> None:
    batch = load_endpoint_frame_batch(
        store=FakeStore(),
        raw_anchor_indices=np.arange(2, 26),
        input_steps=2,
        truth_prefix_steps=5,
        loss_mode="sparse_steps",
        supervised_step_indices=(4, 7, 11, 15, 19, 23),
        task_config=TASK,
        dt=pd.Timedelta("6h"),
    )
    assert len(batch.input_frames) == 6
    assert batch.report.truth_target_steps == 6
    assert [float(value["x"].values[0, 0, 0]) for value in batch.truths] == [
        7.0,
        10.0,
        14.0,
        18.0,
        22.0,
        26.0,
    ]


def test_truth_workspace_reuses_owned_storage_without_mutating_source() -> None:
    store = FakeStore()
    source_before = np.array(store.data_vars["x"].data, copy=True)
    workspace = EndpointFrameWorkspace()
    first = load_endpoint_frame_batch(
        store=store,
        raw_anchor_indices=np.arange(2, 6),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
        truth_workspace=workspace,
    )
    first_leaf = first.truths[0]["x"].values
    second = load_endpoint_frame_batch(
        store=store,
        raw_anchor_indices=np.arange(8, 12),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
        truth_workspace=workspace,
    )
    second_leaf = second.truths[0]["x"].values

    assert np.shares_memory(first_leaf, second_leaf)
    assert first_leaf.flags.writeable
    np.testing.assert_array_equal(second_leaf[0, 0], [9.0, 109.0])
    np.testing.assert_array_equal(store.data_vars["x"].data, source_before)


def test_truth_workspace_keeps_replica_slots_independent() -> None:
    store = FakeStore()
    workspace = EndpointFrameWorkspace()
    first = load_endpoint_frame_batch(
        store=store,
        raw_anchor_indices=np.arange(2, 6),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
        truth_workspace=workspace,
        truth_workspace_slot=0,
    )
    second = load_endpoint_frame_batch(
        store=store,
        raw_anchor_indices=np.arange(8, 12),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
        truth_workspace=workspace,
        truth_workspace_slot=1,
    )
    assert not np.shares_memory(
        first.truths[0]["x"].values,
        second.truths[0]["x"].values,
    )


def test_truth_workspace_converts_non_fp32_source_once() -> None:
    store = FakeStore()
    store.data_vars["x"].data = store.data_vars["x"].data.astype(np.float64)
    workspace = EndpointFrameWorkspace()
    first = load_endpoint_frame_batch(
        store=store,
        raw_anchor_indices=np.arange(2, 6),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
        truth_workspace=workspace,
    )
    first_leaf = first.truths[0]["x"].values
    second = load_endpoint_frame_batch(
        store=store,
        raw_anchor_indices=np.arange(8, 12),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
        truth_workspace=workspace,
    )
    assert first_leaf.dtype == np.float32
    assert np.shares_memory(first_leaf, second.truths[0]["x"].values)


def test_truth_workspace_reads_lazy_source_through_bounded_take() -> None:
    store = FakeStore()
    lazy_source = FakeLazyArray(store.data_vars["x"].data.astype(np.float64))
    store.data_vars["x"].data = lazy_source
    workspace = EndpointFrameWorkspace()

    batch = load_endpoint_frame_batch(
        store=store,
        raw_anchor_indices=np.arange(2, 6),
        input_steps=2,
        truth_prefix_steps=2,
        loss_mode="all_steps",
        task_config=TASK,
        dt=pd.Timedelta("6h"),
        truth_workspace=workspace,
    )

    truth_values = np.concatenate(
        [truth["x"].values[:, 0] for truth in batch.truths],
        axis=0,
    )
    assert truth_values.dtype == np.float32
    np.testing.assert_array_equal(
        truth_values,
        np.array(
            [[3.0, 103.0], [4.0, 104.0], [5.0, 105.0], [6.0, 106.0]],
            dtype=np.float32,
        ),
    )
    assert any(
        axis == 0 and np.array_equal(indices, np.arange(3, 7))
        for indices, axis in lazy_source.take_calls
    )
