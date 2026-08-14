from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models.mamba.v23_Ilya.training.frame_data import (
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

