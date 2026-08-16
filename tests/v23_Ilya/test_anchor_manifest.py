from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_operations.preprocessing.build_graphcast37_prepared_stream import (
    PreparedStoreV2Writer,
    _month_windows,
    _time_shards,
)
from src.models.mamba.v23_Ilya import anchor_manifest
from src.models.mamba.v23_Ilya.training.data import build_segments


def _task_config() -> SimpleNamespace:
    return SimpleNamespace(
        input_variables=("temperature", "land_sea_mask"),
        target_variables=("temperature",),
        forcing_variables=("toa_incident_solar_radiation",),
        pressure_levels=(500, 850),
        input_duration="12h",
    )


def _january_dataset(times: pd.DatetimeIndex) -> xr.Dataset:
    levels = np.asarray([500, 850], dtype=np.int64)
    lat = np.asarray([-10.0, 10.0], dtype=np.float32)
    lon = np.asarray([0.0, 120.0, 240.0], dtype=np.float32)
    temperature = np.arange(
        len(times) * len(levels) * len(lat) * len(lon),
        dtype=np.float32,
    ).reshape(len(times), len(levels), len(lat), len(lon))
    forcing = np.zeros((len(times), len(lat), len(lon)), dtype=np.float32)
    static = np.ones((len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset(
        {
            "temperature": (
                ("time", "level", "lat", "lon"),
                temperature,
            ),
            "toa_incident_solar_radiation": (
                ("time", "lat", "lon"),
                forcing,
            ),
            "land_sea_mask": (("lat", "lon"), static),
        },
        coords={
            "time": times,
            "datetime": ("time", times),
            "level": levels,
            "lat": lat,
            "lon": lon,
        },
    )


def test_january_manifest_requires_empty_validation_opt_in_and_has_five_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = pd.date_range(
        "2021-01-01T00:00:00",
        "2021-01-31T18:00:00",
        freq="6h",
    )
    prepared_root = tmp_path / "prepared"
    writer = PreparedStoreV2Writer(
        root=prepared_root,
        expected_times=times,
        task_cfg=_task_config(),
        checkpoint_path=Path("graphcast37.npz"),
        checkpoint_sha256="abc123",
        source_uri="synthetic.zarr",
        resume=True,
    )
    window = _month_windows(times, _time_shards(times))[0]
    writer.write_window(_january_dataset(times), window, chunk_time=16)
    writer.finalize()

    checkpoint = SimpleNamespace(
        task_config=_task_config(),
        model_config=SimpleNamespace(
            resolution=0.25,
            mesh_size=6,
            gnn_msg_steps=16,
        ),
    )
    monkeypatch.setattr(
        anchor_manifest,
        "load_graphcast_checkpoint",
        lambda _path: checkpoint,
    )

    with pytest.raises(ValueError, match="allow_empty_validation"):
        anchor_manifest.build_anchor_manifest(
            prepared_root=prepared_root,
            baseline_checkpoint=Path("graphcast37.npz"),
            output_root=tmp_path / "rejected",
        )

    output_root = tmp_path / "accepted"
    metadata = anchor_manifest.build_anchor_manifest(
        prepared_root=prepared_root,
        baseline_checkpoint=Path("graphcast37.npz"),
        output_root=output_root,
        allow_empty_validation=True,
    )
    anchors = np.load(output_root / "anchors/anchor_indices.npy")
    train = np.load(output_root / "anchors/split_train.npy")
    validation = np.load(output_root / "anchors/split_val.npy")
    segments = build_segments(train, segment_steps=120)

    assert metadata["n_anchors_total"] == 120
    assert metadata["n_anchors_train"] == 120
    assert metadata["n_anchors_val"] == 0
    np.testing.assert_array_equal(anchors, np.arange(2, 122))
    assert validation.size == 0
    assert len(segments) == 1
    assert segments[0].size == 120
    assert list(range(0, segments[0].size, 24)) == [0, 24, 48, 72, 96]
