from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
import zarr

from src.models.mamba.v24_Ilya.spectral_observer import (
    FIELD_SPECS, SPECTRAL_VARIABLES, SpectralObserver, extract_fields,
)


def fields(scale=1):
    lat = np.asarray([-90, -60, -45, -30, 0, 30, 45, 60, 90.])
    lon = np.arange(16) * 360 / 16
    wave = scale * np.cos(np.deg2rad(lon) * 3)
    plane = np.broadcast_to(wave, (1, 1, len(lat), len(lon))).astype(np.float32)
    surface = {name: (("batch", "time", "lat", "lon"), plane.copy())
               for _, name, level, _ in FIELD_SPECS if level is None}
    pressure = {name: (("batch", "time", "level", "lat", "lon"),
                       np.stack((plane, 2 * plane), axis=2))
                for _, name, level, _ in FIELD_SPECS if level is not None}
    return xr.Dataset(surface | pressure, coords={
        "lat": lat, "lon": lon, "batch": [0], "time": [np.timedelta64(6, "h")], "level": [700, 850],
    })


def record():
    return dict(initialization_id="20220115T000000Z", initialization_time="2022-01-15T00:00:00Z",
                lead_hours=[6, 12], valid_times=["2022-01-15T06:00:00Z", "2022-01-15T12:00:00Z"])


def observer(tmp_path, **kwargs):
    return SpectralObserver(tmp_path, dict(model_id="M1", protocol_hash="protocol", checkpoint_sha256="checkpoint"),
                            target_steps=2, **kwargs)


def test_complete_fields_and_statistics_publish_atomically(tmp_path):
    writer = observer(tmp_path)
    writer.begin_sample(record())
    for step in range(2):
        writer.update_step(step, fields(), fields(.75), fields(.5))
    assert not list(tmp_path.glob("*.npz"))
    assert not list(tmp_path.glob("*.fields.zarr"))
    writer.finish_sample()
    result = list(tmp_path.glob("*.npz"))[0]
    with np.load(result, allow_pickle=False) as saved:
        assert tuple(saved["variables"]) == SPECTRAL_VARIABLES
        assert saved["full_power"].shape == (2, 11, 9)
        np.testing.assert_allclose(saved["full_power"], .25 * saved["truth_power"], atol=1e-13)
        metadata = json.loads(saved["metadata"].item())
        assert metadata["valid_times"] == record()["valid_times"]
        assert metadata["status"] == "complete"
        store = zarr.open_consolidated(metadata["field_store"], mode="r")
        np.testing.assert_array_equal(store["truth"][0], extract_fields(fields()))
        assert store["truth"].dtype == np.dtype("float32")
        assert store["truth"].shape == (2, 9, 9, 16)
        assert store.attrs["status"] == "complete"
        assert store["truth"].attrs["_ARRAY_DIMENSIONS"] == ["lead_hours", "variable", "lat", "lon"]
        with xr.open_zarr(metadata["field_store"], consolidated=True) as decoded:
            np.testing.assert_array_equal(decoded.lat.values, fields().lat.values)
            np.testing.assert_array_equal(decoded.lon.values, fields().lon.values)
    with pytest.raises(FileExistsError):
        writer.begin_sample(record())


def test_partial_duplicate_and_wrong_grid_never_publish(tmp_path):
    writer = observer(tmp_path, export_fields=False)
    writer.begin_sample(record())
    with pytest.raises(ValueError, match="incomplete"):
        writer.finish_sample()
    with pytest.raises(ValueError, match="out-of-order"):
        writer.update_step(1, fields(), fields(), fields())
    writer.update_step(0, fields(), fields(), fields())
    with pytest.raises(ValueError, match="out-of-order"):
        writer.update_step(0, fields(), fields(), fields())
    with pytest.raises(ValueError, match="grids differ"):
        writer.update_step(1, fields(), fields().assign_coords(lon=fields().lon + 1), fields())
    assert not list(tmp_path.glob("*.npz"))


def test_plane_selection_and_ambiguous_batch_rejection():
    dataset = fields()
    selected = extract_fields(dataset)
    # Temperature at 850 is twice the 700-hPa wave; Q700 keeps the 700-hPa wave.
    np.testing.assert_array_equal(selected[5], 2 * selected[6])
    with pytest.raises(ValueError, match="one batch"):
        extract_fields(xr.concat([dataset, dataset], dim="batch"))
    with pytest.raises(KeyError):
        extract_fields(dataset.sel(level=[850]))


def test_longitude_endpoint_duplication_is_rejected(tmp_path):
    writer = observer(tmp_path)
    writer.begin_sample(record())
    invalid = fields().assign_coords(lon=np.linspace(0, 360, 16))
    with pytest.raises(ValueError, match="periodic"):
        writer.update_step(0, invalid, invalid, invalid)
