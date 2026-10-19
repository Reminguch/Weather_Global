"""Receipt, timestamp, and paired-scoring contracts for the common-grid runner."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from scripts.analyze_models import eval_common_grid_2023 as runner


def staged_truth(directory, timestamp, helper):
    data = xr.Dataset({key: (("lat", "lon"), np.zeros((121, 240), np.float32))
                       for key in helper.DIAGNOSTICS},
                      coords={"lat": helper.DEFAULT_LAT, "lon": helper.DEFAULT_LON},
                      attrs={"valid_time": timestamp})
    path = directory / (runner.timestamp_id(timestamp) + ".nc")
    data.to_netcdf(path)
    receipt = {"schema_version": 1, "path": str(path.resolve()),
               "sha256": runner.sha256(path), "size_bytes": path.stat().st_size,
               "kind": "common_truth", "timestamp": timestamp}
    Path(str(path) + ".READY.json").write_text(json.dumps(receipt))
    return data, path, receipt


def test_observer_preserves_pairing_and_common_truth_identity(tmp_path):
    helper = runner.load_helper()
    times = ["2023-01-01T06:00:00Z", "2023-01-01T12:00:00Z"]
    staged = [staged_truth(tmp_path, stamp, helper) for stamp in times]
    observer = runner.CommonGridObserver(runner.CommonTruth(tmp_path), helper, 2)
    observer.begin_sample({"initialization_time": "2023-01-01T00:00:00Z",
                           "valid_times": times, "lead_hours": [6, 12]})
    for index, (truth, _, _) in enumerate(staged):
        observer.update_step(index, None, truth + 2, truth)
    observer.finish_sample()
    record = observer.records[0]
    assert record["truth_sha256"] == [receipt["sha256"] for _, _, receipt in staged]
    assert record["valid_times"] == times
    for key in helper.DIAGNOSTICS:
        assert record["mse"]["gc1"][key] == pytest.approx([4., 4.])
        assert record["mse"]["m1"][key] == pytest.approx([0., 0.])


def test_truth_rejects_wrong_receipt_time_and_tampered_file(tmp_path):
    _, path, receipt = staged_truth(tmp_path, "2023-01-01T06:00:00Z", runner.load_helper())
    with pytest.raises(ValueError, match="receipt"):
        runner.verify_ready(path, "2023-01-01T12:00:00Z", "common_truth")
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="receipt"):
        runner.verify_ready(path, receipt["timestamp"], "common_truth")


def test_manifest_preserves_year_boundary_and_rejects_duplicates(tmp_path):
    path = tmp_path / "initializations.json"
    payload = {"schema_version": 1, "initializations": ["2023-12-30T00:00:00Z"]}
    path.write_text(json.dumps(payload))
    record = runner.records_from_manifest(path)[0]
    assert record["valid_times"][-1] == "2024-01-09T00:00:00Z"
    assert record["lead_hours"] == list(range(6, 241, 6))
    payload["initializations"] *= 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique"):
        runner.records_from_manifest(path)


def test_generated_calendar_forcing_uses_valid_time_not_initialization(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "third_party/graphcast"))
    task = SimpleNamespace(forcing_variables=("year_progress_sin", "year_progress_cos",
                                               "day_progress_sin", "day_progress_cos"))
    inputs = xr.Dataset(coords={"lat": [-90., 0., 90.], "lon": [0., 90., 180., 270.]})
    first = runner.native_forcings(inputs, "2023-01-01T06:00:00Z", 6, task)
    second = runner.native_forcings(inputs, "2023-01-01T12:00:00Z", 12, task)
    assert first.time.values[0] == np.timedelta64(6, "h")
    assert second.time.values[0] == np.timedelta64(12, "h")
    assert first.sizes["batch"] == 1
    assert not np.array_equal(first.day_progress_sin, second.day_progress_sin)
