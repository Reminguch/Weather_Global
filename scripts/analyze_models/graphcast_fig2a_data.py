"""Read native physical ERA5 fields from the existing year-sharded store."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def array_sha(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode())
    digest.update(memoryview(values.view(np.uint8)).cast("B"))
    return digest.hexdigest()


def dataset_sha(dataset):
    identity = {name: {"dims": list(dataset[name].dims), "sha256": array_sha(dataset[name].values)}
                for name in sorted(dataset.variables)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class LocalERA5:
    def __init__(self, root, expected_identity=None):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata["resolution"] != .25 or len(self.metadata["pressure_levels"]) != 37:
            raise ValueError("Expected native 0.25-degree, 37-level physical fields")
        self.coords = {n: np.load(self.root / "coords" / f"{n}.npy") for n in ("lat", "lon", "level")}
        np.testing.assert_array_equal(np.sort(self.coords["lat"]), np.arange(-90, 90.25, .25))
        np.testing.assert_array_equal(self.coords["lon"], np.arange(0, 360, .25))
        self.year_times, self.validity = {}, {}
        files = [self.root / "metadata.json"]
        files += [self.root / "coords" / f"{n}.npy" for n in self.coords]
        for year in (2018, 2019):
            base = self.root / "years" / str(year)
            self.year_times[year] = np.load(base / "coords/time.npy")
            self.validity[year] = json.loads((base / "validity.json").read_text())
            files += [base / "coords/time.npy", base / "validity.json"]
        self.identity = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        if expected_identity is not None and self.identity != expected_identity:
            raise ValueError("Prepared ERA5 metadata/coordinates/validity changed")

    def field(self, name, stamp=None, level=None):
        info = self.metadata["variables"][name]
        dims = list(info["dims"])
        if "time" in dims:
            stamp = pd.Timestamp(stamp).tz_localize(None)
            times = self.year_times[stamp.year]
            index = int(np.searchsorted(times, stamp.to_datetime64()))
            if index == len(times) or times[index] != stamp.to_datetime64():
                raise ValueError(f"Missing source timestamp: {stamp}")
            validity = self.validity[stamp.year]
            if not validity["written_time_mask"][index] or not validity["time_finite_by_variable"][name][index]:
                raise ValueError(f"Invalid source field: {name} at {stamp}")
            path = self.root / "years" / str(stamp.year) / "vars" / f"{name}.npy"
            dims.remove("time")
        else:
            index = slice(None)
            path = self.root / "vars" / f"{name}.npy"
        mapped = np.load(path, mmap_mode="r")
        values = mapped[index] if "time" in info["dims"] else mapped
        if level is not None:
            values = np.take(values, int(np.flatnonzero(self.coords["level"] == level)[0]), axis=dims.index("level"))
            dims.remove("level")
        if self.coords["lat"][0] > self.coords["lat"][-1]:
            values = np.flip(values, axis=dims.index("lat"))
        values = np.array(values, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite physical field {name}")
        return values, dims

    def inputs(self, stamp):
        times = pd.date_range(pd.Timestamp(stamp).tz_localize(None) - pd.Timedelta("6h"), periods=2, freq="6h")
        variables = {}
        names = set(self.metadata["task_input_variables"]) - set(self.metadata["task_forcing_variables"])
        for name in sorted(names):
            if "time" in self.metadata["variables"][name]["dims"]:
                first, dims = self.field(name, times[0])
                second, _ = self.field(name, times[1])
                variables[name] = (("time", *dims), np.stack([first, second]))
            else:
                values, dims = self.field(name)
                variables[name] = (dims, values)
        return xr.Dataset(variables, coords={"time": times, "lat": np.sort(self.coords["lat"]),
                                            "lon": self.coords["lon"], "level": self.coords["level"]})

    def truth(self, stamp):
        values, dims = self.field("geopotential", stamp, level=500)
        if dims != ["lat", "lon"]:
            raise ValueError("Unexpected Z500 dimension order")
        return values
