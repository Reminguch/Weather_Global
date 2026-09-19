"""Bounded spectral diagnostics and physical-field export during evaluation.

Each completed initialization publishes one NPZ containing sufficient
statistics, plus nine physical FP32 planes in a chunked Zarr store. A partial
initialization is never advertised as complete. These are raw-field zonal
diagnostics, not the spherical-harmonic confirmation study.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import numpy as np

from .spectral_diagnostics import zonal_spectral_metadata, zonal_spectral_statistics


FIELD_SPECS = (
    ("T2m", "2m_temperature", None, "K"),
    ("U10", "10m_u_component_of_wind", None, "m s-1"),
    ("V10", "10m_v_component_of_wind", None, "m s-1"),
    ("MSLP", "mean_sea_level_pressure", None, "Pa"),
    ("TP6", "total_precipitation_6hr", None, "m"),
    ("T850", "temperature", 850, "K"),
    ("Q700", "specific_humidity", 700, "kg kg-1"),
    ("U850", "u_component_of_wind", 850, "m s-1"),
    ("V850", "v_component_of_wind", 850, "m s-1"),
)
SPECTRAL_VARIABLES = tuple(spec[0] for spec in FIELD_SPECS) + ("WS10", "WS850")
SPECTRAL_UNITS = tuple(spec[3] for spec in FIELD_SPECS) + ("m s-1", "m s-1")


def extract_fields(dataset):
    """Select actual pressure levels, rejecting ambiguous batches or times."""
    planes = []
    for _, name, level, _ in FIELD_SPECS:
        field = dataset[name]
        if level is not None:
            field = field.sel(level=level)
        for dimension in tuple(field.dims):
            if dimension not in ("lat", "lon"):
                if field.sizes[dimension] != 1:
                    raise ValueError(f"Expected one {dimension} in {name}")
                field = field.isel({dimension: 0}, drop=True)
        planes.append(np.asarray(field.transpose("lat", "lon").values, dtype=np.float32))
    result = np.stack(planes)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite physical diagnostic fields")
    return result


def spectral_fields(primitive):
    """Wind speed must be derived in physical space before the transform."""
    return np.concatenate((primitive, np.stack((
        np.hypot(primitive[1], primitive[2]),
        np.hypot(primitive[7], primitive[8]),
    ))), axis=0)


class SpectralObserver:
    def __init__(self, output_dir, metadata, *, target_steps=40, export_fields=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        for key in ("protocol_hash", "model_id", "checkpoint_sha256"):
            if not self.metadata.get(key):
                raise ValueError(f"Missing observer provenance: {key}")
        if target_steps < 1:
            raise ValueError("target_steps must be positive")
        self.target_steps = target_steps
        self.export_fields = export_fields
        self._sample = None

    def begin_sample(self, metadata):
        if self._sample is not None:
            raise ValueError("Previous initialization is unfinished")
        identity = metadata["initialization_id"]
        if not re.fullmatch(r"\d{8}T\d{6}Z", identity):
            raise ValueError("Invalid initialization ID")
        if metadata["lead_hours"] != list(range(6, 6 * self.target_steps + 1, 6)):
            raise ValueError("Expected complete six-hour forecast leads")
        if len(metadata["valid_times"]) != self.target_steps:
            raise ValueError("Missing valid times")
        self._destination = self.output_dir / f"{identity}.npz"
        self._field_destination = self.output_dir / f"{identity}.fields.zarr"
        self._field_partial = self.output_dir / f"{identity}.fields.partial.zarr"
        if self._destination.exists() or self._field_destination.exists() or self._field_partial.exists():
            raise FileExistsError(f"Existing initialization output: {identity}")
        self._sample = {**metadata, **self.metadata, "spectral_metadata": zonal_spectral_metadata()}
        self._records = []
        self._group = None
        self._latitude = self._longitude = None

    def _check_grid(self, datasets):
        latitude = np.asarray(datasets[0].lat.values, dtype=np.float64)
        longitude = np.asarray(datasets[0].lon.values, dtype=np.float64)
        if latitude.ndim != 1 or longitude.ndim != 1 or longitude.size < 4:
            raise ValueError("Spectral diagnostics require a regular latitude/longitude grid")
        spacing = 360.0 / longitude.size
        if not np.isfinite(longitude).all() or not np.allclose(np.diff(longitude), spacing, rtol=0, atol=1e-5):
            raise ValueError("Longitude must cover one uniform, periodic 360-degree ring")
        if not np.isfinite(latitude).all() or np.any(np.abs(latitude) > 90) or not (
            np.all(np.diff(latitude) > 0) or np.all(np.diff(latitude) < 0)
        ):
            raise ValueError("Latitude must be unique, ordered and within [-90,90]")
        for dataset in datasets[1:]:
            if not np.array_equal(dataset.lat.values, latitude) or not np.array_equal(dataset.lon.values, longitude):
                raise ValueError("Truth, baseline and model grids differ")
        if self._latitude is not None:
            if not np.array_equal(self._latitude, latitude) or not np.array_equal(self._longitude, longitude):
                raise ValueError("Grid changed within initialization")
        else:
            self._latitude, self._longitude = latitude, longitude

    def _create_field_store(self):
        import zarr
        from numcodecs import Blosc

        self._group = zarr.open_group(str(self._field_partial), mode="w", zarr_format=2)
        self._group.attrs.update({**self._sample, "status": "partial", "field_units": list(SPECTRAL_UNITS[:9]),
                                 "description": "Physical FP32 fields, pressure levels in hPa, TP6 in metres"})
        for key, values in (("lat", self._latitude), ("lon", self._longitude),
                            ("lead_hours", self._sample["lead_hours"]),
                            ("variable", np.asarray(SPECTRAL_VARIABLES[:9], dtype="U8"))):
            array = self._group.create_array(key, data=np.asarray(values), fill_value=None, overwrite=False)
            array.attrs["_ARRAY_DIMENSIONS"] = [key]
        for branch in ("truth", "baseline", "full"):
            array = self._group.create_array(
                branch, shape=(self.target_steps, 9, len(self._latitude), len(self._longitude)),
                chunks=(1, 1, len(self._latitude), len(self._longitude)), dtype="f4",
                compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE), fill_value=np.nan,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["lead_hours", "variable", "lat", "lon"]

    def update_step(self, step_index, truth, baseline, full):
        if self._sample is None:
            raise ValueError("begin_sample must precede predictions")
        if step_index != len(self._records) or step_index >= self.target_steps:
            raise ValueError("Missing, duplicate or out-of-order forecast lead")
        datasets = (truth, baseline, full)
        self._check_grid(datasets)
        fields = [extract_fields(dataset) for dataset in datasets]
        if self.export_fields:
            if self._group is None:
                self._create_field_store()
            for branch, values in zip(("truth", "baseline", "full"), fields, strict=True):
                self._group[branch][step_index] = values
        transformed = [spectral_fields(values) for values in fields]
        baseline_stats = zonal_spectral_statistics(transformed[0], transformed[1], self._latitude)
        full_stats = zonal_spectral_statistics(transformed[0], transformed[2], self._latitude)
        self._wavenumber = baseline_stats["wavenumber"]
        self._records.append({
            "truth_power": baseline_stats["power_truth"],
            "baseline_power": baseline_stats["power_forecast"], "full_power": full_stats["power_forecast"],
            "baseline_cross_power": baseline_stats["cross_power"], "full_cross_power": full_stats["cross_power"],
            "baseline_error_power": baseline_stats["error_power"], "full_error_power": full_stats["error_power"],
        })

    def finish_sample(self):
        if self._sample is None or len(self._records) != self.target_steps:
            raise ValueError("Cannot publish incomplete initialization")
        metadata = {**self._sample, "status": "complete", "field_store": (
            str(self._field_destination) if self.export_fields else None), "field_dtype": "float32"}
        if self.export_fields:
            import zarr
            self._group.attrs.update(status="complete")
            zarr.consolidate_metadata(str(self._field_partial))
            self._group = None
            self._field_partial.rename(self._field_destination)
        payload = {key: np.stack([record[key] for record in self._records]) for key in self._records[0]}
        payload.update(wavenumber=self._wavenumber, lead_hours=np.asarray(self._sample["lead_hours"]),
                       variables=np.asarray(SPECTRAL_VARIABLES), units=np.asarray(SPECTRAL_UNITS),
                       metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
        temporary = self._destination.with_suffix(".partial.npz")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self._destination)
        print(f"[spectra] complete {self._sample['model_id']} {self._sample['initialization_time']}: {self._destination}", flush=True)
        self._sample = None
        self._records = []
