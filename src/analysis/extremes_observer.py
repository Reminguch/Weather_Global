"""Streaming, paired occurrence and intensity diagnostics for fixed extremes.

Scores use fixed historical thresholds and spherical cell areas.  The saved
sufficient statistics retain false alarms, observed-tail errors and raw counts;
undefined ratios are deliberately left to the report instead of set to zero.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np


EVENTS = ("hot", "cold", "wind", "rain")
THRESHOLDS = ("primary", "sensitivity")
DOMAINS = ("global", "land")
BRANCHES = ("baseline", "full")
OBSERVABLES = ("T2m", "WS10", "TP24")
QUANTILES = (.02, .05, .95, .98)
STATISTICS = (
    "hit_area", "miss_area", "false_alarm_area", "correct_negative_area", "eligible_area",
    "hit_count", "miss_count", "false_alarm_count", "correct_negative_count", "eligible_count",
    "tail_absolute_error_sum", "tail_signed_error_sum", "tail_squared_error_sum",
    "tail_truth_sum", "tail_forecast_sum",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(record):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Extreme diagnostic input changed: {path}")
    return path


def utc_datetime(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("Extreme diagnostic times must explicitly use UTC")
    return result.astimezone(timezone.utc)


def calendar_day_index(value):
    """Map month/day to the nonleap climatology; January of any year is valid."""
    return (datetime(2001, value.month, value.day) - datetime(2001, 1, 1)).days


def spherical_cell_weights(latitude, longitude):
    """Global unit-area weights, including nonzero finite polar caps."""
    latitude, longitude = np.asarray(latitude, float), np.asarray(longitude, float)
    if latitude.ndim != 1 or longitude.ndim != 1 or len(latitude) < 2 or len(longitude) < 2:
        raise ValueError("Expected one-dimensional global latitude/longitude coordinates")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("Nonfinite grid coordinates")
    descending = bool(latitude[0] > latitude[-1])
    ordered = latitude[::-1] if descending else latitude
    if np.any(np.diff(ordered) <= 0) or not np.allclose(ordered[[0, -1]], [-90, 90], atol=1e-6):
        raise ValueError("Latitude must be ordered and include both poles")
    if not np.allclose(np.diff(longitude), 360 / len(longitude), atol=1e-6, rtol=0):
        raise ValueError("Longitude must cover one uniform periodic global ring")
    edges = np.concatenate(([-90.], (ordered[:-1] + ordered[1:]) / 2, [90.]))
    bands = np.diff(np.sin(np.deg2rad(edges))) / 2
    if descending:
        bands = bands[::-1]
    result = np.broadcast_to(bands[:, None] / len(longitude), (len(latitude), len(longitude))).copy()
    if not np.all(result > 0) or not np.isclose(result.sum(), 1, atol=1e-12):
        raise ValueError("Invalid spherical cell areas")
    return result


def occurrence_statistics(truth, forecast, threshold, eligible, weights, *, cold=False):
    """Additive full-domain contingency and truth-tail intensity statistics."""
    truth, forecast, threshold, weights = (np.asarray(x, np.float64) for x in
                                          (truth, forecast, threshold, weights))
    eligible = np.asarray(eligible, bool)
    if not all(x.shape == truth.shape for x in (forecast, threshold, eligible, weights)):
        raise ValueError("Extreme scoring shapes differ")
    if not all(np.isfinite(x).all() for x in (truth, forecast, threshold, weights)):
        raise ValueError("Nonfinite extreme fields or thresholds")
    observed = truth < threshold if cold else truth > threshold
    predicted = forecast < threshold if cold else forecast > threshold
    masks = (eligible & observed & predicted, eligible & observed & ~predicted,
             eligible & ~observed & predicted, eligible & ~observed & ~predicted, eligible)
    tail = eligible & observed
    error = forecast[tail] - truth[tail]
    area = weights[tail]
    values = [weights[mask].sum() for mask in masks] + [np.count_nonzero(mask) for mask in masks]
    values.extend((np.dot(area, np.abs(error)), np.dot(area, error), np.dot(area, error**2),
                   np.dot(area, truth[tail]), np.dot(area, forecast[tail])))
    result = np.asarray(values, np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite extreme sufficient statistics")
    return result


class FrozenClimatology:
    def __init__(self, directory, metadata_record):
        self.directory = Path(directory)
        metadata_path = verify_file(metadata_record)
        self.metadata_record = dict(metadata_record)
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata.get("status") != "complete" or self.metadata.get("quantiles") != list(QUANTILES):
            raise ValueError("Climatology must be complete with the prescribed quantile ordering")
        self.latitude = np.load(self.directory / "lat.npy", allow_pickle=False)
        self.longitude = np.load(self.directory / "lon.npy", allow_pickle=False)
        records = self.metadata["arrays"]
        if isinstance(records, dict):
            records = list(records.values())
        checked = set()
        for record in records:
            record = dict(record)
            path = Path(record["path"])
            if not path.is_absolute():
                path = self.directory / path
            record["path"] = str(path)
            verify_file(record)
            checked.add(path.resolve())
        self.arrays = {}
        for observable in OBSERVABLES:
            for hour in (0, 6, 12, 18):
                path = self.directory / f"{observable}_h{hour:02d}.npy"
                if path.resolve() not in checked:
                    raise ValueError(f"Missing climatology array integrity record: {path}")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != (365, 4, len(self.latitude), len(self.longitude)):
                    raise ValueError(f"Unexpected threshold dimensions: {path}")
                self.arrays[observable, hour] = array
        for coordinate in ("lat.npy", "lon.npy"):
            if (self.directory / coordinate).resolve() not in checked:
                raise ValueError(f"Missing climatology coordinate integrity record: {coordinate}")

    def thresholds(self, observable, valid_time):
        values = np.asarray(self.arrays[observable, valid_time.hour][calendar_day_index(valid_time)], np.float64)
        if not np.isfinite(values).all() or np.any(np.diff(values, axis=0) < 0):
            raise ValueError("Nonfinite or unordered climatological quantiles")
        return values


def surface_plane(dataset, name):
    field = dataset[name]
    for dimension in tuple(field.dims):
        if dimension not in ("lat", "lon"):
            if field.sizes[dimension] != 1:
                raise ValueError(f"Expected one {dimension} in {name}")
            field = field.isel({dimension: 0}, drop=True)
    values = np.asarray(field.transpose("lat", "lon").values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite physical field: {name}")
    return values


class ExtremesObserver:
    def __init__(self, destination, identity, climatology, land_mask, *, target_steps=40, provenance=None):
        self.destination = Path(destination) / "extremes"
        self.identity = dict(identity)
        for key in ("protocol_hash", "model_id", "checkpoint_sha256", "baseline_checkpoint_sha256"):
            if not self.identity.get(key):
                raise ValueError(f"Missing extreme provenance: {key}")
        self.climatology = climatology
        self.latitude = np.asarray(climatology.latitude)
        self.longitude = np.asarray(climatology.longitude)
        self.weights = spherical_cell_weights(self.latitude, self.longitude)
        land_mask = np.asarray(land_mask)
        if land_mask.shape != self.weights.shape or not np.isfinite(land_mask).all():
            raise ValueError("Invalid land-sea mask")
        if np.any(land_mask < 0) or np.any(land_mask > 1):
            raise ValueError("Land-sea fractions must lie in [0,1]")
        self.domain_masks = (np.ones_like(land_mask, dtype=bool), land_mask >= .5)
        if type(target_steps) is not int or target_steps < 1:
            raise ValueError("target_steps must be positive")
        self.target_steps = target_steps
        self.provenance = dict(provenance or {})
        self.sample = None

    def begin_sample(self, metadata):
        if self.sample is not None:
            raise ValueError("Previous extreme sample is unfinished")
        identity = metadata["initialization_id"]
        if not re.fullmatch(r"\d{8}T\d{6}Z", identity):
            raise ValueError("Invalid initialization identifier")
        initial = utc_datetime(metadata["initialization_time"])
        leads = list(range(6, 6 * self.target_steps + 1, 6))
        times = [utc_datetime(t) for t in metadata["valid_times"]]
        if metadata["lead_hours"] != leads or times != [initial + timedelta(hours=h) for h in leads]:
            raise ValueError("Incomplete or inconsistent extreme forecast times")
        if initial.strftime("%Y%m%dT%H%M%SZ") != identity:
            raise ValueError("Initialization identifier differs from its timestamp")
        path = self.destination / f"{identity}.npz"
        if path.exists() or path.with_suffix(".partial.npz").exists():
            raise FileExistsError(f"Existing extreme output: {path}")
        self.sample = dict(metadata)
        self.times = times
        self.path = path
        self.stats = np.full((self.target_steps, 4, 2, 2, 2, len(STATISTICS)), np.nan)
        self.valid = np.zeros((self.target_steps, 4), bool)
        self.negative_tp6_counts = np.zeros((self.target_steps, 3), np.int64)
        self.rain_windows = [deque(maxlen=4) for _ in range(3)]
        self.next_step = 0

    def update_step(self, step_index, truth, baseline, full):
        if self.sample is None or step_index != self.next_step or step_index >= self.target_steps:
            raise ValueError("Missing, duplicate or out-of-order extreme forecast lead")
        valid_time = self.times[step_index]
        fields = []
        for bi, dataset in enumerate((truth, baseline, full)):
            if not np.array_equal(dataset.lat.values, self.latitude) or not np.array_equal(dataset.lon.values, self.longitude):
                raise ValueError("Forecast and climatology grids differ")
            t2m = surface_plane(dataset, "2m_temperature")
            ws10 = np.hypot(surface_plane(dataset, "10m_u_component_of_wind"),
                            surface_plane(dataset, "10m_v_component_of_wind"))
            tp6 = surface_plane(dataset, "total_precipitation_6hr") * 1000
            self.negative_tp6_counts[step_index, bi] = np.count_nonzero(tp6 < 0)
            self.rain_windows[bi].append(tp6)
            tp24 = np.sum(self.rain_windows[bi], axis=0) if len(self.rain_windows[bi]) == 4 else None
            fields.append((t2m, ws10, tp24))
        threshold_fields = {obs: self.climatology.thresholds(obs, valid_time)
                            for obs in OBSERVABLES if obs != "TP24" or step_index >= 3}
        for ei, event in enumerate(EVENTS):
            if event == "rain" and step_index < 3:
                continue
            observable, fi = ("T2m", 0) if event in ("hot", "cold") else (("WS10", 1) if event == "wind" else ("TP24", 2))
            quantiles = threshold_fields[observable]
            if quantiles.shape != (4, *self.weights.shape) or not np.isfinite(quantiles).all():
                raise ValueError("Invalid extreme threshold field")
            eligible_rain = quantiles[2] > 1 if event == "rain" else np.ones_like(self.weights, dtype=bool)
            qi = (0, 1) if event == "cold" else (3, 2)
            for ti, quantile_index in enumerate(qi):
                for di, domain in enumerate(self.domain_masks):
                    eligible = domain & eligible_rain
                    for bi in range(2):
                        self.stats[step_index, ei, ti, di, bi] = occurrence_statistics(
                            fields[0][fi], fields[bi + 1][fi], quantiles[quantile_index], eligible,
                            self.weights, cold=event == "cold")
            self.valid[step_index, ei] = True
        self.next_step += 1

    def finish_sample(self):
        if self.sample is None or self.next_step != self.target_steps:
            raise ValueError("Cannot publish incomplete extreme sample")
        if not np.isfinite(self.stats[self.valid]).all() or not np.isnan(self.stats[~self.valid]).all():
            raise ValueError("Extreme statistics do not match valid-lead mask")
        self.destination.mkdir(parents=True, exist_ok=True)
        metadata = {**self.sample, **self.identity, "status": "complete", "schema_version": 1,
                    "format": "v24_extremes_sufficient_statistics_v1", "provenance": self.provenance,
                    "area_normalization": "Spherical cell areas sum to one globally; finite polar caps",
                    "land_definition": "fixed land-sea fraction >= 0.5",
                    "event_definition": "strictly > q98/q95 for hot,wind,rain; strictly < q02/q05 for cold",
                    "rain_eligibility": "historical local q95(TP24) > 1 mm, identical mask for both thresholds and branches",
                    "rain_accumulation": "sum of four consecutive forecast TP6 values; no pre-initialization rain; leads >=24h",
                    "negative_precipitation": "retained without clipping; counts are diagnostic warnings",
                    "tail_errors": "forecast minus truth, conditional on observed threshold exceedance; false alarms retained separately",
                    "units": {"hot": "K", "cold": "K", "wind": "m s-1", "rain": "mm"}}
        temporary = self.path.with_suffix(".partial.npz")
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, stats=self.stats, valid_step_event=self.valid,
                                events=np.asarray(EVENTS), thresholds=np.asarray(THRESHOLDS),
                                domains=np.asarray(DOMAINS), branches=np.asarray(BRANCHES),
                                statistics=np.asarray(STATISTICS), lead_hours=np.asarray(self.sample["lead_hours"]),
                                negative_tp6_counts=self.negative_tp6_counts,
                                negative_tp6_branches=np.asarray(("truth", *BRANCHES)),
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)
        print(f"[extremes] complete {self.identity['model_id']} {self.sample['initialization_time']}: {self.path}", flush=True)
        self.sample = None


def make_observer(spec, destination, identity):
    manifest = json.loads(verify_file(spec["experiment_manifest"]).read_text())
    config = manifest["extremes"]
    climatology = FrozenClimatology(config["climatology_dir"], config["climatology_metadata"])
    land_mask = np.load(verify_file(config["land_mask"]), allow_pickle=False)
    return ExtremesObserver(destination, identity, climatology, land_mask, target_steps=spec["target_steps"],
                            provenance={"extremes_protocol": config,
                                        "experiment_manifest": spec["experiment_manifest"]})


# Stable entry point used by the frozen evaluation campaign manifests.
factory = make_observer
