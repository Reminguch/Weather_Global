"""Streaming ERA5 preparation, immutable frame stores and causal origin manifests."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
from .config import FIELDS, FORCING_FIELDS
from .io import atomic_bytes, digest, sha256, write_json, read_json

STEP = np.timedelta64(6, "h")
ARCO_SOURCE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DECODED_UNITS = {"temperature": "K", "geopotential": "m2/s2", "u_component_of_wind": "m/s",
                 "v_component_of_wind": "m/s", "specific_humidity": "kg/kg",
                 "specific_cloud_ice_water_content": "kg/kg", "specific_cloud_liquid_water_content": "kg/kg",
                 "sea_surface_temperature": "K", "sea_ice_cover": "1"}


def split_for(time):
    year = int(str(np.datetime64(time, "Y"))[:4])
    return "train" if 2015 <= year <= 2021 else {2022: "val", 2023: "test"}.get(year)


def stamp(time):
    return str(np.datetime64(time, "h"))


def complete_segments(times, length=96):
    """No padding. Report omitted records at gaps and boundaries explicitly."""
    t = np.asarray(times, dtype="datetime64[h]")
    groups, omitted, current = [], [], []
    for i in range(len(t)):
        if current and (t[i] - t[current[-1]] != STEP or split_for(t[i]) != split_for(t[current[-1]])):
            omitted.extend(current)
            current = []
        current.append(i)
        if len(current) == length:
            groups.append(current)
            current = []
    omitted.extend(current)
    return groups, omitted


def eligible_origins(times, split, steps, *, warm_hours=24, daily=True):
    available = set(np.asarray(times, dtype="datetime64[h]"))
    origins = []
    for t in sorted(available):
        if split_for(t) != split or (daily and int(t.astype(np.int64)) % 24):
            continue
        # Warm history needs its own causal lag, so oldest accessed data is t-48h.
        required = [t + i * STEP for i in range(-warm_hours // 6 - 4, steps + 1)]
        if all(x in available and split_for(x) == split for x in required):
            origins.append(stamp(t))
    return origins


def seasonal_order(origins, seed=22):
    rng = np.random.default_rng(seed)
    seasons = [[] for _ in range(4)]
    for t in origins:
        month = int(t[5:7])
        seasons[(month % 12) // 3].append(t)
    for values in seasons:
        rng.shuffle(values)
    return [values[i] for i in range(max(map(len, seasons), default=0))
            for values in seasons if i < len(values)]


def origin_manifest(times, split, count, steps, dataset_id):
    candidates = seasonal_order(eligible_origins(times, split, steps))
    if len(candidates) < count:
        raise ValueError(f"Need {count} eligible {split} origins, found {len(candidates)}")
    origins = candidates[:count]
    return {"version": "origins_v1", "split": split, "dataset_id": dataset_id,
            "origins": origins, "steps": steps, "warm_hours": 24,
            "selection": "seed22_seasonally_balanced_daily", "identity": digest(origins)}


class PreparedStore:
    def __init__(self, root, *, verify=True):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        ready = read_json(self.root / "READY.json")
        if ready["manifest_sha256"] != sha256(self.root / "manifest.json"):
            raise ValueError("Prepared store completeness identity differs")
        self.identity = digest(self.manifest)
        self.records = {r["time"]: r for r in self.manifest["records"]}
        self.times = np.array(sorted(self.records), dtype="datetime64[h]")
        self.verify = verify

    def frame(self, time):
        record = self.records[stamp(time)]
        path = self.root / record["file"]
        if self.verify and sha256(path) != record["sha256"]:
            raise ValueError(f"Prepared frame changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    def inputs_and_forcing(self, model, time):
        t = np.datetime64(time, "h")
        lag = t - np.timedelta64(24, "h")
        if split_for(t) != split_for(lag):
            raise ValueError("Forcing history crosses split boundary")
        origin, past = self.frame(t), self.frame(lag)
        sim_time = np.asarray(model.datetime64_to_sim_time(t)).astype(np.float32)
        inputs = {k: origin[k] for k in FIELDS}
        inputs["sim_time"] = sim_time
        forcing = {k: past[k][None, ...] for k in FORCING_FIELDS}
        forcing["sim_time"] = sim_time
        return inputs, forcing


def require_complete_experiment_data(store):
    """Partial prepared shards may be inspected, but cannot start production."""
    expected = np.arange(np.datetime64("2015-01-01T00", "h"),
                         np.datetime64("2024-01-01T00", "h"), STEP)
    if not np.array_equal(store.times, expected):
        missing = np.setdiff1d(expected, store.times)
        extra = np.setdiff1d(store.times, expected)
        raise ValueError(f"Production requires all 13,148 six-hour frames from 2015–2023; "
                         f"missing={len(missing)}, unexpected={len(extra)}")


def prepare_era5(source, model, root, start, end):
    """Read/regrid exactly one timestamp at a time, including all seven fields.

    A store is immutable and READY is written only after its entire declared interval.
    Supply separate roots for time shards and merge via merge_prepared afterwards.
    """
    import xarray as xr
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Use a new prepared shard directory: {root}")
    opts = {"chunks": None}
    if str(source).startswith("gs://"):
        opts["storage_options"] = {"token": "anon"}
    ds = xr.open_zarr(source, **opts)
    ds = ds.rename({k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in ds.dims})
    missing = set(FIELDS + FORCING_FIELDS) - set(ds)
    if missing:
        raise ValueError(f"ERA5 store lacks required NeuralGCM fields: {sorted(missing)}")
    ds = ds[list(FIELDS + FORCING_FIELDS)].sel(level=np.asarray(model.data_coords.vertical.centers))
    source_grid = spherical_harmonic.Grid(
        longitude_nodes=ds.sizes["longitude"], latitude_nodes=ds.sizes["latitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude))
    regridder = horizontal_interpolation.ConservativeRegridder(source_grid, model.data_coords.horizontal, skipna=True)
    times = np.arange(np.datetime64(start, "h"), np.datetime64(end, "h"), STEP)
    if not len(times):
        raise ValueError("Empty preparation interval")
    records = []
    for t in times:
        frame = xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(ds.sel(time=t).compute(), regridder))
        values = {k: np.asarray(frame[k].transpose(*(("level",) if k in FIELDS else ()),
                  "longitude", "latitude").values, dtype=np.float32) for k in FIELDS + FORCING_FIELDS}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError(f"Nonfinite regridded ERA5 at {t}")
        name = "frames/" + stamp(t).replace(":", "") + ".npz"
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **values)
        atomic_bytes(root / name, buffer.getvalue())
        records.append({"time": stamp(t), "file": name, "sha256": sha256(root / name)})
    manifest = {"version": "prepared_era5_v1", "source": str(source),
                "source_metadata": dict(ds.attrs), "interval": [stamp(start), stamp(end)],
                "data_grid": {"longitude": model.data_coords.horizontal.longitudes.tolist(),
                              "latitude": model.data_coords.horizontal.latitudes.tolist(),
                              "levels": np.asarray(model.data_coords.vertical.centers).tolist()},
                "variables": list(FIELDS + FORCING_FIELDS), "units": DECODED_UNITS, "dtype": "float32",
                "regrid": "dinosaur.ConservativeRegridder_skipna_then_fill_nearest",
                "records": records}
    write_json(root / "manifest.json", manifest, immutable=True)
    write_json(root / "READY.json", {"manifest_sha256": sha256(root / "manifest.json")}, immutable=True)
    return manifest


def merge_prepared(shards, output):
    root = Path(output)
    stores = [PreparedStore(s) for s in shards]
    if not stores:
        raise ValueError("No prepared shards")
    manifest = dict(stores[0].manifest)
    records = []
    for store in stores:
        if store.manifest["data_grid"] != manifest["data_grid"] or store.manifest["source"] != manifest["source"]:
            raise ValueError("Incompatible prepared shards")
        for record in store.manifest["records"]:
            record = dict(record, file=str((store.root / record["file"]).resolve()))
            records.append(record)
    records.sort(key=lambda r: r["time"])
    if len({r["time"] for r in records}) != len(records):
        raise ValueError("Overlapping prepared shards")
    manifest.update(records=records, interval=[records[0]["time"], stamp(np.datetime64(records[-1]["time"]) + STEP)])
    write_json(root / "manifest.json", manifest, immutable=True)
    write_json(root / "READY.json", {"manifest_sha256": sha256(root / "manifest.json")}, immutable=True)


def prepare_era5_pair(source, models, roots, start, end):
    """Download each original ERA5 timestamp once and prepare both model grids.

    Per-frame receipts permit exact restart of interrupted monthly downloads.
    These receipts do not make a partial monthly store READY.
    """
    import time
    import xarray as xr
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
    options = {"chunks": None}
    if str(source).startswith("gs://"):
        options["storage_options"] = {"token": "anon"}
    ds = xr.open_zarr(source, **options)
    missing = set(FIELDS + FORCING_FIELDS) - set(ds)
    if missing:
        raise ValueError(f"Missing NeuralGCM ERA5 variables: {missing}")
    first = next(iter(models.values()))
    levels = np.asarray(first.data_coords.vertical.centers)
    if any(not np.array_equal(model.data_coords.vertical.centers, levels) for model in models.values()):
        raise ValueError("Pair preparation requires identical pressure levels")
    ds = ds[list(FIELDS + FORCING_FIELDS)].sel(level=levels)
    source_grid = spherical_harmonic.Grid(longitude_nodes=ds.sizes["longitude"], latitude_nodes=ds.sizes["latitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude))
    regridders = {rid: horizontal_interpolation.ConservativeRegridder(source_grid, model.data_coords.horizontal, skipna=True)
                  for rid, model in models.items()}
    times = np.arange(np.datetime64(start, "h"), np.datetime64(end, "h"), STEP)
    if not len(times) or int(times[0].astype(int)) % 6:
        raise ValueError("Nonempty interval must start on a six-hour boundary")
    records, timings = {rid: [] for rid in models}, []
    roots = {rid: Path(path) for rid, path in roots.items()}
    contract = {"source": source, "start": stamp(start), "end": stamp(end), "fields": list(FIELDS + FORCING_FIELDS),
                "levels": levels.tolist(), "method": "conservative_skipna_then_nearest_v1"}
    for rid, model in models.items():
        grid = model.data_coords.horizontal
        write_json(roots[rid] / "download_contract.json", {**contract, "grid": {
            "longitude": grid.longitudes.tolist(), "latitude": grid.latitudes.tolist()}}, immutable=True)
    for t in times:
        started = time.perf_counter()
        name = f"{split_for(t)}/frames/{stamp(t)}.npz"
        needed = []
        for rid in models:
            receipt = roots[rid] / (name + ".json")
            if receipt.exists():
                record = read_json(receipt)
                if sha256(roots[rid] / name) != record["sha256"]:
                    raise ValueError(f"Previously downloaded frame changed: {receipt}")
                records[rid].append(record)
            else:
                needed.append(rid)
        if not needed:
            continue
        frame = ds.sel(time=t).compute()  # One global raw frame, shared by the two resolutions.
        downloaded = time.perf_counter()
        for rid in needed:
            regridded = xarray_utils.fill_nan_with_nearest(xarray_utils.regrid_horizontal(frame, regridders[rid]))
            arrays = {k: np.asarray(regridded[k].transpose(*(("level",) if k in FIELDS else ()),
                      "longitude", "latitude").values, dtype=np.float32) for k in FIELDS + FORCING_FIELDS}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError(f"Nonfinite prepared frame: {rid}/{t}")
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            atomic_bytes(roots[rid] / name, buffer.getvalue())
            record = {"time": stamp(t), "file": name, "sha256": sha256(roots[rid] / name),
                      "bytes": (roots[rid] / name).stat().st_size}
            write_json(roots[rid] / (name + ".json"), record, immutable=True)
            records[rid].append(record)
        timings.append({"time": stamp(t), "download_seconds": downloaded - started,
                        "total_seconds": time.perf_counter() - started, "raw_frame_bytes": frame.nbytes})
        print(f"prepared {stamp(t)} both resolutions in {timings[-1]['total_seconds']:.2f}s", flush=True)
    for rid, model in models.items():
        grid = model.data_coords.horizontal
        records[rid].sort(key=lambda r: r["time"])
        manifest = {"version": "prepared_era5_v1", "source": str(source), "source_metadata": dict(ds.attrs),
                    "interval": [stamp(start), stamp(end)], "data_grid": {"longitude": grid.longitudes.tolist(),
                    "latitude": grid.latitudes.tolist(), "levels": levels.tolist()},
                    "variables": list(FIELDS + FORCING_FIELDS), "units": DECODED_UNITS, "dtype": "float32",
                    "regrid": "dinosaur.ConservativeRegridder_skipna_then_fill_nearest", "records": records[rid]}
        write_json(roots[rid] / "manifest.json", manifest, immutable=True)
        write_json(roots[rid] / "READY.json", {"manifest_sha256": sha256(roots[rid] / "manifest.json")}, immutable=True)
    return {"interval": [stamp(start), stamp(end)], "new_frame_timings": timings,
            "resolutions": {rid: {"root": str(roots[rid]), "records": len(records[rid]),
                                  "stored_bytes": sum(r["bytes"] for r in records[rid])} for rid in models}}
