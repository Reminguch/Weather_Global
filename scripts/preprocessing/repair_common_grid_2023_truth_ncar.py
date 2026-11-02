#!/usr/bin/env python3
"""Recover three common-grid truth records from independently curated NCAR ERA5.

Only the five independently identified stale pressure-level diagnostics are
replaced. The original campaign is immutable. HTTP range reads fetch four
compressed hourly HDF5 chunks per variable, not the full daily files. All 37
levels are retained for an independent native-data audit. A midnight control
must agree with the original archive before any corrected truth is published.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from scripts.analyze_models.common_grid_metrics import DIAGNOSTICS, _operators
from scripts.analyze_models.eval_common_grid_2023 import CommonTruth, sha256, verify_ready

OLD = ROOT / "data/graphcast/graphcast/dataset/common_grid_2023_20261015/common_truth"
NATIVE = ROOT / "data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr"
CAMPAIGN = ROOT / "artifacts/evaluations/graphcast_common_grid_2023_truth_repair_20261018"
LEVELS = np.array([1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,
                   400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000])
SOURCES = {
    "geopotential": {"code": "128_129_z", "field": "Z", "units": "m**2 s**-2"},
    "temperature": {"code": "128_130_t", "field": "T", "units": "K"},
    "specific_humidity": {"code": "128_133_q", "field": "Q", "units": "kg kg**-1"},
}
REPLACED = ["Z500", "Z850", "T500", "Q700", "Q850"]
# Precision-control limits in the plotted SI units, intentionally far below
# the diagnosed stale-field discrepancy. These are not fitted to the result.
CONTROL_RMSE_LIMIT = {"Z500": 1.0, "Z850": 1.0, "T500": .01, "Q700": 1e-6, "Q850": 1e-6}


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def serializable(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [serializable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def publish_json(path, body):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def utc(hour):
    return f"2023-11-10T{hour:02d}:00:00Z"


def filename(hour):
    return f"20231110T{hour:02d}0000Z"


def download(out):
    import fsspec
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    records = []
    for variable, spec in SOURCES.items():
        url = ("https://nsf-ncar-era5.s3.amazonaws.com/e5.oper.an.pl/202311/"
               f"e5.oper.an.pl.{spec['code']}.ll025sc.2023111000_2023111023.nc")
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            headers = dict(response.headers)
        # A fresh range-backed handle per variable bounds the remote transfer;
        # each read below is exactly one compressed all-level hourly chunk.
        with fsspec.open(url, "rb", block_size=65536) as source, h5py.File(source, "r") as handle:
            lat, lon = handle["latitude"][:], handle["longitude"][:]
            levels, hours = handle["level"][:], handle["time"][:]
            coordinate_attrs = {name:{k:serializable(v) for k,v in handle[name].attrs.items()
                                     if not k.startswith("_") and k != "DIMENSION_LIST"}
                                for name in ("latitude","longitude","level")}
            time_attrs = {k: serializable(v) for k,v in handle["time"].attrs.items() if not k.startswith("_")}
            assert np.array_equal(lat, np.arange(90, -90.25, -.25))
            assert np.array_equal(lon, np.arange(0, 360, .25))
            assert np.array_equal(levels, LEVELS)
            assert str(coordinate_attrs["level"]["units"]).lower() in ("millibar","millibars","hpa","mbar")
            assert coordinate_attrs["latitude"]["units"] == "degrees_north"
            assert coordinate_attrs["longitude"]["units"] == "degrees_east"
            assert time_attrs["units"] == "hours since 1900-01-01 00:00:00"
            actual = np.datetime64("1900-01-01T00", "h") + hours.astype("timedelta64[h]")
            assert np.array_equal(actual, np.arange(np.datetime64("2023-11-10T00", "h"),
                                                   np.datetime64("2023-11-11T00", "h")))
            field = handle[spec["field"]]
            attrs = {k: serializable(v) for k,v in field.attrs.items()
                     if not k.startswith("_") and k != "DIMENSION_LIST"}
            assert field.shape == (24,37,721,1440) and field.chunks == (1,37,721,1440)
            assert np.dtype(field.dtype) == np.dtype("float32")
            assert attrs["units"] == spec["units"], attrs
            source_record = {"variable": variable, "url": url,
                             "http_headers": headers, "field": spec["field"],
                             "shape": list(field.shape), "chunks": list(field.chunks),
                             "dtype": str(field.dtype), "attributes": attrs,
                             "time_attributes": time_attrs,
                             "coordinate_attributes":coordinate_attrs,
                             "global_attributes": {k: serializable(v) for k,v in handle.attrs.items()},
                             "coordinate_sha256": {"latitude": array_sha(lat), "longitude": array_sha(lon),
                                                   "level": array_sha(levels), "time": array_sha(hours)}}
            publish_json(out / f"{variable}.source.json", source_record)
            coord_path = out / "coordinates.npz"
            if not coord_path.exists():
                np.savez(coord_path, latitude=lat, longitude=lon, level=levels, time=actual)
            else:
                with np.load(coord_path) as previous:
                    for key, value in (("latitude",lat),("longitude",lon),("level",levels),("time",actual)):
                        assert np.array_equal(previous[key], value)
            for hour in (0,6,12,18):
                path = out / f"{filename(hour)}_{variable}.npy"
                receipt_path = Path(str(path) + ".json")
                if path.exists() and receipt_path.exists():
                    record = json.loads(receipt_path.read_text())
                    assert record["sha256"] == sha256(path)
                    assert record["url"] == url and record["valid_time"] == utc(hour)
                    print(f"Verified cached {variable} {utc(hour)}", flush=True)
                else:
                    print(f"Reading NCAR {variable} {utc(hour)} (one compressed hourly chunk)", flush=True)
                    values = field[hour,:,:,:]
                    assert values.shape == (37,721,1440) and values.dtype == np.float32
                    assert np.isfinite(values).all() and np.max(np.abs(values)) < 1e10
                    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
                    with temporary.open("wb") as stream:
                        np.save(stream, values, allow_pickle=False)
                    os.replace(temporary, path)
                    record = {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
                              "url": url, "source_headers": headers, "variable": variable,
                              "valid_time": utc(hour), "array_sha256": array_sha(values),
                              "level_sha256": {str(int(level)):array_sha(values[i]) for i,level in enumerate(levels)},
                              "shape": list(values.shape), "dtype": str(values.dtype), "units": spec["units"]}
                    publish_json(receipt_path, record)
                    del values
                    print(f"Saved {path.name}: {record['sha256']}", flush=True)
                records.append(record)
    return records


def weighted_difference(left, right, lat):
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    weights = np.cos(np.deg2rad(lat)); weights /= weights.mean()
    return {"cosine_latitude_rmse": float(np.sqrt(np.mean(difference**2 * weights[:,None]))),
            "max_abs_difference": float(np.max(np.abs(difference))),
            "bitwise_equal": bool(np.array_equal(left, right))}


def prepare(campaign, source_records):
    ncar = campaign / "data/ncar"
    truth_out = campaign / "data/common_truth"
    truth_out.mkdir(parents=True, exist_ok=True)
    with np.load(ncar / "coordinates.npz") as coords:
        lat, lon, levels = coords["latitude"], coords["longitude"], coords["level"]
    original = CommonTruth(OLD)
    print("Reading midnight common-grid control",flush=True)
    midnight, midnight_hash = original.at(utc(0))
    a,b,li,oi,ul,uo = _operators(tuple(lat), tuple(lon), tuple(midnight.lat.values),tuple(midnight.lon.values))

    def remap(values):
        values = np.asarray(values, dtype=np.float64)[np.ix_(li,oi)]
        return b.dot(a.dot(values).T).T[np.ix_(ul,uo)]

    fields = {}
    native_comparisons = {}
    print("Opening independently staged native 1-degree truth for controls",flush=True)
    native_store = xr.open_zarr(NATIVE, consolidated=False, chunks=None)
    for hour in (0,6,12,18):
        print(f"Validating native fields and remapping {utc(hour)}",flush=True)
        fields[hour] = {}
        native_comparisons[utc(hour)] = {}
        native_time = native_store.sel(time=np.datetime64(utc(hour)[:-1]))
        assert np.array_equal(np.sort(native_time.lat.values), np.sort(lat[::4]))
        assert np.array_equal(native_time.lon.values, lon[::4])
        latitude_indices = [int(np.flatnonzero(lat == x)[0]) for x in native_time.lat.values]
        longitude_indices = [int(np.flatnonzero(lon == x)[0]) for x in native_time.lon.values]
        level_indices = [int(np.flatnonzero(levels == x)[0]) for x in native_time.level.values]
        for variable in SOURCES:
            path = ncar / f"{filename(hour)}_{variable}.npy"
            receipt = json.loads(Path(str(path) + ".json").read_text())
            assert sha256(path) == receipt["sha256"]
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            assert array_sha(values) == receipt["array_sha256"]
            old_values = native_time[variable].values
            new_values = values[np.ix_(level_indices,latitude_indices,longitude_indices)]
            native_comparisons[utc(hour)][variable] = {
                str(int(level)): weighted_difference(new_values[i],old_values[i],native_time.lat.values)
                for i,level in enumerate(native_time.level.values)}
            for key in REPLACED:
                spec = DIAGNOSTICS[key]
                if spec.variable == variable:
                    plane = values[int(np.flatnonzero(levels == spec.level)[0])]
                    fields[hour][key] = remap(plane)
            del values
    native_store.close()
    control = {key: weighted_difference(fields[0][key],midnight[key].values,midnight.lat.values)
               for key in REPLACED}
    for key, value in control.items():
        if value["cosine_latitude_rmse"] > CONTROL_RMSE_LIMIT[key]:
            raise ValueError(f"Cross-source midnight precision control failed: {key}: {value}")
    print("Midnight controls: " + json.dumps(control),flush=True)
    same_time_control = {}
    for hour in (6,12,18):
        same_time_control[utc(hour)] = {}
        for variable,level,limit in (("geopotential","50",1.0),("temperature","850",.01),
                                     ("specific_humidity","500",1e-6)):
            value = native_comparisons[utc(hour)][variable][level]
            if value["cosine_latitude_rmse"] > limit:
                raise ValueError(f"Unaffected same-time native control failed: {hour}/{variable}/{level}: {value}")
            same_time_control[utc(hour)][f"{variable}_{level}"] = {**value,"rmse_limit":limit}
    configuration = {"schema_version":1, "original_truth_root":str(OLD), "source":"NSF NCAR ERA5 ds633.0",
                     "replaced_diagnostics":REPLACED, "replaced_valid_times":[utc(h) for h in (6,12,18)],
                     "retained_diagnostics":[k for k in DIAGNOSTICS if k not in REPLACED],
                     "remapping":"common_grid_metrics._operators first_order_conservative_spherical_cell_overlap",
                     "remapping_source_sha256":sha256(ROOT / "scripts/analyze_models/common_grid_metrics.py"),
                     "recovery_script_sha256":sha256(Path(__file__)),
                     "native_source_file_sha256":{r["path"]:r["sha256"] for r in source_records},
                     "source_metadata_file_sha256":{str(p):sha256(p) for p in sorted(ncar.glob("*.source.json"))},
                     "coordinate_file_sha256":sha256(ncar / "coordinates.npz")}
    configuration_hash = hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    outputs = []
    for hour in (6,12,18):
        old, old_hash = original.at(utc(hour))
        result = old.copy(deep=True)
        for key in REPLACED:
            result[key].data = fields[hour][key]
            result[key].attrs.update({"recovery_source":"NSF NCAR ERA5 ds633.0", "recovery_valid_time":utc(hour)})
        result.attrs.update({"valid_time":utc(hour), "truth_recovery":"Replace five stale ARCO diagnostic planes using NCAR ERA5",
                             "original_truth_sha256":old_hash, "recovery_configuration_sha256":configuration_hash})
        path = truth_out / (filename(hour) + ".nc")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite published truth: {path}")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
        # Do not inherit backend storage encodings from the original NetCDF.
        for name in result.variables:
            result[name].encoding = {}
        encoding = {k:{"dtype":"float64", "zlib":True, "complevel":1, "shuffle":True} for k in DIAGNOSTICS}
        result.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        os.replace(temporary,path)
        receipt = {"schema_version":1, "kind":"common_truth", "path":str(path.resolve()),
                   "timestamp":utc(hour), "size_bytes":path.stat().st_size, "sha256":sha256(path),
                   "configuration_sha256":configuration_hash,
                   "created_at":datetime.now(timezone.utc).isoformat(),
                   "source_uri":"https://nsf-ncar-era5.s3.amazonaws.com/e5.oper.an.pl/202311/",
                   "source_structure_sha256":hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),
                   "original_truth_path":str(OLD / path.name), "original_truth_sha256":old_hash,
                   "replaced_diagnostics":REPLACED}
        publish_json(Path(str(path)+".READY.json"),receipt)
        checked, checked_hash = CommonTruth(truth_out).at(utc(hour))
        assert checked_hash == receipt["sha256"] and set(checked.data_vars) == set(DIAGNOSTICS)
        for coordinate in ("lat","lon"):
            assert np.array_equal(checked[coordinate].values,old[coordinate].values)
        retained = {}
        replaced = {}
        for key in DIAGNOSTICS:
            assert checked[key].shape == (121,240) and np.isfinite(checked[key].values).all()
            if key in REPLACED:
                assert np.array_equal(checked[key].values, fields[hour][key])
                assert not np.array_equal(checked[key].values, old[key].values)
                replaced[key] = {"array_sha256":array_sha(checked[key].values),
                                 "original_array_sha256":array_sha(old[key].values),
                                 "change":weighted_difference(checked[key].values,old[key].values,old.lat.values)}
            else:
                assert np.array_equal(checked[key].values,old[key].values)
                assert checked[key].values.dtype == old[key].values.dtype
                retained[key] = {"byte_identical":True,"array_sha256":array_sha(checked[key].values)}
        # Prove the old source is still the exact artifact verified before repair.
        assert verify_ready(OLD / path.name,utc(hour),"common_truth")["sha256"] == old_hash
        outputs.append({**receipt,"retained_fields":retained,"replaced_fields":replaced,
                        "ready_receipt_sha256":sha256(Path(str(path)+".READY.json"))})
    report = {"status":"complete", "configuration":configuration,"configuration_sha256":configuration_hash,
              "corrected_common_truth_root":str(truth_out),
              "repaired_valid_times":[utc(hour) for hour in (6,12,18)],
              "output_truth_files":[{"valid_time":r["timestamp"],"path":r["path"],"sha256":r["sha256"],
                                     "original_sha256":r["original_truth_sha256"],
                                     "replaced_diagnostics":r["replaced_diagnostics"]} for r in outputs],
              "source_records":source_records,"midnight_control_truth_sha256":midnight_hash,
              "control_rmse_limits":CONTROL_RMSE_LIMIT,"midnight_control_comparisons":control,
              "same_time_unaffected_native_controls":same_time_control,
              "native_1degree_comparisons":native_comparisons,"outputs":outputs,
              "scope":"Only the three common-grid truth files are replaced. Native all-level downloads are audit evidence; native dataset and all original campaign files are unchanged.",
              "source_public_documentation":["https://registry.opendata.aws/nsf-ncar-era5/",
                                              "https://gdex.ucar.edu/datasets/d633000/"]}
    publish_json(campaign / "recovery_report.json", report)
    print(json.dumps({"status":report["status"],"report":str(campaign / "recovery_report.json"),
                      "midnight_control_comparisons":control,"corrected_truth_files":[r["path"] for r in outputs]},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,default=CAMPAIGN)
    parser.add_argument("--phase",choices=("all","download","prepare"),default="all")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True,exist_ok=True)
    if args.phase in ("all","download"):
        records = download(campaign / "data/ncar")
    else:
        records = [json.loads(path.read_text()) for path in sorted((campaign / "data/ncar").glob("*.npy.json"))]
        assert len(records) == 12
    if args.phase in ("all","prepare"):
        prepare(campaign,records)


if __name__ == "__main__":
    main()
