#!/usr/bin/env python3
"""Score DeepMind's archived 2022 GraphCast (37 levels) on our 2022 starts.

Apples-to-apples check for the 2022 common-grid campaign: the public
WeatherBench2 store `graphcast_v2/2022-240x121_equiangular_with_poles_conservative.zarr`
holds DeepMind's own 37-level, ERA5-initialized GraphCast forecasts on the
1.5° grid. For each of our 122 starts this scores those forecasts with the
campaign's frozen `common_grid_metrics.score_step` against the same staged
truth (hashes checked against our GC025 shards). Only the model run differs.

The store's metadata is complete; chunks are decoded serially because the
threaded HDF5/Blosc path segfaulted in score_ens_common_grid_2023.py.

    python scripts/analyze_models/score_wb2_graphcast_archive_2022.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analyze_models"))
from score_ens_common_grid_2023 import BLOSC, BUCKET, fetch, sha256  # noqa: E402

CAMPAIGN = ROOT / "artifacts/evaluations/graphcast_common_grid_2022_20261016"
OUT = CAMPAIGN / "wb2_archive/graphcast_v2_2022_mse.json"
STORE = "datasets/graphcast_v2/2022-240x121_equiangular_with_poles_conservative.zarr/"
LEAD_CHUNK = 8
SOURCES = {
    "Z500": ("geopotential", 500), "Z850": ("geopotential", 850),
    "T500": ("temperature", 500), "T850": ("temperature", 850),
    "T2m": ("2m_temperature", None), "MSL": ("mean_sea_level_pressure", None),
    "Q700": ("specific_humidity", 700), "Q850": ("specific_humidity", 850),
    "U850": ("u_component_of_wind", 850), "V850": ("v_component_of_wind", 850),
    "U10": ("10m_u_component_of_wind", None), "V10": ("10m_v_component_of_wind", None),
}


def coordinate(meta, name):
    spec = meta[name + "/.zarray"]
    raw = fetch(BUCKET + STORE + name + "/0")
    return np.frombuffer(BLOSC.decode(raw) if spec["compressor"] else raw,
                         dtype=spec["dtype"]).copy()


def layout():
    meta = json.loads(fetch(BUCKET + STORE + ".zmetadata"))["metadata"]
    coords = {name: coordinate(meta, name)
              for name in ("time", "prediction_timedelta", "level", "latitude", "longitude")}
    assert meta["time/.zattrs"]["units"] == "hours since 2021-12-01"
    assert meta["prediction_timedelta/.zattrs"]["units"] == "hours"
    assert coords["prediction_timedelta"].tolist() == list(range(6, 241, 6))
    arrays = {}
    for var in {v for v, _ in SOURCES.values()}:
        spec, dims = meta[var + "/.zarray"], meta[var + "/.zattrs"]["_ARRAY_DIMENSIONS"]
        assert dims[:2] == ["time", "prediction_timedelta"] and dims[-2:] == ["longitude", "latitude"]
        assert spec["chunks"][:2] == [1, LEAD_CHUNK] and spec["dtype"] == "<f4"
        arrays[var] = spec
    return meta, coords, arrays


def time_index(coords, when):
    hours = int((when - dt.datetime(2021, 12, 1)).total_seconds() // 3600)
    matches = np.flatnonzero(coords["time"] == hours)
    if matches.size != 1:
        raise ValueError(f"Archive has no unique forecast for {when}")
    return int(matches[0])


def fetch_start(t, arrays, leads):
    raw = {}
    for var, spec in arrays.items():
        for c in sorted({(h // 6 - 1) // LEAD_CHUNK for h in leads}):
            key = f"{var}/{t}.{c}" + ".0" * (len(spec["shape"]) - 2)
            data = fetch(BUCKET + STORE + key)
            if data is None:
                raise ValueError(f"Missing archive chunk {key}")
            raw[var, c] = data
    return raw


def score_start(start, raw, *, arrays, coords, scorer, protocol, hashes):
    when = dt.datetime.fromisoformat(start.replace("Z", ""))
    chunks = {key: np.frombuffer(BLOSC.decode(data), dtype="<f4").reshape(arrays[key[0]]["chunks"])
              for key, data in raw.items()}
    level_index = {int(level): i for i, level in enumerate(coords["level"])}
    mse = {key: [] for key in SOURCES}
    for hours in protocol["lead_hours"]:
        valid = when + dt.timedelta(hours=hours)
        truth_path = Path(protocol["common_truth"]) / valid.strftime("%Y%m%dT%H%M%SZ.nc")
        if sha256(truth_path) != hashes[start][valid.strftime("%Y-%m-%dT%H:%M:%SZ")]:
            raise ValueError(f"Truth hash differs from the GC025 shard: {truth_path}")
        truth = xr.load_dataset(truth_path)
        k = hours // 6 - 1
        fields = {}
        for key, (var, level) in SOURCES.items():
            values = chunks[var, k // LEAD_CHUNK][0, k % LEAD_CHUNK]
            if level is not None:
                values = values[level_index[level]]
            fields[key] = (("lon", "lat"), np.asarray(values, dtype=np.float64))
        prediction = xr.Dataset(fields, coords={"lat": coords["latitude"], "lon": coords["longitude"]})
        scores = scorer.score_step(prediction, truth)
        for key in SOURCES:
            mse[key].append(scores[key])
    return {"initialization_time": start, "lead_hours": protocol["lead_hours"], "mse": mse}


def main():
    import importlib.util
    protocol = json.loads((CAMPAIGN / "protocol.json").read_text())
    path = CAMPAIGN / "source/scripts/analyze_models/common_grid_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_common_grid_metrics", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    meta, coords, arrays = layout()
    sample = xr.open_dataset(next(Path(protocol["common_truth"]).glob("*.nc")))
    assert np.allclose(np.sort(coords["latitude"]), sample.lat.values, rtol=0, atol=1e-9)
    assert np.allclose(np.sort(coords["longitude"]), sample.lon.values, rtol=0, atol=1e-9)
    sample.close()
    hashes = {}
    for shard in sorted((CAMPAIGN / "shards").glob("gc025_*.json")):
        for record in json.loads(shard.read_text())["per_initialization"]:
            hashes[record["initialization_time"]] = dict(zip(record["valid_times"],
                                                             record["truth_sha256"]))
    starts = protocol["initialization_times"]
    assert set(starts) == set(hashes), "GC025 shards must cover the full manifest"
    indices = [time_index(coords, dt.datetime.fromisoformat(s.replace("Z", ""))) for s in starts]
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch_start, t, arrays, protocol["lead_hours"]) for t in indices]
        for i, future in enumerate(futures):
            results.append(score_start(starts[i], future.result(), arrays=arrays, coords=coords,
                                       scorer=scorer, protocol=protocol, hashes=hashes))
            futures[i] = None
            print(f"{i + 1}/{len(starts)} {starts[i]}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "schema_version": 1,
        "model": "wb2_graphcast_v2_2022",
        "store": BUCKET + STORE,
        "store_attrs": meta[".zattrs"],
        "protocol_hash": protocol["protocol_hash"],
        "scorer_sha256": sha256(path),
        "script_sha256": sha256(__file__),
        "truth": "campaign common_truth; per-file SHA256 matched to GC025 shards",
        "per_initialization": results,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
