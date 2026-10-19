#!/usr/bin/env python3
"""Score the WeatherBench2 IFS ENS mean on the 2023 common-grid campaign.

WeatherBench2 publishes no 2023 ENS scores. Its public store
`ifs_ens/2016-2024-240x121_equiangular_with_poles_conservative_mean.zarr`
holds 1.5° ENS-mean forecasts but lacks all zarr metadata and coordinate
arrays, and roughly a fifth of its chunks are absent. The chunk layout is
borrowed from the complete `2018-2022` sibling store and verified first by
decoding overlapping 2018-2022 chunks from both stores.

Scoring reuses the campaign's frozen `common_grid_metrics.score_step` and the
same staged 1.5° truth files (hashes checked against the M1 shards). The ENS
grid equals the truth grid, so the conservative remap is an identity. Missing
chunks are recorded as null MSEs; nothing is interpolated or substituted.

    python scripts/analyze_models/score_ens_common_grid_2023.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

import numcodecs
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "artifacts/evaluations/graphcast_common_grid_2023_20261015"
OUT = CAMPAIGN / "ens/ens_mean_mse.json"
BUCKET = "https://storage.googleapis.com/weatherbench2/"
STORE = "datasets/ifs_ens/2016-2024-240x121_equiangular_with_poles_conservative_mean.zarr/"
REFERENCE = "datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr/"
STORE_EPOCH = dt.datetime(2016, 1, 1)
REFERENCE_EPOCH = dt.datetime(2018, 1, 1)
STEP_HOURS = 12
LEAD_CHUNK = 8
LEVELS = (500, 700, 850)
# Diagnostic -> (store variable, level or None).
SOURCES = {
    "Z500": ("geopotential", 500), "Z850": ("geopotential", 850),
    "T500": ("temperature", 500), "T850": ("temperature", 850),
    "T2m": ("2m_temperature", None), "MSL": ("mean_sea_level_pressure", None),
    "Q700": ("specific_humidity", 700), "Q850": ("specific_humidity", 850),
    "U850": ("u_component_of_wind", 850), "V850": ("v_component_of_wind", 850),
    "U10": ("10m_u_component_of_wind", None), "V10": ("10m_v_component_of_wind", None),
}
BLOSC = numcodecs.Blosc()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(url, attempts=4):
    """Return object bytes, or None when the object does not exist."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def list_objects(prefix):
    names, token = set(), None
    while True:
        query = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        url = ("https://storage.googleapis.com/storage/v1/b/weatherbench2/o?"
               + urllib.parse.urlencode(query))
        page = json.loads(fetch(url))
        names.update(item["name"][len(prefix):] for item in page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return names


def time_index(epoch, when):
    hours = (when - epoch).total_seconds() / 3600
    assert hours % STEP_HOURS == 0
    return int(hours // STEP_HOURS)


def reference_layout():
    """Borrow array metadata and coordinates from the complete sibling store."""
    meta = json.loads(fetch(BUCKET + REFERENCE + ".zmetadata"))["metadata"]
    coords = {}
    for name in ("prediction_timedelta", "level", "latitude", "longitude"):
        spec = meta[name + "/.zarray"]
        raw = BLOSC.decode(fetch(BUCKET + REFERENCE + name + "/0"))
        coords[name] = np.frombuffer(raw, dtype=spec["dtype"]).copy()
    arrays = {var: meta[var + "/.zarray"] for var in {v for v, _ in SOURCES.values()}}
    attrs = {var: meta[var + "/.zattrs"] for var in arrays}
    assert coords["prediction_timedelta"].tolist() == list(range(0, 361, 6))
    assert tuple(coords["level"].tolist()) == LEVELS
    for var, spec in arrays.items():
        dims = attrs[var]["_ARRAY_DIMENSIONS"]
        assert dims[-2:] == ["longitude", "latitude"], (var, dims)
        assert spec["chunks"][:2] == [1, LEAD_CHUNK] and spec["dtype"] == "<f4", var
        assert spec["compressor"]["id"] == "blosc", var
    return arrays, attrs, coords, meta


def chunk_key(var, spec, t, c):
    return f"{var}/{t}.{c}" + ".0" * (len(spec["shape"]) - 2)


def decode(raw, spec):
    shape = list(spec["chunks"])
    return np.frombuffer(BLOSC.decode(raw), dtype=spec["dtype"]).reshape(shape)


def verify_layout(arrays):
    """Decode overlapping 2018-2022 chunks from both stores and compare values."""
    checks = []
    dates = [dt.datetime(2018, 1, 1), dt.datetime(2019, 6, 15, 12),
             dt.datetime(2020, 2, 29), dt.datetime(2021, 9, 1), dt.datetime(2022, 12, 31, 12)]
    for when in dates:
        for var in sorted(arrays):
            spec = arrays[var]
            for c in (0, 4):
                new = fetch(BUCKET + STORE + chunk_key(var, spec, time_index(STORE_EPOCH, when), c))
                old = fetch(BUCKET + REFERENCE + chunk_key(var, spec, time_index(REFERENCE_EPOCH, when), c))
                if new is None or old is None:
                    continue
                a, b = decode(new, spec), decode(old, spec)
                scale = float(np.nanmax(np.abs(b)))
                diff = float(np.nanmax(np.abs(a - b)))
                checks.append({"time": when.isoformat(), "variable": var, "lead_chunk": c,
                               "max_abs_diff": diff, "max_rel_diff": diff / scale})
    worst = max(check["max_rel_diff"] for check in checks)
    variables = {check["variable"] for check in checks}
    # Float32 recomputation noise only; a transposed or shifted layout would
    # produce order-one relative differences.
    if worst > 1e-5 or variables != set(arrays):
        raise ValueError(f"ENS store layout check failed: worst={worst}, variables={variables}")
    return {"compared_chunks": len(checks), "max_rel_diff": worst, "checks": checks}


def load_scorer():
    path = CAMPAIGN / "source/scripts/analyze_models/common_grid_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_common_grid_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(path)


def truth_hashes():
    hashes = {}
    for shard in sorted((CAMPAIGN / "shards").glob("m1_*.json")):
        for record in json.loads(shard.read_text())["per_initialization"]:
            hashes[record["initialization_time"]] = dict(zip(record["valid_times"],
                                                             record["truth_sha256"]))
    return hashes


def fetch_start(start, *, arrays, available, leads):
    """Download one start's raw chunks; thread-safe (no decoding or HDF5)."""
    t = time_index(STORE_EPOCH, dt.datetime.fromisoformat(start.replace("Z", "")))
    raw = {}
    for var, spec in arrays.items():
        for c in sorted({(h // 6) // LEAD_CHUNK for h in leads}):
            key = chunk_key(var, spec, t, c)
            raw[var, c] = fetch(BUCKET + STORE + key) if key in available else None
    return raw


def score_start(start, raw, *, arrays, coords, scorer, protocol, hashes):
    when = dt.datetime.fromisoformat(start.replace("Z", ""))
    t = time_index(STORE_EPOCH, when)
    leads = protocol["lead_hours"]
    chunks = {key: None if data is None else decode(data, arrays[key[0]])
              for key, data in raw.items()}
    level_index = {level: i for i, level in enumerate(coords["level"].tolist())}
    truth_root = Path(protocol["common_truth"])
    mse = {key: [] for key in SOURCES}
    for hours in leads:
        valid = when + dt.timedelta(hours=hours)
        valid_id = valid.strftime("%Y%m%dT%H%M%SZ")
        valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
        truth_path = truth_root / f"{valid_id}.nc"
        if sha256(truth_path) != hashes[start][valid_iso]:
            raise ValueError(f"Truth hash differs from the M1 shard: {truth_path}")
        truth = xr.load_dataset(truth_path)
        k = hours // 6
        fields, present = {}, {}
        for key, (var, level) in SOURCES.items():
            block = chunks[var, k // LEAD_CHUNK]
            values = None
            if block is not None:
                values = block[0, k % LEAD_CHUNK]
                if level is not None:
                    values = values[level_index[level]]
                values = np.asarray(values, dtype=np.float64).T  # (lon, lat) -> (lat, lon)
                if not np.isfinite(values).all():
                    values = None
            present[key] = values is not None
            # score_step needs all twelve fields; an absent field is filled
            # with truth only to satisfy the interface and is reported as null.
            fields[key] = (("lat", "lon"), truth[key].values if values is None else values,
                           {"units": truth[key].attrs.get("units", "")})
        prediction = xr.Dataset(fields, coords={"lat": coords["latitude"],
                                                "lon": coords["longitude"]})
        scores = scorer.score_step(prediction, truth)
        for key in SOURCES:
            mse[key].append(scores[key] if present[key] else None)
    return {"initialization_time": start, "store_time_index": t, "lead_hours": leads, "mse": mse}


def main():
    protocol = json.loads((CAMPAIGN / "protocol.json").read_text())
    scorer, scorer_sha = load_scorer()
    arrays, attrs, coords, _ = reference_layout()
    # Same grid as the staged truth, so score_step's conservative remap is an
    # identity; tolerance covers only float formatting of the 1.5° spacing.
    sample = xr.open_dataset(next(Path(protocol["common_truth"]).glob("*.nc")))
    assert np.allclose(coords["latitude"], sample.lat.values, rtol=0, atol=1e-9)
    assert np.allclose(coords["longitude"], sample.lon.values, rtol=0, atol=1e-9)
    coords["latitude"], coords["longitude"] = sample.lat.values, sample.lon.values
    sample.close()
    layout = verify_layout(arrays)
    print(f"layout verified on {layout['compared_chunks']} chunks, "
          f"max rel diff {layout['max_rel_diff']:.2e}", flush=True)
    available = set()
    for var in arrays:
        available.update(f"{var}/{name}" for name in list_objects(STORE + var + "/"))
    hashes = truth_hashes()
    starts = protocol["initialization_times"]
    results = []
    # Threads only download; Blosc decoding and NetCDF/HDF5 reads are not
    # thread-safe here (the threaded version segfaulted), so scoring is serial.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_start, s, arrays=arrays, available=available,
                               leads=protocol["lead_hours"]) for s in starts]
        for i, future in enumerate(futures):
            results.append(score_start(starts[i], future.result(), arrays=arrays,
                                       coords=coords, scorer=scorer, protocol=protocol,
                                       hashes=hashes))
            futures[i] = None  # release raw bytes
            print(f"{i + 1}/{len(starts)} {starts[i]}", flush=True)
    present = sum(v is not None for r in results for series in r["mse"].values() for v in series)
    total = len(results) * len(SOURCES) * len(protocol["lead_hours"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "schema_version": 1,
        "model": "ifs_ens_mean",
        "protocol_hash": json.loads((CAMPAIGN / "shards/m1_00.json").read_text())["protocol_hash"],
        "campaign": str(CAMPAIGN.relative_to(ROOT)),
        "store": BUCKET + STORE,
        "store_note": ("Public store without zarr metadata or coordinates; layout borrowed "
                       "from the complete 2018-2022 sibling and verified on overlapping chunks. "
                       "Absent chunks yield null MSE."),
        "layout_reference": BUCKET + REFERENCE,
        "layout_verification": layout,
        "scorer": str((CAMPAIGN / "source/scripts/analyze_models/common_grid_metrics.py").relative_to(ROOT)),
        "scorer_sha256": scorer_sha,
        "script_sha256": sha256(__file__),
        "truth": "campaign common_truth; per-file SHA256 matched to M1 shards",
        "available_entries": present,
        "total_entries": total,
        "per_initialization": results,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {present}/{total} MSE entries present")


if __name__ == "__main__":
    main()
