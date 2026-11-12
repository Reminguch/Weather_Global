"""Build verified native-grid 1991–2020 extreme thresholds from staged WB2 surfaces."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from stage_extremes_surfaces import CAMPAIGN, HISTORY, ROOT, START, STOP, LocalV2, json_atomic, require, sha256


FIRST_YEAR = 1991
YEARS = 30
DAYS = 365
HOURS = (0, 6, 12, 18)
HALF_WINDOW = 30
QUANTILES = {"t2m": (2, 5, 95, 98), "ws10": (95, 98), "tp24": (95, 98)}
UNITS = {"t2m": "K", "ws10": "m/s", "tp24": "mm"}
EVAL = ROOT / "data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval_ncar_qc.zarr"


def precipitation_24h(previous, current):
    """Preserve real leap dates and month boundaries during 24-hour accumulation."""
    require(previous.shape[0] == 3 and previous.shape[1:] == current.shape[1:], "Need three preceding TP6 fields")
    combined = np.concatenate((previous, current), axis=0)
    count = len(current)
    total = combined[:count].astype(np.float64)
    for offset in (1, 2, 3):
        total += combined[offset:offset + count]
    return total * 1000.0, combined[-3:].copy()


def load_calendar(stage, config, lat, lon):
    """Place samples on a 365-day calendar after forming the physical variables."""
    spatial = (len(lat), len(lon))
    cube = np.empty((len(QUANTILES), len(HOURS), YEARS, DAYS, *spatial), dtype=np.float32)
    received = np.zeros((YEARS, DAYS), dtype=bool)
    receipts = {}
    expected_months = [str(p) for p in pd.period_range(HISTORY, STOP, freq="M")]
    require(config["months"] == expected_months and config["reference_start"] == START and config["reference_end"] == STOP,
            "Staged surface coverage does not match the fixed climate reference")
    previous = None
    for month in expected_months:
        path = stage / f"{month}.npy"
        receipt = json.loads((stage / f"{month}.json").read_text())
        receipts[month] = receipt
        require(receipt["month"] == month and sha256(path) == receipt["sha256"], f"Invalid staged source receipt: {month}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(list(array.shape) == receipt["shape"] and array.dtype == np.float32, f"Invalid surface array: {month}")
        if month == "1990-12":
            require(array.shape == (1, 3, *spatial) and receipt["first_time"] == "1990-12-31 06:00:00"
                    and receipt["last_time"] == "1990-12-31 18:00:00", "Missing prior-day TP6 history")
            previous = array[0].copy()
            continue
        period = pd.Period(month, freq="M")
        dates = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
        require(array.shape == (3, len(dates) * 4, *spatial)
                and receipt["first_time"] == str(dates[0])
                and receipt["last_time"] == str(dates[-1] + pd.Timedelta(hours=18)),
                f"Incomplete six-hour coverage in {month}")
        tp24, previous = precipitation_24h(previous, array[2])
        years = period.year - FIRST_YEAR
        dates_keep = ~((dates.month == 2) & (dates.day == 29))
        calendar_days = (dates.dayofyear - 1 - (dates.is_leap_year & (dates.month > 2))).to_numpy()[dates_keep]
        require(not received[years, calendar_days].any(), f"Duplicate climate dates in {month}")
        for index, values in enumerate((array[0], array[1], tp24)):
            ordered = values.reshape(len(dates), len(HOURS), *spatial)
            cube[index, :, years][:, calendar_days] = ordered[dates_keep].transpose(1, 0, 2, 3)
        received[years, calendar_days] = True
        if month.endswith("-12"):
            print(f"Loaded climate reference year {period.year}", flush=True)
    require(received.all(), "Some reference dates are missing")
    return cube.reshape(*cube.shape[:-2], -1), receipts


def linear_window_quantiles(series, windows, quantiles):
    """Exact empirical quantiles from order statistics, in spatial tiles."""
    # series layout is [spatial point, year, calendar day]. The indexed samples
    # become the contiguous last axis so partition never sorts the full window.
    count = series.shape[1] * windows.shape[1]
    ranks = (count - 1) * np.asarray(quantiles, dtype=np.float64) / 100.0
    lo = np.floor(ranks).astype(np.intp)
    hi = np.ceil(ranks).astype(np.intp)
    fraction = ranks - lo
    kth = np.unique(np.concatenate((lo, hi)))
    for days in windows:
        samples = np.take(series, days, axis=2).reshape(series.shape[0], count)
        samples.partition(kth, axis=1)
        lower = samples[:, lo].astype(np.float64)
        upper = samples[:, hi].astype(np.float64)
        delta = upper - lower
        yield np.where(fraction < 0.5, lower + fraction * delta, upper - (1.0 - fraction) * delta)


def compute(cube, writers, workers, tile_points, status, status_path):
    windows = (np.arange(DAYS)[:, None] + np.arange(-HALF_WINDOW, HALF_WINDOW + 1)[None, :]) % DAYS
    tasks = [(i, variable, hour, start, min(start + tile_points, cube.shape[-1]))
             for i, variable in enumerate(QUANTILES) for hour in range(len(HOURS))
             for start in range(0, cube.shape[-1], tile_points)]

    def compute_tile(index, variable, hour, start, stop):
        series = np.ascontiguousarray(cube[index, hour, :, :, start:stop].transpose(2, 0, 1))
        quantiles = QUANTILES[variable]
        outputs = [writers[(variable, q)] for q in quantiles]
        for day, values in enumerate(linear_window_quantiles(series, windows, quantiles)):
            for i, output in enumerate(outputs):
                output[day, hour, start:stop] = values[:, i]

    started = time.monotonic()
    status.update(status="computing", completed_tiles=0, total_tiles=len(tasks))
    json_atomic(status_path, status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compute_tile, *task) for task in tasks]
        try:
            for finished, future in enumerate(as_completed(futures), start=1):
                future.result()
                if finished == 1 or finished % 25 == 0 or finished == len(tasks):
                    status.update(completed_tiles=finished, compute_seconds=time.monotonic() - started)
                    json_atomic(status_path, status)
                    print(f"Computed {finished}/{len(tasks)} threshold tiles", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise


def publish(args, status, status_path):
    stage = args.campaign / "surface_stage"
    ready_stage = json.loads((args.campaign / "climatology_stage_status.json").read_text())
    require(ready_stage["status"] == "staged" and ready_stage["completed_months"] == ready_stage["total_months"],
            "Surface acquisition is incomplete")
    output = args.campaign / "climatology"
    if output.exists():
        raise FileExistsError(output)
    partial = Path(tempfile.mkdtemp(prefix=".climatology-", dir=args.campaign))
    status.update(status="loading", partial_output=str(partial))
    json_atomic(status_path, status)
    reference = LocalV2(args.evaluation)
    lat, lon, land = (np.load(stage / f"{name}.npy", allow_pickle=False) for name in ("lat", "lon", "land_mask"))
    require(np.array_equal(lat, reference.vector("lat")) and np.array_equal(lon, reference.vector("lon"))
            and np.array_equal(land, reference.field("land_sea_mask")), "Climatology and evaluation grid or land mask differ")
    for name in ("lat", "lon", "land_mask"):
        shutil.copy2(stage / f"{name}.npy", partial / f"{name}.npy")
    config_path = stage / "config.json"
    config = json.loads(config_path.read_text())
    cube, receipts = load_calendar(stage, config, lat, lon)
    shape = (DAYS, len(HOURS), len(lat), len(lon))
    writers = {}
    arrays = {}
    for variable, qs in QUANTILES.items():
        for q in qs:
            name = f"{variable}_q{q:02d}"
            path = partial / f"{name}.npy"
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            arrays[name] = array
            writers[(variable, q)] = array.reshape(DAYS, len(HOURS), -1)
    compute(cube, writers, args.workers, args.tile_points, status, status_path)
    del cube, writers
    status.update(status="validating")
    json_atomic(status_path, status)
    for variable, qs in QUANTILES.items():
        previous = None
        for q in qs:
            name = f"{variable}_q{q:02d}"
            current = arrays[name]
            current.flush()
            require(np.isfinite(current).all() and (previous is None or np.all(previous <= current)),
                    f"Nonfinite or unordered quantiles in {name}")
            previous = current
    variables = {name: {"file": f"{name}.npy", "sha256": sha256(partial / f"{name}.npy"),
                        "shape": list(shape), "dtype": "float32", "units": UNITS[name.split("_")[0]]}
                 for name in arrays}
    protocol = {
        "schema_version": 1, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": config, "source_parity": json.loads((stage / "source_parity.json").read_text()),
        "surface_stage_config_sha256": sha256(config_path), "surface_stage_summary": ready_stage,
        "month_receipts_sha256": hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest(),
        "evaluation_zmetadata_sha256": sha256(args.evaluation / ".zmetadata"), "builder_sha256": sha256(Path(__file__)),
        "reference_years": [1991, 2020], "calendar_days": 365, "utc_hours": list(HOURS),
        "window": "circular ±30 calendar days inclusive within a 365-day calendar; February 29 excluded from reference samples",
        "day_index": "January 1 is index 0; array order [calendar_day, utc_hour, latitude, longitude]",
        "quantile_method": "linear empirical quantile interpolation over order statistics; float64 interpolation, float32 storage",
        "ws10": "hypot of 10m zonal/meridional wind; m/s",
        "tp24": "four consecutive WB2 TP6 fields ending at valid UTC, summed in float64 and converted from metres to mm",
        "rain_eligibility": "tp24_q95 > 1 mm at each calendar day, UTC hour and grid point",
        "land_mask_rule": "land_mask >= 0.5 on the native evaluation grid",
        "sample_count_by_day_utc": [[YEARS * (2 * HALF_WINDOW + 1)] * len(HOURS)] * DAYS,
        "variables": variables,
        "coordinates": {name: {"file": f"{name}.npy", "sha256": sha256(partial / f"{name}.npy")}
                        for name in ("lat", "lon", "land_mask")},
    }
    json_atomic(partial / "metadata.json", protocol)
    partial.rename(output)
    status.update(status="complete", output=str(output), metadata_sha256=sha256(output / "metadata.json"))
    json_atomic(status_path, status)
    print(f"Ready: {output / 'metadata.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--evaluation", type=Path, default=EVAL)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--tile-points", type=int, default=256)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve(strict=True)
    args.evaluation = args.evaluation.resolve(strict=True)
    require(args.workers > 0 and args.tile_points > 0, "Expected positive workers and tile size")
    status_path = args.campaign / "climatology_compute_status.json"
    status = {"pid": os.getpid()}
    with (args.campaign / ".climatology.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            publish(args, status, status_path)
        except Exception as error:
            status.update(status="failed", error=str(error))
            json_atomic(status_path, status)
            raise


if __name__ == "__main__":
    main()
