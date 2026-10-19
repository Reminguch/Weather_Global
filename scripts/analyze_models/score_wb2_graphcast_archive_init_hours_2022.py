#!/usr/bin/env python3
"""Init-hour sensitivity of DeepMind's archived 2022 GraphCast forecasts.

Scores the WB2 `graphcast_v2/2022` archive from starts at 06, 12 and 18 UTC
on every fourth date of the 2022 campaign manifest, reusing the 00 UTC scores
from score_wb2_graphcast_archive_2022.py for the same dates. Same frozen
scorer and staged truth; off-manifest truth files are hash-checked against
their READY receipts instead of the GC025 shards.

    python scripts/analyze_models/score_wb2_graphcast_archive_init_hours_2022.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analyze_models"))
import score_wb2_graphcast_archive_2022 as archive  # noqa: E402
from score_ens_common_grid_2023 import sha256  # noqa: E402

OUT = archive.CAMPAIGN / "wb2_archive/graphcast_v2_2022_init_hours_mse.json"
OFFSETS = (6, 12, 18)
DATE_STRIDE = 4


def main():
    protocol = json.loads((archive.CAMPAIGN / "protocol.json").read_text())
    path = archive.CAMPAIGN / "source/scripts/analyze_models/common_grid_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_common_grid_metrics", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    meta, coords, arrays = archive.layout()
    truth_root = Path(protocol["common_truth"])
    dates = protocol["initialization_times"][::DATE_STRIDE]
    jobs = []
    for start in dates:
        base = dt.datetime.fromisoformat(start.replace("Z", ""))
        for offset in OFFSETS:
            when = base + dt.timedelta(hours=offset)
            last = when + dt.timedelta(hours=protocol["lead_hours"][-1])
            if not (truth_root / last.strftime("%Y%m%dT%H%M%SZ.nc")).exists():
                continue
            jobs.append((start, offset, when))
    hashes = {}
    for _, _, when in jobs:
        iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        hashes[iso] = {}
        for hours in protocol["lead_hours"]:
            valid = when + dt.timedelta(hours=hours)
            receipt = truth_root / valid.strftime("%Y%m%dT%H%M%SZ.nc.READY.json")
            hashes[iso][valid.strftime("%Y-%m-%dT%H:%M:%SZ")] = json.loads(receipt.read_text())["sha256"]
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(archive.fetch_start, archive.time_index(coords, when),
                               arrays, protocol["lead_hours"]) for _, _, when in jobs]
        for i, (date, offset, when) in enumerate(jobs):
            iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
            record = archive.score_start(iso, futures[i].result(), arrays=arrays, coords=coords,
                                         scorer=scorer, protocol=protocol, hashes=hashes)
            futures[i] = None
            record.update({"manifest_date": date, "init_hour_offset": offset})
            results.append(record)
            print(f"{i + 1}/{len(jobs)} {iso}", flush=True)
    OUT.write_text(json.dumps({
        "schema_version": 1,
        "model": "wb2_graphcast_v2_2022",
        "store": archive.BUCKET + archive.STORE,
        "dates": dates,
        "offsets_hours": list(OFFSETS),
        "zero_offset_source": "wb2_archive/graphcast_v2_2022_mse.json",
        "scorer_sha256": sha256(path),
        "script_sha256": sha256(__file__),
        "truth": "campaign common_truth; per-file SHA256 matched to READY receipts",
        "per_initialization": results,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
