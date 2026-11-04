#!/usr/bin/env python3
"""Resume the original 2.8 degree ERA5 store; independent of the model package.

Run a CPU Slurm pilot first. Monthly workers reuse SHA256 frame receipts.
No raw 0.25 degree weather files are retained. All output paths must be scratch.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.neuralgcm_residual.io import read_json, write_json, sha256, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "month", "merge"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith("/scratch/"):
        raise ValueError("Data, caches and logs must be on scratch")
    from src.models.neuralgcm_residual.data import (
        ARCO_SOURCE, PreparedStore, prepare_era5_pair, merge_prepared,
        require_complete_experiment_data,
    )
    import numpy as np
    identity = {"checkpoint_sha256": sha256(args.checkpoint), "libraries": versions(),
                "data_source": ARCO_SOURCE, "resolution": "2.8 degrees, Gaussian 128x64",
                "producer_sha256": sha256(ROOT / "src/models/neuralgcm_residual/data.py"),
                "runner_sha256": sha256(__file__)}
    if identity["checkpoint_sha256"] != "bdec1b4612c7385fc492aa031db252c66fb74b788a7efbb118b8b60b06644d3e":
        raise ValueError("Expected original official deterministic 2.8 degree checkpoint")
    write_json(root / "restore_contract.json", identity, immutable=True)
    if args.stage == "merge":
        paths = [root / "shards" / f"{2015+i//12}-{1+i%12:02d}" / "res2p8" for i in range(108)]
        # Verify every file before publishing the complete store.
        for path in paths:
            store = PreparedStore(path)
            for record in store.manifest["records"]:
                if sha256(path / record["file"]) != record["sha256"]:
                    raise ValueError(f"Corrupt frame: {record['time']}")
        merge_prepared(paths, root / "res2p8")
        store = PreparedStore(root / "res2p8")
        require_complete_experiment_data(store)
        write_json(root / "COMPLETE.json", {"frames": len(store.times), "store_identity": store.identity}, immutable=True)
        return
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    bb = FrozenBackbone.load(args.checkpoint, identity["checkpoint_sha256"])
    if args.stage == "pilot":
        start, end, output = "2015-01-01T00", "2015-01-01T12", root / "pilot" / "res2p8"
    else:
        if not (root / "PILOT_OK.json").exists():
            raise ValueError("Run and verify the two-frame pilot first")
        if not 0 <= args.index < 108:
            raise ValueError("Monthly index must be 0..107")
        year, month = 2015 + args.index // 12, 1 + args.index % 12
        start = f"{year}-{month:02d}-01T00"
        end = f"{year+1}-01-01T00" if month == 12 else f"{year}-{month+1:02d}-01T00"
        output = root / "shards" / start[:7] / "res2p8"
    profile = prepare_era5_pair(ARCO_SOURCE, {"res2p8": bb.model}, {"res2p8": output}, start, end)
    store = PreparedStore(output)
    for time in store.times:
        frame = store.frame(time)
        for field, array in frame.items():
            expected = (128, 64) if field in ("sea_surface_temperature", "sea_ice_cover") else (37, 128, 64)
            if array.shape != expected or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid restored {field} at {time}")
    write_json(output / "profile.json", profile)
    if args.stage == "pilot":
        resumed = prepare_era5_pair(ARCO_SOURCE, {"res2p8": bb.model}, {"res2p8": output}, start, end)
        if resumed["new_frame_timings"]:
            raise ValueError("Resume unexpectedly downloaded an existing frame")
        write_json(root / "PILOT_OK.json", {"frames": 2, "identity": store.identity, "resume_verified": True}, immutable=True)
    print(f"Verified {len(store.times)} frames at {output}", flush=True)


if __name__ == "__main__":
    main()
