#!/usr/bin/env python3
"""Print structure of a GraphCast ERA5 dataset (NetCDF or Zarr).

Usage (with graphcast311 env):
  python scripts/inspect_graphcast_dataset.py [path]
  python scripts/inspect_graphcast_dataset.py data/graphcast/graphcast/dataset/source-era5_wb13_latest-1y_res-1.0_levels-13_steps-all.zarr
  python scripts/inspect_graphcast_dataset.py data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ZARR = ROOT / "data/graphcast/graphcast/dataset/source-era5_wb13_latest-1y_res-1.0_levels-13_steps-all.zarr"
DEFAULT_NC = ROOT / "data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc"


def _inspect_zarr_metadata(path: Path) -> None:
    """Print structure from .zmetadata only (no xarray/zarr deps)."""
    meta_file = path / ".zmetadata"
    if not meta_file.exists():
        print("No .zmetadata found; need xarray to open Zarr.")
        return
    with open(meta_file) as f:
        meta = json.load(f)
    md = meta.get("metadata", {})
    print("Zarr structure (from .zmetadata):")
    root_keys = sorted({k.split("/")[0] for k in md if not k.startswith(".")})
    print("  Root keys:", root_keys)
    for key in sorted(md):
        if key.endswith("/.zarray"):
            arr = md[key]
            shape = arr.get("shape", [])
            dims = md.get(key.replace("/.zarray", "/.zattrs"), {}).get("_ARRAY_DIMENSIONS", [])
            print(f"  {key.split('/')[0]}: shape={shape} dims={dims}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GraphCast ERA5 dataset structure.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to .nc or .zarr (default: Zarr if exists else NetCDF).",
    )
    parser.add_argument(
        "--compat",
        action="store_true",
        help="Also open with open_graphcast_era5 and print CDS-compatible layout.",
    )
    args = parser.parse_args()

    path = Path(args.path) if args.path else (DEFAULT_ZARR if DEFAULT_ZARR.exists() else DEFAULT_NC)
    if not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)

    print(f"Path: {path}\n")

    if path.is_dir() and (path / ".zmetadata").exists():
        _inspect_zarr_metadata(path)
    elif path.suffix == ".nc":
        print("NetCDF: use --compat to load with open_graphcast_era5 and print layout.\n")

    if args.compat:
        try:
            from src.data.graphcast_dataset import open_graphcast_era5

            ds = open_graphcast_era5(path, time_slice=slice(0, 2))
            print("CDS-compatible layout (open_graphcast_era5, first 2 time steps):")
            print("  sizes:", dict(ds.sizes))
            print("  data_vars:", list(ds.data_vars))
            for v in list(ds.data_vars)[:5]:
                print(f"    {v}: {ds[v].shape}")
            if len(ds.data_vars) > 5:
                print(f"    ... and {len(ds.data_vars) - 5} more")
        except Exception as e:
            print(f"Failed to open with open_graphcast_era5: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
