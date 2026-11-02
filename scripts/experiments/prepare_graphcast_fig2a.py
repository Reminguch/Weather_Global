#!/usr/bin/env python3
"""Freeze the public GraphCast checkpoint identity and Figure 2a pilot protocol."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import requests

from scripts.experiments.frozen_runtime import freeze_source
from scripts.analyze_models.graphcast_fig2a_data import LocalERA5
from src.data_operations.staging.stage_arco_era5_res1 import DEFAULT_URI

CHECKPOINT_NAME = ("GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - "
                   "mesh 2to6 - precipitation input and output.npz")
EXPECTED_SHA256 = "4e8cf600f3f402ebb430483e782b94648cef57bb757dec24dd7b11760d3fbacc"
UPSTREAM_COMMIT = "08cf73625c9d12bd9aaa038868bcb2fe488f2a22"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(workflow, prepared_root=None, full_year=False):
    if workflow.exists():
        raise FileExistsError(workflow)
    workflow.mkdir(parents=True)
    for folder in ("references", "logs", "results", "forecasts", "report", "control"):
        (workflow / folder).mkdir()
    checkpoint = ROOT / "data/graphcast/graphcast/params" / CHECKPOINT_NAME
    stats = ROOT / "data/graphcast/graphcast/stats"
    identities = []
    for path in [checkpoint] + [stats / f"{name}.nc" for name in (
            "mean_by_level", "stddev_by_level", "diffs_stddev_by_level")]:
        obj = ("params/" if path.suffix == ".npz" else "stats/") + path.name
        url = "https://storage.googleapis.com/storage/v1/b/dm_graphcast/o/" + quote(obj, safe="")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        meta = response.json()
        content = path.read_bytes()
        md5 = base64.b64encode(hashlib.md5(content).digest()).decode()
        if len(content) != int(meta["size"]) or md5 != meta["md5Hash"]:
            raise ValueError(f"Official release mismatch: {path}")
        identities.append({"path": str(path), "sha256": sha(path), "size_bytes": len(content),
                           "md5_base64": md5, "public_metadata_url": url,
                           "public_download_url": "https://storage.googleapis.com/dm_graphcast/" + quote(obj),
                           "official_metadata": meta})
    assert identities[0]["sha256"] == EXPECTED_SHA256
    with np.load(checkpoint, allow_pickle=False) as z:
        config = {k.removeprefix("model_config:"): z[k].tolist() for k in z.files if k.startswith("model_config:")}
        levels = [z[k].item() for k in z.files if k.startswith("task_config:pressure_levels:")]
        description = z["description"].item()
    assert config["resolution"] == .25 and config["mesh_size"] == 6
    assert config["latent_size"] == 512 and config["gnn_msg_steps"] == 16 and len(levels) == 37
    write(workflow / "references/official_release_identity.json", identities)
    upstream = ROOT / "artifacts/evaluations/graphcast_common_grid_2023_20261015/audits/upstream_graphcast"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise ValueError("Official source checkout differs or has tracked changes")
    files = {}
    for source in sorted((upstream / "graphcast").rglob("*.py")):
        target = workflow / "upstream" / source.relative_to(upstream)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[str(target.relative_to(workflow / "upstream"))] = sha(target)
    write(workflow / "upstream/manifest.json", {"commit": commit, "files": files})
    figure_url = "https://arxiv.org/html/2212.12794v2/fig2.png"
    response = requests.get(figure_url, timeout=60)
    response.raise_for_status()
    figure = workflow / "references/paper_figure2.png"
    figure.write_bytes(response.content)
    starts = [f"2018-{month:02d}-15T{hour:02d}:00:00Z" for month in range(1, 13) for hour in (6, 18)]
    if full_year:
        starts = [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in pd.date_range("2018-01-01T06:00", "2018-12-31T18:00", freq="12h")]
    protocol = {
        "schema_version": 1, "experiment": "graphcast_paper_fig2a_2018",
        "prepared_date": "2026-10-19", "timezone": "America/New_York",
        "workspace": str(ROOT), "workflow": str(workflow),
        "scope": "24-start seasonal pilot of the GraphCast curve; paper references are digitized, not rerun HRES",
        "initialization_times": starts, "selection": "15th of each month, 06/18 UTC, fixed before evaluation",
        "paper_full_year_count": 730, "target_steps": 40, "step_hours": 6,
        "lead_hours": list(range(12, 241, 12)), "n_shards": 6,
        "checkpoint": identities[0], "statistics": identities[1:],
        "model_config": config, "pressure_levels": levels, "checkpoint_description": description,
        "upstream_commit": commit, "upstream_directory": str(workflow / "upstream"),
        "era5_source": DEFAULT_URI,
        "data_root": str(ROOT / "data/graphcast/graphcast/dataset/fig2a_2018_20261019"),
        "stage_source": str(workflow / "stage_source"), "eval_source": str(workflow / "eval_source"),
        "numerics": {"internal": "bfloat16", "physical_fields": "float32", "metric": "float32",
                     "xla_flags": "--xla_gpu_enable_triton_gemm=false"},
        "verification": {"field": "geopotential", "pressure_level_hpa": 500,
                         "units": "m2 s-2", "grid": "native 721x1440, 0.25 degrees",
                         "spatial_weighting": "upstream normalized_latitude_weights including finite polar cells",
                         "aggregation": "Eq.20: mean over starts of sqrt(area-weighted spatial MSE)",
                         "uncertainty": "paired-hours date-cluster bootstrap over 12 sampled dates; pilot only"},
        "paper": {"url": "https://arxiv.org/html/2212.12794v2", "figure": "2a",
                  "image_url": figure_url, "image_sha256": sha(figure),
                  "reference_status": "approximate raster digitization of published annual curves"},
    }
    if full_year:
        protocol.update(scope="Full 730-start annual GraphCast curve; paper references are digitized, not rerun HRES",
                        selection="Every 2018 day at 06 and 18 UTC", initialization_times=starts,
                        n_shards=20)
        protocol["verification"]["uncertainty"] = "14-day circular block bootstrap; does not reproduce paper AR(2) intervals"
    if prepared_root is not None:
        local = LocalERA5(prepared_root)
        if local.metadata["checkpoint_sha256"] != EXPECTED_SHA256:
            raise ValueError("Prepared store was built for another checkpoint")
        protocol.update(prepared_root=str(prepared_root), prepared_identity=local.identity,
                        era5_source=local.metadata["source_data_path"],
                        data_root=str(prepared_root), data_mode="local physical float32 year-sharded arrays")
    write(workflow / "protocol.json", protocol)
    freeze_source(ROOT, workflow / ("eval_source" if prepared_root is not None else "stage_source"))
    print(workflow / "protocol.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--full-year", action="store_true")
    args = parser.parse_args()
    prepare(args.workflow.resolve(), args.prepared_root.resolve() if args.prepared_root else None, args.full_year)
