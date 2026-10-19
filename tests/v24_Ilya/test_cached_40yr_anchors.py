"""Campaign anchor reuse must verify raw index/time/split and pinned inputs."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.experiments import run_v24_cached_40yr as campaign
from src.models.graphcast.training.core import model


@pytest.fixture
def anchored(tmp_path, monkeypatch):
    source = tmp_path / "source.zarr"
    source.mkdir()
    (source / ".zmetadata").write_text("{}")
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"pinned checkpoint")
    prepared = tmp_path / "prepared"
    (prepared / "coords").mkdir(parents=True)
    times = pd.date_range("1982-01-01", "2022-12-31 18:00", freq="6h").values
    np.save(prepared / "coords/time.npy", times)
    campaign.write(prepared / "metadata.json", {"source_data_path": str(source),
                                                 "checkpoint_sha256": campaign.sha(checkpoint)})
    campaign.write(prepared / "READY.json", {"metadata_sha256": campaign.sha(prepared / "metadata.json"),
        "years": campaign.YEARS, "status": "complete", "identity": {
            "checkpoint_sha256": campaign.sha(checkpoint), "source_metadata_sha256": campaign.sha(source / ".zmetadata")},
        "root_file_sha256": {"coords/time.npy": campaign.sha(prepared / "coords/time.npy")}})
    target = tmp_path / "anchors"
    (target / "anchors").mkdir(parents=True)
    task = SimpleNamespace(input_duration="12h", pressure_levels=(500, 850), target_variables=("temperature",))
    monkeypatch.setattr(model, "load_graphcast_checkpoint", lambda _: SimpleNamespace(task_config=task))
    indices = np.arange(2, len(times)-2, dtype=np.int64)
    years = pd.DatetimeIndex(times[indices]).year.to_numpy()
    train, val = np.flatnonzero(years <= 2021), np.flatnonzero(years == 2022)
    for name, value in {"anchor_indices": indices, "anchor_times": times[indices], "split_train": train, "split_val": val}.items():
        np.save(target / "anchors" / f"{name}.npy", value)
    campaign.write(target / "metadata.json", {
        "source_prepared_root": str(prepared), "baseline_checkpoint": str(checkpoint),
        "train_end_year": 2021, "validation_year": 2022, "input_steps": 2, "target_steps": 1, "input_duration": "12h",
        "pressure_levels": [500, 850], "target_variables": ["temperature"], "n_anchors_total": len(indices),
        "n_anchors_train": len(train), "n_anchors_val": len(val), "source_time_start": str(times[0]),
        "source_time_end": str(times[-1]), "campaign_prepared_metadata_sha256": campaign.sha(prepared / "metadata.json"),
        "campaign_baseline_checkpoint_sha256": campaign.sha(checkpoint), "campaign_workflow_id": "workflow"})
    flow = {"workflow_id": "workflow", "paths": {"source_zarr": str(source), "prepared_root": str(prepared),
            "baseline_checkpoint": str(checkpoint), "anchor_root": str(target)},
            "assets": {str(checkpoint): campaign.sha(checkpoint)}}
    return flow, target


def test_verified_anchor_reuse_preserves_files(anchored):
    flow, target = anchored
    before = {path: path.stat().st_mtime_ns for path in target.rglob("*") if path.is_file()}
    campaign.prepare_anchors(flow)
    assert before == {path: path.stat().st_mtime_ns for path in target.rglob("*") if path.is_file()}


@pytest.mark.parametrize("name", ["anchor_indices", "anchor_times", "split_train", "split_val"])
def test_anchor_reuse_rejects_changed_index_time_or_split(anchored, name):
    flow, target = anchored
    path = target / "anchors" / f"{name}.npy"
    values = np.load(path)
    values[0] = values[1]
    np.save(path, values)
    with pytest.raises(ValueError, match="indices/timestamps/split"):
        campaign.prepare_anchors(flow)


def test_anchor_reuse_rejects_replaced_checkpoint(anchored):
    flow, _ = anchored
    Path(flow["paths"]["baseline_checkpoint"]).write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="identity"):
        campaign.prepare_anchors(flow)


def test_anchor_reuse_rejects_replaced_prepared_time(anchored):
    flow, _ = anchored
    path = Path(flow["paths"]["prepared_root"]) / "coords/time.npy"
    times = np.load(path)
    times[1] = times[0]
    np.save(path, times)
    with pytest.raises(ValueError, match="coordinate fingerprint"):
        campaign.prepare_anchors(flow)


def test_anchor_reuse_rejects_changed_campaign_identity(anchored):
    flow, target = anchored
    path = target / "metadata.json"
    metadata = campaign.read(path)
    metadata["campaign_workflow_id"] = "other"
    campaign.write(path, metadata)
    with pytest.raises(ValueError, match="metadata differs"):
        campaign.prepare_anchors(flow)
