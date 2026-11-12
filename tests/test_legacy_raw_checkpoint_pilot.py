"""Checks for raw legacy endpoints, reference reuse and frozen-code inheritance."""
import json
from pathlib import Path

import pytest

from scripts.experiments import prepare_v24_legacy_raw_pilot as legacy
from scripts.experiments import run_v24_raw_checkpoint_pilot as pilot
from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_legacy_endpoints_keep_separate_controls_and_training_histories(tmp_path):
    for identity, _, campaign, name, step, weight, _ in legacy.SPECS:
        directory = tmp_path / legacy.BASE / campaign / "runs" / name
        objective = {"loss_kind": "mse_fp32"} if weight is None else {
            "loss_kind": "amse", "amse_lambda": weight}
        write_file(directory / "run_config.json", json.dumps({"objective": objective}))
        write_file(directory / "checkpoints" / f"checkpoint_step{step:08d}.pkl", identity)

    models = legacy.legacy_models(tmp_path)
    assert len(models) == len({m["id"] for m in models}) == 8
    assert len({m["checkpoint"]["path"] for m in models}) == 8
    assert all("/checkpoints/" in m["checkpoint"]["path"] for m in models)
    assert all("/swa/" not in m["checkpoint"]["path"] for m in models)
    by_id = {m["id"]: m for m in models}
    assert {m["step"] for m in models[:4]} == {2000}
    assert {m["step"] for m in models[4:]} == {500}
    assert {m["plot_group"] for m in models} == {
        "September λ0.1 pair", "September λ1 pair", "October 500-update continuation"}
    assert by_id["sep24_mse"]["checkpoint"] != by_id["sep19_mse"]["checkpoint"]
    assert by_id["oct04_mse"]["lambda"] is None
    assert by_id["oct04_mse"]["objective"]["loss_kind"] == "mse_fp32"
    assert by_id["oct04_l000"]["lambda"] == 0
    assert by_id["oct04_l000"]["objective"]["loss_kind"] == "amse"
    assert [by_id[key]["lambda"] for key in (
        "sep24_l010", "sep19_l100", "oct04_l000", "oct04_l025", "oct04_l050")] == [.1, 1., 0., .25, .5]


def test_submission_omits_reused_references_and_keeps_full_dependency(tmp_path):
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source"),
                "models": [{"id": "M1_raw", "reuse": {}}, {"id": "M1_swa", "reuse": {}}]
                          + [{"id": f"legacy_{i}"} for i in range(8)]}
    command = pilot.submission_command(tmp_path, manifest, "evaluate")
    assert [a for a in command if a.startswith("--array=")] == ["--array=2,3,4,5,6,7,8,9"]
    assert not any("%" in a for a in command if a.startswith("--array="))
    dependency = "100:200_0:200_1"
    report = pilot.submission_command(tmp_path, manifest, "report", dependency)
    assert f"--dependency=afterok:{dependency}" in report
    assert not any(a.startswith("--gres=") or a.startswith("--array=") for a in report)
    manifest["models"] = manifest["models"][:2]
    with pytest.raises(ValueError, match="No new evaluations"):
        pilot.submission_command(tmp_path, manifest, "evaluate")


def test_freeze_extension_inherits_computational_code_and_preserves_parent(tmp_path):
    workspace, parent, destination = [tmp_path / name for name in ("workspace", "parent", "extension")]
    contents = {
        "src/forecast.py": "# frozen forecast\n",
        "scripts/experiments/v24_raw_checkpoint_pilot.slurm": "# frozen wrapper\n",
        "scripts/graphcast_env.sh": "# frozen environment\n",
        "third_party/graphcast/graphcast/graphcast.py": "# frozen GraphCast\n",
        legacy.OVERLAYS[1]: "# prior orchestration\n",
        legacy.OVERLAYS[2]: "# prior report\n",
    }
    for relative, contents_text in contents.items():
        write_file(parent / relative, contents_text)
        write_file(workspace / relative, "# unrelated mutable workspace version\n")
    files = {relative: file_sha256(parent / relative) for relative in contents}
    snapshot_path = parent / "snapshot.json"
    snapshot_path.write_text(json.dumps({"format": FORMAT, "workspace": str(workspace),
                                         "files": files, "source_digest": _digest(files)}))
    before = snapshot_path.read_bytes()
    for relative in legacy.OVERLAYS:
        write_file(workspace / relative, f"# intended orchestration overlay: {relative}\n")

    result = legacy.freeze_extension(workspace, parent, destination)
    assert snapshot_path.read_bytes() == before
    assert verify_source(parent)["source_digest"] == _digest(files)
    assert result["parent_snapshot"]["sha256"] == file_sha256(snapshot_path)
    assert set(result["files"]) == set(files) | set(legacy.OVERLAYS)
    for relative in result["files"]:
        expected = workspace if relative in legacy.OVERLAYS else parent
        assert (destination / relative).read_bytes() == (expected / relative).read_bytes()
    assert (destination / "src/forecast.py").read_bytes() != (workspace / "src/forecast.py").read_bytes()
    with pytest.raises(FileExistsError):
        legacy.freeze_extension(workspace, parent, destination)
