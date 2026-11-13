"""Fresh low-rate trajectories, preserved computation, and exact allocation resume."""
import copy
from pathlib import Path

import pytest

from scripts.experiments import run_v24_full_gc_low_lr_restart as restart
from scripts.experiments.frozen_runtime import FORMAT, VENDORED_ENTRYPOINT, _digest, file_sha256


def task(weight=.25):
    return {"id": f"gcft_s22_l{round(weight * 100):03d}_lr3em7_fresh_20261112",
            "logical_id": f"gcft_s22_l{round(weight * 100):03d}",
            "model_family": "full_gc", "lambda": weight, "learning_rate": 3e-7, "resume_step": 0}


def original_config(weight=.25):
    selected = restart.unseal(restart.ROOT / restart.REFERENCE / "selected_arms_full_gc.json")
    original = next(value for value in selected["tasks"] if value["lambda"] == weight)
    return restart.read(original["config"]["path"])


@pytest.mark.parametrize("weight", restart.LAMBDAS)
def test_restart_changes_only_rate_and_output_from_actual_old_config(tmp_path, weight):
    original = original_config(weight)
    before = copy.deepcopy(original)
    identity = task(weight)
    config = restart.make_config(original, tmp_path, identity)
    restart.validate_config(config, identity)
    restored = copy.deepcopy(config)
    restored["optimizer"]["learning_rate"] = original["optimizer"]["learning_rate"]
    restored["optimizer"]["end_learning_rate"] = original["optimizer"]["end_learning_rate"]
    restored["output"] = original["output"]
    assert restored == original == before
    assert config["optimizer"]["max_steps"] == 2500
    assert config["data"]["train_order_manifest"] == original["data"]["train_order_manifest"]


def test_first_allocation_is_fresh_and_second_exactly_resumes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(restart.scalar, "run_training_stage", lambda *args, **kwargs: calls.append(kwargs))
    manifest = {"tasks": [task()]}
    restart.train(tmp_path, manifest, 0, 0)
    restart.train(tmp_path, manifest, 0, 1)
    assert calls == [{"stop": 1250, "previous": None}, {"stop": 2500, "previous": 1250}]
    with pytest.raises(ValueError, match="Invalid"):
        restart.train(tmp_path, manifest, 1, 0)


def test_submission_has_only_four_fresh_arms_and_corresponding_continuations(tmp_path):
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source")}
    plan = restart.submission_plan(tmp_path, manifest)
    assert [node["id"] for node in plan] == ["train-part1", "train-part2", "history", "finish"]
    assert "--array=0-3" in plan[0]["command"]
    assert "--time=1800" in plan[0]["command"] and "--mem=384G" in plan[0]["command"]
    assert "--dependency=aftercorr:<train-part1>" in plan[1]["command"]
    assert "--dependency=aftercorr:<train-part2>" in plan[2]["command"]
    assert "--dependency=afterok:<train-part2>" in plan[3]["command"]
    assert not any("%" in part for node in plan for part in node["command"] if part.startswith("--array"))


def test_extension_preserves_parent_computation_instead_of_workspace(tmp_path):
    workspace, parent, destination = (tmp_path / name for name in ("workspace", "parent", "source"))
    relative = "src/model.py"
    (parent / "src").mkdir(parents=True)
    (workspace / "src").mkdir(parents=True)
    (parent / relative).write_text("original computation\n")
    (workspace / relative).write_text("different working computation\n")
    (parent / VENDORED_ENTRYPOINT).parent.mkdir(parents=True)
    (parent / VENDORED_ENTRYPOINT).write_text("original GraphCast\n")
    files = {name: file_sha256(parent / name) for name in (relative, VENDORED_ENTRYPOINT)}
    restart.write(parent / "snapshot.json", {"format": FORMAT, "workspace": str(workspace),
        "files": files, "source_digest": _digest(files)})
    for overlay in restart.OVERLAYS:
        path = workspace / overlay
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new orchestration\n")
    result = restart.freeze_extension(workspace, parent, destination)
    assert (destination / relative).read_bytes() == (parent / relative).read_bytes()
    assert result["files"][relative] == files[relative]
    assert set(result["files"]) == {relative, VENDORED_ENTRYPOINT, *restart.OVERLAYS}


def test_high_rate_or_inherited_initialization_is_rejected(tmp_path):
    identity = task()
    config = restart.make_config(original_config(), tmp_path, identity)
    config["optimizer"]["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="learning rates"):
        restart.validate_config(config, identity)
    config["optimizer"]["learning_rate"] = 3e-7
    config["init_from"] = "/archive/old_high_rate.pkl"
    with pytest.raises(ValueError, match="inherit"):
        restart.validate_config(config, identity)
