from pathlib import Path

import pytest

from scripts.experiments import run_v24_smoothing_sweep as sweep
from scripts.experiments.frozen_runtime import freeze_source, verify_source


def test_sweep_submission_has_complete_unthrottled_array_and_afterok_report(tmp_path):
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source"),
        "model_id": "M1", "shards": {"evaluate": [{"count": 16}] * 8},
        "resource_measurement": {"memory_gib": 20, "minutes": 180}}
    command = sweep.sbatch(tmp_path, manifest, "evaluate")
    assert "--array=0-7" in command and "--mem=20G" in command
    assert "--time=180" in command and "--gres=gpu:1" in command
    assert command[-5] == str(tmp_path / "source" / sweep.WRAPPER)
    assert not any("%" in arg for arg in command if arg.startswith("--array="))
    report = sweep.sbatch(tmp_path, manifest, "report", dependency="123")
    assert "--dependency=afterok:123" in report
    assert not any(arg.startswith("--gres=") for arg in report)


def test_snapshot_extension_preserves_parent_and_detects_changed_source(tmp_path, monkeypatch):
    workspace, parent, result = [tmp_path / name for name in ("workspace", "parent", "result")]
    for relative, content in {
        "src/model.py": "ORIGINAL = True\n",
        "scripts/eval.py": "# old evaluation\n",
        "third_party/graphcast/graphcast/graphcast.py": "# vendored\n",
    }.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    before = freeze_source(workspace, parent)
    (workspace / "src/model.py").write_text("UNRELATED_CHANGE = True\n")
    (workspace / "scripts/new.py").write_text("# authorized extension\n")
    monkeypatch.setattr(sweep, "OVERLAYS", ("scripts/new.py",))
    frozen = sweep.freeze_extension(workspace, parent, result)
    assert (result / "src/model.py").read_text() == "ORIGINAL = True\n"
    assert frozen["files"]["scripts/new.py"]
    assert verify_source(parent)["source_digest"] == before["source_digest"]
    (result / "src/model.py").write_text("# corrupt\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        verify_source(result)
