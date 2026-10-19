"""Recovery must reject modified sources and use snapshot subprocess paths."""
import pytest

from scripts.experiments import run_v24_eval_recovery as recovery


def test_source_mutation_fails_before_copy(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    expected = recovery.sha(source)
    source.write_text("modified")
    target = tmp_path / "runtime/source.py"
    with pytest.raises(ValueError, match="Source checksum"):
        recovery.copy_verified(source, target, expected)
    assert not target.exists()


def test_subprocess_uses_frozen_script_and_absolute_data(tmp_path):
    runtime = tmp_path / "snapshot"
    workspace = tmp_path / "workspace"
    command = recovery.isolated_command(runtime, workspace, [
        "scripts/analyze_models/eval_v24_Ilya.py", "--ckpt-in", "data/baseline.npz",
        "--stats-dir", "data/stats", "--ckpt", "/preserved/model.pkl"])
    assert command[2] == str(runtime / "scripts/analyze_models/eval_v24_Ilya.py")
    assert command[4] == str(workspace / "data/baseline.npz")
    assert command[6] == str(workspace / "data/stats")
    assert command[8] == "/preserved/model.pkl"


def test_verify_rejects_changed_checkpoint(tmp_path):
    runtime = tmp_path / "source"
    runtime.mkdir()
    (runtime / "model.py").write_text("model")
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"weights")
    recovery.write(tmp_path / "manifest.json", {
        "runtime": str(runtime), "files": {"model.py": recovery.sha(runtime / "model.py")},
        "inputs": {str(checkpoint): recovery.sha(checkpoint)}})
    recovery.verify(tmp_path)
    checkpoint.write_bytes(b"different")
    with pytest.raises(ValueError, match="Recovery input changed"):
        recovery.verify(tmp_path)
