"""Frozen jobs must execute their saved code while retaining workspace data paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

from scripts.experiments import frozen_runtime as runtime


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "working checkout"
    _write(root / "src/__init__.py", "")
    _write(root / "src/marker.py", "VALUE = 'saved source'\n")
    _write(root / "scripts/__init__.py", "")
    _write(root / "scripts/experiments/__init__.py", "")
    shutil.copy2(Path(runtime.__file__), root / "scripts/experiments/frozen_runtime.py")
    _write(root / "third_party/graphcast/graphcast/__init__.py", "")
    _write(root / "third_party/graphcast/graphcast/graphcast.py", "VALUE = 'saved vendor'\n")
    _write(root / "inputs/data.txt", "workspace data\n")
    _write(root / "scripts/child.py", """
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        from src import marker
        from graphcast import graphcast

        # Python -c normally puts cwd before PYTHONPATH. The inherited frozen
        # environment must prevent workspace imports in this nested worker too.
        nested = subprocess.run(
            [sys.executable, "-c",
             "import json; from src import marker; from graphcast import graphcast; "
             "print(json.dumps({'source_path': marker.__file__, 'vendor_path': graphcast.__file__}))"],
            capture_output=True, text=True, check=True,
        )
        print(json.dumps({
            "source": marker.VALUE,
            "source_path": str(Path(marker.__file__).resolve()),
            "vendor": graphcast.VALUE,
            "vendor_path": str(Path(graphcast.__file__).resolve()),
            "cwd": str(Path.cwd()),
            "data": Path(sys.argv[1]).read_text(),
            "pythonpath": os.environ["PYTHONPATH"],
            "nested": json.loads(nested.stdout),
        }))
    """)
    _write(root / "scripts/parent.py", """
        import argparse
        import json
        import os
        from pathlib import Path
        import subprocess
        from src import marker
        from graphcast import graphcast
        from scripts.experiments.frozen_runtime import frozen_command

        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        command, options = frozen_command(
            os.environ["WEATHER_SOURCE_ROOT"], os.environ["WEATHER_WORKSPACE"],
            "scripts/child.py", [args.input],
        )
        child = subprocess.run(command, capture_output=True, text=True, check=True, **options)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "source": marker.VALUE,
            "source_path": str(Path(marker.__file__).resolve()),
            "vendor": graphcast.VALUE,
            "vendor_path": str(Path(graphcast.__file__).resolve()),
            "cwd": str(Path.cwd()),
            "project_root": os.environ["PROJECT_ROOT"],
            "child": json.loads(child.stdout),
        }))
    """)
    return root


def test_saved_parent_and_nested_child_ignore_mutated_workspace_and_pythonpath(workspace, tmp_path):
    snapshot = tmp_path / "saved source"
    runtime.freeze_source(workspace, snapshot)
    _write(workspace / "src/marker.py", "raise RuntimeError('workspace source executed')\n")
    _write(workspace / "third_party/graphcast/graphcast/graphcast.py",
           "raise RuntimeError('workspace vendor executed')\n")
    for name in ("parent", "child"):
        _write(workspace / f"scripts/{name}.py", "raise RuntimeError('workspace script executed')\n")
    environment = dict(os.environ, PYTHONPATH=f"{workspace}:{workspace / 'third_party/graphcast'}")
    result = subprocess.run(
        [sys.executable, str(Path(runtime.__file__).resolve()), "exec",
         "--source", str(snapshot), "--workspace", str(workspace),
         "--script", "scripts/parent.py", "--",
         "--input", "inputs/data.txt", "--output", "outputs/result.json"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = json.loads((workspace / "outputs/result.json").read_text())
    for record in (result, result["child"]):
        assert record["source"] == "saved source"
        assert record["source_path"] == str(snapshot / "src/marker.py")
        assert record["vendor"] == "saved vendor"
        assert record["vendor_path"] == str(snapshot / "third_party/graphcast/graphcast/graphcast.py")
        assert record["cwd"] == str(workspace)
    assert result["project_root"] == str(workspace)
    assert result["child"]["data"] == "workspace data\n"
    assert result["child"]["pythonpath"].split(os.pathsep) == [
        str(snapshot), str(snapshot / "third_party/graphcast")]
    assert result["child"]["nested"] == {
        "source_path": str(snapshot / "src/marker.py"),
        "vendor_path": str(snapshot / "third_party/graphcast/graphcast/graphcast.py"),
    }
    assert not (snapshot / "outputs").exists()


def test_repeated_freeze_verifies_saved_copy_without_refreshing_workspace(workspace, tmp_path):
    snapshot = tmp_path / "source"
    original = runtime.freeze_source(workspace, snapshot)
    _write(workspace / "src/marker.py", "VALUE = 'later edit'\n")
    _write(workspace / "src/new_module.py", "VALUE = 'added later'\n")
    (workspace / "scripts/child.py").unlink()

    assert runtime.freeze_source(workspace, snapshot) == original
    assert (snapshot / "src/marker.py").read_text() == "VALUE = 'saved source'\n"
    assert not (snapshot / "src/new_module.py").exists()
    assert (snapshot / "scripts/child.py").is_file()


@pytest.mark.parametrize("operation", ["verify", "command", "refreeze"])
def test_modified_frozen_source_is_rejected(workspace, tmp_path, operation):
    snapshot = tmp_path / "source"
    runtime.freeze_source(workspace, snapshot)
    _write(snapshot / "src/marker.py", "VALUE = 'changed saved source'\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        if operation == "verify":
            runtime.verify_source(snapshot)
        elif operation == "command":
            runtime.frozen_command(snapshot, workspace, "scripts/parent.py", [])
        else:
            runtime.freeze_source(workspace, snapshot)


def test_unrecorded_executable_source_is_rejected(workspace, tmp_path):
    snapshot = tmp_path / "source"
    runtime.freeze_source(workspace, snapshot)
    _write(snapshot / "src/unrecorded.py", "VALUE = 'unverified'\n")
    with pytest.raises(ValueError, match="unrecorded"):
        runtime.verify_source(snapshot)


def test_incomplete_snapshot_cannot_execute_or_be_refreshed(workspace, tmp_path):
    snapshot = tmp_path / "source"
    _write(snapshot / "scripts/parent.py", "raise RuntimeError('incomplete snapshot executed')\n")
    with pytest.raises(FileNotFoundError, match="complete source snapshot"):
        runtime.frozen_command(snapshot, workspace, "scripts/parent.py", [])
    with pytest.raises(FileExistsError, match="Incomplete/legacy"):
        runtime.freeze_source(workspace, snapshot)


def test_freeze_rejects_missing_vendored_graphcast_entrypoint(workspace, tmp_path):
    (workspace / "third_party/graphcast/graphcast/graphcast.py").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="[Gg]raph[Cc]ast|graphcast"):
        runtime.freeze_source(workspace, tmp_path / "source")


def test_manifest_digest_tampering_is_rejected(workspace, tmp_path):
    snapshot = tmp_path / "source"
    runtime.freeze_source(workspace, snapshot)
    path = snapshot / "snapshot.json"
    manifest = json.loads(path.read_text())
    manifest["source_digest"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest differs"):
        runtime.verify_source(snapshot)


def test_command_rejects_workspace_entrypoint(workspace, tmp_path):
    snapshot = tmp_path / "source"
    runtime.freeze_source(workspace, snapshot)
    with pytest.raises(ValueError, match="Entry point is not recorded"):
        runtime.frozen_command(snapshot, workspace, workspace / "scripts/parent.py", [])


def test_command_requires_saved_workspace_identity_and_accepts_resolved_alias(workspace, tmp_path):
    snapshot = tmp_path / "source"
    runtime.freeze_source(workspace, snapshot)
    other_workspace = tmp_path / "other workspace"
    other_workspace.mkdir()
    with pytest.raises(ValueError, match="workspace differs"):
        runtime.frozen_command(snapshot, other_workspace, "scripts/parent.py", [])
    alias = tmp_path / "workspace alias"
    alias.symlink_to(workspace, target_is_directory=True)
    _, options = runtime.frozen_command(snapshot, alias, "scripts/parent.py", [])
    assert options["cwd"] == workspace
    assert options["env"]["WEATHER_WORKSPACE"] == str(workspace)
