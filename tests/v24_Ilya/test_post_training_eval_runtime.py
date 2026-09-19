"""Keep frozen post-training evaluations on the Mamba-enabled GraphCast."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[2]


def _snapshot(tmp_path: Path, *, include_graphcast: bool) -> Path:
    source = tmp_path / "source"
    for relative in (
        "scripts/experiments/run_v24_post_training_eval.py",
        "src/models/graphcast/training/core/bootstrap.py",
        "src/models/mamba/modules/temporal_mesh_mamba.py",
        "src/models/mamba/modules/temporal_mesh_mamba_Ilya.py",
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    if include_graphcast:
        shutil.copytree(
            ROOT / "third_party/graphcast/graphcast",
            source / "third_party/graphcast/graphcast",
            ignore=shutil.ignore_patterns("__pycache__", "*_test.py"),
        )
    return source


def _validate_in_fresh_process(source: Path, *, import_mode: str) -> dict:
    # -I excludes the working repository and inherited PYTHONPATH, which would
    # otherwise hide an incomplete snapshot behind this test runner's imports.
    code = textwrap.dedent(
        """
        import inspect
        import json
        from pathlib import Path
        import sys

        from jax._src import xla_bridge

        def forbid_backend_initialization(platform):
            raise AssertionError("Runtime validation must not initialize JAX devices")

        xla_bridge._init_backend = forbid_backend_initialization
        source = Path(sys.argv[1])
        import_mode = sys.argv[2]
        installed_path = None
        if import_mode == "installed_first":
            from graphcast import graphcast as installed_gc
            installed_path = str(Path(installed_gc.__file__).resolve())

        sys.path.insert(0, str(source))
        if import_mode == "missing_vendor":
            from src.models.graphcast.training.core import bootstrap
            from graphcast import graphcast as installed_gc
            installed_path = str(Path(installed_gc.__file__).resolve())

        from scripts.experiments.run_v24_post_training_eval import validate_graphcast_runtime

        try:
            validate_graphcast_runtime()
        except RuntimeError as exc:
            result = {"ok": False, "error": str(exc)}
        else:
            from graphcast import graphcast as gc
            result = {
                "ok": True,
                "graphcast_path": str(Path(gc.__file__).resolve()),
                "accepts_is_training": "is_training" in inspect.signature(gc.GraphCast._run_mesh_gnn).parameters,
                "has_mamba": callable(getattr(gc.GraphCast, "_run_mesh_gnn_interleaved", None)),
            }
        result["installed_path"] = installed_path
        result["backend_initialized"] = xla_bridge.backends_are_initialized()
        print(json.dumps(result))
        """
    )
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source), import_mode],
        cwd=source.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert not payload["backend_initialized"]
    return payload


def test_complete_snapshot_loads_its_mamba_graphcast_without_devices(tmp_path):
    source = _snapshot(tmp_path, include_graphcast=True)
    result = _validate_in_fresh_process(source, import_mode="fresh")

    assert result["ok"], result
    assert result["graphcast_path"] == str(
        source / "third_party/graphcast/graphcast/graphcast.py"
    )
    assert result["accepts_is_training"]
    assert result["has_mamba"]


def test_incomplete_snapshot_rejects_installed_graphcast_without_devices(tmp_path):
    source = _snapshot(tmp_path, include_graphcast=False)
    result = _validate_in_fresh_process(source, import_mode="missing_vendor")

    assert not Path(result["installed_path"]).is_relative_to(source)
    assert not result["ok"], result
    assert "graphcast" in result["error"].lower()
    assert str(source / "third_party/graphcast") in result["error"]


def test_complete_snapshot_rejects_previously_imported_installed_graphcast(tmp_path):
    source = _snapshot(tmp_path, include_graphcast=True)
    result = _validate_in_fresh_process(source, import_mode="installed_first")

    assert not Path(result["installed_path"]).is_relative_to(source)
    assert not result["ok"], result
    assert "graphcast" in result["error"].lower()
    assert result["installed_path"] in result["error"]
