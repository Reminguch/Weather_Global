"""New batch/spatial jobs pin code, configs and data before entering the queue."""
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
from scripts.experiments import run_v24_res1_batch4_lr as driver


REPO = Path(__file__).resolve().parents[2]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(value))


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace with spaces"
    for package in ("scripts", "scripts/experiments", "src", "src/data_operations",
                    "src/data_operations/loaders", "third_party/graphcast/graphcast"):
        _write(workspace / package / "__init__.py", "")
    for relative in (
        "scripts/experiments/frozen_runtime.py",
        "scripts/experiments/run_v24_res1_batch4_lr.py",
        "scripts/experiments/run_v24_res1_slow_spatial.py",
        "scripts/experiments/run_v24_res1_slow_spatial_8k.py",
        "scripts/experiments/run_frozen_experiment.sh",
        "scripts/experiments/v24_res1_batch4_lr.slurm",
        "scripts/experiments/v24_res1_slow_spatial.slurm",
        "scripts/experiments/v24_res1_slow_spatial_8k.slurm",
        "src/data_operations/loaders/zarr_metadata.py",
    ):
        shutil.copy2(REPO / relative, workspace / relative)
    _write(workspace / "third_party/graphcast/graphcast/graphcast.py", "SAVED_VENDOR = True\n")
    # Model-free activation stub deliberately reproduces the real script's cd.
    _write(workspace / "scripts/graphcast_env.sh", 'cd "${PROJECT_ROOT}"\n')
    _write(workspace / "scripts/analyze_models/eval_v24_Ilya.py", """
        import argparse
        import json
        from pathlib import Path
        import sys
        from graphcast import graphcast

        assert graphcast.SAVED_VENDOR
        parser = argparse.ArgumentParser()
        parser.add_argument("--out-json", type=Path, required=True)
        parser.add_argument("--data-path", type=Path, required=True)
        parser.add_argument("--ckpt", type=Path, required=True)
        args, _ = parser.parse_known_args()
        assert args.data_path.is_dir()
        args.out_json.write_text(json.dumps({
            "chosen_idx": [1, 2], "evaluated_samples": 32,
            "original_graphcast_loss": {"full_per_step": [1.0] * 40},
            "script": str(Path(__file__).resolve()), "cwd": str(Path.cwd()),
            "checkpoint": str(args.ckpt),
        }))
    """)
    template_path = workspace / driver.BASE_CONFIG.relative_to(driver.WORKSPACE)
    template = {
        "data": {name: f"data/{name}" for name in (
            "prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir")},
        "distributed": {}, "validation": {}, "optimizer": {},
        "output": {}, "architecture": {},
    }
    _write(template_path, json.dumps(template))
    eval_data = workspace / "data/eval.zarr"
    _write(eval_data / "zarr.json", json.dumps({
        "zarr_format": 3, "node_type": "group",
        "consolidated_metadata": {"kind": "inline", "metadata": {"field": {"node_type": "array"}}},
    }))
    reference = workspace / "reference.json"
    _write(reference, json.dumps({"chosen_idx": [1, 2]}))
    experiment = workspace / "artifacts/new_experiment"
    for name, value in {
        "ROOT": workspace, "WORKSPACE": workspace, "BASE_CONFIG": template_path,
        "EXPERIMENT": experiment, "EVAL_DATA": str(eval_data), "REFERENCE": reference,
        "BRANCH_UPDATES": 2000, "SWA_STEPS": (500, 1000, 1500, 2000),
        "ARMS": [("joint", 1., 1.), ("slow_mamba", 1., .3), ("frozen_spatial", 0., 1.)],
    }.items():
        monkeypatch.setattr(driver, name, value)
    return workspace, experiment, eval_data


def _manifest(experiment):
    return json.loads((experiment / "manifest.json").read_text())


def test_metadata_failure_prevents_freezing_or_creating_experiment(workflow, monkeypatch):
    _, experiment, eval_data = workflow
    (eval_data / "zarr.json").write_text("not valid metadata")
    monkeypatch.setattr(driver, "freeze_source", lambda *args: pytest.fail("Must validate metadata first"))
    with pytest.raises(ValueError, match="Invalid Zarr metadata"):
        driver.prepare()
    assert not experiment.exists()


def test_prepare_pins_complete_source_configs_metadata_and_workspace_paths(workflow):
    workspace, experiment, eval_data = workflow
    driver.prepare()
    manifest = _manifest(experiment)
    source = experiment / "source"
    saved = runtime.verify_source(source)
    assert "third_party/graphcast/graphcast/graphcast.py" in saved["files"]
    assert "scripts/experiments/run_v24_res1_batch4_lr.py" in saved["files"]
    assert manifest["source_root"] == str(source)
    assert manifest["workspace"] == str(workspace)
    assert manifest["source_manifest_sha256"] == runtime.file_sha256(source / "snapshot.json")
    assert manifest["eval_data"] == str(eval_data)
    assert manifest["eval_data_metadata"] == {
        "path": str(eval_data / "zarr.json"), "sha256": runtime.file_sha256(eval_data / "zarr.json")}
    assert len(manifest["entries"]) == 6
    assert len(manifest["config_hashes"]) == 8  # Two shared seeds and six branch arms.
    for path, digest in manifest["config_hashes"].items():
        assert Path(path).is_absolute()
        assert runtime.file_sha256(path) == digest
        config = json.loads(Path(path).read_text())
        for key, data_path in config["data"].items():
            assert data_path == str(workspace / f"data/{key}")
        assert Path(config["output"]["output_root"]).is_relative_to(experiment)
    for entry in manifest["entries"]:
        assert Path(entry["config"]).is_absolute()
        assert Path(entry["init_from"]).is_relative_to(experiment)
        assert entry["warmup_config"] in manifest["config_hashes"]
    original = (experiment / "manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match="already prepared"):
        driver.prepare()
    assert (experiment / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("mutation", ["configuration", "source", "metadata", "live_execution", "source_identity"])
def test_runtime_rejects_changed_inputs_and_live_workspace(workflow, monkeypatch, mutation):
    workspace, experiment, eval_data = workflow
    driver.prepare()
    manifest = _manifest(experiment)
    source = experiment / "source"
    monkeypatch.setattr(driver, "ROOT", source)
    driver.require_runtime(manifest)
    if mutation == "configuration":
        path = Path(next(iter(manifest["config_hashes"])))
        path.write_text(path.read_text() + "\n")
        error, message = ValueError, "configuration changed"
    elif mutation == "source":
        path = source / "scripts/experiments/run_v24_res1_batch4_lr.py"
        path.write_text(path.read_text() + "\n# changed\n")
        error, message = ValueError, "Frozen source changed"
    elif mutation == "metadata":
        path = eval_data / "zarr.json"
        path.write_text(path.read_text() + "\n")
        error, message = ValueError, "dataset metadata changed"
    elif mutation == "live_execution":
        monkeypatch.setattr(driver, "ROOT", workspace)
        error, message = RuntimeError, "prepared source snapshot"
    else:
        manifest["source_manifest_sha256"] = "0" * 64
        error, message = ValueError, "source identity differs"
    with pytest.raises(error, match=message):
        driver.require_runtime(manifest)


def test_spatial_8k_prepares_and_launches_maintained_frozen_driver(workflow, tmp_path):
    workspace, experiment, eval_data = workflow
    # There is intentionally no historical artifacts/.../source/driver_8k.py.
    # Importing the actual wrapper exercises its overrides on the maintained driver.
    code = """
        from pathlib import Path
        import runpy
        import sys
        workspace, experiment, eval_data = map(Path, sys.argv[1:])
        sys.path.insert(0, str(workspace))
        from scripts.experiments import run_v24_res1_batch4_lr as driver
        driver.EVAL_DATA = str(eval_data)
        driver.REFERENCE = workspace / 'reference.json'
        sys.argv = ['spatial', 'prepare', '--experiment', str(experiment)]
        runpy.run_path(str(workspace / 'scripts/experiments/run_v24_res1_slow_spatial_8k.py'), run_name='__main__')
    """
    environment = dict(os.environ, WEATHER_WORKSPACE=str(workspace))
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code), str(workspace), str(experiment), str(eval_data)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = _manifest(experiment)
    assert manifest["branch_updates"] == 8000
    assert manifest["swa_steps"] == [2000, 4000, 6000, 8000]
    assert {entry["spatial_lr_multiplier"] for entry in manifest["entries"]} == {1., .2, .1}
    assert {entry["mamba_lr_multiplier"] for entry in manifest["entries"]} == {1.}
    for entry in manifest["entries"]:
        assert json.loads(Path(entry["config"]).read_text())["optimizer"]["max_steps"] == 8000

    # Queue delay can change every live entry point without affecting the job.
    for relative in ("scripts/experiments/run_v24_res1_slow_spatial_8k.py",
                     "scripts/experiments/run_v24_res1_batch4_lr.py",
                     "scripts/analyze_models/eval_v24_Ilya.py"):
        _write(workspace / relative, "raise RuntimeError('live workspace executed')\n")
    source = experiment / "source"
    environment.update(SLURM_SUBMIT_DIR=str(tmp_path), SLURM_ARRAY_TASK_ID="0",
                       PYTHONPATH=str(workspace))
    result = subprocess.run(
        ["bash", str(source / "scripts/experiments/v24_res1_slow_spatial_8k.slurm"),
         "eval", str(experiment), str(workspace)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    entry = manifest["entries"][0]
    outputs = experiment / "runs" / entry["name"] / "eval/cold_full_zero_exact_gc/matched32"
    assert {path.name for path in outputs.glob("*.json")} == {
        "shared_start.json", "step8000.json", "swa_step02000-08000.json"}
    for path in outputs.glob("*.json"):
        saved = json.loads(path.read_text())
        assert saved["script"] == str(source / "scripts/analyze_models/eval_v24_Ilya.py")
        assert saved["cwd"] == str(workspace)
    assert json.loads((outputs / "step8000.json").read_text())["checkpoint"].endswith(
        "/checkpoints/checkpoint_step00008000.pkl")
