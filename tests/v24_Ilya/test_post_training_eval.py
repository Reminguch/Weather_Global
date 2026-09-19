from pathlib import Path

from scripts.experiments.run_v24_post_training_eval import discover, eval_args, swa_windows
from src.models.mamba.v24_Ilya.config import parse_args
import json


def test_discovery_includes_only_completed_checkpoint_names(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("checkpoint_step00000300.pkl", "checkpoint_step00000100.pkl",
                 "checkpoint_step00000400.pkl.tmp", "checkpoint_stepbad.pkl"):
        (checkpoints / name).touch()
    assert [step for step, _ in discover(tmp_path)] == [100, 300]


def test_swa_windows_adapt_to_timeout_and_deduplicate():
    assert swa_windows([100]) == []
    assert swa_windows([100, 200, 300]) == [("swa_all", [100, 200, 300])]
    assert swa_windows([100, 200, 300, 400]) == [
        ("swa_all", [100, 200, 300, 400]), ("swa_last3", [200, 300, 400])]


def test_real_run_architecture_and_shard_cli(tmp_path):
    config = json.loads(Path(
        "configs/experiments/v24_Ilya/res0p25_uniform_full24_legacy_di64_bcg2_bf16_dp4_lr1em5_sg500.json"
    ).read_text())
    args = eval_args(config, tmp_path / "checkpoint.pkl", tmp_path / "eval.json", 3)
    parsed = parse_args(args)
    assert parsed.architecture.temporal_d_inner == 64
    assert parsed.architecture.temporal_bc_groups == 2
    assert parsed.architecture.temporal_zero_init_out
    assert parsed.architecture.resolution == 0.25
    assert parsed.anchor_shard_count == 4
    assert parsed.anchor_shard_index == 3
    assert parsed.n_samples == 32
    assert parsed.target_steps == 40
    assert parsed.residual_state_init == "zero"


def test_prepare_submits_discovered_variants_and_exact_merge(tmp_path, monkeypatch):
    from scripts.experiments import run_v24_post_training_eval as driver
    monkeypatch.setattr(driver, "validate_graphcast_runtime", lambda: None)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    for step in (100, 200, 300, 400):
        (run / "checkpoints" / f"checkpoint_step{step:08d}.pkl").write_bytes(b"checkpoint")
    (run / "run_config.json").write_text("{}")
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    (workflow / "settings.json").write_text(json.dumps({"run_dir": str(run), "training_job": "123"}))
    def fake_build(args):
        output = Path(args[-1])
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"swa")
    submitted = []
    def fake_submit(workflow, phase, options):
        submitted.append((phase, options))
        return "456"
    monkeypatch.setattr(driver, "command", fake_build)
    monkeypatch.setattr(driver, "submit", fake_submit)
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    driver.prepare(workflow)
    manifest = json.loads((workflow / "manifest.json").read_text())
    assert len(manifest["variants"]) == 6
    assert "--array=0-23" in submitted[0][1]
    assert "--dependency=afterok:124" in submitted[0][1]
    assert "--dependency=afterok:456" in submitted[1][1]
