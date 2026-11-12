"""Protect the historical schedule and the two different state-transfer boundaries."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_v24_m1_matched_gcm_lambda as campaign


def task(weight=0.):
    return {"id": f"gcm_m1matched_s22_l{round(weight * 100):03d}",
            "logical_id": "logical", "model_family": "gc_mamba", "lambda": weight}


def test_stage_configs_reproduce_m1_schedule_and_only_add_requested_penalty(tmp_path):
    for stage, relative in campaign.TEMPLATES.items():
        template = campaign.read(campaign.ROOT / relative)
        original = copy.deepcopy(template)
        for weight in campaign.LAMBDAS:
            cfg = campaign.make_config(template, tmp_path, task(weight), campaign.ROOT, stage)
            assert cfg["model_family"] == "gc_mamba"
            assert cfg["sequence"] == template["sequence"]
            assert cfg["memory"] == template["memory"]
            assert "train_order_manifest" not in cfg["data"]
            assert cfg["architecture"]["residual_initialization"] == "baseline_overlay"
            for key in ("max_steps", "learning_rate", "warmup_steps", "end_learning_rate",
                        "learning_rate_schedule", "adam_beta1", "adam_beta2", "grad_clip",
                        "weight_decay", "weight_decay_policy", "precision", "seed"):
                assert cfg["optimizer"][key] == template["optimizer"][key]
            assert cfg["optimizer"]["mamba_lr_multiplier"] == (1. if stage == "initial" else 3.)
            assert cfg["objective"]["loss_kind"] == ("native_mse_amse_excess" if weight else "mse")
            assert cfg["objective"].get("spectral_excess_weight", 0.) == weight
        assert original == template


def test_total_update_mapping_is_not_a_filename_alias(tmp_path):
    expected = {500: ("initial", 500), 1000: ("main", 500), 1250: ("main", 750),
                1500: ("main", 1000), 2000: ("main", 1500), 2500: ("main", 2000)}
    for total, wanted in expected.items():
        stage, stored, path = campaign.checkpoint_for_total(tmp_path, task(), total)
        assert (stage, stored) == wanted
        assert path.name == f"checkpoint_step{stored:08d}.pkl"
    with pytest.raises(ValueError, match="Unsupported"):
        campaign.checkpoint_for_total(tmp_path, task(), 750)


def test_initial_stage_transfer_resets_state_but_operational_resume_does_not(tmp_path):
    t = task()
    t.update(initial_config={"path": "initial.json"}, main_config={"path": "main.json"})
    manifest = {"source_root": str(tmp_path / "source")}
    initial = campaign.training_command(tmp_path, manifest, t, "initial", 500)
    assert "--init-from" not in initial and "--resume" not in initial
    main = campaign.training_command(tmp_path, manifest, t, "main", 750, init_from="initial500.pkl")
    assert main[main.index("--init-from") + 1] == "initial500.pkl"
    assert "--zero-state-on-init" in main and "--resume" not in main
    resume = campaign.training_command(tmp_path, manifest, t, "main", 2000, resume="main750.pkl")
    assert resume[resume.index("--resume") + 1] == "main750.pkl"
    assert "--init-from" not in resume and "--zero-state-on-init" not in resume


def test_first_part_is_500_plus750_and_second_resumes750(tmp_path, monkeypatch):
    t = task()
    manifest = {"tasks": [t]}
    calls = []
    monkeypatch.setattr(campaign, "gpu_required", lambda: None)
    initial = {"stage": "initial", "completed_steps": 500}
    prior = {"stage": "main", "completed_steps": 750}
    def run(experiment, manifest, task, stage, stop, total, **kwargs):
        calls.append((stage, stop, total, kwargs))
        return initial if stage == "initial" else prior
    monkeypatch.setattr(campaign, "run_stage", run)
    campaign.train(tmp_path, manifest, 0, 1)
    assert calls == [("initial", 500, 500, {}),
                     ("main", 750, 1250, {"initial_proof": initial})]
    monkeypatch.setattr(campaign, "unseal", lambda path: initial if "initial" in path.name else prior)
    calls.clear()
    campaign.train(tmp_path, manifest, 0, 2)
    assert calls == [("main", 2000, 2500, {"previous": prior, "initial_proof": initial})]


def test_jobs_pair_same_arm_and_do_not_throttle_or_schedule_full_gc(tmp_path):
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source")}
    plan = campaign.submission_plan(tmp_path, manifest)
    assert [node["id"] for node in plan] == ["train-part1", "train-part2", "history", "finish"]
    assert "--dependency=aftercorr:<train-part1>" in plan[1]["command"]
    for node in plan[:2]:
        cmd = node["command"]
        assert "--array=0-4" in cmd and "--mem=384G" in cmd and "--time=1800" in cmd
        assert not any("%" in arg for arg in cmd if arg.startswith("--array="))
        assert not any("full_gc" in arg for arg in cmd)
    wrapper = (campaign.ROOT / campaign.WRAPPER).read_text()
    assert "unset JAX_PLATFORMS JAX_PLATFORM_NAME" in wrapper


def fake_stage(tmp_path, monkeypatch):
    from src.models.mamba.v24_Ilya import checkpoint as checkpoint_module
    t = task()
    config = tmp_path / "main.json"
    config.write_text("{}")
    t.update(main_config=campaign.pinned(config), initial_config=campaign.pinned(config))
    manifest = {"source_root": str(tmp_path / "source"), "record_hash": "campaign", "source_digest": "source",
                "baseline": {"sha256": "baseline"}}
    (tmp_path / "checks").mkdir()
    calls = []
    def execute(cmd):
        calls.append(cmd)
        stop = int(cmd[cmd.index("--stop-after-steps") + 1])
        stage = "main"
        directory = campaign.run_dir(tmp_path, t, stage)
        (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        for step in (0, stop):
            campaign.checkpoint_path(tmp_path, t, stage, step).write_text(str(step))
        (directory / "run_config.json").write_text("{}")
        rows = [{"step": step, "loss": float(step), "gradient_norm": float(step),
                 "end_to_end_seconds": 1.} for step in range(1, stop + 1)]
        (directory / "train_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(campaign, "call", execute)
    monkeypatch.setattr(campaign, "numeric_initial_identity", lambda path: {"residual_params": "same"})
    monkeypatch.setattr(campaign, "verify_training_checkpoint", lambda *args: None)
    monkeypatch.setattr(checkpoint_module, "load_v24_Ilya_training_checkpoint",
                        lambda path: SimpleNamespace(completed_step=int(path.read_text()), model_family="gc_mamba"))
    return t, manifest, calls


def test_finite_increasing_loss_completes_and_split_proof_retains_total_counter(tmp_path, monkeypatch):
    t, manifest, calls = fake_stage(tmp_path, monkeypatch)
    record = campaign.run_stage(tmp_path, manifest, t, "main", 750, 1250)
    assert record["completed_steps"] == 750 and record["total_training_updates"] == 1250
    assert record["scientific_improvement_required"] is False
    resumed = campaign.run_stage(tmp_path, manifest, t, "main", 2000, 2500, previous=record)
    assert resumed["completed_steps"] == 2000 and resumed["started_from_step"] == 750
    assert resumed["total_training_updates"] == 2500
    assert "--resume" in calls[-1] and "--zero-state-on-init" not in calls[-1]


def test_modified_training_history_cannot_silently_resume(tmp_path, monkeypatch):
    t, manifest, calls = fake_stage(tmp_path, monkeypatch)
    record = campaign.run_stage(tmp_path, manifest, t, "main", 750, 1250)
    with (campaign.run_dir(tmp_path, t, "main") / "train_metrics.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Live training history"):
        campaign.run_stage(tmp_path, manifest, t, "main", 2000, 2500, previous=record)
    assert len(calls) == 1


def test_checkpoint_verification_rejects_wrong_config_or_nonfinite_saved_weights(tmp_path, monkeypatch):
    import numpy as np
    from src.models.mamba.v24_Ilya import checkpoint as checkpoint_module
    from src.models.mamba.v24_Ilya.training import config as config_module
    cfg = {"sequence": {"feedback_mode": "closed_loop_sg"}}
    saved = SimpleNamespace(completed_step=750, model_family="gc_mamba",
        baseline_checkpoint_fingerprint="baseline", resolved_training_config=copy.deepcopy(cfg),
        residual_params={"x": np.array([1.])}, residual_state={}, optimizer_state={})
    monkeypatch.setattr(checkpoint_module, "load_v24_Ilya_training_checkpoint", lambda path: saved)
    monkeypatch.setattr(config_module, "load_training_config", lambda path: SimpleNamespace(to_dict=lambda: cfg))
    campaign.verify_training_checkpoint(tmp_path / "checkpoint", tmp_path / "cfg", "baseline", 750)
    saved.resolved_training_config = {"different": True}
    with pytest.raises(ValueError, match="training configuration"):
        campaign.verify_training_checkpoint(tmp_path / "checkpoint", tmp_path / "cfg", "baseline", 750)
    saved.resolved_training_config = cfg
    saved.residual_params["x"][0] = np.inf
    with pytest.raises(ValueError, match="Nonfinite saved"):
        campaign.verify_training_checkpoint(tmp_path / "checkpoint", tmp_path / "cfg", "baseline", 750)
