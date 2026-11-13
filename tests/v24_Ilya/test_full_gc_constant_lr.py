"""Five fresh constant-LR trajectories and optimizer persistence across 1250."""
import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scripts.experiments import run_v24_full_gc_constant_lr as campaign
from src.models.mamba.v24_Ilya.training.config import load_training_config
from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer


def task(weight=0.):
    return {"id": f"gcft_s22_l{round(weight * 100):03d}_lr3em7_constant_20261112",
            "logical_id": f"gcft_s22_l{round(weight * 100):03d}",
            "model_family": "full_gc", "lambda": weight, "learning_rate": 3e-7, "resume_step": 0}


def original_config(weight=0.):
    selected = campaign.unseal(campaign.ROOT / campaign.REFERENCE / "selected_arms_full_gc.json")
    original = next(value for value in selected["tasks"] if value["lambda"] == weight)
    return campaign.read(original["config"]["path"])


@pytest.mark.parametrize("weight", campaign.LAMBDAS)
def test_all_five_configs_change_only_constant_schedule_and_fresh_output(tmp_path, weight):
    original = original_config(weight)
    before = copy.deepcopy(original)
    identity = task(weight)
    config = campaign.make_config(original, tmp_path, identity)
    campaign.validate_config(config, identity)
    restored = copy.deepcopy(config)
    for key in ("learning_rate", "end_learning_rate", "learning_rate_schedule", "warmup_steps"):
        restored["optimizer"][key] = original["optimizer"][key]
    restored["output"] = original["output"]
    assert restored == original == before
    assert config["optimizer"]["max_steps"] == 2500
    assert config["optimizer"]["learning_rate_schedule"] == "constant"
    assert config["optimizer"]["end_learning_rate"] is None
    assert config["optimizer"]["warmup_steps"] == 0
    assert config["data"]["train_order_manifest"] == original["data"]["train_order_manifest"]


def test_actual_optimizer_constant_rate_and_serialized_resume(tmp_path):
    identity = task()
    path = tmp_path / "config.json"
    campaign.write(path, campaign.make_config(original_config(), tmp_path, identity))
    evidence = campaign.validate_optimizer_schedule(path)
    assert evidence["rate_at_update_index"] == {str(step): 3e-7 for step in (0, 99, 100, 1249, 1250, 2499)}
    config = load_training_config(path)
    optimizer, rate = build_optimizer(config)
    assert rate == 3e-7 and not callable(rate)
    params = {"spatial": {"w": jnp.asarray([1., -2.], dtype=jnp.float32)}}
    state = optimizer.init(params)

    def step(carry, index):
        params, state = carry
        gradients = {"spatial": {"w": jnp.asarray([.25, -.5]) * (1. + (index % 7) / 10.)}}
        updates, state = optimizer.update(gradients, state, params)
        return (optax.apply_updates(params, updates), state), updates["spatial"]["w"]

    run = jax.jit(lambda carry, indices: jax.lax.scan(step, carry, indices))
    midpoint, _ = run((params, state), jnp.arange(1250))
    restored = pickle.loads(pickle.dumps(midpoint))
    uninterrupted, original_updates = run(midpoint, jnp.arange(1250, 2500))
    resumed, resumed_updates = run(restored, jnp.arange(1250, 2500))
    np.testing.assert_array_equal(original_updates, resumed_updates)
    for actual, expected in zip(jax.tree_util.tree_leaves(resumed), jax.tree_util.tree_leaves(uninterrupted)):
        np.testing.assert_array_equal(actual, expected)


def test_every_first_allocation_is_fresh_and_second_exactly_resumes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign.scalar, "run_training_stage", lambda *args, **kwargs: calls.append((args[2], kwargs)))
    manifest = {"tasks": [task(weight) for weight in campaign.LAMBDAS]}
    for index in range(5):
        campaign.train(tmp_path, manifest, index, 0)
        campaign.train(tmp_path, manifest, index, 1)
    assert len(calls) == 10
    for index in range(5):
        assert calls[2 * index] == (manifest["tasks"][index], {"stop": 1250, "previous": None})
        assert calls[2 * index + 1] == (manifest["tasks"][index], {"stop": 2500, "previous": 1250})


def test_five_arm_mapping_and_dependencies(tmp_path):
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source")}
    plan = campaign.submission_plan(tmp_path, manifest)
    assert [node["id"] for node in plan] == ["train-part1", "train-part2", "history", "finish"]
    for node in plan[:3]:
        assert "--array=0-4" in node["command"]
    assert "--time=1800" in plan[0]["command"] and "--mem=384G" in plan[0]["command"]
    assert "--dependency=aftercorr:<train-part1>" in plan[1]["command"]
    assert "--dependency=aftercorr:<train-part2>" in plan[2]["command"]
    assert "--dependency=afterok:<train-part2>" in plan[3]["command"]
    assert not any("%" in part for node in plan for part in node["command"] if part.startswith("--array"))


def test_decay_warmup_or_inherited_weights_cannot_enter_constant_campaign(tmp_path):
    identity = task()
    for change in ({"learning_rate_schedule": "cosine"}, {"warmup_steps": 100}, {"end_learning_rate": 3e-8}):
        config = campaign.make_config(original_config(), tmp_path, identity)
        config["optimizer"].update(change)
        with pytest.raises(ValueError):
            campaign.validate_config(config, identity)
    config = campaign.make_config(original_config(), tmp_path, identity)
    config["init_from"] = "/archive/cosine_step1250.pkl"
    with pytest.raises(ValueError, match="inherit"):
        campaign.validate_config(config, identity)
