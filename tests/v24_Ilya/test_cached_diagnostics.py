from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.models.mamba.v24_Ilya.training.cached_diagnostics import (
    capture_gradients, gradient_comparison, timing_summary, tree_comparison, tree_digest,
)


def test_comparison_requires_direction_and_magnitude_agreement():
    expected = {"layer": {"w": np.array([0.2, 0.5], np.float32)}}
    assert gradient_comparison(expected, expected)["passed"]
    scaled = {"layer": {"w": expected["layer"]["w"] * 2}}
    assert not gradient_comparison(scaled, expected)["passed"]
    opposite = {"layer": {"w": -expected["layer"]["w"]}}
    assert not gradient_comparison(opposite, expected)["passed"]


def test_zero_tolerance_and_nonfinite_checks():
    assert tree_comparison(np.array([1e-6]), np.zeros(1))["passed"]
    assert not tree_comparison(np.array([1e-4]), np.zeros(1))["passed"]
    assert not tree_comparison(np.array([np.nan]), np.zeros(1))["passed"]
    assert not tree_comparison({"a": np.zeros(1)}, {"b": np.zeros(1)})["passed"]


def test_capture_optimizer_preserves_updates_and_exposes_unclipped_gradients():
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adamw(1e-4, b2=0.98))
    captured = capture_gradients(optimizer)
    params = {"w": jnp.array([1., 2.])}
    state, diagnostic_state = optimizer.init(params), captured.init(params)
    for grads in ({"w": jnp.array([30., 40.])}, {"w": jnp.array([2., -3.])}):
        update, state = jax.jit(optimizer.update)(grads, state, params)
        actual, diagnostic_state = jax.jit(captured.update)(grads, diagnostic_state, params)
        assert tree_digest(actual) == tree_digest(update)
        assert tree_digest(diagnostic_state[0]) == tree_digest(state)
        assert tree_digest(diagnostic_state[1]) == tree_digest(grads)
        params = optax.apply_updates(params, update)


def test_benchmark_excludes_twenty_warmup_updates():
    records = [{"compute_seconds": 100 if i < 20 else 2,
                "end_to_end_seconds": 200 if i < 20 else 3} for i in range(70)]
    result = timing_summary(records)
    assert result["compute_median_seconds"] == 2
    assert result["end_to_end_p90_seconds"] == 3
    assert result["valid_weather_steps_per_second"] == 8
    assert timing_summary(records, weather_steps=6)["valid_weather_steps_per_second"] == 2


@pytest.mark.parametrize("policy,copied", [("fresh", 0), ("baseline_overlay", 1)])
def test_initialization_gate_matches_requested_spatial_policy(policy, copied):
    from types import SimpleNamespace
    from src.models.mamba.v24_Ilya.training.cached_parity import _initialization_checks

    params = {"spatial": {"w": np.ones(1)},
              "temporal_residual_head": {"w": np.zeros(1)},
              "temporal/out_proj": {"w": np.zeros(1)}}
    config = SimpleNamespace(architecture=SimpleNamespace(residual_initialization=policy))
    metadata = {"residual_initialization": policy, "residual_copied": copied,
                "residual_fresh": 3 - copied, "baseline_copied": 1, "baseline_fresh": 0}
    source = SimpleNamespace(residual_params=params, residual_state={},
                             parameter_overlay_metadata=metadata)
    assert _initialization_checks(source, config)["passed"]
    metadata["residual_initialization"] = "baseline_overlay" if policy == "fresh" else "fresh"
    assert not _initialization_checks(source, config)["passed"]
    metadata["residual_initialization"] = policy
    metadata["residual_fresh"] += 1
    assert not _initialization_checks(source, config)["passed"]
    metadata["residual_fresh"] -= 1
    source.residual_state = {"memory": np.ones(1)}
    assert not _initialization_checks(source, config)["passed"]
    source.residual_state = {}
    params["temporal/out_proj"]["w"] = np.ones(1)
    assert not _initialization_checks(source, config)["passed"]


@pytest.mark.parametrize("reset", [False, True])
def test_online_diagnostic_forward_obeys_prediction_reset_policy(reset):
    from types import SimpleNamespace
    from graphcast import xarray_jax
    from src.models.mamba.v24_Ilya.training.cached_parity import _diagnostic_forward

    def apply(params, state, key, inputs, target, forcing):
        del params, key, target, forcing
        prediction = state["memory"] + inputs
        return ((xarray_jax.DataArray(jnp.mean(prediction ** 2)), {}), prediction), {"memory": prediction}

    config = SimpleNamespace(temporal_state_policy="reset_every_anchor" if reset else "carry")
    transforms = SimpleNamespace(residual_loss_and_predictions=SimpleNamespace(apply=apply))
    forward = _diagnostic_forward(transforms, config)
    state = {"memory": jnp.array([7.])}
    predictions = []
    for _ in range(2):
        _, prediction, state = forward({}, state, jax.random.PRNGKey(0), jnp.ones(1), None, None)
        predictions.append(float(prediction[0]))
    assert predictions == ([1., 1.] if reset else [8., 9.])
    assert float(state["memory"][0]) == (0. if reset else 9.)


def test_resource_validation_coverage_uses_cache_and_rejects_partial_results(tmp_path):
    import json
    from pathlib import Path
    from src.models.mamba.v24_Ilya.training.cached_parity import _validation_coverage

    config_path = Path("configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json").resolve()
    cache = tmp_path / "cache"
    cache.mkdir()
    chunks = [{"split": "val", "segment_id": segment_id, "segment_offset": offset}
              for segment_id in range(2) for offset in (0, 24, 48, 72)]
    (cache / "manifest.json").write_text(json.dumps({"chunks": chunks}))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "configs": {"cached": str(config_path)}, "cache_root": str(cache)}))
    validation = {"selected_segments": 2, "available_segments": 2, "segment_ids": [0, 1],
                  "num_chunks": 8, "num_anchors": 192, "bptt_steps": 24}
    assert _validation_coverage(tmp_path, validation)["passed"]
    assert not _validation_coverage(tmp_path, {**validation, "num_chunks": 7})["passed"]
    assert not _validation_coverage(tmp_path, {**validation, "segment_ids": [0, 2]})["passed"]
    (cache / "manifest.json").write_text(json.dumps({"chunks": chunks[:-1]}))
    assert not _validation_coverage(tmp_path, validation)["passed"]


def test_exact_checkpoint_comparison_still_rejects_identical_nonfinite_values(monkeypatch):
    from types import SimpleNamespace
    from src.models.mamba.v24_Ilya import checkpoint
    from src.models.mamba.v24_Ilya.training.cached_parity import _checkpoint_compare

    source = SimpleNamespace(residual_params={"w": np.array([np.nan])}, residual_state={},
                             optimizer_state={}, rng_key=np.zeros(2, np.uint32), training_cursor={})
    monkeypatch.setattr(checkpoint, "load_v24_Ilya_training_checkpoint", lambda _: source)
    result = _checkpoint_compare("left", "right", exact=True)
    assert result["checks"]["residual_params"]["exact_hash_match"]
    assert not result["passed"]


def test_worker_launch_uses_importable_module_when_parent_is_main(tmp_path, monkeypatch):
    import subprocess
    from src.models.mamba.v24_Ilya.training import cached_parity

    monkeypatch.setattr(cached_parity, "__name__", "__main__")
    command = cached_parity._worker_command(tmp_path, worker="preflight", backend="online")
    assert command[2] == "src.models.mamba.v24_Ilya.training.cached_parity"
    # Exercise the actual Python module launcher, without constructing a model.
    result = subprocess.run([*command, "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--worker" in result.stdout
