"""AMSE split mathematics, two-gradient routing and calibration provenance."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from graphcast import xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.amse_split_loss import (
    STAT_FILES, build_amse_split_objectives, calibration_fingerprint,
    combine_amse_split_calibrations, make_amse_split_calibration_record,
    validate_amse_split_calibration,
)
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig, load_training_config
from src.models.mamba.v24_Ilya.training.data import TrainingCursor
from src.models.mamba.v24_Ilya.training.mgda_step import make_multiobjective_train_step
from src.models.mamba.v24_Ilya.training.spectral_loss import amse_per_degree, build_full_field_loss
from src.models.mamba.v24_Ilya.training import runner


def config(width=16, root=Path("split_test")):
    return V24IlyaTrainConfig(
        prepared_root=root / "prepared", anchor_manifest_root=root / "anchors",
        baseline_checkpoint=root / "baseline", output_root=root / "runs", stats_dir=root / "stats",
        run_name=f"di{width}", architecture=V24IlyaArchitectureConfig(temporal_d_inner=width),
        objective_pair="amse_split", loss_kind="amse", amse_lmax=179,
        feedback_mode="closed_loop_sg", loss_mode="all_steps", segment_steps=8, bptt_steps=4,
        ar_tail_k=2, seed=22, max_steps=2000, learning_rate=1e-5, end_learning_rate=3e-6,
        warmup_steps=100, decay_end_step=1000, learning_rate_schedule="warmup_cosine_constant",
        weight_decay=0, mamba_lr_multiplier=3.)


def record(cfg, values=((1., 3.), (2., 4.))):
    return make_amse_split_calibration_record(values,
        [{"cursor": {"segment_index": 0, "segment_offset": i * 4}, "anchor_indices": [i]}
         for i in range(len(values))],
        init_from_sha256=("1" if cfg.architecture.temporal_d_inner == 16 else "2") * 64,
        source_config_sha256="3" * 64, source_config=cfg.to_dict(),
        baseline_checkpoint_fingerprint="4" * 64, anchor_manifest_fingerprint="5" * 64,
        prepared_metadata_sha256="6" * 64, stats_sha256={name: "7" * 64 for name in STAT_FILES})


def pooled(root=Path("split_test")):
    return combine_amse_split_calibrations([record(config(16, root)),
                                            record(config(32, root), ((2., 6.), (4., 8.)))])


def test_shared_scale_matches_raw_evidence_and_preserves_both_initializations():
    value = pooled()
    assert value["common_scale"] == 7.5
    assert [r["init_from_sha256"] for r in value["initializations"]] == ["1" * 64, "2" * 64]
    assert validate_amse_split_calibration(value) == value
    assert calibration_fingerprint(value) == calibration_fingerprint(json.loads(json.dumps(value)))
    corrupted = deepcopy(value); corrupted["common_scale"] *= 2
    with pytest.raises(ValueError, match="pooled raw evidence"):
        validate_amse_split_calibration(corrupted)


@pytest.mark.parametrize("mutation", ["batch_manifest", "stats", "baseline", "sequence", "same_hash"])
def test_pool_rejects_mismatched_forward_evidence(mutation):
    records = [record(config(16)), record(config(32))]
    other = records[1]
    if mutation == "batch_manifest":
        other["batch_manifest"][0]["anchor_indices"] = [100]
    elif mutation == "stats":
        other["stats_sha256"][STAT_FILES[0]] = "8" * 64
    elif mutation == "baseline":
        other["baseline_checkpoint_fingerprint"] = "8" * 64
    elif mutation == "sequence":
        other["source_config"]["sequence"]["ar_tail_k"] = 1
    else:
        other["init_from_sha256"] = records[0]["init_from_sha256"]
    with pytest.raises(ValueError):
        combine_amse_split_calibrations(records)


def test_config_serialization_requires_shared_scale_and_preserves_old_schema(tmp_path):
    bootstrap = config(root=tmp_path)
    with pytest.raises(ValueError, match="frozen shared calibration"):
        replace(bootstrap, optimizer_kind="mgda_rms")
    active = replace(bootstrap, optimizer_kind="mgda_rms", amse_split_calibration=pooled(tmp_path))
    path = tmp_path / "config.json"; path.write_text(json.dumps(active.to_dict()))
    assert load_training_config(path).to_dict() == active.to_dict()
    with pytest.raises(ValueError, match="unmodified lambda=1"):
        replace(bootstrap, amse_lambda=.25)
    with pytest.raises(ValueError, match="requires objective_pair"):
        replace(active, objective_pair=None)
    old = replace(bootstrap, objective_pair=None)
    assert "objective_pair" not in old.to_dict()["objective"]
    assert "amse_split_calibration" not in old.to_dict()["objective"]


def test_unknown_initialization_is_rejected_before_training():
    active = replace(config(), optimizer_kind="mgda_rms", amse_split_calibration=pooled())
    with pytest.raises(ValueError, match="init_from_sha256"):
        runner._validate_calibration_provenance(active, "4" * 64, "5" * 64,
                                               {"init_from_sha256": "8" * 64})


def test_bootstrap_production_and_resume_preserve_calibrated_initialization(tmp_path):
    bootstrap = config(root=tmp_path)
    bootstrap.prepared_root.mkdir()
    bootstrap.stats_dir.mkdir()
    metadata = bootstrap.prepared_root / "metadata.json"
    metadata.write_text('{"schema": 1}')
    for name in STAT_FILES:
        (bootstrap.stats_dir / name).write_text(name)
    records = [record(config(width, tmp_path)) for width in (16, 32)]
    for item in records:
        item["prepared_metadata_sha256"] = runner.file_sha256(metadata)
        item["stats_sha256"] = {name: runner.file_sha256(bootstrap.stats_dir / name) for name in STAT_FILES}
    calibration = combine_amse_split_calibrations(records)
    active = replace(bootstrap, optimizer_kind="mgda_rms", amse_split_calibration=calibration)
    # Resume copies the original SWA identity from parameter_overlay_metadata;
    # it must not reinterpret the checkpoint100 hash as the calibration source.
    overlay = {"init_from_sha256": "1" * 64, "init_from": "original_swa.pkl"}
    runner._validate_calibration_provenance(active, "4" * 64, "5" * 64, overlay)
    restored = json.loads(json.dumps(overlay))
    runner._validate_calibration_provenance(active, "4" * 64, "5" * 64, restored)
    wrong_width = replace(active, architecture=config(32, tmp_path).architecture)
    with pytest.raises(ValueError, match="architecture differs"):
        runner._validate_calibration_provenance(wrong_width, "4" * 64, "5" * 64, overlay)
    (bootstrap.stats_dir / STAT_FILES[0]).write_text("changed")
    with pytest.raises(ValueError, match="statistics differ"):
        runner._validate_calibration_provenance(active, "4" * 64, "5" * 64, overlay)


def fields83():
    lat, lon, levels = np.linspace(-90, 90, 7), np.arange(12) * 30., np.arange(1, 14) * 75
    rng = np.random.default_rng(23)
    variables, scales = {}, {}
    surface = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
               "mean_sea_level_pressure", "total_precipitation_6hr")
    upper = ("temperature", "u_component_of_wind", "v_component_of_wind",
             "specific_humidity", "geopotential", "vertical_velocity")
    for i, name in enumerate((*surface, *upper)):
        is_upper = name in upper
        dims = ("batch", "time", "level", "lat", "lon") if is_upper else ("batch", "time", "lat", "lon")
        shape = (2, 1, 13, 7, 12) if is_upper else (2, 1, 7, 12)
        variables[name] = (dims, jnp.asarray(rng.normal(size=shape), jnp.float32))
        scales[name] = (("level", np.arange(1, 14, dtype=np.float32) / 13 + .5)
                        if is_upper else ((), np.float32(i + 1)))
    data = xr.Dataset(variables, coords={"batch": [0, 1], "time": [0], "level": levels,
                                        "lat": lat, "lon": lon})
    return data, xr.Dataset(scales, coords={"level": levels})


def test_full_83_channel_split_sum_and_gradients_match_existing_amse():
    truth, stats = fields83()
    amse = build_full_field_loss(stats, "amse", lmax=5)
    gc = build_full_field_loss(stats, "mse_fp32")
    pair = build_amse_split_objectives(amse)
    prediction = truth * .73 + .1
    values, diagnostics = pair(prediction, truth)
    np.testing.assert_allclose(values.sum(axis=0), amse(prediction, truth)[0], rtol=2e-6)
    np.testing.assert_allclose(diagnostics["gc_loss"], gc(prediction, truth)[0], rtol=1e-6)
    direct = jax.grad(lambda scale: amse(truth * scale + .1, truth)[0].sum())(.73)
    components = jax.jacrev(lambda scale: pair(truth * scale + .1, truth)[0].sum(axis=1))(.73)
    np.testing.assert_allclose(components.sum(), direct, rtol=3e-5)
    # Upper-air vertical velocity is outside the old nine-field power objective.
    changed = prediction.copy(); changed["vertical_velocity"] = truth["vertical_velocity"] * 10
    assert np.max(np.abs(np.asarray(pair(changed, truth)[0] - values))) > 1
    zero = truth * 0
    np.testing.assert_allclose(pair(zero, zero)[0], 0, atol=1e-10)
    assert np.all(np.isfinite(jax.jacrev(lambda scale: pair(truth * scale, zero)[0].sum(axis=1))(0.)))


def toy_pair(prediction, truth):
    f, y = (xarray_jax.unwrap_data(item["x"])[:, 0, :] for item in (prediction, truth))
    _, terms = amse_per_degree(f[:, None, :], y[:, None, :])
    return jnp.stack((terms["amplitude"][:, 0], terms["pattern"][:, 0])), {
        "gc_loss": jnp.mean((f - y) ** 2, axis=-1)}


def toy_setup():
    # Reuse the independently checked stochastic/recurrent trajectory fixture.
    from test_mgda_step import setup
    cfg, transforms, optimizer, params, state, args, _ = setup()
    cfg.objective_pair, cfg.loss_kind, cfg.amse_split_calibration = "amse_split", "amse", {"common_scale": 3.}
    transforms = replace(transforms, full_field_objectives=toy_pair)
    step = make_multiobjective_train_step(transforms=transforms, optimizer=optimizer,
        baseline_params={}, baseline_state={}, config=cfg, time_step=pd.Timedelta("6h"), input_steps=2)
    return cfg, transforms, optimizer, params, state, args, step


def test_dual_reverse_uses_common_scale_and_retains_exact_gc_diagnostic():
    from test_mgda_step import assert_tree_close
    cfg, transforms, optimizer, params, state, args, step = toy_setup()
    def direct(candidate):
        keys, initial_frames, _, truths, _ = args
        frames = [xarray_jax.unwrap_data(frame["x"])[:, 0] for frame in initial_frames]
        carried = state["s"]
        values, gc = [], []
        for i in range(4):
            mean = jnp.mean(jnp.stack((frames[i], frames[i + 1])), axis=(0, 2))
            carried = .7 * carried + candidate["a"] * mean + .01 * jax.random.normal(keys[i], mean.shape)
            f = .5 + candidate["b"] * carried[:, None] * jnp.ones((4, 2))
            y = xarray_jax.unwrap_data(truths[i]["x"])[:, 0]
            _, terms = amse_per_degree(f[:, None, :], y[:, None, :])
            values.append(jnp.stack((terms["amplitude"].mean(), terms["pattern"].mean())))
            gc.append(jnp.mean((f - y) ** 2))
            if 1 <= i < 3:
                frames.append(jax.lax.stop_gradient(f))
        return jnp.mean(jnp.stack(values), axis=0), jnp.mean(jnp.stack(gc)), carried
    expected = tuple(jax.grad(lambda p: direct(p)[0][i])(params) for i in range(2))
    assert_tree_close(step.objective_gradients(params, state, args), expected)
    opt_state = optimizer.init(params)
    result = step(params, state, opt_state, *args)
    import optax
    normalized = jax.tree_util.tree_map(lambda x: x / 3, expected)
    updates, next_opt, _ = optimizer.update_pair(*normalized, opt_state, params)
    assert_tree_close(result.params, optax.apply_updates(params, updates))
    assert_tree_close(result.optimizer_state, next_opt)
    values, exact_gc, carried = direct(params)
    np.testing.assert_allclose(result.state["s"], carried, rtol=3e-6)
    assert float(result.loss) == pytest.approx(float(exact_gc))
    assert result.diagnostics["gc_loss"] == pytest.approx(float(exact_gc))
    assert result.diagnostics["amplitude_loss"] == pytest.approx(float(values[0]))
    assert result.diagnostics["pattern_loss"] == pytest.approx(float(values[1]))
    assert "power_loss" not in result.diagnostics and "gc_gradient_norm" not in result.diagnostics
    assert "amplitude_displacement_dot" in result.diagnostics
    assert result.diagnostics["finite"] and result.diagnostics["descent_valid"]


def test_split_calibration_is_single_forward_pass_without_parameter_updates():
    from test_mgda_step import assert_tree_close
    cfg, transforms, _, params, state, args, _ = toy_setup()
    calls = []
    def chunk(cursor, *_):
        calls.append(cursor)
        return SimpleNamespace(input_frames=args[1], static_inputs=args[2], truths=args[3],
            forcings=args[4], raw_anchor_indices=np.array([cursor.segment_index * 8 + cursor.segment_offset]),
            next_cursor=(TrainingCursor(segment_index=cursor.segment_index + 1) if cursor.segment_offset else
                         TrainingCursor(segment_index=cursor.segment_index, segment_offset=4)))
    data = SimpleNamespace(build_chunk=chunk, input_steps=2, time_step=pd.Timedelta("6h"))
    before = jax.tree_util.tree_map(np.asarray, (params, state))
    evidence = record(config())
    provenance = {key: value for key, value in evidence.items()
                  if key not in ("version", "kind", "batches", "raw_values", "batch_manifest")}
    first = runner._calibrate_initialized_training(config=cfg, transforms=transforms, training_data=data,
        task_config=None, params=params, zero_state=state, rng_key=jax.random.PRNGKey(19),
        baseline_params={}, baseline_state={}, batches=4, provenance=provenance)
    assert len(calls) == 4
    assert [cursor.segment_offset for cursor in calls] == [0, 4, 0, 4]
    assert np.asarray(first["raw_values"]).shape == (4, 2)
    assert_tree_close((params, state), before)
