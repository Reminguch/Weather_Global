from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from src.models.mamba.v24_Ilya.checkpoint import (
    SWA_CHECKPOINT_FORMAT,
    build_v24_Ilya_swa,
    data_parallel_training_checkpoint_payload,
    load_v24_Ilya_checkpoint,
    load_v24_Ilya_training_checkpoint,
    overlay_frozen_baseline_params,
)
from src.models.mamba.v24_Ilya.config import (
    ARCHITECTURE_ID,
    V24IlyaArchitectureConfig,
)
from src.models.mamba.v24_Ilya.training.config import (
    BPTT_BACKEND,
    V24IlyaDistributedConfig,
    V24IlyaTrainConfig,
    load_training_config,
    parse_cli,
)
from src.models.mamba.v24_Ilya.training.endpoint_step import (
    mamba_standard_weight_decay_mask,
)


def _config() -> V24IlyaTrainConfig:
    return V24IlyaTrainConfig(
        prepared_root=Path("prepared"),
        anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline.npz"),
        output_root=Path("output"),
        run_name="endpoint",
        architecture=V24IlyaArchitectureConfig(),
        segment_steps=24,
        bptt_steps=24,
        ar_tail_k=20,
    )


def test_endpoint_defaults_and_roundtrip(tmp_path: Path) -> None:
    config = _config()
    assert config.loss_mode == "last_step"
    assert config.weather_tape_precision == "fp32"
    assert config.architecture.temporal_init_scheme == "legacy_haiku"
    assert config.weight_decay_policy == "all"
    assert config.truth_prefix_steps == 4
    payload = config.to_dict()
    assert payload["objective"] == {"loss_mode": "last_step"}
    assert payload["memory"] == {
        "weather_tape_precision": "fp32",
        "bptt_backend": BPTT_BACKEND,
    }
    assert payload["distributed"] == {
        "mode": "single",
        "num_devices": 1,
        "per_device_batch_size": 1,
        "drop_incomplete_replica_group": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    assert load_training_config(path) == config


def test_mamba1_initialization_and_decay_policy_roundtrip(tmp_path: Path) -> None:
    architecture = V24IlyaArchitectureConfig(
        temporal_d_inner=16,
        temporal_init_scheme="mamba1",
        temporal_dt_init="random",
        temporal_dt_min=0.001,
        temporal_dt_max=0.1,
        temporal_dt_scale=1.0,
        temporal_dt_init_floor=1e-4,
        temporal_zero_init_out=True,
    )
    config = V24IlyaTrainConfig(
        **{
            **_config().__dict__,
            "architecture": architecture,
            "weight_decay_policy": "mamba_standard",
        }
    )
    path = tmp_path / "mamba1.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_training_config(path) == config


def test_old_config_without_initialization_fields_keeps_legacy_behavior(
    tmp_path: Path,
) -> None:
    payload = _config().to_dict()
    for name in (
        "temporal_init_scheme",
        "temporal_dt_init",
        "temporal_dt_min",
        "temporal_dt_max",
        "temporal_dt_scale",
        "temporal_dt_init_floor",
    ):
        payload["architecture"].pop(name)
    payload["optimizer"].pop("weight_decay_policy")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))

    loaded = load_training_config(path)
    assert loaded.architecture.temporal_init_scheme == "legacy_haiku"
    assert loaded.weight_decay_policy == "all"


def test_mamba_standard_weight_decay_mask() -> None:
    params = {
        "mamba/layer_norm_0": {
            "scale": np.ones((4,), np.float32),
            "offset": np.zeros((4,), np.float32),
        },
        "mamba/block": {
            "A_log": np.ones((4, 3), np.float32),
            "D": np.ones((4,), np.float32),
            "w": np.ones((4, 4), np.float32),
            "b": np.zeros((4,), np.float32),
        },
    }
    mask = mamba_standard_weight_decay_mask(params)
    assert mask["mamba/layer_norm_0"] == {"scale": False, "offset": False}
    assert mask["mamba/block"] == {
        "A_log": False,
        "D": False,
        "w": True,
        "b": False,
    }


def test_baseline_validation_only_cli_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config().to_dict()))
    invocation = parse_cli(
        ["--config", str(path), "--baseline-validation-only"]
    )
    assert invocation.baseline_validation_only is True
    assert invocation.resume is None
    assert invocation.init_from is None


def test_validation_compare_cli_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config().to_dict()))
    checkpoint = tmp_path / "checkpoint.pkl"
    invocation = parse_cli(
        ["--config", str(path), "--validation-compare", str(checkpoint)]
    )
    assert invocation.validation_compare == checkpoint
    assert invocation.baseline_validation_only is False


def test_frozen_baseline_overlay_accepts_and_records_source_only_head() -> None:
    initialized = {
        "encoder": {"w": np.zeros((2, 3), np.float32)},
        "decoder/grid": {"b": np.zeros((4,), np.float32)},
    }
    source = {
        "encoder": {"w": np.ones((2, 3), np.float32)},
        "decoder/grid": {"b": np.ones((4,), np.float32)},
        "decoder/mesh": {
            "w": np.ones((3, 4), np.float32),
            "b": np.ones((4,), np.float32),
        },
    }

    loaded, stats = overlay_frozen_baseline_params(initialized, source)

    assert set(loaded) == set(initialized)
    assert np.array_equal(loaded["encoder"]["w"], source["encoder"]["w"])
    assert np.array_equal(loaded["decoder/grid"]["b"], source["decoder/grid"]["b"])
    assert stats.copied == 2
    assert stats.ignored_source == (
        "decoder/mesh/b",
        "decoder/mesh/w",
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "missing_modules"),
        ({"module": {}}, "missing_params"),
        (
            {"module": {"w": np.zeros((3,), np.float32)}},
            "mismatched_shapes",
        ),
    ],
)
def test_frozen_baseline_overlay_rejects_incomplete_target_coverage(
    source: dict,
    message: str,
) -> None:
    initialized = {"module": {"w": np.zeros((2,), np.float32)}}
    with pytest.raises(ValueError, match=message):
        overlay_frozen_baseline_params(initialized, source)


@pytest.mark.parametrize("loss_mode", ["last_step", "all_steps"])
def test_supported_objective_and_fp32_tape(loss_mode: str) -> None:
    config = V24IlyaTrainConfig(
        **{
            **_config().__dict__,
            "loss_mode": loss_mode,
            "weather_tape_precision": "fp32",
        }
    )
    assert config.loss_mode == loss_mode
    assert config.weather_tape_precision == "fp32"


def test_res0p25_full24_experiment_isolated_from_current_recipe() -> None:
    candidate = load_training_config(
        Path(
            "configs/experiments/v24_Ilya/"
            "res0p25_v22compat_all24_legacy_di16_bcg1_fp32_sg500.json"
        )
    )
    control = load_training_config(
        Path(
            "configs/experiments/v24_Ilya/"
            "res0p25_sparse_uniform_legacy_di16_bcg1_fp32_sg500.json"
        )
    )

    assert candidate.loss_mode == "all_steps"
    assert candidate.ar_tail_k == 20
    assert candidate.supervised_step_indices == tuple(range(24))
    assert candidate.normalized_supervised_weights == (1.0 / 24.0,) * 24
    assert candidate.max_steps == 500
    assert candidate.checkpoint_every == 50
    assert candidate.architecture.temporal_zero_init_out is False

    candidate_payload = candidate.to_dict()
    control_payload = control.to_dict()
    for payload in (candidate_payload, control_payload):
        payload.pop("objective")
        payload.pop("output")
        payload["sequence"].pop("ar_tail_k")
    assert candidate_payload == control_payload


def test_raw_bf16_weather_tape_is_rejected() -> None:
    with pytest.raises(ValueError, match="raw physical BF16 weather trajectories"):
        V24IlyaTrainConfig(
            **{
                **_config().__dict__,
                "weather_tape_precision": "bf16",
            }
        )


def test_sparse_objective_roundtrip_and_exact_index_mapping(tmp_path: Path) -> None:
    config = V24IlyaTrainConfig(
        **{
            **_config().__dict__,
            "ar_tail_k": 19,
            "loss_mode": "sparse_steps",
            "supervised_horizons": (1, 4, 8, 12, 16, 20),
            "supervised_weights": (1, 1, 2, 2, 4, 8),
        }
    )
    assert config.truth_prefix_steps == 5
    assert config.supervised_step_indices == (4, 7, 11, 15, 19, 23)
    np.testing.assert_allclose(
        config.normalized_supervised_weights,
        np.asarray([1, 1, 2, 2, 4, 8]) / 18.0,
    )
    path = tmp_path / "sparse.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_training_config(path) == config


@pytest.mark.parametrize(
    ("horizons", "weights", "message"),
    [
        ((1, 20), (1,), "matching lengths"),
        ((4, 1, 20), (1, 1, 1), "ordered and unique"),
        ((1, 1, 20), (1, 1, 1), "ordered and unique"),
        ((1, 20), (1, 0), "positive finite"),
        ((1, 20), (1, float("nan")), "positive finite"),
        ((1, 16), (1, 1), "final AR endpoint"),
    ],
)
def test_sparse_objective_rejects_malformed_values(
    horizons: tuple[int, ...],
    weights: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        V24IlyaTrainConfig(
            **{
                **_config().__dict__,
                "ar_tail_k": 19,
                "loss_mode": "sparse_steps",
                "supervised_horizons": horizons,
                "supervised_weights": weights,
            }
        )


def test_existing_modes_reject_sparse_only_fields() -> None:
    with pytest.raises(ValueError, match="only valid"):
        V24IlyaTrainConfig(
            **{
                **_config().__dict__,
                "supervised_horizons": (21,),
                "supervised_weights": (1,),
            }
        )


def test_config_rejects_unknown_keys_and_backend(tmp_path: Path) -> None:
    payload = _config().to_dict()
    payload["objective"]["unexpected"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown keys in objective"):
        load_training_config(path)

    payload = _config().to_dict()
    payload["memory"]["bptt_backend"] = "ordinary_scan"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bptt_backend"):
        load_training_config(path)


def test_data_parallel_config_is_strict_and_roundtrips(tmp_path: Path) -> None:
    distributed = V24IlyaDistributedConfig(
        mode="data_parallel",
        num_devices=4,
        per_device_batch_size=1,
        drop_incomplete_replica_group=True,
    )
    config = V24IlyaTrainConfig(
        **{**_config().__dict__, "distributed": distributed}
    )
    path = tmp_path / "distributed.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_training_config(path) == config
    assert config.distributed.global_batch_size == 4

    payload = config.to_dict()
    payload["distributed"]["unexpected"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown keys in distributed"):
        load_training_config(path)


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "unknown"},
        {"mode": "single", "num_devices": 4},
        {"mode": "data_parallel", "num_devices": 1},
        {"mode": "data_parallel", "num_devices": 4, "per_device_batch_size": 2},
        {
            "mode": "data_parallel",
            "num_devices": 4,
            "drop_incomplete_replica_group": False,
        },
    ],
)
def test_distributed_config_rejects_invalid_values(values: dict) -> None:
    with pytest.raises(ValueError, match="distributed|per_device|drop_incomplete"):
        V24IlyaDistributedConfig(**values)


@pytest.mark.parametrize(
    ("architecture_id", "checkpoint_format"),
    [
        ("v23_Ilya", "v23_Ilya_training_pickle"),
        ("v22_final", "v22_final_training_pickle"),
    ],
)
def test_v24_checkpoint_parser_rejects_older_payloads(
    tmp_path: Path,
    architecture_id: str,
    checkpoint_format: str,
) -> None:
    path = tmp_path / "legacy.pkl"
    payload = {
        "architecture_id": architecture_id,
        "schema_version": 1,
        "checkpoint_format": checkpoint_format,
        "checkpoint_kind": "training",
        "residual_params": {"module": {"w": np.ones((2,), np.float32)}},
        "residual_state": {"module": {"state": np.zeros((1,), np.float32)}},
    }
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="v23/v22 checkpoints cannot be used"):
        load_v24_Ilya_checkpoint(path)
    with pytest.raises(ValueError, match=f"not a {ARCHITECTURE_ID} checkpoint"):
        load_v24_Ilya_training_checkpoint(path)


def test_v24_checkpoint_parser_rejects_bf16_parameter_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bf16.pkl"
    payload = {
        "architecture_id": ARCHITECTURE_ID,
        "schema_version": 1,
        "checkpoint_format": SWA_CHECKPOINT_FORMAT,
        "checkpoint_kind": "swa",
        "residual_params": {
            "module": {"w": jnp.ones((2,), dtype=jnp.bfloat16)}
        },
        "residual_state": None,
    }
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="must be float32"):
        load_v24_Ilya_checkpoint(path)


def _write_data_parallel_checkpoint(
    path: Path,
    *,
    step: int,
    value: float,
    shape: tuple[int, ...] = (2,),
    config_tag: str = "matched",
    baseline_fingerprint: str = "baseline-fingerprint",
    manifest_fingerprint: str = "manifest-fingerprint",
) -> None:
    payload = data_parallel_training_checkpoint_payload(
        completed_step=step,
        residual_params={"module": {"w": np.full(shape, value, np.float32)}},
        replica_states={
            "module": {"state": np.full((2, 1), value, np.float32)}
        },
        optimizer_state={"count": np.asarray(step, dtype=np.int32)},
        rng_key=np.asarray([0, step], dtype=np.uint32),
        replica_group_cursor={"group_index": step},
        active_segment_ids=(0, 1),
        num_devices=2,
        resolved_training_config={"tag": config_tag, "optimizer": {"max_steps": 400}},
        baseline_checkpoint_path="baseline.npz",
        baseline_checkpoint_fingerprint=baseline_fingerprint,
        anchor_manifest_fingerprint=manifest_fingerprint,
        parameter_overlay_metadata={"copied": 1},
    )
    path.write_bytes(pickle.dumps(payload))


def test_swa_accepts_data_parallel_training_checkpoints(tmp_path: Path) -> None:
    first = tmp_path / "step100.pkl"
    second = tmp_path / "step200.pkl"
    output = tmp_path / "swa.pkl"
    _write_data_parallel_checkpoint(first, step=100, value=1.0)
    _write_data_parallel_checkpoint(second, step=200, value=3.0)

    payload = build_v24_Ilya_swa([first, second], [100, 200], output)

    assert payload["checkpoint_format"] == SWA_CHECKPOINT_FORMAT
    assert payload["swa_source_steps"] == [100, 200]
    assert payload["residual_state"] is None
    np.testing.assert_allclose(payload["residual_params"]["module"]["w"], [2.0, 2.0])
    loaded = load_v24_Ilya_checkpoint(output)
    np.testing.assert_allclose(loaded.residual_params["module"]["w"], [2.0, 2.0])
    assert not loaded.has_residual_state


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"step": 201}, "completed_step=201"),
        ({"shape": (3,)}, "shape/dtype differs"),
        ({"config_tag": "different"}, "resolved_training_config"),
        ({"baseline_fingerprint": "different"}, "baseline_checkpoint_fingerprint"),
        ({"manifest_fingerprint": "different"}, "anchor_manifest_fingerprint"),
    ],
)
def test_swa_rejects_mismatched_data_parallel_inputs(
    tmp_path: Path,
    override: dict,
    message: str,
) -> None:
    first = tmp_path / "step100.pkl"
    second = tmp_path / "step200.pkl"
    _write_data_parallel_checkpoint(first, step=100, value=1.0)
    second_options = {"step": 200, "value": 3.0, **override}
    _write_data_parallel_checkpoint(
        second,
        **second_options,
    )

    with pytest.raises(ValueError, match=message):
        build_v24_Ilya_swa(
            [first, second],
            [100, 200],
            tmp_path / "swa.pkl",
        )


def test_data_selection_fields_are_strict_and_roundtrip(tmp_path: Path) -> None:
    config = V24IlyaTrainConfig(
        **{
            **_config().__dict__,
            "time_start": "2021-01-01T00:00:00",
            "time_end": "2021-01-31T18:00:00",
            "allow_incomplete_prepared_store": True,
        }
    )
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_training_config(path) == config

    payload = config.to_dict()
    payload["data"]["unknown_partial_option"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown keys in data"):
        load_training_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"time_start": "2021-01-01T00:00:00"},
        {"time_end": "2021-01-31T18:00:00"},
        {"allow_incomplete_prepared_store": True},
        {"allow_incomplete_prepared_store": "true"},
    ],
)
def test_data_selection_fields_reject_invalid_combinations(overrides: dict) -> None:
    with pytest.raises(ValueError, match=r"data\.|provided together|requires"):
        V24IlyaTrainConfig(**{**_config().__dict__, **overrides})
