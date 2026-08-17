from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.mamba.v22_final.training.config import (
    V22FinalValidationConfig,
    apply_operational_overrides,
    load_training_config,
    parse_cli,
    validate_resume_config,
)
from src.models.mamba.v22_final.training.data import (
    TrainingCursor,
    advance_cursor,
    build_segments,
    open_training_data,
)
from src.models.mamba.v22_final.training import data as training_data_module


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = ROOT / "configs/experiments/v22_final/res2_gc500k_k12_open.json"
CARRY_CONFIG = ROOT / "configs/experiments/v22_final/res2_gc500k_k12_di16_full_mamba_carry_20k.json"
RESET_CONFIG = ROOT / "configs/experiments/v22_final/res2_gc500k_k12_di16_full_mamba_reset_every_anchor_20k.json"


def test_reference_config_and_operational_overrides() -> None:
    config = load_training_config(REFERENCE_CONFIG)
    assert config.architecture.resolution == 2.0
    assert config.architecture.temporal_bc_groups == 1
    assert config.to_dict()["architecture"]["temporal_bc_groups"] == 1
    assert "temporal_hidden_size" not in config.to_dict()["architecture"]
    assert config.truth_prefix_steps == 4
    assert config.ar_tail_k == 12
    assert config.feedback_mode == "baseline"
    assert config.temporal_state_policy == "carry"
    assert config.to_dict()["sequence"]["temporal_state_policy"] == "carry"
    assert config.validation == V22FinalValidationConfig(
        enabled=True,
        every_steps=2_000,
        num_segments=16,
        final_num_segments=None,
    )
    overridden = apply_operational_overrides(
        config,
        max_steps=20,
        checkpoint_every=10,
        output_root=Path("smoke"),
        run_name="open",
        learning_rate=3e-5,
    )
    assert overridden.max_steps == 20
    assert overridden.checkpoint_every == 10
    assert overridden.learning_rate == 3e-5
    assert overridden.run_dir == Path("smoke/open")


def test_warm_start_cli_can_zero_state_and_override_learning_rate() -> None:
    invocation = parse_cli(
        [
            "--config",
            str(REFERENCE_CONFIG),
            "--init-from",
            "average.pkl",
            "--zero-state-on-init",
            "--learning-rate",
            "1e-5",
        ]
    )
    assert invocation.init_from == Path("average.pkl")
    assert invocation.zero_state_on_init
    assert invocation.config.learning_rate == 1e-5


def test_full_mamba_state_ablation_configs_differ_only_by_policy_and_run_name() -> None:
    carry = load_training_config(CARRY_CONFIG)
    reset = load_training_config(RESET_CONFIG)
    assert carry.architecture.temporal_stateful is True
    assert carry.architecture == reset.architecture
    assert carry.temporal_state_policy == "carry"
    assert reset.temporal_state_policy == "reset_every_anchor"
    assert dataclasses.replace(
        carry,
        temporal_state_policy=reset.temporal_state_policy,
        run_name=reset.run_name,
    ) == reset


def test_config_rejects_unknown_and_invalid_values(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    payload["sequence"]["unknown"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys in sequence"):
        load_training_config(path)

    payload = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    payload["architecture"]["temporal_hidden_size"] = 128
    path = tmp_path / "removed-hidden-size.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="temporal_hidden_size"):
        load_training_config(path)

    config = load_training_config(REFERENCE_CONFIG)
    with pytest.raises(ValueError, match="ar_tail_k"):
        dataclasses.replace(config, ar_tail_k=config.bptt_steps)
    with pytest.raises(ValueError, match="divisible"):
        dataclasses.replace(config, segment_steps=65)
    with pytest.raises(ValueError, match="temporal_state_policy"):
        dataclasses.replace(config, temporal_state_policy="sometimes")

    with pytest.raises(ValueError, match="temporal_bc_groups"):
        dataclasses.replace(config.architecture, temporal_bc_groups=0)
    with pytest.raises(ValueError, match="must not exceed"):
        dataclasses.replace(config.architecture, temporal_bc_groups=17)
    with pytest.raises(ValueError, match="divisible"):
        dataclasses.replace(config.architecture, temporal_bc_groups=3)
    with pytest.raises(ValueError, match="every_steps"):
        V22FinalValidationConfig(enabled=True, every_steps=0)
    with pytest.raises(ValueError, match="num_segments"):
        V22FinalValidationConfig(enabled=True, num_segments=0)

    payload = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    payload.pop("validation")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_training_config(legacy_path).validation.enabled is False


def test_resume_allows_only_operational_step_cadence() -> None:
    config = load_training_config(REFERENCE_CONFIG)
    resumed = dataclasses.replace(config, max_steps=60_000, checkpoint_every=1_000)
    legacy_saved = config.to_dict()
    legacy_saved["sequence"].pop("temporal_state_policy")
    legacy_saved["architecture"].pop("temporal_bc_groups")
    legacy_saved["architecture"]["temporal_hidden_size"] = 128
    validate_resume_config(resumed, legacy_saved, completed_step=50_000)

    validate_resume_config(resumed, config.to_dict(), completed_step=50_000)
    validate_resume_config(
        dataclasses.replace(
            resumed,
            validation=V22FinalValidationConfig(
                enabled=True, every_steps=500, num_segments=2
            ),
        ),
        config.to_dict(),
        completed_step=50_000,
    )
    with pytest.raises(ValueError, match="differs"):
        validate_resume_config(
            dataclasses.replace(resumed, feedback_mode="closed_loop_sg"),
            config.to_dict(),
            completed_step=50_000,
        )
    with pytest.raises(ValueError, match="differs"):
        validate_resume_config(
            dataclasses.replace(resumed, temporal_state_policy="reset_every_anchor"),
            config.to_dict(),
            completed_step=50_000,
        )
    with pytest.raises(ValueError, match="only increase"):
        validate_resume_config(
            dataclasses.replace(config, max_steps=40_000),
            config.to_dict(),
            completed_step=20_000,
        )


def test_segments_preserve_order_and_discard_remainder() -> None:
    split = np.arange(10, dtype=np.int64)
    segments = build_segments(split, 4)
    assert [segment.tolist() for segment in segments] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_cursor_advances_with_segment_and_epoch_boundaries() -> None:
    config = dataclasses.replace(
        load_training_config(REFERENCE_CONFIG),
        segment_steps=4,
        bptt_steps=2,
        ar_tail_k=1,
    )
    assert advance_cursor(TrainingCursor(), config, 2) == TrainingCursor(0, 0, 2)
    assert advance_cursor(TrainingCursor(0, 0, 2), config, 2) == TrainingCursor(0, 1, 0)
    assert advance_cursor(TrainingCursor(0, 1, 2), config, 2) == TrainingCursor(1, 0, 0)


@pytest.mark.parametrize(("bptt", "tail", "prefix"), [(1, 0, 1), (16, 12, 4), (8, 7, 1)])
def test_truth_prefix_boundaries(bptt: int, tail: int, prefix: int) -> None:
    config = dataclasses.replace(
        load_training_config(REFERENCE_CONFIG),
        segment_steps=bptt,
        bptt_steps=bptt,
        ar_tail_k=tail,
    )
    assert config.truth_prefix_steps == prefix


def _write_manifest(root: Path, *, resolution: float = 2.0) -> None:
    (root / "anchors").mkdir(parents=True)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "resolution": resolution,
                "input_steps": 2,
                "target_variables": ["x"],
            }
        ),
        encoding="utf-8",
    )
    np.save(root / "anchors/anchor_indices.npy", np.arange(1, 17, dtype=np.int64))
    np.save(root / "anchors/split_train.npy", np.arange(8, dtype=np.int64))
    np.save(root / "anchors/split_val.npy", np.arange(8, 16, dtype=np.int64))


def test_manifest_validation_and_ordered_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self, root, *, label):
            del root, label
            self.time = SimpleNamespace(
                values=np.arange(
                    np.datetime64("2020-01-01T00"),
                    np.datetime64("2020-01-26T00"),
                    np.timedelta64(6, "h"),
                )
            )
            self.sizes = {"time": self.time.values.size}

        def validate(self, *, resolution, task_cfg):
            assert resolution == 2.0
            assert task_cfg.target_variables == ("x",)

    monkeypatch.setattr(training_data_module, "PreparedArrayStore", FakeStore)
    manifest_root = tmp_path / "manifest"
    _write_manifest(manifest_root)
    config = dataclasses.replace(
        load_training_config(REFERENCE_CONFIG),
        prepared_root=tmp_path / "prepared",
        anchor_manifest_root=manifest_root,
        segment_steps=4,
        bptt_steps=2,
        ar_tail_k=1,
    )
    task = SimpleNamespace(input_duration="12h", target_variables=("x",))
    data = open_training_data(config, task)
    assert [segment.tolist() for segment in data.segments] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert len(data.manifest_fingerprint) == 64
    assert [segment.tolist() for segment in data.validation_segments] == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert data.fixed_validation_segment_ids.tolist() == [0, 1]
    assert data.validation_subset_policy == "all"
    assert len(data.validation_subset_fingerprint or "") == 64
    assert data.fingerprint_validation_subset(np.asarray([0, 1])) == (
        data.validation_subset_fingerprint
    )

    (manifest_root / "metadata.json").write_text(
        json.dumps(
            {"resolution": 1.0, "input_steps": 2, "target_variables": ["x"]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resolution"):
        open_training_data(config, task)
