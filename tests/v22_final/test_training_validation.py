from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.mamba.v22_final.training.config import (
    V22FinalValidationConfig,
    load_training_config,
)
from src.models.mamba.v22_final.training.data import BPTTChunk
from src.models.mamba.v22_final.training.validation import (
    append_validation_record,
    load_validation_records,
    run_fixed_validation,
    select_validation_segment_ids,
    update_best_validation,
    validation_step_keys,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = ROOT / "configs/experiments/v22_final/res2_gc500k_k12_open.json"


class _FakeValidationData:
    def __init__(self) -> None:
        self.validation_segments = tuple(
            np.arange(start, start + 4, dtype=np.int64) for start in range(0, 16, 4)
        )
        self.anchor_indices = np.arange(16, dtype=np.int64)
        self.store = SimpleNamespace(
            time=SimpleNamespace(
                values=np.arange(
                    np.datetime64("2022-01-01T00"),
                    np.datetime64("2022-01-05T00"),
                    np.timedelta64(6, "h"),
                )
            )
        )

    def load_segment_chunk(self, segment, segment_offset, config, task_config):
        del task_config
        raw = np.asarray(segment[segment_offset : segment_offset + config.bptt_steps])
        first = float(raw[0] + 1)
        return BPTTChunk(
            truth_inputs=(first,),
            targets=tuple(first for _ in range(config.bptt_steps)),
            forcings=tuple(0.0 for _ in range(config.bptt_steps)),
            raw_anchor_indices=raw,
        )

    def fingerprint_validation_subset(self, segment_ids):
        return hashlib.sha256(np.asarray(segment_ids, dtype=np.int64).tobytes()).hexdigest()


def _toy_config():
    return dataclasses.replace(
        load_training_config(REFERENCE_CONFIG),
        segment_steps=4,
        bptt_steps=2,
        ar_tail_k=1,
        validation=V22FinalValidationConfig(
            enabled=True,
            every_steps=2,
            num_segments=2,
            final_num_segments=None,
        ),
    )


def test_validation_rng_and_stratified_subset_are_deterministic() -> None:
    first = validation_step_keys(seed=18, segment_id=2, chunk_index=1, bptt_steps=2)
    second = validation_step_keys(seed=18, segment_id=2, chunk_index=1, bptt_steps=2)
    different = validation_step_keys(seed=18, segment_id=3, chunk_index=1, bptt_steps=2)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)

    data = _FakeValidationData()
    ids1, policy1 = select_validation_segment_ids(data, 2)
    ids2, policy2 = select_validation_segment_ids(data, 2)
    np.testing.assert_array_equal(ids1, ids2)
    assert ids1.tolist() == [0, 3]
    assert policy1 == policy2 == "stratified_fixed"


def test_validation_uses_anchor_weighting_and_resets_each_segment() -> None:
    data = _FakeValidationData()
    observed_states = []

    def validation_step(params, state, keys, truth_inputs, targets, forcings):
        del params, keys, truth_inputs, forcings
        observed_states.append(int(state))
        return jnp.asarray(targets[0], dtype=jnp.float32), state + 1

    params = {"weight": jnp.asarray(7.0)}
    result = run_fixed_validation(
        validation_step=validation_step,
        residual_params=params,
        zero_residual_state=jnp.asarray(0),
        training_data=data,
        task_config=SimpleNamespace(),
        config=_toy_config(),
        segment_ids=np.asarray([0, 1]),
        step=2,
        role="fixed_checkpoint",
        subset_policy="stratified_fixed",
    )
    assert observed_states == [0, 1, 0, 1]
    assert result["loss"] == pytest.approx(np.mean([1.0, 3.0, 5.0, 7.0]))
    assert result["num_chunks"] == 4
    assert result["num_anchors"] == 8
    np.testing.assert_array_equal(params["weight"], 7.0)


def test_validation_log_is_idempotent_and_best_ties_keep_earlier(tmp_path: Path) -> None:
    path = tmp_path / "validation_metrics.jsonl"
    base = {
        "step": 2,
        "loss": 1.0,
        "duration_seconds": 3.0,
        "role": "fixed_checkpoint",
        "subset_fingerprint": "same",
    }
    assert append_validation_record(path, base)
    assert not append_validation_record(path, base)
    with pytest.raises(ValueError, match="Conflicting"):
        append_validation_record(path, {**base, "loss": 2.0})
    assert append_validation_record(
        path, {**base, "loss": 2.0, "subset_fingerprint": "changed"}
    )
    assert append_validation_record(path, {**base, "step": 4})
    assert append_validation_record(
        path,
        {**base, "step": 4, "role": "final_full"},
    )

    best_path = tmp_path / "best_validation.json"
    best = update_best_validation(
        records=load_validation_records(path),
        checkpoint_for_step=lambda step: tmp_path / f"checkpoint-{step}.pkl",
        output_path=best_path,
        subset_fingerprint="same",
    )
    assert best is not None
    assert best["step"] == 2
    assert json.loads(best_path.read_text())["checkpoint"].endswith("checkpoint-2.pkl")
