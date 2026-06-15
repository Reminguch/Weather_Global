import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from src.models.mamba.training.segments_training import (
    ar_gradient_alignment_masks,
    ar_tail_group_indices,
    build_ar_gradient_alignment_record,
    safe_cosine,
)


def test_ar_tail_group_indices_k12_bptt16() -> None:
    groups = ar_tail_group_indices(bptt_steps=16, target_steps=12)
    assert groups == {
        "early": [4, 5, 6, 7],
        "mid": [8, 9, 10, 11],
        "late": [12, 13, 14, 15],
    }


def test_ar_gradient_alignment_masks_include_uniform_tail() -> None:
    masks = ar_gradient_alignment_masks(bptt_steps=16, target_steps=12)
    assert masks["early"] == [0.0] * 4 + [1.0] * 4 + [0.0] * 8
    assert masks["mid"] == [0.0] * 8 + [1.0] * 4 + [0.0] * 4
    assert masks["late"] == [0.0] * 12 + [1.0] * 4
    assert masks["uniform"] == [0.0] * 4 + [1.0] * 12
    assert masks["prefix"] == [1.0] * 4 + [0.0] * 12
    assert masks["all_bptt"] == [1.0] * 16


def test_safe_cosine_handles_zero_norm_gradients() -> None:
    assert safe_cosine(1.0, 0.0, 2.0) == 0.0
    assert safe_cosine(1.0, 2.0, 0.0) == 0.0
    assert safe_cosine(3.0, 2.0, 3.0) == 0.5


def test_build_ar_gradient_alignment_record_required_fields() -> None:
    record = build_ar_gradient_alignment_record(
        step=20000,
        chunk_index=0,
        bptt_steps=16,
        target_steps=12,
        truth_prefix_steps=4,
        tail_groups={
            "early": [4, 5, 6, 7],
            "mid": [8, 9, 10, 11],
            "late": [12, 13, 14, 15],
        },
        losses={"early": 1.0, "mid": 2.0, "late": 3.0, "uniform": 2.0},
        param_groups={
            "all": {
                "gradient_norms": {"early": 1.0, "mid": 2.0, "late": 3.0, "uniform": 4.0},
                "lr_weighted_update_norms": {
                    "early": 1e-5,
                    "mid": 2e-5,
                    "late": 3e-5,
                    "uniform": 4e-5,
                },
                "dot_to_late": {"early": 0.1, "mid": 0.2, "late": 9.0, "uniform": 0.3},
                "cosine_to_late": {"early": 0.1, "mid": 0.2, "late": 1.0, "uniform": 0.3},
            }
        },
        graphcast_lr=1e-6,
        mamba_lr=1e-5,
        lora_lr=2e-5,
    )
    assert record["step"] == 20000
    assert record["truth_prefix_steps"] == 4
    assert record["prefix_groups"] == {"prefix": [0, 1, 2, 3]}
    assert record["tail_horizon_groups"]["early"] == [1, 2, 3, 4]
    assert record["tail_horizon_groups"]["mid"] == [5, 6, 7, 8]
    assert record["tail_horizon_groups"]["late"] == [9, 10, 11, 12]
    assert record["learning_rates"] == {"graphcast": 1e-6, "mamba": 1e-5, "lora": 2e-5}
    assert "all" in record["param_groups"]
