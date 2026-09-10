"""Separate spatial capacity while preserving the pretrained baseline contract."""
import dataclasses
import json
from pathlib import Path

import pytest

from graphcast import graphcast as gc
from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig, parse_args
from src.models.mamba.v24_Ilya.model import build_model_configs
from src.models.mamba.v24_Ilya.training.config import (
    load_training_config, validate_resume_config,
)


REFERENCE = Path("configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904/"
                 "mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k.json")


def test_width_changes_only_residual_model():
    base = gc.ModelConfig(resolution=1, mesh_size=5, latent_size=512,
                          gnn_msg_steps=16, hidden_layers=1,
                          radius_query_fraction_edge_length=0.6)
    for width, expected in ((None, 512), (128, 128)):
        configs = build_model_configs(base, V24IlyaArchitectureConfig(residual_width=width))
        assert configs.baseline == base
        assert configs.residual.latent_size == expected
        assert configs.residual.gnn_msg_steps == 2
        assert configs.residual.mesh_size == 5


def test_training_eval_and_resume_preserve_width_and_initialization(tmp_path):
    original = load_training_config(REFERENCE)
    architecture = dataclasses.replace(original.architecture, residual_width=128,
                                       residual_initialization="fresh", temporal_dt_rank="32")
    narrow = dataclasses.replace(original, architecture=architecture)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(narrow.to_dict()))
    assert load_training_config(path) == narrow
    evaluated = parse_args(["--ckpt", "narrow.pkl", "--out-json", "eval.json",
                            "--eval-mode", "cold_full", "--residual-width", "128",
                            "--residual-initialization", "fresh", "--temporal-dt-rank", "32"])
    assert evaluated.architecture.residual_width == 128
    assert evaluated.architecture.width == 512
    assert evaluated.architecture.residual_initialization == "fresh"
    validate_resume_config(narrow, narrow.to_dict(), completed_step=1000)
    with pytest.raises(ValueError, match="Resume configuration differs"):
        validate_resume_config(original, narrow.to_dict(), completed_step=1000)
    # Old checkpoint metadata omits both new fields.
    old = original.to_dict()
    old["architecture"].pop("residual_width")
    old["architecture"].pop("residual_initialization")
    validate_resume_config(original, old, completed_step=1000)


@pytest.mark.parametrize("kwargs", [{"residual_width": 0}, {"residual_width": -128},
                                   {"residual_initialization": "partial"}])
def test_invalid_architecture_rejected(kwargs):
    with pytest.raises(ValueError):
        V24IlyaArchitectureConfig(**kwargs)
