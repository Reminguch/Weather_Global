from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.models.mamba.v22_final.evaluation import compare_metric_outputs


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = os.environ.get("V22_FINAL_PARITY_CKPT")
COMMON_ARGS_JSON = os.environ.get("V22_FINAL_PARITY_ARGS_JSON")


@pytest.mark.skipif(
    not CHECKPOINT or not COMMON_ARGS_JSON,
    reason="Set V22_FINAL_PARITY_CKPT and V22_FINAL_PARITY_ARGS_JSON for integration parity",
)
@pytest.mark.parametrize(
    ("eval_mode", "state_init"),
    [
        ("cold_bp", "zero"),
        ("cold_full", "zero"),
        ("warm_full", "zero"),
        ("cold_bp", "ckpt"),
    ],
)
def test_legacy_and_v22_final_json_parity(
    tmp_path: Path,
    eval_mode: str,
    state_init: str,
) -> None:
    common_args = json.loads(COMMON_ARGS_JSON)
    if not isinstance(common_args, list) or not all(isinstance(value, str) for value in common_args):
        raise ValueError("V22_FINAL_PARITY_ARGS_JSON must be a JSON array of CLI argument strings")

    suffix = f"{eval_mode}_{state_init}"
    legacy_output = tmp_path / f"legacy_{suffix}.json"
    final_output = tmp_path / f"final_{suffix}.json"
    fixed_args = [
        "--ckpt",
        str(CHECKPOINT),
        "--eval-mode",
        eval_mode,
        "--residual-state-init",
        state_init,
        "--target-steps",
        "2",
        "--warmup-steps",
        "2",
        "--n-samples",
        "1",
        "--seed",
        "0",
    ]
    legacy_command = [
        sys.executable,
        "scripts/training/full_mamba_v23/eval_v22_clean.py",
        *common_args,
        *fixed_args,
        "--out-json",
        str(legacy_output),
    ]
    final_command = [
        sys.executable,
        "scripts/analyze_models/eval_v22_final.py",
        *common_args,
        *fixed_args,
        "--out-json",
        str(final_output),
    ]
    subprocess.run(legacy_command, cwd=ROOT, check=True)
    subprocess.run(final_command, cwd=ROOT, check=True)

    legacy = json.loads(legacy_output.read_text(encoding="utf-8"))
    candidate = json.loads(final_output.read_text(encoding="utf-8"))
    compare_metric_outputs(legacy, candidate, rtol=1e-6, atol=1e-6)
