from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/models/mamba/v24_Ilya"


def test_v24_has_no_old_training_imports() -> None:
    forbidden = ("v23_Ilya", "v22_final", "full_mamba_v")
    violations = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(token in name for token in forbidden):
                    violations.append((path.relative_to(ROOT), name))
    assert not violations


def test_reference_config_uses_only_v24_artifacts() -> None:
    config = (
        ROOT
        / "configs/experiments/v24_Ilya/"
        "res1_sparse_h20_di16_bcg1_fp32_smoke.json"
    ).read_text()
    assert "lianghong_v22" not in config
    assert '"architecture_id": "v24_Ilya"' in config
    assert '"loss_mode": "sparse_steps"' in config
    assert '"weather_tape_precision": "fp32"' in config
    assert '"bptt_backend": "explicit_reverse_vjp"' in config
