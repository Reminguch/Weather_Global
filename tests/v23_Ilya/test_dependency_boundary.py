from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/models/mamba/v23_Ilya"


def test_v23_has_no_old_training_imports() -> None:
    forbidden = ("v22_final", "full_mamba_v")
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


def test_reference_config_uses_only_v23_artifacts() -> None:
    config = (
        ROOT
        / "configs/experiments/v23_Ilya/"
        "res1_endpoint_k24_di16_bcg1_closed_sg_stateful_20k.json"
    ).read_text()
    assert "lianghong_v22" not in config
    assert '"architecture_id": "v23_Ilya"' in config
    assert '"loss_mode": "last_step"' in config
    assert '"bptt_backend": "explicit_reverse_vjp"' in config
