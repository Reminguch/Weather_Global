#!/usr/bin/env python3
"""Generate the explicit v24 res1 initialization/capacity/optimizer matrix."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904"
OUTPUT_ROOT = "artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904"
BASE_CONFIG = (
    ROOT
    / "configs/experiments/v24_Ilya/"
    "res1_v22compat_uniform_legacy_di16_bcg1_fp32_10k.json"
)

MODELS = (
    ("legacy", "legacy_haiku", 16, 1),
    ("mamba1", "mamba1", 16, 1),
    ("legacy", "legacy_haiku", 64, 4),
    ("mamba1", "mamba1", 64, 4),
)
OPTIMIZERS = (
    ("lr1em4_b1p9_b2p999", 1e-4, 0.9, 0.999),
    ("lr5em5_b1p9_b2p999", 5e-5, 0.9, 0.999),
    ("lr1em4_b1p9_b2p98", 1e-4, 0.9, 0.98),
    ("lr5em5_b1p9_b2p98", 5e-5, 0.9, 0.98),
    ("lr1em4_b1p8_b2p98", 1e-4, 0.8, 0.98),
    ("lr5em5_b1p8_b2p98", 5e-5, 0.8, 0.98),
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    generated: list[Path] = []
    for init_label, init_scheme, d_inner, bc_groups in MODELS:
        for optimizer_label, learning_rate, beta1, beta2 in OPTIMIZERS:
            payload = copy.deepcopy(base)
            architecture = payload["architecture"]
            architecture["temporal_init_scheme"] = init_scheme
            architecture["temporal_d_inner"] = d_inner
            architecture["temporal_bc_groups"] = bc_groups

            optimizer = payload["optimizer"]
            optimizer.update(
                max_steps=10_000,
                checkpoint_every=1_000,
                learning_rate=learning_rate,
                learning_rate_schedule="cosine",
                end_learning_rate=learning_rate * 0.1,
                adam_beta1=beta1,
                adam_beta2=beta2,
                warmup_steps=200,
                seed=22,
            )

            run_name = (
                f"{init_label}_di{d_inner}_bcg{bc_groups}_"
                f"{optimizer_label}_cos10k"
            )
            payload["output"] = {
                "output_root": OUTPUT_ROOT,
                "run_name": run_name,
            }
            path = CONFIG_DIR / f"{run_name}.json"
            write_json(path, payload)
            generated.append(path)

    # The new capacity needs a short compile/update check before its 12-run
    # array is released. Optimizer behavior is covered by unit tests; these
    # preflights isolate the two initialization-dependent parameter trees.
    for init_label, init_scheme in (("legacy", "legacy_haiku"), ("mamba1", "mamba1")):
        payload = copy.deepcopy(base)
        architecture = payload["architecture"]
        architecture["temporal_init_scheme"] = init_scheme
        architecture["temporal_d_inner"] = 64
        architecture["temporal_bc_groups"] = 4
        optimizer = payload["optimizer"]
        optimizer.update(
            max_steps=2,
            checkpoint_every=2,
            learning_rate=1e-4,
            learning_rate_schedule="constant",
            end_learning_rate=None,
            adam_beta1=0.9,
            adam_beta2=0.999,
            warmup_steps=1,
            seed=22,
        )
        run_name = f"{init_label}_di64_bcg4_smoke2"
        payload["output"] = {
            "output_root": f"{OUTPUT_ROOT}/preflight",
            "run_name": run_name,
        }
        path = CONFIG_DIR / "preflight" / f"{run_name}.json"
        write_json(path, payload)
        generated.append(path)

    print(f"generated {len(generated)} configs under {CONFIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
