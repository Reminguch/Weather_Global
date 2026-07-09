#!/usr/bin/env python
"""Post-K=1 sanity check for v23.

Run after v23 K=1 finishes, BEFORE submitting v23_K22.slurm. Verifies:
  1. step23000 ckpt exists at expected path
  2. run_config.json has temporal_d_conv = 8
  3. Mamba conv1d kernel shape is [128, 8] (NOT [128, 4]) in ckpt
  4. K=1 loss trajectory exists and didn't blow up (last 20 steps < 1.0)
  5. Initial loss (step ~20010) is close to v22 K=1's initial loss (sanity that
     architecture wiring + warm-start are correct).

Exits 0 on success, 1 on any failure. Print summary either way.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

CKPT_PATH = Path(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23/K1_from_v15v2_20k/v13_residual_step23000.pkl"
)
RUN_CONFIG = CKPT_PATH.parent / "run_config.json"
TRAIN_LOG = CKPT_PATH.parent / "train_log.json"
V22_TRAIN_LOG = Path(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K1_from_v15v2_20k/train_log.json"
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    all_ok = True

    # 1. ckpt exists
    all_ok &= check(
        "v23 K=1 step23000 ckpt exists",
        CKPT_PATH.is_file(),
        str(CKPT_PATH),
    )
    if not CKPT_PATH.is_file():
        return 1  # rest of checks need ckpt

    # 2. run_config has d_conv = 8
    try:
        cfg = json.loads(RUN_CONFIG.read_text())["config"]
        d_conv = cfg.get("temporal_d_conv")
        all_ok &= check(
            "run_config.json has temporal_d_conv == 8",
            d_conv == 8,
            f"temporal_d_conv={d_conv}",
        )
        # Bonus: confirm other knobs unchanged from v22
        expected = {
            "temporal_d_state": 16,
            "temporal_hidden_size": 128,
            "temporal_layers": 2,
            "temporal_dt_rank": "auto",
            "ar_tail_K": 0,
            "bptt_steps": 24,
            "sequential_segment_steps": 96,
        }
        mismatches = [
            f"{k}={cfg.get(k)}!={v}" for k, v in expected.items() if cfg.get(k) != v
        ]
        all_ok &= check(
            "all other knobs match v22 K=1",
            not mismatches,
            ", ".join(mismatches) if mismatches else "",
        )
    except Exception as e:
        all_ok &= check("run_config.json parse", False, str(e))

    # 3. ckpt conv1d kernel shape is [128, 8]
    try:
        ckpt = pickle.loads(CKPT_PATH.read_bytes())
        params = ckpt["residual_params"]
        conv_shapes = []
        for mod_name, mod_params in params.items():
            if "conv1d" in mod_name:
                for pname, parr in mod_params.items():
                    if pname == "kernel":
                        conv_shapes.append((mod_name, tuple(parr.shape)))
        ok = bool(conv_shapes) and all(
            shape == (128, 8) for _, shape in conv_shapes
        )
        detail = (
            f"{len(conv_shapes)} conv1d kernels, all shape (128, 8)"
            if ok
            else f"shapes seen: {set(s for _, s in conv_shapes)}"
        )
        all_ok &= check(
            "all conv1d kernels have shape (128, 8)",
            ok,
            detail,
        )
    except Exception as e:
        all_ok &= check("ckpt conv1d shape inspection", False, str(e))

    # 4 + 5. train log sanity
    try:
        log = json.loads(TRAIN_LOG.read_text())
        # train_log.json is a list of records {step, loss, ...}; format may vary.
        # Be defensive about field names.
        if isinstance(log, list) and log:
            losses = [r.get("loss") for r in log if isinstance(r, dict)]
            steps = [r.get("step") for r in log if isinstance(r, dict)]
        else:
            raise ValueError(f"train_log.json format unexpected: top-level type={type(log).__name__}")

        last20 = [l for l in losses[-20:] if l is not None]
        last20_ok = bool(last20) and max(last20) < 1.0
        all_ok &= check(
            "last 20 K=1 losses < 1.0 (no blowup)",
            last20_ok,
            f"last 20 loss range = [{min(last20):.3f}, {max(last20):.3f}]" if last20 else "no losses found",
        )

        # Compare initial loss to v22 K=1
        first_few = [l for l in losses[:5] if l is not None]
        if first_few and V22_TRAIN_LOG.is_file():
            v22_log = json.loads(V22_TRAIN_LOG.read_text())
            v22_first = [
                r.get("loss") for r in v22_log[:5]
                if isinstance(r, dict) and r.get("loss") is not None
            ]
            if v22_first:
                v23_init = sum(first_few) / len(first_few)
                v22_init = sum(v22_first) / len(v22_first)
                rel_gap = abs(v23_init - v22_init) / max(v22_init, 1e-6)
                # Tolerance: 20% relative gap. Different seeds + fresh conv at d_conv=8
                # means v23 will not exactly match v22, but should be within reasonable range.
                all_ok &= check(
                    "initial loss within 20% of v22 K=1",
                    rel_gap < 0.20,
                    f"v23={v23_init:.3f}, v22={v22_init:.3f}, rel_gap={rel_gap:.1%}",
                )
    except FileNotFoundError:
        all_ok &= check("train log exists", False, str(TRAIN_LOG))
    except Exception as e:
        all_ok &= check("train log sanity", False, str(e))

    print()
    print("=" * 50)
    print("PASS — safe to sbatch slurm_pilots/v23_K22.slurm" if all_ok else "FAIL — fix issues above before launching K=22")
    print("=" * 50)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
