"""Zero-init equivalence test for GC-Mamba_v2.

Foundational sanity: with --zero-init-temporal-out + --init-from-graphcast-ckpt,
the Mamba block's output projection is zero, so it acts as identity at init.
Therefore GC-Mamba prediction at init should equal vanilla DeepMind GraphCast
prediction within numerical noise.

If this fails, ALL downstream training/eval is meaningless because the
"frozen GraphCast" baseline is not actually identical to vanilla GraphCast.

Tests both temporal locations:
  - mesh_processor_interleaved (insert_count=2, layers=1)
  - mesh_post_processor       (insert_count=1, layers=1)

Pass criterion: max |pred_vanilla - pred_gcmamba| < 1e-3 (bf16 noise floor).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc_pkg, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.data.prepared_array import PreparedArrayStore  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    build_predictor,
    derive_model_config_from_checkpoint,
    load_graphcast_checkpoint,
    load_stats,
)
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-in",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
                           "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
                           "mesh 2to5 - precipitation input and output.npz")
    p.add_argument("--prepared-root",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/"
                           "prepared_stream_2015_2022_v2")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--processor-msg-steps", type=int, default=16)
    p.add_argument("--temporal-location",
                   choices=["mesh_post_encoder", "mesh_processor_interleaved", "mesh_post_processor"],
                   default="mesh_post_processor")
    p.add_argument("--temporal-d-inner", type=int, default=128)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--temporal-insert-count", type=int, default=None)
    p.add_argument("--anchor-idx", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-3,
                   help="Pass if max|diff| <= this. bf16 noise floor ~1e-3.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    cfg = parse_args()
    print(f"\n=== Zero-init equivalence test ===")
    print(f"  temporal-location  = {cfg.temporal_location}")
    print(f"  temporal-d-inner   = {cfg.temporal_d_inner}")
    print(f"  temporal-d-state   = {cfg.temporal_d_state}")
    print(f"  temporal-layers    = {cfg.temporal_layers}")
    print(f"  insert-count       = {cfg.temporal_insert_count}")
    print()

    # --- Load task / build configs ---
    ckpt_in = load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = dataclasses.replace(ckpt_in.task_config, input_duration="12h")

    model_cfg = derive_model_config_from_checkpoint(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.processor_msg_steps,
        hidden_layers=1,
    )
    norm_stats = load_stats(Path(cfg.stats_dir))

    # --- Build one batch from PreparedArrayStore ---
    store = PreparedArrayStore(cfg.prepared_root + "/res1", label="zero-init-test")
    store.validate(resolution=cfg.resolution, task_cfg=task_cfg)
    import pandas as pd
    dt = pd.Timedelta(np.diff(store.time.values)[0])
    inp, tgt, frc = store.build_batch_from_indices(
        indices=[cfg.anchor_idx], input_steps=2, target_steps=1,
        task_cfg=task_cfg, dt=dt)
    print(f"Built batch: anchor_idx={cfg.anchor_idx}, time={inp.time.values}")

    rng = jax.random.PRNGKey(cfg.seed)

    # Common Mamba kwargs for build_predictor (vanilla path also requires
    # them since build_predictor lists them as kw-only required).
    mamba_kwargs = dict(
        temporal_location=cfg.temporal_location,
        temporal_d_inner=cfg.temporal_d_inner,
        temporal_d_state=cfg.temporal_d_state,
        temporal_d_conv=cfg.temporal_d_conv,
        temporal_dt_rank=cfg.temporal_dt_rank,
        temporal_bias=False,
        temporal_conv_bias=True,
        temporal_layers=cfg.temporal_layers,
        temporal_dropout=0.0,
    )

    # --- Vanilla GraphCast forward fn ---
    def vanilla_forward(inputs, targets_template, forcings):
        predictor = build_predictor(
            model_cfg, task_cfg, norm_stats,
            use_bf16=True,
            gradient_checkpointing=False,
            temporal_backbone="none",
            **mamba_kwargs,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    vanilla_t = hk.transform_with_state(vanilla_forward)
    rng, vk = jax.random.split(rng)
    vanilla_params, vanilla_state = vanilla_t.init(vk, inp, tgt, frc)
    vanilla_params, ov = overlay_matching_params(vanilla_params, ckpt_in.params)
    print(f"Vanilla: copied={ov.copied}, new_init={ov.initialized}")
    pred_vanilla, _ = vanilla_t.apply(vanilla_params, vanilla_state, vk, inp, tgt, frc)

    # --- GC-Mamba forward fn (zero-init temporal out) ---
    def gcmamba_forward(inputs, targets_template, forcings):
        predictor = build_predictor(
            model_cfg, task_cfg, norm_stats,
            use_bf16=True,
            gradient_checkpointing=False,
            temporal_backbone="mamba",
            temporal_stateful=True,
            temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=True,
            **mamba_kwargs,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    gcmamba_t = hk.transform_with_state(gcmamba_forward)
    rng, gk = jax.random.split(rng)
    gcmamba_params, gcmamba_state = gcmamba_t.init(gk, inp, tgt, frc)
    gcmamba_params, og = overlay_matching_params(gcmamba_params, ckpt_in.params)
    print(f"GC-Mamba: copied={og.copied}, new_init={og.initialized}")
    pred_gcmamba, _ = gcmamba_t.apply(gcmamba_params, gcmamba_state, gk, inp, tgt, frc)

    # --- Compare ---
    print(f"\n=== Prediction diff (max |pred_vanilla - pred_gcmamba|) ===")
    print(f"{'variable':<32}{'max|diff|':>14}{'mean|diff|':>14}{'rel max':>12}{'verdict':>10}")
    print("-" * 82)

    all_pass = True
    for var in sorted(pred_vanilla.data_vars):
        if var not in pred_gcmamba.data_vars:
            continue
        v = np.asarray(pred_vanilla[var].values).astype(np.float32)
        g = np.asarray(pred_gcmamba[var].values).astype(np.float32)
        diff = np.abs(v - g)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        denom = float(np.abs(v).mean()) + 1e-9
        rel = max_d / denom
        ok = max_d <= cfg.tolerance
        if not ok: all_pass = False
        verdict = "PASS" if ok else "FAIL"
        print(f"{var:<32}{max_d:>14.4e}{mean_d:>14.4e}{rel:>12.3e}{verdict:>10}")
    print("-" * 82)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'} (tolerance = {cfg.tolerance:.0e})")
    print(f"If PASS: zero-init Mamba is identity → frozen GC behaves exactly like vanilla.")
    print(f"If FAIL: Mamba is perturbing latent at init → overlay or zero-init is broken.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
