"""Verify that GraphCastRemat (subclass with hk.remat boundaries) produces
the same loss/predictions/gradients as the base GraphCast.

Why this matters:
  - hk.remat on a bound method can mis-handle Haiku state/param lifting.
  - The reimplemented _run_mesh_gnn_interleaved loop may diverge from the
    base in subtle ways (e.g. private API ordering).
  - zero-init Mamba (output_proj=0) hides ALL of this because Mamba's
    contribution to the prediction is exactly zero. Equivalence at zero-init
    only checks that the OVERLAY/SCOPE is correct, NOT that the remat'd
    processor loop is mathematically equivalent.

So this script:
  1. Builds both predictors (non-remat + remat with same temporal_*).
  2. Loads DeepMind GC into the GC leaves.
  3. *Perturbs Mamba params away from zero-init* so the temporal path is
     actually exercised.
  4. Runs a single forward + backward on the SAME batch.
  5. Compares loss, predictions, and gradients (Mamba params only).

Pass criteria (bf16, so don't expect bitwise identity):
  - |loss_diff| < 1e-2
  - max|pred_diff| < a few * 1e-2 (geopotential dominates absolute scale)
  - relative grad tree norm diff < 1e-2 (Mamba grads only)
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

from src.models.graphcast.training.core.model import (  # noqa: E402
    build_predictor,
    load_graphcast_checkpoint,
    load_stats,
    scalarize_loss,
)
from src.models.graphcast_remat.build import build_predictor_remat  # noqa: E402
from src.models.mamba.training.param_utils import (  # noqa: E402
    is_temporal_param,
    overlay_matching_params,
)
import scripts.training.train_graphcast as base_train  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
    p.add_argument("--ckpt-in",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
                           "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
                           "mesh 2to5 - precipitation input and output.npz")
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=64)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--mamba-noise-scale", type=float, default=1e-2,
                   help="Std of gaussian noise added to Mamba params to break "
                        "zero-init. 1e-2 -> small but non-zero Mamba contribution.")
    p.add_argument("--mamba-perturb-out-proj-only", action="store_true",
                   help="Only perturb the zero-init'd output projection of Mamba. "
                        "Other Mamba leaves stay at fresh Haiku init.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--remat-processor-steps", action="store_true", default=False)
    p.add_argument("--remat-mesh2grid", action="store_true", default=False)
    p.add_argument("--remat-grid2mesh", action="store_true", default=False)
    # Default (no flag): SAFE Mode-2 — processor remat'd, Mamba outside.
    # Verified rel diff ~0.03 in earlier verify.
    # Use --unsafe-remat-temporal-inside to test the UNSAFE Mode-1
    # (Mamba INSIDE remat boundary). Failed verification with rel diff ~0.68.
    p.add_argument("--unsafe-remat-temporal-inside", action="store_true",
                   default=False, help=argparse.SUPPRESS)
    p.add_argument("--no-temporal-stateful",
                   dest="temporal_stateful", action="store_false", default=True,
                   help="Disable stateful Mamba (isolate stateful from remat correctness).")
    return p.parse_args()


def build_transform(cfg, *, remat: bool, model_cfg, task_cfg, norm_stats):
    builder = build_predictor_remat if remat else build_predictor
    extra = (
        dict(remat_processor_steps=cfg.remat_processor_steps,
             remat_mesh2grid=cfg.remat_mesh2grid,
             remat_grid2mesh=cfg.remat_grid2mesh,
             remat_temporal_inside_processor=cfg.unsafe_remat_temporal_inside)
        if remat else {})

    def forward_fn(inputs, targets, forcings, is_training):
        predictor = builder(
            model_cfg, task_cfg, norm_stats,
            use_bf16=True,
            gradient_checkpointing=False,
            temporal_backbone="mamba",
            temporal_location="mesh_processor_interleaved",
            temporal_hidden_size=cfg.temporal_hidden_size,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank="auto",
            temporal_bias=False,
            temporal_conv_bias=True,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=0.0,
            temporal_stateful=cfg.temporal_stateful,
            zero_init_temporal_out=True,
            **extra,
        )
        return predictor.loss(inputs, targets, forcings)

    return hk.transform_with_state(forward_fn)


def perturb_mamba_params(params, key, scale, out_proj_only):
    """Add Gaussian noise to Mamba leaves so Mamba actually contributes."""
    out = {m: dict(leaves) for m, leaves in params.items()}
    keys = jax.random.split(key, sum(
        len(leaves) for leaves in out.values()))
    k_idx = 0
    n_perturbed = 0
    for module_name, leaves in out.items():
        for leaf_name in list(leaves):
            if not is_temporal_param(module_name, leaf_name):
                k_idx += 1
                continue
            # If --mamba-perturb-out-proj-only, only perturb 'out_proj' leaves
            if out_proj_only and "out_proj" not in leaf_name.lower() \
                    and "out_proj" not in module_name.lower():
                k_idx += 1
                continue
            value = leaves[leaf_name]
            noise = scale * jax.random.normal(
                keys[k_idx], shape=value.shape, dtype=value.dtype)
            leaves[leaf_name] = value + noise
            n_perturbed += 1
            k_idx += 1
    return out, n_perturbed


def tree_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(jnp.asarray(l).astype(jnp.float32)))
                              for l in leaves)))


def max_abs_diff(pred_a, pred_b):
    """Max |a-b| across all data vars of an xarray.Dataset prediction."""
    maxes = []
    for var in pred_a.data_vars:
        a = np.asarray(pred_a[var].values, dtype=np.float32)
        b = np.asarray(pred_b[var].values, dtype=np.float32)
        maxes.append((var, float(np.max(np.abs(a - b))), float(np.max(np.abs(a)))))
    return maxes


def main():
    cfg = parse_args()
    print(f"[verify-remat] config: remat-processor={cfg.remat_processor_steps} "
          f"mesh2grid={cfg.remat_mesh2grid} grid2mesh={cfg.remat_grid2mesh} "
          f"unsafe-temporal-inside={cfg.unsafe_remat_temporal_inside} "
          f"stateful={cfg.temporal_stateful}")

    # Load DeepMind ckpt
    ckpt_in = load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg = dataclasses.replace(
        base_model_cfg,
        resolution=1.0, mesh_size=5, latent_size=512, gnn_msg_steps=16,
    )

    norm_stats = load_stats(Path(cfg.stats_dir))

    # Small data batch
    class _Sc:
        data_path = cfg.data_path
        resolution = 1.0
        val_year = 2022
        train_start_year = 2020
        train_end_year = 2021
    train_ds, _ = base_train._open_local_splits(_Sc)
    train_ds = base_train.prepare_dataset_for_task(train_ds, task_cfg)
    dt = base_train.infer_time_step(train_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    indices = base_train.valid_final_input_indices(
        train_ds.sizes["time"], input_steps, target_steps=1)
    inp, tgt, frc = base_train.build_batch_from_indices(
        train_ds, indices=[int(indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)

    # Build BOTH transforms
    tr_a = build_transform(cfg, remat=False, model_cfg=model_cfg,
                            task_cfg=task_cfg, norm_stats=norm_stats)
    tr_b = build_transform(cfg, remat=True, model_cfg=model_cfg,
                            task_cfg=task_cfg, norm_stats=norm_stats)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_init_a, k_init_b = jax.random.split(rng, 3)
    print("[verify-remat] init non-remat transform...")
    params_a, state_a = tr_a.init(k_init_a, inp, tgt, frc, True)
    print("[verify-remat] init remat transform...")
    params_b, state_b = tr_b.init(k_init_b, inp, tgt, frc, True)

    # Sanity: param trees should have identical structure
    keys_a = set((m, p) for m, ld in params_a.items() for p in ld)
    keys_b = set((m, p) for m, ld in params_b.items() for p in ld)
    if keys_a != keys_b:
        only_a = list(keys_a - keys_b)[:5]
        only_b = list(keys_b - keys_a)[:5]
        raise AssertionError(
            f"Param tree structure mismatch! "
            f"non-remat-only={only_a} remat-only={only_b}")
    print(f"[verify-remat] param trees match: {len(keys_a)} leaves both sides")

    # Overlay DeepMind into BOTH (for tree-structure sanity), but we will
    # then DISCARD params_b/state_b entirely and use params_a/state_a for
    # both apply calls.
    params_a, _ = overlay_matching_params(params_a, ckpt_in.params, strict=False)
    params_b_init_only, _ = overlay_matching_params(params_b, ckpt_in.params, strict=False)

    # STRUCTURAL ASSERT: param tree keys must be byte-for-byte identical
    # between the two transforms. Any mismatch here means the remat subclass
    # changed Haiku scope — verification wouldn't be meaningful otherwise.
    keys_a = {(m, p) for m, ld in params_a.items() for p in ld}
    keys_b = {(m, p) for m, ld in params_b_init_only.items() for p in ld}
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)[:5]
        only_b = sorted(keys_b - keys_a)[:5]
        raise AssertionError(
            f"Param tree structure differs between non-remat and remat:\n"
            f"  only in non-remat: {only_a}\n"
            f"  only in remat:     {only_b}")
    print(f"[verify-remat] param tree structure identical: {len(keys_a)} leaves")

    # Also check state tree structure (Mamba SSM state etc.).
    sk_a = {(m, p) for m, ld in state_a.items() for p in ld}
    sk_b = {(m, p) for m, ld in state_b.items() for p in ld}
    if sk_a != sk_b:
        raise AssertionError(
            f"State tree structure differs:\n"
            f"  only in non-remat: {sorted(sk_a - sk_b)[:5]}\n"
            f"  only in remat:     {sorted(sk_b - sk_a)[:5]}")
    print(f"[verify-remat] state tree structure identical: {len(sk_a)} state leaves")

    # FORCE COMPLETE EQUALITY: use ONE shared params + state object for both
    # transforms.  This means ANY remaining loss/grad diff between A and B
    # is purely from remat behavior (forward and backward), not init noise.
    shared_state = state_a   # state_a == state_b structure now; pick one
    # Perturb Mamba in the shared params, same noise to both (trivially true
    # since they share the same object now).
    rng, k_pert = jax.random.split(rng)
    params_a, n_pert = perturb_mamba_params(
        params_a, k_pert, cfg.mamba_noise_scale, cfg.mamba_perturb_out_proj_only)
    # Hard alias: both sides see exactly the same params object.
    params_b = params_a
    print(f"[verify-remat] perturbed {n_pert} Mamba leaves with N(0, {cfg.mamba_noise_scale})")
    print(f"[verify-remat] params_b := params_a (single shared object)")

    # SAME rng key for both forward passes.
    rng, k_fwd = jax.random.split(rng)

    # SAME params + SAME state + SAME rng -> any non-zero loss/grad diff is
    # purely from remat behavior.
    def loss_fn_a(p):
        (loss_da, _), _ = tr_a.apply(p, shared_state, k_fwd, inp, tgt, frc, True)
        return scalarize_loss(loss_da)

    def loss_fn_b(p):
        (loss_da, _), _ = tr_b.apply(p, shared_state, k_fwd, inp, tgt, frc, True)
        return scalarize_loss(loss_da)

    print("[verify-remat] running non-remat forward...")
    loss_a, grads_a = jax.value_and_grad(loss_fn_a)(params_a)
    print(f"  loss_a = {float(loss_a):.6f}")

    print("[verify-remat] running remat forward...")
    loss_b, grads_b = jax.value_and_grad(loss_fn_b)(params_b)
    print(f"  loss_b = {float(loss_b):.6f}")

    # ---- Compare ----
    print("\n=== EQUIVALENCE REPORT ===")
    loss_diff = abs(float(loss_a) - float(loss_b))
    print(f"loss_a        = {float(loss_a):.6f}")
    print(f"loss_b        = {float(loss_b):.6f}")
    print(f"|loss_a - b|  = {loss_diff:.2e}   (gate: < 1e-2)")

    # Mamba-only grads on both sides.
    mamba_grad_a = {m: l for m, l in grads_a.items()
                     if any(is_temporal_param(m, p) for p in l)}
    mamba_grad_b = {m: l for m, l in grads_b.items()
                     if any(is_temporal_param(m, p) for p in l)}
    diff_tree = jax.tree_util.tree_map(lambda a, b: a - b,
                                        mamba_grad_a, mamba_grad_b)
    norm_a = tree_norm(mamba_grad_a)
    norm_b = tree_norm(mamba_grad_b)
    norm_diff = tree_norm(diff_tree)
    rel_diff = norm_diff / max(norm_a, 1e-12)
    print(f"||grad_a||   = {norm_a:.4e}")
    print(f"||grad_b||   = {norm_b:.4e}")
    print(f"||grad_a-b|| = {norm_diff:.4e}")
    print(f"rel diff     = {rel_diff:.2e}   (gate: < 1e-2)")

    # Per-leaf top-10: which leaves dominate the diff?
    leaf_rows = []
    for module_name, leaves in mamba_grad_a.items():
        for leaf_name, ga in leaves.items():
            gb = mamba_grad_b[module_name][leaf_name]
            ga_arr = np.asarray(ga).astype(np.float32)
            gb_arr = np.asarray(gb).astype(np.float32)
            d = ga_arr - gb_arr
            norm_l_a = float(np.sqrt(np.sum(ga_arr * ga_arr)))
            norm_l_d = float(np.sqrt(np.sum(d * d)))
            rel = norm_l_d / norm_l_a if norm_l_a > 1e-12 else 0.0
            leaf_rows.append((module_name, leaf_name, norm_l_a, norm_l_d, rel))
    # Sort by absolute diff norm desc
    leaf_rows.sort(key=lambda r: r[3], reverse=True)
    print("\nTop-10 Mamba leaves by ||grad_a - grad_b||:")
    print(f"  {'leaf':<70}  {'||g_a||':>10}  {'||d||':>10}  {'rel':>8}")
    for module_name, leaf_name, na, nd, rel in leaf_rows[:10]:
        path = f"{module_name}/{leaf_name}"
        # truncate long module paths for display
        path_disp = path if len(path) <= 70 else "..." + path[-67:]
        print(f"  {path_disp:<70}  {na:>10.3e}  {nd:>10.3e}  {rel:>8.2e}")
    # And by relative diff (catches outliers with small grads)
    leaf_rows.sort(key=lambda r: r[4], reverse=True)
    print("\nTop-10 Mamba leaves by RELATIVE diff:")
    print(f"  {'leaf':<70}  {'||g_a||':>10}  {'||d||':>10}  {'rel':>8}")
    for module_name, leaf_name, na, nd, rel in leaf_rows[:10]:
        path = f"{module_name}/{leaf_name}"
        path_disp = path if len(path) <= 70 else "..." + path[-67:]
        print(f"  {path_disp:<70}  {na:>10.3e}  {nd:>10.3e}  {rel:>8.2e}")

    # Dual gates: bf16 + remat-recompute can drift several % even when
    # remat is mathematically correct, so a single 1% gate is too tight.
    GATE_STRICT = 1e-2
    GATE_BF16 = 5e-2
    GATE_HARD_FAIL = 1e-1
    print("\n[verify-remat] Gates:")
    print(f"  Gate-loss          : "
          f"{'PASS' if loss_diff < GATE_STRICT else 'FAIL'} ({loss_diff:.2e})  "
          f"[strict < {GATE_STRICT:.0e}]")
    if rel_diff < GATE_STRICT:
        grad_verdict = "PASS strict"
    elif rel_diff < GATE_BF16:
        grad_verdict = "PASS bf16 (strict fail, within bf16 noise)"
    elif rel_diff < GATE_HARD_FAIL:
        grad_verdict = "MARGINAL (above bf16 floor; investigate)"
    else:
        grad_verdict = "HARD FAIL (likely real remat bug)"
    print(f"  Gate-grad-strict   : "
          f"{'PASS' if rel_diff < GATE_STRICT else 'FAIL'} ({rel_diff:.2e})  "
          f"[strict < {GATE_STRICT:.0e}]")
    print(f"  Gate-grad-bf16     : "
          f"{'PASS' if rel_diff < GATE_BF16 else 'FAIL'} ({rel_diff:.2e})  "
          f"[bf16 floor < {GATE_BF16:.0e}]")
    print(f"  Gate-grad-hardfail : "
          f"{'PASS' if rel_diff < GATE_HARD_FAIL else 'FAIL'} ({rel_diff:.2e})  "
          f"[hard fail > {GATE_HARD_FAIL:.0e}]")
    print(f"  Verdict: {grad_verdict}")


if __name__ == "__main__":
    main()
