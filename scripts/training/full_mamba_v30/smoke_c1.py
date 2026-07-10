"""v30 C1 smoke — predictor-level JIT smoke with real GC forward + event_batch.

Verifies the FULL loss path runs end-to-end:
  - GraphCast mesh2grid forward → z_pred_grid published as state
  - gather_event_patch reshapes grid-node-flat → patch-mean-pooled embedding
  - TargetEncoder encodes truth_tube_pos + truth_tube_neg
  - info_nce_loss returns finite scalar
  - grad flows into event_trunk / event_proj / event_gate / event_delta /
    event_damp / target_encoder (selectively, per adapter_scale)
  - effective_adapter_rms = 0 when adapter_scale=0

This driver does NOT touch the BPTT loop. It calls residual_loss.init +
residual_loss.apply once each with a dummy event_batch.

Reuses train_mz_v20's predictor construction by importing _build pieces.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    DirectResidualNormalizer,
)
from src.models.mamba.training.param_utils import (  # noqa: E402
    overlay_matching_params,
)
from src.data.prepared_array import PreparedArrayStore  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import _attach_temporal  # noqa: E402
from scripts.training.full_mamba_v30.event_adapter_head import (  # noqa: E402
    GCResidualWithEventAdapter,
)

# Use xarray_jax for unwrapping aux DataArrays / JaxArrayWrappers.
from graphcast import xarray_jax  # noqa: E402


def _to_float(x):
    """Best-effort scalarize: handles (value, weight) tuples, xarray DataArrays,
    JaxArrayWrappers, jax/np arrays, and Python scalars."""
    if isinstance(x, tuple) and len(x) >= 1:
        x = x[0]
    try:
        x = xarray_jax.unwrap_data(x)
    except Exception:
        pass
    if hasattr(x, "values"):
        x = x.values
    arr = np.asarray(jax.device_get(x))
    if arr.shape == ():
        return float(arr)
    return float(np.mean(arr))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--residual-root", required=True)
    p.add_argument("--ckpt-in", required=True)
    p.add_argument("--init-from-residual-ckpt", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--mesh-size", type=int, default=4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-msg-steps", type=int, default=6)
    p.add_argument("--residual-msg-steps", type=int, default=2)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=128)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--temporal-bias", action="store_true")
    p.add_argument("--temporal-conv-bias", action="store_true", default=True)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--temporal-zero-init-out", action="store_true", default=True)
    # v30 knobs
    p.add_argument("--adapter-scale", type=float, default=0.0)
    p.add_argument("--lambda-con", type=float, default=0.01)
    p.add_argument("--lambda-delta-reg", type=float, default=0.0)
    p.add_argument("--tau-contrastive", type=float, default=0.1)
    p.add_argument("--event-patch-size", type=int, default=16)
    p.add_argument("--event-trunk-dim", type=int, default=64)
    p.add_argument("--event-proj-dim", type=int, default=64)
    p.add_argument("--gate-bias-init", type=float, default=-2.0)
    p.add_argument("--damp-bias-init", type=float, default=-4.0)
    p.add_argument("--delta-cap", type=float, default=0.5)
    p.add_argument("--n-neg", type=int, default=4,
                   help="Number of negative tubes per positive.")
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    return p.parse_args()


def main():
    cfg = parse()

    # ---- ckpt + task_cfg ----
    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)

    model_cfg_residual = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.residual_msg_steps,
    )

    # ---- data store ----
    store = PreparedArrayStore(Path(cfg.prepared_root), label="c1-source")
    residual_root = Path(cfg.residual_root)
    meta = json.loads((residual_root / "metadata.json").read_text())
    target_vars = list(meta["target_variables"])
    anchor_indices = np.load(residual_root / "anchors" / "anchor_indices.npy")
    train_split = np.load(residual_root / "anchors" / "split_train.npy")

    norm_stats = base_train.load_stats(Path(cfg.stats_dir))
    base_train.validate_stats_coverage(task_cfg, norm_stats)
    use_bf16 = cfg.precision == "bf16"

    def _build_residual_predictor():
        p = GCResidualWithEventAdapter(
            model_cfg_residual, task_cfg,
            adapter_scale=cfg.adapter_scale,
            trunk_dim=cfg.event_trunk_dim,
            proj_dim=cfg.event_proj_dim,
            gate_bias_init=cfg.gate_bias_init,
            damp_bias_init=cfg.damp_bias_init,
            delta_cap=cfg.delta_cap,
            lambda_con=cfg.lambda_con,
            lambda_delta_reg=cfg.lambda_delta_reg,
            tau_contrastive=cfg.tau_contrastive,
            force_init_target_encoder=False,   # we'll pass real event_batch
            event_patch_size=cfg.event_patch_size,
        )
        _attach_temporal(p, cfg)
        if use_bf16:
            p = casting.Bfloat16Cast(p)
        p = DirectResidualNormalizer(
            p,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"],
        )
        return p

    def residual_loss_fn(inputs, targets, forcings, is_training, event_batch=None):
        del is_training
        return _build_residual_predictor().loss(
            inputs, targets, forcings, event_batch=event_batch)

    residual_loss = hk.transform_with_state(residual_loss_fn)

    # FROZEN baseline GC1 — needed to compute residual target = truth - baseline.
    model_cfg_baseline = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.baseline_msg_steps,
    )
    def _build_baseline_predictor():
        p = gc.GraphCast(model_cfg_baseline, task_cfg)
        if use_bf16:
            p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"],
        )
        return p
    def baseline_pred_fn(inputs, targets_template, forcings):
        return _build_baseline_predictor()(
            inputs, targets_template=targets_template, forcings=forcings)
    baseline_predict = hk.transform_with_state(baseline_pred_fn)

    # ---- sample batch ----
    rng = jax.random.PRNGKey(cfg.seed)
    dt = pd.Timedelta(
        np.diff(np.asarray(store.time.values).astype("datetime64[ns]"))[0])
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    sample_anchor = int(anchor_indices[int(train_split[0])])
    sample_inputs, sample_targets, sample_forcings = store.build_batch_from_indices(
        indices=[sample_anchor],
        input_steps=input_steps,
        target_steps=1,
        task_cfg=task_cfg,
        dt=dt,
    )

    # ---- dummy event_batch (pos + neg with right shapes) ----
    P = int(cfg.event_patch_size)
    B = sample_inputs.sizes.get("batch", 1)
    Nneg = int(cfg.n_neg)
    C_vars = 4   # 2m_T, u10, v10, mslp surface vars  (matches TargetEncoder C)
    T_win = 4
    rng_np = np.random.default_rng(cfg.seed)
    event_batch = dict(
        truth_tube_pos=jnp.asarray(
            rng_np.standard_normal((B, T_win, P, P, C_vars)).astype(np.float32)),
        truth_tube_neg=jnp.asarray(
            rng_np.standard_normal((B, Nneg, T_win, P, P, C_vars)).astype(np.float32)),
        center_lat_idx=jnp.asarray(
            rng_np.integers(P, sample_inputs.sizes["lat"] - P, size=(B,)),
            dtype=jnp.int32),
        center_lon_idx=jnp.asarray(
            rng_np.integers(0, sample_inputs.sizes["lon"], size=(B,)),
            dtype=jnp.int32),
    )
    print(f"[C1] event_batch shapes: "
          f"pos={tuple(event_batch['truth_tube_pos'].shape)} "
          f"neg={tuple(event_batch['truth_tube_neg'].shape)} "
          f"center_lat={tuple(event_batch['center_lat_idx'].shape)}")

    # ---- init residual ----
    rng, k = jax.random.split(rng)
    residual_params, residual_state = residual_loss.init(
        k, sample_inputs, sample_targets, sample_forcings, True,
        event_batch=event_batch)

    # Overlay v22 K=18 weights so Mamba+base_head are realistic.
    residual_params, r_stats = overlay_matching_params(
        residual_params, ckpt_in.params, strict=False)
    print(f"[C1] DM overlay: {r_stats.copied} copied, {r_stats.initialized} fresh")

    # ---- init + load frozen baseline GC1 ----
    rng, kb = jax.random.split(rng)
    baseline_params, baseline_state = baseline_predict.init(
        kb, sample_inputs, sample_targets, sample_forcings)
    baseline_params, b_stats = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    print(f"[C1] baseline (FROZEN): {b_stats.copied} copied from DM ckpt")

    # ---- compute residual target = truth - baseline_pred ----
    rng, kpb = jax.random.split(rng)
    baseline_pred, _ = baseline_predict.apply(
        baseline_params, baseline_state, kpb,
        sample_inputs, sample_targets, sample_forcings)
    residual_targets = jax.tree_util.tree_map(
        lambda t, b: t - jax.lax.stop_gradient(b),
        sample_targets, baseline_pred)
    print("[C1] residual targets = truth - baseline_pred computed.")

    # Overlay v22 K=18 residual ckpt
    import pickle
    with Path(cfg.init_from_residual_ckpt).open("rb") as f:
        src = pickle.load(f)
    src_params = src["residual_params"]
    overlaid, skipped = 0, 0
    def _merge(dst, srcd, path=()):
        nonlocal overlaid, skipped
        if isinstance(dst, dict):
            out = {}
            for kk, v in dst.items():
                if isinstance(srcd, dict) and kk in srcd:
                    out[kk] = _merge(v, srcd[kk], path + (kk,))
                else:
                    out[kk] = v
                    if hasattr(v, "shape"): skipped += 1
            return out
        else:
            if hasattr(srcd, "shape") and srcd.shape == dst.shape:
                overlaid += 1; return srcd
            else:
                skipped += 1; return dst
    residual_params = _merge(residual_params, src_params)
    print(f"[C1] v22 K=18 overlay: {overlaid} leaves overlaid, {skipped} kept (event heads + encoder)")

    # ---- one forward + grad ----
    from src.models.graphcast.training.core.model import scalarize_loss
    def loss_fn(params, state, rng_in, inputs, targets, forcings, event_batch):
        (loss_da, aux), new_state = residual_loss.apply(
            params, state, rng_in, inputs, targets, forcings, True,
            event_batch=event_batch)
        return scalarize_loss(loss_da), (aux, new_state)

    rng, k = jax.random.split(rng)
    print(f"[C1] JIT compile + first forward (with residual targets)...")
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    (loss_scalar, (aux, _)), grads = grad_fn(
        residual_params, residual_state, k,
        sample_inputs, residual_targets, sample_forcings, event_batch)
    loss_v = _to_float(loss_scalar)
    print(f"[C1] total loss (scalarized, normalized) = {loss_v:.6f}")

    # ---- aux readout ----
    print(f"\n[C1] aux scalars (key v30 signals):")
    interesting_keys = [
        "aux_L_con", "aux_con_pos_sim", "aux_con_neg_sim",
        "aux_con_gap", "aux_con_top1",
        "aux_gate_mean", "aux_damp_mean",
        "aux_delta_rms", "aux_adapter_rms", "aux_effective_adapter_rms",
    ]
    for k_aux in interesting_keys:
        if k_aux in aux:
            try:
                v = _to_float(aux[k_aux])
                print(f"   {k_aux:32s} = {v:>14.6f}")
            except Exception as e:
                print(f"   {k_aux:32s} = (could not extract: {e})")
        else:
            print(f"   {k_aux:32s} = (MISSING)")

    print(f"\n[C1] per-variable weighted-MSE aux (from weighted_mse_per_level):")
    for k_aux in sorted(aux.keys()):
        if k_aux.startswith("aux_"): continue
        try:
            v = _to_float(aux[k_aux])
            print(f"   {k_aux:32s} = {v:>14.6f}")
        except Exception as e:
            print(f"   {k_aux:32s} = (could not extract: {e})")

    # ---- grad norm per parameter group ----
    print(f"\n[C1] gradient norms by group:")
    groups = {
        "mamba_temporal":       lambda p: "temporal_mesh_mamba" in p or "mesh_interleaved_temporal" in p,
        "base_residual_head":   lambda p: "temporal_residual_head" in p,
        "event_trunk":          lambda p: "temporal_event_trunk" in p,
        "event_proj":           lambda p: "temporal_event_proj" in p,
        "event_gate":           lambda p: "temporal_event_gate" in p,
        "event_delta":          lambda p: "temporal_event_delta" in p,
        "event_damp":           lambda p: "temporal_event_damp" in p,
        "target_encoder":       lambda p: "temporal_event_target_encoder" in p,
        "graphcast_other":      lambda p: not any(s in p for s in [
            "temporal_mesh_mamba", "mesh_interleaved_temporal",
            "temporal_residual_head", "temporal_event_"]),
    }
    def _walk(tree, path=()):
        if isinstance(tree, dict):
            for kk, v in tree.items():
                yield from _walk(v, path + (kk,))
        else:
            yield "/".join(path), tree
    flat = list(_walk(grads))
    for gname, predicate in groups.items():
        total = 0.0
        n_leaves = 0
        for path, leaf in flat:
            if predicate(path) and hasattr(leaf, "shape"):
                total += float(jnp.sum(leaf * leaf))
                n_leaves += 1
        norm = total ** 0.5
        print(f"   {gname:24s} L2={norm:>12.4e}  (leaves={n_leaves})")

    print(f"\n[C1 sanity expectations]")
    def _check(name, ok, val=None):
        tag = "PASS" if ok else "FAIL"
        if val is not None:
            print(f"   {name:35s} : {tag}  ({val})")
        else:
            print(f"   {name:35s} : {tag}")
    _check("loss finite", np.isfinite(loss_v), val=f"{loss_v:.6f}")

    if "aux_L_con" in aux:
        L_con_val = _to_float(aux["aux_L_con"])
        # log(1+Nneg) baseline; allow generous range since tau=0.1 sharpens softmax.
        Nneg = int(cfg.n_neg)
        log_baseline = float(np.log(1 + Nneg))
        _check("aux_L_con finite + nonzero",
               np.isfinite(L_con_val) and L_con_val > 0,
               val=f"{L_con_val:.4f} (random-init ref log(1+Nneg)={log_baseline:.4f})")
    else:
        _check("aux_L_con present", False)

    if "aux_effective_adapter_rms" in aux:
        eff = _to_float(aux["aux_effective_adapter_rms"])
        _check("effective_adapter_rms = 0", eff == 0.0, val=f"{eff:.4e}")

    if "aux_gate_mean" in aux:
        gm = _to_float(aux["aux_gate_mean"])
        # init bias=-2 → sigmoid(-2)=0.119
        _check("aux_gate_mean ≈ 0.119", abs(gm - 0.119) < 0.02, val=f"{gm:.4f}")

    if "aux_damp_mean" in aux:
        dm = _to_float(aux["aux_damp_mean"])
        # init bias=-4 → sigmoid(-4)=0.018
        _check("aux_damp_mean ≈ 0.018", abs(dm - 0.018) < 0.005, val=f"{dm:.4f}")


if __name__ == "__main__":
    main()
