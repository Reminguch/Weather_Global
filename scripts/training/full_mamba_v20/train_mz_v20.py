"""v20 = v18 baseline-stream residual + fixed AR forcings index.

CRITICAL FIX vs v18: in the AR feedback loop, when we shift the input window
forward by 1 step, the input_forcings at the new time slot (= time of the
just-predicted state) must be the *just-used* forcings_list[i] -- the same
target_forcings that produced baseline_pred -- NOT forcings_list[i+1]
(which is the target_forcings for the next step, sitting one tick into the
future). v18 trained the model on systematically off-by-one forcings
(solar-zenith / day-of-year / etc. shifted by 6h at every AR step).

NOTE: rest of training is identical to v18.

Original v18 docstring follows:



v17 (Two-stream A) failed because Mamba's input (corrected state) and
Mamba's target (truth - GC(baseline_state)) lived on DIFFERENT trajectories
that diverged with K, breaking input-target relevance. v18 fixes this by
making BOTH GC and Mamba see the SAME baseline-self-rollout input:

  current_inputs_{i+1} = shift(current_inputs_i, baseline_pred_i)
  baseline_pred_i      = GC(current_inputs_i)
  residual_pred_i      = Mamba(current_inputs_i)
  target_i             = truth_i - stop_gradient(baseline_pred_i)
  full_pred_i          = baseline_pred_i + residual_pred_i

This is "frozen GC + residual corrector" done properly:
  * GC pure self-rollout (no Mamba pollution).
  * Mamba sees the same GC-self-rollout state -> input matches target context.
  * Mamba's hidden state still propagates corrections across AR steps,
    so memory of past corrections is preserved IMPLICITLY via state, just
    not via explicit input.

K=1 (ar-tail-K=0) is BYTE-IDENTICAL to v15 v2 / v16 / v17.

What's reused:
  - GCResidualWithZeroHead, _attach_temporal from v9 (model + Mamba config)

What's reused:
  - GCResidualWithZeroHead, _attach_temporal from v9 (model + Mamba config)
  - PreparedArrayStore from src/data/prepared_array.py (zarr -> memmap)
  - Precomputed residual_target memmaps from precompute_residual_targets.py
    (saved at /scratch/.../precomputed_residuals/v11_setup_res1/residuals/{var}.npy)
  - DirectResidualNormalizer wrapping the residual model

What's new:
  - Train step skips baseline_predict_transform.apply (would re-run frozen
    DeepMind GC1 every BPTT step). Instead loads residual_target from disk.
  - Loads inputs/forcings via PreparedArrayStore.build_batch_from_indices,
    which is numpy-memmap-backed (no xarray/zarr in the inner loop).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    DirectResidualNormalizer, scalarize_loss,
)
from src.models.mamba.training.param_utils import (  # noqa: E402
    overlay_matching_params,
)
# Prepared-array support moved with the GraphCast training core.  v20 uses
# exactly the same public store API (memmap batches/metadata validation).
from src.models.graphcast.training.core.prepared_array import PreparedArrayStore  # noqa: E402

from scripts.training.full_mamba_v9.train_mz_v9 import (  # noqa: E402
    GCResidualWithZeroHead, _attach_temporal,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="v13 = v9 arch + fast data pipeline (precomputed residuals).")
    p.add_argument("--prepared-root", required=True,
                   help="Path to prepared_stream/res<N>")
    p.add_argument("--residual-root", required=True,
                   help="Path to precomputed_residuals/<setup>/")
    p.add_argument("--ckpt-in", default=base_train.DEFAULT_CKPT)
    p.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512,
                   help="GC2 latent_size (DeepMind GC_small uses 512)")
    p.add_argument("--baseline-msg-steps", type=int, default=16,
                   help="frozen baseline GC1 mesh GNN msg steps")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Clip global gradient norm to this. 0 disables.")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear LR warmup from 0 to --lr over this many steps.")
    p.add_argument("--residual-msg-steps", type=int, default=2,
                   help="GC2 mesh_gnn processor depth (v9 used 2)")
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--sequential-segment-steps", type=int, default=32)
    p.add_argument("--bptt-steps", type=int, default=8)
    p.add_argument("--ar-tail-K", type=int, default=None,
                   help="Number of tail anchors that use closed-loop AR. The "
                        "first (bptt - K) anchors use REAL ERA5 inputs (truth-"
                        "anchored), the last K anchors feed back model preds. "
                        "K=0 means all truth (= v15 v2). K=bptt-1 means only "
                        "anchor 1 is truth, rest are AR (= original v16, default).")
    p.add_argument("--temporal-location",
                   choices=["mesh_post_encoder", "mesh_processor_interleaved"],
                   default="mesh_processor_interleaved",
                   help="Where Mamba sits inside GC2 (v9 used interleaved)")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=None,
                   help="Mamba SSM inner dim. None -> hidden_size")
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--temporal-bias", action="store_true", default=False)
    p.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias",
                   action="store_false", default=True)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--no-zero-init-out", dest="temporal_zero_init_out",
                   action="store_false", default=True)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--start-step", type=int, default=1)
    # Feedback mode for AR-tail trajectory advancement.
    #   baseline       (default, v22 open-loop): cur_{t+1} <- shift(cur_t, bp_t)
    #   closed_loop_sg                          : cur_{t+1} <- shift(cur_t, bp_t + sg(rp_t))
    # The latter exposes Mamba SSM to closed-loop input distribution during training
    # while stop_gradient on rp_t keeps memory/BPTT graph identical to open-loop.
    p.add_argument("--feedback-mode",
                   choices=["baseline", "closed_loop_sg"], default="baseline")
    # --- Gated extreme residual boost head (v25) ---
    # When enabled, the residual head is split into:
    #   r = r_base + sigmoid(gate) * r_ext
    # r_base is the existing residual head (overlaid from --init-from-residual-ckpt if
    # given). r_ext and gate are new heads. Loss adds BCE(gate, |y-bp|>q95) + λ·E[gate].
    p.add_argument("--gated-extreme-head", action="store_true",
                   help="Enable gated extreme residual boost head.")
    p.add_argument("--init-from-residual-ckpt", default=None,
                   help="Path to existing residual ckpt (.pkl) to overlay Mamba + "
                        "base_head from. Ext/gate heads stay at fresh init.")
    p.add_argument("--gate-phase", choices=["1", "2"], default="1",
                   help="Phase 1: freeze all except ext+gate heads. Phase 2: train all.")
    p.add_argument("--gate-bias-init", type=float, default=-3.5)
    p.add_argument("--gate-beta", type=float, default=0.05,
                   help="BCE weight on gate.")
    p.add_argument("--gate-lambda-g", type=float, default=0.003,
                   help="Sparsity weight on gate (E[gate]).")
    p.add_argument("--gate-extreme-quantile", type=float, default=0.95,
                   help="Per-variable quantile of |y-bp| for extreme mask.")
    p.add_argument("--gate-alpha-extreme", type=float, default=0.0,
                   help="Alpha-weighted MSE on extreme cells. 0 disables (only BCE+sparsity).")
    # Phase 1.5: separate LR for new heads vs base/backbone.
    # If --ext-head-lr (or --gate-head-lr) is set, those params get that LR;
    # rest of params use --lr. Implemented via optax.multi_transform.
    p.add_argument("--ext-head-lr", type=float, default=None,
                   help="Separate LR for temporal_residual_ext_head. None = use --lr.")
    p.add_argument("--gate-head-lr", type=float, default=None,
                   help="Separate LR for temporal_residual_gate_head. None = use --lr.")
    # --- v26 Physical Tail Calibration Head ---
    # r_final = r_base + Delta_tail, where Delta_tail is bounded tail correction
    # gated by physical anomaly masks. Only 2m_T + 10m wind in v1.
    p.add_argument("--tail-calib-head", action="store_true",
                   help="Enable v26 Physical Tail Calibration Head.")
    p.add_argument("--tail-alpha", type=float, default=1.0)
    p.add_argument("--tail-lambda-quiet", type=float, default=0.01)
    p.add_argument("--tail-cap-T", type=float, default=0.5)
    p.add_argument("--tail-cap-wind", type=float, default=0.5)
    p.add_argument("--tail-z95", type=float, default=1.64)
    p.add_argument("--tail-tau", type=float, default=0.5)
    p.add_argument("--tail-head-lr", type=float, default=None,
                   help="Separate LR for temporal_tail_calibration_head. None = use --lr.")
    # --- v30 Contrastive Event Adapter Head ---
    # r_final = r_base + adapter_scale * (g * delta - damp * r_base)
    # New small heads (trunk/proj/gate/delta/damp + target_encoder) train via
    # weighted-MSE + optional InfoNCE on event-tube contrastive matching.
    p.add_argument("--event-adapter-head", action="store_true",
                   help="Enable v30 Contrastive Event Adapter Head.")
    p.add_argument("--adapter-scale", type=float, default=0.0,
                   help="Scale on adapter contribution to r_final. 0 = byte-identical v22.")
    p.add_argument("--event-trunk-dim", type=int, default=64)
    p.add_argument("--event-proj-dim", type=int, default=64)
    p.add_argument("--event-gate-bias-init", type=float, default=-2.0)
    p.add_argument("--damp-bias-init", type=float, default=-4.0,
                   help="Bias init for temporal_event_damp head. -4 => damp~0.018 at init.")
    p.add_argument("--event-delta-cap", type=float, default=0.5)
    p.add_argument("--lambda-con", type=float, default=0.0,
                   help="InfoNCE weight. 0 = contrastive off.")
    p.add_argument("--lambda-delta-reg", type=float, default=0.0,
                   help="Amplitude regularizer on effective_adapter (scale*(g*delta - damp*r_base)).")
    p.add_argument("--tau-contrastive", type=float, default=0.1)
    p.add_argument("--force-init-target-encoder", action="store_true",
                   help="Run target_encoder with 0-weight at training graph init so its "
                        "params live in the tree even when lambda_con=0 (lets you smoke "
                        "with lambda_con=0 then resume with lambda_con>0).")
    p.add_argument("--event-head-lr", type=float, default=None,
                   help="Separate LR for event adapter heads (trunk/proj/gate/delta/damp).")
    p.add_argument("--event-target-encoder-lr", type=float, default=None,
                   help="Separate LR for temporal_event_target_encoder/*.")
    p.add_argument("--event-patch-size", type=int, default=16,
                   help="Static patch size for event contrastive tubes. Must "
                        "match target_encoder's expected spatial input. Static "
                        "Python int — not part of event_batch pytree.")
    p.add_argument("--contrastive-anchor-idx", type=int, default=-1,
                   help="Which BPTT anchor gets event_batch for InfoNCE. "
                        "-1 = last anchor (bptt-1). Only one anchor gets it per "
                        "BPTT chunk to keep z_pred / truth_tube time-aligned.")
    p.add_argument("--n-neg", type=int, default=4,
                   help="Number of negative tubes per positive.")
    p.add_argument("--event-dummy", action="store_true",
                   help="Use dummy random event_batch (C2 smoke). Real event "
                        "sampling is Plan B.")
    p.add_argument("--event-t-win", type=int, default=1,
                   help="Time-window length for event tubes (frames of 6h "
                        "each). B0 default 1. Encoder input dim scales with T.")
    p.add_argument("--event-c-vars", type=int, default=4,
                   help="Number of surface vars used by target encoder. Must "
                        "match the sampler's DEFAULT_VARIABLES count (default 4: "
                        "2m_T, u10, v10, mslp).")
    p.add_argument("--event-min-time-gap-days", type=int, default=7,
                   help="B0 negative sampling: minimum |dt| in days between "
                        "positive and negative anchors.")
    return p.parse_args()


def _load_residual_metadata(residual_root: Path) -> dict:
    return json.loads((residual_root / "metadata.json").read_text())


def _load_anchor_split(residual_root: Path):
    anchor_indices = np.load(residual_root / "anchors" / "anchor_indices.npy")
    train_split = np.load(residual_root / "anchors" / "split_train.npy")
    val_split = np.load(residual_root / "anchors" / "split_val.npy")
    return anchor_indices, train_split, val_split


def _open_residual_memmaps(residual_root: Path, target_vars: list[str]) -> dict:
    return {
        v: np.load(residual_root / "residuals" / f"{v}.npy", mmap_mode="r")
        for v in target_vars
    }


def _build_residual_target_xr(
    residual_memmaps: dict,
    target_template: xr.Dataset,
    *,
    indices_in_residual: np.ndarray,
) -> xr.Dataset:
    out_vars = {}
    for v, arr in residual_memmaps.items():
        gathered = np.take(arr, indices_in_residual, axis=0)  # [B, T, lat, lon, level?]
        tmpl = target_template[v]
        out_vars[v] = xr.DataArray(
            gathered.astype("float32"),
            dims=tmpl.dims,
            coords={d: tmpl.coords[d] for d in tmpl.dims if d in tmpl.coords},
        )
    return xr.Dataset(out_vars, coords=target_template.coords)


def _build_segments(split_indices: np.ndarray, seg_len: int) -> list[np.ndarray]:
    segments = []
    n = len(split_indices)
    for s in range(0, n - seg_len + 1, seg_len):
        segments.append(split_indices[s:s + seg_len])
    return segments


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Checkpoints + task config ----------
    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)

    # GC2 (residual model) config: full DeepMind topology, just shorter
    # processor (2 msg steps vs DeepMind's 16).
    model_cfg_residual = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.residual_msg_steps,
    )
    print(f"[v13] GC2 (residual) latent_size={cfg.width} "
          f"msg_steps={cfg.residual_msg_steps}  (Mamba {cfg.temporal_location})")
    print(f"[v13] Mamba: hidden={cfg.temporal_hidden_size} "
          f"d_inner={cfg.temporal_d_inner} d_state={cfg.temporal_d_state} "
          f"d_conv={cfg.temporal_d_conv} layers={cfg.temporal_layers}")

    # ---------- 2) Open prepared store + residual memmaps ----------
    prepared_root = Path(cfg.prepared_root)
    residual_root = Path(cfg.residual_root)
    store = PreparedArrayStore(prepared_root, label="v13-source")
    store.validate(resolution=cfg.resolution, task_cfg=task_cfg)

    res_meta = _load_residual_metadata(residual_root)
    target_vars = res_meta["target_variables"]
    if list(task_cfg.target_variables) != list(target_vars):
        raise ValueError("target_variables mismatch with precomputed residuals")
    anchor_indices, train_split, val_split = _load_anchor_split(residual_root)
    residual_memmaps = _open_residual_memmaps(residual_root, target_vars)
    print(f"[v13] {len(target_vars)} target vars, {anchor_indices.size} total anchors "
          f"(train={train_split.size}, val={val_split.size})")

    # ---------- 3) Build residual loss transform ----------
    norm_stats = base_train.load_stats(Path(cfg.stats_dir))
    base_train.validate_stats_coverage(task_cfg, norm_stats)
    use_bf16 = cfg.precision == "bf16"

    # Mutex check: at most one auxiliary head at a time.
    _n_heads = sum([bool(cfg.gated_extreme_head),
                    bool(cfg.tail_calib_head),
                    bool(cfg.event_adapter_head)])
    if _n_heads > 1:
        raise ValueError(
            "Choose at most one of --gated-extreme-head / --tail-calib-head / "
            "--event-adapter-head.")

    def _build_residual_predictor():
        if cfg.tail_calib_head:
            from scripts.training.full_mamba_v26.tail_calibration_head import (
                GCResidualWithTailCalibrationHead)
            p = GCResidualWithTailCalibrationHead(
                model_cfg_residual, task_cfg,
                alpha_tail=cfg.tail_alpha,
                lambda_quiet=cfg.tail_lambda_quiet,
                cap_T=cfg.tail_cap_T,
                cap_wind=cfg.tail_cap_wind,
                z95=cfg.tail_z95,
                tau=cfg.tail_tau,
            )
        elif cfg.gated_extreme_head:
            from scripts.training.full_mamba_v25.gated_extreme_head import (
                GCResidualWithGatedExtremeHead)
            p = GCResidualWithGatedExtremeHead(
                model_cfg_residual, task_cfg,
                gate_bias_init=cfg.gate_bias_init,
                beta=cfg.gate_beta,
                lambda_g=cfg.gate_lambda_g,
                extreme_quantile=cfg.gate_extreme_quantile,
                alpha_extreme=cfg.gate_alpha_extreme,
            )
        elif cfg.event_adapter_head:
            from scripts.training.full_mamba_v30.event_adapter_head import (
                GCResidualWithEventAdapter)
            p = GCResidualWithEventAdapter(
                model_cfg_residual, task_cfg,
                adapter_scale=cfg.adapter_scale,
                trunk_dim=cfg.event_trunk_dim,
                proj_dim=cfg.event_proj_dim,
                gate_bias_init=cfg.event_gate_bias_init,
                damp_bias_init=cfg.damp_bias_init,
                delta_cap=cfg.event_delta_cap,
                lambda_con=cfg.lambda_con,
                lambda_delta_reg=cfg.lambda_delta_reg,
                tau_contrastive=cfg.tau_contrastive,
                force_init_target_encoder=cfg.force_init_target_encoder,
                event_patch_size=cfg.event_patch_size,
            )
        else:
            p = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
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

    def residual_loss_fn(inputs, residual_targets, forcings, is_training,
                         event_batch=None):
        del is_training
        predictor = _build_residual_predictor()
        # v30 head accepts event_batch kwarg; older heads ignore it. We dispatch
        # based on whether the predictor (or its DirectResidualNormalizer wrapper)
        # advertises support. For A0/A1 smoke, event_batch is None and the call
        # reduces to the v22 path.
        if cfg.event_adapter_head:
            # DirectResidualNormalizer forwards kwargs to .loss/.loss_and_predictions
            # only if the inner predictor's .loss accepts them. v30's loss does.
            return predictor.loss(inputs, residual_targets, forcings,
                                  event_batch=event_batch)
        return predictor.loss(inputs, residual_targets, forcings)

    def residual_pred_fn(inputs, targets_template, forcings):
        return _build_residual_predictor()(
            inputs, targets_template=targets_template, forcings=forcings)

    residual_loss = hk.transform_with_state(residual_loss_fn)
    residual_predict = hk.transform_with_state(residual_pred_fn)

    # --- v16: frozen baseline GC1 for live forward in AR chain ---
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

    # ---------- 4) Init params (overlay DeepMind weights) ----------
    rng = jax.random.PRNGKey(cfg.seed)
    dt = pd.Timedelta(np.diff(np.asarray(store.time.values).astype("datetime64[ns]"))[0])
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    target_steps = cfg.target_steps
    if target_steps != 1:
        raise ValueError(
            f"v18 bptt-chunk AR expects --target-steps=1 (each anchor predicts "
            f"one step ahead); use --ar-tail-K to control AR-fed anchors instead. "
            f"Got --target-steps={target_steps}.")

    sample_anchor = int(anchor_indices[int(train_split[0])])
    sample_inputs, sample_targets, sample_forcings = store.build_batch_from_indices(
        indices=[sample_anchor],
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt,
    )
    rng, k = jax.random.split(rng)
    # v30: when force_init_target_encoder is set we must pass a dummy event_batch
    # at init time so the target_encoder gets traced and its params land in the
    # Haiku tree (otherwise resume with lambda_con>0 would hit shape mismatches).
    _init_kwargs = {}
    # Auto-force init of target_encoder whenever InfoNCE will be used at apply
    # time (lambda_con>0). Otherwise the encoder module wouldn't be traced
    # during init → its params won't exist → apply-time forward would fail with
    # "Unable to retrieve parameter 'w' for module temporal_event_target_encoder".
    _need_init_encoder = cfg.event_adapter_head and (
        cfg.force_init_target_encoder or cfg.lambda_con > 0.0)
    if _need_init_encoder:
        _b = sample_inputs.sizes.get("batch", 1)
        _P = int(cfg.event_patch_size)
        _T = int(cfg.event_t_win)
        _C = int(cfg.event_c_vars)
        _Nneg_init = int(getattr(cfg, "n_neg", 1)) if cfg.lambda_con > 0 else 1
        # patch_size / T_win / C are STATIC on the predictor (jit safety).
        _init_batch = dict(
            truth_tube_pos=jnp.zeros((_b, _T, _P, _P, _C), dtype=jnp.float32),
            center_lat_idx=jnp.zeros((_b,), dtype=jnp.int32),
            center_lon_idx=jnp.zeros((_b,), dtype=jnp.int32),
        )
        if cfg.lambda_con > 0.0:
            _init_batch["truth_tube_neg"] = jnp.zeros(
                (_b, _Nneg_init, _T, _P, _P, _C), dtype=jnp.float32)
        _init_kwargs["event_batch"] = _init_batch
        print(f"[v30] dummy event_batch threaded into init "
              f"(B={_b}, T={_T}, P={_P}, C={_C}, "
              f"Nneg={_Nneg_init if cfg.lambda_con>0 else 'N/A'}) "
              f"to populate temporal_event_target_encoder/* params.")
    residual_params, residual_state = residual_loss.init(
        k, sample_inputs, sample_targets, sample_forcings, True,
        **_init_kwargs)
    _, _residual_state_for_pred = residual_predict.init(
        k, sample_inputs, sample_targets, sample_forcings)
    # Overlay DeepMind GraphCast weights into the residual_params for matching
    # leaves (encoder/processor/decoder). Lenient because GC1 has 16 msg steps
    # but our GC2 has only `residual_msg_steps`.
    residual_params, r_stats = overlay_matching_params(
        residual_params, ckpt_in.params, strict=False)
    n_residual = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[v16] residual: overlaid {r_stats.copied} GC params from DeepMind, "
          f"{r_stats.initialized} fresh (Mamba + residual_head). "
          f"Total trainable: {n_residual:,}")

    # --- v16: init FROZEN baseline params from DeepMind ckpt ---
    rng, kb = jax.random.split(rng)
    baseline_params, baseline_state = baseline_predict.init(
        kb, sample_inputs, sample_targets, sample_forcings)
    baseline_params, b_stats = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    n_baseline = sum(p.size for p in jax.tree_util.tree_leaves(baseline_params))
    print(f"[v16] baseline (FROZEN): {n_baseline:,} params loaded from DeepMind ckpt")

    if cfg.resume_from is not None:
        import pickle
        with Path(cfg.resume_from).open("rb") as f:
            ck = pickle.load(f)
        residual_params = ck["residual_params"]
        if "residual_state" in ck and ck["residual_state"]:
            residual_state = ck["residual_state"]
        print(f"[v13] resumed from {cfg.resume_from} at step {cfg.start_step}")

    # --- v25/v26/v30: overlay existing residual ckpt's Mamba + base residual head into
    # the new model's params. New heads (ext/gate for v25, tail for v26, event adapter
    # for v30) stay at fresh init since they don't exist in the source ckpt.
    if ((cfg.gated_extreme_head or cfg.tail_calib_head or cfg.event_adapter_head)
            and cfg.init_from_residual_ckpt):
        import pickle
        with Path(cfg.init_from_residual_ckpt).open("rb") as f:
            src = pickle.load(f)
        src_params = src["residual_params"]
        # Walk both param trees; for each leaf in residual_params, if a leaf
        # with matching (module path, leaf name) exists in src_params and has
        # the same shape, overlay it. Specifically catches:
        #   - all GraphCast encoder/processor/decoder
        #   - all Mamba modules (mesh_interleaved_temporal_*)
        #   - temporal_residual_head (the base head)
        # ext_head and gate_head do NOT exist in src — left at fresh init.
        overlaid, skipped = 0, 0
        def _merge(dst_tree, src_tree, path=()):
            nonlocal overlaid, skipped
            if isinstance(dst_tree, dict):
                out = {}
                for k, v in dst_tree.items():
                    if isinstance(src_tree, dict) and k in src_tree:
                        out[k] = _merge(v, src_tree[k], path + (k,))
                    else:
                        out[k] = v
                        if hasattr(v, "shape"):
                            skipped += 1
                return out
            else:
                # Leaf
                if hasattr(src_tree, "shape") and src_tree.shape == dst_tree.shape:
                    overlaid += 1
                    return src_tree
                else:
                    skipped += 1
                    return dst_tree
        residual_params = _merge(residual_params, src_params)
        # State: overlay matching keys ONLY (don't drop our new gate__<var> state).
        if "residual_state" in src and src["residual_state"]:
            residual_state = _merge(residual_state, src["residual_state"])
        print(f"[v25/v26/v30] overlaid {overlaid} leaves from {cfg.init_from_residual_ckpt}, "
              f"{skipped} kept at fresh init (new auxiliary heads).")

    # LR schedule with optional warmup
    if cfg.warmup_steps > 0:
        lr_schedule = optax.warmup_constant_schedule(
            init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup_steps)
    else:
        lr_schedule = cfg.lr
    # Optimizer chain: optional grad clipping then AdamW.
    # If any *-head-lr is set, use optax.multi_transform to apply per-head LRs
    # against the rest of the params.
    use_multi_lr = ((cfg.ext_head_lr is not None) or (cfg.gate_head_lr is not None)
                    or (cfg.tail_head_lr is not None)
                    or (cfg.event_head_lr is not None)
                    or (cfg.event_target_encoder_lr is not None))
    if use_multi_lr:
        ext_lr  = cfg.ext_head_lr  if cfg.ext_head_lr  is not None else cfg.lr
        gate_lr = cfg.gate_head_lr if cfg.gate_head_lr is not None else cfg.lr
        tail_lr = cfg.tail_head_lr if cfg.tail_head_lr is not None else cfg.lr
        # v30: group all event adapter heads (trunk/proj/gate/delta/damp) into one
        # label, target_encoder into another — per Step-1 review, don't split 5 LRs.
        event_lr     = cfg.event_head_lr           if cfg.event_head_lr           is not None else cfg.lr
        event_enc_lr = cfg.event_target_encoder_lr if cfg.event_target_encoder_lr is not None else cfg.lr
        def _label_path(path_tuple):
            full = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path_tuple)
            # v30 labels (must come BEFORE generic substring matches that share prefixes).
            if "temporal_event_target_encoder" in full: return "event_target"
            if ("temporal_event_trunk" in full
                    or "temporal_event_proj" in full
                    or "temporal_event_gate" in full
                    or "temporal_event_delta" in full
                    or "temporal_event_damp" in full):
                return "event_adapter"
            # v25 / v26 labels.
            if "temporal_tail_calibration_head" in full: return "tail"
            if "temporal_residual_ext_head"     in full: return "ext"
            if "temporal_residual_gate_head"    in full: return "gate"
            return "main"
        param_labels = jax.tree_util.tree_map_with_path(
            lambda path, _leaf: _label_path(path), residual_params)
        ext_opt   = optax.adamw(ext_lr,      weight_decay=cfg.weight_decay)
        gate_opt  = optax.adamw(gate_lr,     weight_decay=cfg.weight_decay)
        tail_opt  = optax.adamw(tail_lr,     weight_decay=cfg.weight_decay)
        event_opt = optax.adamw(event_lr,    weight_decay=cfg.weight_decay)
        event_enc_opt = optax.adamw(event_enc_lr, weight_decay=cfg.weight_decay)
        main_opt  = optax.adamw(lr_schedule, weight_decay=cfg.weight_decay)
        opt = optax.multi_transform(
            {"ext": ext_opt, "gate": gate_opt, "tail": tail_opt,
             "event_adapter": event_opt, "event_target": event_enc_opt,
             "main": main_opt},
            param_labels)
        if cfg.grad_clip > 0:
            opt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), opt)
        print(f"[v25/v26/v30] multi-LR opt: ext_lr={ext_lr}, gate_lr={gate_lr}, "
              f"tail_lr={tail_lr}, event_lr={event_lr}, "
              f"event_target_lr={event_enc_lr}, main_lr={cfg.lr}")
    else:
        opt_tx = []
        if cfg.grad_clip > 0:
            opt_tx.append(optax.clip_by_global_norm(cfg.grad_clip))
        opt_tx.append(optax.adamw(lr_schedule, weight_decay=cfg.weight_decay))
        opt = optax.chain(*opt_tx) if len(opt_tx) > 1 else opt_tx[0]
        print(f"[v16] opt: grad_clip={cfg.grad_clip}, warmup={cfg.warmup_steps}, lr={cfg.lr}")

    # --- Phase-1 freeze for v25 (gated extreme), v26 (tail calibration), or
    # v30 (event adapter): only the new heads remain trainable; everything else
    # has gradients zeroed.
    _phase1_freeze = ((cfg.gated_extreme_head or cfg.tail_calib_head
                       or cfg.event_adapter_head)
                      and cfg.gate_phase == "1")
    if _phase1_freeze:
        def _is_trainable(path_tuple):
            full = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path_tuple)
            return ("temporal_residual_ext_head" in full
                    or "temporal_residual_gate_head" in full
                    or "temporal_tail_calibration_head" in full
                    or "temporal_event_trunk" in full
                    or "temporal_event_proj" in full
                    or "temporal_event_gate" in full
                    or "temporal_event_delta" in full
                    or "temporal_event_damp" in full
                    or "temporal_event_target_encoder" in full)
        _train_mask = jax.tree_util.tree_map_with_path(
            lambda path, _leaf: _is_trainable(path), residual_params)
        n_train = sum(int(m) * leaf.size
                      for m, leaf in zip(jax.tree_util.tree_leaves(_train_mask),
                                          jax.tree_util.tree_leaves(residual_params)))
        print(f"[v25/v26/v30] PHASE 1: freezing all except new heads. "
              f"Trainable param count: {n_train:,}")
    else:
        _train_mask = None

    opt_state = opt.init(residual_params)

    run_config = {
        "config": {k: getattr(cfg, k) for k in vars(cfg)},
        "model_cfg_residual": dataclasses.asdict(model_cfg_residual),
        "input_steps": int(input_steps),
        "n_residual_params": int(n_residual),
        "n_anchors_train": int(train_split.size),
        "n_anchors_val": int(val_split.size),
        "deepmind_overlay_copied": int(r_stats.copied),
        "deepmind_overlay_fresh_init": int(r_stats.initialized),
        "data_pipeline": "prepared_array+precomputed_residual",
        "architecture": "v9_corrected_GCResidualWithZeroHead",
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # ---------- 5) Train step: AR rollout within each BPTT chunk ----------
    bptt = cfg.bptt_steps
    seg_len = cfg.sequential_segment_steps
    if seg_len % bptt != 0:
        raise ValueError(f"seg_len ({seg_len}) must be divisible by bptt ({bptt})")
    chunks_per_segment = seg_len // bptt

    def _shift_inputs_with_state(prev_inputs, new_state, forcings_next):
        """Build next anchor's input window from an arbitrary state dataset.
        prev_inputs has time=input_steps (=2 for 12h window).
        new_state is whatever 1-step state we want to feed forward
        (baseline_pred for baseline stream, baseline_pred + residual_pred for
        residual stream)."""
        next_frame = xr.merge([new_state, forcings_next])
        if "datetime" in next_frame.coords:
            next_frame = next_frame.drop_vars("datetime")
        keys_in_next = [k for k in next_frame.data_vars if k in prev_inputs.data_vars]
        next_inputs_part = next_frame[keys_in_next]
        next_inputs_part = next_inputs_part.assign_coords(
            time=prev_inputs.time.values[-1:] + dt
        )
        # Pin the legacy compatibility mode explicitly.  This preserves the
        # original concat semantics and avoids one warning per AR anchor.
        merged = xr.concat(
            [prev_inputs, next_inputs_part], dim="time", data_vars="different",
            compat="equals")
        return merged.tail(time=input_steps)

    # v18 SINGLE-STREAM (baseline rollout): GC and Mamba both see the same
    # baseline-self-rollout input. Mamba's residual_pred is summed into
    # full_pred but does NOT feed back into next-step input.
    def _one_ar_step(rp, rs, key, current_inputs, truth_i, forcings_i):
        baseline_pred, _ = baseline_predict.apply(
            baseline_params, baseline_state, key,
            current_inputs, truth_i, forcings_i)
        residual_pred, new_rs = residual_predict.apply(
            rp, rs, key, current_inputs, truth_i, forcings_i)
        # Target = truth - GC(current_inputs). stop_gradient is redundant since
        # baseline_params is frozen but kept for clarity.
        target_da = jax.tree_util.tree_map(
            lambda t, b: t - jax.lax.stop_gradient(b), truth_i, baseline_pred)
        (loss_da, _), _ = residual_loss.apply(
            rp, rs, key, current_inputs, target_da, forcings_i, True)
        return baseline_pred, residual_pred, scalarize_loss(loss_da), new_rs
    _one_ar_step_ckpt = jax.checkpoint(_one_ar_step, static_argnums=())

    # v30 (C2 / B0): parallel variant with event_batch threaded through for
    # InfoNCE. Returns a dict of aux scalars so the main loop can log them.
    # Only used at the Python-static `contrastive_anchor_idx`.
    _AUX_KEYS = ("aux_L_con", "aux_con_pos_sim", "aux_con_neg_sim",
                 "aux_con_neg_sim_max", "aux_con_gap", "aux_con_top1",
                 "aux_z_pred_std", "aux_z_truth_std")

    def _extract_scalar(x):
        # aux values in graphcast convention are (value, weight) tuples.
        if isinstance(x, tuple) and len(x) >= 1:
            return jnp.asarray(x[0]).astype(jnp.float32)
        return jnp.asarray(x).astype(jnp.float32)

    def _one_ar_step_with_event(rp, rs, key, current_inputs, truth_i, forcings_i,
                                event_batch_i):
        baseline_pred, _ = baseline_predict.apply(
            baseline_params, baseline_state, key,
            current_inputs, truth_i, forcings_i)
        residual_pred, new_rs = residual_predict.apply(
            rp, rs, key, current_inputs, truth_i, forcings_i)
        target_da = jax.tree_util.tree_map(
            lambda t, b: t - jax.lax.stop_gradient(b), truth_i, baseline_pred)
        (loss_da, aux), _ = residual_loss.apply(
            rp, rs, key, current_inputs, target_da, forcings_i, True,
            event_batch=event_batch_i)
        # Extract a fixed-key dict of scalars — JIT-friendly (static keys).
        # Missing keys → 0.0 (should not happen when InfoNCE branch fires).
        aux_out = {k: _extract_scalar(aux.get(k, jnp.array(0.0)))
                   for k in _AUX_KEYS}
        return (baseline_pred, residual_pred, scalarize_loss(loss_da),
                new_rs, aux_out)
    _one_ar_step_with_event_ckpt = jax.checkpoint(
        _one_ar_step_with_event, static_argnums=())

    # v30 (C2): Python-static contrastive_anchor_idx. Only this one anchor gets
    # event_batch threaded into loss (so z_pred and truth tube stay time-aligned).
    _c_anchor = cfg.contrastive_anchor_idx
    if _c_anchor < 0:
        _c_anchor = bptt - 1
    _use_contrastive = (cfg.event_adapter_head and cfg.lambda_con > 0
                        and 0 <= _c_anchor < bptt)
    if _use_contrastive:
        print(f"[v30] BPTT anchor {_c_anchor}/{bptt-1} will get event_batch "
              f"(InfoNCE, lambda_con={cfg.lambda_con}).")

    # ar_start (Python int) is captured as a static closure variable per
    # train_step recompile (it doesn't change between steps within a stage).
    def _make_train_step(ar_start_static: int):
        @jax.jit
        def train_step(residual_params, residual_state, opt_state, keys,
                       inputs_truth_list, truths_list, forcings_list,
                       event_batch):
            """inputs_truth_list: list of real ERA5 inputs for anchors
            0..ar_start_static-1 (truth zone). Anchors ar_start_static..bptt-1
            use AR-fed inputs (built from previous anchor's prediction).
            truths_list: ground truth at each anchor's target time.
            forcings_list: forcings at each anchor.
            event_batch: dict passed only to anchor `_c_anchor` (v30 InfoNCE);
                         ignored when v30 is off (loop takes the plain path).
            """
            def f(rp):
                rs = residual_state
                losses = []
                # Fixed-key dict of aux scalars (JIT-friendly).
                aux_captured = {k: jnp.array(0.0, dtype=jnp.float32)
                                for k in _AUX_KEYS}
                # v18 SINGLE-STREAM (baseline rollout): both GC and Mamba see
                # the same input at every AR step. AR feedback is baseline_pred
                # only (NOT corrected). Mamba's residual prediction is
                # discarded from the input chain -- it only contributes to the
                # output via baseline + residual.
                current_inputs = inputs_truth_list[0]
                for i in range(bptt):
                    if _use_contrastive and i == _c_anchor:
                        baseline_pred, residual_pred, loss_i, rs, aux_dict_i = (
                            _one_ar_step_with_event_ckpt(
                                rp, rs, keys[i],
                                current_inputs,
                                truths_list[i], forcings_list[i],
                                event_batch))
                        aux_captured = aux_dict_i
                    else:
                        baseline_pred, residual_pred, loss_i, rs = _one_ar_step_ckpt(
                            rp, rs, keys[i],
                            current_inputs,
                            truths_list[i], forcings_list[i])
                    losses.append(loss_i)
                    if i < bptt - 1:
                        next_i = i + 1
                        if next_i < ar_start_static:
                            current_inputs = inputs_truth_list[next_i]
                        else:
                            # v20 fix: the new input slot is at time
                            # t+(i+1)*6h. Its input_forcings must be forcings
                            # AT that time = forcings_list[i] (the same
                            # target_forcings we just used to predict
                            # baseline_pred), NOT forcings_list[i+1] which is
                            # AT time t+(i+2)*6h. v18 had the off-by-one here.
                            if cfg.feedback_mode == "closed_loop_sg":
                                # Feed SSM with closed-loop trajectory but
                                # stop-grad on rp so BPTT graph == open-loop.
                                fb_state = jax.tree_util.tree_map(
                                    lambda b, r: b + jax.lax.stop_gradient(r),
                                    baseline_pred, residual_pred)
                            else:
                                fb_state = baseline_pred
                            current_inputs = _shift_inputs_with_state(
                                current_inputs, fb_state,
                                forcings_list[i])
                return jnp.stack(losses).mean(), (rs, aux_captured)

            (loss, (new_rs, aux_out)), grads = jax.value_and_grad(
                f, has_aux=True)(residual_params)
            # v25 phase-1: zero out grads for frozen params.
            if _train_mask is not None:
                grads = jax.tree_util.tree_map(
                    lambda g, m: g * jnp.asarray(m, dtype=g.dtype),
                    grads, _train_mask)
            grad_norm = optax.global_norm(grads)
            updates, new_opt_state = opt.update(grads, opt_state, residual_params)
            new_rp = optax.apply_updates(residual_params, updates)
            new_rs = jax.tree_util.tree_map(jax.lax.stop_gradient, new_rs)
            return new_rp, new_rs, new_opt_state, loss, grad_norm, aux_out
        return train_step

    # Determine ar_start once (constant per run) and build train_step accordingly.
    _ar_tail_K = cfg.ar_tail_K if cfg.ar_tail_K is not None else (bptt - 1)
    if not (0 <= _ar_tail_K <= bptt - 1):
        raise ValueError(f"--ar-tail-K must be in [0, bptt-1={bptt-1}], got {_ar_tail_K}")
    ar_start_static = bptt - _ar_tail_K   # captured by train_step closure
    print(f"[v16] ar-tail-K = {_ar_tail_K} (anchors 0..{ar_start_static-1} use truth, "
          f"{ar_start_static}..{bptt-1} use closed-loop AR)")
    train_step = _make_train_step(ar_start_static)

    # ---------- 6) Build train segments + main loop ----------
    train_segments = _build_segments(train_split, seg_len)
    print(f"[v13] {len(train_segments)} segments × {chunks_per_segment} "
          f"chunks/segment = {len(train_segments)*chunks_per_segment} steps/epoch")

    # v30: build the per-step event_batch source (Plan B0 sampler or Plan C dummy).
    # Shapes must be JIT-consistent across steps — patch_size, T_win, C_vars,
    # Nneg are static.
    _P = int(cfg.event_patch_size)
    _Nneg = int(cfg.n_neg)
    _T_win = int(cfg.event_t_win)
    _C_vars = int(cfg.event_c_vars)
    _B_dummy = 1

    # Persistent dummy (used when --event-dummy or _use_contrastive is False).
    # Kept device-resident so re-passing it every step doesn't re-upload.
    _rng_np_dummy = np.random.default_rng(cfg.seed + 1000)
    event_batch_dummy = dict(
        truth_tube_pos=jnp.asarray(
            _rng_np_dummy.standard_normal(
                (_B_dummy, _T_win, _P, _P, _C_vars)).astype(np.float32)),
        truth_tube_neg=jnp.asarray(
            _rng_np_dummy.standard_normal(
                (_B_dummy, _Nneg, _T_win, _P, _P, _C_vars)).astype(np.float32)),
        center_lat_idx=jnp.asarray(
            _rng_np_dummy.integers(_P, 88, size=(_B_dummy,)), dtype=jnp.int32),
        center_lon_idx=jnp.asarray(
            _rng_np_dummy.integers(0, 180, size=(_B_dummy,)), dtype=jnp.int32),
    )
    if _use_contrastive:
        print(f"[v30] event_batch shape: pos "
              f"{tuple(event_batch_dummy['truth_tube_pos'].shape)}, "
              f"neg {tuple(event_batch_dummy['truth_tube_neg'].shape)}")

    # Plan B0: real climate-matched sampler (used when --event-dummy is OFF).
    event_sampler = None
    if _use_contrastive and not cfg.event_dummy:
        from scripts.training.full_mamba_v30.event_tube_sampler import (
            PerStepEventSampler)
        event_sampler = PerStepEventSampler(
            store=store,
            anchor_indices=anchor_indices,
            split_indices=train_split,
            input_steps=input_steps,
            dt=dt,
            task_cfg=task_cfg,
            patch_size=_P,
            n_neg=_Nneg,
            time_window=_T_win,
            min_time_gap_days=int(cfg.event_min_time_gap_days),
            mean_by_level=norm_stats["mean_by_level"],
            stddev_by_level=norm_stats["stddev_by_level"],
        )
        _sampler_rng = np.random.default_rng(cfg.seed + 2000)
        print(f"[v30] B0 sampler active: T={_T_win}, P={_P}, Nneg={_Nneg}, "
              f"min_time_gap_days={cfg.event_min_time_gap_days}, "
              f"tube_normalized={len(event_sampler._var_mean) > 0}")

    train_log = []
    seg_iter = iter(train_segments)
    cur_segment = next(seg_iter)
    seg_pos = 0

    for step in range(cfg.start_step, cfg.max_steps + 1):
        while seg_pos + bptt > len(cur_segment):
            try:
                cur_segment = next(seg_iter)
            except StopIteration:
                seg_iter = iter(train_segments)
                cur_segment = next(seg_iter)
            seg_pos = 0
            residual_state = jax.tree_util.tree_map(jnp.zeros_like, residual_state)

        chunk_residual_idxs = np.asarray(
            [int(cur_segment[seg_pos + j]) for j in range(bptt)], dtype=np.int64)
        seg_pos += bptt
        raw_anchor_idxs = anchor_indices[chunk_residual_idxs]

        rng, *split_keys = jax.random.split(rng, bptt + 1)
        keys = jnp.stack(split_keys)

        # v16: load real ERA5 inputs for the first (bptt - K) anchors (truth
        # zone), then anchors (bptt-K)..bptt-1 use AR-fed inputs inside
        # train_step. truth + forcings still loaded for every anchor as targets.
        ar_tail_K = cfg.ar_tail_K if cfg.ar_tail_K is not None else (bptt - 1)
        if not (0 <= ar_tail_K <= bptt - 1):
            raise ValueError(f"ar_tail_K must be in [0, bptt-1=={bptt-1}], got {ar_tail_K}")
        ar_start = bptt - ar_tail_K   # anchors 0..ar_start-1 use real inputs
        inputs_truth = [None] * bptt
        tgt_list, frc_list = [], []
        for i in range(bptt):
            t_idx = int(raw_anchor_idxs[i])
            inp_i, tgt_i, frc_i = store.build_batch_from_indices(
                indices=[t_idx],
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt,
            )
            if i < ar_start:
                inputs_truth[i] = inp_i
            tgt_list.append(tgt_i)
            frc_list.append(frc_i)
        # Pass only the truth-zone inputs (anchors 0..ar_start-1) into train_step.
        # The closure ar_start_static == ar_start determines how many of these
        # are actually read inside train_step.
        inputs_truth_tuple = tuple(inputs_truth[i] for i in range(ar_start))

        # Build event_batch for this step. If B0 sampler active, sample fresh
        # per step. Otherwise reuse the fixed dummy (Plan C fallback).
        if event_sampler is not None:
            _pos_anchor_store_idx = int(raw_anchor_idxs[_c_anchor])
            eb_np = event_sampler.sample(_pos_anchor_store_idx, _sampler_rng)
            event_batch_step = dict(
                truth_tube_pos=jnp.asarray(eb_np["truth_tube_pos"]),
                truth_tube_neg=jnp.asarray(eb_np["truth_tube_neg"]),
                center_lat_idx=jnp.asarray(eb_np["center_lat_idx"], dtype=jnp.int32),
                center_lon_idx=jnp.asarray(eb_np["center_lon_idx"], dtype=jnp.int32),
            )
            # One-time time-alignment sanity print.
            if step == cfg.start_step:
                _pos_time = store.time.values[_pos_anchor_store_idx]
                _pos_lat_idx = int(eb_np["center_lat_idx"][0])
                _pos_lon_idx = int(eb_np["center_lon_idx"][0])
                _pos_lat_val = float(store.coords["lat"][_pos_lat_idx])
                _pos_lon_val = float(store.coords["lon"][_pos_lon_idx])
                _valid_time = _pos_time + np.timedelta64(6, "h")
                _pos_tube_val = float(eb_np["truth_tube_pos"][0, 0, 8, 8, 0])
                print(f"[v30 sanity] contrastive anchor store idx={_pos_anchor_store_idx}, "
                      f"anchor_time={_pos_time}, valid_time (target)={_valid_time}")
                print(f"[v30 sanity] patch center (lat_idx={_pos_lat_idx}, "
                      f"lon_idx={_pos_lon_idx}) → (lat={_pos_lat_val:.2f}°, "
                      f"lon={_pos_lon_val:.2f}°)")
                print(f"[v30 sanity] truth_tube_pos[0,0,center_lat,center_lon,var0(2m_T)] = {_pos_tube_val:.3f}")
        else:
            event_batch_step = event_batch_dummy

        t0 = time.time()
        (residual_params, residual_state, opt_state,
         loss, grad_norm, aux_out) = train_step(
            residual_params, residual_state, opt_state, keys,
            inputs_truth_tuple, tuple(tgt_list), tuple(frc_list),
            event_batch_step)
        loss = float(loss); grad_norm = float(grad_norm)
        aux_floats = {k: float(v) for k, v in aux_out.items()}
        step_t = time.time() - t0

        if step <= 5 or step % 10 == 0:
            if _use_contrastive:
                _extra = (
                    f" L_con {aux_floats['aux_L_con']:.4f}"
                    f" pos {aux_floats['aux_con_pos_sim']:+.4f}"
                    f" neg {aux_floats['aux_con_neg_sim']:+.4f}"
                    f" negMax {aux_floats['aux_con_neg_sim_max']:+.4f}"
                    f" gap {aux_floats['aux_con_gap']:+.4f}"
                    f" top1 {aux_floats['aux_con_top1']:.3f}"
                    f" zpStd {aux_floats['aux_z_pred_std']:.3f}"
                    f" ztStd {aux_floats['aux_z_truth_std']:.3f}")
            else:
                _extra = ""
            print(f"step {step}/{cfg.max_steps} loss {loss:.5f} "
                  f"grad_norm {grad_norm:.4f}{_extra} step_time {step_t:.2f}s")
        train_log.append({"step": step, "loss": loss, "grad_norm": grad_norm,
                          "aux": aux_floats, "step_time": float(step_t)})

        if step % cfg.checkpoint_every == 0:
            import pickle
            ckpt_path = out_dir / f"v13_residual_step{step}.pkl"
            with ckpt_path.open("wb") as f:
                pickle.dump({"residual_params": residual_params,
                             "residual_state": residual_state}, f)
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_log, f, indent=2)
            print(f"[v13] saved ckpt {ckpt_path}")

    print(f"[v13] training done after {cfg.max_steps} steps")


if __name__ == "__main__":
    main()
