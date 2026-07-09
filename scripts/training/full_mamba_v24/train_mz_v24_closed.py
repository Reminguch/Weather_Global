"""v24-CLOSED = v24 architecture + CLOSED-LOOP AR feedback.

Differs from train_mz_v24.py in EXACTLY ONE place: the AR feedback in the
BPTT train step uses full_pred = baseline_pred + residual_pred instead of
baseline_pred only. Matches the user's "step k feeds full_k into step k+1"
design and matches Ilya's branch_rollout / our cold_full eval semantics.

Logical AR loop (closed-loop):
  step 1: bp_1 = G(truth_0); rp_1 = R(truth_0, s_0); full_1 = bp_1 + rp_1; loss_1 = ||full_1 - truth_1||^2
  step 2: cur_1 = shift(truth_0, full_1); bp_2 = G(cur_1); rp_2 = R(cur_1, s_1);
           full_2 = bp_2 + rp_2; loss_2 = ||full_2 - truth_2||^2
  step k: cur_{k-1} = shift(cur_{k-2}, full_{k-1}); ...
This trains Mamba on its OWN error compounding (not on the artificial
"baseline self-rollout" distribution that train_mz_v20.py / v22 used).

Identical to v24 in arch + freeze + AdamW + jax.checkpoint + AR-tail-K.

Original v24 docstring follows:

v24 = v20 + freeze GC2 encoder/decoder (graph-level) + larger Mamba.

Goal: move trainable-param budget from GC2's grid2mesh / mesh2grid GNN into
a wider/deeper Mamba block, while keeping AR rollout protocol identical to
v22 (BPTT=24, K=22). Same data + baseline + AR feedback semantics as v20/v22.

Two levels of freeze, both applied:

  1. Graph-level (saves activation memory):
       latent_mesh_nodes, latent_grid_nodes = grid2mesh(grid_features)
       latent_mesh_nodes = jax.lax.stop_gradient(latent_mesh_nodes)
       latent_grid_nodes = jax.lax.stop_gradient(latent_grid_nodes)
     This cuts backward through grid2mesh entirely. JAX no longer needs to
     retain grid2mesh's internal activations during the rematerialized
     backward (we still use jax.checkpoint per AR step).
     mesh2grid does NOT get output stop_gradient (that would zero out the
     residual head's gradient signal), but its params are frozen via the
     optimizer-level mechanism below.

  2. Optimizer-level (saves AdamW state, prevents unintended updates):
       optax.multi_transform({
           'trainable': AdamW,
           'frozen': optax.set_to_zero(),
       })
     Same approach as v21. Frozen substrs: 'grid2mesh_gnn', 'mesh2grid_gnn'.

Defaults (= the "scheme Z+" config we converged on for memory):
  --temporal-hidden-size 512   # d_inner: 128 → 512 (4×)
  --temporal-d-state 16        # unchanged (d_state scaling is a memory trap)
  --temporal-layers 3          # 2 → 3 layers per insertion (1.5×)
  --temporal-d-conv 8          # 4 → 8 (matches v23 ablation)
  --bptt-steps 24, --ar-tail-K 22  # same as v22 for direct comparison

Mamba per-block (d_inner=512, d_state=16, d_conv=8, dt_rank=32 auto):
  ~852k params/block × 6 blocks (3 layers × 2 insertions) = ~5.1M Mamba

Total trainable ≈ mesh_gnn (3.94M, processor) + Mamba (5.1M) + head (7k)
  = ~9.05M  (vs v22's 11.09M; the 6.29M of grid2mesh+mesh2grid are now frozen)

Memory: layered freeze + jax.checkpoint per step is borderline-feasible on
80GB A100. Always run smoke test (--max-steps 2) before committing K=1.

K=1 (ar-tail-K=0) starts the residual head from the zero-init pattern
established in v9/v15/v18/v20/v22 (see docs/experiments/zero_init_residual_head.md).
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

from graphcast import casting, graphcast as gc, normalization, xarray_jax  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    DirectResidualNormalizer, scalarize_loss,
)
from src.models.mamba.training.param_utils import (  # noqa: E402
    overlay_matching_params,
)
from src.data.prepared_array import PreparedArrayStore  # noqa: E402

from scripts.training.full_mamba_v9.train_mz_v9 import (  # noqa: E402
    GCResidualWithZeroHead, _attach_temporal,
)


class GCResidualWithZeroHeadV24(GCResidualWithZeroHead):
    """v24: graph-level stop_gradient on grid2mesh outputs.

    Same forward as v9 GCResidualWithZeroHead except that the latent mesh
    and latent grid features produced by grid2mesh_gnn are detached from
    the autodiff graph. JAX therefore does not need to retain grid2mesh's
    internal activations during backward (which is rematerialized once per
    AR step via jax.checkpoint).

    mesh2grid is NOT detached here -- its input gradient must propagate
    back to Mamba/processor for the residual head to receive any training
    signal. mesh2grid's parameters are kept frozen via the optimizer-level
    optax.multi_transform({'frozen': set_to_zero()}) in the main trainer.
    """

    def __call__(self, inputs, targets_template, forcings, is_training=False):
        self._maybe_init(inputs)
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(
            grid_node_features)
        # v24 graph-level freeze: detach grid2mesh outputs. Backward through
        # grid2mesh no longer needs to materialize its activations.
        latent_mesh_nodes = jax.lax.stop_gradient(latent_mesh_nodes)
        latent_grid_nodes = jax.lax.stop_gradient(latent_grid_nodes)
        if (self._temporal_backbone != "none"
                and self._temporal_location == "mesh_post_encoder"):
            latent_mesh_nodes = self._run_temporal_mesh_block(
                latent_mesh_nodes, is_training=is_training)
        updated_latent_mesh_nodes = self._run_mesh_gnn(
            latent_mesh_nodes, is_training=is_training)
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes)
        out_size = output_grid_nodes.shape[-1]
        residual_head = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_residual_head",
        )
        residual_output = residual_head(output_grid_nodes)
        return self._grid_node_outputs_to_prediction(
            residual_output, targets_template)


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

    def _build_residual_predictor():
        p = GCResidualWithZeroHeadV24(model_cfg_residual, task_cfg)
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

    def residual_loss_fn(inputs, residual_targets, forcings, is_training):
        del is_training
        return _build_residual_predictor().loss(inputs, residual_targets, forcings)

    def residual_pred_fn(inputs, targets_template, forcings):
        return _build_residual_predictor()(
            inputs, targets_template=targets_template, forcings=forcings)

    # v24-CLOSED Fix #1: single forward returns (loss, scalars, predictions).
    # Avoids the v22-era double-forward where residual_predict.apply was called
    # to get residual_pred for AR feedback AND residual_loss.apply was called
    # to get the gradient. Now state update + loss + AR feedback all come from
    # ONE forward pass through the residual predictor.
    def residual_loss_and_pred_fn(inputs, residual_targets, forcings, is_training):
        del is_training
        return _build_residual_predictor().loss_and_predictions(
            inputs, residual_targets, forcings)

    residual_loss = hk.transform_with_state(residual_loss_fn)
    residual_predict = hk.transform_with_state(residual_pred_fn)
    residual_loss_and_pred = hk.transform_with_state(residual_loss_and_pred_fn)

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
    residual_params, residual_state = residual_loss.init(
        k, sample_inputs, sample_targets, sample_forcings, True)
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
        # Fix #3: do NOT load ckpt residual_state — it's a random snapshot from
        # the training's final batch, NOT a meaningful initialization for the
        # next run's first batch. residual_state stays at zero-init. The
        # `residual_state` key in ckpt is preserved for debugging only.
        print(f"[v13] resumed params only from {cfg.resume_from} at step "
              f"{cfg.start_step}; residual_state kept at zero-init")

    # Fix #4: set up cos(lat) weights for lat-weighted signed bias diagnostic.
    # Computed once outside the JIT'd train_step; captured by closure.
    lat_np = np.asarray(sample_inputs["lat"].values, dtype=np.float32)
    cos_lat_np = np.cos(np.deg2rad(lat_np))
    cos_lat_np = cos_lat_np / cos_lat_np.mean()  # normalize so global mean weight = 1
    cos_lat_jnp = jnp.asarray(cos_lat_np, dtype=jnp.float32)
    print(f"[v24-closed] lat-weighted bias diagnostic: lat range {lat_np.min():.1f}..{lat_np.max():.1f}, "
          f"{len(lat_np)} lat points")

    # LR schedule with optional warmup
    if cfg.warmup_steps > 0:
        lr_schedule = optax.warmup_constant_schedule(
            init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup_steps)
    else:
        lr_schedule = cfg.lr
    # v24: freeze grid2mesh_gnn and mesh2grid_gnn via optax.multi_transform.
    # Graph-level stop_gradient on grid2mesh outputs (in GCResidualWithZeroHeadV24)
    # already kills grid2mesh's gradient/activation cost; this multi_transform
    # is the second line of defense: even mesh2grid (whose inputs still carry
    # gradient) doesn't receive parameter updates.
    FROZEN_SUBSTRS = ("grid2mesh_gnn", "mesh2grid_gnn")
    def _module_is_frozen(module_key):
        return any(sub in module_key for sub in FROZEN_SUBSTRS)
    def _label_for(mod_key):
        return 'frozen' if _module_is_frozen(mod_key) else 'trainable'
    label_tree = {
        mod_key: jax.tree_util.tree_map(
            lambda _, lbl=_label_for(mod_key): lbl, mod_val)
        for mod_key, mod_val in residual_params.items()
    }
    n_train, n_frozen = 0, 0
    frozen_modules, trainable_modules = [], []
    for mod_key, mod_val in residual_params.items():
        sz = sum(p.size for p in jax.tree_util.tree_leaves(mod_val))
        if _module_is_frozen(mod_key):
            n_frozen += sz; frozen_modules.append((mod_key, sz))
        else:
            n_train += sz; trainable_modules.append((mod_key, sz))
    print(f"[v24] FREEZE grid2mesh + mesh2grid (optimizer-level set_to_zero + "
          f"graph-level stop_gradient on grid2mesh outputs):")
    print(f"[v24]   trainable params: {n_train:,} ({100*n_train/(n_train+n_frozen):.1f}%) "
          f"across {len(trainable_modules)} modules")
    print(f"[v24]   frozen params:    {n_frozen:,} ({100*n_frozen/(n_train+n_frozen):.1f}%) "
          f"across {len(frozen_modules)} modules")
    print(f"[v24]   total residual params: {n_train+n_frozen:,}")
    print(f"[v24]   first 3 frozen modules: {[m[0] for m in frozen_modules[:3]]}")
    print(f"[v24]   first 3 trainable modules: {[m[0] for m in trainable_modules[:3]]}")

    trainable_adamw = optax.adamw(lr_schedule, weight_decay=cfg.weight_decay)
    multi_opt = optax.multi_transform(
        {'trainable': trainable_adamw, 'frozen': optax.set_to_zero()},
        param_labels=label_tree,
    )
    opt_tx = []
    if cfg.grad_clip > 0:
        opt_tx.append(optax.clip_by_global_norm(cfg.grad_clip))
    opt_tx.append(multi_opt)
    opt = optax.chain(*opt_tx) if len(opt_tx) > 1 else opt_tx[0]
    print(f"[v24] opt: grad_clip={cfg.grad_clip}, warmup={cfg.warmup_steps}, "
          f"lr={cfg.lr}, multi_transform=True")
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
        "architecture": "v24_closed_GCResidualWithZeroHead_grid2mesh_stopgrad_freeze_encdec_bigM_FULLPRED_AR_FEEDBACK",
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
        merged = xr.concat([prev_inputs, next_inputs_part], dim="time", data_vars="different")
        return merged.tail(time=input_steps)

    # v24-CLOSED: CLOSED-LOOP AR. The shift call in train_step (below) uses
    # full_pred = baseline_pred + residual_pred to advance the input window
    # for the next AR step. Both GC and Mamba see the corrected (full-pred)
    # trajectory at every step.
    #
    # Fix #1: ONE residual forward pass via loss_and_predictions. State
    # update + loss gradient + AR feedback all come from same forward.
    #
    # Fix #2: stop_gradient on baseline_pred IMMEDIATELY after computing it.
    # baseline_params is frozen, so we don't want gradients flowing through
    # GraphCast's internal Jacobian — that just wastes activation memory
    # during BPTT backward. With closed-loop AR, full_pred = bp + rp is
    # fed forward, so without this stop_gradient JAX would have to retain
    # baseline activations through the entire K-step chain. Critical for
    # K=22 BPTT memory.
    def _one_ar_step(rp, rs, key, current_inputs, truth_i, forcings_i):
        baseline_pred_raw, _ = baseline_predict.apply(
            baseline_params, baseline_state, key,
            current_inputs, truth_i, forcings_i)
        baseline_pred = jax.tree_util.tree_map(
            jax.lax.stop_gradient, baseline_pred_raw)
        # baseline_pred is already stopped; use it as-is in target.
        target_da = jax.tree_util.tree_map(
            lambda t, b: t - b, truth_i, baseline_pred)
        ((loss_da, _), residual_pred), new_rs = residual_loss_and_pred.apply(
            rp, rs, key, current_inputs, target_da, forcings_i, True)
        return baseline_pred, residual_pred, scalarize_loss(loss_da), new_rs
    _one_ar_step_ckpt = jax.checkpoint(_one_ar_step, static_argnums=())

    # ar_start (Python int) is captured as a static closure variable per
    # train_step recompile (it doesn't change between steps within a stage).
    def _make_train_step(ar_start_static: int):
        @jax.jit
        def train_step(residual_params, residual_state, opt_state, keys,
                       inputs_truth_list, truths_list, forcings_list):
            """inputs_truth_list: list of real ERA5 inputs for anchors
            0..ar_start_static-1 (truth zone). Anchors ar_start_static..bptt-1
            use AR-fed inputs (built from previous anchor's prediction).
            truths_list: ground truth at each anchor's target time.
            forcings_list: forcings at each anchor.
            """
            def f(rp):
                rs = residual_state
                losses = []
                # v24-CLOSED: CLOSED-LOOP. AR feedback is full_pred
                # (= baseline_pred + residual_pred). Both GC and Mamba see
                # the corrected trajectory at every step. Errors compound
                # exactly as they will at eval — train_distribution matches
                # eval_distribution (closed-loop).
                current_inputs = inputs_truth_list[0]
                # Fix #2: bias diagnostic for the LAST AR step. Tracks global
                # mean drift of full_pred vs truth for 3 sentinel vars. Tells
                # us if Mamba is paying for variance reduction with global bias
                # (the failure mode we identified in cold-eval mean traces).
                bias_2mt = jnp.float32(0.0)
                bias_u10 = jnp.float32(0.0)
                bias_mslp = jnp.float32(0.0)
                for i in range(bptt):
                    baseline_pred, residual_pred, loss_i, rs = _one_ar_step_ckpt(
                        rp, rs, keys[i],
                        current_inputs,
                        truths_list[i], forcings_list[i])
                    losses.append(loss_i)
                    # Build full_pred at every step so we can both (a) feed it
                    # into the next AR step (if needed) and (b) compute the
                    # bias diagnostic on the final step.
                    full_pred = jax.tree_util.tree_map(
                        lambda b, r: b + r, baseline_pred, residual_pred)
                    if i == bptt - 1:
                        # Fix #4: final-step lat-weighted SIGNED global mean
                        # bias of (full_pred - truth). Robust to dim order:
                        # detect lat axis from xarray dim names (static).
                        def _latw_signed_bias(pred_da, truth_da):
                            err = xarray_jax.unwrap_data(pred_da - truth_da)
                            lat_axis = pred_da.dims.index("lat")
                            weight_shape = [1] * err.ndim
                            weight_shape[lat_axis] = -1
                            w = cos_lat_jnp.reshape(weight_shape)
                            return jnp.mean(err * w)
                        bias_2mt = _latw_signed_bias(
                            full_pred["2m_temperature"],
                            truths_list[i]["2m_temperature"])
                        bias_u10 = _latw_signed_bias(
                            full_pred["10m_u_component_of_wind"],
                            truths_list[i]["10m_u_component_of_wind"])
                        bias_mslp = _latw_signed_bias(
                            full_pred["mean_sea_level_pressure"],
                            truths_list[i]["mean_sea_level_pressure"])
                    if i < bptt - 1:
                        next_i = i + 1
                        if next_i < ar_start_static:
                            current_inputs = inputs_truth_list[next_i]
                        else:
                            # v24-CLOSED: feed full_pred = bp + rp forward.
                            # v20 forcings fix preserved.
                            current_inputs = _shift_inputs_with_state(
                                current_inputs, full_pred,
                                forcings_list[i])
                return jnp.stack(losses).mean(), (rs, bias_2mt, bias_u10, bias_mslp)

            (loss, (new_rs, bias_2mt, bias_u10, bias_mslp)), grads = (
                jax.value_and_grad(f, has_aux=True)(residual_params))
            grad_norm = optax.global_norm(grads)
            updates, new_opt_state = opt.update(grads, opt_state, residual_params)
            new_rp = optax.apply_updates(residual_params, updates)
            new_rs = jax.tree_util.tree_map(jax.lax.stop_gradient, new_rs)
            return (new_rp, new_rs, new_opt_state, loss, grad_norm,
                    bias_2mt, bias_u10, bias_mslp)
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

        t0 = time.time()
        (residual_params, residual_state, opt_state, loss, grad_norm,
         bias_2mt, bias_u10, bias_mslp) = train_step(
            residual_params, residual_state, opt_state, keys,
            inputs_truth_tuple, tuple(tgt_list), tuple(frc_list))
        loss = float(loss); grad_norm = float(grad_norm)
        bias_2mt = float(bias_2mt); bias_u10 = float(bias_u10); bias_mslp = float(bias_mslp)
        step_t = time.time() - t0

        if step <= 5 or step % 10 == 0:
            print(f"step {step}/{cfg.max_steps} loss {loss:.5f} "
                  f"grad_norm {grad_norm:.4f} "
                  f"bias_2mt {bias_2mt:+.3e}K bias_u10 {bias_u10:+.3e}m/s "
                  f"bias_mslp {bias_mslp:+.2f}Pa step_time {step_t:.2f}s")
        train_log.append({"step": step, "loss": loss, "grad_norm": grad_norm,
                          "bias_2mt_K": bias_2mt, "bias_u10_ms": bias_u10,
                          "bias_mslp_Pa": bias_mslp,
                          "step_time": float(step_t)})

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
