"""v11: v3-style fixed geometric grid<->mesh + Mamba (no learned GNN encoder/decoder).

Architecture:
  GC1 baseline branch (FROZEN, full DeepMind public GC_small, msg_steps=16):
      inputs --> baseline_pred

  Residual branch (TRAINABLE, ~280K params total):
      inputs (xarray)
        --> _inputs_to_grid_node_features          (gc.GraphCast helper)
        --> reshape to [B, P_grid, F_in]
        --> Linear F_in -> H (input_proj)          ~21K
        --> grid_to_mesh fixed Gaussian aggregate  (0 params, KNN+exp(-d^2/2sigma^2))
        --> [B, M, H]
        --> Mamba (the only learned spatial mixer) ~244K
        --> mesh_to_grid fixed Gaussian aggregate  (0 params)
        --> Linear H -> output_size (zero-init)    ~10K
        --> _grid_node_outputs_to_prediction (xarray)

  prediction = baseline_pred + residual_pred

Compared to v10:
  - Replaces 6.5M-param DeepMind GNN encoder/decoder with ~30K-param fixed
    geometric Gaussian aggregation (v3 style). Total residual params ~280K vs
    v10's 6.7M.
  - Tests whether the learned GNN encoder/decoder were actually contributing
    capacity, or whether geometry alone is enough.

Why this matters:
  v10 (full GNN encoder/decoder, no processor) and v9 corrected (full GNN
  encoder/decoder, with processor, all trainable) hit the same step-500 loss
  (0.74). If v11 (~24x smaller) also hits 0.74, it confirms the bottleneck
  is the residual task itself or Mamba capacity, not spatial mixing.
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
from src.models.mz.meshed_mamba.mesh_ops import (  # noqa: E402
    build_grid_mesh_projections,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="v11: fixed geometric grid<->mesh + Mamba residual.")
    p.add_argument("--data-path", default=base_train.DEFAULT_DATA_PATH)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5,
                   help="icosphere splits (3->642 nodes, 4->2562, 5->10242)")
    p.add_argument("--hidden-size", type=int, default=128,
                   help="latent dim in mesh space (matches v3 default)")
    p.add_argument("--baseline-msg-steps", type=int, default=16,
                   help="GC1 baseline processor depth.")
    p.add_argument("--n-grid-neighbors", type=int, default=6)
    p.add_argument("--n-mesh-neighbors", type=int, default=3)
    p.add_argument("--sigma-scale", type=float, default=1.0)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--train-start-year", type=int, default=2020)
    p.add_argument("--train-end-year", type=int, default=2021)
    p.add_argument("--ckpt-in", required=True)
    p.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=1)
    p.add_argument("--sequential-segment-steps", type=int, default=32)
    p.add_argument("--bptt-steps", type=int, default=8)
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
    p.add_argument("--resume-from", default=None,
                   help="path to a v11 ckpt pickle to warm-start residual_params from")
    p.add_argument("--start-step", type=int, default=1,
                   help="step counter offset (e.g., 5001 if resuming from step 5000 ckpt)")
    return p.parse_args()


def _grid_to_mesh_aggregate(x_b_p_h, g2m_indices, g2m_weights):
    """Fixed Gaussian KNN aggregation [B, P, H] -> [B, M, H]."""
    gathered = x_b_p_h[:, g2m_indices, :]  # [B, M, K, H]
    return jnp.sum(gathered * g2m_weights.astype(gathered.dtype)[None, :, :, None],
                   axis=2)


def _mesh_to_grid_aggregate(x_b_m_h, m2g_indices, m2g_weights):
    """Fixed Gaussian KNN aggregation [B, M, H] -> [B, P, H]."""
    gathered = x_b_m_h[:, m2g_indices, :]
    return jnp.sum(gathered * m2g_weights.astype(gathered.dtype)[None, :, :, None],
                   axis=2)


def _attach_temporal(predictor, cfg):
    predictor._temporal_backbone = "mamba"
    predictor._temporal_location = "mesh_post_encoder"
    predictor._temporal_hidden_size = cfg.hidden_size
    predictor._temporal_d_inner = getattr(cfg, "temporal_d_inner", None)
    predictor._temporal_d_state = cfg.temporal_d_state
    predictor._temporal_d_conv = cfg.temporal_d_conv
    predictor._temporal_dt_rank = cfg.temporal_dt_rank
    predictor._temporal_bias = cfg.temporal_bias
    predictor._temporal_conv_bias = cfg.temporal_conv_bias
    predictor._temporal_layers = cfg.temporal_layers
    predictor._temporal_dropout = cfg.temporal_dropout
    predictor._temporal_stateful = True
    predictor._temporal_zero_init_out = cfg.temporal_zero_init_out
    return predictor


class GCFixedMeshMamba(gc.GraphCast):
    """v3-style: input_proj -> fixed g2m -> Mamba -> fixed m2g -> zero-init head.

    Skips DeepMind's _run_grid2mesh_gnn / _run_mesh_gnn / _run_mesh2grid_gnn
    entirely. Reuses gc.GraphCast only for the xarray <-> tensor reshape
    helpers (_inputs_to_grid_node_features, _grid_node_outputs_to_prediction).

    Attributes set externally (after __init__, before first call):
        _g2m_idx, _g2m_w   [M, K_g] int32 / float32
        _m2g_idx, _m2g_w   [P, K_m] int32 / float32
        _hidden_size       int
        _output_size       int
    Plus the temporal_* attributes from _attach_temporal().
    """

    def __call__(self, inputs, targets_template, forcings, is_training=False):
        self._maybe_init(inputs)
        # [P_grid, B, F_in]
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
        # transpose to v3 convention [B, P_grid, F_in]
        x = jnp.transpose(grid_node_features, (1, 0, 2))
        # input projection F_in -> H
        h = hk.Linear(self._hidden_size, name="input_proj")(x)  # [B, P, H]
        # fixed grid -> mesh
        h_mesh = _grid_to_mesh_aggregate(
            h, self._g2m_idx, self._g2m_w)                       # [B, M, H]
        # gc.GraphCast convention for _run_temporal_mesh_block is [M, B, H]
        h_mesh = jnp.transpose(h_mesh, (1, 0, 2))                # [M, B, H]
        # Mamba on mesh latents
        if self._temporal_backbone != "none":
            h_mesh = self._run_temporal_mesh_block(
                h_mesh, is_training=is_training)
        # back to [B, M, H]
        h_mesh = jnp.transpose(h_mesh, (1, 0, 2))
        # fixed mesh -> grid
        h = _mesh_to_grid_aggregate(
            h_mesh, self._m2g_idx, self._m2g_w)                  # [B, P, H]
        # zero-init residual head H -> output_size
        residual_head = hk.Linear(
            self._output_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_residual_head",
        )
        out = residual_head(h)                                    # [B, P, output_size]
        # transpose back to gc.GraphCast convention [P, B, output_size]
        out = jnp.transpose(out, (1, 0, 2))
        return self._grid_node_outputs_to_prediction(out, targets_template)


def _compute_output_size(task_cfg):
    """Number of output channels = surface_vars + levels * atmospheric_vars."""
    num_surface = len(set(task_cfg.target_variables) - set(gc.ALL_ATMOSPHERIC_VARS))
    num_atm = len(set(task_cfg.target_variables) & set(gc.ALL_ATMOSPHERIC_VARS))
    return num_surface + len(task_cfg.pressure_levels) * num_atm


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg_baseline = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=512,  # match DeepMind GC_small
        gnn_msg_steps=cfg.baseline_msg_steps,
    )
    # Residual branch: latent_size and msg_steps don't matter (we skip the
    # GNNs entirely), but model_cfg validation may want them.
    model_cfg_residual = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.hidden_size,
        gnn_msg_steps=1,
    )
    output_size = _compute_output_size(task_cfg)
    print(f"[v11] baseline GC1: latent=512 msg_steps={cfg.baseline_msg_steps}")
    print(f"[v11] residual: hidden={cfg.hidden_size} output_size={output_size}")
    print(f"[v11] Mamba: d_state={cfg.temporal_d_state} d_conv={cfg.temporal_d_conv} "
          f"layers={cfg.temporal_layers}")

    # Precompute fixed grid<->mesh projections.
    base_lat = np.linspace(-90.0, 90.0, int(180 / cfg.resolution) + 1)
    base_lon = np.arange(0.0, 360.0, cfg.resolution)
    proj_arrays, n_mesh = build_grid_mesh_projections(
        lat_deg=base_lat, lon_deg=base_lon,
        mesh_size=cfg.mesh_size,
        n_grid_neighbors=cfg.n_grid_neighbors,
        n_mesh_neighbors=cfg.n_mesh_neighbors,
        sigma_scale=cfg.sigma_scale,
    )
    print(f"[v11] mesh nodes M={n_mesh}, grid P={len(base_lat)*len(base_lon)}, "
          f"K_g2m={cfg.n_grid_neighbors}, K_m2g={cfg.n_mesh_neighbors}")

    norm_stats = base_train.load_stats(Path(cfg.stats_dir))
    base_train.validate_stats_coverage(task_cfg, norm_stats)

    class _SplitCfg:
        data_path = cfg.data_path
        resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = cfg.train_start_year
        train_end_year = cfg.train_end_year
    train_ds, _eval_ds = base_train._open_local_splits(_SplitCfg)
    train_ds = base_train.prepare_dataset_for_task(train_ds, task_cfg)
    dt = base_train.infer_time_step(train_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    use_bf16 = cfg.precision == "bf16"

    g2m_idx_jax = jnp.asarray(proj_arrays["g2m_indices"], dtype=jnp.int32)
    g2m_w_jax = jnp.asarray(proj_arrays["g2m_weights"], dtype=jnp.float32)
    m2g_idx_jax = jnp.asarray(proj_arrays["m2g_indices"], dtype=jnp.int32)
    m2g_w_jax = jnp.asarray(proj_arrays["m2g_weights"], dtype=jnp.float32)

    def _build_baseline_predictor():
        predictor = gc.GraphCast(model_cfg_baseline, task_cfg)
        if use_bf16:
            predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"],
        )
        return predictor

    def _build_residual_predictor():
        predictor = GCFixedMeshMamba(model_cfg_residual, task_cfg)
        # Inject fixed projections and sizes.
        predictor._g2m_idx = g2m_idx_jax
        predictor._g2m_w = g2m_w_jax
        predictor._m2g_idx = m2g_idx_jax
        predictor._m2g_w = m2g_w_jax
        predictor._hidden_size = cfg.hidden_size
        predictor._output_size = output_size
        _attach_temporal(predictor, cfg)
        if use_bf16:
            predictor = casting.Bfloat16Cast(predictor)
        predictor = DirectResidualNormalizer(
            predictor,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"],
        )
        return predictor

    def baseline_predict_fn(inputs, targets, forcings, is_training):
        del is_training
        return _build_baseline_predictor()(
            inputs, targets_template=targets, forcings=forcings)

    def residual_loss_fn(inputs, residual_targets, forcings, is_training):
        del is_training
        return _build_residual_predictor().loss(inputs, residual_targets, forcings)

    baseline_predict = hk.transform_with_state(baseline_predict_fn)
    residual_loss = hk.transform_with_state(residual_loss_fn)

    rng = jax.random.PRNGKey(cfg.seed)
    train_indices = base_train.valid_final_input_indices(
        train_ds.sizes["time"], input_steps, cfg.target_steps)
    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        train_ds, indices=[int(train_indices[0])],
        input_steps=input_steps, target_steps=cfg.target_steps,
        task_cfg=task_cfg, dt=dt)
    print(f"[v11] sample_inputs sizes: {dict(sample_inputs.sizes)}")

    rng, k_b = jax.random.split(rng)
    baseline_params, baseline_state = baseline_predict.init(
        k_b, sample_inputs, sample_targets, sample_forcings, False)
    baseline_params, b_stats = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    print(f"[v11] baseline GC1: overlaid {b_stats.copied} params from DeepMind.")

    rng, k_r = jax.random.split(rng)
    residual_params, residual_state = residual_loss.init(
        k_r, sample_inputs, sample_targets, sample_forcings, True)
    n_residual = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[v11] residual params: {n_residual:,} (input_proj + Mamba + head, "
          f"all trainable from scratch)")

    if cfg.resume_from is not None:
        import pickle
        rp_path = Path(cfg.resume_from)
        print(f"[v11] resuming from {rp_path}")
        with rp_path.open("rb") as f:
            ck = pickle.load(f)
        residual_params = ck["residual_params"]
        if "residual_state" in ck and ck["residual_state"]:
            residual_state = ck["residual_state"]
        print(f"[v11] resumed; will start at step {cfg.start_step}")

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(residual_params)

    run_config = {
        "config": {k: getattr(cfg, k) for k in vars(cfg)},
        "model_cfg_baseline": dataclasses.asdict(model_cfg_baseline),
        "model_cfg_residual": dataclasses.asdict(model_cfg_residual),
        "input_steps": int(input_steps),
        "n_residual_params": int(n_residual),
        "n_mesh_nodes": int(n_mesh),
        "output_size": int(output_size),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    bptt = cfg.bptt_steps
    seg_len = cfg.sequential_segment_steps
    if seg_len % bptt != 0:
        raise ValueError(f"seg_len ({seg_len}) must be divisible by bptt ({bptt})")
    chunks_per_segment = seg_len // bptt

    @jax.jit
    def train_step(residual_params, residual_state, opt_state, keys,
                   inputs_list, targets_list, forcings_list):
        def f(rp):
            rs = residual_state
            losses = []
            for i in range(bptt):
                baseline_pred, _ = baseline_predict.apply(
                    baseline_params, baseline_state, keys[i],
                    inputs_list[i], targets_list[i], forcings_list[i], False)
                baseline_pred = jax.tree_util.tree_map(
                    jax.lax.stop_gradient, baseline_pred)
                residual_target = targets_list[i] - baseline_pred
                (loss_da, _scalars), rs = residual_loss.apply(
                    rp, rs, keys[i],
                    inputs_list[i], residual_target, forcings_list[i], True)
                losses.append(scalarize_loss(loss_da))
            return jnp.stack(losses).mean(), rs

        (loss, new_rs), grads = jax.value_and_grad(f, has_aux=True)(residual_params)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = opt.update(grads, opt_state, residual_params)
        new_rp = optax.apply_updates(residual_params, updates)
        new_rs = jax.tree_util.tree_map(jax.lax.stop_gradient, new_rs)
        return new_rp, new_rs, new_opt_state, loss, grad_norm

    print(f"[v11] sequential_segment_steps={seg_len}  bptt_steps={bptt}  "
          f"chunks_per_segment={chunks_per_segment}")
    segments = base_train.build_sequential_segments(train_indices, seg_len)
    print(f"[v11] {len(segments)} segments × {chunks_per_segment} chunks/segment")

    train_log = []
    seg_iter = iter(segments)
    cur_segment = next(seg_iter)
    seg_pos = 0

    for step in range(cfg.start_step, cfg.max_steps + 1):
        while seg_pos + bptt > len(cur_segment):
            try:
                cur_segment = next(seg_iter)
            except StopIteration:
                seg_iter = iter(segments)
                cur_segment = next(seg_iter)
            seg_pos = 0
            residual_state = jax.tree_util.tree_map(jnp.zeros_like, residual_state)

        chunk_indices = [int(cur_segment[seg_pos + i]) for i in range(bptt)]
        seg_pos += bptt

        rng, *split_keys = jax.random.split(rng, bptt + 1)
        keys = jnp.stack(split_keys)

        inp_list, tgt_list, frc_list = [], [], []
        for idx in chunk_indices:
            inp, tgt, frc = base_train.build_batch_from_indices(
                train_ds, indices=[idx], input_steps=input_steps,
                target_steps=cfg.target_steps, task_cfg=task_cfg, dt=dt)
            inp_list.append(inp); tgt_list.append(tgt); frc_list.append(frc)

        t0 = time.time()
        residual_params, residual_state, opt_state, loss, grad_norm = train_step(
            residual_params, residual_state, opt_state, keys,
            tuple(inp_list), tuple(tgt_list), tuple(frc_list))
        step_t = time.time() - t0

        if step <= 5 or step % 10 == 0:
            print(f"step {step}/{cfg.max_steps} loss {float(loss):.5f} "
                  f"grad_norm {float(grad_norm):.4f} step_time {step_t:.2f}s")

        train_log.append({"step": step, "loss": float(loss),
                          "grad_norm": float(grad_norm), "step_time": float(step_t)})

        if step % cfg.checkpoint_every == 0:
            import pickle
            ckpt_path = out_dir / f"v11_residual_step{step}.pkl"
            with ckpt_path.open("wb") as f:
                pickle.dump({"residual_params": residual_params,
                             "residual_state": residual_state}, f)
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_log, f, indent=2)
            print(f"[v11] saved ckpt {ckpt_path}")

    print(f"[v11] training done after {cfg.max_steps} steps")


if __name__ == "__main__":
    main()
