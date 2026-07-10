"""v9: asymmetric residual_mamba on DeepMind public GC_small with a fresh
zero-init residual head (replacing v8's scalar gate).

Architecture (per-cycle):
  GC1 baseline (FROZEN, full DeepMind public GC_small, msg_steps=16):
    inputs --> encoder --> 16 msg_steps --> decoder --> baseline_pred

  GC2 residual branch (frozen GC layers + Mamba interleaved + new head):
    inputs --> encoder --> N msg_steps with Mamba interleaved
            --> mesh2grid GNN --> output_grid_nodes [N_grid, B, output_size]
            --> RESIDUAL HEAD (Linear w_init=0, b_init=0)  <-- fresh, trainable
            --> grid_node_outputs_to_prediction --> residual_pred

  prediction = baseline_pred + residual_pred
  Loss      = ||residual_pred - (truth - baseline_pred)||^2
              (computed in normalized residual space via DirectResidualNormalizer)

Why this specific shape:
  v3 / v4 all use a per-output-channel `Linear(w=0, b=0)` as their residual
  head. At step 0, residual_pred is exactly 0 -> full_pred = baseline_pred,
  loss = baseline GC's intrinsic residual loss (small). v8 instead relied on
  zero-init Mamba out_proj alone, but that only zeros Mamba's CONTRIBUTION to
  GC2's mesh latent -- the rest of GC2 (frozen DeepMind weights) still
  produces a non-zero raw output. Loss was stuck at ~5.3 because Mamba had to
  fight the GC2 frozen-bias before learning residual dynamics.

  v9 replaces the (non-trainable, DeepMind-overlaid) implicit identity
  projection at the end of mesh2grid_gnn with a fresh zero-init Linear that
  IS trainable. At init residual_pred = 0; the head + Mamba train together.

Memory:
  Same as v8: GC2 with `--residual-msg-steps N` (default 4) cuts processor
  activation memory by 16/N relative to full DeepMind. With N=4 + bptt=8
  fits on A100 80GB.
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
    overlay_matching_params, build_trainable_labels,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="v9: asymmetric residual_mamba + zero-init residual head.")
    p.add_argument("--data-path", default=base_train.DEFAULT_DATA_PATH)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-msg-steps", type=int, default=16,
                   help="GC1 (baseline) processor depth — match DeepMind ckpt.")
    p.add_argument("--residual-msg-steps", type=int, default=4,
                   help="GC2 (residual) processor depth — shallower to save memory.")
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
    p.add_argument("--temporal-location",
                   choices=["mesh_post_encoder", "mesh_processor_interleaved"],
                   default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=None)
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
    p.add_argument("--trainable-part", choices=["all", "mamba", "graphcast"],
                   default="mamba")
    return p.parse_args()


def _attach_temporal(predictor, cfg):
    predictor._temporal_backbone = "mamba"
    predictor._temporal_location = cfg.temporal_location
    predictor._temporal_hidden_size = cfg.temporal_hidden_size
    predictor._temporal_d_inner = cfg.temporal_d_inner
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


class GCResidualWithZeroHead(gc.GraphCast):
    """gc.GraphCast subclass that, after the mesh2grid_gnn produces the
    grid output features, applies a fresh `hk.Linear(output_size, w=0, b=0)`
    residual head. Step-0 output is exactly 0 -> residual_pred = 0 ->
    full_pred = baseline_pred -> initial loss = baseline GC's residual loss.

    Naming: the head is called `temporal_residual_head` so it falls in the
    "temporal/mamba" trainable group and trains alongside Mamba.
    """

    def __call__(self, inputs, targets_template, forcings, is_training=False):
        self._maybe_init(inputs)
        # Same forward as gc.GraphCast up to and including mesh2grid_gnn.
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(
            grid_node_features)
        if (self._temporal_backbone != "none"
                and self._temporal_location == "mesh_post_encoder"):
            latent_mesh_nodes = self._run_temporal_mesh_block(
                latent_mesh_nodes, is_training=is_training)
        updated_latent_mesh_nodes = self._run_mesh_gnn(
            latent_mesh_nodes, is_training=is_training)
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes)
        # Fresh zero-init residual head on top of mesh2grid output.
        # output_grid_nodes: [num_grid_nodes, batch, output_size]
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
        latent_size=cfg.width,
        gnn_msg_steps=cfg.baseline_msg_steps,
    )
    model_cfg_residual = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.residual_msg_steps,
    )
    print(f"[v9] baseline GC1: latent={model_cfg_baseline.latent_size} "
          f"msg_steps={model_cfg_baseline.gnn_msg_steps}")
    print(f"[v9] residual GC2: latent={model_cfg_residual.latent_size} "
          f"msg_steps={model_cfg_residual.gnn_msg_steps}  (shallow)")
    print(f"[v9] Mamba: location={cfg.temporal_location} "
          f"hidden={cfg.temporal_hidden_size} d_state={cfg.temporal_d_state} "
          f"d_conv={cfg.temporal_d_conv} layers={cfg.temporal_layers}")

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
        # Subclass with zero-init residual head: at init residual_pred = 0.
        predictor = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
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
    print(f"[v9] sample_inputs sizes: {dict(sample_inputs.sizes)}")

    rng, k_b = jax.random.split(rng)
    baseline_params, baseline_state = baseline_predict.init(
        k_b, sample_inputs, sample_targets, sample_forcings, False)
    baseline_params, b_stats = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    print(f"[v9] baseline GC1: overlaid {b_stats.copied} params from DeepMind, "
          f"{b_stats.initialized} fresh (should be 0).")

    rng, k_r = jax.random.split(rng)
    residual_params, residual_state = residual_loss.init(
        k_r, sample_inputs, sample_targets, sample_forcings, True)
    residual_params, r_stats = overlay_matching_params(
        residual_params, ckpt_in.params, strict=False)
    n_residual = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[v9] residual GC2: overlaid {r_stats.copied} GC params from DeepMind "
          f"(lenient: extra {cfg.baseline_msg_steps - cfg.residual_msg_steps} "
          f"processor steps in source skipped); "
          f"{r_stats.initialized} fresh (Mamba + residual_head + any unmatched).")
    print(f"[v9] residual GC2 total params: {n_residual:,}")

    if cfg.trainable_part == "all":
        opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    else:
        labels = build_trainable_labels(residual_params, cfg.trainable_part)
        opt = optax.multi_transform(
            {"train": optax.adamw(cfg.lr, weight_decay=cfg.weight_decay),
             "freeze": optax.set_to_zero()},
            labels,
        )
        n_train = sum(1 for m in labels.values() for v in m.values() if v == "train")
        n_freeze = sum(1 for m in labels.values() for v in m.values() if v == "freeze")
        print(f"[v9] trainable_part={cfg.trainable_part}: train={n_train} freeze={n_freeze}")
    opt_state = opt.init(residual_params)

    run_config = {
        "config": {k: getattr(cfg, k) for k in vars(cfg)},
        "model_cfg_baseline": dataclasses.asdict(model_cfg_baseline),
        "model_cfg_residual": dataclasses.asdict(model_cfg_residual),
        "input_steps": int(input_steps),
        "n_residual_params": int(n_residual),
        "overlaid_baseline_gc_params": int(b_stats.copied),
        "overlaid_residual_gc_params": int(r_stats.copied),
        "fresh_init_residual_params": int(r_stats.initialized),
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

    print(f"[v9] sequential_segment_steps={seg_len}  bptt_steps={bptt}  "
          f"chunks_per_segment={chunks_per_segment}")
    segments = base_train.build_sequential_segments(train_indices, seg_len)
    print(f"[v9] {len(segments)} segments × {chunks_per_segment} chunks/segment "
          f"= {len(segments)*chunks_per_segment} optimiser updates per epoch")

    train_log = []
    seg_iter = iter(segments)
    cur_segment = next(seg_iter)
    seg_pos = 0

    for step in range(1, cfg.max_steps + 1):
        # Advance to a segment that has at least `bptt` remaining samples.
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
            ckpt_path = out_dir / f"v9_residual_step{step}.pkl"
            with ckpt_path.open("wb") as f:
                pickle.dump({"residual_params": residual_params,
                             "residual_state": residual_state}, f)
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_log, f, indent=2)
            print(f"[v9] saved ckpt {ckpt_path}")

    print(f"[v9] training done after {cfg.max_steps} steps")


if __name__ == "__main__":
    main()
