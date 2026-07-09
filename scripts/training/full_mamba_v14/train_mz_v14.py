"""v14 = v13 architecture (full DeepMind GC2 trainable + Mamba interleaved
+ DeepMind init) + v12 fast pipeline + **specialist heads** (replaces v13's
single Linear(83->83) with two zero-init Linears: surface_head(5->5) and
upper_head(78->78)).

Channel layout in mesh2grid_gnn output (verified at task_cfg load time):
  indices 0-4:  surface vars (2m_t, MSLP, 10m_v, 10m_u, tp)
  indices 5-82: atmos channels (T, Z, U, V, W, q each x 13 levels)

Specialist heads decouple surface vs atmospheric weight matrices so:
  - atmos overfit gradient cannot pull surface_head weights
  - surface overfit gradient cannot pull upper_head weights

Everything else identical to v13.
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
from src.data.prepared_array import PreparedArrayStore  # noqa: E402

from scripts.training.full_mamba_v9.train_mz_v9 import _attach_temporal  # noqa: E402

# Channel layout in target_variables order (verified):
#   target_variables = [
#     0: 2m_temperature,                                  ← surface
#     1: mean_sea_level_pressure,                         ← surface
#     2: 10m_v_component_of_wind,                         ← surface
#     3: 10m_u_component_of_wind,                         ← surface
#     4: total_precipitation_6hr,                         ← surface
#     5: temperature       (atmos, 13 levels: 5..17)
#     6: geopotential      (atmos, 13 levels: 18..30)
#     7: u_component_of_wind  (atmos, 13 levels: 31..43)
#     8: v_component_of_wind  (atmos, 13 levels: 44..56)
#     9: vertical_velocity (atmos, 13 levels: 57..69)
#     10: specific_humidity (atmos, 13 levels: 70..82)
#   ]
N_SURFACE_CHANNELS = 5
N_ATMOS_CHANNELS = 78


class GCResidualWithSpecialistHeads(gc.GraphCast):
    """gc.GraphCast subclass with two specialist zero-init heads at the
    output layer. Architecture is otherwise identical to v9's
    GCResidualWithZeroHead.

    Channels [0:5]   -> surface_head Linear(5->5,  w=0, b=0)
    Channels [5:83]  -> upper_head   Linear(78->78, w=0, b=0)
    Outputs concatenated back to [N_grid, B, 83] in the original order.
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
        # output_grid_nodes: [num_grid_nodes, batch, output_size=83]

        # Sanity-check we have exactly 5 surface + 78 atmospheric.
        out_size = output_grid_nodes.shape[-1]
        if out_size != N_SURFACE_CHANNELS + N_ATMOS_CHANNELS:
            raise ValueError(
                f"Expected output_size={N_SURFACE_CHANNELS + N_ATMOS_CHANNELS}, "
                f"got {out_size}; channel layout assumption violated."
            )

        surface_in = output_grid_nodes[..., :N_SURFACE_CHANNELS]
        atmos_in = output_grid_nodes[..., N_SURFACE_CHANNELS:]

        surface_head = hk.Linear(
            N_SURFACE_CHANNELS,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="surface_head",
        )
        upper_head = hk.Linear(
            N_ATMOS_CHANNELS,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="upper_head",
        )

        surface_out = surface_head(surface_in)  # [N_grid, B, 5]
        upper_out = upper_head(atmos_in)        # [N_grid, B, 78]
        residual_output = jnp.concatenate([surface_out, upper_out], axis=-1)
        return self._grid_node_outputs_to_prediction(
            residual_output, targets_template)


def parse_args():
    p = argparse.ArgumentParser(
        description="v14 = v13 architecture + specialist heads (surface + upper).")
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--residual-root", required=True)
    p.add_argument("--ckpt-in", default=base_train.DEFAULT_CKPT)
    p.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--residual-msg-steps", type=int, default=2)
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
    p.add_argument("--resume-from", default=None)
    p.add_argument("--start-step", type=int, default=1)
    return p.parse_args()


def _load_residual_metadata(residual_root: Path) -> dict:
    return json.loads((residual_root / "metadata.json").read_text())


def _load_anchor_split(residual_root: Path):
    return (
        np.load(residual_root / "anchors" / "anchor_indices.npy"),
        np.load(residual_root / "anchors" / "split_train.npy"),
        np.load(residual_root / "anchors" / "split_val.npy"),
    )


def _open_residual_memmaps(residual_root: Path, target_vars):
    return {
        v: np.load(residual_root / "residuals" / f"{v}.npy", mmap_mode="r")
        for v in target_vars
    }


def _build_residual_target_xr(residual_memmaps, target_template, *, indices_in_residual):
    out_vars = {}
    for v, arr in residual_memmaps.items():
        gathered = np.take(arr, indices_in_residual, axis=0)
        tmpl = target_template[v]
        out_vars[v] = xr.DataArray(
            gathered.astype("float32"),
            dims=tmpl.dims,
            coords={d: tmpl.coords[d] for d in tmpl.dims if d in tmpl.coords},
        )
    return xr.Dataset(out_vars, coords=target_template.coords)


def _build_segments(split_indices, seg_len):
    return [split_indices[s:s + seg_len]
            for s in range(0, len(split_indices) - seg_len + 1, seg_len)]


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)

    # Verify channel layout assumption matches target_variables ordering.
    expected_surface = ["2m_temperature", "mean_sea_level_pressure",
                        "10m_v_component_of_wind", "10m_u_component_of_wind",
                        "total_precipitation_6hr"]
    actual_first5 = list(task_cfg.target_variables)[:5]
    if actual_first5 != expected_surface:
        raise ValueError(
            f"Channel layout assumption violated: target_variables[:5]={actual_first5}, "
            f"expected={expected_surface}")

    model_cfg_residual = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.residual_msg_steps,
    )
    print(f"[v14] GC2 (residual) latent_size={cfg.width} "
          f"msg_steps={cfg.residual_msg_steps}  Mamba {cfg.temporal_location}")
    print(f"[v14] specialist heads: surface_head(5->5) + upper_head(78->78), zero-init")

    prepared_root = Path(cfg.prepared_root)
    residual_root = Path(cfg.residual_root)
    store = PreparedArrayStore(prepared_root, label="v14-source")
    store.validate(resolution=cfg.resolution, task_cfg=task_cfg)

    res_meta = _load_residual_metadata(residual_root)
    target_vars = res_meta["target_variables"]
    if list(task_cfg.target_variables) != list(target_vars):
        raise ValueError("target_variables mismatch with precomputed residuals")
    anchor_indices, train_split, val_split = _load_anchor_split(residual_root)
    residual_memmaps = _open_residual_memmaps(residual_root, target_vars)
    print(f"[v14] {len(target_vars)} target vars, {anchor_indices.size} total anchors "
          f"(train={train_split.size}, val={val_split.size})")

    norm_stats = base_train.load_stats(Path(cfg.stats_dir))
    base_train.validate_stats_coverage(task_cfg, norm_stats)
    use_bf16 = cfg.precision == "bf16"

    def _build_residual_predictor():
        p = GCResidualWithSpecialistHeads(model_cfg_residual, task_cfg)
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

    residual_loss = hk.transform_with_state(residual_loss_fn)

    rng = jax.random.PRNGKey(cfg.seed)
    dt = pd.Timedelta(np.diff(np.asarray(store.time.values).astype("datetime64[ns]"))[0])
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    target_steps = cfg.target_steps

    sample_anchor = int(anchor_indices[int(train_split[0])])
    sample_inputs, sample_targets, sample_forcings = store.build_batch_from_indices(
        indices=[sample_anchor], input_steps=input_steps,
        target_steps=target_steps, task_cfg=task_cfg, dt=dt,
    )
    rng, k = jax.random.split(rng)
    residual_params, residual_state = residual_loss.init(
        k, sample_inputs, sample_targets, sample_forcings, True)
    residual_params, r_stats = overlay_matching_params(
        residual_params, ckpt_in.params, strict=False)
    n_residual = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[v14] residual: overlaid {r_stats.copied} GC params from DeepMind, "
          f"{r_stats.initialized} fresh (Mamba + surface_head + upper_head). "
          f"Total trainable: {n_residual:,}")

    if cfg.resume_from is not None:
        import pickle
        with Path(cfg.resume_from).open("rb") as f:
            ck = pickle.load(f)
        residual_params = ck["residual_params"]
        if "residual_state" in ck and ck["residual_state"]:
            residual_state = ck["residual_state"]
        print(f"[v14] resumed from {cfg.resume_from} at step {cfg.start_step}")

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
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
        "architecture": "v9_corrected_GCResidualWithSpecialistHeads",
        "specialist_heads": {"surface_head": [5, 5], "upper_head": [78, 78]},
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
                (loss_da, _scalars), rs = residual_loss.apply(
                    rp, rs, keys[i],
                    inputs_list[i], targets_list[i], forcings_list[i], True)
                losses.append(scalarize_loss(loss_da))
            return jnp.stack(losses).mean(), rs

        (loss, new_rs), grads = jax.value_and_grad(f, has_aux=True)(residual_params)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = opt.update(grads, opt_state, residual_params)
        new_rp = optax.apply_updates(residual_params, updates)
        new_rs = jax.tree_util.tree_map(jax.lax.stop_gradient, new_rs)
        return new_rp, new_rs, new_opt_state, loss, grad_norm

    train_segments = _build_segments(train_split, seg_len)
    print(f"[v14] {len(train_segments)} segments × {chunks_per_segment} "
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

        inp_list, tgt_list, frc_list = [], [], []
        for i in range(bptt):
            t_idx = int(raw_anchor_idxs[i])
            r_idx = int(chunk_residual_idxs[i])
            inp, tgt, frc = store.build_batch_from_indices(
                indices=[t_idx], input_steps=input_steps,
                target_steps=target_steps, task_cfg=task_cfg, dt=dt,
            )
            res_tgt = _build_residual_target_xr(
                residual_memmaps, tgt,
                indices_in_residual=np.array([r_idx], dtype=np.int64),
            )
            inp_list.append(inp); tgt_list.append(res_tgt); frc_list.append(frc)

        t0 = time.time()
        residual_params, residual_state, opt_state, loss, grad_norm = train_step(
            residual_params, residual_state, opt_state, keys,
            tuple(inp_list), tuple(tgt_list), tuple(frc_list))
        loss = float(loss); grad_norm = float(grad_norm)
        step_t = time.time() - t0

        if step <= 5 or step % 10 == 0:
            print(f"step {step}/{cfg.max_steps} loss {loss:.5f} "
                  f"grad_norm {grad_norm:.4f} step_time {step_t:.2f}s")
        train_log.append({"step": step, "loss": loss, "grad_norm": grad_norm,
                          "step_time": float(step_t)})

        if step % cfg.checkpoint_every == 0:
            import pickle
            ckpt_path = out_dir / f"v14_residual_step{step}.pkl"
            with ckpt_path.open("wb") as f:
                pickle.dump({"residual_params": residual_params,
                             "residual_state": residual_state}, f)
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_log, f, indent=2)
            print(f"[v14] saved ckpt {ckpt_path}")

    print(f"[v14] training done after {cfg.max_steps} steps")


if __name__ == "__main__":
    main()
