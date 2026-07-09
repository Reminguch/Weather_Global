"""v12: v11's GCFixedMeshMamba + fast data pipeline (PreparedArrayStore +
precomputed residual_target).

Same residual model as v11. The only changes are upstream of the residual
forward:
  - Inputs come from a PreparedArrayStore (numpy memmap, no xarray/zarr in
    the inner loop).
  - residual_target = truth - baseline_GC1_pred is loaded from disk
    (precomputed once by src/data/precompute_residual_targets.py), NOT
    recomputed every step. This skips the most expensive forward in the
    train loop (16-step DeepMind GC small, ~1.5-2 s per step).

Pipeline shape:
  prepared_stream/res1/                  (memmap inputs/forcings/truth)
  precomputed_residuals/v11_setup_res1/  (memmap residual_target per var)
  ----- joined by anchor index -----
  per train step:
    inputs, residual_target, forcings, anchor_idx <- load from memmap
    residual_pred = GCFixedMeshMamba(inputs)
    loss = ||residual_pred - residual_target||^2 (DirectResidualNormalizer space)
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
from src.models.mz.meshed_mamba.mesh_ops import (  # noqa: E402
    build_grid_mesh_projections,
)
from src.data.prepared_array import PreparedArrayStore  # noqa: E402

from scripts.training.full_mamba_v11.train_mz_v11 import (  # noqa: E402
    GCFixedMeshMamba, _attach_temporal, _compute_output_size,
)


def parse_args():
    p = argparse.ArgumentParser(description="v12: v11 with fast precomputed-residual pipeline.")
    p.add_argument("--prepared-root", required=True,
                   help="Path to prepared_stream/res<N> (writer's output).")
    p.add_argument("--residual-root", required=True,
                   help="Path to precomputed_residuals/<setup>/ (precompute writer's output).")
    p.add_argument("--ckpt-in", default=base_train.DEFAULT_CKPT)
    p.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-grid-neighbors", type=int, default=6)
    p.add_argument("--n-mesh-neighbors", type=int, default=3)
    p.add_argument("--sigma-scale", type=float, default=1.0)
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
    p.add_argument("--temporal-d-inner", type=int, default=None,
                   help="Mamba SSM inner dim. None -> use hidden_size.")
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
    out = {}
    for v in target_vars:
        path = residual_root / "residuals" / f"{v}.npy"
        out[v] = np.load(path, mmap_mode="r")
    return out


def _build_residual_target_xr(
    residual_memmaps: dict,
    target_template: xr.Dataset,
    *,
    indices_in_residual: np.ndarray,
) -> xr.Dataset:
    """Gather residual_target per anchor and assemble an xarray Dataset whose
    dims/coords match `target_template` (output of PreparedArrayStore batch
    builder)."""
    out_vars = {}
    batch_size = indices_in_residual.size
    for v, arr in residual_memmaps.items():
        gathered = np.take(arr, indices_in_residual, axis=0)  # [B, target_steps, lat, lon, level?]
        tmpl = target_template[v]
        # Match dimension order of template: (batch, time, lat, lon, [level])
        out_vars[v] = xr.DataArray(
            gathered.astype("float32"),
            dims=tmpl.dims,
            coords={d: tmpl.coords[d] for d in tmpl.dims if d in tmpl.coords},
        )
    return xr.Dataset(out_vars, coords=target_template.coords)


def _build_segments_from_split_indices(split_anchor_idx: np.ndarray, seg_len: int) -> list[np.ndarray]:
    """Build sequential segments (each seg_len consecutive anchors) from a
    split's anchor_in_residual_array indices.
    Note: the input is positions WITHIN the precomputed-residual table, not
    raw timestep indices. We assume the precomputed table already has anchors
    in chronological order within each split, so consecutive entries in
    split_anchor_idx are temporally adjacent.
    """
    segments = []
    n = len(split_anchor_idx)
    for s in range(0, n - seg_len + 1, seg_len):
        segments.append(split_anchor_idx[s:s + seg_len])
    return segments


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Load metadata + checkpoint ----------
    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    output_size = _compute_output_size(task_cfg)

    model_cfg_residual = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.hidden_size,
        gnn_msg_steps=1,
    )
    print(f"[v12] residual: hidden={cfg.hidden_size} output_size={output_size}")
    print(f"[v12] Mamba: d_state={cfg.temporal_d_state} d_conv={cfg.temporal_d_conv} "
          f"layers={cfg.temporal_layers}")

    # ---------- 2) Open prepared store + residual memmaps ----------
    prepared_root = Path(cfg.prepared_root)
    residual_root = Path(cfg.residual_root)
    store = PreparedArrayStore(prepared_root, label="v12-source")
    store.validate(resolution=cfg.resolution, task_cfg=task_cfg)

    res_meta = _load_residual_metadata(residual_root)
    target_vars = res_meta["target_variables"]
    if list(task_cfg.target_variables) != list(target_vars):
        raise ValueError(
            f"Target variable mismatch:\n"
            f"  task_cfg.target_variables = {list(task_cfg.target_variables)}\n"
            f"  precomputed                 = {list(target_vars)}")
    anchor_indices, train_split, val_split = _load_anchor_split(residual_root)
    residual_memmaps = _open_residual_memmaps(residual_root, target_vars)
    print(f"[v12] {len(target_vars)} target vars, {anchor_indices.size} total anchors "
          f"(train={train_split.size}, val={val_split.size})")

    # ---------- 3) Build geometric projections (same as v11) ----------
    base_lat = np.linspace(-90.0, 90.0, int(180 / cfg.resolution) + 1)
    base_lon = np.arange(0.0, 360.0, cfg.resolution)
    proj_arrays, n_mesh = build_grid_mesh_projections(
        lat_deg=base_lat, lon_deg=base_lon, mesh_size=cfg.mesh_size,
        n_grid_neighbors=cfg.n_grid_neighbors,
        n_mesh_neighbors=cfg.n_mesh_neighbors,
        sigma_scale=cfg.sigma_scale,
    )
    g2m_idx_jax = jnp.asarray(proj_arrays["g2m_indices"], dtype=jnp.int32)
    g2m_w_jax = jnp.asarray(proj_arrays["g2m_weights"], dtype=jnp.float32)
    m2g_idx_jax = jnp.asarray(proj_arrays["m2g_indices"], dtype=jnp.int32)
    m2g_w_jax = jnp.asarray(proj_arrays["m2g_weights"], dtype=jnp.float32)
    print(f"[v12] mesh nodes M={n_mesh}")

    # ---------- 4) Build residual predictor + loss transform ----------
    norm_stats = base_train.load_stats(Path(cfg.stats_dir))
    base_train.validate_stats_coverage(task_cfg, norm_stats)
    use_bf16 = cfg.precision == "bf16"

    def _build_residual_predictor():
        p = GCFixedMeshMamba(model_cfg_residual, task_cfg)
        p._g2m_idx = g2m_idx_jax
        p._g2m_w = g2m_w_jax
        p._m2g_idx = m2g_idx_jax
        p._m2g_w = m2g_w_jax
        p._hidden_size = cfg.hidden_size
        p._output_size = output_size
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

    # ---------- 5) Init params from a sample anchor ----------
    rng = jax.random.PRNGKey(cfg.seed)
    dt = pd.Timedelta(np.diff(np.asarray(store.time.values).astype("datetime64[ns]"))[0])
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    target_steps = cfg.target_steps

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
    n_residual = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[v12] residual params: {n_residual:,} (input_proj + Mamba + head)")

    if cfg.resume_from is not None:
        import pickle
        with Path(cfg.resume_from).open("rb") as f:
            ck = pickle.load(f)
        residual_params = ck["residual_params"]
        if "residual_state" in ck and ck["residual_state"]:
            residual_state = ck["residual_state"]
        print(f"[v12] resumed from {cfg.resume_from}, start step {cfg.start_step}")

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(residual_params)

    run_config = {
        "config": {k: getattr(cfg, k) for k in vars(cfg)},
        "model_cfg_residual": dataclasses.asdict(model_cfg_residual),
        "input_steps": int(input_steps),
        "n_residual_params": int(n_residual),
        "n_mesh_nodes": int(n_mesh),
        "n_anchors_train": int(train_split.size),
        "n_anchors_val": int(val_split.size),
        "output_size": int(output_size),
        "data_pipeline": "prepared_array+precomputed_residual",
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # ---------- 6) JIT'd train step (residual ONLY; no baseline GC1) ----------
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

    # ---------- 7) Build train segments over the train-split anchors ----------
    train_segments = _build_segments_from_split_indices(train_split, seg_len)
    print(f"[v12] {len(train_segments)} segments × {chunks_per_segment} chunks/segment "
          f"= {len(train_segments)*chunks_per_segment} steps/epoch")

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

        # `cur_segment[seg_pos+i]` is an INDEX INTO precomputed-residual table.
        chunk_residual_idxs = np.asarray(
            [int(cur_segment[seg_pos + j]) for j in range(bptt)], dtype=np.int64)
        seg_pos += bptt

        # Map from residual-table index to raw timestep index for prepared store.
        raw_anchor_idxs = anchor_indices[chunk_residual_idxs]

        rng, *split_keys = jax.random.split(rng, bptt + 1)
        keys = jnp.stack(split_keys)

        inp_list, tgt_list, frc_list = [], [], []
        for i in range(bptt):
            t_idx = int(raw_anchor_idxs[i])
            r_idx = int(chunk_residual_idxs[i])
            inp, tgt, frc = store.build_batch_from_indices(
                indices=[t_idx],
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt,
            )
            # Replace tgt with PRE-COMPUTED residual target
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
            ckpt_path = out_dir / f"v12_residual_step{step}.pkl"
            with ckpt_path.open("wb") as f:
                pickle.dump({"residual_params": residual_params,
                             "residual_state": residual_state}, f)
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_log, f, indent=2)
            print(f"[v12] saved ckpt {ckpt_path}")

    print(f"[v12] training done after {cfg.max_steps} steps")


if __name__ == "__main__":
    main()
