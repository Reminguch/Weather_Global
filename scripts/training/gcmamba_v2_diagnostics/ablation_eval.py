"""Mamba functional ablation: reset-state, zero-D, normal.

Per the user's "最判别" check:
- If reset-state eval loss ≈ normal eval loss → Mamba is NOT using temporal memory.
- If zero-D eval loss >> normal loss → D skip path dominates (Mamba is just a
  feed-forward correction).
- If reset-state >> normal AND zero-D ≈ normal → recurrent state path is the
  one doing the work.

This script loads a TRAINED gc_mamba checkpoint and evaluates 4 modes on N
val anchors, K=1 truth-fed bptt=24 chunks. Reports a comparison table.

Usage:
    python -m scripts.training.gcmamba_v2_diagnostics.ablation_eval \\
        --ckpt <ckpt_step{N}.npz>  --temporal-location mesh_post_processor \\
        --n-anchors 32

NOTE per user: 5-step training is far too short for this to be meaningful.
Use a checkpoint from >= 1000-2000 steps. The script reports a warning
when ckpt step is too low.
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.data.prepared_array import PreparedArrayStore  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    build_predictor,
    derive_model_config_from_checkpoint,
    load_graphcast_checkpoint,
    load_stats,
    scalarize_loss,
)
from src.models.graphcast.training.core.segments import _reset_temporal_state_lanes  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to trained gc_mamba checkpoint (npz).")
    p.add_argument("--ckpt-in",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
                           "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
                           "mesh 2to5 - precipitation input and output.npz",
                   help="DeepMind GC small (used for task config + as fallback).")
    p.add_argument("--prepared-root",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/"
                           "prepared_stream_2015_2022_v2")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
    p.add_argument("--temporal-location",
                   choices=["mesh_post_encoder", "mesh_processor_interleaved", "mesh_post_processor"],
                   default="mesh_post_processor")
    p.add_argument("--temporal-insert-count", type=int, default=1)
    p.add_argument("--temporal-d-inner", type=int, default=128)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--processor-msg-steps", type=int, default=16)
    p.add_argument("--bptt-steps", type=int, default=24)
    p.add_argument("--n-anchors", type=int, default=32,
                   help="Number of independent val anchors to average over.")
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablations", default="normal,reset_state,zero_D,reset_state+zero_D",
                   help="Comma-separated list of ablation modes to run.")
    p.add_argument("--out-json", default=None,
                   help="Optional path to dump results JSON.")
    return p.parse_args()


def _flat_npz_to_haiku_params(npz_path: Path) -> hk.Params:
    """Load checkpoint in segments_training's flat npz format -> Haiku params dict."""
    data = np.load(npz_path, allow_pickle=True)
    params: dict[str, dict[str, jnp.ndarray]] = {}
    for key in data.keys():
        if not key.startswith("params:"):
            continue
        # Format: "params:<module>:<leaf>"
        rest = key[len("params:"):]
        if ":" not in rest:
            continue
        module, leaf = rest.rsplit(":", 1)
        params.setdefault(module, {})[leaf] = jnp.asarray(data[key])
    return params


def _zero_out_D(params: hk.Params) -> hk.Params:
    """Zero the D parameter in every Mamba block. Returns a new params dict."""
    out: dict = {}
    n_zeroed = 0
    for module, leaves in params.items():
        new_leaves = dict(leaves)
        if "mamba_block" in module and "D" in new_leaves:
            new_leaves["D"] = jnp.zeros_like(new_leaves["D"])
            n_zeroed += 1
        out[module] = new_leaves
    print(f"[zero-D] zeroed D in {n_zeroed} Mamba block(s)")
    return out


def _ckpt_step_from_filename(path: Path) -> int | None:
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else None


def main():
    cfg = parse_args()
    ckpt_path = Path(cfg.ckpt)
    step = _ckpt_step_from_filename(ckpt_path)
    if step is not None and step < 500:
        print(f"\nWARNING: ckpt step={step} is too small for meaningful ablation.")
        print(f"  At step<500, Mamba has barely moved from zero-init; reset-state and")
        print(f"  zero-D will all give ~equal results because Mamba is ~identity.")
        print(f"  Re-run after >=1000 step K=1 training.\n")

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

    # --- Data ---
    store = PreparedArrayStore(cfg.prepared_root + "/res1", label="ablation-eval")
    store.validate(resolution=cfg.resolution, task_cfg=task_cfg)
    all_times = pd.DatetimeIndex(pd.to_datetime(store.time.values))
    years = all_times.year.to_numpy()
    val_indices = np.where(years == cfg.val_year)[0]
    val_store = store.split_by_time_indices(val_indices, label="val")
    dt = pd.Timedelta(np.diff(val_store.time.values)[0])

    # Sample N anchors from val that admit a full bptt_steps chunk
    rng_np = np.random.default_rng(cfg.seed)
    max_idx = val_store.sizes["time"] - cfg.bptt_steps - 2
    candidates = rng_np.choice(max_idx, size=cfg.n_anchors, replace=False)
    anchors = sorted(int(c) + 1 for c in candidates)
    print(f"Sampled {cfg.n_anchors} val anchors, bptt={cfg.bptt_steps}, K=1 truth-fed.")

    # --- Build model (loss forward) ---
    def loss_forward(inputs, targets, forcings, is_training):
        predictor = build_predictor(
            model_cfg, task_cfg, norm_stats,
            use_bf16=True,
            gradient_checkpointing=False,
            temporal_backbone="mamba",
            temporal_location=cfg.temporal_location,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank=cfg.temporal_dt_rank,
            temporal_bias=False,
            temporal_conv_bias=True,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=0.0,
            temporal_stateful=True,
            temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=True,
            memory_mode="standard",
        )
        return predictor.loss(inputs, targets, forcings)

    transformed = hk.transform_with_state(loss_forward)
    rng = jax.random.PRNGKey(cfg.seed)

    # Init for state shape; then overwrite params from ckpt
    sample_inp, sample_tgt, sample_frc = val_store.build_batch_from_indices(
        indices=[anchors[0]], input_steps=2, target_steps=1, task_cfg=task_cfg, dt=dt)
    rng, k = jax.random.split(rng)
    _, init_state = transformed.init(k, sample_inp, sample_tgt, sample_frc, True)
    print(f"Loading trained params from {ckpt_path}")
    trained_params = _flat_npz_to_haiku_params(ckpt_path)
    print(f"  loaded {sum(len(v) for v in trained_params.values())} param leaves "
          f"across {len(trained_params)} modules")

    apply_fn = jax.jit(transformed.apply, static_argnums=())

    def run_one_mode(params, reset_each_step: bool) -> float:
        """Average loss across n_anchors, K=1 truth-fed bptt rollout."""
        losses = []
        rng_local = jax.random.PRNGKey(cfg.seed + 7)
        for ai, idx in enumerate(anchors):
            inp, tgt, frc = val_store.build_batch_from_indices(
                indices=[idx], input_steps=2, target_steps=cfg.bptt_steps,
                task_cfg=task_cfg, dt=dt)
            state = init_state
            chunk_losses = []
            for bptt_i in range(cfg.bptt_steps):
                if reset_each_step:
                    reset_mask = jnp.ones((1,), dtype=jnp.bool_)
                    state = _reset_temporal_state_lanes(state, reset_mask)
                tgt_k = tgt.isel(time=slice(bptt_i, bptt_i + 1))
                frc_k = frc.isel(time=slice(bptt_i, bptt_i + 1))
                rng_local, k = jax.random.split(rng_local)
                (loss_and_diag, state) = apply_fn(params, state, k, inp, tgt_k, frc_k, True)
                chunk_losses.append(float(scalarize_loss(loss_and_diag[0])))
                # advance input window with truth (K=1)
                inp = _shift_input_window_truth(inp, tgt_k, frc_k, dt)
            losses.append(float(np.mean(chunk_losses)))
            if (ai + 1) % max(1, cfg.n_anchors // 4) == 0:
                print(f"    anchor {ai+1}/{cfg.n_anchors}: mean_loss = {losses[-1]:.4f}")
        return float(np.mean(losses)), float(np.std(losses) / np.sqrt(len(losses)))

    # --- Run ablations ---
    modes = [m.strip() for m in cfg.ablations.split(",")]
    results: dict[str, dict] = {}
    for mode in modes:
        print(f"\n=== Ablation: {mode} ===")
        reset = "reset_state" in mode
        zd = "zero_D" in mode
        params = _zero_out_D(trained_params) if zd else trained_params
        mean, sem = run_one_mode(params, reset_each_step=reset)
        results[mode] = {"mean_loss": mean, "sem_loss": sem,
                         "reset_state": reset, "zero_D": zd}
        print(f"  mean loss = {mean:.6f} +- {sem:.6f} (SEM)")

    # --- Report table ---
    base = results.get("normal", {}).get("mean_loss")
    print(f"\n{'='*78}")
    print(f"{'mode':<35}{'mean_loss':>14}{'+- SEM':>14}{'vs normal':>14}")
    print("-"*78)
    for mode in modes:
        r = results[mode]
        if base is not None and r["mean_loss"] is not None:
            delta = r["mean_loss"] - base
            pct = 100 * delta / base
            vs = f"{pct:+.2f}%"
        else:
            vs = "—"
        print(f"{mode:<35}{r['mean_loss']:>14.6f}{r['sem_loss']:>14.6f}{vs:>14}")
    print("-"*78)
    print(f"Verdict guide:")
    print(f"  reset_state ≈ normal  → Mamba state is NOT used (memory not helping)")
    print(f"  zero_D     >> normal  → D path was carrying the model (state OR D-only)")
    print(f"  reset_state >> normal AND zero_D ≈ normal → recurrent state IS doing work")

    if cfg.out_json:
        import json
        Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.out_json, "w") as f:
            json.dump({"ckpt": str(ckpt_path), "step": step,
                       "n_anchors": cfg.n_anchors, "bptt_steps": cfg.bptt_steps,
                       "temporal_location": cfg.temporal_location,
                       "results": results}, f, indent=2)
        print(f"\nsaved {cfg.out_json}")


def _shift_input_window_truth(prev_inputs, new_target_ds, forcings_next, dt):
    """K=1 truth-fed: next input = current target frame."""
    import xarray as xr
    target_time = prev_inputs.time.values[-1:] + dt
    ns = new_target_ds.assign_coords(time=target_time)
    fn = forcings_next.assign_coords(time=target_time)
    next_frame = xr.merge([ns, fn])
    if "datetime" in next_frame.coords:
        next_frame = next_frame.drop_vars("datetime")
    keys_in_next = [k for k in next_frame.data_vars if k in prev_inputs.data_vars]
    next_inputs_part = next_frame[keys_in_next]
    merged = xr.concat([prev_inputs, next_inputs_part], dim="time", data_vars="different")
    return merged.tail(time=prev_inputs.sizes["time"])


if __name__ == "__main__":
    main()
