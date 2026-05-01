#!/usr/bin/env python3
"""Run a trained MZ ckpt over the full 2022 eval set and compute per-level
MAE/RMSE for baseline vs MZ-corrected. Outputs a JSON with one entry per
(variable, level) channel.

Used to count how many of the 83 atmospheric channels (5 surface + 6 upper-air
× 13 levels) MZ improves on a paper baseline. The full GraphCast paper uses
the same metric layout but with 37 levels (5 + 6×37 = 227).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "scripts" / "training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

# v3 trainer holds the anchor-as-batch implementation of _segment_to_tensors;
# add it to the path so we can route v3 ckpts through that build path during
# eval. Falls back transparently to the legacy [T=S*K, B=1, ...] layout when
# anchor_as_batch=False, so v1 ckpts keep working.
TRAIN_V3_DIR = ROOT / "scripts" / "training" / "full_mamba_v3"
if str(TRAIN_V3_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_V3_DIR))

import train_graphcast as base_train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--num-segments", type=int, default=64)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--horizon", type=int, default=1,
        help="Forecast horizon K (1=6h only, 2=6h+12h, ...). MAE is averaged "
             "over all T=S*K positions, matching the K-step training loss.",
    )
    p.add_argument(
        "--reset-hidden", action="store_true", default=False,
        help="Ablation: zero the SSM hidden state at every step inside "
             "rollout_ar, isolating Mamba memory contribution. If a normal "
             "eval and a --reset-hidden eval produce identical metrics, "
             "Mamba is acting as a per-step residual head and its hidden "
             "state carries no useful long-range memory.",
    )
    p.add_argument(
        "--zero-prev-residual", action="store_true", default=False,
        help="Ablation: force the previous-residual encoder input to zero "
             "at every step (kills both the TF prev-residual injection and "
             "the AR self-fed residual). Combined with --reset-hidden, the "
             "model acts as a pure per-step residual head with no temporal "
             "context of any kind.",
    )
    p.add_argument(
        "--disable-state-feedback", action="store_true", default=False,
        help="Ablation: skip the state_feedback branch (state_from_feedback = "
             "bsln_n[t-1] + prev_residual * rescale_f) so cs_t is always "
             "the precomputed baseline anchor state. Tests whether the "
             "Option-2 state feedback materially shapes the SSM trajectory.",
    )
    p.add_argument(
        "--mode", choices=["tf", "ar"], default="ar",
        help=(
            "tf: anchor positions teacher-forced (training-time eval); "
            "ar: pure autoregressive self-feed (deployment honest, paper-comparable)."
        ),
    )
    p.add_argument(
        "--segment-steps", type=int, default=0,
        help="Override segment_steps from run_config. 0 = use the value the "
             "ckpt was trained with. Set explicitly to keep eval memory bounded "
             "when comparing ckpts trained with different segment_steps "
             "(e.g. K=2 ckpt has seg=16, K=4 ckpt has seg=8 — at horizon=4 the "
             "K=2 ckpt would otherwise hit T=64 and OOM)."
    )
    args = p.parse_args()
    horizon = max(1, int(args.horizon))

    run_dir = Path(args.run_dir)
    cfg_blob = json.load(open(run_dir / "run_config.json"))
    cfg_dict = cfg_blob["config"]

    class _C:
        pass
    cfg = _C()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    print(f"[eval] run_dir={run_dir}  step={args.step}")
    print(f"[eval] meshed={cfg.meshed}  full_mamba={getattr(cfg,'full_mamba',False)}")

    if cfg.meshed and getattr(cfg, "full_mamba", False):
        from src.models.mz.full_mamba import (
            MZResidualFullMambaConfig, MZResidualFullMambaMeshed,
        )
        variant = "full_mamba"
    elif cfg.meshed:
        from src.models.mz.meshed_mamba import (
            MZResidualMeshedConfig, MZResidualMeshedMamba,
        )
        variant = "meshed_simplified"
    else:
        raise NotImplementedError("Per-level eval only supported for meshed variants")
    from src.models.mz.meshed_mamba import build_grid_mesh_projections

    # Variable set
    full_variables = getattr(cfg, "full_variables", None)
    if full_variables is None:
        full_variables = len(cfg_blob.get("resolved_variables", [])) > 4
    if full_variables:
        RESOLVED_VARIABLES = (
            "2m_temperature", "mean_sea_level_pressure",
            "10m_u_component_of_wind", "10m_v_component_of_wind",
            "total_precipitation_6hr",
            "temperature", "geopotential", "u_component_of_wind",
            "v_component_of_wind", "vertical_velocity", "specific_humidity",
        )
    else:
        RESOLVED_VARIABLES = (
            "mean_sea_level_pressure", "geopotential",
            "u_component_of_wind", "v_component_of_wind",
        )
    PRESSURE_LEVEL_VARS = {
        "geopotential", "u_component_of_wind", "v_component_of_wind",
        "temperature", "vertical_velocity", "specific_humidity",
    }

    ckpt = base_train.load_graphcast_checkpoint(Path(cfg.baseline_ckpt))
    task_cfg = ckpt.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg = dataclasses.replace(
        ckpt.model_config, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
    )
    norm_stats = base_train.load_stats(Path(cfg.stats_dir))

    class _SplitCfg:
        data_path = cfg.data_path
        resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = cfg.train_start_year
        train_end_year = cfg.train_end_year

    _train_ds, eval_ds = base_train._open_local_splits(_SplitCfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    # Build cos(lat) area-proportional weights (matches GraphCast paper /
    # WeatherBench2 standard). Normalised to mean=1 so multiplying abs(error)
    # by these weights and summing keeps the same number of "effective points"
    # as uniform averaging would, but each element's contribution is scaled
    # by its true grid-cell area (cos(lat) for our 1°x1° grid which uses
    # mid-cell latitudes -89.5..89.5 and never hits the exact poles).
    lat_deg_eval = np.asarray(eval_ds.lat.values, dtype=np.float64)
    if np.any(np.isclose(np.abs(lat_deg_eval), 90.0)):
        d_lat = float(np.abs(lat_deg_eval[1] - lat_deg_eval[0]))
        pole_mask = np.isclose(np.abs(lat_deg_eval), 90.0)
        pole_w = 2.0 * np.sin(np.deg2rad(d_lat) / 4.0) ** 2
        non_pole_w = 2.0 * np.sin(np.deg2rad(d_lat) / 2.0) * np.cos(np.deg2rad(lat_deg_eval))
        lat_w = np.where(pole_mask, pole_w, non_pole_w)
    else:
        lat_w = np.cos(np.deg2rad(lat_deg_eval))
    lat_w = (lat_w / lat_w.mean()).astype(np.float32)
    print(
        f"[eval] lat weights: shape={lat_w.shape}  "
        f"min/mean/max = {lat_w.min():.3f}/{lat_w.mean():.3f}/{lat_w.max():.3f}"
    )

    pressure_levels = list(task_cfg.pressure_levels)

    def resolved_layout():
        layout, slices, cursor = [], {}, 0
        for name in RESOLVED_VARIABLES:
            width = len(pressure_levels) if name in PRESSURE_LEVEL_VARS else 1
            slices[name] = slice(cursor, cursor + width)
            layout.append(name)
            cursor += width
        return tuple(layout), slices, cursor
    feature_order, feature_slices, feature_dim = resolved_layout()

    def _stack(ds):
        chunks = []
        for name in feature_order:
            da = ds[name]
            if "level" in da.dims:
                arr = da.sel(level=pressure_levels).values.astype(np.float32)
            else:
                arr = np.asarray([float(da.values)], dtype=np.float32)
            chunks.append(arr)
        return np.concatenate(chunks, axis=0)

    diffs_std = _stack(norm_stats["diffs_stddev_by_level"])
    mean_f = _stack(norm_stats["mean_by_level"])
    std_f = _stack(norm_stats["stddev_by_level"])
    input_mean_f = jnp.asarray(mean_f, dtype=jnp.float32)
    input_std_f = jnp.asarray(std_f, dtype=jnp.float32)
    output_denorm_f = jnp.asarray(diffs_std, dtype=jnp.float32)
    residual_input_std_f = jnp.asarray(diffs_std, dtype=jnp.float32)

    # Use v3 trainer's helpers — its _segment_to_tensors supports
    # anchor_as_batch=True (the layout v3 ckpts were trained with). When
    # anchor_as_batch=False it falls back to the same [T=S*K, B=1, ...]
    # path the resume trainer uses, so v1/v2 ckpts get the same behaviour.
    import train_mz_fullmamba_v3 as mz_train

    # Resolve effective segment_steps: --segment-steps > 0 overrides the run
    # config's value (use this to bound T = seg × horizon and avoid OOM when a
    # K=2 ckpt was trained with seg=16 but we now want horizon=4 -> T=64).
    eff_segment_steps = (
        int(args.segment_steps) if args.segment_steps > 0 else int(cfg.segment_steps)
    )
    if eff_segment_steps != cfg.segment_steps:
        print(
            f"[eval] overriding segment_steps from run_config "
            f"({cfg.segment_steps}) -> CLI ({eff_segment_steps}); "
            f"horizon={horizon} -> T = {eff_segment_steps * horizon} positions/segment"
        )

    # Resolve effective layout (Mod A — anchor-as-batch).
    #   * v3 ckpts were trained with anchor_as_batch=True, fallback at K<min_k.
    #   * v1/v2 ckpts have no anchor_as_batch flag in run_config -> default False.
    # The `effective` flag must mirror v3 trainer logic:
    #   effective_anchor_as_batch = anchor_as_batch and horizon >= anchor_as_batch_min_k
    # When True, _segment_to_tensors yields [T=K, B=S, ...] with tf_mask=[1,0,...,0]
    # and rollout_ar must be called with allow_tf_at_t0=True so tf_mask[0]
    # actually injects the previous-anchor residual.
    cfg_anchor_as_batch = bool(getattr(cfg, "anchor_as_batch", False))
    cfg_anchor_as_batch_min_k = int(getattr(cfg, "anchor_as_batch_min_k", 2))
    effective_anchor_as_batch = (
        cfg_anchor_as_batch and horizon >= cfg_anchor_as_batch_min_k
    )
    print(
        f"[eval] layout: cfg.anchor_as_batch={cfg_anchor_as_batch}  "
        f"min_k={cfg_anchor_as_batch_min_k}  horizon={horizon}  "
        f"effective_anchor_as_batch={effective_anchor_as_batch}  "
        f"allow_tf_at_t0={effective_anchor_as_batch}"
    )
    print(
        f"[eval] ablations: reset_hidden={args.reset_hidden}  "
        f"zero_prev_residual={args.zero_prev_residual}  "
        f"disable_state_feedback={args.disable_state_feedback}"
    )

    # Bound the valid pool to the horizon we will actually evaluate at —
    # otherwise a horizon=2 eval can land on an index whose t+12h falls off
    # the array.
    eval_indices_raw = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, horizon)
    eval_indices = mz_train._filter_time_continuous_indices(
        eval_ds, eval_indices_raw, input_steps=input_steps,
        target_steps=horizon, dt=dt)
    eval_segments = [
        seg for seg in mz_train._time_continuous_segments(
            eval_ds, eval_indices, eff_segment_steps, dt)
        if len(seg) == eff_segment_steps
    ]
    print(f"[eval] eval_segments available = {len(eval_segments)}")
    eval_segments = eval_segments[:args.num_segments]
    print(f"[eval] using {len(eval_segments)} segments")

    def baseline_predict_fn(inputs, targets, forcings, is_training):
        predictor = base_train.build_predictor(
            model_cfg, task_cfg, norm_stats,
            use_bf16=(cfg.baseline_precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone="none", temporal_location="mesh_post_encoder",
            temporal_hidden_size=model_cfg.latent_size,
            temporal_layers=1, temporal_dropout=0.0,
        )
        return predictor(inputs, targets_template=targets, forcings=forcings,
                         is_training=is_training)
    baseline_predict = hk.transform_with_state(baseline_predict_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(eval_indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
    sample_inputs = mz_train._to_jax_dataset(sample_inputs)
    sample_targets = mz_train._to_jax_dataset(sample_targets)
    sample_forcings = mz_train._to_jax_dataset(sample_forcings)
    _, baseline_state = baseline_predict.init(
        rng, sample_inputs, sample_targets, sample_forcings, False)
    baseline_params = ckpt.params

    lat_deg = np.asarray(eval_ds.lat.values, dtype=np.float64)
    lon_deg = np.asarray(eval_ds.lon.values, dtype=np.float64)
    proj, n_mesh = build_grid_mesh_projections(
        lat_deg=lat_deg, lon_deg=lon_deg,
        mesh_size=getattr(cfg, "mz_mesh_size", 5),
        n_grid_neighbors=getattr(cfg, "n_grid_neighbors", 6),
        n_mesh_neighbors=getattr(cfg, "n_mesh_neighbors", 3),
    )

    if variant == "full_mamba":
        # v2 ckpt support: if run_config has use_specialist_heads, build the
        # model with split upper/surface heads so the param tree matches.
        use_specialist_heads = bool(getattr(cfg, "use_specialist_heads", False))
        upper_idx_t = ()
        surface_idx_t = ()
        if use_specialist_heads:
            surface_names = {
                "2m_temperature", "mean_sea_level_pressure",
                "10m_u_component_of_wind", "10m_v_component_of_wind",
                "total_precipitation_6hr",
            }
            upper_idx, surface_idx = [], []
            for name in feature_order:
                sl = feature_slices[name]
                target = surface_idx if name in surface_names else upper_idx
                target.extend(range(sl.start, sl.stop))
            upper_idx_t = tuple(upper_idx)
            surface_idx_t = tuple(surface_idx)
            print(
                f"[eval] specialist_heads on: upper_dim={len(upper_idx_t)} "
                f"surface_dim={len(surface_idx_t)} total={len(upper_idx_t)+len(surface_idx_t)}"
            )
        mz_cfg = MZResidualFullMambaConfig(
            input_size=feature_dim*2, output_size=feature_dim,
            hidden_size=cfg.hidden_size, d_state=cfg.d_state, expand=cfg.expand,
            layers=cfg.layers, dropout=getattr(cfg, "dropout", 0.0),
            a_log_init_min=cfg.a_log_init_min, a_log_init_max=cfg.a_log_init_max,
            use_specialist_heads=use_specialist_heads,
            upper_channel_indices=upper_idx_t,
            surface_channel_indices=surface_idx_t,
        )
        mk = lambda: MZResidualFullMambaMeshed(mz_cfg, n_mesh_nodes=n_mesh, **proj)
    else:
        mz_cfg = MZResidualMeshedConfig(
            input_size=feature_dim*2, output_size=feature_dim,
            hidden_size=cfg.hidden_size, layers=cfg.layers,
            dropout=getattr(cfg, "dropout", 0.0),
            a_log_init=getattr(cfg, "a_log_init", -0.1),
        )
        mk = lambda: MZResidualMeshedMamba(mz_cfg, n_mesh_nodes=n_mesh, **proj)

    # Mirror training-time normalisation/denormalisation and Option-2 state
    # feedback so eval matches what the model saw during training.
    residual_to_state_rescale_f = output_denorm_f / input_std_f

    def _normalize_inputs(seq_inputs):
        cs, pr = jnp.split(seq_inputs, 2, axis=-1)
        cs_n = (cs - input_mean_f[None,None,None,None,:]) / input_std_f[None,None,None,None,:]
        pr_n = pr / residual_input_std_f[None,None,None,None,:]
        return cs_n, pr_n

    def _pred_to_real(pred_n):
        return pred_n * output_denorm_f[None,None,None,None,:]

    def residual_objective_tf(seq_inputs, baseline_next, truth_next, tf_mask, is_training):
        """Teacher-forced K-step rollout: anchor positions (mask=1) use the
        true previous residual; intra-sample positions (mask=0) self-feed.
        Matches training's eval_step_teacher under train_mode=target_rollout.
        For K=1 (mask all-ones), this collapses to a single teacher-forced step.
        """
        current_state_n, true_prev_residual_n = _normalize_inputs(seq_inputs)
        m = mk()
        baseline_absolute_n = (
            baseline_next - input_mean_f[None,None,None,None,:]
        ) / input_std_f[None,None,None,None,:]
        pred_residual_n = m.rollout_ar(
            current_state_n,
            is_training=is_training,
            true_prev_residual_n_tblnf=true_prev_residual_n,
            tf_mask_per_step=tf_mask,
            baseline_absolute_n_tblnf=baseline_absolute_n,
            residual_to_state_rescale_f=residual_to_state_rescale_f,
            allow_tf_at_t0=effective_anchor_as_batch,
            reset_hidden_each_step=args.reset_hidden,
            zero_prev_residual_input=args.zero_prev_residual,
            disable_state_feedback=args.disable_state_feedback,
        )
        pred = _pred_to_real(pred_residual_n)
        return {"corrected": baseline_next + pred}

    def residual_objective_ar(seq_inputs, baseline_next, truth_next, is_training):
        """Pure autoregressive K-step rollout: prev_residual is always the
        model's own r_hat (deployment-honest). Matches training's
        eval_step_ar / residual_objective_ar_eval. For K=1 this collapses to
        zero-prev-residual single step (same as TF for the first step)."""
        current_state_n, _ = _normalize_inputs(seq_inputs)
        m = mk()
        baseline_absolute_n = (
            baseline_next - input_mean_f[None,None,None,None,:]
        ) / input_std_f[None,None,None,None,:]
        pred_residual_n = m.rollout_ar(
            current_state_n,
            is_training=is_training,
            true_prev_residual_n_tblnf=None,
            teacher_forcing_prob=0.0,
            baseline_absolute_n_tblnf=baseline_absolute_n,
            residual_to_state_rescale_f=residual_to_state_rescale_f,
            reset_hidden_each_step=args.reset_hidden,
            zero_prev_residual_input=args.zero_prev_residual,
            disable_state_feedback=args.disable_state_feedback,
        )
        pred = _pred_to_real(pred_residual_n)
        return {"corrected": baseline_next + pred}

    if args.mode == "tf":
        residual_model = hk.transform(residual_objective_tf)
    else:
        residual_model = hk.transform(residual_objective_ar)
    mem_params = pickle.load(open(run_dir / f"mz_residual_step{args.step}.pkl", "rb"))

    if args.mode == "tf":
        @jax.jit
        def infer_step(p, k, si, bn, tn, tfm):
            return residual_model.apply(p, k, si, bn, tn, tfm, False)
    else:
        @jax.jit
        def infer_step(p, k, si, bn, tn, tfm):  # tfm unused in AR path
            return residual_model.apply(p, k, si, bn, tn, False)

    # Accumulate per-channel sum of |error| (uniform AND lat-weighted) + sq.
    # Lat-weight tensor lat_w has shape [n_lat], normalized to mean=1; we
    # broadcast it across (T, B, lat, lon, F) before summing -> the resulting
    # weighted "count" equals the unweighted count, so MAE_latw = sum_w / count.
    #
    # Per-lead-time accumulators: when each segment has T = S * K positions
    # (S anchors × K lead-time steps), we additionally keep a (K, F) accumulator
    # so the JSON output can break improvements down by lead time, matching
    # GraphCast / WB2 scorecard convention. Aggregation: sum over (S, B, lat, lon)
    # but keep (K, F) -> per-lead-time per-channel MAE/RMSE.
    F = feature_dim
    K_lead = horizon  # number of distinct lead times per segment
    abs_err_base = np.zeros(F, dtype=np.float64)
    abs_err_corr = np.zeros(F, dtype=np.float64)
    sq_err_base = np.zeros(F, dtype=np.float64)
    sq_err_corr = np.zeros(F, dtype=np.float64)
    abs_err_base_latw = np.zeros(F, dtype=np.float64)
    abs_err_corr_latw = np.zeros(F, dtype=np.float64)
    sq_err_base_latw = np.zeros(F, dtype=np.float64)
    sq_err_corr_latw = np.zeros(F, dtype=np.float64)
    # (lead_time, channel) — only filled when K_lead > 1 to avoid duplicate work
    abs_err_base_lead_latw = np.zeros((K_lead, F), dtype=np.float64)
    abs_err_corr_lead_latw = np.zeros((K_lead, F), dtype=np.float64)
    sq_err_base_lead_latw = np.zeros((K_lead, F), dtype=np.float64)
    sq_err_corr_lead_latw = np.zeros((K_lead, F), dtype=np.float64)
    count = 0
    count_per_lead = 0  # total (S, B, lat, lon) elements per lead-time slot across segments

    lat_w_b = lat_w[None, None, :, None, None]  # broadcast over T,B,lon,F
    # For per-lead-time: lat_w broadcast over (S, B, lon, F) inside [S, K, B, lat, lon, F]
    lat_w_skblnf = lat_w[None, None, None, :, None, None]

    for i, seg in enumerate(eval_segments):
        rng, key = jax.random.split(rng)
        seq_inputs, baseline_next, truth_next, tf_mask = mz_train._segment_to_tensors(
            baseline_predict, baseline_params, baseline_state, key,
            eval_ds, seg, input_steps=input_steps, task_cfg=task_cfg,
            dt=dt, feature_order=feature_order, target_steps=horizon,
            anchor_as_batch=effective_anchor_as_batch)
        out = infer_step(mem_params, key, seq_inputs, baseline_next, truth_next, tf_mask)
        b = np.asarray(baseline_next, dtype=np.float32)
        c = np.asarray(out["corrected"], dtype=np.float32)
        t = np.asarray(truth_next, dtype=np.float32)
        diff_b = b - t
        diff_c = c - t
        # Layout depends on effective_anchor_as_batch:
        #   anchor_as_batch=True  -> [T=K_lead, B=S, lat, lon, F]
        #     T axis IS the lead-time axis directly; no reshape needed.
        #   anchor_as_batch=False -> [T=S*K, B=1, lat, lon, F]
        #     T axis interleaves (anchor, lead-step) row-major; reshape T->(S,K).
        T_total = b.shape[0]
        B_eff = b.shape[1]
        if effective_anchor_as_batch:
            if T_total != K_lead:
                raise RuntimeError(
                    f"anchor_as_batch=True expects T_total==K_lead but got "
                    f"T_total={T_total}, K_lead={K_lead}"
                )
            S_anchors = B_eff
            # ab/sq tensors shaped [K, S, lat, lon, F]; per-lead axis is axis 0.
            per_lead_axes = (1, 2, 3)  # sum over (B=S, lat, lon)
        else:
            if T_total % K_lead != 0:
                raise RuntimeError(
                    f"T_total={T_total} not divisible by K_lead={K_lead}; cannot reshape to (S, K)"
                )
            S_anchors = T_total // K_lead
        n = b.shape[0] * b.shape[1] * b.shape[2] * b.shape[3] * b.shape[4]
        ab_b = np.abs(diff_b)
        ab_c = np.abs(diff_c)
        sq_b = np.square(diff_b)
        sq_c = np.square(diff_c)
        # Uniform (unweighted) — T*B regardless of layout
        abs_err_base += ab_b.reshape(-1, F).sum(axis=0)
        abs_err_corr += ab_c.reshape(-1, F).sum(axis=0)
        sq_err_base += sq_b.reshape(-1, F).sum(axis=0)
        sq_err_corr += sq_c.reshape(-1, F).sum(axis=0)
        # Lat-weighted: multiply by lat_w broadcast across (T, B, lon, F)
        abs_err_base_latw += (ab_b * lat_w_b).reshape(-1, F).sum(axis=0)
        abs_err_corr_latw += (ab_c * lat_w_b).reshape(-1, F).sum(axis=0)
        sq_err_base_latw += (sq_b * lat_w_b).reshape(-1, F).sum(axis=0)
        sq_err_corr_latw += (sq_c * lat_w_b).reshape(-1, F).sum(axis=0)
        # Per-lead-time lat-weighted accumulator. Two layouts:
        if effective_anchor_as_batch:
            # [K, S, lat, lon, F]: axis 0 is lead time. Keep K, F; sum over B,lat,lon.
            lat_w_klbnf = lat_w[None, None, :, None, None]   # broadcast over K,B,lon,F
            abs_err_base_lead_latw += (ab_b * lat_w_klbnf).sum(axis=(1, 2, 3))
            abs_err_corr_lead_latw += (ab_c * lat_w_klbnf).sum(axis=(1, 2, 3))
            sq_err_base_lead_latw += (sq_b * lat_w_klbnf).sum(axis=(1, 2, 3))
            sq_err_corr_lead_latw += (sq_c * lat_w_klbnf).sum(axis=(1, 2, 3))
        else:
            # [S*K, 1, lat, lon, F] -> reshape T to (S, K), sum over (S, B, lat, lon)
            ab_b_skblnf = ab_b.reshape(S_anchors, K_lead, *ab_b.shape[1:])
            ab_c_skblnf = ab_c.reshape(S_anchors, K_lead, *ab_c.shape[1:])
            sq_b_skblnf = sq_b.reshape(S_anchors, K_lead, *sq_b.shape[1:])
            sq_c_skblnf = sq_c.reshape(S_anchors, K_lead, *sq_c.shape[1:])
            abs_err_base_lead_latw += (ab_b_skblnf * lat_w_skblnf).sum(axis=(0, 2, 3, 4))
            abs_err_corr_lead_latw += (ab_c_skblnf * lat_w_skblnf).sum(axis=(0, 2, 3, 4))
            sq_err_base_lead_latw += (sq_b_skblnf * lat_w_skblnf).sum(axis=(0, 2, 3, 4))
            sq_err_corr_lead_latw += (sq_c_skblnf * lat_w_skblnf).sum(axis=(0, 2, 3, 4))
        # Count: total over (T, B, lat, lon) for global; (S, lat, lon) per lead slot.
        count += n
        count_per_lead += S_anchors * b.shape[2] * b.shape[3]
        if i == 0 or (i+1) % 8 == 0:
            print(f"  segment {i+1}/{len(eval_segments)}  "
                  f"shape={tuple(b.shape)}  S_anchors={S_anchors}  layout={'anchor_as_batch' if effective_anchor_as_batch else 'legacy'}")

    base_mae = abs_err_base / count
    corr_mae = abs_err_corr / count
    base_rmse = np.sqrt(sq_err_base / count)
    corr_rmse = np.sqrt(sq_err_corr / count)
    # Lat-weighted: weights have mean 1 and we broadcast over (T,B,lon,F)
    # which contributes the SAME total count as uniform, so divide by count
    # to get the area-weighted mean (this matches GraphCast / WB2 convention).
    base_mae_latw = abs_err_base_latw / count
    corr_mae_latw = abs_err_corr_latw / count
    base_rmse_latw = np.sqrt(sq_err_base_latw / count)
    corr_rmse_latw = np.sqrt(sq_err_corr_latw / count)
    # Per-lead-time lat-weighted (shape [K, F])
    base_mae_lead_latw = abs_err_base_lead_latw / count_per_lead
    corr_mae_lead_latw = abs_err_corr_lead_latw / count_per_lead
    base_rmse_lead_latw = np.sqrt(sq_err_base_lead_latw / count_per_lead)
    corr_rmse_lead_latw = np.sqrt(sq_err_corr_lead_latw / count_per_lead)

    # Build per-channel breakdown: (variable_name, level_or_None, channel_idx)
    per_channel = []
    for name in feature_order:
        sl = feature_slices[name]
        for i, ci in enumerate(range(sl.start, sl.stop)):
            level = pressure_levels[i] if name in PRESSURE_LEVEL_VARS else None
            per_channel.append({
                "variable": name,
                "level": level,
                "channel_idx": ci,
                # Uniform (legacy)
                "baseline_MAE": float(base_mae[ci]),
                "corrected_MAE": float(corr_mae[ci]),
                "baseline_RMSE": float(base_rmse[ci]),
                "corrected_RMSE": float(corr_rmse[ci]),
                "delta_MAE_pct": 100 * float(base_mae[ci] - corr_mae[ci]) / float(base_mae[ci]),
                # Lat-weighted (GraphCast / WB2 convention)
                "baseline_MAE_latw": float(base_mae_latw[ci]),
                "corrected_MAE_latw": float(corr_mae_latw[ci]),
                "baseline_RMSE_latw": float(base_rmse_latw[ci]),
                "corrected_RMSE_latw": float(corr_rmse_latw[ci]),
                "delta_MAE_pct_latw": 100 * float(base_mae_latw[ci] - corr_mae_latw[ci]) / float(base_mae_latw[ci]),
                "delta_RMSE_pct_latw": 100 * float(base_rmse_latw[ci] - corr_rmse_latw[ci]) / float(base_rmse_latw[ci]),
            })

    n_better = sum(1 for c in per_channel if c["delta_MAE_pct"] > 0)
    n_worse = sum(1 for c in per_channel if c["delta_MAE_pct"] < 0)
    n_better_latw = sum(1 for c in per_channel if c["delta_MAE_pct_latw"] > 0)
    n_worse_latw = sum(1 for c in per_channel if c["delta_MAE_pct_latw"] < 0)

    # Per-channel per-lead-time scorecard (GraphCast / WB2 convention).
    # Each entry is a (variable, level, lead_time) target with lat-weighted
    # MAE/RMSE -> Δ%. Total entries = K_lead × F = horizon × 83 channels.
    per_channel_per_leadtime = []
    for k in range(K_lead):
        lead_h = (k + 1) * 6  # +6h, +12h, ..., +K_lead*6h
        for name in feature_order:
            sl = feature_slices[name]
            for i, ci in enumerate(range(sl.start, sl.stop)):
                level = pressure_levels[i] if name in PRESSURE_LEVEL_VARS else None
                bm = float(base_mae_lead_latw[k, ci])
                cm = float(corr_mae_lead_latw[k, ci])
                br = float(base_rmse_lead_latw[k, ci])
                cr = float(corr_rmse_lead_latw[k, ci])
                per_channel_per_leadtime.append({
                    "variable": name,
                    "level": level,
                    "channel_idx": ci,
                    "lead_time_h": lead_h,
                    "lead_time_idx": k,
                    "baseline_MAE_latw": bm,
                    "corrected_MAE_latw": cm,
                    "baseline_RMSE_latw": br,
                    "corrected_RMSE_latw": cr,
                    "delta_MAE_pct_latw": 100 * (bm - cm) / bm if bm > 0 else 0.0,
                    "delta_RMSE_pct_latw": 100 * (br - cr) / br if br > 0 else 0.0,
                })
    n_targets_total = len(per_channel_per_leadtime)
    n_targets_better_RMSE_latw = sum(
        1 for e in per_channel_per_leadtime if e["delta_RMSE_pct_latw"] > 0
    )
    n_targets_better_MAE_latw = sum(
        1 for e in per_channel_per_leadtime if e["delta_MAE_pct_latw"] > 0
    )

    payload = {
        "run_dir": str(run_dir),
        "step": args.step,
        "horizon": horizon,
        "mode": args.mode,
        "segment_steps_used": eff_segment_steps,
        "num_segments": len(eval_segments),
        "n_channels_total": len(per_channel),
        "n_channels_better": n_better,
        "n_channels_worse": n_worse,
        "n_channels_better_latw": n_better_latw,
        "n_channels_worse_latw": n_worse_latw,
        "n_lead_times": K_lead,
        "n_targets_total": n_targets_total,
        "n_targets_better_RMSE_latw": n_targets_better_RMSE_latw,
        "n_targets_better_MAE_latw": n_targets_better_MAE_latw,
        "per_channel": per_channel,
        "per_channel_per_leadtime": per_channel_per_leadtime,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n=== summary ===")
    print(f"  total channels: {len(per_channel)}")
    print(f"  uniform:       improved={n_better}/{len(per_channel)}  degraded={n_worse}")
    print(f"  lat-weighted:  improved={n_better_latw}/{len(per_channel)}  degraded={n_worse_latw}")
    mean_d = np.mean([c["delta_MAE_pct"] for c in per_channel])
    mean_d_latw = np.mean([c["delta_MAE_pct_latw"] for c in per_channel])
    print(f"  simple 83-ch mean Δ MAE: uniform={mean_d:+.3f}%   lat-weighted={mean_d_latw:+.3f}%")
    if K_lead > 1:
        print(f"\n  Per-lead-time scorecard (lat-weighted):")
        print(f"    total targets (K x F): {n_targets_total}")
        print(f"    targets improved RMSE: {n_targets_better_RMSE_latw}/{n_targets_total}")
        print(f"    targets improved MAE:  {n_targets_better_MAE_latw}/{n_targets_total}")
        # Per-lead-time aggregate
        for k in range(K_lead):
            lead_h = (k + 1) * 6
            entries = [e for e in per_channel_per_leadtime if e["lead_time_idx"] == k]
            mean_rmse = np.mean([e["delta_RMSE_pct_latw"] for e in entries])
            n_imp = sum(1 for e in entries if e["delta_RMSE_pct_latw"] > 0)
            print(f"    +{lead_h:3d}h: mean ΔRMSE_latw {mean_rmse:+.3f}%   improved {n_imp}/{len(entries)}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
