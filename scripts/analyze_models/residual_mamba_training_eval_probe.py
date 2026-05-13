#!/usr/bin/env python3
"""Compare residual-Mamba training-eval loss with runtime loss on one segment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import haiku as hk
import jax
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if GRAPHCAST_LOCAL.exists() and str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))

from graphcast import checkpoint, graphcast, xarray_jax  # noqa: E402

from scripts.analyze_models.legacy.analysis_metrics import (  # noqa: E402
    GRAPHCAST_PER_VARIABLE_WEIGHTS,
    normalized_weighted_mse_allvars,
)
from scripts.analyze_models.unified_resolution_eval import DEFAULT_PREPARED_DATA_ROOT, HOURS_PER_STEP, STATS_DIR, _load_stats  # noqa: E402
from src.models.graphcast.runtime import _dataset_to_numpy, load_run_config  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    advance_residual_inputs,
    build_residual_correction_predictor,
    build_zero_residual_inputs,
    reset_residual_input_lanes,
)
from src.models.graphcast.training.core.prepared_data import open_prepared_store, select_prepared_eval_window  # noqa: E402
from src.models.graphcast.training.core.segments import (  # noqa: E402
    SegmentBlockBatchLoader,
    _reset_temporal_state_lanes,
    build_full_segments,
    iter_eval_segment_chunk_infos,
    valid_contiguous_final_input_indices,
)
from src.models.mamba.residual_mamba.runtime import _build_residual_rollout_bundle  # noqa: E402
from src.models.mamba.residual_mamba.training.model import (  # noqa: E402
    build_eval_loss_transform,
    build_predict_transform,
    _use_zero_init_temporal_out,
    compute_residual_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data-root", type=Path, default=ROOT / DEFAULT_PREPARED_DATA_ROOT)
    parser.add_argument("--stats-dir", type=Path, default=ROOT / STATS_DIR)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--eval-year", type=int, default=None)
    parser.add_argument("--segment-position", type=int, default=28)
    parser.add_argument(
        "--segment-positions",
        type=str,
        default=None,
        help="Comma-separated segment positions to evaluate as one batch; overrides --segment-position.",
    )
    parser.add_argument("--len-segment", type=int, default=30)
    parser.add_argument("--bptt-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _cfg_from_run_config(run_cfg: dict) -> SimpleNamespace:
    temporal = run_cfg.get("temporal_config", {})
    return SimpleNamespace(
        precision=run_cfg.get("precision", "bf16"),
        temporal_backbone=temporal.get("backbone", "none"),
        temporal_location=temporal.get("location", "mesh_processor_interleaved"),
        temporal_d_inner=temporal.get("d_inner"),
        temporal_d_state=temporal.get("d_state", 16),
        temporal_d_conv=temporal.get("d_conv", 4),
        temporal_dt_rank=temporal.get("dt_rank", "auto"),
        temporal_bias=temporal.get("bias", False),
        temporal_conv_bias=temporal.get("conv_bias", True),
        temporal_layers=temporal.get("layers", 1),
        temporal_dropout=temporal.get("dropout", 0.0),
        temporal_stateful=bool(temporal.get("stateful", False)),
        temporal_insert_count=temporal.get("insert_count"),
    )


def _load_baseline_checkpoint(run_cfg: dict):
    baseline_path = run_cfg.get("residual_training", {}).get("baseline_checkpoint")
    if not baseline_path:
        raise ValueError("Residual checkpoint run_config is missing residual_training.baseline_checkpoint")
    baseline_path = Path(baseline_path)
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    with baseline_path.open("rb") as f:
        return baseline_path, checkpoint.load(f, graphcast.CheckPoint)


def _build_training_predict_transform(
    model_cfg: graphcast.ModelConfig,
    task_cfg: graphcast.TaskConfig,
    norm_stats: dict[str, xr.Dataset],
    cfg: SimpleNamespace,
) -> hk.TransformedWithState:
    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_residual_correction_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone=cfg.temporal_backbone,
            temporal_location=cfg.temporal_location,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank=cfg.temporal_dt_rank,
            temporal_bias=cfg.temporal_bias,
            temporal_conv_bias=cfg.temporal_conv_bias,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=cfg.temporal_dropout,
            temporal_stateful=cfg.temporal_stateful,
            temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=_use_zero_init_temporal_out(cfg, cfg.temporal_backbone),
        )
        return predictor(inputs, targets_template=targets, forcings=forcings, is_training=True)

    return hk.transform_with_state(forward_fn)


def _weighted_loss(pred: xr.Dataset, target: xr.Dataset, stats: dict[str, xr.Dataset]) -> float:
    values = _weighted_loss_values(pred, target, stats)
    return float(np.asarray(values, dtype=np.float64).mean())


def _weighted_loss_values(pred: xr.Dataset, target: xr.Dataset, stats: dict[str, xr.Dataset]) -> np.ndarray:
    pred = _dataset_to_numpy(pred)
    target = _dataset_to_numpy(target)
    loss = normalized_weighted_mse_allvars(
        pred,
        target,
        per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
        use_latitude_weights=True,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    return np.asarray(loss.values, dtype=np.float64).reshape(-1)


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    run_cfg = load_run_config(ckpt_path)
    cfg = _cfg_from_run_config(run_cfg)
    stats = _load_stats(args.stats_dir)
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    baseline_path, baseline_ckpt = _load_baseline_checkpoint(run_cfg)

    task_cfg = ckpt_obj.task_config
    model_cfg = ckpt_obj.model_config
    residual_eval_transform = build_eval_loss_transform(
        model_cfg,
        task_cfg,
        stats,
        cfg,
        temporal_backbone=cfg.temporal_backbone,
        temporal_location=cfg.temporal_location,
        temporal_d_inner=cfg.temporal_d_inner,
        temporal_d_state=cfg.temporal_d_state,
        temporal_d_conv=cfg.temporal_d_conv,
        temporal_dt_rank=cfg.temporal_dt_rank,
        temporal_bias=cfg.temporal_bias,
        temporal_conv_bias=cfg.temporal_conv_bias,
        temporal_layers=cfg.temporal_layers,
        temporal_dropout=cfg.temporal_dropout,
        temporal_stateful=cfg.temporal_stateful,
    )
    residual_training_predict_transform = _build_training_predict_transform(
        model_cfg,
        task_cfg,
        stats,
        cfg,
    )
    baseline_predict_transform = build_predict_transform(
        baseline_ckpt.model_config,
        task_cfg,
        stats,
        cfg,
        temporal_backbone="none",
        temporal_location="mesh_post_encoder",
        temporal_d_inner=None,
        temporal_d_state=cfg.temporal_d_state,
        temporal_d_conv=cfg.temporal_d_conv,
        temporal_dt_rank="auto",
        temporal_bias=False,
        temporal_conv_bias=True,
        temporal_layers=1,
        temporal_dropout=0.0,
        temporal_stateful=False,
    )
    residual_bundle, _baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)

    store0 = open_prepared_store(args.prepared_data_root, args.resolution, task_cfg, label="training-eval-probe")
    selected = select_prepared_eval_window(
        store0,
        eval_year=int(args.eval_year if args.eval_year is not None else run_cfg.get("val_year", 2022)),
    )
    store = selected.store
    eval_indices = valid_contiguous_final_input_indices(
        store,
        input_steps=2,
        target_steps=1,
        dt=pd.Timedelta(hours=HOURS_PER_STEP),
    )
    segments = build_full_segments(eval_indices, args.len_segment)
    if args.segment_positions is None:
        segment_positions = [int(args.segment_position)]
    else:
        segment_positions = [int(piece) for piece in args.segment_positions.split(",") if piece.strip()]
    if not segment_positions:
        raise ValueError("No segment positions requested.")
    for segment_position in segment_positions:
        if segment_position < 0 or segment_position >= len(segments):
            raise IndexError(f"segment-position {segment_position} outside 0..{len(segments) - 1}")
    selected_segments = [segments[position] for position in segment_positions]
    batch_size = len(selected_segments)
    chunks = list(
        iter_eval_segment_chunk_infos(
            selected_segments,
            batch_size=batch_size,
            bptt_steps=args.bptt_steps,
            segment_ids=np.arange(batch_size, dtype=np.int64),
        )
    )
    loader = SegmentBlockBatchLoader(
        store,
        selected_segments,
        input_steps=2,
        target_steps=1,
        task_cfg=task_cfg,
        dt=pd.Timedelta(hours=HOURS_PER_STEP),
        label="training-eval-probe",
    )

    rng = jax.random.PRNGKey(args.seed)
    first_inputs, first_targets, first_forcings, _stats = loader.load_chunk(chunks[0])
    _, baseline_state = baseline_predict_transform.init(
        rng,
        first_inputs[0],
        first_targets[0],
        first_forcings[0],
        False,
    )
    residual_inputs_train_loss = build_zero_residual_inputs(first_inputs[0], first_targets[0])
    _, residual_state_train = residual_eval_transform.init(
        rng,
        residual_inputs_train_loss,
        first_targets[0],
        first_forcings[0],
        False,
    )
    residual_inputs_train_pred = build_zero_residual_inputs(first_inputs[0], first_targets[0])
    _, residual_state_train_pred = residual_training_predict_transform.init(
        rng,
        residual_inputs_train_pred,
        first_targets[0],
        first_forcings[0],
        False,
    )
    residual_inputs_runtime = build_zero_residual_inputs(first_inputs[0], first_targets[0])
    _, residual_state_runtime = residual_bundle["transformed"].init(
        rng,
        residual_inputs_runtime,
        first_targets[0],
        first_forcings[0],
    )

    records: list[dict] = []
    time_index = pd.DatetimeIndex(pd.to_datetime(store.time.values))
    for chunk_i, chunk in enumerate(chunks):
        chunk_inputs, chunk_targets, chunk_forcings, _chunk_load_stats = loader.load_chunk(chunk)
        reset_mask = jax.numpy.asarray(chunk.reset_mask)
        residual_state_train = _reset_temporal_state_lanes(residual_state_train, reset_mask)
        residual_state_train_pred = _reset_temporal_state_lanes(residual_state_train_pred, reset_mask)
        residual_state_runtime = _reset_temporal_state_lanes(residual_state_runtime, reset_mask)
        residual_inputs_train_loss = reset_residual_input_lanes(
            residual_inputs_train_loss,
            chunk_targets[0],
            reset_mask,
        )
        residual_inputs_train_pred = reset_residual_input_lanes(
            residual_inputs_train_pred,
            chunk_targets[0],
            reset_mask,
        )
        residual_inputs_runtime = reset_residual_input_lanes(residual_inputs_runtime, chunk_targets[0], reset_mask)

        keys = jax.random.split(jax.random.fold_in(rng, chunk_i), args.bptt_steps * 3)
        for bptt_i in range(args.bptt_steps):
            baseline_preds, _ = baseline_predict_transform.apply(
                baseline_ckpt.params,
                baseline_state,
                keys[3 * bptt_i],
                chunk_inputs[bptt_i],
                chunk_targets[bptt_i],
                chunk_forcings[bptt_i],
                False,
            )
            residual_targets = compute_residual_targets(chunk_targets[bptt_i], baseline_preds)
            loss_and_diag, residual_state_train = residual_eval_transform.apply(
                ckpt_obj.params,
                residual_state_train,
                keys[3 * bptt_i + 1],
                residual_inputs_train_loss,
                residual_targets,
                chunk_forcings[bptt_i],
                False,
            )
            training_pred, residual_state_train_pred = residual_training_predict_transform.apply(
                ckpt_obj.params,
                residual_state_train_pred,
                keys[3 * bptt_i + 1],
                residual_inputs_train_pred,
                residual_targets,
                chunk_forcings[bptt_i],
                False,
            )
            train_loss_values = np.asarray(jax.device_get(xarray_jax.unwrap_data(loss_and_diag[0]))).reshape(-1)
            training_allvars_values = _weighted_loss_values(training_pred, residual_targets, stats)
            runtime_pred, residual_state_runtime = residual_bundle["transformed"].apply(
                residual_bundle["params"],
                residual_state_runtime,
                keys[3 * bptt_i + 2],
                residual_inputs_runtime,
                residual_targets,
                chunk_forcings[bptt_i],
            )
            runtime_values = _weighted_loss_values(runtime_pred, residual_targets, stats)
            baseline_values = _weighted_loss_values(baseline_preds, chunk_targets[bptt_i], stats)
            reset_mask_np = np.asarray(chunk.reset_mask, dtype=bool)
            for lane_i in range(batch_size):
                final_input_idx = int(chunk.chunk_indices[bptt_i][lane_i])
                target_idx = final_input_idx + 1
                scored_by_training_eval = not (bool(reset_mask_np[lane_i]) and bptt_i == 0)
                train_loss = float(train_loss_values[lane_i])
                training_allvars_loss = float(training_allvars_values[lane_i])
                runtime_loss = float(runtime_values[lane_i])
                records.append(
                    {
                        "chunk_i": int(chunk_i),
                        "lane_i": int(lane_i),
                        "segment_position": int(segment_positions[lane_i]),
                        "offset": int(chunk.lane_offsets[lane_i] + bptt_i),
                        "final_input_index": final_input_idx,
                        "target_time": str(time_index[target_idx]),
                        "scored_by_training_eval": bool(scored_by_training_eval),
                        "training_graphcast_loss": train_loss,
                        "training_allvars_loss": training_allvars_loss,
                        "runtime_residual_loss": runtime_loss,
                        "baseline_only_loss": float(baseline_values[lane_i]),
                        "runtime_minus_training_allvars": runtime_loss - training_allvars_loss,
                        "training_allvars_minus_graphcast": training_allvars_loss - train_loss,
                    }
                )
            residual_inputs_train_loss = advance_residual_inputs(residual_inputs_train_loss, residual_targets)
            residual_inputs_train_pred = advance_residual_inputs(residual_inputs_train_pred, residual_targets)
            residual_inputs_runtime = advance_residual_inputs(residual_inputs_runtime, residual_targets)

    scored = [record for record in records if record["scored_by_training_eval"]]
    by_segment: dict[str, dict[str, float]] = {}
    for segment_position in segment_positions:
        segment_records = [r for r in scored if r["segment_position"] == segment_position]
        if not segment_records:
            continue
        by_segment[str(segment_position)] = {
            "training_graphcast_mean": float(np.mean([r["training_graphcast_loss"] for r in segment_records])),
            "training_allvars_mean": float(np.mean([r["training_allvars_loss"] for r in segment_records])),
            "runtime_mean": float(np.mean([r["runtime_residual_loss"] for r in segment_records])),
            "baseline_mean": float(np.mean([r["baseline_only_loss"] for r in segment_records])),
        }
    payload = {
        "metadata": {
            "checkpoint": str(ckpt_path),
            "baseline_checkpoint": str(baseline_path),
            "eval_start": selected.eval_start,
            "eval_end": selected.eval_end,
            "segment_positions": [int(position) for position in segment_positions],
            "segment_starts": [str(time_index[int(segment[0])]) for segment in selected_segments],
            "segment_end_targets": [str(time_index[int(segment[-1]) + 1]) for segment in selected_segments],
            "batch_size": int(batch_size),
            "len_segment": int(args.len_segment),
            "bptt_steps": int(args.bptt_steps),
            "scored_training_graphcast_mean": float(np.mean([r["training_graphcast_loss"] for r in scored])),
            "scored_training_allvars_mean": float(np.mean([r["training_allvars_loss"] for r in scored])),
            "scored_runtime_mean": float(np.mean([r["runtime_residual_loss"] for r in scored])),
            "scored_baseline_mean": float(np.mean([r["baseline_only_loss"] for r in scored])),
            "scored_by_segment": by_segment,
        },
        "records": records,
    }
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    print("\nfirst/last records:")
    for record in records[:3] + records[-3:]:
        print(json.dumps(record, sort_keys=True))
    if args.output_json is not None:
        output_path = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
