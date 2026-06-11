from __future__ import annotations

import argparse
import dataclasses

from src.models.graphcast.training.core.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_PREPARED_DATA_ROOT,
    DEFAULT_STATS_DIR,
    MEMORY_MODE_CHOICES,
    RunConfig,
)
from src.models.mamba.residual_mamba.feedback import RESIDUAL_AR_FEEDBACK, RESIDUAL_AR_FEEDBACK_CHOICES


@dataclasses.dataclass(frozen=True)
class ResidualSegmentRunConfig:
    base_cfg: RunConfig
    len_segment: int
    bptt_steps: int
    chunk_load_workers: int
    baseline_ckpt: str
    resume_ckpt: str | None
    training_target: str
    residual_output_head_mode: str = "auto"
    eval_num_segments: int | None = 16
    final_eval_num_segments: int | None = None
    eval_subset_policy: str = "stratified_fixed"
    eval_rotating_diagnostics: bool = True
    residual_ar_feedback: str = RESIDUAL_AR_FEEDBACK


def _positive_int_or_all(value: str) -> int | None:
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'")
    return parsed


def parse_args(argv: list[str] | None = None) -> ResidualSegmentRunConfig:
    parser = argparse.ArgumentParser(
        description="Train GraphCast/Mamba on one-step residuals over shuffled chronological segments."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--data-source", choices=["raw", "prepared_array"], default="raw")
    parser.add_argument("--prepared-data-root", default=DEFAULT_PREPARED_DATA_ROOT)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--width", type=int, choices=[32, 128, 256, 512, 1024], default=128)
    parser.add_argument("--processor-msg-steps", type=int, default=1)
    parser.add_argument("--val-year", type=int, default=2021)
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--baseline-ckpt", default=None)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", default="artifacts/checkpoints/graphcast_mamba_interleaved_segments_residual")
    parser.add_argument("--run-name", default="residual_segments_res2_m4_w128_mp1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-num-segments", type=_positive_int_or_all, default=16)
    parser.add_argument("--final-eval-num-segments", type=_positive_int_or_all, default=None)
    parser.add_argument(
        "--eval-subset-policy",
        choices=["first", "stratified_fixed"],
        default="stratified_fixed",
        help="Policy for capped regular validation evals. Default selects a fixed full-year stratified subset.",
    )
    parser.add_argument(
        "--no-eval-rotating-diagnostics",
        dest="eval_rotating_diagnostics",
        action="store_false",
        default=True,
        help="Disable the second rotating stratified diagnostic eval for capped regular validation evals.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--input-duration", default=None)
    parser.add_argument("--target-steps", type=int, default=1)
    parser.add_argument("--len-segment", type=int, default=30)
    parser.add_argument("--bptt-steps", type=int, default=6)
    parser.add_argument("--chunk-load-workers", type=int, default=6)
    parser.add_argument("--temporal-backbone", choices=["none", "mamba"], default="none")
    parser.add_argument(
        "--temporal-location",
        choices=["mesh_post_encoder", "mesh_processor_interleaved"],
        default="mesh_processor_interleaved",
    )
    parser.add_argument("--temporal-d-inner", type=int, default=None)
    parser.add_argument("--temporal-d-state", type=int, default=16)
    parser.add_argument("--temporal-d-conv", type=int, default=4)
    parser.add_argument("--temporal-dt-rank", default="auto")
    parser.add_argument("--temporal-bias", action="store_true", default=False)
    parser.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias", action="store_false", default=True)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-stateful", action="store_true", default=False)
    parser.add_argument("--temporal-insert-count", type=int, default=None)
    parser.add_argument(
        "--memory-mode",
        choices=MEMORY_MODE_CHOICES,
        default="standard",
        help=(
            "Training memory behavior: standard preserves current behavior, "
            "conservative stops gradients through frozen baseline outputs and "
            "checkpoints each residual AR step, and optimal also rematerializes "
            "processor steps plus mesh2grid."
        ),
    )
    parser.add_argument(
        "--residual-output-head",
        choices=["auto", "enabled", "disabled"],
        default="auto",
        help=(
            "Final zero-init residual head policy. auto enables it for fresh runs "
            "and preserves the existing run_config setting when resuming."
        ),
    )
    parser.add_argument("--data-cache-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--data-cache-max-gib", type=float, default=48.0)
    parser.add_argument("--batch-builder", choices=["legacy", "vectorized", "direct", "numpy", "prepared_array"], default="numpy")
    parser.add_argument(
        "--training-target",
        choices=["residual"],
        default="residual",
        help="Residual target definition. 'residual' means y_true - y_base.",
    )
    parser.add_argument(
        "--residual-ar-feedback",
        choices=RESIDUAL_AR_FEEDBACK_CHOICES,
        default=RESIDUAL_AR_FEEDBACK,
        help=(
            "Physical autoregressive feedback during residual AR tail training/eval. "
            "'baseline_plus_residual' feeds the corrected forecast back; 'baseline' "
            "scores baseline+residual output but feeds only the frozen baseline forecast."
        ),
    )
    args = parser.parse_args(argv)

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.len_segment <= 0:
        raise ValueError("--len-segment must be > 0")
    if args.bptt_steps <= 0:
        raise ValueError("--bptt-steps must be > 0")
    if args.chunk_load_workers <= 0:
        raise ValueError("--chunk-load-workers must be > 0")
    if args.len_segment % args.bptt_steps != 0:
        raise ValueError("--bptt-steps must divide --len-segment")
    if args.target_steps <= 0:
        raise ValueError("--target-steps must be > 0")
    if args.target_steps > 1 and args.target_steps >= args.bptt_steps:
        raise ValueError("--target-steps must be < --bptt-steps for chunk-local AR tail training")
    if args.train_start_year is not None and args.train_end_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_end_year is not None and args.train_start_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_start_year is not None and args.train_start_year > args.train_end_year:
        raise ValueError("--train-start-year must be <= --train-end-year")
    if args.resume_step is not None and args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if not args.baseline_ckpt:
        raise ValueError("--baseline-ckpt is required for residual_mamba training.")
    if args.resume_step is not None and not args.resume_ckpt:
        raise ValueError("--resume-ckpt is required when --resume-step is set.")
    if args.temporal_d_inner is not None and args.temporal_d_inner <= 0:
        raise ValueError("--temporal-d-inner must be > 0")
    if args.temporal_backbone == "mamba" and args.temporal_d_inner is None:
        raise ValueError("--temporal-d-inner is required when --temporal-backbone=mamba")
    if args.temporal_d_state <= 0:
        raise ValueError("--temporal-d-state must be > 0")
    if args.temporal_d_conv <= 0:
        raise ValueError("--temporal-d-conv must be > 0")
    if args.temporal_dt_rank != "auto" and int(args.temporal_dt_rank) <= 0:
        raise ValueError("--temporal-dt-rank must be 'auto' or a positive integer")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be > 0")
    if args.temporal_insert_count is not None and args.temporal_insert_count <= 0:
        raise ValueError("--temporal-insert-count must be > 0")
    if args.temporal_insert_count is not None and args.temporal_insert_count > args.processor_msg_steps:
        raise ValueError("--temporal-insert-count must be <= --processor-msg-steps")
    if not (0.0 <= args.temporal_dropout < 1.0):
        raise ValueError("--temporal-dropout must be in [0, 1)")
    if args.data_cache_max_gib <= 0:
        raise ValueError("--data-cache-max-gib must be > 0")

    base_cfg = RunConfig(
        data_path=args.data_path,
        data_source=args.data_source,
        prepared_data_root=args.prepared_data_root,
        resolution=args.resolution,
        mesh_size=args.mesh_size,
        width=args.width,
        processor_msg_steps=args.processor_msg_steps,
        grad_accum_steps=1,
        val_year=args.val_year,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        ckpt_in=args.baseline_ckpt,
        stats_dir=args.stats_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        eval_num_batches=None,
        final_eval_num_batches=None,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        precision=args.precision,
        resume_step=args.resume_step,
        input_duration=args.input_duration,
        temporal_backbone=args.temporal_backbone,
        temporal_location=args.temporal_location,
        temporal_d_inner=args.temporal_d_inner,
        temporal_d_state=args.temporal_d_state,
        temporal_d_conv=args.temporal_d_conv,
        temporal_dt_rank=args.temporal_dt_rank,
        temporal_bias=args.temporal_bias,
        temporal_conv_bias=args.temporal_conv_bias,
        temporal_layers=args.temporal_layers,
        temporal_dropout=args.temporal_dropout,
        temporal_stateful=args.temporal_stateful,
        temporal_insert_count=args.temporal_insert_count,
        target_steps=args.target_steps,
        sequential_segment_steps=None,
        data_cache_mode=args.data_cache_mode,
        data_cache_max_gib=args.data_cache_max_gib,
        batch_builder=args.batch_builder,
        prefetch_workers=0,
        prefetch_depth=0,
        prefetch_device_depth=0,
        usage_every=1,
        eval_only=False,
        residual_output_head=False,
        memory_mode=args.memory_mode,
    )
    return ResidualSegmentRunConfig(
        base_cfg=base_cfg,
        len_segment=args.len_segment,
        bptt_steps=args.bptt_steps,
        chunk_load_workers=args.chunk_load_workers,
        baseline_ckpt=args.baseline_ckpt,
        resume_ckpt=args.resume_ckpt,
        training_target=args.training_target,
        residual_output_head_mode=args.residual_output_head,
        eval_num_segments=args.eval_num_segments,
        final_eval_num_segments=args.final_eval_num_segments,
        eval_subset_policy=args.eval_subset_policy,
        eval_rotating_diagnostics=args.eval_rotating_diagnostics,
        residual_ar_feedback=args.residual_ar_feedback,
    )
