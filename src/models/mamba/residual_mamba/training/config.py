from __future__ import annotations

import argparse
import dataclasses

from src.models.graphcast.training.core.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_STATS_DIR,
    RunConfig,
)


@dataclasses.dataclass(frozen=True)
class ResidualSegmentRunConfig:
    base_cfg: RunConfig
    len_segment: int
    bptt_steps: int
    chunk_load_workers: int
    baseline_ckpt: str
    resume_ckpt: str | None
    training_target: str


def parse_args(argv: list[str] | None = None) -> ResidualSegmentRunConfig:
    parser = argparse.ArgumentParser(
        description="Train GraphCast/Mamba on one-step residuals over shuffled chronological segments."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--width", type=int, choices=[32, 128, 256, 512, 1024], default=128)
    parser.add_argument("--processor-msg-steps", type=int, choices=[1, 2, 3, 4], default=1)
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
    parser.add_argument("--temporal-hidden-size", type=int, default=128)
    parser.add_argument("--temporal-d-inner", type=int, default=None)
    parser.add_argument("--temporal-d-state", type=int, default=16)
    parser.add_argument("--temporal-d-conv", type=int, default=4)
    parser.add_argument("--temporal-dt-rank", default="auto")
    parser.add_argument("--temporal-bias", action="store_true", default=False)
    parser.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias", action="store_false", default=True)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-stateful", action="store_true", default=False)
    parser.add_argument("--data-cache-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--data-cache-max-gib", type=float, default=48.0)
    parser.add_argument("--batch-builder", choices=["legacy", "vectorized", "numpy"], default="numpy")
    parser.add_argument(
        "--training-target",
        choices=["residual"],
        default="residual",
        help="Residual target definition. 'residual' means y_true - y_base.",
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
    if args.target_steps != 1:
        raise ValueError("Residual segment training currently requires --target-steps 1.")
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
    if args.temporal_hidden_size <= 0:
        raise ValueError("--temporal-hidden-size must be > 0")
    if args.temporal_d_inner is not None and args.temporal_d_inner <= 0:
        raise ValueError("--temporal-d-inner must be > 0")
    if args.temporal_d_state <= 0:
        raise ValueError("--temporal-d-state must be > 0")
    if args.temporal_d_conv <= 0:
        raise ValueError("--temporal-d-conv must be > 0")
    if args.temporal_dt_rank != "auto" and int(args.temporal_dt_rank) <= 0:
        raise ValueError("--temporal-dt-rank must be 'auto' or a positive integer")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be > 0")
    if not (0.0 <= args.temporal_dropout < 1.0):
        raise ValueError("--temporal-dropout must be in [0, 1)")
    if args.data_cache_max_gib <= 0:
        raise ValueError("--data-cache-max-gib must be > 0")

    base_cfg = RunConfig(
        data_path=args.data_path,
        resolution=args.resolution,
        mesh_size=args.mesh_size,
        width=args.width,
        processor_msg_steps=args.processor_msg_steps,
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
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        precision=args.precision,
        resume_step=args.resume_step,
        input_duration=args.input_duration,
        temporal_backbone=args.temporal_backbone,
        temporal_location=args.temporal_location,
        temporal_hidden_size=args.temporal_hidden_size,
        temporal_d_inner=args.temporal_d_inner,
        temporal_d_state=args.temporal_d_state,
        temporal_d_conv=args.temporal_d_conv,
        temporal_dt_rank=args.temporal_dt_rank,
        temporal_bias=args.temporal_bias,
        temporal_conv_bias=args.temporal_conv_bias,
        temporal_layers=args.temporal_layers,
        temporal_dropout=args.temporal_dropout,
        temporal_stateful=args.temporal_stateful,
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
    )
    return ResidualSegmentRunConfig(
        base_cfg=base_cfg,
        len_segment=args.len_segment,
        bptt_steps=args.bptt_steps,
        chunk_load_workers=args.chunk_load_workers,
        baseline_ckpt=args.baseline_ckpt,
        resume_ckpt=args.resume_ckpt,
        training_target=args.training_target,
    )
