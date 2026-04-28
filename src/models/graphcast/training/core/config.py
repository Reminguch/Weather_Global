from __future__ import annotations

import argparse
import dataclasses

DEFAULT_DATA_PATH = "data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
DEFAULT_CKPT = (
    "data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
DEFAULT_STATS_DIR = "data/graphcast/graphcast/stats"
DEFAULT_OUT_DIR = "artifacts/checkpoints/graphcast_res2_stream"

GRAPHCAST_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "geopotential_at_surface",
    "land_sea_mask",
    "toa_incident_solar_radiation",
]


@dataclasses.dataclass
class RunConfig:
    data_path: str
    resolution: float
    mesh_size: int
    width: int
    processor_msg_steps: int
    val_year: int
    train_start_year: int | None
    train_end_year: int | None
    ckpt_in: str
    stats_dir: str
    out_dir: str
    run_name: str
    batch_size: int
    max_steps: int
    eval_every: int
    eval_batch_size: int
    checkpoint_every: int
    lr: float
    weight_decay: float
    seed: int
    precision: str
    resume_step: int | None
    input_duration: str | None
    temporal_backbone: str
    temporal_location: str
    temporal_hidden_size: int
    temporal_d_inner: int | None
    temporal_d_state: int
    temporal_d_conv: int
    temporal_dt_rank: str
    temporal_bias: bool
    temporal_conv_bias: bool
    temporal_layers: int
    temporal_dropout: float
    temporal_stateful: bool
    target_steps: int
    sequential_segment_steps: int | None
    data_cache_mode: str
    data_cache_max_gib: float
    batch_builder: str
    prefetch_workers: int
    prefetch_depth: int
    prefetch_device_depth: int
    usage_every: int
    eval_only: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train GraphCast at 2.0deg from local ERA5 data.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Local dataset path (.zarr or .nc).")
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--width", type=int, choices=[128, 256, 512, 1024], default=128)
    parser.add_argument("--processor-msg-steps", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--val-year", type=int, default=2021, help="Validation year (excluded from train split).")
    parser.add_argument("--train-start-year", type=int, default=None, help="Optional lower bound for train years.")
    parser.add_argument("--train-end-year", type=int, default=None, help="Optional upper bound for train years.")
    parser.add_argument("--ckpt-in", default=DEFAULT_CKPT)
    parser.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="res2_m4_w128_mp1_h6_bs4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume-step", type=int, default=None, help="Resume from this step (load params from --ckpt-in).")
    parser.add_argument(
        "--input-duration",
        default=None,
        help="Override task input duration (e.g. 12h/24h/36h/48h). Default: use checkpoint task config.",
    )
    parser.add_argument(
        "--temporal-backbone",
        choices=["none", "mamba"],
        default="none",
        help="Temporal module type. 'none' preserves existing GraphCast behavior.",
    )
    parser.add_argument(
        "--temporal-location",
        choices=["mesh_post_encoder", "mesh_processor_interleaved"],
        default="mesh_post_encoder",
        help="Where to insert temporal module when enabled.",
    )
    parser.add_argument("--temporal-hidden-size", type=int, default=128)
    parser.add_argument("--temporal-d-inner", type=int, default=None,
                        help="Mamba internal channel width. Default: use --temporal-hidden-size.")
    parser.add_argument("--temporal-d-state", type=int, default=16,
                        help="Mamba SSM state size per internal channel.")
    parser.add_argument("--temporal-d-conv", type=int, default=4,
                        help="Mamba causal convolution width. Use 1 to disable local temporal mixing.")
    parser.add_argument("--temporal-dt-rank", default="auto",
                        help="Mamba dt rank: 'auto' or a positive integer.")
    parser.add_argument("--temporal-bias", action="store_true", default=False,
                        help="Enable bias in Mamba linear projections.")
    parser.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias",
                        action="store_false", default=True,
                        help="Disable bias in the Mamba causal convolution.")
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-stateful", action="store_true", default=False,
                        help="Use stateful Mamba (preserves SSM state across autoregressive steps).")
    parser.add_argument("--target-steps", type=int, default=1,
                        help="Number of autoregressive target steps (default 1 = 6h single step).")
    parser.add_argument("--sequential-segment-steps", type=int, default=None,
                        help="Enable chunked sequential sampling: segment length in time steps. "
                             "E.g. 120 = 30 days. Segments are shuffled across epochs, sequential within. "
                             "Mamba state carries across samples within a segment (truncated BPTT).")
    parser.add_argument("--data-cache-mode", choices=["auto", "always", "never"], default="auto",
                        help="Cache the prepared training split in RAM. 'auto' uses --data-cache-max-gib.")
    parser.add_argument("--data-cache-max-gib", type=float, default=48.0,
                        help="Maximum estimated train split size for --data-cache-mode=auto.")
    parser.add_argument("--batch-builder", choices=["legacy", "vectorized", "numpy"], default="vectorized",
                        help="Batch construction implementation. Numpy bypasses xarray gathers when cached.")
    parser.add_argument("--prefetch-workers", type=int, default=4,
                        help="Background workers used to build future random-sampling batches.")
    parser.add_argument("--prefetch-depth", type=int, default=8,
                        help="Maximum number of random-sampling batches queued ahead of training.")
    parser.add_argument("--prefetch-device-depth", type=int, default=1,
                        help="Number of prefetched batches to stage to GPU. Set 0 for host-only prefetch.")
    parser.add_argument("--usage-every", type=int, default=50,
                        help="Sample process/GPU memory every N steps. Set 0 to disable periodic sampling.")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Skip training, only run eval on the loaded checkpoint.")
    args = parser.parse_args()

    if not args.eval_only and args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.train_start_year is not None and args.train_end_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_end_year is not None and args.train_start_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_start_year is not None and args.train_start_year > args.train_end_year:
        raise ValueError("--train-start-year must be <= --train-end-year")
    if args.resume_step is not None and args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if args.temporal_hidden_size <= 0:
        raise ValueError("--temporal-hidden-size must be > 0")
    if args.temporal_d_inner is not None and args.temporal_d_inner <= 0:
        raise ValueError("--temporal-d-inner must be > 0")
    if args.temporal_d_state <= 0:
        raise ValueError("--temporal-d-state must be > 0")
    if args.temporal_d_conv <= 0:
        raise ValueError("--temporal-d-conv must be > 0")
    if args.temporal_dt_rank != "auto":
        try:
            dt_rank = int(args.temporal_dt_rank)
        except ValueError as exc:
            raise ValueError("--temporal-dt-rank must be 'auto' or a positive integer") from exc
        if dt_rank <= 0:
            raise ValueError("--temporal-dt-rank must be 'auto' or a positive integer")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be > 0")
    if not (0.0 <= args.temporal_dropout < 1.0):
        raise ValueError("--temporal-dropout must be in [0, 1)")
    if args.data_cache_max_gib <= 0:
        raise ValueError("--data-cache-max-gib must be > 0")
    if args.prefetch_workers < 0:
        raise ValueError("--prefetch-workers must be >= 0")
    if args.prefetch_depth < 0:
        raise ValueError("--prefetch-depth must be >= 0")
    if args.prefetch_device_depth < 0:
        raise ValueError("--prefetch-device-depth must be >= 0")
    if args.usage_every < 0:
        raise ValueError("--usage-every must be >= 0")

    return RunConfig(
        data_path=args.data_path,
        resolution=args.resolution,
        mesh_size=args.mesh_size,
        width=args.width,
        processor_msg_steps=args.processor_msg_steps,
        val_year=args.val_year,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        ckpt_in=args.ckpt_in,
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
        sequential_segment_steps=args.sequential_segment_steps,
        data_cache_mode=args.data_cache_mode,
        data_cache_max_gib=args.data_cache_max_gib,
        batch_builder=args.batch_builder,
        prefetch_workers=args.prefetch_workers,
        prefetch_depth=args.prefetch_depth,
        prefetch_device_depth=args.prefetch_device_depth,
        usage_every=args.usage_every,
        eval_only=args.eval_only,
    )
