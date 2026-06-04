#!/usr/bin/env python3
"""Plot residual-Mamba training-eval losses for the res2 target-step sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16"
)
DEFAULT_OUTPUT_DATA_DIR = (
    ROOT / "plots/analyze_models/data/resolution_eval/7y_mp6_mamba_res2_target_steps_bptt16_training_eval"
)
DEFAULT_OUTPUT_IMAGE_DIR = (
    ROOT / "plots/analyze_models/images/resolution_eval/7y_mp6_mamba_res2_target_steps_bptt16_warm_rollout"
)

RUN_RE = re.compile(
    r"residual_mamba.*_di(?P<di>\d+)_ds(?P<ds>\d+)_20k_target_step(?P<target_step>\d+)_bptt16"
)
PARAM_STYLES = {
    (64, 32): {"marker": "o", "linestyle": "-", "color": "#2f6f9f"},
    (128, 64): {"marker": "s", "linestyle": "--", "color": "#b14b2d"},
}
TARGET_COLORS = {4: "#2f6f9f", 8: "#b14b2d", 12: "#2f7d4f"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _final_detail(details: list[dict]) -> dict:
    for item in reversed(details):
        if item.get("final"):
            return item
    return details[-1] if details else {}


def _rows(checkpoint_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows: list[dict] = []
    history_rows: list[dict] = []
    for eval_path in sorted(checkpoint_root.glob("*residual_mamba*/eval_loss.json")):
        run_dir = eval_path.parent
        match = RUN_RE.search(run_dir.name)
        if match is None:
            continue
        run_cfg = _load_json(run_dir / "run_config.json")
        eval_loss = _load_json(eval_path)
        details = _load_json(run_dir / "eval_details.json") if (run_dir / "eval_details.json").exists() else []
        target_step = int(match.group("target_step"))
        bptt_steps = int(run_cfg.get("segment_training", {}).get("bptt_steps", 16))
        row_base = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "di": int(match.group("di")),
            "ds": int(match.group("ds")),
            "target_step": target_step,
            "bptt_steps": bptt_steps,
            "truth_prefix_steps": bptt_steps - target_step,
            "eval_semantics": "run_residual_eval_chunk_tail_mean",
        }
        final = _final_detail(details)
        final_loss = float(final.get("total", eval_loss[-1][1]))
        final_rows.append(
            {
                **row_base,
                "step": int(eval_loss[-1][0]),
                "value": final_loss,
                "segments": final.get("segments"),
                "chunks": final.get("chunks"),
                "eval_subset_policy": final.get("eval_subset_policy", ""),
                "eval_subset_role": final.get("eval_subset_role", ""),
                "eval_subset_selected_segments": final.get("eval_subset_selected_segments"),
            }
        )
        for step, value in eval_loss:
            history_rows.append({**row_base, "step": int(step), "value": float(value)})

    if not final_rows:
        raise FileNotFoundError(f"No residual-Mamba eval_loss.json files found under {checkpoint_root}")
    final_df = pd.DataFrame(final_rows).sort_values(["target_step", "di", "ds"]).reset_index(drop=True)
    history_df = pd.DataFrame(history_rows).sort_values(["target_step", "di", "ds", "step"]).reset_index(drop=True)
    return final_df, history_df


def _plot_final(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for (di, ds), sub in df.groupby(["di", "ds"], sort=True):
        sub = sub.sort_values("target_step")
        style = PARAM_STYLES.get((int(di), int(ds)), {"marker": "o", "linestyle": "-"})
        ax.plot(
            sub["target_step"],
            sub["value"],
            label=f"di{int(di)}/ds{int(ds)}",
            linewidth=2.2,
            markersize=6,
            **style,
        )
    ax.set_xticks(sorted(df["target_step"].unique()))
    ax.set_xlabel("target_steps K")
    ax.set_ylabel("Training eval loss")
    ax.set_title("Residual Mamba | final training eval loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def _plot_history(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for (target_step, di, ds), sub in df.groupby(["target_step", "di", "ds"], sort=True):
        sub = sub.sort_values("step")
        style = PARAM_STYLES.get((int(di), int(ds)), {"marker": "o", "linestyle": "-"})
        ax.plot(
            sub["step"],
            sub["value"],
            color=TARGET_COLORS.get(int(target_step)),
            label=f"K={int(target_step)}, di{int(di)}/ds{int(ds)}",
            linewidth=2.0,
            markersize=4,
            **{k: v for k, v in style.items() if k != "color"},
        )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Training eval loss")
    ax.set_title("Residual Mamba | training eval history")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    final_df, history_df = _rows(args.checkpoint_root)
    final_csv = args.output_data_dir / "residual_mamba_training_eval_final.csv"
    history_csv = args.output_data_dir / "residual_mamba_training_eval_history.csv"
    final_df.to_csv(final_csv, index=False)
    history_df.to_csv(history_csv, index=False)
    print(f"Saved CSV: {final_csv}")
    print(f"Saved CSV: {history_csv}")

    _plot_final(
        final_df,
        args.output_image_dir
        / "res2_target_steps_bptt16_training_eval_residual_mamba_weighted_allvars_vs_target_step.png",
    )
    _plot_history(
        history_df,
        args.output_image_dir
        / "res2_target_steps_bptt16_training_eval_residual_mamba_weighted_allvars_vs_train_step.png",
    )


if __name__ == "__main__":
    main()
