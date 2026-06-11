from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else start
    for candidate in (start, *start.parents):
        if (candidate / "scripts/graphcast_env.sh").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find Weather_global repo root from current working directory.")


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.graphcast.training.core.model import load_graphcast_checkpoint


INSERTION_RE = re.compile(r"mesh_interleaved_temporal_r(?P<rep>\d+)_s(?P<step>\d+)")
BASE_DIR = ROOT / "artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16"
CHECKPOINT_FILE = "ckpt_best.npz"
RUNS = {
    "di64_ds32": BASE_DIR
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds32_20k_target_step4_bptt16",
    "di128_ds64": BASE_DIR
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di128_ds64_20k_target_step4_bptt16",
}
OUT_DIR = ROOT / "plots/analyze_models/images/gc_mamba_tensors"


def tensor_kind(path: str) -> str:
    if "layer_norm" in path:
        return "layernorm"
    if "mamba_block" in path:
        return "mamba"
    return "temporal"


def insertion_name(path: str) -> str:
    match = INSERTION_RE.search(path)
    if not match:
        return "unknown"
    return f"r{match.group('rep')}_s{match.group('step')}"


def short_tensor_name(path: str) -> str:
    marker = "~_run_sequence/"
    if marker in path:
        return path.split(marker, 1)[1]
    return path


def iter_param_leaves(params):
    for module_name, module_params in params.items():
        for param_name, value in module_params.items():
            yield module_name, param_name, value


def extract_mamba_tensors(ckpt, include_layernorm: bool = True) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for module_name, param_name, value in iter_param_leaves(ckpt.params):
        full_path = f"{module_name}/{param_name}"
        lower = full_path.lower()
        is_temporal = "temporal" in lower or "mamba" in lower
        is_ln = "layer_norm" in lower
        if not is_temporal:
            continue
        if is_ln and not include_layernorm:
            continue
        tensors[full_path] = np.asarray(value)
    return dict(sorted(tensors.items()))


def matrix_view(tensor: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim < 2:
        return None
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def svd_values(tensor: np.ndarray, normalize: bool = True) -> np.ndarray:
    mat = matrix_view(tensor)
    if mat is None:
        return np.array([], dtype=np.float64)
    values = np.linalg.svd(mat, compute_uv=False)
    if normalize and values.size and values[0] != 0:
        values = values / values[0]
    return values


def selected_tensors(
    all_tensors: dict[str, dict[str, np.ndarray]],
    pattern: str | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    selected = {}
    for run_label, tensors in all_tensors.items():
        for path, arr in tensors.items():
            if pattern is not None and pattern not in path:
                continue
            if tensor_kind(path) == "layernorm":
                continue
            if matrix_view(arr) is None:
                continue
            selected[(run_label, path)] = arr
    return selected


def safe_name(pattern: str | None) -> str:
    if pattern is None:
        return "all_matrix_like_mamba_tensors"
    return pattern.replace("~", "").replace("/", "_").replace(" ", "_").strip("_")


def make_svd_distribution(
    all_tensors: dict[str, dict[str, np.ndarray]],
    pattern: str | None,
    *,
    normalize: bool,
    bins: int = 60,
) -> pd.DataFrame:
    rows = []
    for (run_label, path), arr in selected_tensors(all_tensors, pattern=pattern).items():
        for value in svd_values(arr, normalize=normalize):
            rows.append(
                {
                    "selection": pattern or "all_matrix_like_mamba_tensors",
                    "normalized": normalize,
                    "run": run_label,
                    "insertion": insertion_name(path),
                    "tensor": short_tensor_name(path),
                    "path": path,
                    "svd_value": float(value),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No singular values matched pattern={pattern!r}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for run_label, group in df.groupby("run"):
        ax.hist(group["svd_value"], bins=bins, alpha=0.45, density=True, label=run_label)
    title = "all matrix-like Mamba tensors" if pattern is None else pattern
    ax.set_title(f"SVD value distribution: {title}")
    ax.set_xlabel("Normalized SVD value" if normalize else "Raw SVD value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    suffix = "" if normalize else "_raw"
    fig.savefig(OUT_DIR / f"svd_distribution_{safe_name(pattern)}{suffix}.png", dpi=200)
    plt.close(fig)
    return df


def make_svd_spectrum(
    all_tensors: dict[str, dict[str, np.ndarray]],
    pattern: str,
    *,
    normalize: bool,
) -> None:
    chosen = selected_tensors(all_tensors, pattern=pattern)
    if not chosen:
        raise ValueError(f"No matrix-like tensors matched pattern={pattern!r}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for (run_label, path), arr in chosen.items():
        values = svd_values(arr, normalize=normalize)
        if values.size == 0:
            continue
        label = f"{run_label} | {insertion_name(path)} | {short_tensor_name(path)}"
        ax.plot(np.arange(1, values.size + 1), values, marker=".", linewidth=1.2, label=label)
    ax.set_title(f"SVD spectrum: {pattern}")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized SVD value" if normalize else "Raw SVD value")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    suffix = "" if normalize else "_raw"
    fig.savefig(OUT_DIR / f"svd_spectrum_{safe_name(pattern)}{suffix}.png", dpi=200)
    plt.close(fig)


def make_direct_a_plots(all_tensors: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    a_arrays: dict[tuple[str, str], np.ndarray] = {}
    for (run_label, path), a_log in selected_tensors(all_tensors, pattern="A_log").items():
        a = -np.exp(np.asarray(a_log, dtype=np.float64))
        a_arrays[(run_label, path)] = a
        for channel_idx, row in enumerate(a):
            for state_idx, value in enumerate(row):
                rows.append(
                    {
                        "run": run_label,
                        "insertion": insertion_name(path),
                        "tensor": short_tensor_name(path),
                        "path": path,
                        "channel": channel_idx,
                        "state": state_idx,
                        "A_value": float(value),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No A_log tensors found for direct A plots.")

    fig, ax = plt.subplots(figsize=(10, 5))
    for run_label, group in df.groupby("run"):
        ax.hist(group["A_value"], bins=80, alpha=0.45, density=True, label=run_label)
    ax.set_title("Direct SSM A values from A = -exp(A_log)")
    ax.set_xlabel("A value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "direct_A_values_histogram.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for run_label, group in df.groupby("run"):
        ax.hist(-group["A_value"], bins=80, alpha=0.45, density=True, label=run_label)
    ax.set_title("Direct positive decay rates from -A = exp(A_log)")
    ax.set_xlabel("-A = exp(A_log)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "direct_A_decay_rates_histogram.png", dpi=200)
    plt.close(fig)

    for (run_label, path), a in a_arrays.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(a, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(f"A = -exp(A_log): {run_label} | {insertion_name(path)}")
        ax.set_xlabel("SSM state index")
        ax.set_ylabel("Mamba channel index")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("A value")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"direct_A_heatmap_{run_label}_{insertion_name(path)}.png", dpi=200)
        plt.close(fig)

    return df


def x_proj_splits(all_tensors: dict[str, dict[str, np.ndarray]]) -> dict[tuple[str, str, str], np.ndarray]:
    splits = {}
    for run_label, tensors in all_tensors.items():
        for path, x_proj_w in tensors.items():
            if not path.endswith("~ssm/x_proj/w"):
                continue
            a_log_path = path.removesuffix("/~ssm/x_proj/w") + "/A_log"
            if a_log_path not in tensors:
                raise KeyError(f"Could not find paired A_log for {path}")
            d_state = int(np.asarray(tensors[a_log_path]).shape[-1])
            x_proj_w = np.asarray(x_proj_w)
            dt_rank = int(x_proj_w.shape[-1] - 2 * d_state)
            if dt_rank <= 0:
                raise ValueError(f"Invalid x_proj split for {path}: shape={x_proj_w.shape}, d_state={d_state}")
            delta_raw_w, b_w, c_w = np.split(x_proj_w, [dt_rank, dt_rank + d_state], axis=-1)
            base = path.removesuffix("/w")
            splits[(run_label, f"{base}/delta_raw_w", "delta_raw")] = delta_raw_w
            splits[(run_label, f"{base}/B_w", "B")] = b_w
            splits[(run_label, f"{base}/C_w", "C")] = c_w
    if not splits:
        raise ValueError("No ~ssm/x_proj/w tensors found to split.")
    return splits


def make_x_proj_split_plots(all_tensors: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    splits = x_proj_splits(all_tensors)
    svd_rows = []
    value_rows = []

    for (run_label, path, component), arr in splits.items():
        for normalize in (True, False):
            for value in svd_values(arr, normalize=normalize):
                svd_rows.append(
                    {
                        "component": component,
                        "normalized": normalize,
                        "run": run_label,
                        "insertion": insertion_name(path),
                        "path": path,
                        "svd_value": float(value),
                    }
                )
        for input_idx, row in enumerate(np.asarray(arr, dtype=np.float64)):
            for output_idx, value in enumerate(row):
                value_rows.append(
                    {
                        "component": component,
                        "run": run_label,
                        "insertion": insertion_name(path),
                        "path": path,
                        "input_channel": input_idx,
                        "output_channel": output_idx,
                        "weight": float(value),
                    }
                )

    svd_df = pd.DataFrame(svd_rows)
    value_df = pd.DataFrame(value_rows)

    for component, component_df in svd_df.groupby("component"):
        for normalize, norm_df in component_df.groupby("normalized"):
            fig, ax = plt.subplots(figsize=(10, 5))
            for run_label, group in norm_df.groupby("run"):
                ax.hist(group["svd_value"], bins=60, alpha=0.45, density=True, label=run_label)
            ax.set_title(f"x_proj {component} SVD distribution")
            ax.set_xlabel("Normalized SVD value" if normalize else "Raw SVD value")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            suffix = "" if normalize else "_raw"
            fig.savefig(OUT_DIR / f"x_proj_{component}_svd_distribution{suffix}.png", dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 6))
            for (run_label, insertion), group in norm_df.groupby(["run", "insertion"]):
                values = group["svd_value"].to_numpy()
                ax.plot(np.arange(1, values.size + 1), values, marker=".", linewidth=1.2, label=f"{run_label} | {insertion}")
            ax.set_title(f"x_proj {component} SVD spectrum")
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Normalized SVD value" if normalize else "Raw SVD value")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"x_proj_{component}_svd_spectrum{suffix}.png", dpi=200)
            plt.close(fig)

    for component, component_df in value_df.groupby("component"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for run_label, group in component_df.groupby("run"):
            ax.hist(group["weight"], bins=80, alpha=0.45, density=True, label=run_label)
        ax.set_title(f"x_proj {component} raw weight distribution")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"x_proj_{component}_weight_histogram.png", dpi=200)
        plt.close(fig)

    value_df.to_csv(OUT_DIR / "x_proj_split_weights_long.csv", index=False)
    return svd_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_paths = {label: run_dir / CHECKPOINT_FILE for label, run_dir in RUNS.items()}

    print("Loading checkpoints...")
    for label, path in ckpt_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"  {label}: {path}")

    checkpoints = {label: load_graphcast_checkpoint(path) for label, path in ckpt_paths.items()}
    all_tensors = {label: extract_mamba_tensors(ckpt, include_layernorm=True) for label, ckpt in checkpoints.items()}
    for label, tensors in all_tensors.items():
        print(f"{label}: extracted {len(tensors)} leaves")

    distribution_patterns = [
        None,
        "in_proj/w",
        "out_proj/w",
        "~ssm/x_proj/w",
        "~ssm/dt_proj/w",
        "A_log",
        "conv1d/kernel",
    ]
    frames = []
    for pattern in distribution_patterns:
        print(f"Saving normalized distribution for {pattern or 'all matrix-like tensors'}")
        frames.append(make_svd_distribution(all_tensors, pattern=pattern, normalize=True, bins=60))
        print(f"Saving raw distribution for {pattern or 'all matrix-like tensors'}")
        frames.append(make_svd_distribution(all_tensors, pattern=pattern, normalize=False, bins=60))

    for pattern in ["in_proj/w", "out_proj/w", "~ssm/x_proj/w", "~ssm/dt_proj/w", "A_log"]:
        print(f"Saving normalized spectrum for {pattern}")
        make_svd_spectrum(all_tensors, pattern=pattern, normalize=True)
        print(f"Saving raw spectrum for {pattern}")
        make_svd_spectrum(all_tensors, pattern=pattern, normalize=False)

    print("Saving direct A = -exp(A_log) plots")
    make_direct_a_plots(all_tensors).to_csv(OUT_DIR / "direct_A_values_long.csv", index=False)

    print("Saving split x_proj diagnostics for delta_raw, B, and C")
    make_x_proj_split_plots(all_tensors).to_csv(OUT_DIR / "x_proj_split_svd_values_long.csv", index=False)

    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "svd_values_long.csv", index=False)
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
