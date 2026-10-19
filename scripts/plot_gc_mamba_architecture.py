"""Draw editable, vector GraphCast–Mamba architecture figures for the paper.

Run from the repository root after sourcing scripts/graphcast_env.sh.
The figures describe the principal v24_Ilya / Overleaf/main_v2.tex architecture.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gc_mamba_matplotlib")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parents[1] / "Overleaf/Pics/gc_mamba_architecture"
INK = "#243447"
GRAY = "#667583"
BLUE = "#246A9A"
ORANGE = "#B45C16"
PALE_BLUE = "#EDF5FB"
PALE_ORANGE = "#FFF3E6"
PALE_GRAY = "#F1F3F5"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": INK, "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def canvas(title, subtitle, height=7.2):
    fig, ax = plt.subplots(figsize=(16, height))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)
    ax.set(xlim=(0, 16), ylim=(0, height))
    ax.axis("off")
    ax.text(0.25, height - 0.42, title, fontsize=21, weight="bold", va="center")
    ax.text(0.25, height - 0.9, subtitle, fontsize=12, color=GRAY, va="center")
    return fig, ax


def box(ax, x, y, w, h, label, color=BLUE, fill=PALE_BLUE, size=12):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, edgecolor=color, facecolor=fill, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=size, linespacing=1.5, zorder=4)


def arrow(ax, points, color=INK, dashed=False, width=1.7, zorder=2,
          mutation_scale=16):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    ax.add_patch(FancyArrowPatch(
        path=MplPath(points, codes), arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=width, color=color, linestyle="--" if dashed else "-",
        zorder=zorder,
    ))


def add(ax, x, y):
    ax.add_patch(Circle((x, y), 0.21, facecolor="white", edgecolor=INK, lw=1.6, zorder=4))
    ax.text(x, y, "+", ha="center", va="center", fontsize=19, zorder=5)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = canvas("a   GraphCast–Mamba: one forecast step",
                     "A frozen spatial forecast plus a trainable correction with temporal memory")
    box(ax, 0.4, 2.75, 2.5, 1.2, "$X_t$\nTwo weather frames\n+ known forcings", color=GRAY, fill="white")
    box(ax, 4.25, 4.05, 4.1, 1.0, "Frozen GraphCast  $G_\u03b8$\nEncoder → graph processor → decoder", color=GRAY, fill=PALE_GRAY)
    box(ax, 4.25, 1.95, 4.1, 1.1, "Mamba correction branch  $R_\u03c6$\nOwn encoder → GNN + Mamba → decoder")
    arrow(ax, [(2.9, 3.35), (3.5, 3.35), (3.5, 4.55), (4.25, 4.55)])
    arrow(ax, [(3.5, 3.35), (3.5, 2.5), (4.25, 2.5)])
    ax.plot(3.5, 3.35, "o", color=INK, ms=4)
    arrow(ax, [(8.35, 4.55), (11.1, 4.55), (11.1, 3.55)])
    arrow(ax, [(8.35, 2.5), (11.1, 2.5), (11.1, 3.13)])
    ax.text(9.65, 4.8, "$G_\u03b8(X_t)$", ha="center", color=GRAY)
    ax.text(9.65, 2.73, "$c_t$", ha="center", color=BLUE)
    add(ax, 11.1, 3.34)
    arrow(ax, [(11.31, 3.34), (12.2, 3.34)])
    box(ax, 12.2, 2.78, 3.35, 1.12, "Final forecast\n$\\hat{Y}_t = G_\u03b8(X_t) + c_t$", color=INK, fill="white", size=13)
    box(ax, 4.25, 3.38, 1.6, 0.4, "$m_{t-1}$", color=ORANGE, fill=PALE_ORANGE)
    box(ax, 6.75, 3.38, 1.6, 0.4, "$m_t$", color=ORANGE, fill=PALE_ORANGE)
    arrow(ax, [(5.05, 3.38), (5.05, 3.05)], color=ORANGE, dashed=True)
    arrow(ax, [(7.55, 3.05), (7.55, 3.38)], color=ORANGE, dashed=True)
    ax.text(6.3, 1.56, "$m$: SSM states + causal-convolution caches", color=ORANGE, ha="center", fontsize=11)
    arrow(ax, [(13.85, 2.78), (13.85, 0.9), (1.65, 0.9), (1.65, 2.75)])
    ax.text(8, 1.06, "Advance the two-frame input window using the combined prediction", ha="center", fontsize=12)
    ax.text(8, 0.32, "Shared meteorological input • Separate branch parameters • Baseline weights fixed", ha="center", color=GRAY, fontsize=11)
    save(fig, "01_full_model")


def correction():
    fig, ax = plt.subplots(figsize=(16, 3.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)
    ax.set(xlim=(0, 16), ylim=(2.9, 6.3))
    ax.axis("off")
    xs = [0.45, 2.75, 5.05, 7.35, 9.65, 11.95, 14.0]
    widths = [1.6, 1.65, 1.65, 1.65, 1.65, 1.5, 1.5]
    labels = ["Grid-to-mesh\nencoder", "Graph message-\npassing step 1", "Mamba module 1\n(2 layers)", "Graph message-\npassing step 2", "Mamba module 2\n(2 layers)", "Mesh-to-grid\ndecoder", "Output head\nZero init."]
    for i, (x, w, label) in enumerate(zip(xs, widths, labels)):
        temporal = i in (2, 4)
        box(ax, x, 4.05, w, 1.0, label, color=ORANGE if temporal else BLUE,
            fill=PALE_ORANGE if temporal else PALE_BLUE, size=11)
        if i:
            arrow(ax, [(xs[i-1] + widths[i-1], 4.55), (x, 4.55)])
    ax.text(0.45, 5.34, "$X_t$ → grid features", fontsize=11, color=BLUE)
    ax.text(15.5, 3.65, "Forecast correction $c_t$", ha="right", fontsize=12, color=BLUE)
    arrow(ax, [(14.75, 4.05), (14.75, 3.87)])
    for x, k in ((5.05, 1), (9.65, 2)):
        ax.text(x + 0.1, 5.9, rf"$m_{{t-1}}^{{({k})}}$", color=ORANGE, ha="center", fontsize=13)
        ax.text(x + 1.55, 5.9, rf"$m_t^{{({k})}}$", color=ORANGE, ha="center", fontsize=13)
        arrow(ax, [(x + 0.15, 5.7), (x + 0.15, 5.05)], color=ORANGE, dashed=True)
        arrow(ax, [(x + 1.5, 5.05), (x + 1.5, 5.7)], color=ORANGE, dashed=True)
    # The GraphCast encoder retains grid features for mesh-to-grid decoding.
    arrow(ax, [(1.25, 4.05), (1.25, 3.15), (12.7, 3.15), (12.7, 4.05)], color=BLUE)
    ax.text(7.1, 3.32, "Encoder grid features → decoder", ha="center", color=BLUE, fontsize=11)
    save(fig, "02_correction_branch")


def rollout():
    # Reserve space for the model title above the step labels and for the
    # small pictorial input/forecast cues beneath each state box.
    fig, ax = plt.subplots(figsize=(16, 7.5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)
    ax.set(xlim=(0, 16), ylim=(-0.2, 7.2))
    ax.axis("off")
    ax.text(8, 6.85, "GraphCast–Mamba", ha="center", va="center",
            fontsize=21, weight="bold")
    earth_steps = [
        plt.imread(OUT / name)
        for name in (
            "rollout_earth_globe.png",
            "rollout_earth_globe_6h.png",
            "rollout_earth_globe_12h.png",
            "rollout_earth_globe_18h.png",
        )
    ]
    mesh = plt.imread(OUT / "rollout_mesh_globe.png")
    # Keep the bright geodesic lines from the mesh illustration and let the
    # globe underneath show through everywhere else.
    mesh_overlay = np.ones((*mesh.shape[:2], 4), dtype=np.float32)
    mesh_overlay[..., 3] = np.clip(
        (mesh[..., :3].min(axis=2) - 0.56) / 0.38, 0, 0.72
    )
    centers = [3.0, 8.0, 13.0]
    ax.text(0.4, 5.9, "Temporal memory state  $m_{-1}=0$", color=ORANGE, ha="left", fontsize=11)
    for i, c in enumerate(centers):
        ax.text(c, 6.15, f"Step {i+1}  ·  +{6*(i+1)} h", ha="center", fontsize=14, weight="bold")
        box(ax, c-1.6, 4.43, 3.2, 0.85, "Mamba correction branch $R_\u03c6$", color=ORANGE, fill=PALE_ORANGE, size=13)
        box(ax, c-1.6, 2.88, 3.2, 0.85, "Frozen GraphCast $G_\u03b8$", color=GRAY, fill=PALE_GRAY, size=12)
        box(ax, c-1.6, 1.05, 1.5, 0.65, f"$X_{i}$", color=INK, fill="white", size=13)
        input_earth = earth_steps[i]
        input_extent = (c-1.41, c-0.29, -0.12, 1.00)
        input_globe = ax.imshow(input_earth, extent=input_extent, zorder=5)
        input_mesh = ax.imshow(mesh_overlay, extent=input_extent, zorder=6)
        input_clip = Circle((c-0.85, 0.44), 0.53, transform=ax.transData)
        input_globe.set_clip_path(input_clip)
        input_mesh.set_clip_path(input_clip)
        arrow(ax, [(c-0.85, 0.96), (c-0.85, 1.03)], color=BLUE, width=1.35)
        # Shared weather input forks outside the left edge of each model.
        arrow(ax, [(c-0.85, 1.7), (c-0.85, 2.15), (c-1.93, 2.15), (c-1.93, 3.3), (c-1.6, 3.3)])
        arrow(ax, [(c-1.93, 3.3), (c-1.93, 4.855), (c-1.6, 4.855)])
        # Both forecasts reach an output addition on the right.
        add(ax, c+1.02, 2.2)
        arrow(ax, [(c+1.02, 2.88), (c+1.02, 2.41)], color=GRAY)
        arrow(ax, [(c+1.6, 4.855), (c+1.97, 4.855), (c+1.97, 2.2), (c+1.23, 2.2)], color=BLUE)
        ax.text(c+1.82, 4.04, f"$c_{i}$", ha="center", color=BLUE, fontsize=11)
        box(ax, c+0.3, 1.05, 1.45, 0.65, f"$\\hat{{Y}}_{i}$", color=INK, fill="white", size=13)
        output_earth = earth_steps[i+1]
        output_extent = (c+0.465, c+1.585, -0.12, 1.00)
        output_globe = ax.imshow(output_earth, extent=output_extent, zorder=5)
        output_mesh = ax.imshow(mesh_overlay, extent=output_extent, zorder=6)
        output_clip = Circle((c+1.025, 0.44), 0.53, transform=ax.transData)
        output_globe.set_clip_path(output_clip)
        output_mesh.set_clip_path(output_clip)
        arrow(ax, [(c+1.025, 1.03), (c+1.025, 0.96)], color=BLUE, width=1.35, zorder=7)
        arrow(ax, [(c+1.02, 1.99), (c+1.02, 1.7)])
        # Memory track above the models, avoiding all weather edges.
        arrow(ax, [(c-0.75, 5.68), (c-0.75, 5.28)], color=ORANGE, dashed=True)
        arrow(ax, [(c+0.75, 5.28), (c+0.75, 5.68), (c+(4.25 if i<2 else 2.5), 5.68)], color=ORANGE, dashed=True)
        ax.text(c+2.5 if i<2 else c+2.0, 5.9, f"$m_{i}$", ha="center", color=ORANGE, fontsize=12)
        if i<2:
            arrow(ax, [(c+1.75, 1.375), (c+3.4, 1.375)], color=BLUE,
                  width=2.4, zorder=8, mutation_scale=24)
            ax.text(c+2.56, 1.93, "Shift window\n+ next forcings", ha="center", color=GRAY, fontsize=10)
    arrow(ax, [(0.4, 5.48), (0.4, 5.68), (2.25, 5.68)], color=ORANGE, dashed=True)
    save(fig, "03_temporal_rollout")


if __name__ == "__main__":
    overview()
    correction()
    rollout()
    print(f"Saved PDF, SVG and PNG figures to {OUT}")
