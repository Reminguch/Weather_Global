#!/usr/bin/env python3
"""
Plot available SLURM node configurations (partitions, GPUs, memory).
Run on the cluster: python scripts/slurm_plot_configs.py

Requires: sinfo on PATH (run on a cluster login node).
Optional: matplotlib for the plot (pip install matplotlib); otherwise prints a table.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict

# Try sinfo (only works on cluster)
def run_sinfo(fmt: str) -> str:
    result = subprocess.run(
        ["sinfo", "-o", fmt],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        print("sinfo failed (are you on a SLURM cluster?). stderr:", result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def main() -> None:
    # Get partition summary: Partition, Nodes, State, Memory, CPUs, GRES
    # %P partition, %D nodes, %t state, %m mem, %c cpus, %G gres, %a alloc mem
    raw = run_sinfo("%P|%D|%t|%m|%c|%G|%a")
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        print("No sinfo output.")
        return

    headers = ["partition", "nodes", "state", "mem_mb", "cpus", "gres", "alloc_mem"]
    # Parse header (first line might be PARTITION|NODES|STATE|MEM|CPUS|GRES|ALLOC_MEM)
    data_lines = lines[1:] if "PARTITION" in lines[0].upper() or "|" in lines[0] else lines

    rows = []
    for ln in data_lines:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 6:
            continue
        part, nodes, state, mem, cpus, gres = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        # Normalize memory (sinfo often shows e.g. 120000 for 120G)
        try:
            mem_val = int(mem) if mem.isdigit() else mem
        except Exception:
            mem_val = mem
        rows.append({
            "partition": part,
            "nodes": nodes,
            "state": state,
            "mem": mem,
            "cpus": cpus,
            "gres": gres or "(none)",
        })

    # Print table
    col_widths = [max(len(str(r.get(h, ""))) for r in rows) for h in headers[:6]]
    col_widths = [max(c, len(h)) for c, h in zip(col_widths, headers[:6])]
    sep = "  "
    header_line = sep.join(h[:col_widths[i]].ljust(col_widths[i]) for i, h in enumerate(headers[:6]))
    print("Available configs (partition → nodes, state, mem, cpus, gres):")
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(sep.join(str(r.get(headers[i], ""))[:col_widths[i]].ljust(col_widths[i]) for i in range(6)))

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(Install matplotlib to get a plot: pip install matplotlib)")
        return

    # Build a simple plot: one bar per partition showing memory (if numeric)
    part_mem = []
    part_labels = []
    for r in rows:
        try:
            m = int(r["mem"])
            part_mem.append(m / 1000.0)  # assume MB -> GB for display
            part_labels.append(r["partition"][:12])
        except (ValueError, TypeError):
            part_mem.append(0)
            part_labels.append(r["partition"][:12])

    if not part_mem:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(part_labels) * 0.8), 5))
    x = range(len(part_labels))
    bars = ax.bar(x, part_mem, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(part_labels, rotation=45, ha="right")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("SLURM partitions: node memory")
    plt.tight_layout()
    out = "logs/slurm_configs.png"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
