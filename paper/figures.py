"""
Figure generation for the paper.

Produces publication-quality plots (PDF format) for:
- Flow length distributions
- Context length vs performance
- Timing feature importance
- Attention heatmaps
- Architecture diagram

Usage:
    python paper/figures.py --results_dir results/ --output_dir paper/figures/
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]


def fig_flow_length_cdf(stats_csv, output_path):
    """Fig 1: CDF of packets per flow across datasets."""
    import pandas as pd
    # This would load from the flow_statistics.py output
    # For now, show the plotting template
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Placeholder data — replace with actual stats
    datasets = {
        "CSTNET-TLS": np.random.lognormal(3, 1.5, 5000).astype(int).clip(3, 1000),
        "USTC-TFC": np.random.lognormal(2.5, 1.2, 5000).astype(int).clip(3, 500),
        "ISCX-VPN": np.random.lognormal(3.5, 1.0, 5000).astype(int).clip(3, 800),
    }

    for i, (name, pkts) in enumerate(datasets.items()):
        sorted_vals = np.sort(pkts)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=name, color=COLORS[i])

    ax.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate("TrafficFormer\nlimit (5 pkts)", xy=(5, 0.3), xytext=(15, 0.2),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
                color="red")

    ax.set_xlabel("Packets per flow")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def fig_context_vs_f1(results_json, output_path):
    """Fig 3: Context length ablation — F1 vs number of packets."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Placeholder — replace with actual ablation results
    packets = [5, 10, 20, 50, 100, 200]
    f1_naive = [0.92, 0.91, 0.89, 0.85, 0.80, 0.75]   # TrafficFormer naive extension
    f1_ours = [0.92, 0.93, 0.94, 0.95, 0.95, 0.96]     # Our model

    ax.plot(packets, f1_naive, "s--", label="TrafficFormer (naive)", color=COLORS[1], markersize=6)
    ax.plot(packets, f1_ours, "o-", label="TrafficLongFormer (ours)", color=COLORS[0], markersize=6)
    ax.axvline(x=5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Number of input packets")
    ax.set_ylabel("Macro F1 Score")
    ax.set_xscale("log")
    ax.set_xticks(packets)
    ax.set_xticklabels(packets)
    ax.set_ylim(0.7, 1.0)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def fig_timing_benchmark(results_json, output_path):
    """Fig 5: Timing benchmark results — TrafficFormer vs ours."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    tasks = ["Slow-rate DoS\n(Task A)", "Covert Channel\n(Task B)", "Beaconing\n(Task C)"]
    # Placeholder — TrafficFormer should be ~50% (random) since it has no timing
    tf_acc = [0.52, 0.51, 0.53]
    our_acc = [0.94, 0.91, 0.88]

    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width / 2, tf_acc, width, label="TrafficFormer", color=COLORS[1], alpha=0.85)
    ax.bar(x + width / 2, our_acc, width, label="TrafficLongFormer", color=COLORS[0], alpha=0.85)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.annotate("Random baseline", xy=(2.3, 0.51), fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def fig_architecture_diagram(output_path):
    """Fig 2: Architecture diagram of TrafficLongFormer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw boxes for each component
    components = [
        # (x, y, width, height, label, color)
        (0.5, 0.3, 2.5, 1.0, "Raw Packets\n(up to 200 × 64B)", "#E3F2FD"),
        (0.5, 1.8, 2.5, 1.0, "Packet Encoder\n(TrafficFormer BERT)", "#BBDEFB"),
        (3.5, 0.3, 2.5, 1.0, "Timestamps +\nMetadata", "#FFF3E0"),
        (3.5, 1.8, 2.5, 1.0, "Temporal Encoding\n(Time2Vec + Bias)", "#FFE0B2"),
        (1.5, 3.3, 5.0, 1.0, "Flow Encoder\n(Longformer + Temporal Attention Bias)", "#C8E6C9"),
        (3.0, 4.8, 2.5, 0.8, "Task Heads\n(Classification / DFP / IPTP / TOV)", "#F3E5F5"),
    ]

    for x, y, w, h, label, color in components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#333", linewidth=1)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8, fontweight="bold")

    # Arrows
    arrow_kwargs = dict(arrowstyle="->", color="#333", lw=1.2)
    ax.annotate("", xy=(1.75, 1.8), xytext=(1.75, 1.3), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(4.75, 1.8), xytext=(4.75, 1.3), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(2.75, 3.3), xytext=(1.75, 2.8), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(5.0, 3.3), xytext=(4.75, 2.8), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(4.25, 4.8), xytext=(4.25, 4.3), arrowprops=arrow_kwargs)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def fig_attention_heatmap(attention_weights, output_path, title=""):
    """Visualise attention patterns from the flow encoder."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attention_weights, cmap="Blues", aspect="auto")
    ax.set_xlabel("Key position (packet)")
    ax.set_ylabel("Query position (packet)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output_dir", default="paper/figures/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating paper figures...")

    fig_flow_length_cdf(
        os.path.join(args.results_dir, "flow_stats", "summary_statistics.csv"),
        os.path.join(args.output_dir, "flow_length_cdf.pdf"),
    )

    fig_context_vs_f1(
        os.path.join(args.results_dir, "context_ablation", "ablation_results.json"),
        os.path.join(args.output_dir, "context_vs_f1.pdf"),
    )

    fig_timing_benchmark(
        os.path.join(args.results_dir, "all_results.json"),
        os.path.join(args.output_dir, "timing_benchmark.pdf"),
    )

    fig_architecture_diagram(
        os.path.join(args.output_dir, "architecture.pdf"),
    )

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
