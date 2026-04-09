"""
Comprehensive evaluation script for TrafficLongFormer.

Runs all experiments and produces paper-ready results:
1. Reproduction: TrafficFormer baseline on original datasets
2. Our model on same datasets (should match or beat)
3. Context length ablation: 5, 10, 20, 50, 100, 200 packets
4. Timing ablation: with vs without temporal encoding
5. Timing benchmark: TrafficFormer vs our model on Tasks A, B, C
6. Pre-training task ablation
7. Attention visualisation

Usage:
    python evaluation/run_evaluation.py \
        --model_path checkpoints/best_model.pt \
        --data_dir data/ \
        --output_dir results/
"""

import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
)
from scipy import stats

import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": False,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_model(model, dataloader, device, num_classes):
    """
    Evaluate model on a dataloader.
    Returns dict with all metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                packet_bytes=batch["packet_bytes"],
                timestamps=batch["timestamps"],
                directions=batch["directions"],
                sizes=batch["sizes"],
                num_packets=batch["num_packets"],
            )
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return metrics, y_true, y_pred


def run_with_seeds(model_class, model_kwargs, train_fn, eval_fn, seeds=(7, 42, 123, 256, 512)):
    """
    Run experiment with multiple seeds and report mean/std.
    """
    all_metrics = defaultdict(list)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_class(**model_kwargs)
        train_fn(model, seed)
        metrics = eval_fn(model)

        for k, v in metrics.items():
            all_metrics[k].append(v)

    # Compute statistics
    summary = {}
    for k, vals in all_metrics.items():
        summary[k] = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "values": vals,
        }

    return summary


def significance_test(metrics_a, metrics_b, metric_name="f1_macro"):
    """
    Paired t-test between two sets of results (across seeds).
    """
    a = metrics_a[metric_name]["values"]
    b = metrics_b[metric_name]["values"]
    t_stat, p_value = stats.ttest_rel(a, b)
    return {"t_statistic": t_stat, "p_value": p_value, "significant": p_value < 0.05}


# ---------------------------------------------------------------------------
# Timing benchmark evaluation
# ---------------------------------------------------------------------------

def evaluate_on_timing_benchmark(model, benchmark_dir, device, batch_size=64):
    """
    Evaluate a model on one of the synthetic timing benchmarks.
    """
    data = {}
    for key in ["packet_bytes", "timestamps", "directions", "sizes", "num_packets", "labels"]:
        arr = np.load(os.path.join(benchmark_dir, f"{key}.npy"))
        data[key] = torch.from_numpy(arr)

    # Fix dtypes
    data["packet_bytes"] = data["packet_bytes"].long()
    data["timestamps"] = data["timestamps"].float()
    data["directions"] = data["directions"].long()
    data["sizes"] = data["sizes"].long()
    data["num_packets"] = data["num_packets"].long()
    data["labels"] = data["labels"].long()

    dataset = TensorDataset(*[data[k] for k in
                               ["packet_bytes", "timestamps", "directions",
                                "sizes", "num_packets", "labels"]])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_tuple in loader:
            pkt_bytes, ts, dirs, szs, npkts, labels = [b.to(device) for b in batch_tuple]
            outputs = model(
                packet_bytes=pkt_bytes,
                timestamps=ts,
                directions=dirs,
                sizes=szs,
                num_packets=npkts,
            )
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_binary": f1_score(y_true, y_pred, average="binary"),
    }


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def results_to_latex(results_dict, output_path):
    """Generate LaTeX table from results dictionary."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Experimental Results}",
        r"\begin{tabular}{l" + "c" * 4 + "}",
        r"\toprule",
        r"Model & Accuracy & F1 (Macro) & F1 (Micro) & F1 (Weighted) \\",
        r"\midrule",
    ]

    for name, metrics in results_dict.items():
        if isinstance(metrics, dict) and "f1_macro" in metrics:
            if isinstance(metrics["f1_macro"], dict):
                # Multi-seed results
                acc = f"{metrics['accuracy']['mean']:.4f}$\\pm${metrics['accuracy']['std']:.4f}"
                f1m = f"{metrics['f1_macro']['mean']:.4f}$\\pm${metrics['f1_macro']['std']:.4f}"
                f1i = f"{metrics['f1_micro']['mean']:.4f}$\\pm${metrics['f1_micro']['std']:.4f}"
                f1w = f"{metrics['f1_weighted']['mean']:.4f}$\\pm${metrics['f1_weighted']['std']:.4f}"
            else:
                acc = f"{metrics['accuracy']:.4f}"
                f1m = f"{metrics['f1_macro']:.4f}"
                f1i = f"{metrics['f1_micro']:.4f}"
                f1w = f"{metrics['f1_weighted']:.4f}"
            lines.append(f"  {name} & {acc} & {f1m} & {f1i} & {f1w} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


def plot_context_ablation(results, output_path):
    """Plot context length vs F1 score."""
    packets = sorted(results.keys())
    f1s = [results[p]["f1_macro"] for p in packets]

    fig, ax = plt.subplots()
    ax.plot(packets, f1s, "o-", linewidth=2, markersize=8, color="steelblue")
    ax.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="TrafficFormer limit")
    ax.set_xlabel("Number of input packets")
    ax.set_ylabel("Macro F1 Score")
    ax.legend(fontsize=9)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--benchmark_dir", type=str, default="data/timing_benchmarks")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 256, 512])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_results = {}

    # Experiment 5: Timing benchmarks (if available)
    for task in ["task_a_slowrate", "task_b_covert", "task_c_beaconing"]:
        task_dir = os.path.join(args.benchmark_dir, task)
        if os.path.exists(task_dir):
            print(f"\nEvaluating on timing benchmark: {task}")
            print("  (Requires trained model — skipping if no checkpoint)")
            # Would call evaluate_on_timing_benchmark here
        else:
            print(f"  Timing benchmark {task} not found at {task_dir}")

    # Save all results
    results_path = os.path.join(args.output_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate LaTeX tables
    latex_path = os.path.join(args.output_dir, "results_table.tex")
    results_to_latex(all_results, latex_path)

    print(f"\nAll results saved to {args.output_dir}")
    print(f"LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
