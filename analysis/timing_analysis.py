"""
Timing pattern analysis for traffic data.

Quantifies the value of inter-packet timing as a classification signal,
directly measuring what TrafficFormer misses by ignoring timestamps.

Usage:
    python analysis/timing_analysis.py --pcap_dirs /path/to/dataset1 /path/to/dataset2 \
                                        --label_dirs /path/to/labels1 /path/to/labels2 \
                                        --names CSTNET USTC-TFC \
                                        --output_dir results/timing
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Optional

import dpkt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": False,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Flow extraction with timing (reuses logic from flow_statistics.py)
# ---------------------------------------------------------------------------

def extract_flows_with_labels(pcap_dir: str, label_dir: Optional[str] = None,
                               max_flows: int = 20_000):
    """
    Extract flows with timing features.

    If label_dir is provided, each subdirectory name is the label.
    If pcap files are already split per-class in subdirs, use that structure.

    Returns:
        flows: list of dicts with keys:
            'ipts': inter-packet times
            'sizes': packet sizes
            'directions': packet directions
            'bytes_first5': concatenated first 5 packet bytes
            'label': class label (str or int)
    """
    flows = []
    pcap_dir = Path(pcap_dir)

    # If pcap_dir has class subdirectories, use them as labels
    subdirs = [d for d in sorted(pcap_dir.iterdir()) if d.is_dir()]
    if subdirs:
        for class_dir in subdirs:
            label = class_dir.name
            for pcap_file in sorted(class_dir.rglob("*.pcap")):
                if len(flows) >= max_flows:
                    break
                flow = _parse_single_flow_pcap(str(pcap_file), label)
                if flow is not None:
                    flows.append(flow)
    else:
        # Single directory with all pcaps; try to infer labels from filenames
        for pcap_file in sorted(pcap_dir.rglob("*.pcap")):
            if len(flows) >= max_flows:
                break
            label = pcap_file.parent.name  # fallback
            flow = _parse_single_flow_pcap(str(pcap_file), label)
            if flow is not None:
                flows.append(flow)

    print(f"  Extracted {len(flows)} labelled flows")
    return flows


def _parse_single_flow_pcap(pcap_path: str, label: str):
    """Parse a pcap file expected to contain a single flow."""
    try:
        with open(pcap_path, "rb") as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)

            timestamps = []
            sizes = []
            directions = []
            byte_chunks = []
            first_src = None

            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except dpkt.UnpackError:
                    continue
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                timestamps.append(ts)
                sizes.append(len(ip))
                if first_src is None:
                    first_src = ip.src
                directions.append(0 if ip.src == first_src else 1)
                byte_chunks.append(bytes(ip)[:64])

            if len(timestamps) < 3:
                return None

            ipts = np.diff(timestamps)
            return {
                "ipts": ipts,
                "sizes": np.array(sizes),
                "directions": np.array(directions),
                "byte_chunks": byte_chunks,
                "label": label,
                "num_packets": len(timestamps),
                "duration": timestamps[-1] - timestamps[0],
            }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Timing feature extraction
# ---------------------------------------------------------------------------

def extract_timing_features(flow: dict) -> np.ndarray:
    """Extract timing-only features from a flow."""
    ipts = flow["ipts"]
    if len(ipts) == 0:
        return np.zeros(15)

    log_ipts = np.log1p(ipts)

    features = [
        np.mean(log_ipts),
        np.std(log_ipts),
        np.median(log_ipts),
        np.min(log_ipts),
        np.max(log_ipts),
        np.percentile(log_ipts, 25),
        np.percentile(log_ipts, 75),
        # Autocorrelation at lag 1
        np.corrcoef(log_ipts[:-1], log_ipts[1:])[0, 1] if len(log_ipts) > 2 else 0.0,
        # Coefficient of variation
        np.std(ipts) / (np.mean(ipts) + 1e-9),
        # Burstiness
        (np.std(ipts) - np.mean(ipts)) / (np.std(ipts) + np.mean(ipts) + 1e-9),
        flow["duration"],
        np.log1p(flow["duration"]),
        # Direction change rate (timing between direction changes)
        np.sum(np.diff(flow["directions"]) != 0) / (len(flow["directions"]) - 1 + 1e-9),
        # Number of packets
        float(flow["num_packets"]),
        # Entropy of IPT histogram (binned)
        _ipt_entropy(ipts),
    ]
    return np.array(features, dtype=np.float64)


def extract_byte_features(flow: dict) -> np.ndarray:
    """Extract byte-content features from first 5 packets."""
    byte_features = []
    for i in range(min(5, len(flow["byte_chunks"]))):
        raw = flow["byte_chunks"][i]
        byte_counts = np.bincount(list(raw), minlength=256)
        byte_features.extend([
            np.mean(raw),
            np.std(list(raw)),
            len(set(raw)),  # unique bytes
            -np.sum((byte_counts / max(sum(byte_counts), 1)) *
                    np.log2(byte_counts / max(sum(byte_counts), 1) + 1e-12)),
        ])
    # Pad if fewer than 5 packets
    while len(byte_features) < 20:
        byte_features.append(0.0)
    return np.array(byte_features[:20], dtype=np.float64)


def _ipt_entropy(ipts):
    """Compute entropy of IPT distribution (log-binned)."""
    if len(ipts) < 2:
        return 0.0
    log_ipts = np.log10(ipts + 1e-9)
    bins = np.linspace(log_ipts.min() - 0.1, log_ipts.max() + 0.1, 20)
    counts, _ = np.histogram(log_ipts, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def ipt_distribution_analysis(flows, name, output_dir):
    """Analyse inter-packet time distribution."""
    all_ipts = np.concatenate([f["ipts"] for f in flows if len(f["ipts"]) > 0])

    # Conditional on direction change
    same_dir_ipts = []
    diff_dir_ipts = []
    for f in flows:
        dirs = f["directions"]
        ipts = f["ipts"]
        for i in range(len(ipts)):
            if i + 1 < len(dirs):
                if dirs[i] == dirs[i + 1]:
                    same_dir_ipts.append(ipts[i])
                else:
                    diff_dir_ipts.append(ipts[i])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Overall distribution
    log_ipts = np.log10(all_ipts + 1e-9)
    axes[0].hist(log_ipts, bins=100, density=True, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("log10(IPT) [seconds]")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{name}: Overall IPT distribution")

    # Conditional on direction
    if same_dir_ipts and diff_dir_ipts:
        axes[1].hist(np.log10(np.array(same_dir_ipts) + 1e-9), bins=80, density=True,
                     alpha=0.5, label="Same direction", color="steelblue")
        axes[1].hist(np.log10(np.array(diff_dir_ipts) + 1e-9), bins=80, density=True,
                     alpha=0.5, label="Direction change", color="coral")
        axes[1].set_xlabel("log10(IPT) [seconds]")
        axes[1].set_ylabel("Density")
        axes[1].legend(fontsize=9)
        axes[1].set_title("Conditional on direction")

    # Autocorrelation
    autocorrs = []
    for f in flows:
        if len(f["ipts"]) >= 20:
            log_ipt = np.log1p(f["ipts"])
            log_ipt = (log_ipt - log_ipt.mean()) / (log_ipt.std() + 1e-9)
            acf = np.correlate(log_ipt, log_ipt, mode="full")
            acf = acf[len(acf) // 2:]
            acf = acf / (acf[0] + 1e-9)
            autocorrs.append(acf[:min(50, len(acf))])

    if autocorrs:
        # Pad to same length and average
        max_lag = max(len(a) for a in autocorrs)
        padded = np.zeros((len(autocorrs), min(50, max_lag)))
        for i, a in enumerate(autocorrs):
            padded[i, :len(a)] = a[:min(50, max_lag)]
        mean_acf = np.nanmean(padded, axis=0)
        axes[2].plot(mean_acf, linewidth=1.5, color="steelblue")
        axes[2].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        axes[2].set_xlabel("Lag (packets)")
        axes[2].set_ylabel("Autocorrelation")
        axes[2].set_title("Average IPT autocorrelation")

    fig.savefig(os.path.join(output_dir, f"ipt_analysis_{name}.pdf"))
    plt.close(fig)


def classification_comparison(flows, name, output_dir):
    """
    Compare classification accuracy using:
    1. Timing features only
    2. Byte features only
    3. Both combined
    """
    # Encode labels
    labels = [f["label"] for f in flows]
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        print(f"  Skipping classification for {name}: only {len(unique_labels)} class(es)")
        return None

    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels])

    # Extract features
    X_timing = np.array([extract_timing_features(f) for f in flows])
    X_bytes = np.array([extract_byte_features(f) for f in flows])
    X_both = np.hstack([X_timing, X_bytes])

    # Handle NaN/Inf
    for X in [X_timing, X_bytes, X_both]:
        X[~np.isfinite(X)] = 0.0

    # 5-fold cross-validation
    results = {}
    for feat_name, X in [("Timing only", X_timing), ("Bytes only", X_bytes), ("Both", X_both)]:
        accs, f1s = [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y[train_idx])
            y_pred = clf.predict(X_test)
            accs.append(accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, average="macro"))

        results[feat_name] = {
            "accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
            "f1_macro": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
            "acc_mean": np.mean(accs),
            "f1_mean": np.mean(f1s),
        }

    # Print results
    print(f"\n  Classification comparison for {name} ({len(unique_labels)} classes):")
    for feat_name, r in results.items():
        print(f"    {feat_name:15s}: Acc={r['accuracy']}, F1={r['f1_macro']}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    feat_names = list(results.keys())
    acc_vals = [results[n]["acc_mean"] for n in feat_names]
    f1_vals = [results[n]["f1_mean"] for n in feat_names]
    x = np.arange(len(feat_names))
    width = 0.35
    ax.bar(x - width / 2, acc_vals, width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, f1_vals, width, label="Macro F1", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"{name}: Timing vs Byte features")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(output_dir, f"classification_comparison_{name}.pdf"))
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Timing pattern analysis")
    parser.add_argument("--pcap_dirs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", default="results/timing")
    parser.add_argument("--max_flows", type=int, default=20000)
    args = parser.parse_args()

    assert len(args.pcap_dirs) == len(args.names)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for pcap_dir, name in zip(args.pcap_dirs, args.names):
        print(f"\n{'='*60}")
        print(f"Analysing: {name}")
        print(f"{'='*60}")

        flows = extract_flows_with_labels(pcap_dir, max_flows=args.max_flows)
        if not flows:
            print(f"  No flows found in {pcap_dir}")
            continue

        ipt_distribution_analysis(flows, name, args.output_dir)
        results = classification_comparison(flows, name, args.output_dir)
        if results:
            all_results[name] = results

    # Save combined results table
    if all_results:
        rows = []
        for dataset, results in all_results.items():
            for feat_type, metrics in results.items():
                rows.append({
                    "Dataset": dataset,
                    "Features": feat_type,
                    "Accuracy": metrics["accuracy"],
                    "F1 Macro": metrics["f1_macro"],
                })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.output_dir, "classification_results.csv"), index=False)

        latex = df.to_latex(index=False)
        with open(os.path.join(args.output_dir, "classification_results.tex"), "w") as f:
            f.write(latex)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
