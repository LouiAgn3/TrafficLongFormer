"""
Flow-level statistics analysis across datasets.

Produces motivating evidence for the paper by characterising what
information TrafficFormer discards (packets 6+, timing, long flows).

Usage:
    python analysis/flow_statistics.py --pcap_dirs /path/to/dataset1 /path/to/dataset2 \
                                        --names CSTNET USTC-TFC \
                                        --output_dir results/flow_stats
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

import dpkt


# ---------------------------------------------------------------------------
# Plotting defaults (publication-quality)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": False,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Pcap parsing utilities
# ---------------------------------------------------------------------------

def iter_flows_from_pcap(pcap_path: str):
    """
    Yield flows from a single pcap.  Each flow is a list of dicts:
        {
            'timestamp': float,
            'size': int,          # IP payload length
            'raw_bytes': bytes,   # first 64 bytes of IP payload
            'direction': int,     # 0 = client->server, 1 = reverse
            'tcp_flags': int | None,
            'ip_proto': int,
        }

    Flows are keyed by 5-tuple (src, dst, sport, dport, proto).
    """
    flows: dict[tuple, list[dict]] = defaultdict(list)

    try:
        with open(pcap_path, "rb") as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)

            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except dpkt.UnpackError:
                    continue

                if not isinstance(eth.data, dpkt.ip.IP):
                    continue

                ip = eth.data
                proto = ip.p
                src = ip.src
                dst = ip.dst

                if isinstance(ip.data, dpkt.tcp.TCP):
                    tp = ip.data
                    sport, dport = tp.sport, tp.dport
                    tcp_flags = tp.flags
                elif isinstance(ip.data, dpkt.udp.UDP):
                    tp = ip.data
                    sport, dport = tp.sport, tp.dport
                    tcp_flags = None
                else:
                    continue

                # Canonical 5-tuple (smaller IP first for bidirectional)
                if (src, sport) <= (dst, dport):
                    key = (src, dst, sport, dport, proto)
                    direction = 0
                else:
                    key = (dst, src, dport, sport, proto)
                    direction = 1

                raw = bytes(ip)[:64]
                flows[key].append({
                    "timestamp": ts,
                    "size": len(ip),
                    "raw_bytes": raw,
                    "direction": direction,
                    "tcp_flags": tcp_flags,
                    "ip_proto": proto,
                })
    except Exception as e:
        print(f"  Warning: could not parse {pcap_path}: {e}")

    for key, packets in flows.items():
        if len(packets) >= 3:  # skip trivially short flows
            yield packets


def iter_flows_from_dir(pcap_dir: str):
    """Walk a directory tree and yield flows from all pcap/pcapng files."""
    pcap_dir = Path(pcap_dir)
    extensions = {".pcap", ".pcapng", ".cap"}
    for p in sorted(pcap_dir.rglob("*")):
        if p.suffix.lower() in extensions and p.is_file():
            yield from iter_flows_from_pcap(str(p))


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_flow_stats(pcap_dir: str, name: str, max_flows: int = 50_000):
    """Compute all statistics for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Analysing dataset: {name}  ({pcap_dir})")
    print(f"{'='*60}")

    packets_per_flow = []
    flow_durations = []
    all_ipts = []           # inter-packet times
    per_position_bytes = defaultdict(list)   # position -> list of 64-byte arrays
    tls_state_positions = {"certificate": [], "finished": [], "app_data": []}
    tcp_flag_positions = {"SYN": [], "FIN": [], "RST": []}

    flow_count = 0
    for flow in iter_flows_from_dir(pcap_dir):
        if flow_count >= max_flows:
            break
        flow_count += 1

        n = len(flow)
        packets_per_flow.append(n)

        ts_sorted = sorted([p["timestamp"] for p in flow])
        if len(ts_sorted) >= 2:
            flow_durations.append(ts_sorted[-1] - ts_sorted[0])
            for i in range(1, len(ts_sorted)):
                ipt = ts_sorted[i] - ts_sorted[i - 1]
                if ipt >= 0:
                    all_ipts.append(ipt)

        # Per-position byte content (up to packet 100)
        for idx, pkt in enumerate(flow[:100]):
            per_position_bytes[idx].append(pkt["raw_bytes"])

        # TCP flag tracking
        for idx, pkt in enumerate(flow):
            if pkt["tcp_flags"] is not None:
                flags = pkt["tcp_flags"]
                if flags & dpkt.tcp.TH_SYN:
                    tcp_flag_positions["SYN"].append(idx)
                if flags & dpkt.tcp.TH_FIN:
                    tcp_flag_positions["FIN"].append(idx)
                if flags & dpkt.tcp.TH_RST:
                    tcp_flag_positions["RST"].append(idx)

        # TLS state heuristic: look for content type bytes
        for idx, pkt in enumerate(flow):
            raw = pkt["raw_bytes"]
            if len(raw) > 20:
                # Check for TLS record types in payload
                # This is a rough heuristic; proper parsing would need scapy TLS layer
                payload_start = 20 if pkt["ip_proto"] == 6 else 8  # TCP/UDP header
                if len(raw) > payload_start + 5:
                    content_type = raw[payload_start]
                    if content_type == 22:  # Handshake
                        if len(raw) > payload_start + 5:
                            hs_type = raw[payload_start + 5]
                            if hs_type == 11:
                                tls_state_positions["certificate"].append(idx)
                            elif hs_type == 20:
                                tls_state_positions["finished"].append(idx)
                    elif content_type == 23:  # Application Data
                        tls_state_positions["app_data"].append(idx)

        if flow_count % 5000 == 0:
            print(f"  Processed {flow_count} flows...")

    print(f"  Total flows analysed: {flow_count}")

    return {
        "name": name,
        "num_flows": flow_count,
        "packets_per_flow": np.array(packets_per_flow),
        "flow_durations": np.array(flow_durations) if flow_durations else np.array([0.0]),
        "inter_packet_times": np.array(all_ipts) if all_ipts else np.array([0.0]),
        "per_position_bytes": dict(per_position_bytes),
        "tcp_flag_positions": tcp_flag_positions,
        "tls_state_positions": tls_state_positions,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_packets_per_flow_cdf(stats_list, output_dir):
    """CDF of packets per flow for each dataset."""
    fig, ax = plt.subplots()
    for s in stats_list:
        sorted_vals = np.sort(s["packets_per_flow"])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=s["name"], linewidth=1.5)
    ax.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="TrafficFormer limit (5)")
    ax.set_xlabel("Packets per flow")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.set_xlim(left=1)
    fig.savefig(os.path.join(output_dir, "packets_per_flow_cdf.pdf"))
    plt.close(fig)


def plot_flow_duration_cdf(stats_list, output_dir):
    """CDF of flow durations."""
    fig, ax = plt.subplots()
    for s in stats_list:
        durations = s["flow_durations"]
        durations = durations[durations > 0]
        if len(durations) == 0:
            continue
        sorted_vals = np.sort(durations)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=s["name"], linewidth=1.5)
    ax.set_xlabel("Flow duration (seconds)")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(output_dir, "flow_duration_cdf.pdf"))
    plt.close(fig)


def plot_ipt_distribution(stats_list, output_dir):
    """Inter-packet time distribution (histogram + CDF)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for s in stats_list:
        ipts = s["inter_packet_times"]
        ipts = ipts[ipts > 0]
        if len(ipts) == 0:
            continue
        log_ipts = np.log10(ipts + 1e-9)
        ax1.hist(log_ipts, bins=100, alpha=0.5, label=s["name"], density=True)
        sorted_ipts = np.sort(ipts)
        cdf = np.arange(1, len(sorted_ipts) + 1) / len(sorted_ipts)
        ax2.plot(sorted_ipts, cdf, label=s["name"], linewidth=1.5)

    ax1.set_xlabel("log10(inter-packet time) [seconds]")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Inter-packet time (seconds)")
    ax2.set_ylabel("CDF")
    ax2.set_xscale("log")
    ax2.legend(fontsize=8)
    fig.savefig(os.path.join(output_dir, "ipt_distribution.pdf"))
    plt.close(fig)


def plot_information_density(stats_list, output_dir):
    """
    Per-position entropy and novelty curves.
    Shows whether packets beyond position 5 carry new information.
    """
    for s in stats_list:
        positions = sorted(s["per_position_bytes"].keys())
        if len(positions) < 6:
            continue

        entropies = []
        novelty_fracs = []

        # Collect byte patterns from first 5 packets
        first5_patterns = set()
        for pos in range(min(5, len(positions))):
            for raw in s["per_position_bytes"].get(pos, []):
                first5_patterns.add(raw)

        for pos in positions:
            byte_arrays = s["per_position_bytes"][pos]
            if not byte_arrays:
                entropies.append(0)
                novelty_fracs.append(0)
                continue

            # Byte entropy: average over all flows at this position
            pos_entropies = []
            novel_count = 0
            for raw in byte_arrays:
                byte_counts = np.bincount(list(raw), minlength=256)
                probs = byte_counts / max(byte_counts.sum(), 1)
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                pos_entropies.append(entropy)
                if raw not in first5_patterns:
                    novel_count += 1

            entropies.append(np.mean(pos_entropies))
            novelty_fracs.append(novel_count / len(byte_arrays))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(positions, entropies, linewidth=1.5)
        ax1.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax1.set_xlabel("Packet position in flow")
        ax1.set_ylabel("Average byte entropy (bits)")
        ax1.set_title(f"{s['name']}: Entropy by position")

        ax2.plot(positions, novelty_fracs, linewidth=1.5, color="orange")
        ax2.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_xlabel("Packet position in flow")
        ax2.set_ylabel("Fraction of novel patterns")
        ax2.set_title(f"{s['name']}: Novelty beyond packets 1-5")

        fig.savefig(os.path.join(output_dir, f"info_density_{s['name']}.pdf"))
        plt.close(fig)


def plot_protocol_state_analysis(stats_list, output_dir):
    """Where TLS/TCP state transitions occur relative to packet 5."""
    for s in stats_list:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # TCP flags
        ax = axes[0]
        for flag_name, positions in s["tcp_flag_positions"].items():
            if positions:
                ax.hist(positions, bins=50, alpha=0.5, label=flag_name, density=True)
        ax.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Packet position")
        ax.set_ylabel("Density")
        ax.set_title(f"{s['name']}: TCP flag positions")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 50)

        # TLS states
        ax = axes[1]
        for state_name, positions in s["tls_state_positions"].items():
            if positions:
                ax.hist(positions, bins=50, alpha=0.5, label=state_name, density=True)
        ax.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Packet position")
        ax.set_ylabel("Density")
        ax.set_title(f"{s['name']}: TLS state positions")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 50)

        fig.savefig(os.path.join(output_dir, f"protocol_states_{s['name']}.pdf"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary_table(stats_list, output_dir):
    """Generate summary statistics table (for the paper)."""
    rows = []
    for s in stats_list:
        ppf = s["packets_per_flow"]
        dur = s["flow_durations"]
        ipt = s["inter_packet_times"]

        row = {
            "Dataset": s["name"],
            "Flows": s["num_flows"],
            "Median pkts/flow": int(np.median(ppf)),
            "Mean pkts/flow": f"{np.mean(ppf):.1f}",
            ">5 pkts (%)": f"{100 * np.mean(ppf > 5):.1f}",
            ">50 pkts (%)": f"{100 * np.mean(ppf > 50):.1f}",
            ">100 pkts (%)": f"{100 * np.mean(ppf > 100):.1f}",
            "Median duration (s)": f"{np.median(dur):.3f}" if len(dur) > 0 else "N/A",
            ">1s (%)": f"{100 * np.mean(dur > 1):.1f}" if len(dur) > 0 else "N/A",
            ">10s (%)": f"{100 * np.mean(dur > 10):.1f}" if len(dur) > 0 else "N/A",
            "IPT range": f"{np.min(ipt):.6f}-{np.max(ipt):.1f}" if len(ipt) > 1 else "N/A",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)

    # Also save as LaTeX
    latex = df.to_latex(index=False, float_format="%.1f")
    with open(os.path.join(output_dir, "summary_statistics.tex"), "w") as f:
        f.write(latex)

    print("\nSummary Statistics:")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flow statistics analysis")
    parser.add_argument("--pcap_dirs", nargs="+", required=True,
                        help="Directories containing pcap files (one per dataset)")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Dataset names (must match pcap_dirs)")
    parser.add_argument("--output_dir", default="results/flow_stats",
                        help="Output directory for plots and tables")
    parser.add_argument("--max_flows", type=int, default=50000,
                        help="Maximum flows to analyse per dataset")
    args = parser.parse_args()

    assert len(args.pcap_dirs) == len(args.names), \
        "Number of pcap_dirs must match number of names"

    os.makedirs(args.output_dir, exist_ok=True)

    # Compute statistics for each dataset
    stats_list = []
    for pcap_dir, name in zip(args.pcap_dirs, args.names):
        stats = compute_flow_stats(pcap_dir, name, max_flows=args.max_flows)
        stats_list.append(stats)

    # Generate all plots
    print("\nGenerating plots...")
    plot_packets_per_flow_cdf(stats_list, args.output_dir)
    plot_flow_duration_cdf(stats_list, args.output_dir)
    plot_ipt_distribution(stats_list, args.output_dir)
    plot_information_density(stats_list, args.output_dir)
    plot_protocol_state_analysis(stats_list, args.output_dir)

    # Summary table
    make_summary_table(stats_list, args.output_dir)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
