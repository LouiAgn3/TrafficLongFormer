"""
Synthetic timing-sensitive classification benchmarks.

Constructs tasks where ONLY timing differs between classes,
making them unsolvable by byte-only models like TrafficFormer.

Task A: Slow-rate DoS vs Normal Slow Transfer
Task B: Covert Timing Channel Detection
Task C: Beaconing Detection

Usage:
    python benchmarks/timing_benchmark.py \
        --source_pcap_dir /path/to/benign_flows \
        --output_dir data/timing_benchmarks \
        --num_samples 5000
"""

import os
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict

import dpkt


# ---------------------------------------------------------------------------
# Utility: load benign flows from pcaps
# ---------------------------------------------------------------------------

def load_benign_flows(pcap_dir: str, max_flows: int = 10000, min_packets: int = 20):
    """
    Load flows from a directory of pcap files.
    Returns list of dicts with timestamps, sizes, directions, byte_chunks.
    """
    flows = []
    pcap_dir = Path(pcap_dir)

    for pcap_file in sorted(pcap_dir.rglob("*.pcap")):
        if len(flows) >= max_flows:
            break
        flow = _parse_single_pcap(str(pcap_file))
        if flow and len(flow["timestamps"]) >= min_packets:
            flows.append(flow)

    print(f"Loaded {len(flows)} benign flows with >= {min_packets} packets")
    return flows


def _parse_single_pcap(pcap_path: str):
    try:
        with open(pcap_path, "rb") as f:
            try:
                reader = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                reader = dpkt.pcapng.Reader(f)

            timestamps, sizes, directions, byte_chunks = [], [], [], []
            first_src = None

            for ts, buf in reader:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except dpkt.UnpackError:
                    continue
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                if first_src is None:
                    first_src = ip.src
                timestamps.append(ts)
                sizes.append(len(ip))
                directions.append(0 if ip.src == first_src else 1)
                byte_chunks.append(bytes(ip)[:64])

            if timestamps:
                return {
                    "timestamps": np.array(timestamps),
                    "sizes": np.array(sizes),
                    "directions": np.array(directions),
                    "byte_chunks": byte_chunks,
                }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Task A: Slow-rate DoS vs Normal Slow Transfer
# ---------------------------------------------------------------------------

def generate_task_a(flows: List[Dict], num_samples: int = 2000) -> Dict:
    """
    Slow-rate DoS: artificially regular inter-packet times
    Normal: keep original bursty/irregular timing

    Byte content is IDENTICAL between classes.
    """
    samples = {"packet_bytes": [], "timestamps": [], "directions": [],
               "sizes": [], "num_packets": [], "labels": []}

    used_flows = random.sample(flows, min(num_samples, len(flows)))

    for flow in used_flows:
        n = len(flow["timestamps"])

        # Normal class (label 0): keep original timing
        samples["packet_bytes"].append(flow["byte_chunks"][:n])
        samples["timestamps"].append(flow["timestamps"][:n].copy())
        samples["directions"].append(flow["directions"][:n].copy())
        samples["sizes"].append(flow["sizes"][:n].copy())
        samples["num_packets"].append(n)
        samples["labels"].append(0)

        # Attack class (label 1): regular timing (1 pkt/sec + small jitter)
        regular_ts = np.array([flow["timestamps"][0] + i * 1.0 + random.gauss(0, 0.05)
                               for i in range(n)])
        samples["packet_bytes"].append(flow["byte_chunks"][:n])  # SAME bytes
        samples["timestamps"].append(regular_ts)
        samples["directions"].append(flow["directions"][:n].copy())
        samples["sizes"].append(flow["sizes"][:n].copy())
        samples["num_packets"].append(n)
        samples["labels"].append(1)

    return samples


# ---------------------------------------------------------------------------
# Task B: Covert Timing Channel Detection
# ---------------------------------------------------------------------------

def generate_task_b(flows: List[Dict], num_samples: int = 2000) -> Dict:
    """
    Covert timing channel: encode bits in inter-packet delays.
    Short delay (50ms) = 0, Long delay (200ms) = 1.

    Byte content is IDENTICAL between classes.
    """
    samples = {"packet_bytes": [], "timestamps": [], "directions": [],
               "sizes": [], "num_packets": [], "labels": []}

    used_flows = random.sample(flows, min(num_samples, len(flows)))

    for flow in used_flows:
        n = len(flow["timestamps"])

        # Normal class (label 0): keep original timing
        samples["packet_bytes"].append(flow["byte_chunks"][:n])
        samples["timestamps"].append(flow["timestamps"][:n].copy())
        samples["directions"].append(flow["directions"][:n].copy())
        samples["sizes"].append(flow["sizes"][:n].copy())
        samples["num_packets"].append(n)
        samples["labels"].append(0)

        # Covert class (label 1): encode random bitstring in delays
        covert_ts = [flow["timestamps"][0]]
        bitstring = [random.randint(0, 1) for _ in range(n - 1)]
        for bit in bitstring:
            delay = 0.05 if bit == 0 else 0.2  # 50ms or 200ms
            delay += random.gauss(0, 0.005)     # small jitter
            covert_ts.append(covert_ts[-1] + max(0.01, delay))

        samples["packet_bytes"].append(flow["byte_chunks"][:n])  # SAME bytes
        samples["timestamps"].append(np.array(covert_ts))
        samples["directions"].append(flow["directions"][:n].copy())
        samples["sizes"].append(flow["sizes"][:n].copy())
        samples["num_packets"].append(n)
        samples["labels"].append(1)

    return samples


# ---------------------------------------------------------------------------
# Task C: Beaconing Detection
# ---------------------------------------------------------------------------

def generate_task_c(flows: List[Dict], num_samples: int = 2000) -> Dict:
    """
    C2 beaconing: periodic connections every T seconds with tight jitter.
    Normal periodic: legitimate periodic traffic with wider jitter.

    Both have periodic patterns, but differ in jitter distribution.
    """
    samples = {"packet_bytes": [], "timestamps": [], "directions": [],
               "sizes": [], "num_packets": [], "labels": []}

    used_flows = random.sample(flows, min(num_samples, len(flows)))

    for flow in used_flows:
        n = len(flow["timestamps"])

        # Normal periodic (label 0): period with wide jitter (20% CV)
        period = random.uniform(5.0, 30.0)
        jitter_std = period * 0.20
        normal_ts = [flow["timestamps"][0]]
        for i in range(1, n):
            delay = period + random.gauss(0, jitter_std)
            normal_ts.append(normal_ts[-1] + max(0.1, delay))

        samples["packet_bytes"].append(flow["byte_chunks"][:n])
        samples["timestamps"].append(np.array(normal_ts))
        samples["directions"].append(flow["directions"][:n].copy())
        samples["sizes"].append(flow["sizes"][:n].copy())
        samples["num_packets"].append(n)
        samples["labels"].append(0)

        # C2 beaconing (label 1): same period, very tight jitter (2% CV)
        jitter_std_c2 = period * 0.02
        beacon_ts = [flow["timestamps"][0]]
        for i in range(1, n):
            delay = period + random.gauss(0, jitter_std_c2)
            beacon_ts.append(beacon_ts[-1] + max(0.1, delay))

        samples["packet_bytes"].append(flow["byte_chunks"][:n])  # SAME bytes
        samples["timestamps"].append(np.array(beacon_ts))
        samples["directions"].append(flow["directions"][:n].copy())
        samples["sizes"].append(flow["sizes"][:n].copy())
        samples["num_packets"].append(n)
        samples["labels"].append(1)

    return samples


# ---------------------------------------------------------------------------
# Save / load benchmarks
# ---------------------------------------------------------------------------

def save_benchmark(samples: Dict, output_path: str, task_name: str):
    """Save benchmark to disk as numpy arrays."""
    save_dir = os.path.join(output_path, task_name)
    os.makedirs(save_dir, exist_ok=True)

    # Convert to fixed-length arrays for saving
    max_pkts = max(samples["num_packets"])
    n = len(samples["labels"])

    byte_array = np.zeros((n, max_pkts, 64), dtype=np.uint8)
    ts_array = np.zeros((n, max_pkts), dtype=np.float64)
    dir_array = np.zeros((n, max_pkts), dtype=np.int8)
    size_array = np.zeros((n, max_pkts), dtype=np.int32)

    for i in range(n):
        npkts = samples["num_packets"][i]
        for j in range(npkts):
            chunk = samples["packet_bytes"][i][j]
            byte_array[i, j, :len(chunk)] = list(chunk)
        ts_array[i, :npkts] = samples["timestamps"][i][:npkts]
        dir_array[i, :npkts] = samples["directions"][i][:npkts]
        size_array[i, :npkts] = samples["sizes"][i][:npkts]

    np.save(os.path.join(save_dir, "packet_bytes.npy"), byte_array)
    np.save(os.path.join(save_dir, "timestamps.npy"), ts_array)
    np.save(os.path.join(save_dir, "directions.npy"), dir_array)
    np.save(os.path.join(save_dir, "sizes.npy"), size_array)
    np.save(os.path.join(save_dir, "num_packets.npy"), np.array(samples["num_packets"]))
    np.save(os.path.join(save_dir, "labels.npy"), np.array(samples["labels"]))

    print(f"  Saved {task_name}: {n} samples, max_pkts={max_pkts}")


def load_benchmark(benchmark_dir: str) -> Dict[str, np.ndarray]:
    """Load a saved benchmark."""
    return {
        "packet_bytes": np.load(os.path.join(benchmark_dir, "packet_bytes.npy")),
        "timestamps": np.load(os.path.join(benchmark_dir, "timestamps.npy")),
        "directions": np.load(os.path.join(benchmark_dir, "directions.npy")),
        "sizes": np.load(os.path.join(benchmark_dir, "sizes.npy")),
        "num_packets": np.load(os.path.join(benchmark_dir, "num_packets.npy")),
        "labels": np.load(os.path.join(benchmark_dir, "labels.npy")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Timing-sensitive benchmarks")
    parser.add_argument("--source_pcap_dir", required=True,
                        help="Directory of benign pcap flows to use as base")
    parser.add_argument("--output_dir", default="data/timing_benchmarks")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Flows to use per task (doubled for 2 classes)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading benign flows...")
    flows = load_benign_flows(args.source_pcap_dir, max_flows=args.num_samples)

    if len(flows) < 100:
        print(f"Only found {len(flows)} flows. Need at least 100. Check pcap directory.")
        return

    print("\nGenerating Task A: Slow-rate DoS vs Normal...")
    task_a = generate_task_a(flows, num_samples=min(args.num_samples, len(flows)))
    save_benchmark(task_a, args.output_dir, "task_a_slowrate")

    print("Generating Task B: Covert Timing Channel...")
    task_b = generate_task_b(flows, num_samples=min(args.num_samples, len(flows)))
    save_benchmark(task_b, args.output_dir, "task_b_covert")

    print("Generating Task C: Beaconing Detection...")
    task_c = generate_task_c(flows, num_samples=min(args.num_samples, len(flows)))
    save_benchmark(task_c, args.output_dir, "task_c_beaconing")

    print(f"\nAll benchmarks saved to {args.output_dir}")
    print("These are the 'smoking gun' benchmarks: byte-identical, timing-only tasks.")
    print("TrafficFormer CANNOT solve these. Our model with temporal encoding SHOULD.")


if __name__ == "__main__":
    main()
