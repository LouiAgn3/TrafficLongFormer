"""
Prepare USTC-TFC2016 dataset for TrafficLongFormer.

Splits large per-class pcap files into per-flow pcap files organized as:
    output_dir/
        train/
            BitTorrent/
                flow_0000.pcap
                flow_0001.pcap
                ...
            Facetime/
                ...
        val/
            ...
        test/
            ...

Usage:
    python scripts/prepare_data.py \
        --input_dir ../USTC-TFC2016 \
        --output_dir data/ustc_flows \
        --train_ratio 0.7 --val_ratio 0.15
"""

import os
import sys
import argparse
import random
import tempfile
from pathlib import Path
from collections import defaultdict

import dpkt


def extract_7z_files(input_dir):
    """Extract all .7z files in-place using py7zr."""
    import py7zr
    extracted = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".7z"):
                archive = os.path.join(root, fname)
                print(f"  Extracting {archive}...")
                try:
                    with py7zr.SevenZipFile(archive, mode='r') as z:
                        z.extractall(path=root)
                    extracted.append(archive)
                except Exception as e:
                    print(f"  WARNING: Failed to extract {fname}: {e}")
    return extracted


def extract_flows_from_pcap(pcap_path, max_packets_per_flow=200):
    """
    Extract individual flows from a pcap file.

    Groups packets by 5-tuple (src_ip, dst_ip, src_port, dst_port, proto).
    Returns a list of flows, each flow is a list of (timestamp, raw_packet_bytes).
    """
    flows = defaultdict(list)

    try:
        with open(pcap_path, "rb") as f:
            try:
                reader = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                try:
                    reader = dpkt.pcapng.Reader(f)
                except Exception:
                    print(f"  WARNING: Cannot read {pcap_path}, skipping")
                    return []

            for ts, buf in reader:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                except (dpkt.UnpackError, dpkt.NeedData):
                    try:
                        ip = dpkt.ip.IP(buf)
                    except Exception:
                        continue

                # Extract 5-tuple
                proto = ip.p
                src_ip = ip.src
                dst_ip = ip.dst

                try:
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        sport = ip.data.sport
                        dport = ip.data.dport
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        sport = ip.data.sport
                        dport = ip.data.dport
                    else:
                        sport = 0
                        dport = 0
                except Exception:
                    sport = 0
                    dport = 0

                # Canonical 5-tuple (sort IPs so both directions map to same flow)
                if src_ip < dst_ip:
                    key = (src_ip, dst_ip, sport, dport, proto)
                else:
                    key = (dst_ip, src_ip, dport, sport, proto)

                if len(flows[key]) < max_packets_per_flow:
                    flows[key].append((ts, buf))

    except Exception as e:
        print(f"  WARNING: Error reading {pcap_path}: {e}")
        return []

    # Filter: only keep flows with at least 5 packets
    valid_flows = [pkts for pkts in flows.values() if len(pkts) >= 5]
    return valid_flows


def write_flow_pcap(packets, output_path):
    """Write a list of (timestamp, raw_bytes) packets to a pcap file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for ts, buf in packets:
            writer.writepkt(buf, ts)


def main():
    parser = argparse.ArgumentParser(description="Prepare USTC-TFC2016 for TrafficLongFormer")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to USTC-TFC2016 directory")
    parser.add_argument("--output_dir", type=str, default="data/ustc_flows",
                        help="Output directory for per-flow pcaps")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_packets", type=int, default=200)
    parser.add_argument("--min_packets", type=int, default=5,
                        help="Minimum packets per flow to keep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Step 0: Extract any .7z archives
    print("Checking for .7z archives to extract...")
    extract_7z_files(str(input_dir))

    # Find pcap files — handle USTC-TFC2016 layout:
    #   USTC-TFC2016/Benign/BitTorrent.pcap
    #   USTC-TFC2016/Benign/FTP.pcap  (extracted from FTP.7z)
    #   USTC-TFC2016/Malware/Cridex.pcap (extracted from Cridex.7z)
    pcap_by_class = defaultdict(list)

    for f in sorted(input_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in {".pcap", ".pcapng", ".cap"}:
            # Class name = filename stem (e.g., BitTorrent.pcap -> BitTorrent)
            class_name = f.stem
            pcap_by_class[class_name].append(f)

    if not pcap_by_class:
        print(f"ERROR: No pcap files found in {input_dir}")
        print(f"Contents: {[p.name for p in input_dir.iterdir()]}")
        sys.exit(1)

    print(f"Found {len(pcap_by_class)} classes:")
    for cls, files in sorted(pcap_by_class.items()):
        print(f"  {cls}: {len(files)} pcap file(s)")

    # Extract flows from each class
    total_flows = 0
    for class_name, pcap_files in sorted(pcap_by_class.items()):
        print(f"\nProcessing class: {class_name}")
        all_flows = []

        for pcap_file in pcap_files:
            print(f"  Extracting flows from {pcap_file.name}...")
            flows = extract_flows_from_pcap(str(pcap_file), args.max_packets)
            all_flows.extend(flows)
            print(f"    -> {len(flows)} flows (>= 5 packets)")

        if not all_flows:
            print(f"  WARNING: No valid flows for {class_name}, skipping")
            continue

        # Shuffle and split
        random.shuffle(all_flows)
        n = len(all_flows)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        splits = {
            "train": all_flows[:n_train],
            "val": all_flows[n_train:n_train + n_val],
            "test": all_flows[n_train + n_val:],
        }

        for split_name, split_flows in splits.items():
            for i, flow_pkts in enumerate(split_flows):
                out_path = output_dir / split_name / class_name / f"flow_{i:05d}.pcap"
                write_flow_pcap(flow_pkts, str(out_path))

            print(f"  {split_name}: {len(split_flows)} flows")

        total_flows += n

    print(f"\nDone! {total_flows} total flows written to {output_dir}")
    print(f"  train: {output_dir / 'train'}")
    print(f"  val:   {output_dir / 'val'}")
    print(f"  test:  {output_dir / 'test'}")


if __name__ == "__main__":
    main()
