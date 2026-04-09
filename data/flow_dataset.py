"""
Dataset class for full flows with timing information.

Loads pcap files, extracts ALL packets (up to max_packets=200) per flow,
and returns both byte content and timing/direction/size metadata.

Key differences from TrafficFormer's data loader:
1. Extracts all packets, not just 5
2. Extracts packet timestamps
3. Computes inter-packet times, directions, sizes
4. Returns structured tensors for the hierarchical model
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import dpkt


class FlowDataset(Dataset):
    """
    PyTorch Dataset for full traffic flows with timing.

    Each sample is a dict:
        'packet_bytes':  (num_packets, packet_len) — raw byte values [0-255]
        'timestamps':    (num_packets,)            — absolute packet times
        'directions':    (num_packets,)            — 0 (client) or 1 (server)
        'sizes':         (num_packets,)            — original IP packet sizes
        'num_packets':   int                       — actual packet count
        'label':         int                       — class label

    Args:
        pcap_dir:      Directory with class subdirectories, each containing per-flow pcaps
        max_packets:   Maximum packets to extract per flow
        packet_len:    Bytes to keep per packet (after header skip)
        start_index:   Byte offset to start extraction (28 = skip Ethernet)
        max_flows_per_class: Cap samples per class for balance
    """

    def __init__(
        self,
        pcap_dir: str,
        max_packets: int = 200,
        packet_len: int = 64,
        start_index: int = 28,
        max_flows_per_class: Optional[int] = None,
    ):
        self.max_packets = max_packets
        self.packet_len = packet_len
        self.start_index_bytes = start_index // 2  # Convert hex-char index to byte offset
        self.samples: List[Tuple[str, int]] = []
        self.label_names: Dict[str, int] = {}

        pcap_dir = Path(pcap_dir)
        class_dirs = sorted([d for d in pcap_dir.iterdir() if d.is_dir()])

        for class_idx, class_dir in enumerate(class_dirs):
            self.label_names[class_dir.name] = class_idx
            pcap_files = sorted([
                str(f) for f in class_dir.rglob("*")
                if f.suffix.lower() in {".pcap", ".pcapng", ".cap"} and f.is_file()
            ])
            if max_flows_per_class and len(pcap_files) > max_flows_per_class:
                random.seed(42)
                pcap_files = random.sample(pcap_files, max_flows_per_class)
            for pcap_file in pcap_files:
                self.samples.append((pcap_file, class_idx))

        print(f"FlowDataset: {len(self.samples)} flows, "
              f"{len(self.label_names)} classes from {pcap_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        pcap_path, label = self.samples[idx]
        return self._parse_flow(pcap_path, label)

    def _parse_flow(self, pcap_path: str, label: int) -> Dict[str, torch.Tensor]:
        """Parse a single-flow pcap file into tensors."""
        timestamps = []
        byte_arrays = []
        directions = []
        sizes = []
        first_src = None

        try:
            with open(pcap_path, "rb") as f:
                try:
                    reader = dpkt.pcap.Reader(f)
                except ValueError:
                    f.seek(0)
                    reader = dpkt.pcapng.Reader(f)

                for ts, buf in reader:
                    if len(timestamps) >= self.max_packets:
                        break
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                    except dpkt.UnpackError:
                        # Try raw IP
                        try:
                            ip = dpkt.ip.IP(buf)
                        except dpkt.UnpackError:
                            continue
                    else:
                        if not isinstance(eth.data, dpkt.ip.IP):
                            continue
                        ip = eth.data

                    # Direction
                    if first_src is None:
                        first_src = ip.src
                    direction = 0 if ip.src == first_src else 1

                    # Extract bytes from configured offset
                    raw = bytes(ip)
                    pkt_bytes = raw[self.start_index_bytes:
                                    self.start_index_bytes + self.packet_len]

                    timestamps.append(ts)
                    byte_arrays.append(pkt_bytes)
                    directions.append(direction)
                    sizes.append(len(ip))

        except Exception:
            pass

        # Handle empty/failed flows
        if not timestamps:
            return self._empty_sample(label)

        num_packets = len(timestamps)

        # Pad bytes to packet_len
        padded_bytes = np.zeros((num_packets, self.packet_len), dtype=np.uint8)
        for i, ba in enumerate(byte_arrays):
            padded_bytes[i, :len(ba)] = list(ba)

        return {
            "packet_bytes": torch.from_numpy(padded_bytes).long(),
            "timestamps": torch.tensor(timestamps, dtype=torch.float64),
            "directions": torch.tensor(directions, dtype=torch.long),
            "sizes": torch.tensor(sizes, dtype=torch.long),
            "num_packets": torch.tensor(num_packets, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _empty_sample(self, label: int) -> Dict[str, torch.Tensor]:
        """Return a minimal valid sample for failed parses."""
        return {
            "packet_bytes": torch.zeros(1, self.packet_len, dtype=torch.long),
            "timestamps": torch.zeros(1, dtype=torch.float64),
            "directions": torch.zeros(1, dtype=torch.long),
            "sizes": torch.zeros(1, dtype=torch.long),
            "num_packets": torch.tensor(1, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self):
        return len(self.label_names)


def collate_flows(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that pads flows to the max length in the batch.

    Pads packet_bytes, timestamps, directions, sizes to (batch, max_packets_in_batch, ...).
    """
    max_pkts = max(sample["num_packets"].item() for sample in batch)

    batch_bytes = []
    batch_ts = []
    batch_dirs = []
    batch_sizes = []
    batch_num = []
    batch_labels = []

    for sample in batch:
        n = sample["num_packets"].item()
        pkt_len = sample["packet_bytes"].size(1)

        # Pad to max_pkts
        padded_bytes = torch.zeros(max_pkts, pkt_len, dtype=torch.long)
        padded_bytes[:n] = sample["packet_bytes"][:n]

        padded_ts = torch.zeros(max_pkts, dtype=torch.float64)
        padded_ts[:n] = sample["timestamps"][:n]

        padded_dirs = torch.zeros(max_pkts, dtype=torch.long)
        padded_dirs[:n] = sample["directions"][:n]

        padded_sizes = torch.zeros(max_pkts, dtype=torch.long)
        padded_sizes[:n] = sample["sizes"][:n]

        batch_bytes.append(padded_bytes)
        batch_ts.append(padded_ts)
        batch_dirs.append(padded_dirs)
        batch_sizes.append(padded_sizes)
        batch_num.append(sample["num_packets"])
        batch_labels.append(sample["label"])

    return {
        "packet_bytes": torch.stack(batch_bytes),
        "timestamps": torch.stack(batch_ts).float(),
        "directions": torch.stack(batch_dirs),
        "sizes": torch.stack(batch_sizes),
        "num_packets": torch.stack(batch_num),
        "labels": torch.stack(batch_labels),
    }


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    batch_size: int = 32,
    max_packets: int = 200,
    num_workers: int = 4,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train/val/test dataloaders."""
    train_ds = FlowDataset(train_dir, max_packets=max_packets, **kwargs)
    val_ds = FlowDataset(val_dir, max_packets=max_packets, **kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_flows, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_flows, num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_dir:
        test_ds = FlowDataset(test_dir, max_packets=max_packets, **kwargs)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_flows, num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
