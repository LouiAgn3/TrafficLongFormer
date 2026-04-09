"""
Pre-training task implementations for TrafficLongFormer.

Task 1: Distant Field Prediction (DFP)
    Predict header fields of a distant packet given preceding context.
    Forces learning of long-range protocol state dependencies.

Task 2: Inter-Packet Time Prediction (IPTP)
    Predict the timing bucket of a randomly masked packet's inter-packet time.
    Forces learning of normal timing patterns per protocol/application.

Task 3: Temporal Order Verification (TOV)
    Detect whether packet timestamps have been perturbed.
    Forces learning that specific timing sequences are expected.
"""

import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Timing bucket definitions for IPTP
# ---------------------------------------------------------------------------
# Bucket boundaries in seconds
IPTP_BOUNDARIES = [0.001, 0.01, 0.1, 1.0, 10.0]  # 6 buckets:
# [<1ms, 1-10ms, 10-100ms, 100ms-1s, 1-10s, >10s]
IPTP_NUM_BUCKETS = len(IPTP_BOUNDARIES) + 1


def ipt_to_bucket(ipt_seconds: float) -> int:
    """Convert an inter-packet time to its bucket index."""
    for i, boundary in enumerate(IPTP_BOUNDARIES):
        if ipt_seconds < boundary:
            return i
    return len(IPTP_BOUNDARIES)


def ipt_to_bucket_tensor(ipt: torch.Tensor) -> torch.Tensor:
    """Vectorised bucket assignment. ipt: (batch,) in seconds."""
    boundaries = torch.tensor(IPTP_BOUNDARIES, device=ipt.device)
    # Count how many boundaries each value exceeds
    bucket = (ipt.unsqueeze(-1) >= boundaries.unsqueeze(0)).sum(dim=-1)
    return bucket.long()


# ---------------------------------------------------------------------------
# DFP: Distant Field Prediction
# ---------------------------------------------------------------------------

class DFPTaskBuilder:
    """
    Constructs Distant Field Prediction training targets.

    For each sample, selects a target packet at position N (randomly chosen
    from target_positions) and extracts its header field as the label.

    Fields predicted:
    - TCP flags (8 discrete values): SYN, SYN-ACK, ACK, FIN, RST, PSH-ACK, etc.
    - Or: first byte of payload (256 classes, as proxy for protocol state)
    """

    TCP_FLAG_CLASSES = {
        0x02: 0,   # SYN
        0x12: 1,   # SYN-ACK
        0x10: 2,   # ACK
        0x18: 3,   # PSH-ACK
        0x11: 4,   # FIN-ACK
        0x04: 5,   # RST
        0x14: 6,   # RST-ACK
    }
    NUM_FLAG_CLASSES = 8  # 7 known + 1 other

    def __init__(
        self,
        target_positions: list = (10, 20, 50),
        field_type: str = "tcp_flags",
    ):
        self.target_positions = list(target_positions)
        self.field_type = field_type

    def build_targets(
        self,
        packet_bytes: torch.Tensor,
        num_packets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build DFP targets for a batch.

        Args:
            packet_bytes: (batch, max_packets, 64) — raw bytes
            num_packets:  (batch,) — actual packet count

        Returns:
            Dict with 'positions' (batch,) and 'labels' (batch,)
        """
        batch_size = packet_bytes.size(0)
        positions = torch.zeros(batch_size, dtype=torch.long, device=packet_bytes.device)
        labels = torch.zeros(batch_size, dtype=torch.long, device=packet_bytes.device)

        for i in range(batch_size):
            n = num_packets[i].item()
            # Pick a target position that's within the flow
            valid_targets = [p for p in self.target_positions if p < n]
            if not valid_targets:
                valid_targets = [max(0, n - 1)]
            pos = random.choice(valid_targets)
            positions[i] = pos

            # Extract field value
            pkt = packet_bytes[i, pos]
            if self.field_type == "tcp_flags":
                # TCP flags at byte offset 13 in IP packet (after 20-byte IP header)
                if len(pkt) > 33:
                    flags = pkt[33].item()  # IP header (20) + TCP flags offset (13)
                    labels[i] = self.TCP_FLAG_CLASSES.get(flags, 7)
                else:
                    labels[i] = 7  # "other"
            else:
                # First payload byte as label
                payload_start = 40  # IP(20) + TCP(20)
                if len(pkt) > payload_start:
                    labels[i] = pkt[payload_start].item() % 32  # bucket to 32 classes
                else:
                    labels[i] = 0

        return {"positions": positions, "labels": labels}


# ---------------------------------------------------------------------------
# IPTP: Inter-Packet Time Prediction
# ---------------------------------------------------------------------------

class IPTPTaskBuilder:
    """
    Constructs Inter-Packet Time Prediction training targets.

    Randomly selects a packet, masks its timing information, and
    creates a bucket classification target from its actual IPT.
    """

    def __init__(self, min_position: int = 2, mask_prob: float = 0.15):
        self.min_position = min_position
        self.mask_prob = mask_prob

    def build_targets(
        self,
        timestamps: torch.Tensor,
        num_packets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select packets to mask and compute their IPT bucket labels.

        Args:
            timestamps: (batch, max_packets) — absolute packet times
            num_packets: (batch,)

        Returns:
            masked_timestamps: (batch, max_packets) — timestamps with masked values zeroed
            targets: Dict with 'positions' (batch,) and 'labels' (batch,)
        """
        batch_size, max_pkts = timestamps.shape
        device = timestamps.device

        masked_ts = timestamps.clone()
        positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            n = num_packets[i].item()
            if n <= self.min_position:
                continue

            # Select random position to mask
            pos = random.randint(self.min_position, n - 1)
            positions[i] = pos

            # Compute inter-packet time at this position
            ipt = timestamps[i, pos] - timestamps[i, pos - 1]
            labels[i] = ipt_to_bucket(ipt.item())

            # Zero out the timestamp (model must predict without it)
            masked_ts[i, pos] = 0.0

        return masked_ts, {"positions": positions, "labels": labels}


# ---------------------------------------------------------------------------
# TOV: Temporal Order Verification
# ---------------------------------------------------------------------------

class TOVTaskBuilder:
    """
    Constructs Temporal Order Verification training targets.

    With swap_prob probability, swaps the timestamps of two random packets
    while keeping byte content unchanged. The model must detect the perturbation.
    """

    def __init__(self, swap_prob: float = 0.5):
        self.swap_prob = swap_prob

    def build_targets(
        self,
        timestamps: torch.Tensor,
        num_packets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Optionally perturb timestamps and create binary labels.

        Args:
            timestamps: (batch, max_packets) — original timestamps
            num_packets: (batch,)

        Returns:
            perturbed_timestamps: (batch, max_packets)
            targets: Dict with 'labels' (batch,) — 0=natural, 1=perturbed
        """
        batch_size, max_pkts = timestamps.shape
        device = timestamps.device

        perturbed_ts = timestamps.clone()
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            n = num_packets[i].item()
            if n < 4:
                continue

            if random.random() < self.swap_prob:
                # Swap timestamps of two random packets
                pos_a = random.randint(1, n - 2)
                pos_b = random.randint(1, n - 2)
                while pos_b == pos_a:
                    pos_b = random.randint(1, n - 2)

                perturbed_ts[i, pos_a], perturbed_ts[i, pos_b] = \
                    timestamps[i, pos_b].item(), timestamps[i, pos_a].item()
                labels[i] = 1  # perturbed

        return perturbed_ts, {"labels": labels}


# ---------------------------------------------------------------------------
# Combined task manager
# ---------------------------------------------------------------------------

class PretrainingTaskManager:
    """Manages all pre-training tasks and their target construction."""

    def __init__(self, config: dict):
        self.dfp = DFPTaskBuilder(
            target_positions=config.get("dfp_positions", [10, 20, 50]),
            field_type=config.get("dfp_field_type", "tcp_flags"),
        )
        self.iptp = IPTPTaskBuilder(
            min_position=config.get("iptp_min_position", 2),
        )
        self.tov = TOVTaskBuilder(
            swap_prob=config.get("tov_swap_prob", 0.5),
        )
        self.lambdas = {
            "dfp": config.get("dfp_lambda", 0.5),
            "iptp": config.get("iptp_lambda", 0.5),
            "tov": config.get("tov_lambda", 0.3),
        }

    def build_all_targets(
        self,
        packet_bytes: torch.Tensor,
        timestamps: torch.Tensor,
        num_packets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Build targets for all tasks simultaneously.

        Returns modified timestamps (with IPTP masking and TOV perturbation)
        and a dict of all targets.
        """
        # Build DFP targets (doesn't modify inputs)
        dfp_targets = self.dfp.build_targets(packet_bytes, num_packets)
        dfp_targets["lambda"] = self.lambdas["dfp"]

        # Build IPTP targets (masks some timestamps)
        masked_ts, iptp_targets = self.iptp.build_targets(timestamps, num_packets)
        iptp_targets["lambda"] = self.lambdas["iptp"]

        # Build TOV targets (may perturb timestamps)
        perturbed_ts, tov_targets = self.tov.build_targets(masked_ts, num_packets)
        tov_targets["lambda"] = self.lambdas["tov"]

        return perturbed_ts, {
            "dfp": dfp_targets,
            "iptp": iptp_targets,
            "tov": tov_targets,
        }
