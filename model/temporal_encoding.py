"""
Temporal encoding module for packet sequences.

Three components:
1. Time2Vec: learnable periodic+linear embedding of inter-packet times
2. TemporalAttentionBias: pairwise time-difference bias added to attention scores
3. PacketMetadataEncoder: projects per-packet timing/direction/size features
"""

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Time2Vec embedding (Kazemi et al., 2019).

    Maps scalar time values to a learned representation with one linear
    component and (embed_dim - 1) periodic (sine) components.

    Input:  tau of shape (batch, seq_len, 1) — log-scaled inter-packet times
    Output: embedding of shape (batch, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, embed_dim - 1)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(tau)                    # (batch, seq_len, 1)
        periodic_out = torch.sin(self.periodic(tau))     # (batch, seq_len, embed_dim-1)
        return torch.cat([linear_out, periodic_out], dim=-1)


class TemporalAttentionBias(nn.Module):
    """
    Relative temporal attention bias.

    Computes a per-head bias matrix from pairwise absolute time differences
    between packets, added to attention logits before softmax.

    This lets the model learn that packets close in time should attend
    differently to packets far apart, regardless of sequence position.

    Input:  timestamps of shape (batch, seq_len) — absolute packet times
    Output: bias of shape (batch, num_heads, seq_len, seq_len)
    """

    def __init__(self, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        # Pairwise time differences: (batch, seq_len, seq_len)
        dt = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)
        # Log-scale to handle dynamic range (us to minutes)
        dt = torch.log1p(dt.abs()).unsqueeze(-1)          # (batch, seq, seq, 1)
        bias = self.mlp(dt)                                # (batch, seq, seq, heads)
        return bias.permute(0, 3, 1, 2)                    # (batch, heads, seq, seq)


class PacketMetadataEncoder(nn.Module):
    """
    Project per-packet metadata features into the hidden space.

    Input features (6 total):
        0: log(1 + inter_packet_time_to_prev)
        1: log(1 + inter_packet_time_to_next)  — masked to 0 during inference
        2: direction (0 or 1)
        3: log(1 + packet_size)
        4: log(1 + cumulative_bytes)
        5: relative_position (packet_index / total_packets)

    Input:  metadata of shape (batch, seq_len, 6)
    Output: projection of shape (batch, seq_len, output_dim)
    """

    def __init__(self, output_dim: int, input_dim: int = 6):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.projection(metadata)


def build_metadata_tensor(
    timestamps: torch.Tensor,
    directions: torch.Tensor,
    sizes: torch.Tensor,
    num_packets: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the 6-feature metadata tensor from raw flow data.

    Args:
        timestamps: (batch, max_packets) — absolute packet times
        directions: (batch, max_packets) — 0 or 1
        sizes:      (batch, max_packets) — packet sizes in bytes
        num_packets:(batch,) — actual packet count per sample

    Returns:
        metadata: (batch, max_packets, 6)
    """
    batch, max_pkts = timestamps.shape
    device = timestamps.device

    # Inter-packet times (prev and next)
    ipt_prev = torch.zeros(batch, max_pkts, device=device)
    ipt_next = torch.zeros(batch, max_pkts, device=device)
    ipt_prev[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
    ipt_next[:, :-1] = timestamps[:, 1:] - timestamps[:, :-1]

    # Cumulative bytes
    cum_bytes = torch.cumsum(sizes, dim=1)

    # Relative position
    rel_pos = torch.arange(max_pkts, device=device).float().unsqueeze(0)
    rel_pos = rel_pos / (num_packets.unsqueeze(1).float().clamp(min=1) - 1).clamp(min=1)
    rel_pos = rel_pos.clamp(0, 1)

    metadata = torch.stack([
        torch.log1p(ipt_prev.clamp(min=0)),
        torch.log1p(ipt_next.clamp(min=0)),
        directions.float(),
        torch.log1p(sizes.float()),
        torch.log1p(cum_bytes.float()),
        rel_pos,
    ], dim=-1)  # (batch, max_packets, 6)

    return metadata
