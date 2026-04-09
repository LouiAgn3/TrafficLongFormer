"""
Flow-level encoder with Longformer-style sparse attention.

Processes the sequence of packet representations with:
- Sliding window attention for local context (e.g., window_size=16)
- Global attention on [CLS], first packet, and every Nth packet
- Temporal attention bias from TemporalAttentionBias module

This is the core architectural contribution: enabling efficient attention
over long packet sequences (200+ packets) with timing awareness.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.temporal_encoding import TemporalAttentionBias, Time2Vec, PacketMetadataEncoder


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention with global tokens and temporal bias.

    Implements O(n * w) attention where w is the window size, plus
    O(n * g) for g global tokens. Much cheaper than O(n^2) full attention
    for long sequences.

    For sequences shorter than 2 * window_size, falls back to full attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
        temporal_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden:        (batch, seq_len, hidden_size)
            attention_mask:(batch, seq_len) — 1 for real tokens, 0 for padding
            global_mask:   (batch, seq_len) — 1 for global attention tokens
            temporal_bias: (batch, num_heads, seq_len, seq_len) — temporal attention bias

        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = hidden.shape

        # Project Q, K, V
        q = self.q_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (batch, heads, seq_len, head_dim)

        # For short sequences, use full attention (simpler and not a bottleneck)
        if seq_len <= 2 * self.window_size:
            return self._full_attention(q, k, v, attention_mask, temporal_bias)

        # Compute full attention scores (we'll mask to sliding window)
        # This is a simplification; a production implementation would use
        # the chunked/strided approach from the Longformer paper for O(n*w).
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # scores: (batch, heads, seq_len, seq_len)

        # Add temporal bias
        if temporal_bias is not None:
            scores = scores + temporal_bias

        # Build sliding window mask
        window_mask = self._build_window_mask(seq_len, hidden.device)
        # window_mask: (seq_len, seq_len) — 1 for attending, 0 for masked

        # Add global attention: global tokens attend to/from all tokens
        if global_mask is not None:
            # Global tokens can attend everywhere
            global_rows = global_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            global_cols = global_mask.unsqueeze(-2).float()  # (batch, 1, seq_len)
            combined_mask = window_mask.unsqueeze(0) + global_rows + global_cols
            combined_mask = (combined_mask > 0).float()
        else:
            combined_mask = window_mask.unsqueeze(0).float()

        # Apply padding mask
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()  # (batch, 1, 1, seq_len)
            combined_mask = combined_mask.unsqueeze(1) * pad_mask  # broadcast over heads

        # Apply mask: set masked positions to -inf
        scores = scores.masked_fill(combined_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Handle NaN from all-masked rows (padding)
        attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0.0)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        output = self.out_proj(output)
        return output

    def _full_attention(self, q, k, v, attention_mask, temporal_bias):
        """Standard full attention for short sequences."""
        batch, heads, seq_len, head_dim = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if temporal_bias is not None:
            scores = scores + temporal_bias

        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0.0)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.out_proj(output)

    def _build_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build sliding window attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1.0
        return mask


class FlowEncoderLayer(nn.Module):
    """Single layer of the flow encoder (attention + FFN with pre-norm)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feedforward_size: int,
        window_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SlidingWindowAttention(hidden_size, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
        temporal_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        normed = self.norm1(hidden)
        hidden = hidden + self.attn(normed, attention_mask, global_mask, temporal_bias)
        # Pre-norm FFN with residual
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden


class FlowEncoder(nn.Module):
    """
    Flow-level encoder that processes the sequence of packet embeddings.

    Architecture:
        1. Project packet embeddings + metadata into hidden space
        2. Add Time2Vec temporal embeddings + learned position embeddings
        3. Prepend [CLS] token
        4. Apply N layers of Longformer-style sparse attention with temporal bias
        5. Output [CLS] representation as flow embedding

    Args:
        hidden_size:  Must match packet encoder output dim (768 for BERT-base)
        num_layers:   Number of transformer layers (4-6)
        num_heads:    Number of attention heads (12)
        feedforward_size: FFN intermediate size (3072)
        window_size:  Sliding window size for local attention (16)
        max_packets:  Maximum number of packets per flow (200)
        global_attention_stride: Place global attention every N packets (8)
        dropout:      Dropout rate (0.1)
        time2vec_dim: Dimension of Time2Vec embedding (64)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        feedforward_size: int = 3072,
        window_size: int = 16,
        max_packets: int = 200,
        global_attention_stride: int = 8,
        dropout: float = 0.1,
        time2vec_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_packets = max_packets
        self.global_attention_stride = global_attention_stride

        # Temporal modules
        self.time2vec = Time2Vec(time2vec_dim)
        self.temporal_bias = TemporalAttentionBias(num_heads)
        self.metadata_encoder = PacketMetadataEncoder(hidden_size)

        # Projection: combine packet embedding + time2vec + metadata
        self.input_projection = nn.Linear(
            hidden_size + time2vec_dim + hidden_size,
            hidden_size,
        )
        self.input_norm = nn.LayerNorm(hidden_size)

        # Learned position embedding for packet positions
        self.pos_embedding = nn.Embedding(max_packets + 1, hidden_size)  # +1 for CLS

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            FlowEncoderLayer(hidden_size, num_heads, feedforward_size, window_size, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        packet_embeds: torch.Tensor,
        timestamps: torch.Tensor,
        metadata: torch.Tensor,
        num_packets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            packet_embeds: (batch, max_packets, hidden_size) — from packet encoder
            timestamps:    (batch, max_packets) — absolute packet times
            metadata:      (batch, max_packets, 6) — timing/direction/size features
            num_packets:   (batch,) — actual packet count per sample

        Returns:
            flow_embed:            (batch, hidden_size) — from [CLS] token
            packet_contextualized: (batch, max_packets, hidden_size) — per-packet embeddings
        """
        batch_size, max_pkts, _ = packet_embeds.shape
        device = packet_embeds.device

        # Compute temporal features
        # Time2Vec from inter-packet times (use log-scaled timestamps relative to first)
        relative_ts = timestamps - timestamps[:, :1]  # relative to first packet
        t2v = self.time2vec(torch.log1p(relative_ts.clamp(min=0)).unsqueeze(-1))
        # t2v: (batch, max_pkts, time2vec_dim)

        # Metadata projection
        meta_embed = self.metadata_encoder(metadata)
        # meta_embed: (batch, max_pkts, hidden_size)

        # Combine all inputs
        combined = torch.cat([packet_embeds, t2v, meta_embed], dim=-1)
        hidden = self.input_projection(combined)  # (batch, max_pkts, hidden_size)

        # Add positional embeddings (1-indexed, 0 reserved for CLS)
        positions = torch.arange(1, max_pkts + 1, device=device).unsqueeze(0)
        positions = positions.clamp(max=self.max_packets)
        hidden = hidden + self.pos_embedding(positions)
        hidden = self.input_norm(hidden)
        hidden = self.dropout(hidden)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls = cls + self.pos_embedding(torch.zeros(1, dtype=torch.long, device=device))
        hidden = torch.cat([cls, hidden], dim=1)  # (batch, 1+max_pkts, hidden)

        # Build attention mask (1 = attend, 0 = padding)
        # CLS token is always attended to
        pkt_mask = torch.arange(max_pkts, device=device).unsqueeze(0) < num_packets.unsqueeze(1)
        attention_mask = torch.cat([
            torch.ones(batch_size, 1, device=device, dtype=torch.bool),
            pkt_mask,
        ], dim=1).float()

        # Build global attention mask
        # Global: CLS + first packet + every N-th packet
        global_mask = torch.zeros(batch_size, 1 + max_pkts, device=device)
        global_mask[:, 0] = 1.0   # CLS
        global_mask[:, 1] = 1.0   # First packet
        for i in range(self.global_attention_stride, max_pkts + 1, self.global_attention_stride):
            if i < max_pkts + 1:
                global_mask[:, i] = 1.0

        # Compute temporal attention bias
        # Pad timestamps for CLS (use timestamp 0 for CLS)
        ts_with_cls = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            timestamps,
        ], dim=1)
        temp_bias = self.temporal_bias(ts_with_cls)

        # Apply transformer layers
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, global_mask, temp_bias)

        hidden = self.final_norm(hidden)

        # Extract outputs
        flow_embed = hidden[:, 0, :]            # CLS token
        packet_contextualized = hidden[:, 1:, :]  # Per-packet embeddings

        return flow_embed, packet_contextualized
