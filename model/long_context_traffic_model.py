"""
Full TrafficLongFormer model.

Assembles: PacketEncoder + TemporalEncoding + FlowEncoder + TaskHeads

Forward pass:
    1. Encode each packet independently via PacketEncoder -> packet_embeds
    2. Build metadata tensor from timestamps/directions/sizes
    3. Pass through FlowEncoder with temporal attention bias
    4. Route [CLS] output to appropriate task head
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from model.packet_encoder import TrafficFormerPacketEncoder, SimplePacketEncoder
from model.flow_encoder import FlowEncoder
from model.temporal_encoding import build_metadata_tensor


class TrafficLongFormer(nn.Module):
    """
    Full hierarchical model for long-context traffic classification.

    Args:
        packet_encoder_type: "trafficformer" or "simple"
        trafficformer_dir:   Path to TrafficFormer codebase (if using trafficformer)
        pretrained_path:     Path to TrafficFormer pretrained model
        vocab_path:          Path to TrafficFormer BPE vocabulary
        hidden_size:         Hidden dimension (768)
        flow_num_layers:     Number of flow encoder layers (6)
        flow_num_heads:      Number of attention heads (12)
        flow_feedforward:    FFN intermediate size (3072)
        window_size:         Sliding window size (16)
        max_packets:         Maximum packets per flow (200)
        num_classes:         Number of output classes (for classification)
        dropout:             Dropout rate (0.1)
        freeze_packet_encoder: Whether to freeze packet encoder initially
    """

    def __init__(
        self,
        packet_encoder_type: str = "simple",
        trafficformer_dir: str = "",
        pretrained_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        hidden_size: int = 768,
        flow_num_layers: int = 6,
        flow_num_heads: int = 12,
        flow_feedforward: int = 3072,
        window_size: int = 16,
        max_packets: int = 200,
        num_classes: int = 10,
        dropout: float = 0.1,
        freeze_packet_encoder: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_packets = max_packets

        # Packet encoder
        if packet_encoder_type == "trafficformer":
            self.packet_encoder = TrafficFormerPacketEncoder(
                trafficformer_dir=trafficformer_dir,
                pretrained_path=pretrained_path,
                vocab_path=vocab_path,
                hidden_size=hidden_size,
                freeze=freeze_packet_encoder,
            )
        else:
            self.packet_encoder = SimplePacketEncoder(
                hidden_size=hidden_size,
                num_layers=4,
                num_heads=8,
                max_bytes=64,
            )

        # Flow encoder
        self.flow_encoder = FlowEncoder(
            hidden_size=hidden_size,
            num_layers=flow_num_layers,
            num_heads=flow_num_heads,
            feedforward_size=flow_feedforward,
            window_size=window_size,
            max_packets=max_packets,
            dropout=dropout,
        )

        # Task heads
        self.classifier = ClassificationHead(hidden_size, num_classes, dropout)

    def forward(
        self,
        packet_bytes: torch.Tensor,
        timestamps: torch.Tensor,
        directions: torch.Tensor,
        sizes: torch.Tensor,
        num_packets: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            packet_bytes:  (batch, max_packets, 64) — raw byte values or token IDs
            timestamps:    (batch, max_packets) — absolute packet times
            directions:    (batch, max_packets) — 0 or 1
            sizes:         (batch, max_packets) — packet sizes
            num_packets:   (batch,) — actual packet count per sample
            labels:        (batch,) — class labels (optional, for training)

        Returns:
            Dict with 'logits', optionally 'loss'
        """
        # 1. Encode packets
        if isinstance(self.packet_encoder, SimplePacketEncoder):
            packet_embeds = self.packet_encoder(packet_bytes)
        else:
            # For TrafficFormer, packet_bytes should be pre-tokenized
            seg_ids = torch.ones_like(packet_bytes)
            packet_embeds = self.packet_encoder(packet_bytes, seg_ids)

        # 2. Build metadata
        metadata = build_metadata_tensor(timestamps, directions, sizes, num_packets)

        # 3. Flow encoder
        flow_embed, packet_ctx = self.flow_encoder(
            packet_embeds, timestamps, metadata, num_packets,
        )

        # 4. Classification
        logits = self.classifier(flow_embed)

        output = {"logits": logits, "flow_embed": flow_embed}
        if labels is not None:
            output["loss"] = nn.CrossEntropyLoss()(logits, labels)

        return output

    def unfreeze_packet_encoder(self):
        """Unfreeze packet encoder for end-to-end fine-tuning."""
        if hasattr(self.packet_encoder, "unfreeze"):
            self.packet_encoder.unfreeze()


class ClassificationHead(nn.Module):
    """Classification head matching TrafficFormer's architecture."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, flow_embed: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.tanh(self.dense(flow_embed)))
        return self.classifier(x)


class PretrainingModel(nn.Module):
    """
    TrafficLongFormer with all pre-training task heads.

    Wraps the base model and adds heads for:
    - DFP: Distant Field Prediction
    - IPTP: Inter-Packet Time Prediction
    - TOV: Temporal Order Verification
    """

    def __init__(self, base_model: TrafficLongFormer, config: dict):
        super().__init__()
        self.base_model = base_model
        hidden = base_model.hidden_size

        # DFP head: predict header fields of distant packet
        self.dfp_head = nn.Linear(hidden, config.get("dfp_num_classes", 32))

        # IPTP head: predict timing bucket
        self.iptp_head = nn.Linear(hidden, config.get("iptp_num_buckets", 6))

        # TOV head: binary classification (natural vs perturbed timing)
        self.tov_head = nn.Linear(hidden, 2)

    def forward(
        self,
        packet_bytes: torch.Tensor,
        timestamps: torch.Tensor,
        directions: torch.Tensor,
        sizes: torch.Tensor,
        num_packets: torch.Tensor,
        dfp_targets: Optional[Dict] = None,
        iptp_targets: Optional[Dict] = None,
        tov_targets: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with all pre-training tasks."""
        # Base forward pass (without classification)
        if isinstance(self.base_model.packet_encoder, SimplePacketEncoder):
            packet_embeds = self.base_model.packet_encoder(packet_bytes)
        else:
            seg_ids = torch.ones_like(packet_bytes)
            packet_embeds = self.base_model.packet_encoder(packet_bytes, seg_ids)

        metadata = build_metadata_tensor(timestamps, directions, sizes, num_packets)
        flow_embed, packet_ctx = self.base_model.flow_encoder(
            packet_embeds, timestamps, metadata, num_packets,
        )

        output = {"flow_embed": flow_embed}
        total_loss = torch.tensor(0.0, device=flow_embed.device)

        # DFP: predict field of packet at target position
        if dfp_targets is not None:
            target_pos = dfp_targets["positions"]  # (batch,) — which packet to predict
            target_labels = dfp_targets["labels"]   # (batch,)
            # Gather the contextualised embedding at the prediction position
            batch_idx = torch.arange(packet_ctx.size(0), device=packet_ctx.device)
            target_embeds = packet_ctx[batch_idx, target_pos.clamp(max=packet_ctx.size(1) - 1)]
            dfp_logits = self.dfp_head(target_embeds)
            dfp_loss = nn.CrossEntropyLoss()(dfp_logits, target_labels)
            output["dfp_loss"] = dfp_loss
            total_loss = total_loss + dfp_targets.get("lambda", 0.5) * dfp_loss

        # IPTP: predict timing bucket of masked packet
        if iptp_targets is not None:
            masked_pos = iptp_targets["positions"]
            bucket_labels = iptp_targets["labels"]
            batch_idx = torch.arange(packet_ctx.size(0), device=packet_ctx.device)
            masked_embeds = packet_ctx[batch_idx, masked_pos.clamp(max=packet_ctx.size(1) - 1)]
            iptp_logits = self.iptp_head(masked_embeds)
            iptp_loss = nn.CrossEntropyLoss()(iptp_logits, bucket_labels)
            output["iptp_loss"] = iptp_loss
            total_loss = total_loss + iptp_targets.get("lambda", 0.5) * iptp_loss

        # TOV: binary classification on flow [CLS]
        if tov_targets is not None:
            tov_logits = self.tov_head(flow_embed)
            tov_loss = nn.CrossEntropyLoss()(tov_logits, tov_targets["labels"])
            output["tov_loss"] = tov_loss
            total_loss = total_loss + tov_targets.get("lambda", 0.3) * tov_loss

        output["total_loss"] = total_loss
        return output
