"""
Packet-level encoder wrapping TrafficFormer's pre-trained BERT.

Encodes each packet independently through the BERT encoder and extracts
the [CLS] token representation as the packet-level summary embedding.

This module reuses ALL of TrafficFormer's pre-trained weights.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class TrafficFormerPacketEncoder(nn.Module):
    """
    Wraps TrafficFormer's BERT to encode individual packets.

    Input:  token_ids of shape (batch, num_packets, seq_per_packet)
            seg_ids   of shape (batch, num_packets, seq_per_packet)
    Output: packet_embeds of shape (batch, num_packets, hidden_size)
    """

    def __init__(
        self,
        trafficformer_dir: str,
        pretrained_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        hidden_size: int = 768,
        freeze: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.freeze = freeze

        # Build the TrafficFormer encoder components
        self.embedding, self.encoder, self.tokenizer = self._load_trafficformer(
            trafficformer_dir, pretrained_path, vocab_path
        )

        if freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_trafficformer(self, tf_dir, pretrained_path, vocab_path):
        """Load TrafficFormer's embedding + encoder from its codebase."""
        tf_dir = Path(tf_dir)
        sys.path.insert(0, str(tf_dir))

        from argparse import Namespace
        from uer.layers.embeddings import WordPosSegEmbedding
        from uer.encoders.transformer_encoder import TransformerEncoder
        from uer.utils.vocab import Vocab
        from uer.utils.tokenizers import BertTokenizer

        # Default BERT-base config matching TrafficFormer
        args = Namespace(
            emb_size=self.hidden_size,
            hidden_size=self.hidden_size,
            feedforward_size=self.hidden_size * 4,
            heads_num=12,
            layers_num=12,
            max_seq_length=512,
            dropout=0.1,
            hidden_act="gelu",
            mask="fully_visible",
            parameter_sharing=False,
            factorized_embedding_parameterization=False,
            layernorm_positioning="post",
            relative_position_embedding=False,
            feed_forward="dense",
            remove_transformer_bias=False,
            remove_embedding_layernorm=False,
            remove_attention_scale=False,
            layernorm="normal",
            is_moe=False,
        )

        # Load tokenizer/vocab
        tokenizer = None
        vocab_size = 30522  # default BERT vocab size
        if vocab_path and Path(vocab_path).exists():
            args.vocab_path = vocab_path
            tokenizer = BertTokenizer(args)
            vocab_size = len(tokenizer.vocab)

        embedding = WordPosSegEmbedding(args, vocab_size)
        encoder = TransformerEncoder(args)

        # Load pretrained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # Filter to only embedding + encoder keys
            emb_keys = {k.replace("embedding.", ""): v
                        for k, v in state_dict.items() if k.startswith("embedding.")}
            enc_keys = {k.replace("encoder.", ""): v
                        for k, v in state_dict.items() if k.startswith("encoder.")}
            if emb_keys:
                embedding.load_state_dict(emb_keys, strict=False)
            if enc_keys:
                encoder.load_state_dict(enc_keys, strict=False)
            print(f"Loaded TrafficFormer weights from {pretrained_path}")

        return embedding, encoder, tokenizer

    def forward(
        self,
        token_ids: torch.Tensor,
        seg_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode packets independently.

        Args:
            token_ids: (batch, num_packets, seq_per_packet) — BPE token IDs
            seg_ids:   (batch, num_packets, seq_per_packet) — segment IDs

        Returns:
            packet_embeds: (batch, num_packets, hidden_size) — [CLS] representations
        """
        batch_size, num_packets, seq_len = token_ids.shape

        # Flatten packets for batch processing
        flat_tokens = token_ids.view(batch_size * num_packets, seq_len)
        flat_segs = seg_ids.view(batch_size * num_packets, seq_len)

        # Pass through BERT
        emb = self.embedding(flat_tokens, flat_segs)
        hidden = self.encoder(emb, flat_segs)

        # Extract [CLS] token (position 0) as packet representation
        cls_output = hidden[:, 0, :]  # (batch*num_packets, hidden_size)

        # Reshape back
        packet_embeds = cls_output.view(batch_size, num_packets, self.hidden_size)

        return packet_embeds

    def unfreeze(self):
        """Unfreeze all parameters for end-to-end fine-tuning."""
        self.freeze = False
        for param in self.embedding.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True


class SimplePacketEncoder(nn.Module):
    """
    Lightweight packet encoder for use without pre-trained TrafficFormer.

    Converts raw packet bytes (as integers 0-255) to embeddings via a
    small transformer, useful for testing and ablation.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        max_bytes: int = 64,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size  # Process packets in chunks to save memory

        self.byte_embedding = nn.Embedding(257, hidden_size)  # 0-255 + padding
        self.pos_embedding = nn.Embedding(max_bytes + 1, hidden_size)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, packet_bytes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            packet_bytes: (batch, num_packets, max_bytes) — byte values 0-255

        Returns:
            packet_embeds: (batch, num_packets, hidden_size)
        """
        batch_size, num_packets, max_bytes = packet_bytes.shape

        # Process packets in chunks to avoid OOM
        all_cls = []
        flat_bytes = packet_bytes.view(batch_size * num_packets, max_bytes)

        for start in range(0, flat_bytes.size(0), self.chunk_size):
            end = min(start + self.chunk_size, flat_bytes.size(0))
            chunk = flat_bytes[start:end]

            # Byte embedding + positional
            byte_emb = self.byte_embedding(chunk)
            positions = torch.arange(1, max_bytes + 1, device=chunk.device)
            pos_emb = self.pos_embedding(positions).unsqueeze(0)
            emb = byte_emb + pos_emb

            # Prepend CLS token
            cls = self.cls_token.expand(chunk.size(0), -1, -1)
            cls_pos = self.pos_embedding(torch.zeros(1, dtype=torch.long, device=chunk.device))
            cls = cls + cls_pos.unsqueeze(0)
            emb = torch.cat([cls, emb], dim=1)

            # Encode
            hidden = self.encoder(emb)
            hidden = self.layer_norm(hidden)
            all_cls.append(hidden[:, 0, :])

        cls_output = torch.cat(all_cls, dim=0)
        return cls_output.view(batch_size, num_packets, self.hidden_size)
