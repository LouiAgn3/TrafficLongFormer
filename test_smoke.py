"""
Smoke test: verify the model builds and runs a forward pass.

Run:  python test_smoke.py
"""

import torch
import sys
sys.path.insert(0, ".")

from model.temporal_encoding import Time2Vec, TemporalAttentionBias, PacketMetadataEncoder, build_metadata_tensor
from model.packet_encoder import SimplePacketEncoder
from model.flow_encoder import FlowEncoder
from model.long_context_traffic_model import TrafficLongFormer, PretrainingModel
from model.pretraining_tasks import PretrainingTaskManager, ipt_to_bucket_tensor


def test_temporal_encoding():
    print("Testing temporal encoding modules...")
    batch, seq = 4, 50

    # Time2Vec
    t2v = Time2Vec(64)
    tau = torch.randn(batch, seq, 1)
    out = t2v(tau)
    assert out.shape == (batch, seq, 64), f"Time2Vec shape mismatch: {out.shape}"

    # TemporalAttentionBias
    tab = TemporalAttentionBias(num_heads=12)
    timestamps = torch.cumsum(torch.rand(batch, seq), dim=1)
    bias = tab(timestamps)
    assert bias.shape == (batch, 12, seq, seq), f"TemporalBias shape mismatch: {bias.shape}"

    # PacketMetadataEncoder
    pme = PacketMetadataEncoder(output_dim=768)
    metadata = torch.randn(batch, seq, 6)
    out = pme(metadata)
    assert out.shape == (batch, seq, 768), f"MetadataEncoder shape mismatch: {out.shape}"

    # build_metadata_tensor
    directions = torch.randint(0, 2, (batch, seq))
    sizes = torch.randint(40, 1500, (batch, seq))
    num_packets = torch.randint(10, seq + 1, (batch,))
    meta = build_metadata_tensor(timestamps, directions, sizes, num_packets)
    assert meta.shape == (batch, seq, 6), f"Metadata tensor shape mismatch: {meta.shape}"

    print("  All temporal encoding tests passed!")


def test_packet_encoder():
    print("Testing SimplePacketEncoder...")
    batch, num_pkts, pkt_len = 4, 50, 64
    encoder = SimplePacketEncoder(hidden_size=256, num_layers=2, num_heads=4, max_bytes=pkt_len)
    packet_bytes = torch.randint(0, 256, (batch, num_pkts, pkt_len))
    out = encoder(packet_bytes)
    assert out.shape == (batch, num_pkts, 256), f"PacketEncoder shape mismatch: {out.shape}"
    print("  SimplePacketEncoder test passed!")


def test_flow_encoder():
    print("Testing FlowEncoder...")
    batch, num_pkts, hidden = 4, 50, 256
    encoder = FlowEncoder(
        hidden_size=hidden, num_layers=2, num_heads=4,
        feedforward_size=512, window_size=8, max_packets=num_pkts,
        time2vec_dim=32,
    )

    packet_embeds = torch.randn(batch, num_pkts, hidden)
    timestamps = torch.cumsum(torch.rand(batch, num_pkts), dim=1)
    metadata = torch.randn(batch, num_pkts, 6)
    num_packets_t = torch.randint(10, num_pkts + 1, (batch,))

    flow_embed, pkt_ctx = encoder(packet_embeds, timestamps, metadata, num_packets_t)
    assert flow_embed.shape == (batch, hidden), f"FlowEmbed shape mismatch: {flow_embed.shape}"
    assert pkt_ctx.shape == (batch, num_pkts, hidden), f"PktCtx shape mismatch: {pkt_ctx.shape}"
    print("  FlowEncoder test passed!")


def test_full_model():
    print("Testing full TrafficLongFormer...")
    batch, max_pkts, pkt_len = 4, 50, 64
    num_classes = 10

    model = TrafficLongFormer(
        packet_encoder_type="simple",
        hidden_size=256,
        flow_num_layers=2,
        flow_num_heads=4,
        flow_feedforward=512,
        window_size=8,
        max_packets=max_pkts,
        num_classes=num_classes,
        dropout=0.1,
    )

    packet_bytes = torch.randint(0, 256, (batch, max_pkts, pkt_len))
    timestamps = torch.cumsum(torch.rand(batch, max_pkts), dim=1)
    directions = torch.randint(0, 2, (batch, max_pkts))
    sizes = torch.randint(40, 1500, (batch, max_pkts))
    num_packets = torch.randint(10, max_pkts + 1, (batch,))
    labels = torch.randint(0, num_classes, (batch,))

    # Forward without labels
    output = model(packet_bytes, timestamps, directions, sizes, num_packets)
    assert output["logits"].shape == (batch, num_classes), \
        f"Logits shape mismatch: {output['logits'].shape}"
    assert "loss" not in output

    # Forward with labels
    output = model(packet_bytes, timestamps, directions, sizes, num_packets, labels=labels)
    assert "loss" in output
    assert output["loss"].dim() == 0, "Loss should be scalar"

    # Backward pass
    output["loss"].backward()

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Full model: {param_count:,} params ({trainable:,} trainable)")
    print(f"  Loss: {output['loss'].item():.4f}")
    print("  Full model test passed!")


def test_pretraining_tasks():
    print("Testing pre-training tasks...")
    batch, max_pkts, pkt_len = 4, 50, 64

    # Test IPT bucketing
    ipts = torch.tensor([0.0005, 0.005, 0.05, 0.5, 5.0, 50.0])
    buckets = ipt_to_bucket_tensor(ipts)
    assert buckets.tolist() == [0, 1, 2, 3, 4, 5], f"Bucket mismatch: {buckets.tolist()}"

    # Test task manager
    config = {"dfp_positions": [5, 10, 20], "tov_swap_prob": 0.5}
    manager = PretrainingTaskManager(config)

    packet_bytes = torch.randint(0, 256, (batch, max_pkts, pkt_len))
    timestamps = torch.cumsum(torch.rand(batch, max_pkts), dim=1)
    num_packets = torch.full((batch,), max_pkts, dtype=torch.long)

    modified_ts, targets = manager.build_all_targets(packet_bytes, timestamps, num_packets)
    assert modified_ts.shape == timestamps.shape
    assert "dfp" in targets
    assert "iptp" in targets
    assert "tov" in targets
    print("  Pre-training tasks test passed!")


def test_pretraining_model():
    print("Testing PretrainingModel...")
    batch, max_pkts, pkt_len = 4, 50, 64

    base_model = TrafficLongFormer(
        packet_encoder_type="simple",
        hidden_size=256,
        flow_num_layers=2,
        flow_num_heads=4,
        flow_feedforward=512,
        window_size=8,
        max_packets=max_pkts,
        num_classes=10,
        dropout=0.1,
    )

    pretrain_model = PretrainingModel(base_model, {
        "dfp_num_classes": 8,
        "iptp_num_buckets": 6,
    })

    packet_bytes = torch.randint(0, 256, (batch, max_pkts, pkt_len))
    timestamps = torch.cumsum(torch.rand(batch, max_pkts), dim=1)
    directions = torch.randint(0, 2, (batch, max_pkts))
    sizes = torch.randint(40, 1500, (batch, max_pkts))
    num_packets = torch.full((batch,), max_pkts, dtype=torch.long)

    output = pretrain_model(
        packet_bytes, timestamps, directions, sizes, num_packets,
        dfp_targets={"positions": torch.tensor([5, 10, 15, 20]),
                     "labels": torch.tensor([0, 1, 2, 3]), "lambda": 0.5},
        iptp_targets={"positions": torch.tensor([3, 7, 12, 25]),
                      "labels": torch.tensor([0, 1, 2, 3]), "lambda": 0.5},
        tov_targets={"labels": torch.tensor([0, 1, 0, 1]), "lambda": 0.3},
    )

    assert "total_loss" in output
    assert output["total_loss"].dim() == 0
    output["total_loss"].backward()
    print(f"  Total pre-training loss: {output['total_loss'].item():.4f}")
    print("  PretrainingModel test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("TrafficLongFormer Smoke Tests")
    print("=" * 60)

    test_temporal_encoding()
    test_packet_encoder()
    test_flow_encoder()
    test_full_model()
    test_pretraining_tasks()
    test_pretraining_model()

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
