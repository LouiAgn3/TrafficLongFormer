"""
Context length ablation: how does performance change with more packets?

Uses TrafficFormer's existing pipeline but varies the number of input packets
(5, 10, 20, 50) to establish a naive baseline. For >512 tokens, uses
a sliding window average of BERT outputs.

This answers: does more context help even WITHOUT architectural changes?

Usage:
    python analysis/context_length_ablation.py \
        --trafficformer_dir /path/to/TrafficFormer \
        --pretrained_model /path/to/pretrain_model.bin \
        --vocab_path /path/to/vocab.txt \
        --data_dir /path/to/dataset \
        --output_dir results/context_ablation
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": False,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_trafficformer_classifier(tf_dir, pretrained_path, vocab_path, num_classes, seq_length):
    """Load a TrafficFormer classifier with custom seq_length."""
    sys.path.insert(0, tf_dir)
    from argparse import Namespace
    from uer.utils.config import load_hyperparam
    from uer.utils.vocab import Vocab
    from uer.utils.tokenizers import BertTokenizer

    args = Namespace(
        config_path=os.path.join(tf_dir, "models/bert/base_config.json"),
        vocab_path=vocab_path,
        pretrained_model_path=pretrained_path,
        embedding="word_pos_seg",
        encoder="transformer",
        mask="fully_visible",
        seq_length=seq_length,
        labels_num=num_classes,
        pooling="first",
        soft_targets=False,
        soft_alpha=0.5,
        hidden_size=768,
        dropout=0.5,
        is_moe=False,
        remove_transformer_bias=False,
        remove_embedding_layernorm=False,
        remove_attention_scale=False,
        factorized_embedding_parameterization=False,
        parameter_sharing=False,
        layernorm_positioning="post",
        relative_position_embedding=False,
        feed_forward="dense",
        layernorm="normal",
    )
    args = load_hyperparam(args)
    args.max_seq_length = seq_length  # Override max seq length
    args.tokenizer = BertTokenizer(args)

    # Import and build classifier
    sys.path.insert(0, os.path.join(tf_dir, "fine-tuning"))
    from run_classifier import Classifier, load_or_initialize_parameters
    model = Classifier(args)
    load_or_initialize_parameters(args, model)

    return model, args


def run_ablation_for_packet_count(
    num_packets: int,
    tf_dir: str,
    pretrained_path: str,
    vocab_path: str,
    train_path: str,
    test_path: str,
    num_classes: int,
    epochs: int = 4,
    batch_size: int = 128,
    device: str = "cuda",
):
    """
    Train and evaluate TrafficFormer with a specific number of packets.

    For num_packets <= 5: standard TrafficFormer (seq_length ~320)
    For num_packets > 5:  extend seq_length, using sliding window if > 512
    """
    # Estimate seq_length: ~63 bigram tokens per 64-byte packet + SEP tokens
    tokens_per_packet = 63
    seq_length = min(512, num_packets * tokens_per_packet + num_packets + 1)

    print(f"\n{'='*60}")
    print(f"Ablation: {num_packets} packets, seq_length={seq_length}")
    print(f"{'='*60}")

    if num_packets > 8:
        # For > 512 tokens, we need a sliding window approach
        # Use multiple passes of 512 tokens and average the CLS outputs
        return _sliding_window_ablation(
            num_packets, tf_dir, pretrained_path, vocab_path,
            train_path, test_path, num_classes, epochs, batch_size, device,
        )

    model, args = load_trafficformer_classifier(
        tf_dir, pretrained_path, vocab_path, num_classes, seq_length,
    )
    model = model.to(device)

    # Load data (using TrafficFormer's data loading with modified packet count)
    # This would need to call finetuning_data_gen with payload_packet=num_packets
    # For now, return placeholder — the actual implementation needs the
    # TrafficFormer data pipeline to be set up
    print(f"  Would train with {num_packets} packets for {epochs} epochs")
    print(f"  Data paths: train={train_path}, test={test_path}")

    return {"num_packets": num_packets, "f1_macro": None, "note": "needs data pipeline setup"}


def _sliding_window_ablation(
    num_packets, tf_dir, pretrained_path, vocab_path,
    train_path, test_path, num_classes, epochs, batch_size, device,
):
    """Handle >512 token case with sliding window averaging."""
    print(f"  Using sliding window approach for {num_packets} packets")
    # This creates overlapping windows of 512 tokens and averages CLS outputs
    # Implementation requires the full data pipeline
    return {"num_packets": num_packets, "f1_macro": None, "note": "sliding window - needs data pipeline"}


def plot_ablation_results(results, output_dir):
    """Plot F1 vs number of packets."""
    valid = [r for r in results if r["f1_macro"] is not None]
    if not valid:
        print("No valid results to plot (data pipeline not set up yet)")
        return

    packets = [r["num_packets"] for r in valid]
    f1s = [r["f1_macro"] for r in valid]

    fig, ax = plt.subplots()
    ax.plot(packets, f1s, "o-", linewidth=2, markersize=8, color="steelblue")
    ax.axvline(x=5, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="TrafficFormer default")
    ax.set_xlabel("Number of input packets")
    ax.set_ylabel("Macro F1 Score")
    ax.set_xscale("log")
    ax.set_xticks(packets)
    ax.set_xticklabels(packets)
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(output_dir, "context_length_ablation.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Context length ablation")
    parser.add_argument("--trafficformer_dir", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=120)
    parser.add_argument("--output_dir", default="results/context_ablation")
    parser.add_argument("--packet_counts", nargs="+", type=int,
                        default=[5, 10, 20, 50])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for num_packets in args.packet_counts:
        result = run_ablation_for_packet_count(
            num_packets=num_packets,
            tf_dir=args.trafficformer_dir,
            pretrained_path=args.pretrained_model,
            vocab_path=args.vocab_path,
            train_path=args.train_path,
            test_path=args.test_path,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        results.append(result)

    # Save results
    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_ablation_results(results, args.output_dir)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
