"""
Fine-tuning script for TrafficLongFormer.

Loads a pre-trained checkpoint and fine-tunes on downstream classification.

Usage:
    python scripts/train_finetune.py \
        --config configs/finetune.yaml \
        --pretrained checkpoints/pretrain/pretrain_final.pt \
        --data_dir data/ustc_flows
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.long_context_traffic_model import TrafficLongFormer
from data.flow_dataset import FlowDataset, collate_flows


def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                packet_bytes=batch["packet_bytes"],
                timestamps=batch["timestamps"],
                directions=batch["directions"],
                sizes=batch["sizes"],
                num_packets=batch["num_packets"],
            )
            if criterion is not None:
                loss = criterion(outputs["logits"], batch["labels"])
                total_loss += loss.item()
            else:
                total_loss += outputs.get("loss", torch.tensor(0.0)).item()
            n_batches += 1
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="TrafficLongFormer fine-tuning")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory with train/val/test subdirs")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetune")
    parser.add_argument("--use_simple_encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dcfg = config["data"]
    train_ds = FlowDataset(
        os.path.join(args.data_dir, "train"),
        max_packets=dcfg["max_packets"],
        packet_len=dcfg["packet_bytes"],
        start_index=dcfg["start_index"],
    )
    val_ds = FlowDataset(
        os.path.join(args.data_dir, "val"),
        max_packets=dcfg["max_packets"],
        packet_len=dcfg["packet_bytes"],
        start_index=dcfg["start_index"],
    )
    test_ds = FlowDataset(
        os.path.join(args.data_dir, "test"),
        max_packets=dcfg["max_packets"],
        packet_len=dcfg["packet_bytes"],
        start_index=dcfg["start_index"],
    )

    tcfg = config["training"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              collate_fn=collate_flows, num_workers=tcfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            collate_fn=collate_flows, num_workers=tcfg["num_workers"],
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             collate_fn=collate_flows, num_workers=tcfg["num_workers"],
                             pin_memory=True)

    num_classes = train_ds.num_classes
    print(f"Classes: {num_classes} — {train_ds.label_names}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Compute class weights for imbalanced data
    class_counts = np.zeros(num_classes)
    for _, label_idx in train_ds.samples:
        class_counts[label_idx] += 1
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    fcfg = config["model"]["flow_encoder"]
    model = TrafficLongFormer(
        packet_encoder_type="simple" if args.use_simple_encoder else "simple",
        hidden_size=fcfg["hidden_size"],
        flow_num_layers=fcfg["num_layers"],
        flow_num_heads=fcfg["num_heads"],
        flow_feedforward=fcfg["hidden_size"] * 4,
        window_size=fcfg["window_size"],
        max_packets=fcfg["max_packets"],
        num_classes=num_classes,
        freeze_packet_encoder=False,
    ).to(device)

    # Load pre-trained weights
    if args.pretrained and os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # Filter to base_model keys (strip "base_model." prefix from pretraining checkpoint)
        base_state = {}
        for k, v in state.items():
            clean_k = k.replace("base_model.", "") if k.startswith("base_model.") else k
            # Skip classifier weights (size mismatch expected) and task heads
            if clean_k.startswith("classifier.") or clean_k.startswith("dfp_head.") or \
               clean_k.startswith("iptp_head.") or clean_k.startswith("tov_head."):
                continue
            base_state[clean_k] = v
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        print(f"Loaded pre-trained weights: {len(base_state)} keys")
        if missing:
            print(f"  Missing: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected: {unexpected[:5]}...")
    else:
        print("No pre-trained checkpoint — training from scratch")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    total_train_steps = len(train_loader) * tcfg["epochs"]
    warmup_steps = int(total_train_steps * tcfg["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_train_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    try:
        scaler = torch.amp.GradScaler("cuda") if tcfg.get("fp16") and device.type == "cuda" else None
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler() if tcfg.get("fp16") and device.type == "cuda" else None

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = 0.0
    patience = 0

    # Training
    for epoch in range(tcfg["epochs"]):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        packet_bytes=batch["packet_bytes"],
                        timestamps=batch["timestamps"],
                        directions=batch["directions"],
                        sizes=batch["sizes"],
                        num_packets=batch["num_packets"],
                    )
                    loss = criterion(outputs["logits"], batch["labels"])
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    packet_bytes=batch["packet_bytes"],
                    timestamps=batch["timestamps"],
                    directions=batch["directions"],
                    sizes=batch["sizes"],
                    num_packets=batch["num_packets"],
                )
                loss = criterion(outputs["logits"], batch["labels"])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} | "
                      f"loss={running_loss / (step + 1):.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{tcfg['epochs']} ({elapsed:.0f}s) | "
              f"train_loss={running_loss / len(train_loader):.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1={val_metrics['f1_macro']:.4f}")

        # Early stopping
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> New best F1: {best_f1:.4f}, saved checkpoint")
        else:
            patience += 1
            if patience >= tcfg["early_stop"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test with best model
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (seed={args.seed}):")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weight): {test_metrics['f1_weighted']:.4f}")
    print(f"{'='*60}")

    # Save test results
    import json
    with open(os.path.join(args.output_dir, f"test_results_seed{args.seed}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
