"""
Pre-training script for TrafficLongFormer.

Trains the flow encoder (with frozen packet encoder) on the three
timing-aware pre-training tasks: DFP, IPTP, TOV.

Usage (single GPU):
    python scripts/train_pretrain.py --config configs/pretrain.yaml

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=3 scripts/train_pretrain.py --config configs/pretrain.yaml
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.long_context_traffic_model import TrafficLongFormer, PretrainingModel
from model.pretraining_tasks import PretrainingTaskManager
from data.flow_dataset import FlowDataset, collate_flows


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="TrafficLongFormer pre-training")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data.corpus_path from config")
    parser.add_argument("--trafficformer_dir", type=str, default=None,
                        help="Path to TrafficFormer repo (for packet encoder)")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to nomoe_bertflow_pre-trained_model.bin")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="Path to TrafficFormer BPE vocab")
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--use_simple_encoder", action="store_true",
                        help="Use SimplePacketEncoder instead of TrafficFormer")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(rank, f"Using device: {device} (world_size={world_size})")

    # Data
    data_dir = args.data_dir or config["data"].get("corpus_path")
    if data_dir is None:
        log(rank, "ERROR: Provide --data_dir or set data.corpus_path in config")
        sys.exit(1)

    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        # If no train/ subdir, use data_dir directly
        train_dir = data_dir

    log(rank, f"Loading data from {train_dir}")
    dataset = FlowDataset(
        pcap_dir=train_dir,
        max_packets=config["data"]["max_packets"],
        packet_len=config["data"]["packet_bytes"],
        start_index=config["data"]["start_index"],
    )

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_flows,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Model
    encoder_type = "simple" if args.use_simple_encoder else "trafficformer"
    if encoder_type == "trafficformer" and not args.trafficformer_dir:
        log(rank, "WARNING: No --trafficformer_dir provided, falling back to SimplePacketEncoder")
        encoder_type = "simple"

    flow_cfg = config["model"]["flow_encoder"]
    base_model = TrafficLongFormer(
        packet_encoder_type=encoder_type,
        trafficformer_dir=args.trafficformer_dir or "",
        pretrained_path=args.pretrained_path,
        vocab_path=args.vocab_path,
        hidden_size=flow_cfg["hidden_size"],
        flow_num_layers=flow_cfg["num_layers"],
        flow_num_heads=flow_cfg["num_heads"],
        flow_feedforward=flow_cfg["feedforward_size"],
        window_size=flow_cfg["window_size"],
        max_packets=flow_cfg["max_packets"],
        num_classes=config["tasks"]["sodf"]["num_classes"],
        dropout=flow_cfg["dropout"],
        freeze_packet_encoder=config["model"]["packet_encoder"]["freeze"],
    )

    task_config = {
        "dfp_positions": config["tasks"]["dfp"]["target_positions"],
        "dfp_lambda": config["tasks"]["dfp"]["lambda"],
        "iptp_lambda": config["tasks"]["iptp"]["lambda"],
        "tov_swap_prob": config["tasks"]["tov"]["swap_prob"],
        "tov_lambda": config["tasks"]["tov"]["lambda"],
    }
    model = PretrainingModel(base_model, task_config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Task manager
    task_manager = PretrainingTaskManager(task_config)

    # Optimizer
    tcfg = config["training"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    total_steps = tcfg["total_steps"]
    warmup_steps = int(total_steps * tcfg["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    try:
        scaler = torch.amp.GradScaler("cuda") if tcfg.get("fp16") and device.type == "cuda" else None
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler() if tcfg.get("fp16") and device.type == "cuda" else None

    # Resume
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        log(rank, f"Resumed from step {global_step}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    log(rank, f"Starting pre-training: {total_steps} steps, batch_size={tcfg['batch_size']}x{world_size}")
    log(rank, f"Dataset: {len(dataset)} flows, {dataset.num_classes} classes")
    log(rank, f"Encoder: {encoder_type}, Frozen: {config['model']['packet_encoder']['freeze']}")

    model.train()
    running_loss = 0.0
    log_interval = 50
    t0 = time.time()

    while global_step < total_steps:
        if sampler is not None:
            sampler.set_epoch(global_step // len(dataloader))

        for batch in dataloader:
            if global_step >= total_steps:
                break

            # Move to device
            packet_bytes = batch["packet_bytes"].to(device)
            timestamps = batch["timestamps"].to(device)
            directions = batch["directions"].to(device)
            sizes = batch["sizes"].to(device)
            num_packets = batch["num_packets"].to(device)

            # Build pre-training targets
            modified_ts, targets = task_manager.build_all_targets(
                packet_bytes, timestamps, num_packets
            )

            # Forward
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        packet_bytes=packet_bytes,
                        timestamps=modified_ts,
                        directions=directions,
                        sizes=sizes,
                        num_packets=num_packets,
                        dfp_targets=targets["dfp"],
                        iptp_targets=targets["iptp"],
                        tov_targets=targets["tov"],
                    )
                    loss = outputs["total_loss"]

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    packet_bytes=packet_bytes,
                    timestamps=modified_ts,
                    directions=directions,
                    sizes=sizes,
                    num_packets=num_packets,
                    dfp_targets=targets["dfp"],
                    iptp_targets=targets["iptp"],
                    tov_targets=targets["tov"],
                )
                loss = outputs["total_loss"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                optimizer.step()

            scheduler.step()
            global_step += 1
            running_loss += loss.item()

            # Log
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - t0
                steps_per_sec = log_interval / elapsed
                lr = scheduler.get_last_lr()[0]

                task_losses = ""
                for key in ["dfp_loss", "iptp_loss", "tov_loss"]:
                    if key in outputs:
                        task_losses += f"  {key}={outputs[key].item():.4f}"

                log(rank, f"step {global_step}/{total_steps} | "
                    f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                    f"{steps_per_sec:.2f} steps/s |{task_losses}")

                running_loss = 0.0
                t0 = time.time()

            # Save checkpoint
            if global_step % tcfg["save_steps"] == 0 and rank == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                }, ckpt_path)
                log(rank, f"Saved checkpoint: {ckpt_path}")

    # Final save
    if rank == 0:
        model_to_save = model.module if hasattr(model, "module") else model
        final_path = os.path.join(args.output_dir, "pretrain_final.pt")
        torch.save({
            "global_step": global_step,
            "model_state_dict": model_to_save.state_dict(),
            "config": config,
        }, final_path)
        log(rank, f"Pre-training complete! Final model saved to {final_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
