#!/usr/bin/env python3
"""
Single experiment training script.
Trains one model configuration and saves checkpoints + metrics.

Usage:
    python run_experiment.py --model gtcrn
    python run_experiment.py --model gtcrn_rat --chunk_size 8
"""

import sys

sys.path.append("../..")


import argparse
import json
import os
from torch.utils.data import DataLoader


from models import GTCRN
from rat_models import GTCRN_RAT
from utils import count_parameters, get_device, VCTKDatasetFromList
from train.gtcrn import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train GTCRN(_RAT) model with VCTK dataset"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["gtcrn", "gtcrn_rat"]
    )
    parser.add_argument(
        "--chunk_size", type=int, default=8, help="Chunk size for RAT/GRAT"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment_len", type=int, default=32000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../../VoiceBank+DEMAND")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = get_device()

    if args.model == "gtcrn":
        model = GTCRN()
        exp_name = "gtcrn"
    else:
        model = GTCRN_RAT(chunk_size=args.chunk_size)
        exp_name = f"gtcrn_rat/chunk_{args.chunk_size}"

    save_dir = os.path.join(args.output_dir, exp_name)

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Save dir: {save_dir}")
    print("=" * 60)

    print("\nLoading datasets...")
    train_dataset = VCTKDatasetFromList(
        file_list=f"{args.data_dir}/training.txt",
        clean_dir=f"{args.data_dir}/wavs_clean",
        noisy_dir=f"{args.data_dir}/wavs_noisy",
        segment_len=args.segment_len,
    )
    val_dataset = VCTKDatasetFromList(
        file_list=f"{args.data_dir}/test.txt",
        clean_dir=f"{args.data_dir}/wavs_clean",
        noisy_dir=f"{args.data_dir}/wavs_noisy",
        segment_len=args.segment_len,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    _, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        save_dir=save_dir,
        lr=args.lr,
        save_every=args.save_every,
    )

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    best_pesq = max(history["val_pesq"])
    best_sisnr = max(history["val_sisnr"])
    print("\n" + "=" * 60)
    print(f"FINISHED: {exp_name}")
    print(f"Best PESQ: {best_pesq:.3f}")
    print(f"Best SI-SNR: {best_sisnr:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
