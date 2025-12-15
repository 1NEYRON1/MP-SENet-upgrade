#!/usr/bin/env python3
"""
Single experiment training script for MinGRU experiments.
Trains one model configuration and saves checkpoints + metrics.

Usage:
    python run_experiment.py --model min_gtcrn
"""

import sys

sys.path.append("../..")

import argparse
import json
import os

from torch.utils.data import DataLoader

from models import GTCRN, MPNet
from experiments.min_gru import MinGTCRN, MinMPNet
from utils import count_parameters, load_config, get_device, VCTKDatasetFromList
from train import train_gtcrn, train_mpnet


GTCRN_CONFIG_PATH = "../../models/gtcrn/config.json"
MPNET_CONFIG_PATH = "../../models/mpnet/config.json"


def get_model(name, config=None):
    if name == "gtcrn":
        return GTCRN()
    if name == "min_gtcrn":
        return MinGTCRN()
    if name == "mpnet":
        return MPNet(config)
    if name == "min_mpnet":
        return MinMPNet(config)
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train MinGRU experiments")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gtcrn", "min_gtcrn", "mpnet", "min_mpnet"],
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../../VoiceBank+DEMAND")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    is_mpnet = "mpnet" in args.model
    config_path = MPNET_CONFIG_PATH if is_mpnet else GTCRN_CONFIG_PATH
    config = load_config(config_path)
    model = get_model(args.model, config)
    save_dir = os.path.join(args.output_dir, args.model)
    device = get_device()

    batch_size = config.batch_size
    lr = config.learning_rate
    segment_len = config.segment_size
    num_workers = config.num_workers

    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Segment length: {segment_len}")
    print(f"Epochs: {args.epochs}")
    print(f"Save dir: {save_dir}")
    print("=" * 60)

    train_dataset = VCTKDatasetFromList(
        f"{args.data_dir}/training.txt",
        f"{args.data_dir}/wavs_clean",
        f"{args.data_dir}/wavs_noisy",
        segment_len=segment_len,
        return_audio=is_mpnet,
    )
    val_dataset = VCTKDatasetFromList(
        f"{args.data_dir}/test.txt",
        f"{args.data_dir}/wavs_clean",
        f"{args.data_dir}/wavs_noisy",
        segment_len=segment_len,
        return_audio=is_mpnet,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_fn = train_mpnet if is_mpnet else train_gtcrn
    kwargs = (
        {"config": config}
        if is_mpnet
        else {"n_fft": config.n_fft, "hop_length": config.hop_length}
    )

    _, history = train_fn(
        model,
        train_loader,
        val_loader,
        args.epochs,
        device,
        save_dir,
        lr,
        args.save_every,
        **kwargs,
    )

    if history:
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nBest PESQ: {max(history['val_pesq']):.3f}")
        print(f"Best SI-SNR: {max(history['val_sisnr']):.2f} dB")


if __name__ == "__main__":
    main()
