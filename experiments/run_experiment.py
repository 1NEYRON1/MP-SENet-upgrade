#!/usr/bin/env python3
"""
Unified experiment script: GTCRN, MinGRU, RAT (GTCRN + MP-SENet).
Trains one model and saves checkpoints + metrics.
"""

import sys
import os

# sbatch runs from experiments/<name>, wrapper chdirs to experiments/
sys.path.insert(0, "..")

import argparse
import random
import re

import torch
from torch.utils.data import DataLoader, Subset

from models import GTCRN, MPNet
from experiments.min_gru import MinGTCRN, MinMPNet
from experiments.rat import GTCRN_RAT, MPNet_RAT
from utils import count_parameters, load_config, get_device, VCTKDatasetFromList
from train import train_gtcrn, train_mpnet

GTCRN_CONFIG_PATH = os.path.join("..", "models", "gtcrn", "config.json")
MPNET_CONFIG_PATH = os.path.join("..", "models", "mpnet", "config.json")


def get_model(name, config=None, chunk_size=8):
    match name:
        case "gtcrn":
            return GTCRN()
        case "min_gtcrn":
            return MinGTCRN()
        case "gtcrn_rat":
            return GTCRN_RAT(chunk_size=chunk_size)
        case "mpnet":
            return MPNet(config)
        case "min_mpnet":
            return MinMPNet(config)
        case "mpnet_rat":
            return MPNet_RAT(config, chunk_size=chunk_size)
        case _:
            raise ValueError(f"Unknown model: {name}")


def main():
    default_data_dir = os.path.join("..", "VoiceBank+DEMAND")
    default_output_dir = "checkpoints"
    parser = argparse.ArgumentParser(description="Train GTCRN / MP-SENet (incl. MinGRU, RAT)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gtcrn", "min_gtcrn", "gtcrn_rat", "mpnet", "min_mpnet", "mpnet_rat"],
    )
    parser.add_argument("--chunk_size", type=int, default=8, help="Chunk size for RAT (gtcrn_rat, mpnet_rat)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument(
        "--segment_size",
        type=int,
        default=None,
        help="Override config segment_size (samples). Default: from config.",
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=None,
        help="Use a subset of training data (0 < ratio <= 1). Default: use all.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override config learning_rate.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override config num_workers.")
    args = parser.parse_args()

    data_dir = os.path.normpath(args.data_dir)
    output_dir = os.path.normpath(args.output_dir)
    model_name = args.model

    train_list = os.path.join(data_dir, "train.txt")
    val_list = os.path.join(data_dir, "test.txt")
    clean_dir = os.path.join(data_dir, "wavs_clean")
    noisy_dir = os.path.join(data_dir, "wavs_noisy")
    for p in (train_list, val_list, clean_dir, noisy_dir):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Data path not found: {p}")

    is_mpnet_family = bool(re.search(r"mpnet", model_name))
    config_path = MPNET_CONFIG_PATH if is_mpnet_family else GTCRN_CONFIG_PATH
    config = load_config(config_path)
    segment_len = args.segment_size if args.segment_size is not None else config.segment_size
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    lr = args.lr if args.lr is not None else config.learning_rate
    num_workers = args.num_workers if args.num_workers is not None else getattr(config, "num_workers", 1)
    epochs = args.epochs

    random.seed(42)
    torch.manual_seed(42)

    train_ds = VCTKDatasetFromList(
        train_list,
        clean_dir,
        noisy_dir,
        segment_len=segment_len,
        return_audio=is_mpnet_family,
    )
    if args.subset_ratio is not None:
        r = max(0.0, min(1.0, args.subset_ratio))
        n = max(1, int(len(train_ds) * r))
        train_ds = Subset(train_ds, random.Random(12345).sample(range(len(train_ds)), n))
        print(f"Using subset: {n} samples (ratio={r})")

    val_ds = VCTKDatasetFromList(
        val_list,
        clean_dir,
        noisy_dir,
        segment_len=segment_len,
        return_audio=is_mpnet_family,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if is_mpnet_family:
        model = get_model(model_name, config=config, chunk_size=args.chunk_size)
    else:
        model = get_model(model_name, chunk_size=args.chunk_size)

    print(f"Model: {model_name}, params: {count_parameters(model):,}")

    if "rat" in model_name:
        save_dir = os.path.join(output_dir, f"{model_name}", f"chunk_{args.chunk_size}")
    else:
        save_dir = os.path.join(output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    device = get_device()
    if is_mpnet_family:
        train_mpnet(
            model,
            train_loader,
            val_loader,
            epochs,
            device,
            save_dir,
            lr=lr,
            save_every=args.save_every,
            config=config,
        )
    else:
        train_gtcrn(
            model,
            train_loader,
            val_loader,
            epochs,
            device,
            save_dir,
            lr=lr,
            save_every=args.save_every,
        )
    print("Started training.")


if __name__ == "__main__":
    main()
