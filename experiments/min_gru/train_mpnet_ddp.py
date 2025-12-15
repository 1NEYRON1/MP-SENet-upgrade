#!/usr/bin/env python3
"""
DDP training script for MPNet models (based on original MP-SENet training).
Supports multi-GPU training with DistributedDataParallel.

Usage:
    # Single GPU:
    python train_mpnet_ddp.py --model mpnet --config ../../models/mpnet/config.json
    
    # Multi-GPU (2 GPUs):
    torchrun --nproc_per_node=2 train_mpnet_ddp.py --model mpnet --config ../../models/mpnet/config.json
"""

import sys
sys.path.append("../..")

import os
import argparse
import json
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from pathlib import Path

from models import MPNet
from experiments.min_gru import MinMPNet
from utils import (
    load_config, 
    VCTKDatasetFromList, 
    compute_pesq, 
    compute_sisnr,
    mag_pha_stft, 
    mag_pha_istft,
    count_parameters,
)
from models.mpnet.model import phase_losses
from models.mpnet.discriminator import MetricDiscriminator, batch_pesq

torch.backends.cudnn.benchmark = True


def setup_ddp(rank, world_size):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed process group."""
    destroy_process_group()


def get_model(name, config):
    if name == "mpnet":
        return MPNet(config)
    if name == "min_mpnet":
        return MinMPNet(config)
    raise ValueError(f"Unknown model: {name}")


def save_checkpoint(path, model, discriminator, optim_g, optim_d, epoch, best_pesq):
    """Save checkpoint, extracting module from DDP wrapper if needed."""
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    disc_state = discriminator.module.state_dict() if hasattr(discriminator, "module") else discriminator.state_dict()
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "discriminator_state_dict": disc_state,
        "optimizer_g_state_dict": optim_g.state_dict(),
        "optimizer_d_state_dict": optim_d.state_dict(),
        "best_pesq": best_pesq,
    }, path)


def train(rank, world_size, args, config):
    """Training function for each GPU process."""
    is_main = rank == 0
    
    # Setup DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Create model
    model = get_model(args.model, config).to(device)
    discriminator = MetricDiscriminator().to(device)
    
    if is_main:
        print(f"Model: {args.model}")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"GPUs: {world_size}")
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])
    
    # Optimizers
    h = config
    optim_g = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)
    
    # Dataset
    train_dataset = VCTKDatasetFromList(
        f"{args.data_dir}/training.txt",
        f"{args.data_dir}/wavs_clean",
        f"{args.data_dir}/wavs_noisy",
        segment_len=h.segment_size,
        return_audio=True,
    )
    val_dataset = VCTKDatasetFromList(
        f"{args.data_dir}/test.txt",
        f"{args.data_dir}/wavs_clean",
        f"{args.data_dir}/wavs_noisy",
        segment_len=h.segment_size,
        return_audio=True,
    )
    
    # Samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    
    # Batch size per GPU
    batch_size = h.batch_size // world_size if world_size > 1 else h.batch_size
    if is_main:
        print(f"Batch size per GPU: {batch_size}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=h.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Validation only on main process
    if is_main:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
    
    # Create save directory
    save_dir = Path(args.output_dir) / args.model
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    best_pesq = 0.0
    best_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        discriminator.train()
        
        if is_main:
            start_time = time.time()
            print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss_sum = 0
        
        for i, (noisy_audio, clean_audio) in enumerate(train_loader):
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            clean_audio = clean_audio.to(device, non_blocking=True)
            bs = clean_audio.size(0)
            one_labels = torch.ones(bs, device=device)
            
            # STFT
            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            
            # Generator forward
            mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            
            # PESQ for discriminator
            audio_list_r = list(clean_audio.cpu().numpy())
            audio_list_g = list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)
            
            # Discriminator step
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = torch.tensor(0.0, device=device)
            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()
            
            # Generator step
            optim_g.zero_grad()
            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            
            loss_gen_all = (
                loss_mag * 0.9
                + loss_pha * 0.3
                + loss_com * 0.1
                + loss_stft * 0.1
                + loss_metric * 0.05
                + loss_time * 0.2
            )
            loss_gen_all.backward()
            optim_g.step()
            
            train_loss_sum += loss_gen_all.item()
            
            if is_main and i % 50 == 0:
                print(f"  Step {i}/{len(train_loader)}, Loss: {loss_gen_all.item():.4f}")
        
        scheduler_g.step()
        scheduler_d.step()
        
        # Validation (main process only)
        if is_main:
            epoch_time = time.time() - start_time
            avg_loss = train_loss_sum / len(train_loader)
            print(f"  Train Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
            
            # Validation
            model.eval()
            torch.cuda.empty_cache()
            pesq_scores = []
            
            with torch.no_grad():
                for noisy_audio, clean_audio in val_loader:
                    noisy_audio = noisy_audio.to(device)
                    clean_audio = clean_audio.to(device)
                    
                    noisy_mag, noisy_pha, _ = mag_pha_stft(
                        noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                    )
                    mag_g, pha_g, _ = model(noisy_mag, noisy_pha)
                    audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                    
                    pesq_scores.append(compute_pesq(clean_audio[0].cpu(), audio_g[0].cpu()))
            
            val_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
            print(f"  Val PESQ: {val_pesq:.3f}")
            

            if val_pesq > best_pesq:
                best_pesq = val_pesq
                save_checkpoint(
                    save_dir / "best_pesq.pt",
                    model, discriminator, optim_g, optim_d, epoch, best_pesq
                )
                print(f"  -> Saved best model (PESQ={best_pesq:.3f})")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    save_dir / "best_loss.pt",
                    model, discriminator, optim_g, optim_d, epoch, best_pesq
                )
                print(f"  -> Saved best model (Loss={best_loss:.4f})")
            
            # Periodic save
            if epoch % args.save_every == 0:
                save_checkpoint(
                    save_dir / f"epoch_{epoch}.pt",
                    model, discriminator, optim_g, optim_d, epoch, best_pesq
                )
    
    if is_main:
        print(f"\nTraining complete. Best PESQ: {best_pesq:.3f}, Best Loss: {best_loss:.4f}")
        with open(save_dir / "metrics.json", "w") as f:
            json.dump({"best_pesq": best_pesq, "best_loss": best_loss}, f)
    
    if world_size > 1:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Train MPNet with DDP")
    parser.add_argument("--model", type=str, required=True, choices=["mpnet", "min_mpnet"])
    parser.add_argument("--config", type=str, default="../../models/mpnet/config.json")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../../VoiceBank+DEMAND")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Check for torchrun environment
    if "LOCAL_RANK" in os.environ:
        # Launched with torchrun
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train(rank, world_size, args, config)
    else:
        # Single GPU
        world_size = torch.cuda.device_count()
        if world_size > 1:
            print(f"Detected {world_size} GPUs. Use torchrun for multi-GPU:")
            print(f"  torchrun --nproc_per_node={world_size} train_mpnet_ddp.py ...")
        train(0, 1, args, config)


if __name__ == "__main__":
    main()

