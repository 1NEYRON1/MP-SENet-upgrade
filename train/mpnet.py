"""
Training utilities for MPNet models.
Unified training function with same interface as train_gtcrn.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

from utils import compute_pesq, compute_sisnr, mag_pha_stft, mag_pha_istft
from models.mpnet.model import phase_losses
from models.mpnet.discriminator import MetricDiscriminator, batch_pesq


def _save_checkpoint(
    epoch,
    model,
    discriminator,
    optim_g,
    optim_d,
    scheduler_g,
    scheduler_d,
    best_loss,
    best_pesq,
    history,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optim_g.state_dict(),
        "optimizer_d_state_dict": optim_d.state_dict(),
        "scheduler_g_state_dict": scheduler_g.state_dict(),
        "scheduler_d_state_dict": scheduler_d.state_dict(),
        "best_loss": best_loss,
        "best_pesq": best_pesq,
        "history": history,
    }
    torch.save(checkpoint, path)


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    save_dir,
    lr=5e-4,
    save_every=10,
    *,
    config,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    h = config
    model = model.to(device)
    discriminator = MetricDiscriminator().to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    optim_g = torch.optim.AdamW(model.parameters(), lr, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(
        discriminator.parameters(), lr, betas=[h.adam_b1, h.adam_b2]
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sisnr": [],
        "val_sisnr": [],
        "val_pesq": [],
        "train_mag_loss": [],
        "val_mag_loss": [],
        "train_pha_loss": [],
        "val_pha_loss": [],
    }

    best_loss = float("inf")
    best_pesq = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        discriminator.train()
        train_loss_sum = 0
        train_mag_sum = 0
        train_pha_sum = 0
        train_sisnr_sum = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for noisy_audio, clean_audio in pbar:
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            clean_audio = clean_audio.to(device, non_blocking=True)
            batch_size = clean_audio.size(0)
            one_labels = torch.ones(batch_size, device=device)

            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(
                mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            audio_list_r = list(clean_audio.cpu().numpy())
            audio_list_g = list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(
                    batch_pesq_score.to(device), metric_g.flatten()
                )
            else:
                loss_disc_g = torch.tensor(0.0, device=device)
            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

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

            sisnr = compute_sisnr(clean_audio, audio_g)
            train_loss_sum += loss_gen_all.item()
            train_mag_sum += loss_mag.item()
            train_pha_sum += loss_pha.item()
            train_sisnr_sum += sisnr

            pbar.set_postfix(
                {
                    "loss": f"{loss_gen_all.item():.4f}",
                    "sisnr": f"{sisnr:.2f}",
                }
            )

        scheduler_g.step()
        scheduler_d.step()

        n_train = len(train_loader)
        history["train_loss"].append(train_loss_sum / n_train)
        history["train_mag_loss"].append(train_mag_sum / n_train)
        history["train_pha_loss"].append(train_pha_sum / n_train)
        history["train_sisnr"].append(train_sisnr_sum / n_train)

        # Validation
        model.eval()
        val_loss_sum = 0
        val_mag_sum = 0
        val_pha_sum = 0
        val_sisnr_sum = 0
        pesq_scores = []

        with torch.no_grad():
            for noisy_audio, clean_audio in tqdm(val_loader, desc="Validation"):
                noisy_audio = noisy_audio.to(device, non_blocking=True)
                clean_audio = clean_audio.to(device, non_blocking=True)

                clean_mag, clean_pha, clean_com = mag_pha_stft(
                    clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                )
                noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
                    noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                )

                mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
                audio_g = mag_pha_istft(
                    mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                )

                loss_mag = F.mse_loss(clean_mag, mag_g)
                loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
                loss_pha = loss_ip + loss_gd + loss_iaf

                val_loss_sum += loss_mag.item()
                val_mag_sum += loss_mag.item()
                val_pha_sum += loss_pha.item()
                val_sisnr_sum += compute_sisnr(clean_audio, audio_g)

                for i in range(audio_g.size(0)):
                    pesq_scores.append(
                        compute_pesq(clean_audio[i].cpu(), audio_g[i].cpu())
                    )

        n_val = len(val_loader)
        val_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
        history["val_loss"].append(val_loss_sum / n_val)
        history["val_mag_loss"].append(val_mag_sum / n_val)
        history["val_pha_loss"].append(val_pha_sum / n_val)
        history["val_sisnr"].append(val_sisnr_sum / n_val)
        history["val_pesq"].append(val_pesq)

        print(
            f"\nEpoch {epoch}: "
            f"loss={history['train_loss'][-1]:.4f}/{history['val_loss'][-1]:.4f} "
            f"sisnr={history['train_sisnr'][-1]:.2f}/{history['val_sisnr'][-1]:.2f} "
            f"pesq={val_pesq:.3f} "
            f"lr={optim_g.param_groups[0]['lr']:.6f}"
        )

        val_loss = history["val_loss"][-1]
        if val_pesq > best_pesq:
            best_pesq = val_pesq
            _save_checkpoint(
                epoch,
                model,
                discriminator,
                optim_g,
                optim_d,
                scheduler_g,
                scheduler_d,
                best_loss,
                best_pesq,
                history,
                save_dir / "best_pesq.pt",
            )
            print(f"  -> Saved best model (loss={best_loss:.4f}, PESQ={best_pesq:.3f})")

        if val_loss < best_loss:
            best_loss = val_loss
            _save_checkpoint(
                epoch,
                model,
                discriminator,
                optim_g,
                optim_d,
                scheduler_g,
                scheduler_d,
                best_loss,
                best_pesq,
                history,
                save_dir / "best_loss.pt",
            )
            print(f"  -> Saved best model (loss={best_loss:.4f}, PESQ={best_pesq:.3f})")

        if epoch % save_every == 0:
            _save_checkpoint(
                epoch,
                model,
                discriminator,
                optim_g,
                optim_d,
                scheduler_g,
                scheduler_d,
                best_loss,
                best_pesq,
                history,
                save_dir / f"epoch_{epoch}.pt",
            )
            print(f"  -> Saved checkpoint epoch_{epoch}.pt")

    return model, history
