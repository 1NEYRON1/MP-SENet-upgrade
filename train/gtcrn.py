"""
Training utilities for GTCRN models.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from utils import compute_pesq


def spec_to_wav(spec, n_fft=512, hop_length=256, window=None):
    """Convert spectrogram (F, T, 2) or (B, F, T, 2) to waveform."""
    if window is None:
        window = torch.hann_window(n_fft, device=spec.device).pow(0.5)
    spec_complex = torch.complex(spec[..., 0], spec[..., 1])
    return torch.istft(spec_complex, n_fft, hop_length, n_fft, window)


class HybridLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mse = nn.MSELoss()
        self.register_buffer("window", torch.hann_window(n_fft).pow(0.5))

    def forward(self, pred_stft, true_stft):
        """
        pred_stft, true_stft: (B, F, T, 2)
        """
        pred_real, pred_imag = pred_stft[..., 0], pred_stft[..., 1]
        true_real, true_imag = true_stft[..., 0], true_stft[..., 1]

        pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_real**2 + true_imag**2 + 1e-12)

        pred_mag_07 = pred_mag**0.7
        true_mag_07 = true_mag**0.7

        pred_real_c = pred_real / pred_mag_07
        pred_imag_c = pred_imag / pred_mag_07
        true_real_c = true_real / true_mag_07
        true_imag_c = true_imag / true_mag_07

        real_loss = self.mse(pred_real_c, true_real_c)
        imag_loss = self.mse(pred_imag_c, true_imag_c)
        mag_loss = self.mse(pred_mag**0.3, true_mag**0.3)

        y_pred = torch.istft(
            pred_real + 1j * pred_imag,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self.window,
        )
        y_true = torch.istft(
            true_real + 1j * true_imag,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self.window,
        )
        y_true = (
            torch.sum(y_true * y_pred, dim=-1, keepdim=True)
            * y_true
            / (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-8)
        )

        sisnr = -torch.log10(
            torch.norm(y_true, dim=-1, keepdim=True) ** 2
            / (torch.norm(y_pred - y_true, dim=-1, keepdim=True) ** 2 + 1e-8)
            + 1e-8
        ).mean()

        total_loss = 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr

        loss_dict = {
            "real_loss": real_loss.item(),
            "imag_loss": imag_loss.item(),
            "mag_loss": mag_loss.item(),
            "sisnr": -sisnr.item() * 10,  # convert to dB
        }
        return total_loss, loss_dict


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    metrics = {"real_loss": 0, "imag_loss": 0, "mag_loss": 0, "sisnr": 0}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        pred = model(noisy)
        loss, loss_dict = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            metrics[k] += v

        pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

    n = len(train_loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device, n_fft=512, hop_length=256, verbose=True):
    model.to(device).eval()
    criterion = criterion.to(device)
    total_loss = 0
    metrics = {"real_loss": 0, "imag_loss": 0, "mag_loss": 0, "sisnr": 0}
    pesq_scores = []
    window = torch.hann_window(n_fft).pow(0.5).to(device)

    loader = tqdm(val_loader, desc="Validation") if verbose else val_loader
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        loss, loss_dict = criterion(pred, clean)

        total_loss += loss.item()
        for k, v in loss_dict.items():
            metrics[k] += v

        for i in range(pred.shape[0]):
            pred_wav = spec_to_wav(pred[i], n_fft, hop_length, window)
            clean_wav = spec_to_wav(clean[i], n_fft, hop_length, window)
            pesq_scores.append(compute_pesq(clean_wav, pred_wav))

    n = len(val_loader)
    total_loss /= n
    result = {k: v / n for k, v in metrics.items()}
    result["pesq"] = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0.0
    return total_loss, result


def _save_checkpoint(
    epoch, model, optimizer, scheduler, best_loss, best_pesq, history, path
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
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
    lr=1e-3,
    save_every=10,
    n_fft=512,
    hop_length=256,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = HybridLoss(n_fft=n_fft, hop_length=hop_length).to(device)

    best_loss = float("inf")
    best_pesq = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sisnr": [],
        "val_sisnr": [],
        "val_pesq": [],
        "train_real_loss": [],
        "val_real_loss": [],
        "train_imag_loss": [],
        "val_imag_loss": [],
        "train_mag_loss": [],
        "val_mag_loss": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, n_fft, hop_length
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pesq"].append(val_metrics["pesq"])
        for k in ["sisnr", "real_loss", "imag_loss", "mag_loss"]:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])

        print(
            f"""
            Epoch {epoch}: 
            loss={train_loss:.4f}/{val_loss:.4f}
            sisnr={train_metrics['sisnr']:.2f}/{val_metrics['sisnr']:.2f}
            pesq="N/A"/{val_metrics['pesq']:.3f}
            mag={train_metrics['mag_loss']:.4f}/{val_metrics['mag_loss']:.4f}
            re={train_metrics['real_loss']:.4f}/{val_metrics['real_loss']:.4f}
            im={train_metrics['imag_loss']:.4f}/{val_metrics['imag_loss']:.4f}
            lr={optimizer.param_groups[0]['lr']:.6f}
            """
        )

        if val_metrics["pesq"] > best_pesq:
            best_pesq = val_metrics["pesq"]
            _save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
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
                optimizer,
                scheduler,
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
                optimizer,
                scheduler,
                best_loss,
                best_pesq,
                history,
                save_dir / f"epoch_{epoch}.pt",
            )
            print(f"  -> Saved checkpoint epoch_{epoch}.pt")

    return model, history
