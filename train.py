import argparse
import logging
import os
import shutil
import time

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from torch.amp import autocast
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from dataset import Dataset, get_dataset_filelist, mag_pha_istft, mag_pha_stft
from models.discriminator import AsyncPESQ, MetricDiscriminator
from models.model import MPNet, pesq_score, phase_losses
from utils import load_checkpoint, load_config, save_checkpoint, scan_checkpoint, set_seed

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

logger = logging.getLogger("train")


def train(a, h):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    generator = MPNet(h).to(device)
    discriminator = MetricDiscriminator(dim=h.disc_dim, dropout=h.disc_dropout).to(device)

    if rank == 0:
        console = Console()
        os.makedirs(h.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(h.checkpoint_path, "logs"), exist_ok=True)

        fh = logging.FileHandler(os.path.join(h.checkpoint_path, "train.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)

        num_params = sum(p.numel() for p in generator.parameters())
        logger.info("Generator:\n%s", generator)
        logger.info("Total Parameters: %.3fM", num_params / 1e6)
        logger.info("Checkpoints directory: %s", h.checkpoint_path)
        console.print(
            f"Parameters: [bold]{num_params / 1e6:.3f}M[/bold] | Checkpoints: {h.checkpoint_path}"
        )

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, "g_")
        cp_do = scan_checkpoint(h.checkpoint_path, "do_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        gen_state = {k.removeprefix("_orig_mod."): v for k, v in state_dict_g["generator"].items()}
        disc_state = {
            k.removeprefix("_orig_mod."): v for k, v in state_dict_do["discriminator"].items()
        }
        generator.load_state_dict(gen_state)
        discriminator.load_state_dict(disc_state)
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    generator = torch.compile(generator, mode="reduce-overhead")
    discriminator = torch.compile(discriminator)

    if distributed:
        generator = DistributedDataParallel(generator, device_ids=[local_rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[local_rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = get_cosine_schedule_with_warmup(
        optim_g,
        num_warmup_steps=h.warmup_epochs,
        num_training_steps=h.epochs,
        last_epoch=last_epoch,
    )
    scheduler_d = get_cosine_schedule_with_warmup(
        optim_d,
        num_warmup_steps=h.warmup_epochs,
        num_training_steps=h.epochs,
        last_epoch=last_epoch,
    )

    if state_dict_do is not None:
        scheduler_g.load_state_dict(state_dict_do["scheduler_g"])
        scheduler_d.load_state_dict(state_dict_do["scheduler_d"])

    training_indexes, validation_indexes = get_dataset_filelist(h)

    trainset = Dataset(
        training_indexes,
        h.input_clean_wavs_dir,
        h.input_noisy_wavs_dir,
        h.segment_size,
        h.sampling_rate,
        split=True,
        n_cache_reuse=0,
        shuffle=not distributed,
        device=device,
        seed=h.seed,
    )

    train_sampler = DistributedSampler(trainset) if distributed else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    if rank == 0:
        validset = Dataset(
            validation_indexes,
            h.input_clean_wavs_dir,
            h.input_noisy_wavs_dir,
            h.segment_size,
            h.sampling_rate,
            split=False,
            shuffle=False,
            n_cache_reuse=0,
            device=device,
        )

        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs"))

    generator.train()
    discriminator.train()

    async_pesq = AsyncPESQ(max_workers=h.train_pesq_workers)
    best_pesq = 0
    one_labels = torch.ones(h.batch_size, device=device)

    for epoch in range(max(0, last_epoch), h.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            epoch_start = time.time()
            desc = ""
            progress = Progress(
                TextColumn(f"[bold]Epoch {epoch + 1}/{h.epochs}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeRemainingColumn(compact=True),
                TextColumn("{task.description}"),
                console=console,
            )
            progress.start()
            task_id = progress.add_task("", total=len(train_loader))

        for i, batch in enumerate(train_loader):
            clean_audio, noisy_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)

            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, _ = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            with autocast("cuda", dtype=torch.bfloat16):
                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(
                mag_g.float(), pha_g.float(), h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            audio_list_r, audio_list_g = (
                list(clean_audio.cpu().numpy()),
                list(audio_g.detach().cpu().numpy()),
            )
            async_pesq.submit(audio_list_r, audio_list_g, h.sampling_rate)

            # Discriminator
            optim_d.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                metric_r = discriminator(clean_mag, clean_mag)
                metric_g = discriminator(clean_mag, mag_g_hat.detach())
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

                batch_pesq_score = async_pesq.collect()
                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
                else:
                    loss_disc_g = torch.tensor(0.0, device=device)

                loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                # L2 Magnitude Loss
                loss_mag = F.mse_loss(clean_mag, mag_g)
                # Anti-wrapping Phase Loss
                loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
                loss_pha = loss_ip + loss_gd + loss_iaf
                # L2 Complex Loss
                loss_com = F.mse_loss(clean_com, com_g) * 2
                # L2 Consistency Loss
                loss_stft = F.mse_loss(com_g, com_g_hat) * 2
                # Time Loss
                loss_time = F.l1_loss(clean_audio, audio_g)
                # Metric Loss
                metric_g = discriminator(clean_mag, mag_g_hat)
                loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

                w = h.loss_weights
                loss_gen_all = (
                    loss_mag * w["mag"]
                    + loss_pha * w["pha"]
                    + loss_com * w["com"]
                    + loss_stft * w["stft"]
                    + loss_metric * w["metric"]
                    + loss_time * w["time"]
                )

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                progress.update(task_id, advance=1)

                if steps % h.stdout_interval == 0:
                    pesq_str = (
                        f"PESQ={batch_pesq_score.mean().item() * 3.5 + 1:.2f} "
                        if batch_pesq_score is not None
                        else ""
                    )
                    desc = (
                        f"{pesq_str}Gen={loss_gen_all.item():.3f} Disc={loss_disc_all.item():.3f} "
                        f"Mag={loss_mag.item():.3f} Pha={loss_pha.item():.3f} "
                        f"Time={loss_time.item():.3f}"
                    )
                    progress.update(task_id, description=desc)
                    logger.info(
                        "Step %d | Gen=%.4f Disc=%.4f Metric=%.4f Mag=%.4f Pha=%.4f Com=%.4f Time=%.4f STFT=%.4f",
                        steps,
                        loss_gen_all.item(),
                        loss_disc_all.item(),
                        loss_metric.item(),
                        loss_mag.item(),
                        loss_pha.item(),
                        loss_com.item() / 2,
                        loss_time.item(),
                        loss_stft.item() / 2,
                    )

                # Checkpointing
                if steps % h.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{h.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator.module if distributed else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = f"{h.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "discriminator": (
                                discriminator.module if distributed else discriminator
                            ).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "scheduler_g": scheduler_g.state_dict(),
                            "scheduler_d": scheduler_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % h.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all.item(), steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all.item(), steps)
                    sw.add_scalar("Training/Metric Loss", loss_metric.item(), steps)
                    sw.add_scalar("Training/Magnitude Loss", loss_mag.item(), steps)
                    sw.add_scalar("Training/Phase Loss", loss_pha.item(), steps)
                    sw.add_scalar("Training/Complex Loss", loss_com.item() / 2, steps)
                    sw.add_scalar("Training/Time Loss", loss_time.item(), steps)
                    sw.add_scalar("Training/Consistency Loss", loss_stft.item() / 2, steps)
                    sw.add_scalar("Training/Learning Rate", scheduler_g.get_last_lr()[0], steps)

                # Validation
                if steps % h.validation_interval == 0 and steps != 0:
                    progress.update(task_id, description="[yellow]Validating...[/yellow]")
                    val_task_id = progress.add_task("[yellow]Validation", total=len(validation_loader))
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_stft_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, noisy_audio = batch
                            clean_audio = clean_audio.to(device, non_blocking=True)
                            noisy_audio = noisy_audio.to(device, non_blocking=True)

                            clean_mag, clean_pha, clean_com = mag_pha_stft(
                                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                            )
                            noisy_mag, noisy_pha, _ = mag_pha_stft(
                                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                            )

                            with autocast("cuda", dtype=torch.bfloat16):
                                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

                            audio_g = mag_pha_istft(
                                mag_g.float(),
                                pha_g.float(),
                                h.n_fft,
                                h.hop_size,
                                h.win_size,
                                h.compress_factor,
                            )
                            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                            )
                            audios_r += torch.split(clean_audio, 1, dim=0)  # [1, T] * B
                            audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g.float()).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(
                                clean_pha, pha_g.float()
                            )
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g.float()).item()
                            val_stft_err_tot += F.mse_loss(com_g.float(), com_g_hat).item()
                            progress.update(val_task_id, advance=1)

                        val_mag_err = val_mag_err_tot / (j + 1)
                        val_pha_err = val_pha_err_tot / (j + 1)
                        val_com_err = val_com_err_tot / (j + 1)
                        val_stft_err = val_stft_err_tot / (j + 1)
                        val_pesq_score = pesq_score(audios_r, audios_g, h).item()

                        progress.remove_task(val_task_id)
                        progress.update(task_id, description=desc)
                        console.print(
                            f"  [green]Val[/green] step {steps} | "
                            f"PESQ: [bold]{val_pesq_score:.3f}[/bold] "
                            f"Mag: {val_mag_err:.4f} Pha: {val_pha_err:.4f}"
                        )
                        logger.info(
                            "Validation step %d | PESQ=%.3f Mag=%.4f Pha=%.4f Com=%.4f STFT=%.4f",
                            steps,
                            val_pesq_score,
                            val_mag_err,
                            val_pha_err,
                            val_com_err,
                            val_stft_err,
                        )

                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar("Validation/Consistency Loss", val_stft_err, steps)

                    if epoch >= h.best_checkpoint_start_epoch:
                        if val_pesq_score > best_pesq:
                            best_pesq = val_pesq_score
                            best_checkpoint_path = f"{h.checkpoint_path}/g_best"
                            save_checkpoint(
                                best_checkpoint_path,
                                {
                                    "generator": (
                                        generator.module if distributed else generator
                                    ).state_dict()
                                },
                            )
                            logger.info("New best PESQ: %.3f, saved g_best", best_pesq)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            progress.stop()
            elapsed = int(time.time() - epoch_start)
            mins, secs = divmod(elapsed, 60)
            console.print(f"Epoch {epoch + 1} done in {mins}m{secs:02d}s | {desc}")
            logger.info("Epoch %d done in %ds", epoch + 1, elapsed)

    async_pesq.shutdown()


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    a = parser.parse_args()

    h = load_config(a.config)

    os.makedirs(h.checkpoint_path, exist_ok=True)
    dest = os.path.join(h.checkpoint_path, "config.yaml")
    if a.config != dest:
        shutil.copyfile(a.config, dest)

    set_seed(h.seed)

    train(a, h)


if __name__ == "__main__":
    main()
