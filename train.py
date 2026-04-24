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
from models.model import (
    MPNet,
    PESQEvaluator,
    build_phase_loss_weight,
    build_tfrep_features,
    geometry_consistency_losses,
    geometry_head_losses,
    get_feature_channels,
    phase_losses,
    reliability_loss,
    reliability_sparse_loss,
)
from utils import load_checkpoint, load_config, save_checkpoint, scan_checkpoint, set_seed

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

logger = logging.getLogger("train")


def _get_attr(h, name, default):
    return getattr(h, name, default)


def _resolve_metric_feature_mode(h):
    mode = _get_attr(h, "metric_feature_mode", "mag")
    if mode == "generator":
        return _get_attr(h, "phase_input_feature_mode", "baseline")
    return mode


def _is_geometry_mode(h):
    return _get_attr(h, "phase_decoder_mode", "absolute").lower().startswith("geometry")


def _linear_ramp(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return max(0.0, min(1.0, float(step) / float(warmup_steps)))


def train(a, h):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    metric_feature_mode = _resolve_metric_feature_mode(h)
    metric_in_channel = get_feature_channels(metric_feature_mode) * 2

    generator = MPNet(h).to(device)
    discriminator = MetricDiscriminator(
        dim=h.disc_dim,
        in_channel=metric_in_channel,
        dropout=h.disc_dropout,
    ).to(device)

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
        logger.info("Metric discriminator feature mode: %s", metric_feature_mode)
        console.print(
            f"Parameters: [bold]{num_params / 1e6:.3f}M[/bold] | "
            f"Checkpoints: {h.checkpoint_path} | Disc features: {metric_feature_mode}"
        )

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, "g_")
        cp_do = scan_checkpoint(h.checkpoint_path, "do_")
    else:
        cp_g = None
        cp_do = None

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

    compile_enabled = bool(_get_attr(h, "compile_enabled", True))
    compile_mode = _get_attr(h, "compile_mode", "reduce-overhead")
    compile_validation = bool(_get_attr(h, "compile_validation", False))
    if bool(_get_attr(h, "compile_skip_dynamic_cudagraphs", True)):
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraph_skip_dynamic_graphs = True
            inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
        except Exception:
            if rank == 0:
                logger.warning("Unable to set torch._inductor dynamic-cudagraph options.")

    if compile_enabled:
        generator = torch.compile(generator, mode=compile_mode)
        discriminator = torch.compile(discriminator)

    if distributed:
        generator = DistributedDataParallel(generator, device_ids=[local_rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[local_rank]).to(device)

    train_generator = generator.module if distributed else generator
    val_forward_generator = train_generator
    if not compile_validation:
        val_forward_generator = getattr(train_generator, "_orig_mod", train_generator)

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

    training_entries, validation_entries = get_dataset_filelist(h)

    trainset = Dataset(
        training_entries,
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
            validation_entries,
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
    val_pesq_workers = int(_get_attr(h, "val_pesq_workers", 1))
    val_pesq_fallback_sequential = bool(_get_attr(h, "val_pesq_fallback_sequential", True))
    val_pesq_evaluator = None
    if rank == 0:
        val_pesq_evaluator = PESQEvaluator(
            sampling_rate=h.sampling_rate,
            max_workers=val_pesq_workers,
            fallback_sequential=val_pesq_fallback_sequential,
        )

    best_pesq = 0.0
    one_labels = torch.ones(h.batch_size, device=device)

    curriculum_enabled = bool(_get_attr(h, "curriculum_enabled", False))
    metric_warmup_steps = int(_get_attr(h, "metric_warmup_steps", 0))
    geometry_warmup_steps = int(_get_attr(h, "geometry_warmup_steps", 0))
    geometry_sparse_lambda = float(_get_attr(h, "phase_geometry_sparse_loss", 0.0))

    use_weighted_phase_loss = bool(_get_attr(h, "use_weighted_phase_loss", False))
    phase_weight_power = float(_get_attr(h, "phase_loss_weight_power", 1.0))
    phase_weight_floor = float(_get_attr(h, "phase_loss_weight_floor", 0.0))
    phase_weight_detach = bool(_get_attr(h, "phase_loss_weight_detach", True))
    phase_confidence_mode = _get_attr(h, "phase_decoder_mode", "absolute").lower()
    phase_confidence_lambda = float(_get_attr(h, "phase_confidence_loss", 0.0))
    enable_confidence_supervision = (
        phase_confidence_mode == "residual_blend" and phase_confidence_lambda > 0.0
    )

    use_geometry_decoder = _is_geometry_mode(h)
    geometry_anchor_mode = _get_attr(h, "phase_geometry_anchor_mode", "residual")
    geometry_use_noisy_skip = bool(_get_attr(h, "phase_geometry_use_noisy_skip", True))
    geometry_anchor_lambda = float(_get_attr(h, "phase_geometry_anchor_loss", 0.0))
    geometry_gd_lambda = float(_get_attr(h, "phase_geometry_gd_loss", 0.0))
    geometry_iaf_lambda = float(_get_attr(h, "phase_geometry_iaf_loss", 0.0))
    geometry_reliability_lambda = float(_get_attr(h, "phase_geometry_reliability_loss", 0.0))
    geometry_consistency_lambda = float(_get_attr(h, "phase_geometry_consistency_loss", 0.0))
    geometry_reliability_temperature = float(
        _get_attr(h, "phase_geometry_reliability_temperature", 0.35)
    )
    geometry_sparse_power = float(_get_attr(h, "phase_geometry_sparse_power", 1.0))

    quick_val_subset_size = int(_get_attr(h, "quick_val_subset_size", 0))
    full_val_interval = int(_get_attr(h, "full_val_interval", h.validation_interval))

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

        for _i, batch in enumerate(train_loader):
            clean_audio, noisy_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)

            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, _ = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            phase_weight = None
            if use_weighted_phase_loss:
                phase_weight = build_phase_loss_weight(
                    clean_mag,
                    power=phase_weight_power,
                    floor=phase_weight_floor,
                    detach=phase_weight_detach,
                )

            metric_curr_scale = (
                _linear_ramp(steps, metric_warmup_steps) if curriculum_enabled else 1.0
            )
            geometry_curr_scale = (
                _linear_ramp(steps, geometry_warmup_steps) if curriculum_enabled else 1.0
            )

            with autocast("cuda", dtype=torch.bfloat16):
                mag_g, pha_g, com_g, aux = generator(noisy_mag, noisy_pha, return_aux=True)
                (
                    phase_residual_g,
                    phase_confidence_g,
                    phase_anchor_token_g,
                    phase_gd_g,
                    phase_iaf_g,
                    phase_weight_logits_g,
                    phase_candidate_stack_g,
                ) = aux

            audio_g = mag_pha_istft(
                mag_g.float(), pha_g.float(), h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            audio_list_r = list(clean_audio.cpu().numpy())
            audio_list_g = list(audio_g.detach().cpu().numpy())
            async_pesq.submit(audio_list_r, audio_list_g, h.sampling_rate)

            clean_metric_features = build_tfrep_features(clean_mag, clean_pha, metric_feature_mode)
            generated_metric_features = build_tfrep_features(
                mag_g_hat.detach(), pha_g_hat.detach(), metric_feature_mode
            )

            optim_d.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                metric_r = discriminator(clean_metric_features, clean_metric_features)
                metric_g = discriminator(clean_metric_features, generated_metric_features)
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

                batch_pesq_score = async_pesq.collect()
                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
                else:
                    loss_disc_g = torch.tensor(0.0, device=device)

                loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                loss_mag = F.mse_loss(clean_mag, mag_g)
                loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, weight=phase_weight)
                loss_pha = loss_ip + loss_gd + loss_iaf
                loss_com = F.mse_loss(clean_com, com_g) * 2
                loss_stft = F.mse_loss(com_g, com_g_hat) * 2
                loss_time = F.l1_loss(clean_audio, audio_g)

                metric_g = discriminator(
                    clean_metric_features,
                    build_tfrep_features(mag_g_hat, pha_g_hat, metric_feature_mode),
                )
                loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

                loss_conf = torch.tensor(0.0, device=device)
                if enable_confidence_supervision:
                    confidence_target = build_phase_loss_weight(
                        clean_mag,
                        power=float(_get_attr(h, "phase_confidence_target_power", 0.5)),
                        floor=0.0,
                        detach=True,
                    )
                    loss_conf = F.l1_loss(phase_confidence_g.float(), confidence_target)

                loss_geom_anchor = torch.tensor(0.0, device=device)
                loss_geom_gd = torch.tensor(0.0, device=device)
                loss_geom_iaf = torch.tensor(0.0, device=device)
                loss_geom_rel = torch.tensor(0.0, device=device)
                loss_geom_cons = torch.tensor(0.0, device=device)
                loss_geom_sparse = torch.tensor(0.0, device=device)

                if use_geometry_decoder:
                    loss_geom_anchor, loss_geom_gd, loss_geom_iaf = geometry_head_losses(
                        clean_pha,
                        noisy_pha,
                        phase_anchor_token_g,
                        phase_gd_g,
                        phase_iaf_g,
                        weight=phase_weight,
                        anchor_mode=geometry_anchor_mode,
                    )

                    if geometry_reliability_lambda > 0.0:
                        loss_geom_rel, _ = reliability_loss(
                            phase_weight_logits_g,
                            phase_candidate_stack_g,
                            clean_pha,
                            weight=phase_weight,
                            temperature=geometry_reliability_temperature,
                            use_noisy_skip=geometry_use_noisy_skip,
                        )
                    if geometry_sparse_lambda > 0.0:
                        loss_geom_sparse = reliability_sparse_loss(
                            phase_weight_logits_g,
                            weight=phase_weight,
                            power=geometry_sparse_power,
                        )

                    if geometry_consistency_lambda > 0.0:
                        cons_gd, cons_iaf = geometry_consistency_losses(
                            pha_g,
                            phase_gd_g,
                            phase_iaf_g,
                            weight=phase_weight,
                        )
                        loss_geom_cons = cons_gd + cons_iaf

                w = h.loss_weights
                metric_weight = w["metric"] * metric_curr_scale
                geometry_reliability_weight = geometry_reliability_lambda * geometry_curr_scale
                geometry_sparse_weight = geometry_sparse_lambda * geometry_curr_scale
                loss_gen_all = (
                    loss_mag * w["mag"]
                    + loss_pha * w["pha"]
                    + loss_com * w["com"]
                    + loss_stft * w["stft"]
                    + loss_metric * metric_weight
                    + loss_time * w["time"]
                    + loss_conf * phase_confidence_lambda
                    + loss_geom_anchor * geometry_anchor_lambda
                    + loss_geom_gd * geometry_gd_lambda
                    + loss_geom_iaf * geometry_iaf_lambda
                    + loss_geom_rel * geometry_reliability_weight
                    + loss_geom_sparse * geometry_sparse_weight
                    + loss_geom_cons * geometry_consistency_lambda
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
                        f"GeomA={loss_geom_anchor.item():.3f} GeomG={loss_geom_gd.item():.3f} "
                        f"GeomI={loss_geom_iaf.item():.3f} Rel={loss_geom_rel.item():.3f} "
                        f"GeomS={loss_geom_sparse.item():.3f} "
                        f"mW={metric_curr_scale:.2f} rW={geometry_curr_scale:.2f} "
                        f"Time={loss_time.item():.3f} Conf={phase_confidence_g.mean().item():.3f}"
                    )
                    progress.update(task_id, description=desc)
                    logger.info(
                        (
                            "Step %d | Gen=%.4f Disc=%.4f Metric=%.4f Mag=%.4f Pha=%.4f "
                            "Com=%.4f Time=%.4f STFT=%.4f GeomA=%.4f GeomG=%.4f GeomI=%.4f "
                            "GeomRel=%.4f GeomSparse=%.4f GeomCons=%.4f ConfLoss=%.4f "
                            "ConfMean=%.4f MetricScale=%.4f GeomScale=%.4f"
                        ),
                        steps,
                        loss_gen_all.item(),
                        loss_disc_all.item(),
                        loss_metric.item(),
                        loss_mag.item(),
                        loss_pha.item(),
                        loss_com.item() / 2,
                        loss_time.item(),
                        loss_stft.item() / 2,
                        loss_geom_anchor.item(),
                        loss_geom_gd.item(),
                        loss_geom_iaf.item(),
                        loss_geom_rel.item(),
                        loss_geom_sparse.item(),
                        loss_geom_cons.item(),
                        loss_conf.item(),
                        phase_confidence_g.mean().item(),
                        metric_curr_scale,
                        geometry_curr_scale,
                    )

                if steps % h.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{h.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {"generator": (generator.module if distributed else generator).state_dict()},
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

                if steps % h.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all.item(), steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all.item(), steps)
                    sw.add_scalar("Training/Metric Loss", loss_metric.item(), steps)
                    sw.add_scalar("Training/Metric Loss Weighted", (loss_metric * metric_weight).item(), steps)
                    sw.add_scalar("Training/Magnitude Loss", loss_mag.item(), steps)
                    sw.add_scalar("Training/Phase Loss", loss_pha.item(), steps)
                    sw.add_scalar("Training/Complex Loss", loss_com.item() / 2, steps)
                    sw.add_scalar("Training/Time Loss", loss_time.item(), steps)
                    sw.add_scalar("Training/Consistency Loss", loss_stft.item() / 2, steps)
                    sw.add_scalar("Training/Confidence Loss", loss_conf.item(), steps)
                    sw.add_scalar("Training/Confidence Mean", phase_confidence_g.mean().item(), steps)
                    sw.add_scalar("Training/Geometry Anchor Loss", loss_geom_anchor.item(), steps)
                    sw.add_scalar("Training/Geometry GD Loss", loss_geom_gd.item(), steps)
                    sw.add_scalar("Training/Geometry IAF Loss", loss_geom_iaf.item(), steps)
                    sw.add_scalar("Training/Geometry Reliability Loss", loss_geom_rel.item(), steps)
                    sw.add_scalar("Training/Geometry Sparse Loss", loss_geom_sparse.item(), steps)
                    sw.add_scalar("Training/Geometry Consistency Loss", loss_geom_cons.item(), steps)
                    sw.add_scalar("Training/Curriculum Metric Scale", metric_curr_scale, steps)
                    sw.add_scalar("Training/Curriculum Geometry Scale", geometry_curr_scale, steps)
                    sw.add_scalar("Training/Learning Rate", scheduler_g.get_last_lr()[0], steps)

                if steps % h.validation_interval == 0 and steps != 0:
                    is_full_validation = full_val_interval <= 0 or steps % full_val_interval == 0
                    max_val_batches = len(validation_loader)
                    if not is_full_validation and quick_val_subset_size > 0:
                        max_val_batches = min(max_val_batches, quick_val_subset_size)
                    val_scope = "full" if is_full_validation else "quick"

                    progress.update(task_id, description="[yellow]Validating...[/yellow]")
                    val_task_id = progress.add_task(
                        f"[yellow]Validation ({val_scope})", total=max_val_batches
                    )
                    train_generator.eval()
                    val_forward_generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0.0
                    val_pha_err_tot = 0.0
                    val_com_err_tot = 0.0
                    val_stft_err_tot = 0.0
                    val_conf_mean_tot = 0.0
                    val_geom_anchor_tot = 0.0
                    val_geom_gd_tot = 0.0
                    val_geom_iaf_tot = 0.0
                    val_geom_rel_tot = 0.0
                    processed_batches = 0
                    with torch.no_grad():
                        for batch in validation_loader:
                            if processed_batches >= max_val_batches:
                                break

                            clean_audio, noisy_audio = batch
                            clean_audio = clean_audio.to(device, non_blocking=True)
                            noisy_audio = noisy_audio.to(device, non_blocking=True)

                            clean_mag, clean_pha, clean_com = mag_pha_stft(
                                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                            )
                            noisy_mag, noisy_pha, _ = mag_pha_stft(
                                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
                            )

                            val_phase_weight = None
                            if use_weighted_phase_loss:
                                val_phase_weight = build_phase_loss_weight(
                                    clean_mag,
                                    power=phase_weight_power,
                                    floor=phase_weight_floor,
                                    detach=True,
                                )

                            with autocast("cuda", dtype=torch.bfloat16):
                                mag_g, pha_g, com_g, aux = val_forward_generator(
                                    noisy_mag, noisy_pha, return_aux=True
                                )
                                (
                                    _phase_residual_g,
                                    phase_confidence_g,
                                    phase_anchor_token_g,
                                    phase_gd_g,
                                    phase_iaf_g,
                                    phase_weight_logits_g,
                                    phase_candidate_stack_g,
                                ) = aux

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

                            audios_r += torch.split(clean_audio, 1, dim=0)
                            audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g.float()).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(
                                clean_pha, pha_g.float(), weight=val_phase_weight
                            )
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g.float()).item()
                            val_stft_err_tot += F.mse_loss(com_g.float(), com_g_hat).item()
                            val_conf_mean_tot += phase_confidence_g.float().mean().item()

                            if use_geometry_decoder:
                                vg_a, vg_g, vg_i = geometry_head_losses(
                                    clean_pha,
                                    noisy_pha,
                                    phase_anchor_token_g.float(),
                                    phase_gd_g.float(),
                                    phase_iaf_g.float(),
                                    weight=val_phase_weight,
                                    anchor_mode=geometry_anchor_mode,
                                )
                                val_geom_anchor_tot += vg_a.item()
                                val_geom_gd_tot += vg_g.item()
                                val_geom_iaf_tot += vg_i.item()
                                if geometry_reliability_lambda > 0.0:
                                    vg_rel, _ = reliability_loss(
                                        phase_weight_logits_g.float(),
                                        phase_candidate_stack_g.float(),
                                        clean_pha,
                                        weight=val_phase_weight,
                                        temperature=geometry_reliability_temperature,
                                        use_noisy_skip=geometry_use_noisy_skip,
                                    )
                                    val_geom_rel_tot += vg_rel.item()

                            processed_batches += 1
                            progress.update(val_task_id, advance=1)

                        denom = max(processed_batches, 1)
                        val_mag_err = val_mag_err_tot / denom
                        val_pha_err = val_pha_err_tot / denom
                        val_com_err = val_com_err_tot / denom
                        val_stft_err = val_stft_err_tot / denom
                        val_conf_mean = val_conf_mean_tot / denom
                        val_geom_anchor = val_geom_anchor_tot / denom
                        val_geom_gd = val_geom_gd_tot / denom
                        val_geom_iaf = val_geom_iaf_tot / denom
                        val_geom_rel = val_geom_rel_tot / denom
                        val_pesq_score = (
                            val_pesq_evaluator.score(audios_r, audios_g)
                            if val_pesq_evaluator is not None
                            else -1.0
                        )

                        progress.remove_task(val_task_id)
                        progress.update(task_id, description=desc)
                        console.print(
                            f"  [green]Val-{val_scope}[/green] step {steps} | "
                            f"PESQ: [bold]{val_pesq_score:.3f}[/bold] "
                            f"Mag: {val_mag_err:.4f} Pha: {val_pha_err:.4f} "
                            f"GeomA: {val_geom_anchor:.4f} GeomG: {val_geom_gd:.4f} "
                            f"GeomI: {val_geom_iaf:.4f} Conf: {val_conf_mean:.3f}"
                        )
                        logger.info(
                            (
                                "Validation step %d | Scope=%s PESQ=%.3f Mag=%.4f Pha=%.4f "
                                "Com=%.4f STFT=%.4f GeomA=%.4f GeomG=%.4f GeomI=%.4f "
                                "GeomRel=%.4f ConfMean=%.4f"
                            ),
                            steps,
                            val_scope,
                            val_pesq_score,
                            val_mag_err,
                            val_pha_err,
                            val_com_err,
                            val_stft_err,
                            val_geom_anchor,
                            val_geom_gd,
                            val_geom_iaf,
                            val_geom_rel,
                            val_conf_mean,
                        )

                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar("Validation/Consistency Loss", val_stft_err, steps)
                        sw.add_scalar("Validation/Confidence Mean", val_conf_mean, steps)
                        sw.add_scalar("Validation/Geometry Anchor Loss", val_geom_anchor, steps)
                        sw.add_scalar("Validation/Geometry GD Loss", val_geom_gd, steps)
                        sw.add_scalar("Validation/Geometry IAF Loss", val_geom_iaf, steps)
                        sw.add_scalar("Validation/Geometry Reliability Loss", val_geom_rel, steps)
                        sw.add_scalar("Validation/Is Full", float(is_full_validation), steps)

                    if is_full_validation and epoch >= h.best_checkpoint_start_epoch:
                        if val_pesq_score > best_pesq:
                            best_pesq = val_pesq_score
                            best_checkpoint_path = f"{h.checkpoint_path}/g_best"
                            save_checkpoint(
                                best_checkpoint_path,
                                {"generator": (generator.module if distributed else generator).state_dict()},
                            )
                            logger.info("New best PESQ: %.3f, saved g_best", best_pesq)

                    train_generator.train()
                    val_forward_generator.train()

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
    if val_pesq_evaluator is not None:
        val_pesq_evaluator.shutdown()


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
