import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")
import os
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from dataset import Dataset, mag_pha_stft, mag_pha_istft, get_dataset_filelist
from models.discriminator import MetricDiscriminator, batch_pesq
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True


def build_generator(h):
    """Return (generator_instance, pesq_score_fn, phase_losses_fn)."""
    gen_type = getattr(h, 'generator_type', 'baseline')
    num_tsblocks = getattr(h, 'num_tsconformers', 4)
    if gen_type == 'mingru':
        from models.model_mingru import MPNet, pesq_score, phase_losses
    else:  # 'baseline'
        from models.model import MPNet, pesq_score, phase_losses
    return MPNet(h, num_tsblocks=num_tsblocks), pesq_score, phase_losses


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator that operates on multiple STFT resolutions.
    Each resolution uses a separate MetricDiscriminator.
    """
    def __init__(self, resolutions=None, compress_factor=0.3):
        super().__init__()
        if resolutions is None:
            resolutions = [[400, 100, 400], [1024, 120, 600], [256, 50, 120]]
        self.discriminators = nn.ModuleList([MetricDiscriminator() for _ in resolutions])
        self.resolutions = resolutions
        self.compress_factor = compress_factor

    def forward(self, clean_wav, gen_wav):
        """
        Args:
            clean_wav: Clean audio waveform [B, T]
            gen_wav: Generated audio waveform [B, T]
        Returns:
            metric_rs: List of discriminator outputs for (clean, clean) pairs
            metric_gs: List of discriminator outputs for (clean, generated) pairs
        """
        metric_rs, metric_gs = [], []
        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            clean_mag, _, _ = mag_pha_stft(clean_wav, n_fft, hop, win, compress_factor=self.compress_factor)
            gen_mag, _, _   = mag_pha_stft(gen_wav,   n_fft, hop, win, compress_factor=self.compress_factor)
            metric_rs.append(self.discriminators[i](clean_mag, clean_mag))
            metric_gs.append(self.discriminators[i](clean_mag, gen_mag))
        return metric_rs, metric_gs


def build_discriminator(h):
    disc_type = getattr(h, 'discriminator_type', 'single')
    if disc_type == 'multi_scale':
        resolutions = getattr(h, 'discriminator_resolutions',
                              [[400, 100, 400], [1024, 120, 600], [256, 50, 120]])
        return MultiScaleDiscriminator(resolutions=resolutions, compress_factor=h.compress_factor)
    else:  # 'single'
        return MetricDiscriminator()


def build_ssl_loss(h, device):
    if not getattr(h, 'use_ssl_loss', False):
        return None
    from models.ssl_loss import SSLLossSimple
    model_name = getattr(h, 'ssl_model_name', 'nvidia/ssl_en_nest_xlarge_v1.0')
    return SSLLossSimple(model_name=model_name, device=device)


def get_loss_weights(h):
    defaults = {'mag': 0.9, 'pha': 0.3, 'com': 0.1, 'stft': 0.1, 'metric': 0.05, 'time': 0.2}
    lw = getattr(h, 'loss_weights', {})
    if isinstance(lw, dict):
        defaults.update(lw)
    return AttrDict(defaults)


def train(rank, a, h):
    disc_type = getattr(h, 'discriminator_type', 'single')
    use_ssl = getattr(h, 'use_ssl_loss', False)
    ssl_loss_weight = getattr(h, 'ssl_loss_weight', 0.1)
    lw = get_loss_weights(h)

    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator, pesq_score, phase_losses = build_generator(h)
    generator = generator.to(device)

    discriminator = build_discriminator(h).to(device)

    ssl_loss_module = None
    if use_ssl:
        ssl_loss_module = build_ssl_loss(h, device)
        if rank == 0:
            print(f"SSL Loss enabled — model: {getattr(h, 'ssl_model_name', 'nvidia/ssl_en_nest_xlarge_v1.0')}, "
                  f"weight: {ssl_loss_weight}")

    if rank == 0:
        print(generator)
        num_params_g = sum(p.numel() for p in generator.parameters())
        num_params_d = sum(p.numel() for p in discriminator.parameters())
        print(f'Generator Parameters : {num_params_g/1e6:.3f}M')
        print(f'Discriminator Parameters: {num_params_d/1e6:.3f}M  (type={disc_type})')
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory :", a.checkpoint_path)


    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_indexes, validation_indexes = get_dataset_filelist(a)
    trainset = Dataset(training_indexes, a.input_clean_wavs_dir, a.input_noisy_wavs_dir,
                       h.segment_size, h.sampling_rate,
                       split=True, n_cache_reuse=0,
                       shuffle=(h.num_gpus <= 1), device=device)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler, batch_size=h.batch_size,
                              pin_memory=True, drop_last=True)

    if rank == 0:
        validset = Dataset(validation_indexes, a.input_clean_wavs_dir, a.input_noisy_wavs_dir,
                           h.segment_size, h.sampling_rate,
                           split=False, shuffle=False, n_cache_reuse=0, device=device)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None, batch_size=1,
                                       pin_memory=True, drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    discriminator.train()
    best_pesq = 0

    is_multi = (disc_type == 'multi_scale')
    if is_multi:
        resolutions = getattr(h, 'discriminator_resolutions',
                              [[400, 100, 400], [1024, 120, 600], [256, 50, 120]])
        num_resolutions = len(resolutions)

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            clean_audio, noisy_audio = batch
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            audio_list_r = list(clean_audio.cpu().numpy())
            audio_list_g = list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            optim_d.zero_grad()

            if is_multi:
                metric_rs, metric_gs = discriminator(clean_audio, audio_g.detach())
                loss_disc_r = sum(F.mse_loss(one_labels, mr.flatten()) for mr in metric_rs) / num_resolutions
                if batch_pesq_score is not None:
                    loss_disc_g = sum(
                        F.mse_loss(batch_pesq_score.to(device), mg.flatten()) for mg in metric_gs
                    ) / num_resolutions
                else:
                    if rank == 0 and steps % a.stdout_interval == 0:
                        print('pesq is None!')
                    loss_disc_g = 0
            else:
                metric_r = discriminator(clean_mag, clean_mag)
                metric_g = discriminator(clean_mag, mag_g_hat.detach())
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
                else:
                    if rank == 0 and steps % a.stdout_interval == 0:
                        print('pesq is None!')
                    loss_disc_g = 0

            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad()

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
            if is_multi:
                _, metric_gs = discriminator(clean_audio, audio_g)
                loss_metric = sum(
                    F.mse_loss(mg.flatten(), one_labels) for mg in metric_gs
                ) / num_resolutions
            else:
                metric_g = discriminator(clean_mag, mag_g_hat)
                loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            # Combine generator losses
            loss_gen_all = (loss_mag    * lw.mag  +
                            loss_pha    * lw.pha  +
                            loss_com    * lw.com  +
                            loss_stft   * lw.stft +
                            loss_metric * lw.metric +
                            loss_time   * lw.time)

            # Optional SSL loss
            loss_ssl = None
            if use_ssl and ssl_loss_module is not None:
                loss_ssl = ssl_loss_module(clean_audio, audio_g)
                loss_gen_all = loss_gen_all + loss_ssl * ssl_loss_weight

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = loss_metric.item() if isinstance(loss_metric, torch.Tensor) else loss_metric
                        mag_error  = F.mse_loss(clean_mag, mag_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g)
                        pha_error  = (ip_error + gd_error + iaf_error).item()
                        com_error  = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        stft_error = F.mse_loss(com_g, com_g_hat).item()
                        ssl_error  = loss_ssl.item() if loss_ssl is not None else 0.0

                    msg = (f'Steps: {steps:d}, Gen Loss: {loss_gen_all:4.3f}, '
                           f'Disc Loss: {loss_disc_all:4.3f}, '
                           f'Metric: {metric_error:4.3f}, Mag: {mag_error:4.3f}, '
                           f'Pha: {pha_error:4.3f}, Com: {com_error:4.3f}, '
                           f'Time: {time_error:4.3f}, STFT: {stft_error:4.3f}')
                    if use_ssl:
                        msg += f', SSL: {ssl_error:4.3f}'
                    msg += f', s/b: {time.time() - start_b:4.3f}'
                    print(msg)

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    cp_path_g = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(cp_path_g,
                                    {'generator': (generator.module if h.num_gpus > 1
                                                   else generator).state_dict()})
                    cp_path_do = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(cp_path_do,
                                    {'discriminator': (discriminator.module if h.num_gpus > 1
                                                       else discriminator).state_dict(),
                                     'optim_g': optim_g.state_dict(),
                                     'optim_d': optim_d.state_dict(),
                                     'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)
                    if use_ssl:
                        sw.add_scalar("Training/SSL Loss", ssl_error, steps)

                # Validation 
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_stft_err_tot = 0
                    val_ssl_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio_v, noisy_audio_v = batch
                            clean_audio_v = clean_audio_v.to(device, non_blocking=True)
                            noisy_audio_v = noisy_audio_v.to(device, non_blocking=True)

                            clean_mag_v, clean_pha_v, clean_com_v = mag_pha_stft(
                                clean_audio_v, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            noisy_mag_v, noisy_pha_v, noisy_com_v = mag_pha_stft(
                                noisy_audio_v, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                            mag_g_v, pha_g_v, com_g_v = generator(noisy_mag_v, noisy_pha_v)
                            audio_g_v = mag_pha_istft(
                                mag_g_v, pha_g_v, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                            mag_g_hat_v, _, com_g_hat_v = mag_pha_stft(
                                audio_g_v, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                            audios_r += torch.split(clean_audio_v, 1, dim=0)
                            audios_g += torch.split(audio_g_v, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag_v, mag_g_v).item()
                            vip, vgd, viaf = phase_losses(clean_pha_v, pha_g_v)
                            val_pha_err_tot += (vip + vgd + viaf).item()
                            val_com_err_tot += F.mse_loss(clean_com_v, com_g_v).item()
                            val_stft_err_tot += F.mse_loss(com_g_v, com_g_hat_v).item()
                            if use_ssl and ssl_loss_module is not None:
                                val_ssl_err_tot += ssl_loss_module(clean_audio_v, audio_g_v).item()

                        n_val = j + 1
                        val_mag_err  = val_mag_err_tot  / n_val
                        val_pha_err  = val_pha_err_tot  / n_val
                        val_com_err  = val_com_err_tot  / n_val
                        val_stft_err = val_stft_err_tot / n_val
                        val_ssl_err  = val_ssl_err_tot  / n_val if use_ssl else 0.0
                        val_pesq_score = pesq_score(audios_r, audios_g, h).item()

                        vmsg = f'Steps: {steps:d}, PESQ Score: {val_pesq_score:4.3f}'
                        if use_ssl:
                            vmsg += f', Val SSL Loss: {val_ssl_err:4.3f}'
                        vmsg += f', s/b: {time.time() - start_b:4.3f}'
                        print(vmsg)

                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar("Validation/Consistency Loss", val_stft_err, steps)
                        if use_ssl:
                            sw.add_scalar("Validation/SSL Loss", val_ssl_err, steps)

                    if epoch >= a.best_checkpoint_start_epoch:
                        if val_pesq_score > best_pesq:
                            best_pesq = val_pesq_score
                            best_cp = "{}/g_best".format(a.checkpoint_path)
                            save_checkpoint(best_cp,
                                            {'generator': (generator.module if h.num_gpus > 1
                                                           else generator).state_dict()})
                            print(f"New best PESQ: {best_pesq:.4f}, saved to {best_cp}")

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    parser = argparse.ArgumentParser(description='Unified MP-SENet Training')

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_clean_wavs_dir', default='VoiceBank+DEMAND/wavs_clean')
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/wavs_noisy')
    parser.add_argument('--input_training_file', default='VoiceBank+DEMAND/training.txt')
    parser.add_argument('--input_validation_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--checkpoint_path', default='cp_unified')
    parser.add_argument('--config', default='config_unified.json')

    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=40, type=int)

    parser.add_argument('--generator_type', default=None, choices=['baseline', 'mingru'],
                        help='Generator architecture (overrides config)')
    parser.add_argument('--discriminator_type', default=None, choices=['single', 'multi_scale'],
                        help='Discriminator architecture (overrides config)')

    parser.add_argument('--use_rope', action='store_true',
                        help='Use RoPE in transformer attention (overrides config)')
    parser.add_argument('--use_ssl_loss', action='store_true',
                        help='Enable SSL loss (overrides config)')
    parser.add_argument('--ssl_model_name', default=None,
                        help='NeMo SSL model name (overrides config)')
    parser.add_argument('--ssl_loss_weight', default=None, type=float,
                        help='SSL loss weight (overrides config)')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if a.generator_type is not None:
        h.generator_type = a.generator_type
    if a.discriminator_type is not None:
        h.discriminator_type = a.discriminator_type
    if a.use_rope:
        h.use_rope = True
    if a.use_ssl_loss:
        h.use_ssl_loss = True
    if a.ssl_model_name is not None:
        h.ssl_model_name = a.ssl_model_name
    if a.ssl_loss_weight is not None:
        h.ssl_loss_weight = a.ssl_loss_weight

    gen_type  = getattr(h, 'generator_type', 'baseline')
    disc_type = getattr(h, 'discriminator_type', 'single')
    ssl_flag  = getattr(h, 'use_ssl_loss', False)
    use_rope  = getattr(h, 'use_rope', False)

    print(f'Initializing Unified Training — generator={gen_type}, '
          f'discriminator={disc_type}, ssl={ssl_flag}, use_rope={use_rope}')

    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()

