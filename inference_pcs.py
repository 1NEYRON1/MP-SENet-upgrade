from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import yaml
import torch
from torch.amp import autocast
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from types import SimpleNamespace
from collections import defaultdict
from dataset import mag_pha_stft, mag_pha_istft, _get_hann_window
from models.model import MPNet
from pcs import build_pcs_gains, apply_pcs
from cal_metrics.compute_metrics import compute_metrics
from rich.progress import track

torch.set_float32_matmul_precision('high')

METRIC_NAMES = ['PESQ', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'STOI']

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=True)
    print("Complete.")
    return checkpoint_dict


def reconstruct_pcs(mag_decompressed, pha, pcs_gains, gamma, norm_factor):
    """Apply PCS to decompressed magnitude, then iSTFT + denormalize."""
    mag_pcs = apply_pcs(mag_decompressed, pcs_gains, gamma)
    com = torch.complex(mag_pcs * torch.cos(pha), mag_pcs * torch.sin(pha))
    hann_window = _get_hann_window(h.win_size, com.device)
    audio = torch.istft(com, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=hann_window, center=True)
    return audio / norm_factor


def eval_metrics(clean_path, output_file):
    """Compute 6 metrics from saved WAV files."""
    pesq_val, csig, cbak, covl, ssnr, stoi_val = compute_metrics(clean_path, output_file, h.sampling_rate, path=1)
    return {'PESQ': pesq_val, 'CSIG': csig, 'CBAK': cbak, 'COVL': covl, 'SSNR': ssnr, 'STOI': stoi_val}


def print_comparison_table(baseline_avg, pcs_avg, gamma):
    """Print Baseline vs PCS comparison table."""
    sep = '-' * 49
    print(f"\n{'Metric':<11}| {'Baseline':>10} | {'PCS γ=' + str(gamma):>10} | {'Delta':>10}")
    print(sep)
    for name in METRIC_NAMES:
        b = baseline_avg[name]
        p = pcs_avg[name]
        d = p - b
        sign = '+' if d >= 0 else '-'
        print(f"{name:<11}| {b:>10.4f} | {p:>10.4f} | {sign} {abs(d):>8.4f}")
    print(sep)


def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    gen_state = {k.removeprefix('_orig_mod.'): v for k, v in state_dict['generator'].items()}
    model.load_state_dict(gen_state)

    model = torch.compile(model, dynamic=True)

    pcs_gains = build_pcs_gains(h.n_fft, h.sampling_rate).to(device)
    gammas = sorted(set(a.gamma))

    test_indexes = [f for f in os.listdir(a.input_noisy_wavs_dir) if f.endswith('.wav')]

    # Output dirs: baseline + one per gamma
    has_clean = a.input_clean_wavs_dir is not None
    baseline_dir = f"{a.output_dir}_baseline"
    os.makedirs(baseline_dir, exist_ok=True)
    output_dirs = {}
    for g in gammas:
        d = f"{a.output_dir}_gamma{g}"
        output_dirs[g] = d
        os.makedirs(d, exist_ok=True)

    baseline_accum = defaultdict(float)
    metrics_accum = {g: defaultdict(float) for g in gammas}
    n_files = 0

    model.eval()

    with torch.no_grad():
        for index in track(test_indexes):
            noisy_path = os.path.join(a.input_noisy_wavs_dir, index)
            noisy_wav = AudioDecoder(noisy_path, sample_rate=h.sampling_rate, num_channels=1).get_all_samples().data.squeeze(0).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / (torch.sum(noisy_wav ** 2.0) + 1e-8)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            with autocast('cuda', dtype=torch.bfloat16):
                amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)

            amp_g = amp_g.float()
            pha_g = pha_g.float()

            # Baseline: original mag_pha_istft (no PCS)
            audio_baseline = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_baseline = audio_baseline / norm_factor
            baseline_file = os.path.join(baseline_dir, index)
            AudioEncoder(samples=audio_baseline.cpu(), sample_rate=h.sampling_rate).to_file(baseline_file)

            # PCS variants: decompress -> PCS -> iSTFT
            mag_decompressed = torch.pow(amp_g, 1.0 / h.compress_factor)

            clean_path = os.path.join(a.input_clean_wavs_dir, index) if has_clean else None
            has_this_clean = has_clean and clean_path and os.path.isfile(clean_path)

            if has_this_clean:
                for k, v in eval_metrics(clean_path, baseline_file).items():
                    baseline_accum[k] += v

            for g in gammas:
                audio_pcs = reconstruct_pcs(mag_decompressed, pha_g, pcs_gains, g, norm_factor)
                output_file = os.path.join(output_dirs[g], index)
                AudioEncoder(samples=audio_pcs.cpu(), sample_rate=h.sampling_rate).to_file(output_file)

                if has_this_clean:
                    for k, v in eval_metrics(clean_path, output_file).items():
                        metrics_accum[g][k] += v

            if has_this_clean:
                n_files += 1

    if not has_clean or n_files == 0:
        return

    baseline_avg = {k: v / n_files for k, v in baseline_accum.items()}
    for g in gammas:
        pcs_avg = {k: v / n_files for k, v in metrics_accum[g].items()}
        print_comparison_table(baseline_avg, pcs_avg, g)


def main():
    print('Initializing Inference Process with PCS..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='/work/VoiceBank+DEMAND/testset_noisy')
    parser.add_argument('--input_clean_wavs_dir', default=None, help='Path to clean reference WAVs for metrics')
    parser.add_argument('--output_dir', default='../generated_files_pcs')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--gamma', type=float, nargs='+', default=[1.0], help='PCS gamma values to compare against baseline')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.yaml')
    with open(config_file) as f:
        global h
        h = SimpleNamespace(**yaml.safe_load(f))

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
