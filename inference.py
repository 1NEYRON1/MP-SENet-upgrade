import argparse
import os

import numpy as np
import torch
from joblib import Parallel, delayed
from rich.progress import track
from torch.amp import autocast
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from cal_metrics.compute_metrics import compute_metrics
from dataset import mag_pha_istft, mag_pha_stft
from models.model import MPNet
from utils import load_config, set_seed

torch.set_float32_matmul_precision("high")

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=True)
    print("Complete.")
    return checkpoint_dict


def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    gen_state = {k.removeprefix("_orig_mod."): v for k, v in state_dict["generator"].items()}
    model.load_state_dict(gen_state)

    model = torch.compile(model, dynamic=True)

    with open(h.input_validation_file) as f:
        test_indexes = [os.path.basename(line.split("|")[1].strip()) for line in f]

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    eval_pairs = []

    with torch.no_grad():
        for index in track(test_indexes, description="Inference"):
            wav_path = os.path.join(a.input_noisy_wavs_dir, index)
            noisy_wav = (
                AudioDecoder(wav_path, sample_rate=h.sampling_rate, num_channels=1)
                .get_all_samples()
                .data.squeeze(0)
                .to(device)
            )
            norm_factor = torch.sqrt(len(noisy_wav) / (torch.sum(noisy_wav**2.0) + 1e-8)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, _ = mag_pha_stft(
                noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            with autocast("cuda", dtype=torch.bfloat16):
                amp_g, pha_g, _, _ = model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(
                amp_g.float(), pha_g.float(), h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index)

            AudioEncoder(samples=audio_g.cpu(), sample_rate=h.sampling_rate).to_file(output_file)

            if a.input_clean_wavs_dir:
                clean_path = os.path.join(a.input_clean_wavs_dir, index)
                if os.path.isfile(clean_path):
                    clean_wav = (
                        AudioDecoder(clean_path, sample_rate=h.sampling_rate, num_channels=1)
                        .get_all_samples()
                        .data.squeeze(0)
                        .numpy()
                    )
                    denoised_np = audio_g.squeeze().cpu().numpy()
                    eval_pairs.append((clean_wav, denoised_np))

    if eval_pairs:
        print(f"\nComputing metrics for {len(eval_pairs)} files...")
        results = Parallel(n_jobs=30)(
            delayed(compute_metrics)(c, d, h.sampling_rate, 0) for c, d in eval_pairs
        )
        results = np.array(results)
        means = results.mean(axis=0)
        metric_names = ["PESQ", "CSIG", "CBAK", "COVL", "SSNR", "STOI"]
        print(f"\n{'Metric':<10} {'Mean':>10}")
        print("-" * 22)
        for name, val in zip(metric_names, means, strict=True):
            print(f"{name:<10} {val:>10.4f}")
        print()


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_noisy_wavs_dir", default=None)
    parser.add_argument("--input_clean_wavs_dir", default=None)
    parser.add_argument("--output_dir", default="../generated_files")
    parser.add_argument("--checkpoint_file", required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.yaml")
    global h
    h = load_config(config_file)

    if a.input_noisy_wavs_dir is None:
        a.input_noisy_wavs_dir = h.input_noisy_wavs_dir
    if a.input_clean_wavs_dir is None:
        a.input_clean_wavs_dir = getattr(h, "input_clean_wavs_dir", None)

    set_seed(h.seed)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(a)


if __name__ == "__main__":
    main()
