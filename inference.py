from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import yaml
import numpy as np
import torch
from torch.amp import autocast
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from types import SimpleNamespace
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
from rich.progress import track


class PCS400(torch.nn.Module):
    """Perceptual Contrast Stretching for n_fft=400 (201 frequency bins).
    Reshapes spectral profile (boosting mid-frequencies) while preserving
    per-frame energy to avoid amplitude explosion on raw STFT magnitudes."""
    def __init__(self):
        super().__init__()
        w = np.ones(201)
        w[0:3] = 1.0
        w[3:5] = 1.070175439
        w[5:8] = 1.182456140
        w[8:10] = 1.287719298
        w[10:110] = 1.4
        w[110:130] = 1.322807018
        w[130:160] = 1.238596491
        w[160:190] = 1.161403509
        w[190:] = 1.077192982
        self.register_buffer('weights', torch.from_numpy(w).float().unsqueeze(0).unsqueeze(-1))

    def forward(self, mag):  # [B, F, T] decompressed magnitude
        mag_pcs = torch.exp(self.weights * torch.log(mag + 1e-9))
        energy_in = mag.norm(dim=1, keepdim=True)
        energy_out = mag_pcs.norm(dim=1, keepdim=True)
        return mag_pcs * (energy_in / (energy_out + 1e-9))

torch.set_float32_matmul_precision('high')

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=True)
    print("Complete.")
    return checkpoint_dict

def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    gen_state = {k.removeprefix('_orig_mod.'): v for k, v in state_dict['generator'].items()}
    model.load_state_dict(gen_state)

    model = torch.compile(model, dynamic=True)

    test_indexes = [f for f in os.listdir(a.input_noisy_wavs_dir) if f.endswith('.wav')]

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    if a.use_pcs:
        pcs = PCS400().to(device)

    hann_window = torch.hann_window(h.win_size, device=device)

    with torch.no_grad():
        for index in track(test_indexes):
            wav_path = os.path.join(a.input_noisy_wavs_dir, index)
            noisy_wav = AudioDecoder(wav_path, sample_rate=h.sampling_rate, num_channels=1).get_all_samples().data.squeeze(0).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / (torch.sum(noisy_wav ** 2.0) + 1e-8)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            with autocast('cuda', dtype=torch.bfloat16):
                amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            amp_g = amp_g.float()
            pha_g = pha_g.float()
            mag = torch.pow(amp_g, 1.0 / h.compress_factor)
            if a.use_pcs:
                mag = pcs(mag)
            com = torch.complex(mag * torch.cos(pha_g), mag * torch.sin(pha_g))
            audio_g = torch.istft(com, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=hann_window, center=True)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index)

            AudioEncoder(samples=audio_g.cpu(), sample_rate=h.sampling_rate).to_file(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='/work/VoiceBank+DEMAND/testset_noisy')
    parser.add_argument('--output_dir', default='../generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--use_pcs', action='store_true')
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
