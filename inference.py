from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import yaml
import torch
from torch.amp import autocast
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from types import SimpleNamespace
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
from rich.progress import track

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

    with torch.no_grad():
        for index in track(test_indexes):
            wav_path = os.path.join(a.input_noisy_wavs_dir, index)
            noisy_wav = AudioDecoder(wav_path, sample_rate=h.sampling_rate, num_channels=1).get_all_samples().data.squeeze(0).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / (torch.sum(noisy_wav ** 2.0) + 1e-8)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            with autocast('cuda', dtype=torch.bfloat16):
                amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(amp_g.float(), pha_g.float(), h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index)

            AudioEncoder(samples=audio_g.cpu(), sample_rate=h.sampling_rate).to_file(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='/work/VoiceBank+DEMAND/testset_noisy')
    parser.add_argument('--output_dir', default='../generated_files')
    parser.add_argument('--checkpoint_file', required=True)
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
