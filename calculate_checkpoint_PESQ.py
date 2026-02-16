import os
import json
import torch
import librosa
import numpy as np
from tqdm.auto import tqdm
from models.discriminator import cal_pesq
from env import AttrDict
from models.model import MPNet
from dataset import mag_pha_stft, mag_pha_istft


CHECKPOINT_PATH = "/workspace-SR008.fs2/iatrofimenko/se/MP-SENet/cp_model_multi_ssl_16_seg/g_best"
CONFIG_PATH = "/workspace-SR008.fs2/iatrofimenko/se/MP-SENet/cp_model_multi_ssl_16_seg/config.json"

TESTSET_CLEAN_DIR = "VoiceBank+DEMAND/wavs_clean"
TESTSET_NOISY_DIR = "VoiceBank+DEMAND/wavs_noisy"
TEST_LIST_FILE = "VoiceBank+DEMAND/test.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")
with open(CONFIG_PATH) as f:
    h = AttrDict(json.load(f))

model = MPNet(h, num_tsblocks=4).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["generator"])
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

with open(TEST_LIST_FILE) as f:
    test_files = [line.strip().split("|")[0] for line in f if line.strip()]
print(f"Test files: {len(test_files)}")

def denoise(noisy_wav):
    noisy = torch.FloatTensor(noisy_wav).to(device)
    norm = torch.sqrt(len(noisy) / torch.sum(noisy ** 2.0))
    noisy = (noisy * norm).unsqueeze(0)
    
    noisy_amp, noisy_pha, _ = mag_pha_stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    
    with torch.no_grad():
        amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
    
    audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    return (audio_g / norm).squeeze().cpu().numpy()

pesq_enhanced, pesq_noisy = [], []

for filename in tqdm(test_files):
    clean_path = os.path.join(TESTSET_CLEAN_DIR, filename + ".wav")
    noisy_path = os.path.join(TESTSET_NOISY_DIR, filename + ".wav")
    
    clean_wav, _ = librosa.load(clean_path, sr=h.sampling_rate)
    noisy_wav, _ = librosa.load(noisy_path, sr=h.sampling_rate)
    
    denoised_wav = denoise(noisy_wav)
    
    pesq_enhanced.append(cal_pesq(clean_wav, denoised_wav, h.sampling_rate))
    pesq_noisy.append(cal_pesq(clean_wav, noisy_wav, h.sampling_rate))

print(f"Evaluated: {len(pesq_enhanced)} files")
print(f"\nPESQ Noisy: {np.mean(pesq_noisy):.4f}")
print(f"PESQ Enhanced: {np.mean(pesq_enhanced):.4f}")


