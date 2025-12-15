import os
import random
import torch
import torch.utils.data
import librosa

import soundfile as sf
from pathlib import Path
import torch.nn.functional as F


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1] + (1e-10), stft_spec[:, :, :, 0] + (1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(
    mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True
):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )

    return wav


def get_dataset_filelist(a):
    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        training_indexes = [
            x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0
        ]

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        validation_indexes = [
            x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0
        ]

    return training_indexes, validation_indexes


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_indexes,
        clean_wavs_dir,
        noisy_wavs_dir,
        segment_size,
        sampling_rate,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
    ):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(
                os.path.join(self.clean_wavs_dir, filename + ".wav"),
                sr=self.sampling_rate,
            )
            noisy_audio, _ = librosa.load(
                os.path.join(self.noisy_wavs_dir, filename + ".wav"),
                sr=self.sampling_rate,
            )
            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[:length], noisy_audio[:length]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1

        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(
            noisy_audio
        )
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio**2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[
                    :, audio_start : audio_start + self.segment_size
                ]
                noisy_audio = noisy_audio[
                    :, audio_start : audio_start + self.segment_size
                ]
            else:
                clean_audio = torch.nn.functional.pad(
                    clean_audio,
                    (0, self.segment_size - clean_audio.size(1)),
                    "constant",
                )
                noisy_audio = torch.nn.functional.pad(
                    noisy_audio,
                    (0, self.segment_size - noisy_audio.size(1)),
                    "constant",
                )

        return (clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.audio_indexes)


class VCTKDatasetFromList(Dataset):
    """VCTK-DEMAND dataset. File list format: id|path per line."""

    def __init__(
        self,
        file_list,
        clean_dir,
        noisy_dir,
        n_fft=512,
        hop_length=256,
        segment_len=32000,
        return_audio=False,
    ):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_len = segment_len
        self.return_audio = return_audio
        self.window = torch.hann_window(n_fft).pow(0.5)

        with open(file_list) as f:
            self.files = [line.split("|")[0] + ".wav" for line in f if line.strip()]
        print(f"Loaded {len(self.files)} files from {file_list}")

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path):
        wav, _ = sf.read(
            path, dtype="float32", always_2d=True
        )  # wav: (samples, channels)
        wav = torch.from_numpy(wav).mean(-1)  # stereo -> mono, wav: (samples,)
        return wav

    def _to_stft(self, wav):
        stft = torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self.window,
            return_complex=True,
        )
        return torch.stack([stft.real, stft.imag], dim=-1)

    def __getitem__(self, idx):
        filename = self.files[idx]
        clean = self._load_audio(self.clean_dir / filename)
        noisy = self._load_audio(self.noisy_dir / filename)

        # segment_len = None means use full audio
        if self.segment_len is not None:
            start = random.randint(0, max(0, clean.shape[0] - self.segment_len))
            clean = self._segment_at(clean, start)
            noisy = self._segment_at(noisy, start)

        if self.return_audio:
            return noisy, clean
        return self._to_stft(noisy), self._to_stft(clean)

    def _segment_at(self, wav, start):
        segment = wav[start : start + self.segment_len]
        if segment.shape[0] < self.segment_len:
            segment = F.pad(segment, (0, self.segment_len - segment.shape[0]))
        return segment
