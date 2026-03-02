import csv
import os
import random
import torch
import torch.utils.data
from torchcodec.decoders import AudioDecoder

_hann_window_cache = {}

def _get_hann_window(win_size, device):
    key = (win_size, device)
    if key not in _hann_window_cache:
        _hann_window_cache[key] = torch.hann_window(win_size, device=device)
    return _hann_window_cache[key]

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = _get_hann_window(win_size, y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(torch.clamp(stft_spec.pow(2).sum(-1), min=1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1], stft_spec[:, :, :, 0])
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = _get_hann_window(win_size, com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return training_indexes, validation_indexes


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dir, segment_size,
                sampling_rate, split=True, shuffle=True, n_cache_reuse=1, device=None):
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
            clean_audio = AudioDecoder(os.path.join(self.clean_wavs_dir, filename + '.wav'), sample_rate=self.sampling_rate, num_channels=1).get_all_samples().data.squeeze(0)
            noisy_audio = AudioDecoder(os.path.join(self.noisy_wavs_dir, filename + '.wav'), sample_rate=self.sampling_rate, num_channels=1).get_all_samples().data.squeeze(0)
            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[: length], noisy_audio[: length]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1

        clean_audio, noisy_audio = clean_audio.float(), noisy_audio.float()
        norm_factor = torch.sqrt(len(noisy_audio) / (torch.sum(noisy_audio ** 2.0) + 1e-8))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start+self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start+self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

        return (clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.audio_indexes)

def _read_index_csv(csv_path):
    """
    Reads a CSV with columns:
      gt_path, noisy_path, language, noise_category, speaker
    Returns: rows.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            if not row.get("gt_path") or not row.get("noisy_path"):
                continue
            rows.append(row)
    return rows


def get_dataset_entries(a):
    train_path = a.input_training_file
    val_path = a.input_validation_file

    if train_path.lower().endswith(".csv") and val_path.lower().endswith(".csv"):
        training_entries = _read_index_csv(train_path)
        validation_entries = _read_index_csv(val_path)
        return training_entries, validation_entries

    return get_dataset_filelist(a)


def _infer_dataset_root(clean_dir, noisy_dir):
    if clean_dir is not None:
        cd = os.path.normpath(clean_dir)
        if os.path.basename(cd) == "gt":
            return os.path.dirname(cd)
    if noisy_dir is not None:
        nd = os.path.normpath(noisy_dir)
        if os.path.basename(nd) == "noisy":
            return os.path.dirname(nd)

    candidates = [p for p in [clean_dir, noisy_dir] if p]
    if len(candidates) >= 2:
        try:
            return os.path.commonpath(candidates)
        except Exception:
            pass
    return clean_dir or noisy_dir or "."


class DatasetCSV(torch.utils.data.Dataset):
    def __init__(
        self,
        entries,
        dataset_root,
        segment_size,
        sampling_rate,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
    ):
        self.entries = list(entries)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.entries)

        self.dataset_root = dataset_root
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split

        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.cached_meta = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0

        self.device = device

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]

        if self._cache_ref_count == 0:
            gt_rel = row["gt_path"]
            noisy_rel = row["noisy_path"]

            gt_path = os.path.join(self.dataset_root, gt_rel)
            noisy_path = os.path.join(self.dataset_root, noisy_rel)

            clean_audio = (
                AudioDecoder(gt_path, sample_rate=self.sampling_rate, num_channels=1)
                .get_all_samples()
                .data.squeeze(0)
            )
            noisy_audio = (
                AudioDecoder(noisy_path, sample_rate=self.sampling_rate, num_channels=1)
                .get_all_samples()
                .data.squeeze(0)
            )

            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[:length], noisy_audio[:length]

            # Cache
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self.cached_meta = row
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            row = self.cached_meta
            self._cache_ref_count -= 1

        clean_audio, noisy_audio = clean_audio.float(), noisy_audio.float()

        # Same normalization logic
        norm_factor = torch.sqrt(len(noisy_audio) / (torch.sum(noisy_audio ** 2.0) + 1e-8))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(
                    clean_audio, (0, self.segment_size - clean_audio.size(1)), "constant"
                )
                noisy_audio = torch.nn.functional.pad(
                    noisy_audio, (0, self.segment_size - noisy_audio.size(1)), "constant"
                )

        # Keep all CSV columns available
        meta = dict(row)
        meta.setdefault("language", "")
        meta.setdefault("noise_category", "")
        meta.setdefault("speaker", "")

        return (clean_audio.squeeze(), noisy_audio.squeeze(), meta)
