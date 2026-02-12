import torch
import numpy as np


def build_pcs_gains_257():
    """Hardcoded PCS gains for 257-bin (n_fft=512) from INTERSPEECH 2022 paper."""
    gains = np.ones(257)
    gains[3:6] = 1.070
    gains[6:9] = 1.182
    gains[9:12] = 1.288
    gains[12:138] = 1.4
    gains[138:166] = 1.323
    gains[166:200] = 1.239
    gains[200:241] = 1.161
    gains[241:257] = 1.077
    return torch.from_numpy(gains).float()


def build_pcs_gains(n_fft, sample_rate=16000):
    """Adapt 257-bin PCS gains to arbitrary n_fft via frequency-domain interpolation."""
    n_bins = n_fft // 2 + 1
    gains_257 = build_pcs_gains_257().numpy()
    f_orig = np.linspace(0, sample_rate / 2, 257)
    f_target = np.linspace(0, sample_rate / 2, n_bins)
    gains = np.interp(f_target, f_orig, gains_257)
    return torch.from_numpy(gains).float()


def apply_pcs(mag, pcs_gains, gamma=1.0):
    """Apply PCS to linear (decompressed) magnitude spectrum.

    Args:
        mag: Tensor[B, F, T] — linear magnitude
        pcs_gains: Tensor[F] — per-bin gains from build_pcs_gains
        gamma: strength (0=off, 1=full PCS)
    Returns:
        Tensor[B, F, T] — PCS-enhanced magnitude
    """
    gains = 1.0 + gamma * (pcs_gains - 1.0)
    return mag * gains.view(1, -1, 1)
