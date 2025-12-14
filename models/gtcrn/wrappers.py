"""
Waveform wrappers for STFT-based models.
"""

import torch
import torch.nn as nn


class WaveformWrapper(nn.Module):
    """
    Wrapper that converts waveform audio in order to use STFT-based speech enhancement models.
    Takes raw audio waveform, converts to STFT, runs model, converts back to waveform.
    """

    def __init__(self, model, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = model

    def forward(self, x):
        """
        x: (B, samples) - raw waveform
        Returns: (B, samples) - enhanced waveform
        """
        window = torch.hann_window(self.n_fft, device=x.device).pow(0.5)

        spec_complex = torch.stft(
            x, self.n_fft, self.hop_length, self.n_fft, window, return_complex=True
        )
        spec = torch.stack(
            [spec_complex.real, spec_complex.imag], dim=-1
        )  # (B, F, T, 2)
        spec_enh = self.model(spec)  # (B, F, T, 2)

        spec_enh_complex = torch.complex(spec_enh[..., 0], spec_enh[..., 1])
        enhanced_waveform = torch.istft(
            spec_enh_complex, self.n_fft, self.hop_length, self.n_fft, window
        )

        return enhanced_waveform
