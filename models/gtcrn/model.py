"""
GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources

Full model implementation.
"""

import torch
import torch.nn as nn

from .components import ERB, SFE, DPGRNN, Encoder, Decoder, Mask


class GTCRN(nn.Module):
    """
    GTCRN: Group-Temporal Convolutional Recurrent Network for speech enhancement.

    Architecture:
    - ERB filter banks for frequency decomposition
    - SFE for subband feature extraction
    - Encoder with ConvBlocks and GTConvBlocks
    - Dual-path grouped RNN (DPGRNN) for time-frequency modeling
    - Decoder with skip connections
    - Complex ratio mask for speech estimation

    Input: Spectrogram (B, F, T, 2) where last dim is [real, imag]
    Output: Enhanced spectrogram (B, F, T, 2)
    """

    def __init__(self):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()

        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)

        self.decoder = Decoder()

        self.mask = Mask()

    def forward(self, spec):
        """
        spec: (B, F, T, 2) - complex spectrogram [real, imag]
        Returns: (B, F, T, 2) - enhanced spectrogram
        """
        spec_ref = spec  # (B,F,T,2)

        # Extract features
        spec_real = spec[..., 0].permute(0, 2, 1)  # (B, T, F)
        spec_imag = spec[..., 1].permute(0, 2, 1)  # (B, T, F)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        # ERB band merge and subband feature extraction
        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)  # (B,9,T,129)

        # Encoder
        feat, en_outs = self.encoder(feat)  # (B,16,T,33)

        # Dual-path RNN
        feat = self.dpgrnn1(feat)  # (B,16,T,33)
        feat = self.dpgrnn2(feat)  # (B,16,T,33)

        # Decoder
        m_feat = self.decoder(feat, en_outs)  # (B,2,T,129)

        # ERB band split to get full-band mask
        m = self.erb.bs(m_feat)  # (B,2,T,257)

        # Apply complex mask
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        return spec_enh
