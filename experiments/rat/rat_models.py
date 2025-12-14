"""
GTCRN_RAT: GTCRN with RAT block replacing DPGRNN.

Based on:
- GTCRN: https://github.com/Xiaobin-Rong/gtcrn
- RAT: https://github.com/CLAIRE-Labo/RAT
"""

import sys

sys.path.append("../..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gtcrn import (
    ERB,
    SFE,
    GRNN,
    Encoder,
    Decoder,
    Mask,
)

# ======================= RAT Block =======================


class RATBlock(nn.Module):
    """
    Recurrent Attention Transformer block for audio speech enhancement.
    Based on RAT paper: https://arxiv.org/pdf/2507.04416

    Architecture:
    - Intra-chunk: Bidirectional GRU for local frequency modeling
    - Inter-chunk: Multi-head self-attention between chunk representations
    - Gated fusion: Learnable gate combining intra and inter outputs

    Key difference from original RAT:
    - Original RAT uses associative scan for intra-chunk
    - We use bidirectional GRU which is more suitable for audio (offline processing)
    """

    def __init__(
        self,
        input_size,
        width,
        hidden_size,
        chunk_size=8,
        num_heads=4,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size

        # Input normalization (inspired by RAT's input_norm)
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-8)

        # Intra-chunk: Bidirectional GRU for local frequency modeling
        self.intra_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter-chunk: Self-attention for global temporal context
        # Using scaled dot-product attention like original RAT
        self.inter_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Learnable gate for combining intra and inter outputs (key RAT feature)
        # g * intra + (1-g) * inter
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_bias = nn.Parameter(torch.zeros(hidden_size))

        # Initialize gate bias to prefer local information initially
        nn.init.constant_(self.gate_bias, 1.0)

    def forward(self, x):
        """x: (B, C, T, Freq)"""
        B, C, T, Freq = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, T, Freq, C)
        shortcut = x

        # Input normalization
        x_norm = self.input_norm(x)

        # ===== Intra-chunk: RNN along frequency =====
        intra_x = x_norm.reshape(B * T, Freq, C)  # (B*T, Freq, C)
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(B, T, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = shortcut + intra_x  # Residual connection

        # ===== Inter-chunk: Attention along time =====
        # Reshape for attention: (B*Freq, T, C)
        inter_x = intra_out.permute(0, 2, 1, 3)  # (B, Freq, T, C)
        inter_x = inter_x.reshape(B * Freq, T, C)

        # Pad T to be divisible by chunk_size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            inter_x = F.pad(inter_x, (0, 0, 0, pad_len))

        T_padded = inter_x.shape[1]
        num_chunks = T_padded // self.chunk_size

        # Reshape to chunks: (B*Freq, num_chunks, chunk_size, C)
        chunked = inter_x.view(B * Freq, num_chunks, self.chunk_size, C)

        # Get chunk representations - use last position like original RAT
        # (RAT uses the last hidden state of each chunk as the summary)
        chunk_repr = chunked[:, :, -1, :]  # (B*Freq, num_chunks, C)

        # Self-attention between chunks with causal mask
        # Each chunk can only attend to previous chunks (like RAT's block_causal_mask)
        causal_mask = torch.triu(
            torch.ones(num_chunks, num_chunks, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.inter_attention(
            chunk_repr,
            chunk_repr,
            chunk_repr,
            attn_mask=causal_mask,
            is_causal=False,  # We provide explicit mask
        )

        # Broadcast attention output back to all positions in each chunk
        attn_out = attn_out.unsqueeze(2).expand(-1, -1, self.chunk_size, -1)
        attn_out = attn_out.reshape(B * Freq, T_padded, C)

        # Remove padding
        attn_out = attn_out[:, :T, :]

        attn_out = self.inter_fc(attn_out)
        attn_out = attn_out.reshape(B, Freq, T, self.hidden_size)
        attn_out = attn_out.permute(0, 2, 1, 3)  # (B, T, Freq, C)
        attn_out = self.inter_ln(attn_out)

        # ===== Gated fusion (RAT-style) =====
        # Compute gate from intra output
        gate = torch.sigmoid(self.gate_proj(intra_out) + self.gate_bias)
        out = gate * intra_out + (1 - gate) * attn_out

        return out.permute(0, 3, 1, 2)  # (B, C, T, F)


# ======================= Model =======================


class GTCRN_RAT(nn.Module):
    """GTCRN with RAT blocks replacing DPGRNN."""

    def __init__(self, chunk_size=8, num_heads=4):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        self.rat1 = RATBlock(16, 33, 16, chunk_size=chunk_size, num_heads=num_heads)
        self.rat2 = RATBlock(16, 33, 16, chunk_size=chunk_size, num_heads=num_heads)
        self.decoder = Decoder()
        self.mask = Mask()

    def forward(self, spec):
        """spec: (B, F, T, 2)"""
        spec_ref = spec
        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)

        feat = self.erb.bm(feat)
        feat = self.sfe(feat)
        feat, en_outs = self.encoder(feat)
        feat = self.rat1(feat)
        feat = self.rat2(feat)
        m_feat = self.decoder(feat, en_outs)
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))
        return spec_enh.permute(0, 3, 2, 1)
