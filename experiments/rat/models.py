"""
GTCRN_RAT: GTCRN with RAT block replacing DPGRNN.

Based on:
- GTCRN: https://github.com/Xiaobin-Rong/gtcrn
- RAT: https://github.com/CLAIRE-Labo/RAT
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import einops


# ======================= Original GTCRN Components (from: https://github.com/Xiaobin-Rong/gtcrn/blob/main/gtcrn.py) =======================


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(
            erb_subband_1, erb_subband_2, nfft, high_lim, fs
        )
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(
        self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000
    ):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0] : bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i] : bins[i + 1]] = (
                np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12
            ) / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1] : bins[i + 2]] = (
                bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12
            ) / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2] : bins[-1] + 1] = (
            1 - erb_filters[-2, bins[-2] : bins[-1] + 1]
        )
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., : self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1 :])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., : self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1 :])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""

    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size - 1) // 2),
        )

    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(
            x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3]
        )
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)
        return x * At


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        use_deconv=False,
        is_last=False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        use_deconv=False,
    ):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)

        self.tra = TRA(in_channels // 2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = einops.rearrange(x, "b c g t f -> b (c g) t f")
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x = self.shuffle(h1, x2)
        return x


class GRNN(nn.Module):
    """Grouped RNN"""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h is None:
            size = self.num_layers * 2 if self.bidirectional else self.num_layers
            h = torch.zeros(size, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""

    def __init__(self, input_size, width, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size, bidirectional=False
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(
            x.shape[0] * x.shape[1], x.shape[2], x.shape[3]
        )  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)  # (B*T,F,C)
        intra_x = intra_x.reshape(
            x.shape[0], -1, self.width, self.hidden_size
        )  # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)  # (B*F,T,C)
        inter_x = inter_x.reshape(
            x.shape[0], self.width, -1, self.hidden_size
        )  # (B,F,T,C)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B,T,F,C)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)

        return dual_out



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList(
            [
                ConvBlock(
                    3 * 3,
                    16,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    use_deconv=False,
                    is_last=False,
                ),
                ConvBlock(
                    16,
                    16,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    groups=2,
                    use_deconv=False,
                    is_last=False,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    use_deconv=False,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(2, 1),
                    use_deconv=False,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(5, 1),
                    use_deconv=False,
                ),
            ]
        )

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList(
            [
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(2 * 5, 1),
                    dilation=(5, 1),
                    use_deconv=True,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(2 * 2, 1),
                    dilation=(2, 1),
                    use_deconv=True,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(2 * 1, 1),
                    dilation=(1, 1),
                    use_deconv=True,
                ),
                ConvBlock(
                    16,
                    16,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    groups=2,
                    use_deconv=True,
                    is_last=False,
                ),
                ConvBlock(
                    16,
                    2,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    use_deconv=True,
                    is_last=True,
                ),
            ]
        )

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers - 1 - i])
        return x


class Mask(nn.Module):
    """Complex Ratio Mask"""

    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


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

# ======================= Models =======================


class GTCRN(nn.Module):
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
        spec: (B, F, T, 2)
        """
        spec_ref = spec  # (B,F,T,2)

        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)  # (B,9,T,129)

        feat, en_outs = self.encoder(feat)

        feat = self.dpgrnn1(feat)  # (B,16,T,33)
        feat = self.dpgrnn2(feat)  # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)

        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        return spec_enh

class GTCRN_GRAT(nn.Module):
    """GTCRN with DPGRAT (GRNN replaced by GRAT inside dual-path structure)"""

    def __init__(self, num_heads=4, chunk_size=8):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        # DPGRAT instead of DPGRNN
        self.dpgrat1 = DPGRAT(16, 33, 16, num_heads=num_heads, chunk_size=chunk_size)
        self.dpgrat2 = DPGRAT(16, 33, 16, num_heads=num_heads, chunk_size=chunk_size)
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
        feat = self.dpgrat1(feat)
        feat = self.dpgrat2(feat)
        m_feat = self.decoder(feat, en_outs)
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))
        return spec_enh.permute(0, 3, 2, 1)


class GTCRN_RAT(nn.Module):
    """GTCRN with RAT blocks replacing DPGRNN"""

    def __init__(self, chunk_size=8, num_heads=4):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        # Replace DPGRNN with RATBlock
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


# ======================= GRAT: Grouped RAT (replaces GRNN) =======================


class GRAT(nn.Module):
    """
    Grouped RAT - replaces GRNN with attention-based sequence modeling.
    Same interface as GRNN: (B, seq_len, input_size) -> (B, seq_len, output_size)
    
    Output size matches GRNN behavior:
    - bidirectional=False: output_size = hidden_size
    - bidirectional=True: output_size = hidden_size * 2 (like bidirectional GRU)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
        num_heads=4,
        chunk_size=8,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Each group processes half of input
        # For bidirectional, GRU output is hidden_size * 2, so we match that
        if bidirectional:
            # bidirectional: each group outputs hidden_size, total = 2 * hidden_size
            group_out = hidden_size
        else:
            # unidirectional: each group outputs hidden_size // 2, total = hidden_size
            group_out = hidden_size // 2

        # Adjust num_heads if needed
        num_heads = min(num_heads, group_out)
        while group_out % num_heads != 0:
            num_heads -= 1

        self.proj1 = nn.Linear(input_size // 2, group_out)
        self.proj2 = nn.Linear(input_size // 2, group_out)

        self.attn1 = nn.MultiheadAttention(
            embed_dim=group_out,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=group_out,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, x, h=None):
        """
        x: (B, seq_len, input_size)
        Returns: (output, hidden_state) - hidden_state is dummy for compatibility
        """
        B, T, _ = x.shape

        # Split into 2 groups (like GRNN)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)

        # Project
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        if self.bidirectional:
            # Bidirectional: no causal mask, full attention
            y1, _ = self.attn1(x1, x1, x1)
            y2, _ = self.attn2(x2, x2, x2)
        else:
            # Unidirectional: causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            y1, _ = self.attn1(x1, x1, x1, attn_mask=causal_mask, is_causal=False)
            y2, _ = self.attn2(x2, x2, x2, attn_mask=causal_mask, is_causal=False)

        y = torch.cat([y1, y2], dim=-1)

        # Return dummy hidden state for GRNN interface compatibility
        dummy_h = torch.zeros(1, B, self.hidden_size, device=x.device)
        return y, dummy_h


class DPGRAT(nn.Module):
    """
    Dual-path GRAT - same structure as DPGRNN but uses GRAT instead of GRNN.
    """

    def __init__(
        self, input_size, width, hidden_size, num_heads=4, chunk_size=8, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        # Intra: bidirectional attention along frequency
        self.intra_rat = GRAT(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            num_heads=num_heads,
            chunk_size=chunk_size,
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter: causal attention along time
        self.inter_rat = GRAT(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=False,
            num_heads=num_heads,
            chunk_size=chunk_size,
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        # Intra RAT (along frequency)
        x = x.permute(0, 2, 3, 1)  # (B, T, F, C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T, F, C)
        intra_x = self.intra_rat(intra_x)[0]  # (B*T, F, C)
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)  # (B, T, F, C)
        intra_x = self.intra_ln(intra_x)
        intra_out = x + intra_x

        # Inter RAT (along time)
        x = intra_out.permute(0, 2, 1, 3)  # (B, F, T, C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*F, T, C)
        inter_x = self.inter_rat(inter_x)[0]  # (B*F, T, C)
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)  # (B, F, T, C)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B, T, F, C)
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x

        return inter_out.permute(0, 3, 1, 2)  # (B, C, T, F)


class GTCRN_GRAT(nn.Module):
    """GTCRN with DPGRAT (GRNN replaced by GRAT inside dual-path structure)"""

    def __init__(self, num_heads=4, chunk_size=8):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        # DPGRAT instead of DPGRNN
        self.dpgrat1 = DPGRAT(16, 33, 16, num_heads=num_heads, chunk_size=chunk_size)
        self.dpgrat2 = DPGRAT(16, 33, 16, num_heads=num_heads, chunk_size=chunk_size)
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
        feat = self.dpgrat1(feat)
        feat = self.dpgrat2(feat)
        m_feat = self.decoder(feat, en_outs)
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))
        return spec_enh.permute(0, 3, 2, 1)


# ======================= Waveform Wrappers =======================


class WaveformWrapper(nn.Module):
    """Waveform wrapper for any STFT-based model"""

    def __init__(self, model, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = model

    def forward(self, x):
        """x: (B, samples) -> (B, samples)"""
        window = torch.hann_window(self.n_fft, device=x.device).pow(0.5)
        spec_complex = torch.stft(
            x, self.n_fft, self.hop_length, self.n_fft, window, return_complex=True
        )
        spec = torch.stack([spec_complex.real, spec_complex.imag], dim=-1)
        spec_enh = self.model(spec)
        spec_enh_complex = torch.complex(spec_enh[..., 0], spec_enh[..., 1])
        return torch.istft(
            spec_enh_complex, self.n_fft, self.hop_length, self.n_fft, window
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

