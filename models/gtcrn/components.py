"""
GTCRN components: ERB, SFE, TRA, ConvBlock, GTConvBlock, GRNN, DPGRNN, Encoder, Decoder, Mask.
"""

import torch
import numpy as np
import torch.nn as nn
import einops


class ERB(nn.Module):
    """Equivalent Rectangular Bandwidth filter banks for frequency decomposition."""

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
        """Band merge: x: (B,C,T,F) -> (B,C,T,F_erb)"""
        x_low = x[..., : self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1 :])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """Band split: x: (B,C,T,F_erb) -> (B,C,T,F)"""
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
        """x: (B,C,T,F) -> (B, C*kernel_size, T, F)"""
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
        """x: (B,C,T,F) -> (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)
        return x * At


class ConvBlock(nn.Module):
    """Standard 2D convolution block with BatchNorm and activation."""

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
    """Group Temporal Convolution block with TRA attention."""

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
        """Channel shuffle: x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = einops.rearrange(x, "b c g t f -> b (c g) t f")
        return x

    def forward(self, x):
        """x: (B, C, T, F) -> (B, C, T, F)"""
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
    """Grouped RNN - two parallel GRUs processing split input."""

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
        Returns: (output, hidden_state)
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
    """Grouped Dual-path RNN for time-frequency processing."""

    def __init__(self, input_size, width, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        # Intra-frame (frequency) processing - bidirectional
        self.intra_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter-frame (time) processing - unidirectional (causal)
        self.inter_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size, bidirectional=False
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F) -> (B, C, T, F)"""
        ## Intra RNN (along frequency)
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

        ## Inter RNN (along time)
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
    """GTCRN Encoder: ConvBlocks + GTConvBlocks."""

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
        """x: (B, 9, T, 129) -> (B, 16, T, 33), encoder_outputs"""
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    """GTCRN Decoder: GTConvBlocks + ConvBlocks with skip connections."""

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
        """x: (B, 16, T, 33), en_outs -> (B, 2, T, 129)"""
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers - 1 - i])
        return x


class Mask(nn.Module):
    """Complex Ratio Mask application."""

    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        """
        mask: (B, 2, T, F) - real and imaginary mask
        spec: (B, 2, T, F) - real and imaginary spectrogram
        Returns: (B, 2, T, F) - enhanced spectrogram
        """
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s
