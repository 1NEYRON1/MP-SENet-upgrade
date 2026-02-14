import torch
import torch.nn as nn
import numpy as np
from models.transformer import TransformerBlock
from utils import LearnableSigmoid2d
from pesq import pesq
from joblib import Parallel, delayed

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out
    

class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DenseBlock(h, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super(TSTransformerBlock, self).__init__()
        self.h = h
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class ChannelProjection(nn.Module):
    """Linear projection over channel dim — equivalent to Conv2d(c,c,1) but avoids DDP stride warnings."""
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.PReLU(channels)

    def forward(self, x):  # [B, C, T, F]
        x = self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(self.norm(x))


class TSCrossAttentionBlock(nn.Module):
    def __init__(self, h):
        super(TSCrossAttentionBlock, self).__init__()
        d = h.dense_channel
        self.mag_time_cross = TransformerBlock(d, n_heads=4, is_cross_attention=True)
        self.mag_freq_cross = TransformerBlock(d, n_heads=4, is_cross_attention=True)
        self.pha_time_cross = TransformerBlock(d, n_heads=4, is_cross_attention=True)
        self.pha_freq_cross = TransformerBlock(d, n_heads=4, is_cross_attention=True)

    def forward(self, x_mag, x_pha):  # [B, C, T, F]
        b, c, t, f = x_mag.size()

        # Time-axis cross-attention: [B*F, T, C]
        mag_t = x_mag.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        pha_t = x_pha.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        mag_t = self.mag_time_cross(mag_t, kv=pha_t) + mag_t
        pha_t = self.pha_time_cross(pha_t, kv=mag_t) + pha_t

        # Freq-axis cross-attention: [B*T, F, C]
        mag_f = mag_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        pha_f = pha_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        mag_f = self.mag_freq_cross(mag_f, kv=pha_f) + mag_f
        pha_f = self.pha_freq_cross(pha_f, kv=mag_f) + pha_f

        # Back to [B, C, T, F]
        return (mag_f.view(b, t, f, c).permute(0, 3, 1, 2),
                pha_f.view(b, t, f, c).permute(0, 3, 1, 2))


class MPNet(nn.Module):
    def __init__(self, h):
        super(MPNet, self).__init__()
        self.h = h
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        # Shared TSTransformerBlocks
        self.shared_ts = nn.ModuleList([
            TSTransformerBlock(h) for _ in range(h.num_shared_tsblocks)])

        # Branch projections (Linear instead of Conv2d 1x1 to avoid DDP stride warning)
        self.mag_proj = ChannelProjection(h.dense_channel)
        self.pha_proj = ChannelProjection(h.dense_channel)

        # Branch-specific TSTransformerBlocks
        self.mag_ts = nn.ModuleList([
            TSTransformerBlock(h) for _ in range(h.num_branch_tsblocks)])
        self.pha_ts = nn.ModuleList([
            TSTransformerBlock(h) for _ in range(h.num_branch_tsblocks)])

        # Cross-attention
        self.cross_attn = TSCrossAttentionBlock(h)

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_amp, noisy_pha):  # [B, F, T]
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        encoder_out = x  # Save for skip connection

        for block in self.shared_ts:
            x = block(x)

        x_mag = self.mag_proj(x)
        x_pha = self.pha_proj(x)

        for block in self.mag_ts:
            x_mag = block(x_mag)
        for block in self.pha_ts:
            x_pha = block(x_pha)

        x_mag, x_pha = self.cross_attn(x_mag, x_pha)

        # Skip connections: encoder features → decoder input
        denoised_amp = noisy_amp * self.mask_decoder(x_mag + encoder_out)
        denoised_pha = self.phase_decoder(x_pha + encoder_out)

        denoised_com = torch.stack((denoised_amp*torch.cos(denoised_pha),
                                    denoised_amp*torch.sin(denoised_pha)), dim=-1)

        return denoised_amp, denoised_pha, denoised_com


def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):

    return torch.abs(torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        pesq_score = -1

    return pesq_score