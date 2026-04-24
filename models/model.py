import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pesq import pesq

from models.transformer import TransformerBlock
from utils import LearnableSigmoid2d

EPS = 1e-8
MIN_GEOMETRY_CANDIDATES = 4
DEFAULT_GEOMETRY_CANDIDATES = 4


FEATURE_MODE_TO_CHANNELS = {
    "mag": 1,
    "baseline": 2,
    "raw": 2,
    "mag_phase": 2,
    "mag_ri": 3,
    "ri": 3,
    "mag_ri_gd_iaf": 5,
    "ri_gd_iaf": 5,
    "phase_geometry": 5,
}


def get_feature_channels(feature_mode: str) -> int:
    mode = (feature_mode or "baseline").lower()
    if mode not in FEATURE_MODE_TO_CHANNELS:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    return FEATURE_MODE_TO_CHANNELS[mode]


def resolve_geometry_candidate_count(num_candidates: int | None) -> int:
    if num_candidates is None:
        return DEFAULT_GEOMETRY_CANDIDATES
    return max(MIN_GEOMETRY_CANDIDATES, int(num_candidates))


def noisy_skip_index(num_candidates: int) -> int:
    return num_candidates - 1


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def anti_wrapping_function(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(wrap_to_pi(x))


def phase_difference(phase: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 1:
        anchor = phase[:, :1, :]
    elif dim == 2:
        anchor = phase[:, :, :1]
    else:
        raise ValueError(f"Unsupported dim for phase difference: {dim}")
    return wrap_to_pi(torch.diff(phase, dim=dim, prepend=anchor))


def build_tfrep_features(mag: torch.Tensor, pha: torch.Tensor, feature_mode: str) -> torch.Tensor:
    mode = (feature_mode or "baseline").lower()

    if mode == "mag":
        return mag.unsqueeze(1)

    if mode in {"baseline", "raw", "mag_phase"}:
        return torch.stack((mag, pha), dim=1)

    cos_pha = torch.cos(pha)
    sin_pha = torch.sin(pha)

    if mode in {"mag_ri", "ri"}:
        return torch.stack((mag, cos_pha, sin_pha), dim=1)

    if mode in {"mag_ri_gd_iaf", "ri_gd_iaf", "phase_geometry"}:
        gd = phase_difference(pha, dim=1) / math.pi
        iaf = phase_difference(pha, dim=2) / math.pi
        return torch.stack((mag, cos_pha, sin_pha, gd, iaf), dim=1)

    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def build_phase_loss_weight(
    mag: torch.Tensor,
    power: float = 1.0,
    floor: float = 0.0,
    detach: bool = True,
) -> torch.Tensor:
    weight = mag.clamp_min(EPS)
    if power != 1.0:
        weight = weight.pow(power)
    denom = weight.amax(dim=(1, 2), keepdim=True).clamp_min(EPS)
    weight = weight / denom
    if floor > 0.0:
        weight = floor + (1.0 - floor) * weight
    if detach:
        weight = weight.detach()
    return weight


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return value.mean()
    return (value * weight).sum() / weight.sum().clamp_min(EPS)


def phase_distance(target: torch.Tensor, pred: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    return _weighted_mean(anti_wrapping_function(target - pred), weight)


def phase_losses(
    phase_r: torch.Tensor,
    phase_g: torch.Tensor,
    weight: torch.Tensor | None = None,
):
    ip_loss = anti_wrapping_function(phase_r - phase_g)
    gd_loss = anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1))
    iaf_loss = anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2))

    gd_weight = None
    iaf_weight = None
    if weight is not None:
        gd_weight = 0.5 * (weight[:, 1:, :] + weight[:, :-1, :])
        iaf_weight = 0.5 * (weight[:, :, 1:] + weight[:, :, :-1])

    return (
        _weighted_mean(ip_loss, weight),
        _weighted_mean(gd_loss, gd_weight),
        _weighted_mean(iaf_loss, iaf_weight),
    )


def geometry_head_losses(
    clean_pha: torch.Tensor,
    noisy_pha: torch.Tensor,
    phase_anchor_token: torch.Tensor,
    gd_pred: torch.Tensor,
    iaf_pred: torch.Tensor,
    weight: torch.Tensor | None = None,
    anchor_mode: str = "residual",
):
    anchor_mode = (anchor_mode or "residual").lower()
    if anchor_mode == "absolute":
        anchor_target = clean_pha
    elif anchor_mode == "residual":
        anchor_target = wrap_to_pi(clean_pha - noisy_pha)
    else:
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")

    clean_gd = phase_difference(clean_pha, dim=1)
    clean_iaf = phase_difference(clean_pha, dim=2)

    return (
        phase_distance(anchor_target, phase_anchor_token, weight),
        phase_distance(clean_gd, gd_pred, weight),
        phase_distance(clean_iaf, iaf_pred, weight),
    )


def geometry_consistency_losses(
    phase_g: torch.Tensor,
    gd_pred: torch.Tensor,
    iaf_pred: torch.Tensor,
    weight: torch.Tensor | None = None,
):
    phase_gd = phase_difference(phase_g, dim=1)
    phase_iaf = phase_difference(phase_g, dim=2)
    return (
        phase_distance(phase_gd, gd_pred, weight),
        phase_distance(phase_iaf, iaf_pred, weight),
    )


def reliability_soft_targets(
    candidate_stack: torch.Tensor,
    clean_pha: torch.Tensor,
    temperature: float = 0.35,
    use_noisy_skip: bool = True,
) -> torch.Tensor:
    errors = anti_wrapping_function(candidate_stack - clean_pha.unsqueeze(1))
    if not use_noisy_skip:
        n_candidates = candidate_stack.size(1)
        skip_idx = noisy_skip_index(n_candidates)
        mask = torch.ones(n_candidates, device=errors.device, dtype=errors.dtype)
        mask[skip_idx] = 0.0
        errors = errors + (1.0 - mask.view(1, n_candidates, 1, 1)) * 1e4
    temperature = max(float(temperature), 1e-4)
    return torch.softmax(-errors / temperature, dim=1).detach()


def reliability_loss(
    weight_logits: torch.Tensor,
    candidate_stack: torch.Tensor,
    clean_pha: torch.Tensor,
    weight: torch.Tensor | None = None,
    temperature: float = 0.35,
    use_noisy_skip: bool = True,
):
    target = reliability_soft_targets(
        candidate_stack,
        clean_pha,
        temperature=temperature,
        use_noisy_skip=use_noisy_skip,
    )
    log_probs = torch.log_softmax(weight_logits, dim=1)
    kl_map = (target * (torch.log(target.clamp_min(EPS)) - log_probs)).sum(dim=1)
    return _weighted_mean(kl_map, weight), target


def reliability_sparse_loss(
    weight_logits: torch.Tensor,
    weight: torch.Tensor | None = None,
    power: float = 1.0,
) -> torch.Tensor:
    n_candidates = weight_logits.size(1)
    skip_idx = noisy_skip_index(n_candidates)
    skip_prob = torch.softmax(weight_logits, dim=1)[:, skip_idx]
    if power != 1.0:
        skip_prob = skip_prob.pow(power)
    return _weighted_mean(skip_prob, weight)


def _integrate_phase_from_steps(anchor_phase: torch.Tensor, step_phase: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 1:
        steps = step_phase.clone()
        steps[:, :1, :] = 0.0
        start = anchor_phase[:, :1, :]
        return wrap_to_pi(start + torch.cumsum(steps, dim=1))
    if dim == 2:
        steps = step_phase.clone()
        steps[:, :, :1] = 0.0
        start = anchor_phase[:, :, :1]
        return wrap_to_pi(start + torch.cumsum(steps, dim=2))
    raise ValueError(f"Unsupported dim for integration: {dim}")


def fuse_phase_candidates(candidate_stack: torch.Tensor, weight_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(weight_logits, dim=1)
    real = (weights * torch.cos(candidate_stack)).sum(dim=1)
    imag = (weights * torch.sin(candidate_stack)).sum(dim=1)
    return torch.atan2(imag, real), weights


def _build_repeated_candidates(primary: torch.Tensor, noisy_pha: torch.Tensor, n_candidates: int) -> torch.Tensor:
    candidates = [primary, primary, primary]
    while len(candidates) < n_candidates - 1:
        candidates.append(primary)
    candidates.append(noisy_pha)
    return torch.stack(candidates, dim=1)


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, height, width = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, height, width))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, height, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super().__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2**i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                nn.Conv2d(
                    h.dense_channel * (i + 1), h.dense_channel, kernel_size, dilation=(dilation, 1)
                ),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
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
        super().__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.dense_block = DenseBlock(h, depth=h.dense_depth)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MultiScaleChannelAttention(nn.Module):
    def __init__(self, channels: int, kernel_sizes: tuple[int, ...] = (3, 5, 9), reduction: int = 4):
        super().__init__()
        kernel_sizes = tuple(int(k) for k in kernel_sizes if int(k) > 0)
        if not kernel_sizes:
            kernel_sizes = (3, 5, 9)
        self.dw_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    groups=channels,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )
        hidden = max(channels // max(1, reduction), 8)
        self.project = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aggregated = x
        for conv in self.dw_convs:
            aggregated = aggregated + conv(x)
        pooled = aggregated.mean(dim=(2, 3), keepdim=True)
        gate = self.project(pooled)
        return x * gate


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=h.dense_depth)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft // 2 + 1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x)
        return x


class CircularProjectionHead(nn.Module):
    def __init__(self, in_channels, out_channel=1):
        super().__init__()
        self.proj_r = nn.Conv2d(in_channels, out_channel, (1, 2))
        self.proj_i = nn.Conv2d(in_channels, out_channel, (1, 2))

    def forward(self, x):
        x_r = self.proj_r(x)
        x_i = self.proj_i(x)
        return torch.atan2(x_i, x_r).permute(0, 3, 2, 1).squeeze(-1)


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.mode = getattr(h, "phase_decoder_mode", "absolute").lower()
        self.use_confidence = self.mode == "residual_blend"
        self.num_candidates = resolve_geometry_candidate_count(
            getattr(h, "phase_geometry_num_candidates", DEFAULT_GEOMETRY_CANDIDATES)
        )

        self.dense_block = DenseBlock(h, depth=h.dense_depth)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_head = CircularProjectionHead(h.dense_channel, out_channel=out_channel)
        self.phase_gate = None
        if self.use_confidence:
            self.phase_gate = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
            nn.init.constant_(self.phase_gate.bias, getattr(h, "phase_confidence_bias", 0.0))

    def forward(self, x, noisy_pha):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        phase_token = self.phase_head(x)

        if self.mode == "absolute":
            denoised_pha = phase_token
            phase_residual = wrap_to_pi(denoised_pha - noisy_pha)
            phase_confidence = torch.ones_like(denoised_pha)
            candidate_stack = _build_repeated_candidates(denoised_pha, noisy_pha, self.num_candidates)
            weight_logits = torch.zeros(
                noisy_pha.size(0),
                self.num_candidates,
                noisy_pha.size(1),
                noisy_pha.size(2),
                device=noisy_pha.device,
                dtype=noisy_pha.dtype,
            )
            return (
                denoised_pha,
                phase_residual,
                phase_confidence,
                phase_token,
                phase_difference(denoised_pha, dim=1),
                phase_difference(denoised_pha, dim=2),
                weight_logits,
                candidate_stack,
            )

        phase_residual = phase_token
        candidate_pha = wrap_to_pi(noisy_pha + phase_residual)

        if self.use_confidence:
            gate = torch.sigmoid(self.phase_gate(x)).permute(0, 3, 2, 1).squeeze(-1)
            cand_r = gate * torch.cos(candidate_pha) + (1.0 - gate) * torch.cos(noisy_pha)
            cand_i = gate * torch.sin(candidate_pha) + (1.0 - gate) * torch.sin(noisy_pha)
            denoised_pha = torch.atan2(cand_i, cand_r)
            phase_confidence = gate
        else:
            denoised_pha = candidate_pha
            phase_confidence = torch.ones_like(denoised_pha)

        candidate_stack = _build_repeated_candidates(candidate_pha, noisy_pha, self.num_candidates)
        weight_logits = torch.zeros(
            noisy_pha.size(0),
            self.num_candidates,
            noisy_pha.size(1),
            noisy_pha.size(2),
            device=noisy_pha.device,
            dtype=noisy_pha.dtype,
        )
        return (
            denoised_pha,
            phase_residual,
            phase_confidence,
            phase_token,
            phase_difference(denoised_pha, dim=1),
            phase_difference(denoised_pha, dim=2),
            weight_logits,
            candidate_stack,
        )


class PhaseGeometryDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.anchor_mode = getattr(h, "phase_geometry_anchor_mode", "residual").lower()
        self.use_noisy_skip = bool(getattr(h, "phase_geometry_use_noisy_skip", True))
        self.noisy_init_bias = float(getattr(h, "phase_geometry_noisy_bias", 1.0))
        self.num_candidates = resolve_geometry_candidate_count(
            getattr(h, "phase_geometry_num_candidates", DEFAULT_GEOMETRY_CANDIDATES)
        )
        self.skip_idx = noisy_skip_index(self.num_candidates)
        self.refine_idx = 3 if self.num_candidates >= 5 else None

        self.dense_block = DenseBlock(h, depth=h.dense_depth)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.anchor_head = CircularProjectionHead(h.dense_channel, out_channel=out_channel)
        self.gd_head = CircularProjectionHead(h.dense_channel, out_channel=out_channel)
        self.iaf_head = CircularProjectionHead(h.dense_channel, out_channel=out_channel)
        self.refine_head = (
            CircularProjectionHead(h.dense_channel, out_channel=out_channel)
            if self.refine_idx is not None
            else None
        )
        self.weight_head = nn.Conv2d(h.dense_channel, self.num_candidates, (1, 2))
        with torch.no_grad():
            self.weight_head.bias.zero_()
            self.weight_head.bias[self.skip_idx] = self.noisy_init_bias

    def forward(self, x, noisy_pha):
        x = self.dense_block(x)
        x = self.phase_conv(x)

        phase_anchor_token = self.anchor_head(x)
        gd_pred = self.gd_head(x)
        iaf_pred = self.iaf_head(x)

        if self.anchor_mode == "absolute":
            phase_anchor = phase_anchor_token
            phase_residual = wrap_to_pi(phase_anchor - noisy_pha)
        elif self.anchor_mode == "residual":
            phase_residual = phase_anchor_token
            phase_anchor = wrap_to_pi(noisy_pha + phase_residual)
        else:
            raise ValueError(f"Unsupported phase_geometry_anchor_mode: {self.anchor_mode}")

        phase_from_gd = _integrate_phase_from_steps(phase_anchor, gd_pred, dim=1)
        phase_from_iaf = _integrate_phase_from_steps(phase_anchor, iaf_pred, dim=2)

        candidates = [phase_anchor, phase_from_gd, phase_from_iaf]
        if self.refine_head is not None:
            refine_residual = self.refine_head(x)
            phase_refined = wrap_to_pi(phase_anchor + refine_residual)
            candidates.append(phase_refined)

        while len(candidates) < self.skip_idx:
            candidates.append(phase_anchor)
        candidates.append(noisy_pha)

        candidate_stack = torch.stack(candidates, dim=1)
        weight_logits = self.weight_head(x).permute(0, 1, 3, 2)

        if not self.use_noisy_skip:
            noisy_mask = torch.zeros(self.num_candidates, device=weight_logits.device, dtype=weight_logits.dtype)
            noisy_mask[self.skip_idx] = -1e4
            weight_logits = weight_logits + noisy_mask.view(1, self.num_candidates, 1, 1)

        denoised_pha, phase_weights = fuse_phase_candidates(candidate_stack, weight_logits)
        phase_confidence = 1.0 - phase_weights[:, self.skip_idx]

        return (
            denoised_pha,
            phase_residual,
            phase_confidence,
            phase_anchor_token,
            gd_pred,
            iaf_pred,
            weight_logits,
            candidate_stack,
        )


class LightweightMemoryBranch(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        kernel_size = max(3, int(kernel_size) | 1)
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        self.gru.flatten_parameters()
        y_gru, _ = self.gru(y)
        y_conv = self.dw_conv(y.transpose(1, 2))
        y_conv = self.pw_conv(F.silu(y_conv)).transpose(1, 2)
        y = self.out(y_gru + y_conv)
        return self.dropout(y)


class HybridSequenceBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.attention = TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.memory = LightweightMemoryBranch(d_model=d_model, kernel_size=kernel_size, dropout=dropout)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        mem_out = x + self.memory(x)
        merged = torch.cat((attn_out, mem_out), dim=-1)
        gate = torch.sigmoid(self.gate(merged))
        mixed = gate * attn_out + (1.0 - gate) * mem_out
        mixed = mixed + 0.1 * torch.tanh(self.fuse(merged))
        return self.norm(mixed)


class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=h.n_heads)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=h.n_heads)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class TSHybridBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        kernel = int(getattr(h, "hybrid_memory_kernel", 5))
        dropout = float(getattr(h, "hybrid_dropout", 0.0))
        self.time_block = HybridSequenceBlock(
            d_model=h.dense_channel,
            n_heads=h.n_heads,
            kernel_size=kernel,
            dropout=dropout,
        )
        self.freq_block = HybridSequenceBlock(
            d_model=h.dense_channel,
            n_heads=h.n_heads,
            kernel_size=kernel,
            dropout=dropout,
        )

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_block(x)
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_block(x)
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_tscblocks = h.num_tsblocks
        self.phase_input_feature_mode = getattr(h, "phase_input_feature_mode", "baseline")
        self.phase_decoder_mode = getattr(h, "phase_decoder_mode", "absolute").lower()
        self.ts_block_type = getattr(h, "ts_block_type", "hybrid").lower()
        self.use_mulca = bool(getattr(h, "use_mulca", True))
        encoder_channels = get_feature_channels(self.phase_input_feature_mode)
        self.dense_encoder = DenseEncoder(h, in_channel=encoder_channels)

        self.mulca = None
        if self.use_mulca:
            kernel_sizes = tuple(getattr(h, "mulca_kernel_sizes", (3, 5, 9)))
            reduction = int(getattr(h, "mulca_reduction", 4))
            self.mulca = MultiScaleChannelAttention(
                h.dense_channel,
                kernel_sizes=kernel_sizes,
                reduction=max(1, reduction),
            )

        self.ts_transformer = nn.ModuleList([])
        if self.ts_block_type == "hybrid":
            block_cls = TSHybridBlock
        elif self.ts_block_type in {"transformer", "baseline"}:
            block_cls = TSTransformerBlock
        else:
            raise ValueError(f"Unsupported ts_block_type: {self.ts_block_type}")

        for _i in range(h.num_tsblocks):
            self.ts_transformer.append(block_cls(h))

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        if self.phase_decoder_mode.startswith("geometry"):
            self.phase_decoder = PhaseGeometryDecoder(h, out_channel=1)
        else:
            self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_amp, noisy_pha, return_aux=False):
        features = build_tfrep_features(noisy_amp, noisy_pha, self.phase_input_feature_mode)
        x = features.permute(0, 1, 3, 2)
        x = self.dense_encoder(x)
        if self.mulca is not None:
            x = self.mulca(x)

        for i in range(self.num_tscblocks):
            x = self.ts_transformer[i](x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        (
            denoised_pha,
            phase_residual,
            phase_confidence,
            phase_anchor_token,
            phase_gd,
            phase_iaf,
            phase_weight_logits,
            phase_candidate_stack,
        ) = self.phase_decoder(x, noisy_pha)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)), dim=-1
        )

        if return_aux:
            aux = (
                phase_residual,
                phase_confidence,
                phase_anchor_token,
                phase_gd,
                phase_iaf,
                phase_weight_logits,
                phase_candidate_stack,
            )
            return denoised_amp, denoised_pha, denoised_com, aux
        return denoised_amp, denoised_pha, denoised_com


def _to_numpy_utt(utt):
    if isinstance(utt, torch.Tensor):
        return utt.squeeze().detach().cpu().numpy()
    return np.asarray(utt).squeeze()


class PESQEvaluator:
    def __init__(
        self,
        sampling_rate: int = 16000,
        max_workers: int = 1,
        fallback_sequential: bool = True,
        start_method: str = "spawn",
    ):
        self.sampling_rate = int(sampling_rate)
        self.max_workers = max(1, int(max_workers))
        self.fallback_sequential = bool(fallback_sequential)
        self.start_method = start_method
        self.executor = None
        if self.max_workers > 1:
            self._reset_executor()

    def _reset_executor(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        ctx = get_context(self.start_method)
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)

    def _score_sequential(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
        scores = [eval_pesq(clean, esti, self.sampling_rate) for clean, esti in pairs]
        return float(np.mean(scores)) if scores else -1.0

    def score(self, utts_r, utts_g) -> float:
        pairs = [
            (_to_numpy_utt(utts_r[i]), _to_numpy_utt(utts_g[i]))
            for i in range(min(len(utts_r), len(utts_g)))
        ]
        if not pairs:
            return -1.0

        if self.executor is None:
            return self._score_sequential(pairs)

        try:
            futures = [
                self.executor.submit(eval_pesq, clean, esti, self.sampling_rate)
                for clean, esti in pairs
            ]
            scores = [future.result() for future in futures]
            return float(np.mean(scores))
        except Exception:
            if not self.fallback_sequential:
                raise
            self._reset_executor()
            return self._score_sequential(pairs)

    def shutdown(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


def pesq_score(utts_r, utts_g, h, evaluator: PESQEvaluator | None = None):
    owns_evaluator = evaluator is None
    if evaluator is None:
        evaluator = PESQEvaluator(
            sampling_rate=getattr(h, "sampling_rate", 16000),
            max_workers=getattr(h, "val_pesq_workers", 1),
            fallback_sequential=getattr(h, "val_pesq_fallback_sequential", True),
        )
    try:
        value = evaluator.score(utts_r, utts_g)
    finally:
        if owns_evaluator:
            evaluator.shutdown()
    return torch.tensor(value, dtype=torch.float32)


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score_value = pesq(sr, clean_utt, esti_utt)
    except Exception:
        pesq_score_value = -1
    return pesq_score_value
