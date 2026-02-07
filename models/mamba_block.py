import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from mambapy.mamba import MambaConfig, ResidualBlock, MambaBlock, RMSNorm

def _init_weights_residual(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "dt_proj.weight", "conv1d.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_residual_block(d_model, cfg, layer_idx=0):
    mcfg = MambaConfig(
        d_model=d_model,
        n_layers=1,
        d_state=cfg['model_cfg']['d_state'],
        expand_factor=cfg['model_cfg']['expand'],
        d_conv=cfg['model_cfg']['d_conv'],
        rms_norm_eps=cfg['model_cfg']['norm_epsilon'],
        inner_layernorms=cfg.get('inner_layernorms', False),
    )
    block = ResidualBlock(mcfg)
    block.layer_idx = layer_idx
    return block


class MyMambaBlock(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.fwd = create_residual_block(dim, cfg, layer_idx=0)
        self.bwd = create_residual_block(dim, cfg, layer_idx=0)

        self.apply(partial(_init_weights_residual, n_layer=1))

    def forward(self, x):
        # x: (B, L, D)
        y_fwd = self.fwd(x)  # (B, L, D)
        x_rev = torch.flip(x, dims=[1])
        y_rev = self.bwd(x_rev)
        y_rev = torch.flip(y_rev, dims=[1])
        y = torch.cat([y_fwd, y_rev], dim=-1)
        return y


class TFMambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hid = cfg['model_cfg']['hid_feature']
        self.hid = hid

        self.time_block = MyMambaBlock(hid, cfg)
        self.freq_block = MyMambaBlock(hid, cfg)

        # (batch * other_dim, seq_len, hid)
        self.tlinear = nn.ConvTranspose1d(hid * 2, hid, kernel_size=1, stride=1)
        self.flinear = nn.ConvTranspose1d(hid * 2, hid, kernel_size=1, stride=1)

    def forward(self, x):
        # x: (B, C, T, F)
        B, C, T, F = x.shape

        # (B, F, T, C) → (B*F, T, C)
        xt = x.permute(0, 3, 2, 1).contiguous().view(B * F, T, C)
        # → (B*F, T, 2C)
        yt = self.time_block(xt)
        # (B*F, 2C, T)
        yt_t = self.tlinear(yt.permute(0, 2, 1))  # (B*F, C, T)
        yt = yt_t.permute(0, 2, 1)  # → (B*F, T, C)
        yt = yt + xt
        # (B, T, F, C) → (B*T, F, C)
        yt = yt.view(B, F, T, C).permute(0, 2, 1, 3).contiguous().view(B * T, F, C)

        yf = self.freq_block(yt)  # (B*T, F, 2C)
        yf_t = self.flinear(yf.permute(0, 2, 1))  # (B*T, C, F)
        yf = yf_t.permute(0, 2, 1)  # (B*T, F, C)
        yf = yf + yt
        # (B, T, F, C)
        yf = yf.view(B, T, F, C).permute(0, 3, 1, 2)  # (B, C, T, F)
        return yf
