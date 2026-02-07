import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from .mamba_block import MyMambaBlock
from .xlstm_block import xLSTMBlock

class TFMixedBlock(nn.Module):
    def __init__(self, cfg):
        super(TFMixedBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']
        
        # Initialize xLSTM blocks
        self.time_block = MyMambaBlock(self.hid_feature, cfg)
        self.freq_block = xLSTMBlock(in_channels=self.hid_feature, cfg=cfg)
        
        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
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