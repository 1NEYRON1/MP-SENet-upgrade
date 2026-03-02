# ------------------------------------------------------------------------
# Copyright (c) 2022 Murufeng. All Rights Reserved.
# ------------------------------------------------------------------------
'''
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
# from basicsr.models.archs.arch_util import LayerNorm2d
# from basicsr.models.archs.local_arch import Local_Base
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class LayerNormFunction(torch.autograd.Function):
    @torch._dynamo.disable
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        B, C, T = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        B, C, T = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

#
# class ffn(nn.Module):
#     def __init__(self, in_channels, FFN_Expand=2,  dropout=0.):
#         super(ffn, self).__init__()
#
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * in_channels
#         self.conv4 = nn.Conv1d(in_channels=in_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
#                                groups=1,
#                                bias=True)
#         self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=in_channels, kernel_size=1, padding=0,
#                                stride=1,
#                                groups=1, bias=True)
#
#
#         self.norm2 = LayerNorm1d(in_channels)
#
#
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#
#         self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
#
#     def forward(self, x):
#         inp3 = x
#         x = self.conv4(self.norm2(inp3))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#         x = inp3 + x * self.gamma
#         return x


class LayerNorm1d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LKFCA_Block(nn.Module):
    def __init__(self, in_channels, DW_Expand=2, FFN_Expand=1, drop_out_rate=0., type='sca', kernel_size=31):
        super().__init__()

        dw_channel = in_channels * DW_Expand
        # ConvModule
        # self.lnorm0 = LayerNorm1d(in_channels)
        # self.cmconv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel*2, kernel_size=1)
        # self.cmconv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size,
        #           padding=get_padding(kernel_size), groups=dw_channel)  # DepthWiseConv1d
        # self.cmnorm = nn.InstanceNorm1d(dw_channel, affine=True)
        # self.cmconv3 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        # 注意力

        # self.inter = int(dw_channel // 4)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)



        # Simplified Channel Attention
        # self.type = type
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        # ffn_channel = FFN_Expand * in_channels
        # self.conv4 = nn.Conv1d(in_channels=in_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm1d(in_channels)
        self.norm2 = LayerNorm1d(in_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.lamda = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        # self.LKEFN = FeedForward(in_channels, FFN_Expand, kernel_size=kernel_size, bias=False)
        self.GCGFN = GCGFN(in_channels)


    def forward(self, x):


        inp2 = x
        # x = self.lnorm0(x)
        # x = self.cmconv1(x)
        # x = self.sg(x)
        # x = self.cmconv2(x)
        # x = self.cmnorm(x)
        # x = self.sg(x)
        # x = self.cmconv3(x)
        # inp2 = inp + x * self.lamda


        x = self.norm1(inp2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)


        x = self.dropout1(x)

        inp3 = inp2 + x * self.beta


        x = self.GCGFN(inp3)
        # x = self.sg(x)
        # x = self.conv5(x)
        #
        # x = self.dropout2(x)
        # x = inp3 + x * self.gamma

        # # x = self.norm1(x)#末尾再加一层LN
        # x = self.Rearrange2(x)
        return x


class GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm1d(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.X3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))


    def forward(self, x):
        # shortcut = x.clone()
        shortcut = x
        x = self.norm(x)
        x = self.proj_first(x)
        # a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, kernel_size=3, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv1d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv1d(hidden_features * 2, hidden_features * 2, kernel_size=kernel_size, stride=1, padding=get_padding(kernel_size),
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv1d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GPFCA(nn.Module):

    def __init__(self, in_channels, num_blocks=2):
        super().__init__()

        self.naf_blocks = nn.ModuleList([LKFCA_Block(in_channels) for _ in range(num_blocks)])

        # self.norm1 = LayerNorm1d(in_channels)
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')

    def forward(self, x):
        
        # x = self.norm1(x)

        x = self.Rearrange1(x)


        for block in self.naf_blocks:
            x = block(x)

        x = self.Rearrange2(x)

        return x