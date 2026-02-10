import torch.nn as nn
import torch.nn.functional as F

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(1, 2),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose(1, 2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.layernorm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        b, s, d = x.shape
        x = self.layernorm(x)
        qkv = self.qkv(x).reshape(b, s, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(b, s, d)
        x = self.out_proj(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x
