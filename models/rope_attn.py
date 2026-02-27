import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout

def _rotate_half(x: torch.Tensor):
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def _rope_cache(seq_len: int, head_dim: int, device, dtype, theta: float = 10000.0):
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()[None, None, :, :].to(dtype)
    sin = emb.sin()[None, None, :, :].to(dtype)
    return cos, sin

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (x * cos) + (_rotate_half(x) * sin)


class RotarySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True, rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rope_theta = rope_theta

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        B, L, D = x.shape
        H = self.n_heads

        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)

        cos, sin = _rope_cache(L, self.head_dim, x.device, x.dtype, self.rope_theta)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))
            else:
                attn_scores = attn_scores + attn_mask[None, None, :, :]

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(ctx)
        out = self.proj_drop(out)
        return out


class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super(FFN, self).__init__()
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(d_model*2*2, d_model)
        else:
            self.linear = Linear(d_model*2, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

# FFN без GRU
class MLPFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        inner = d_model * 3  # Взято от балды
        self.fc1 = nn.Linear(d_model, inner)
        self.fc2 = nn.Linear(inner, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RopeTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, bidirectional=True, dropout=0, remove_gru=False):
        super(RopeTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = RotarySelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = Dropout(dropout)
        
        self.norm2 = LayerNorm(d_model)
        if remove_gru:
            self.ffn = MLPFFN(d_model, dropout=dropout)
        else:
            self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.norm1(x)
        h = self.attention(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout2(h)

        x = self.norm3(x)
        return x
    
def main():
    x = torch.randn(4, 64, 401, 201)
    b, c, t, f = x.size()
    x = x.permute(0, 3, 2, 1).contiguous().view(b, f*t, c)
    transformer = TransformerBlock(d_model=64, n_heads=4)
    x = transformer(x)
    x =  x.view(b, f, t, c).permute(0, 3, 2, 1)
    print(x.size())


if __name__ == '__main__':
    main()
