import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout

from models.rat import RAT

class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0, chunk_size=1):
        super().__init__()
        self.rat = RAT(
            d_model=d_model,
            num_head=8,
            chunk_size=chunk_size,
            ngroups=2,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.rat(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0, chunk_size=1):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)
        
        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

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