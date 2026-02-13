"""
MPNet-MinGRU: MPNet model with MinGRU instead of GRU.

Uses inheritance to replace GRU with MinGRU in FFN blocks.
"""

import torch.nn as nn

from .min_gru import MinGRU
from models.mpnet.model import MPNet
from models.mpnet.transformer import FFN, TransformerBlock
from models.mpnet.model import TSTransformerBlock


class MinFFN(FFN):
    def __init__(self, d_model, bidirectional=True, dropout=0.0):
        super().__init__(d_model, bidirectional, dropout)
        self.gru = MinGRU(
            d_model,
            d_model * 2,
            1,
            batch_first=False,
            bidirectional=bidirectional,
        )


class MinTransformerBlock(TransformerBlock):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0.0):
        super().__init__(d_model, n_heads, bidirectional, dropout)
        self.ffn = MinFFN(d_model, bidirectional=bidirectional, dropout=dropout)


class MinTSTransformerBlock(TSTransformerBlock):
    def __init__(self, h):
        super().__init__(h)
        self.time_transformer = MinTransformerBlock(d_model=h.dense_channel, n_heads=4)
        self.freq_transformer = MinTransformerBlock(d_model=h.dense_channel, n_heads=4)


class MinMPNet(MPNet):
    def __init__(self, h, num_tsblocks=4):
        super().__init__(h, num_tsblocks)
        self.TSTransformer = nn.ModuleList(
            [MinTSTransformerBlock(h) for _ in range(num_tsblocks)]
        )
