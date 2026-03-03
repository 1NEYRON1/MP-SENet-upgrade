import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Dropout, LayerNorm, Linear, MultiheadAttention


class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super().__init__()
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.linear = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear = Linear(d_model * 2, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0, moe_config=None):
        super().__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(d_model)

        self.use_moe = moe_config is not None
        if self.use_moe:
            from models.moe import MoEFFN

            self.ffn = MoEFFN(
                d_model=d_model,
                num_experts=moe_config.get("num_experts", 4),
                top_k=moe_config.get("top_k", 2),
                expert_ffn_dim=moe_config.get("expert_ffn_dim", 256),
                balance_loss_weight=moe_config.get("balance_loss_weight", 0.01),
                z_loss_weight=moe_config.get("z_loss_weight", 0.001),
                bias_update_speed=moe_config.get("bias_update_speed", 0.001),
                noise_ctx_dim=moe_config.get("noise_ctx_dim", 0),
                bidirectional=bidirectional,
                dropout=dropout,
            )
        else:
            self.ffn = FFN(d_model, bidirectional=bidirectional)

        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, noise_ctx=None, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(
            xt, xt, xt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        if self.use_moe:
            xt, aux_loss = self.ffn(xt, noise_ctx=noise_ctx)
        else:
            xt = self.ffn(xt)
            aux_loss = 0.0
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x, aux_loss
