import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Dropout


def _trunc_normal_init(tensor, std=0.02, a=-0.06, b=0.06):
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=a, b=b)


class MoEFFN(nn.Module):
    """Expert Choice Mixture-of-Experts FFN (Zhou et al., NeurIPS 2022).

    Each expert selects its top-k tokens (inverted routing), guaranteeing perfect
    load balance by construction. No auxiliary loss, bias, or jitter needed.
    """

    def __init__(
        self,
        d_model=64,
        num_experts=8,
        capacity_factor=2.0,
        expert_ffn_dim=128,
        bidirectional=True,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional, batch_first=True)
        self.gru_out_dim = d_model * 4 if bidirectional else d_model * 2

        self.gate = nn.Linear(self.gru_out_dim, num_experts)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.gru_out_dim, expert_ffn_dim),
                    nn.GELU(),
                    nn.Linear(expert_ffn_dim, d_model),
                )
                for _ in range(num_experts)
            ]
        )

        self.dropout = Dropout(dropout)
        self._last_expert_counts = None
        self._last_token_coverage = None
        self._last_avg_experts_per_token = None
        self._init_weights()

    def _init_weights(self):
        _trunc_normal_init(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for expert in self.experts:
            for module in expert:
                if isinstance(module, nn.Linear):
                    _trunc_normal_init(module.weight)
                    nn.init.zeros_(module.bias)

    @torch.compiler.disable
    def forward(self, x):
        B, T, _ = x.shape
        n = B * T

        # 1. Shared GRU
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)
        gru_out = F.leaky_relu(gru_out)
        gru_out = self.dropout(gru_out)
        flat = gru_out.reshape(n, self.gru_out_dim)

        # 2. Expert Choice gating (float32)
        with torch.amp.autocast("cuda", enabled=False):
            logits = self.gate(flat.float())  # [n, E]
            scores = F.softmax(logits, dim=-1)  # softmax over experts (per-token distribution)
            scores_t = scores.T  # [E, n]
            capacity = int(n * self.capacity_factor / self.num_experts)
            topk_vals, topk_idx = scores_t.topk(capacity, dim=-1)  # [E, cap], [E, cap]

        # 3. Expert dispatch
        output = torch.zeros(n, self.d_model, device=x.device, dtype=torch.float32)
        for e in range(self.num_experts):
            selected = flat[topk_idx[e]]  # [cap, gru_out_dim]
            expert_out = self.experts[e](selected).float()  # [cap, d_model]
            weighted = topk_vals[e].unsqueeze(-1) * expert_out  # [cap, d_model]
            output.scatter_add_(0, topk_idx[e].unsqueeze(-1).expand_as(weighted), weighted)

        # 4. Stats (detached, no grad)
        with torch.no_grad():
            token_expert_count = torch.zeros(n, device=x.device)
            for e in range(self.num_experts):
                token_expert_count.scatter_add_(
                    0, topk_idx[e], torch.ones(capacity, device=x.device)
                )
            self._last_expert_counts = torch.tensor(
                [capacity] * self.num_experts, dtype=torch.float32, device=x.device
            )
            self._last_token_coverage = (token_expert_count > 0).float().mean()
            self._last_avg_experts_per_token = token_expert_count.mean()

        return output.to(x.dtype).reshape(B, T, self.d_model), torch.tensor(0.0, device=x.device)
