import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Dropout


def _trunc_normal_init(tensor, std=0.02, a=-0.06, b=0.06):
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=a, b=b)


class MoEFFN(nn.Module):
    """Token Choice Top-2 MoE FFN with Switch Transformer balance loss + DeepSeek-V3 bias.

    Each token selects its top-2 experts via softmax gating. No token dropping,
    no capacity limits.

    Dual load balancing:
    1. Switch Transformer balance loss (gradient-based, pushes P toward uniform)
    2. DeepSeek-V3 adaptive bias (non-gradient, directly adjusts routing logits)

    Router z-loss stabilizes logit magnitudes.

    References:
    - Switch Transformers (arXiv:2101.03961) — balance loss
    - DeepSeek-V3 (arXiv:2412.19437) — auxiliary-loss-free bias balancing
    - ST-MoE — router z-loss
    """

    def __init__(
        self,
        d_model=64,
        num_experts=4,
        top_k=2,
        expert_ffn_dim=256,
        balance_loss_weight=0.01,
        z_loss_weight=0.001,
        bias_update_speed=0.001,
        noise_ctx_dim=0,
        bidirectional=True,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = balance_loss_weight
        self.z_loss_weight = z_loss_weight
        self.bias_update_speed = bias_update_speed
        self.noise_ctx_dim = noise_ctx_dim

        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional, batch_first=True)
        self.gru_out_dim = d_model * 4 if bidirectional else d_model * 2

        gate_in_dim = self.gru_out_dim + noise_ctx_dim
        self.gate = nn.Linear(gate_in_dim, num_experts)

        # DeepSeek-V3 adaptive bias: non-gradient, updated by load imbalance.
        # persistent=False: not saved in state_dict (resets to zero on load, reconverges quickly)
        self.register_buffer("expert_bias", torch.zeros(num_experts), persistent=False)

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
        self._last_gate_entropy = None
        self._last_expert_load = None
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
    def forward(self, x, noise_ctx=None):
        B, T, _ = x.shape
        n = B * T
        E = self.num_experts

        # 1. Shared GRU (runs in caller's autocast dtype, e.g. bfloat16)
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)
        gru_out = F.leaky_relu(gru_out)
        gru_out = self.dropout(gru_out)
        flat = gru_out.reshape(n, self.gru_out_dim)

        # 2. Gating (float32 for softmax numerical stability)
        with torch.amp.autocast("cuda", enabled=False):
            # Noise-conditioned gate input
            if noise_ctx is not None and self.noise_ctx_dim > 0:
                ctx = noise_ctx.unsqueeze(1).expand(B, T, -1).reshape(n, -1)
                gate_input = torch.cat([flat.float(), ctx.float()], dim=-1)
            else:
                gate_input = flat.float()

            logits = self.gate(gate_input)  # [n, E]

            # DeepSeek-V3 bias: shift logits to balance routing (no gradient)
            logits = logits + self.expert_bias

            # Router z-loss (ST-MoE): stabilize logit magnitudes
            z_loss = self.z_loss_weight * torch.logsumexp(logits, dim=-1).square().mean()

            scores = F.softmax(logits, dim=-1)  # [n, E]

            # Token Choice: each token picks top-k experts
            top_scores, top_idx = scores.topk(self.top_k, dim=-1)  # [n, k]
            weights = top_scores / top_scores.sum(dim=-1, keepdim=True)  # renorm to 1.0

            # Switch Transformer balance loss:
            # f_i = fraction of tokens routed to expert i (no grad)
            # P_i = mean softmax probability for expert i (has grad)
            expert_mask = torch.zeros(n, E, device=x.device, dtype=torch.float32)
            expert_mask.scatter_(1, top_idx, 1.0)
            f = expert_mask.detach().mean(dim=0)  # [E]
            P = scores.mean(dim=0)  # [E]
            balance_loss = self.balance_loss_weight * E * (f * P).sum()

            aux_loss = balance_loss + z_loss

        # 3. Expert dispatch (bf16 compute, f32 accumulation)
        output = torch.zeros(n, self.d_model, device=x.device, dtype=torch.float32)
        for k in range(self.top_k):
            for e in range(E):
                mask = top_idx[:, k] == e
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=True)[0]
                expert_out = self.experts[e](flat[idx])  # bf16
                output[idx] += weights[idx, k : k + 1] * expert_out.float()

        # 4. DeepSeek-V3 bias update (non-gradient, train only)
        with torch.no_grad():
            expert_load = f  # reuse from balance loss (same computation, already detached)
            target_load = self.top_k / E  # 0.5 for top-2/4
            if self.training:
                # DDP: average load across ranks so bias stays identical on all GPUs
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(expert_load, op=torch.distributed.ReduceOp.AVG)
                self.expert_bias += self.bias_update_speed * (target_load - expert_load)

        # 5. Stats (detached, no grad)
        with torch.no_grad():
            gate_entropy = -(scores * scores.clamp(min=1e-8).log()).sum(dim=-1).mean()
            self._last_gate_entropy = gate_entropy.detach()
            self._last_expert_load = expert_load  # [E], already detached via f

        return output.to(x.dtype).reshape(B, T, self.d_model), aux_loss
