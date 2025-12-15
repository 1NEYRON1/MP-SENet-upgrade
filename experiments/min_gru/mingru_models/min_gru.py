"""
MinGRU: https://arxiv.org/abs/2410.01201v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def parallel_scan(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h = a_star + (log_values - a_star).logcumsumexp(dim=1)
    return log_h.exp()


class MinGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
        bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size * self.num_directions
            self.layers.append(nn.Linear(in_size, hidden_size * 2, bias=bias))
            if bidirectional:
                self.layers.append(nn.Linear(in_size, hidden_size * 2, bias=bias))

    def _scan(self, x, linear, reverse=False):
        if reverse:
            x = x.flip(1)

        h, gate = linear(x).chunk(2, dim=-1)
        out = parallel_scan(-F.softplus(gate), -F.softplus(-gate) + log_g(h))

        if reverse:
            out = out.flip(1)
        return out, out[:, -1] if not reverse else out[:, 0]

    def forward(self, x, h0=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        h_list = []
        for i in range(self.num_layers):
            idx = i * self.num_directions
            fwd, h_fwd = self._scan(x, self.layers[idx])
            h_list.append(h_fwd)

            if self.bidirectional:
                bwd, h_bwd = self._scan(x, self.layers[idx + 1], reverse=True)
                h_list.append(h_bwd)
                x = torch.cat([fwd, bwd], dim=-1)
            else:
                x = fwd

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x, torch.stack(h_list)

    def flatten_parameters(self):
        pass
