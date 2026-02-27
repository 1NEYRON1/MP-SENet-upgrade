"""

MinGRU: https://arxiv.org/abs/2410.01201v1

Based on: https://github.com/Fritschek/MinGRU-x-Mamba

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinGRUCell(nn.Module):
    """Single MinGRU cell with separate z and h projections."""
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_z.weight)
        nn.init.xavier_uniform_(self.linear_h.weight)
        if self.linear_z.bias is not None:
            nn.init.zeros_(self.linear_z.bias)
            nn.init.zeros_(self.linear_h.bias)

    def forward(self, x, h_0=None):
        if self.training:
            return self._forward_parallel(x, h_0)
        else:
            return self._forward_sequential(x, h_0)

    def _forward_parallel(self, x, h_0=None):
        """Parallel scan for training."""
        batch_size = x.size(0)
        
        if h_0 is None:
            h_0 = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
        elif h_0.dim() == 2:
            h_0 = h_0.unsqueeze(1)

        # Compute gates
        k = self.linear_z(x)
        log_z = -F.softplus(-k)      # log(sigmoid(k))
        log_coeffs = -F.softplus(k)  # log(1 - sigmoid(k))

        # Compute candidate hidden state
        log_h_0 = self._log_g(h_0)
        log_tilde_h = self._log_g(self.linear_h(x))

        # Concatenate h_0 with values
        log_values = torch.cat([log_h_0, log_z + log_tilde_h], dim=1)

        # Parallel scan
        h = self._parallel_scan_log(log_coeffs, log_values)
        return h

    def _forward_sequential(self, x, h_0=None):
        """Sequential forward for inference."""
        batch_size, seq_len, _ = x.size()
        
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        elif h_0.dim() == 3:
            h_0 = h_0.squeeze(1)

        # Precompute all gates
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = self._g(self.linear_h(x))
        h_prev = self._g(h_0)

        h_all = []
        for t in range(seq_len):
            h_prev = (1 - z[:, t]) * h_prev + z[:, t] * h_tilde[:, t]
            h_all.append(h_prev.unsqueeze(1))

        return torch.cat(h_all, dim=1)

    @staticmethod
    def _g(x):
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    @staticmethod
    def _log_g(x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    @staticmethod
    def _parallel_scan_log(log_coeffs, log_values):
        # log_coeffs: (batch, seq_len, hidden)
        # log_values: (batch, seq_len + 1, hidden)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]


class MinGRU(nn.Module):
    """Multi-layer bidirectional MinGRU with nn.GRU-compatible interface."""

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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size * self.num_directions
            self.layers.append(MinGRUCell(in_size, hidden_size, bias=bias))
            if bidirectional:
                self.layers.append(MinGRUCell(in_size, hidden_size, bias=bias))

    def forward(self, x, h0=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        h_list = []
        for i in range(self.num_layers):
            idx = i * self.num_directions

            # Forward direction
            fwd = self.layers[idx](x)
            h_list.append(fwd[:, -1])

            if self.bidirectional:
                # Backward direction
                x_rev = x.flip(1)
                bwd = self.layers[idx + 1](x_rev)
                bwd = bwd.flip(1)
                h_list.append(bwd[:, 0])
                x = torch.cat([fwd, bwd], dim=-1)
            else:
                x = fwd

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x, torch.stack(h_list)

    def flatten_parameters(self):
        pass

