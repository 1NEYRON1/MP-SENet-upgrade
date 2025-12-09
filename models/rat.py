import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention as fla
from torch._higher_order_ops.associative_scan import associative_scan
from torch.nn import RMSNorm
import math

# apply parallel scan in the second dimension
class AScan(torch.autograd.Function): # (b c l p)
    @staticmethod
    def scan_op(i, j):
        g_i, x_i = i
        g_j, x_j = j
        return g_j * g_i, g_j * x_i + x_j

    @torch.compile
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, g, x):
        _, x_scan = associative_scan(AScan.scan_op, (g, x), dim=2)
        ctx.save_for_backward(g, x_scan)
        return x_scan

    @torch.compile
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda") # (b c l p)
    def backward(ctx, grad):
        g, x_scan = ctx.saved_tensors
        g = F.pad(g, (0, 0, -1, 1))
        _, x_grad = associative_scan(AScan.scan_op, (g, grad), dim=2, reverse=True)
        g_grad = torch.zeros_like(x_scan)
        g_grad[:, :, 1:].add_(x_scan[:, :, :-1] * x_grad[:, :, 1:])
        return g_grad, x_grad

ascan = AScan.apply


"""
An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.
"""


def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(
                Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1]))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)
        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(
                Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2])))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0 : L : 2**k]
            Xa = X[:, :, 0 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @torch.compiler.disable()
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        # cloning is required because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in)  # (B, npo2(L), D, N)
            X = pad_npo2(X_in)  # (B, npo2(L), D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, npo2(L), N)
        X = X.transpose(2, 1)  # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]

    @torch.compiler.disable()
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is required because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in)  # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  # (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(
            A_in[:, :, 1:], (0, 0, 0, 1)
        )  # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]

pscan = PScan.apply


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin): # cos and sin has been taken out based on the position ids
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MergeLastToken(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inter_out, intra_out, inter_lse, intra_lse):
        max_lse = torch.max(inter_lse, intra_lse)
        inter_lse_exp = torch.exp(inter_lse - max_lse)
        intra_lse_exp = torch.exp(intra_lse - max_lse)
        intra_adjust = (intra_lse_exp / (intra_lse_exp + inter_lse_exp)).to(intra_out.dtype).unsqueeze(-1)
        inter_adjust = (inter_lse_exp / (intra_lse_exp + inter_lse_exp)).to(inter_out.dtype).unsqueeze(-1)

        out = inter_out * inter_adjust + intra_adjust * intra_out
        ctx.save_for_backward(inter_out, intra_out, inter_adjust, intra_adjust)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inter_out, intra_out, inter_adjust, intra_adjust = ctx.saved_tensors
        grad_inter_lse = (grad_output * (inter_out - intra_out) * intra_adjust * inter_adjust).sum(-1)
        return grad_output * inter_adjust, grad_output * intra_adjust, grad_inter_lse, -grad_inter_lse

merge_last_token = MergeLastToken.apply

class RAT(nn.Module):
    def __init__(
        self,
        d_model,
        num_head=8,
        bias=False,
        chunk_size=1, # (T=321, F=101)
        ngroups=2,
        init=None,
        **kwargs,
    ):
        super().__init__()

        self.layer_id = kwargs.get("layer_id", 0)
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0

        self.d_head = d_model // num_head
        self.ngroups = ngroups
        self.softmax_scale = self.d_head ** -0.5

        # in_proj: [z, x, g, q, k]
        self.in_proj = nn.Linear(
            d_model,
            3 * d_model + 2 * ngroups * self.d_head,
            bias=bias,
            device=kwargs.get("device", None),
            dtype=kwargs.get("dtype", None),
        )

        self.gate_bias = nn.Parameter(
            torch.empty(d_model, device=kwargs.get("device", None), dtype=kwargs.get("dtype", None))
        )

        self.input_norm = RMSNorm(d_model, eps=1e-6)

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
            device=kwargs.get("device", None),
            dtype=kwargs.get("dtype", None),
        )

        self.init = init

    def init_weights(self, init_config=None):
        if self.chunk_size != 1:
            torch.nn.init.uniform_(self.gate_bias, 1, self.chunk_size - 1)
            self.gate_bias.data = torch.log(self.gate_bias.data)


    def apply_rope(self, q, k, **kwargs):
        rotary_pos_emb = kwargs.get(f"interrope_{self.chunk_size}", None)
        if rotary_pos_emb is None:
            return q, k

        cos, sin = rotary_pos_emb
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        return q_rope.to(k.dtype), k_rope.to(k.dtype)

    def forward(self, hidden_states, cache=None, **kwargs):
        bs, seq_len, _ = hidden_states.shape

        assert seq_len % self.chunk_size == 0
        num_chunk = seq_len // self.chunk_size

        shortcut = hidden_states

        h = self.input_norm(hidden_states)
        z, x, g, q, k = self.prepare_input(h)

        k = k.repeat(1, 1, self.num_head // self.ngroups)
        q = q.repeat(1, 1, self.num_head // self.ngroups)
        q = q.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)

        x, g, k = (
            x.reshape(bs, num_chunk, self.chunk_size, self.d_model),
            g.reshape(bs, num_chunk, self.chunk_size, self.d_model),
            k.reshape(bs, num_chunk, self.chunk_size, self.d_model),
        )
        g = g.repeat(1, 1, 1, 2)

        intra_xk = ascan(g, ((1 - g) * torch.cat([x, k], dim=-1))).to(torch.bfloat16)

        intra_x = intra_xk[..., : self.d_model].reshape(bs, seq_len, self.d_model)
        intra_k = intra_xk[..., self.d_model:].reshape(bs, seq_len, self.d_model)

        intra_x = intra_x.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)
        intra_k = intra_k.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)

        q, intra_k = self.apply_rope(q, intra_k, **kwargs)
        if q.dtype != intra_k.dtype:
            q = q.to(intra_k.dtype)

        chunk_intra_k = intra_k[..., :: self.chunk_size, :]
        chunk_intra_x = intra_x[..., :: self.chunk_size, :]

        block_mask = fla.create_block_mask(
            self.block_causal_mask,
            1,
            1,
            q.shape[2],
            num_chunk,
            device=q.device,
        )
        inter_out, inter_lse = fla.flex_attention(
            q, chunk_intra_k, chunk_intra_x, scale=self.softmax_scale,
            block_mask=block_mask, return_lse=True
        )

        intra_lse = (torch.einsum("balp,balp->bal", q, intra_k) * self.softmax_scale).to(torch.float32)

        out = merge_last_token(inter_out, intra_x, inter_lse, intra_lse)

        out = out.transpose(1, 2).reshape(bs, seq_len, self.d_model)
        return self.prepare_output(out, z) + shortcut

    def block_causal_mask(self, b, h, q_idx, kv_idx):
        return q_idx // self.chunk_size > kv_idx

    def prepare_input(self, hidden_states):
        inp = self.in_proj(hidden_states)
        z, x, g, q, k = torch.split(
            inp,
            [self.d_model, self.d_model, self.d_model,
             self.ngroups * self.d_head, self.ngroups * self.d_head],
            dim=-1,
        )
        return torch.sigmoid(z), x, torch.sigmoid(g + self.gate_bias), q, k

    def prepare_output(self, out, z):
        return self.out_proj(z * out)
