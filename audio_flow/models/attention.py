import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from audio_flow.models.rope import RoPE


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    r"""Modulate input with scale and shift.

    Args:
        x: (b, t, d)
        shift: (b, t, d)
        scale: (b, t, d)

    Outputs:
        out: (b, t, d)
    """
    return x * (1 + scale) + shift


class Block(nn.Module):
    r"""Self attention block.

    Ref: 
        [1] https://github.com/facebookresearch/DiT/blob/main/models.py
        [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """
    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        self.self_attn = SelfAttention(dim, num_heads)
        self.cross_attn = CrossAttention(dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim)
        )

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cx: Tensor,
        rope: RoPE,
    ) -> torch.Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2)
            mask: None | (1, 1, l, l)
            emb: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        e = self.modulation(e).chunk(6, dim=2)

        # Self-attention
        #  e[0], e[1] h [8, 750, 768]
        h = modulate(self.norm1(x), e[0], e[1])
        x = x + e[2] * self.self_attn(h, rope)

        # Cross-attention
        # [8, 750, 768])
        if cx is not None:
            x = x + self.cross_attn(self.norm2(x), cx, rope)

        # FFN
        h = modulate(self.norm3(x), e[3], e[4])
        x = x + e[5] * self.ffn(h)

        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        rope: nn.Module,
    ) -> Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d)
            rope: (l, head_dim/2, 2)
            mask: (1, 1)

        Outputs:
            x: (b, l, d)
        """

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)

        # Apply RoPE
        q = rope(q)  # (b, l, n, h)
        k = rope(k)  # (b, l, n, h)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=rearrange(q, 'b l n h -> b n l h'), 
            key=rearrange(k, 'b l n h -> b n l h'), 
            value=rearrange(v, 'b l n h -> b n l h'), 
            attn_mask=None, 
            dropout_p=0.0
        )  # (b, n, l, h)

        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self, 
        x: Tensor, 
        cx: Tensor,
        rope: RoPE,
    ) -> Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: heads_num
        h: head_dim
        k: rope_dim

        Args:
            x: (b, l, d)
            rope: (l, h/2, 2)
            pos: (l, k)
            mask: (1, 1, )

        Outputs:
            x: (b, l, d)
        """
        B, L, D = x.shape

        # Calculate query, key, values
        q = self.q_linear(x)  # shapes: (b, lx, d)
        k, v = self.kv_linear(cx).chunk(chunks=2, dim=2)  # shapes: (b, lc, d)

        q = rearrange(self.norm_q(q), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        
        # Apply RoPE
        q = rope(q)  # (b, l, n, h)
        k = rope(k)  # (b, l, n, h)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=rearrange(q, 'b l n h -> b n l h'), 
            key=rearrange(k, 'b l n h -> b n l h'), 
            value=rearrange(v, 'b l n h -> b n l h'), 
            attn_mask=None, 
            dropout_p=0.0
        )  # (b, n, l, h)

        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        
        return x


# class MLP(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()

#         # The hyper-parameters follow https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
#         hidden_dim = 4 * config.n_embd

#         self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
#         self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
#         self.proj = nn.Linear(hidden_dim, dim, bias=False)

#     def forward(self, x: Tensor) -> Tensor:
#         r"""Causal self attention.

#         Args:
#             x: (b, l, d)
           
#         Outputs:
#             x: (b, l, d)
#         """

#         x = F.silu(self.fc1(x)) * self.fc2(x)
#         x = self.proj(x)
#         return x