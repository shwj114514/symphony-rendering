from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from audio_flow.models.attention import Block
from audio_flow.models.embedders import LabelEmbedder, MlpEmbedder, TimestepEmbedder
from audio_flow.models.rope import RoPE
from audio_flow.models.pad import pad1d


class Transformer1D(nn.Module):
    def __init__(
        self,
        in_dim=16,
        patch_size=1,
        dim=384,
        mlp_ratio=4.0,
        num_layers=12,
        num_heads=12,
        rope_len=8192,
        **kwargs
    ):
        
        super().__init__()

        self.patch_size = patch_size

        self.patch_x = nn.Conv1d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.unpatch_x = nn.ConvTranspose1d(dim, in_dim, kernel_size=patch_size, stride=patch_size)

        # Time embedder
        self.t_embedder = TimestepEmbedder(dim=dim, freq_size=256, scale=100.)

        self.blocks = nn.ModuleList(Block(dim, num_heads) for _ in range(num_layers))

        head_dim = dim // num_heads
        self.rope = RoPE(head_dim, max_len=rope_len)

    def forward(
        self, 
        t: Tensor, 
        x: Tensor, 
        emb_dict: dict
    ) -> Tensor:
        """Model

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, d, t)
            cond_dict: dict

        Outputs:
            output: (b, d, t)
        """

        assert all(key in ["c", "ct", "cx"] for key in emb_dict.keys()), "Invalid key in emb_dict!"

        c = emb_dict.get("c", None)
        ct = emb_dict.get("ct", None)
        cx = emb_dict.get("cx", None)

        B, D, T = x.shape
        x = pad1d(x, self.patch_size)  # x: (b, d, t)
        x = self.patch_x(x)  # shape: (b, d, t, f)

        e = torch.zeros_like(x)

        # 2.1 Time embedder. Repeat B times for inference
        if t.dim() == 0:
            t = t.repeat(B)

        e += self.t_embedder(t)[:, :, None]
        
        if c is not None: 
            # [1, 768]
            e += c[:, :, None]
        
        if ct is not None: 
            # [8, 768, 750]
            e += ct[:, :, :]

        if cx is not None:
            cx = rearrange(cx, 'b d t -> b t d')

        # x [8, 64, 750] e [8, 750, 768])
        x = rearrange(x, 'b d t -> b t d')
        e = rearrange(e, 'b d t -> b t d')

        for block in self.blocks:
            x = block(x, e, cx, self.rope)
        # [8, 750, 768] -> [8, 768, 750]
        x = rearrange(x, 'b t d -> b d t')
        x = self.unpatch_x(x)
        x = x[:, :, 0 : T]


        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer1D(
        in_dim=16,
        patch_size=4,
        dim=128,
        mlp_ratio=4.0,
        num_layers=2,
        num_heads=4,
        rope_len=512,
    ).to(device)

    B, D, T = 2, 16, 256
    x = torch.randn(B, D, T, device=device)
    t = torch.rand(B, device=device)

    # 只给 c
    emb = {"c": torch.randn(B, 128, device=device)}
    with torch.no_grad():
        y = model(t, x, emb)
    print("Transformer1D out (c only):", y.shape)

    # 给 c + ct（时间长度需与 patch 之后一致）
    with torch.no_grad():
        x_pad = pad1d(x, model.patch_size)
        tlen = model.patch_x(x_pad).shape[-1]
    emb2 = {
        "c": torch.randn(B, 128, device=device),
        "ct": torch.randn(B, 128, tlen, device=device),
    }
    with torch.no_grad():
        y2 = model(t, x, emb2)
    print("Transformer1D out (c+ct):", y2.shape)

    # 测试标量 t
    t_scalar = torch.tensor(0.5, device=device)
    with torch.no_grad():
        y3 = model(t_scalar, x, emb)
    print("Transformer1D out (scalar t):", y3.shape)