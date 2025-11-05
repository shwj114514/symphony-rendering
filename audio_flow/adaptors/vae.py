import torch.nn as nn
from torch import Tensor
from einops import rearrange


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, cond_dict: dict) -> Tensor:
        r"""Compute onehot embedding."""

        ct = rearrange(cond_dict["ct"], 'b d t -> b t d')
        ct = self.mlp(ct)
        ct = rearrange(ct, 'b t d -> b d t')

        emb_dict = {
            "ct": ct,
        }

        return emb_dict