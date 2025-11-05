import torch.nn as nn
from torch import Tensor
from einops import rearrange


class LatentEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, cond_dict: dict) -> Tensor:
        r"""Compute latent embedding."""

        if "c" in cond_dict:
            c = self.mlp(cond_dict["c"])
            return {"c": c}

        elif "ct" in cond_dict:
            ct = rearrange(cond_dict["ct"], 'b d t -> b t d')
            ct = self.mlp(ct)
            ct = rearrange(ct, 'b t d -> b d t')
            return {"ct": ct}

        else:
            raise ValueError