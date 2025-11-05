import torch.nn as nn
from torch import Tensor
from einops import rearrange


class OnehotEncoder(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Embedding(num_classes, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, cond_dict: dict) -> Tensor:
        r"""Compute latent embedding."""

        if cond_dict["id"].ndim == 1:
            c = self.mlp(cond_dict["id"])  # (b, d)
            return {"c": c}

        elif cond_dict["id"].ndim > 1:
            cx = self.mlp(cond_dict["id"])  # (b, t, d)
            cx = rearrange(cx, 'b t d -> b d t')
            return {"cx": cx}

        else:
            raise ValueError