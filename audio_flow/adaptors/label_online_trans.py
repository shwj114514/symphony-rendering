import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch
import torchaudio

class OnehotTransEncoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dim: int,cond_length:int):
        super().__init__()

        self.label_mlp = nn.Sequential(
            nn.Embedding(num_classes, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        self.trans_mlp = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )


        self.cond_length = cond_length

    def forward(self, cond_dict: dict) -> Tensor:
        r"""Compute latent embedding."""

        if cond_dict["id"].ndim == 1:
            # [16, 768]
            c = self.label_mlp(cond_dict["id"])  # (b, d)
        elif cond_dict["id"].ndim > 1:
            cx = self.mlp(cond_dict["id"])  # (b, t, d)
            cx = rearrange(cx, 'b t d -> b d t')
        

        if "frame_roll" in cond_dict:
            frame_roll = cond_dict["frame_roll"]

            # ct = rearrange(frame_roll, 'b d t -> b t d')
            # ct = frame_roll
            ct = torch.nn.functional.interpolate(frame_roll.transpose(1, 2), size=self.cond_length, mode='linear', align_corners=False).transpose(1, 2)
            ct = self.trans_mlp(ct)
            ct = rearrange(ct, 'b t d -> b d t')
        

        return {
            "ct": ct,
            "c" :c
        }


class TransEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int,cond_length:int):
        super().__init__()


        self.trans_mlp = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # loaded_model = torch.jit.load(MODEL_PATH).to(device)

        self.cond_length = cond_length

    def forward(self, cond_dict: dict) -> Tensor:


        if "frame_roll" in cond_dict:
            frame_roll = cond_dict["frame_roll"]

            # ct = rearrange(frame_roll, 'b d t -> b t d')
            # ct = frame_roll
            ct = torch.nn.functional.interpolate(frame_roll.transpose(1, 2), size=self.cond_length, mode='linear', align_corners=False).transpose(1, 2)
            ct = self.trans_mlp(ct)
            ct = rearrange(ct, 'b t d -> b d t')
        

        return {
            "ct": ct,
        }

