import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pad1d(x: Tensor, patch_size: int) -> Tensor:
    r"""Pad a tensor along the last two dims.

    Args:
        x: (b, c, t)
        patch_size: tuple
    
    Outpus:
        out: (b, c, t)
    """

    T = x.shape[2]
    t = patch_size

    pad_t = math.ceil(T / t) * t - T
    x = F.pad(x, pad=(0, pad_t))

    return x
