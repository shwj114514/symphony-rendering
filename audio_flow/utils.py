from __future__ import annotations

import os
import sys
from collections import OrderedDict
from contextlib import contextmanager

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from einops import rearrange


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler."""

    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    """Update EMA model weights and buffers from model."""

    # Parameters
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)

    # Buffers (BN running stats, etc)
    for e, m in zip(ema.buffers(), model.buffers()):
        if m.dtype in [torch.bool, torch.long]:
            continue
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


class CombinedModel(nn.Module):
    def __init__(self, base: nn.Module, adaptor: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adaptor = adaptor


def forward_in_chunks(model: nn.Module, audio: np.array, clip_samples: int) -> np.array:

    device = next(model.parameters()).device
    latents = []
    i = 0
    skip_samples = 10000

    while i < audio.shape[-1]:

        if audio.shape[-1] - i < skip_samples:
            break
        
        x = Tensor(audio[None, :, i : i + clip_samples]).to(device)

        with torch.no_grad():
            model.eval()
            latent = model(x)[0].data.cpu().numpy()  # (d, t)

        latents.append(latent)
        i += clip_samples

    latents = np.concatenate(latents, axis=-1)

    return latents


def align_temporal_features(
    input: Tensor, 
    target: Tensor, 
    input_fps: float, 
    target_fps: float
) -> Tensor:
    r"""Align or stack the input with the target along the temporal axis.

    Args:
        input: (any, t1)
        target: (any, t2)
        input_fps: float
        target_fps: float

    Outputs:
        output: (any, t2) if t1 ≤ t2
                (any, t2*w) if t1 > t2
    """

    if input_fps == target_fps:
        return input

    elif input_fps > target_fps:

        ratio = input_fps / target_fps
        width = round(ratio / 2)
        T = target.shape[-1]
        
        indices = torch.round(torch.arange(0, T) * ratio)  # (t,)
        indices = indices[:, None] + torch.arange(-width, width + 1)  # (t, w)
        indices = torch.clamp(indices, 0, T).long()  # (t, w)

        indices = rearrange(indices, 't w -> (t w)')  # (t*w)
        output = rearrange(input[..., indices], 'b d (t w) -> b (w d) t', t=T)  # (b, w*d, t)
        return output

    else:
        ratio = input_fps / target_fps
        T = target.shape[-1]
        indices = (torch.arange(0, T) * ratio).long()  # (t,)
        output = input[..., indices]  # (b, d, t)
        return output


def logmel(audio: np.ndarray, sr: float) -> np.ndarray:

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    return np.log10(librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=2048, 
        hop_length=round(sr * 0.01), 
        n_mels=128
    )).T  # (t, f)


'''
def fix_length(x: Tensor, size: int) -> Tensor:

    if x.shape[-1] >= size:
        return x[:, :, 0 : size]
    else:
        pad_t = size - x.shape[-1]
        return F.pad(input=x, pad=(0, pad_t))


@contextmanager
def suppress_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


class Logmel:
    def __init__(self, sr: float):
        self.sr = sr
        self.n_fft = 2048
        self.hop_length = round(sr * 0.01)
        self.n_mels = 128

    def __call__(self, audio: np.array) -> np.array:
        
        logmel = np.log10(librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )).T

        return logmel
'''



def calculate_model_params(model:nn.Module):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print( "Total params: {}M, trainable params: {}M, size of saving: {}M".format(
        pytorch_total_params/1024/1024,
        pytorch_train_params/1024/1024,
        pytorch_total_params*4/1024/1024)
    )
    return


from pathlib import Path
import typing as tp
def get_filelist(
    folder: tp.Union[str, os.PathLike],
    extensions: tp.Optional[tp.List[str]] = None
) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)
    
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    filelist = [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]
    return filelist
