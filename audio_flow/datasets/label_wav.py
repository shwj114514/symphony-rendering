r"""Code modified from: https://github.com/AudioFans/audidata/blob/main/audidata/datasets/gtzan.py"""
from __future__ import annotations

import os
import re
from pathlib import Path

import h5py
import random
import pickle
import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.transforms.onehot import OneHot
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal

import torch
import typing as tp

import torchaudio

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


class LabelWavDatset(Dataset):
    r"""
        vae_latent wav_np
    """


    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "test"] = "train",
        test_fold: int = 0,  # E.g., fold 0 is used for testing. Fold 1 - 9 are used for training.
        duration: float = 10.,
        data_sr = 16000,
    ) -> None:
    
        self.root = root
        self.split = split
        self.test_fold = test_fold
        self.duration = duration
        vae_latent_folder = os.path.join(root,"vae_latent")
        LABELS = sorted(os.listdir(vae_latent_folder))

        CLASSES_NUM = len(LABELS)
        LB_TO_IX = {lb: ix for ix, lb in enumerate(LABELS)}
        IX_TO_LB = {ix: lb for ix, lb in enumerate(LABELS)}


        
        self.labels = LABELS
        self.lb_to_ix = LB_TO_IX
        self.ix_to_lb = IX_TO_LB

        self.meta_dict = self.load_meta()
        # vae_list = get_filelist(vae_latent_folder,[".h5"])

    def __getitem__(self, index: int) -> dict:
        try:
            wav_np_path = str(self.meta_dict["path"][index])
            label = self.meta_dict["label"][index] 
            
            large_waveform_mmap = np.load(wav_np_path, mmap_mode='r')
            C, L = large_waveform_mmap.shape

            seg_len = 48000 * 30 
            if L < seg_len:
                # 太短：补零或直接跳过这条样本
                pad = np.zeros((C, seg_len), dtype=large_waveform_mmap.dtype)
                pad[:, :L] = large_waveform_mmap
                chunk_np = pad
            else:
                start = np.random.randint(0, L - seg_len + 1)
                end = start + seg_len
                chunk_np = large_waveform_mmap[:, start:end] 

            chunk_torch = torch.from_numpy(chunk_np.copy())
            resampler = torchaudio.transforms.Resample(48000, 16000)
            chunk_resampled = resampler(chunk_torch)  # (C, L_out)


            target = self.lb_to_ix[label]

            full_data = {
                "label": label,
                "target": target , # shape: (classes_num,)
                "wav": chunk_resampled,

            }


            return full_data
        except Exception as e:
            print(f"e = {e} in getitem index = {index}",flush=True)
            return self.__getitem__(random.randrange(self.__len__()))

    def __len__(self) -> int:
        return len(self.meta_dict["name"])

    def load_meta(self) -> dict:
        r"""Load metadata of the GTZAN dataset.
        """

        meta_dict = {
            "name": [],
            "path": [],
            "label": [],
        }

        out_dir = Path(self.root, "wav_np")

        for genre in self.labels:
            # 'datasets/youtube2/vae_latent' + '勃拉姆斯（Johannes Brahms）'
            names = sorted(os.listdir(Path(out_dir, genre)))
            # train_names 3  test_names 10



            train_names = []
            test_names = []

            # E.g., if test_fold = 3, then test_ids = [30, 31, 32, ..., 39]
            test_names = random.choices(names,k = 1)
            # train_names = names

            train_names = list(set(names) - set(test_names))

            if self.split == "train":
                filtered_names:tp.List= train_names

            elif self.split == "test":
                filtered_names = test_names
            else:
                raise ValueError(self.split)

            for name in filtered_names:
                path = str(Path(out_dir, genre, name))
                meta_dict["name"].append(name)
                meta_dict["path"].append(path)
                meta_dict["label"].append(genre)
        return meta_dict



    def load_latent_data(self, path: str) -> dict:

        with h5py.File(path, 'r') as hf:
            latent = hf["latent"][:]
            fps = hf.attrs["fps"]

        clip_frames = int(self.duration * fps)
        bgn_frame = random.randint(0, latent.shape[-1] - clip_frames)
        bgn_frame = max(0, bgn_frame)    
        latent = latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)

        data = {
            "latent": latent,
            "fps": fps
        }

        return data



    def load_target_data(self, label: str) -> dict:

        target = self.lb_to_ix[label]

        data = {
            "label": label,
            "target": target  # shape: (classes_num,)
        }

        return data